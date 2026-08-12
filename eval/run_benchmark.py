"""Chấm điểm hệ thống theo ĐÚNG công thức vòng sơ tuyển AIC 2026 (N5 - Quang Dũng).

Nguồn luật: "Thong tin vong So tuyen AIC2026.pdf" mục 2.
Tóm tắt ở docs/onboarding/10-luat-so-tuyen-va-cach-cham-diem.md

    R-Score(r_i) ∈ [0, 1]   — độ chính xác của MỘT câu trả lời
    R@k          = max R-Score trong k câu trả lời ĐẦU TIÊN,  k ∈ {1,5,20,50,100}
    Final Score  = trung bình cộng của 5 giá trị R@k

Hệ quả: hạng của câu đúng đầu tiên quyết định điểm —
    hạng 1 → 1.00 | 2-5 → 0.80 | 6-20 → 0.60 | 21-50 → 0.40 | 51-100 → 0.20

Hai chế độ chạy:

    # 1) Gọi API thật, tự dựng bài nộp (chỉ kis + qa)
    python eval/run_benchmark.py --api http://127.0.0.1:8000 \
        --queries eval/queries.jsonl --tag clip_only

    # 2) Chấm một file bài nộp có sẵn (đủ cả 3 dạng, kể cả trake)
    python eval/run_benchmark.py --submissions eval/sub.jsonl \
        --queries eval/queries.jsonl --tag trake_v1
"""
import argparse
import csv
import json
import re
import statistics
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KS = [1, 5, 20, 50, 100]
MAX_ANSWERS = 100  # BTC: tối đa 100 câu trả lời mỗi truy vấn


# --------------------------------------------------------------------------
# R-Score cho từng dạng truy vấn (PDF mục 2.1)
# --------------------------------------------------------------------------

def _in_span(frame_id, s, e):
    return frame_id is not None and s <= frame_id <= e


def rscore_kis(ans, gt):
    """Textual KIS: đúng video VÀ frame_id nằm trong [s, e]."""
    return float(ans.get("video_id") == gt["video_id"]
                 and _in_span(ans.get("frame_id"), gt["s"], gt["e"]))


def _norm(text):
    """Chuẩn hoá thô để so đáp án Q&A. BTC chấm theo NGỮ NGHĨA nên đây chỉ là
    xấp xỉ — điểm Q&A cục bộ luôn là cận dưới của điểm thật."""
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text)).lower().strip()
    return re.sub(r"[\s\.,;:!?\"'()]+", " ", t).strip()


def rscore_qa(ans, gt):
    """Q&A: đúng video, đúng khoảng frame, VÀ đáp án khớp ngữ nghĩa."""
    if not (ans.get("video_id") == gt["video_id"]
            and _in_span(ans.get("frame_id"), gt["s"], gt["e"])):
        return 0.0
    got = _norm(ans.get("answer"))
    if not got:
        return 0.0
    # gt["answer"] cho phép là chuỗi hoặc danh sách các cách diễn đạt chấp nhận được
    accepted = gt["answer"] if isinstance(gt["answer"], list) else [gt["answer"]]
    for a in accepted:
        a = _norm(a)
        if a and (got == a or a in got or got in a):
            return 1.0
    return 0.0


def rscore_trake(ans, gt):
    """TRAKE: sai video = 0 ngay. Đúng video = tỉ lệ khoảnh khắc khớp."""
    if ans.get("video_id") != gt["video_id"]:
        return 0.0
    spans = gt["spans"]
    fids = ans.get("frame_ids") or []
    hit = sum(1 for (s, e), f in zip(spans, fids) if _in_span(f, s, e))
    return hit / len(spans)


RSCORERS = {"kis": rscore_kis, "qa": rscore_qa, "trake": rscore_trake}


# --------------------------------------------------------------------------
# Final Score (PDF mục 2.2)
# --------------------------------------------------------------------------

def final_score(rscores):
    """Trung bình của R@k. rscores theo ĐÚNG thứ tự nộp bài."""
    r_at_k = [max(rscores[:k], default=0.0) for k in KS]
    return sum(r_at_k) / len(KS), r_at_k


def first_hit_rank(rscores, threshold=1.0):
    """Hạng của câu trả lời đúng hoàn toàn đầu tiên (None nếu không có)."""
    for i, s in enumerate(rscores, start=1):
        if s >= threshold:
            return i
    return None


# --------------------------------------------------------------------------
# Nạp dữ liệu
# --------------------------------------------------------------------------

def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"{path} dòng {ln}: JSON hỏng — {e}")
    return rows


def extract_frame_id(seg):
    """Lấy frame_id từ một kết quả /search.

    BTC chấm theo CHỈ SỐ KHUNG HÌNH của video gốc, không phải giây.
    Con số này nằm ở cột frame_idx của map-keyframes CSV.
    """
    for key in ("frame_idx", "frame_id", "frame_index"):
        if seg.get(key) is not None:
            return int(seg[key])
    sig = seg.get("signals") or {}
    for key in ("frame_idx", "frame_id", "frame_index"):
        if sig.get(key) is not None:
            return int(sig[key])
    return None


def fetch_from_api(api, queries, top_k=MAX_ANSWERS):
    """Gọi /search, dựng bài nộp cho kis + qa. TRAKE cần pipeline riêng."""
    submissions, latencies, failures = {}, [], []
    thieu_frame_id = False

    for q in queries:
        if q["type"] == "trake":
            continue  # /search chưa trả chuỗi nhiều frame — chấm qua --submissions
        payload = {"query_text": q["query_vi"], "top_k": top_k, "mode": "kis"}
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{api}/search", json=payload, timeout=60)
            r.raise_for_status()
            results = r.json()["results"]
        except Exception as e:  # noqa: BLE001 — benchmark phải chạy hết bộ câu
            failures.append((q["qid"], str(e)))
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

        answers = []
        for seg in results[:top_k]:
            fid = extract_frame_id(seg)
            if fid is None:
                thieu_frame_id = True
            answers.append({"video_id": seg.get("video_id"), "frame_id": fid,
                            "answer": seg.get("answer")})
        submissions[q["qid"]] = answers

    if thieu_frame_id:
        print("\n*** /search chưa trả về frame_idx — mọi điểm sẽ là 0. ***")
        print("*** BTC chấm theo chỉ số khung hình, không phải giây.     ***")
        print("*** Backend phải thêm frame_idx vào SearchResult (issue #20).\n")
    return submissions, latencies, failures


# --------------------------------------------------------------------------
# Chấm
# --------------------------------------------------------------------------

def grade(queries, submissions):
    rows = []
    for q in queries:
        qtype = q["type"]
        if qtype not in RSCORERS:
            print(f"Bỏ qua {q['qid']}: dạng '{qtype}' không hợp lệ "
                  f"(chỉ có kis/qa/trake)")
            continue
        answers = submissions.get(q["qid"])
        if answers is None:
            continue
        if len(answers) > MAX_ANSWERS:
            print(f"{q['qid']}: nộp {len(answers)} câu, BTC chỉ nhận "
                  f"{MAX_ANSWERS} — cắt bớt phần dư")
            answers = answers[:MAX_ANSWERS]

        scorer = RSCORERS[qtype]
        rscores = [scorer(a, q["gt"]) for a in answers]
        fs, r_at_k = final_score(rscores)
        rows.append({"qid": q["qid"], "type": qtype, "n_nop": len(answers),
                     "final": fs, "r_at_k": r_at_k,
                     "hang_dung_dau": first_hit_rank(rscores),
                     "best": max(rscores, default=0.0)})
    return rows


def report(tag, rows, latencies, failures):
    if not rows:
        sys.exit("Không chấm được truy vấn nào — kiểm tra queries/submissions.")

    summary = {"tag": tag, "time": datetime.now().isoformat(timespec="seconds"),
               "n_query": len(rows)}
    summary["FINAL_SCORE"] = round(statistics.mean(r["final"] for r in rows), 4)
    for i, k in enumerate(KS):
        summary[f"R@{k}"] = round(statistics.mean(r["r_at_k"][i] for r in rows), 4)
    if latencies:
        summary["p50_ms"] = round(statistics.median(latencies))
        summary["p95_ms"] = round(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)])
    summary["failures"] = len(failures)

    cols = list(summary.keys())
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    print("| " + " | ".join(str(summary[c]) for c in cols) + " |")

    # Tách theo dạng truy vấn — biết mảng nào đang yếu
    print("\n| dạng | số câu | Final Score |")
    print("|---|---|---|")
    for t in ("kis", "qa", "trake"):
        sub = [r for r in rows if r["type"] == t]
        if sub:
            print(f"| {t} | {len(sub)} | "
                  f"{statistics.mean(r['final'] for r in sub):.4f} |")

    # Nộp thiếu 100 câu = vứt điểm miễn phí (hạng 100 vẫn được 0.2)
    thieu = [r["qid"] for r in rows if r["n_nop"] < MAX_ANSWERS]
    if thieu:
        print(f"\nNộp CHƯA ĐỦ {MAX_ANSWERS} câu ({len(thieu)} truy vấn): "
              f"{', '.join(thieu[:8])}{' ...' if len(thieu) > 8 else ''}")
        print("  -> Hạng 100 vẫn được 0.2 điểm. Luôn nộp đủ 100.")

    # Câu đã tìm ra nhưng xếp hạng thấp — rerank là ăn điểm ngay
    xep_thap = [(r["qid"], r["hang_dung_dau"]) for r in rows
                if r["hang_dung_dau"] and r["hang_dung_dau"] > 5]
    if xep_thap:
        print(f"\nĐÃ tìm ra nhưng xếp hạng > 5 ({len(xep_thap)} câu) — "
              "đẩy lên top-1 là điểm tăng ngay:")
        for qid, hang in sorted(xep_thap, key=lambda x: x[1])[:10]:
            print(f"  {qid}: hạng {hang}")

    truot = [r["qid"] for r in rows if r["best"] == 0.0]
    if truot:
        print(f"\nTRƯỢT HẲN ({len(truot)}): {', '.join(truot[:12])}"
              f"{' ...' if len(truot) > 12 else ''}")
    for qid, err in failures:
        print(f"LỖI {qid}: {err}")

    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    hist = out / "history.csv"
    write_header = not hist.exists()
    with open(hist, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(summary)
    print(f"\nĐã ghi {hist}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # 127.0.0.1 thay vì localhost: trên Windows, localhost thử IPv6 trước
    # rồi mới rơi về IPv4 -> cộng oan ~2 giây/request vào số đo latency
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--submissions", help="chấm file bài nộp có sẵn thay vì gọi API")
    ap.add_argument("--tag", required=True, help="tên cấu hình, vd: clip_only, with_rrf")
    args = ap.parse_args()

    queries = load_jsonl(args.queries)
    print(f"{len(queries)} truy vấn từ {args.queries} (tag={args.tag})")

    if args.submissions:
        subs = {r["qid"]: r["answers"] for r in load_jsonl(args.submissions)}
        print(f"Chấm {len(subs)} bài nộp từ {args.submissions}")
        lat, fails = [], []
    else:
        print(f"Gọi API {args.api}")
        subs, lat, fails = fetch_from_api(args.api, queries)

    report(args.tag, grade(queries, subs), lat, fails)
