"""Benchmark hệ thống truy hồi AIC 2026 (N5 - Quang Dũng).

Chạy các truy vấn KIS/AVS trong queries.jsonl lên endpoint /search,
tính recall@K (KIS) / precision@K (AVS) + latency, in bảng markdown
và ghi lịch sử vào eval/results/history.csv.

VQA/KISC cần pipeline agent — chấm riêng ở giai đoạn M4 (TODO).

Dùng:
    python eval/run_benchmark.py --api http://localhost:8000 \
        --queries eval/queries.jsonl --tag clip_only
"""
import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests

RECALL_KS = [1, 5, 20, 100]


def load_queries(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def overlap(seg, ans):
    """Segment trả về có giao với khoảng đáp án không (cùng video)."""
    if seg["video_id"] != ans["video_id"]:
        return False
    return seg["start_time"] < ans["end"] and seg["end_time"] > ans["start"]


def eval_kis(q, results):
    """Trả về hạng (1-based) của kết quả đúng đầu tiên, hoặc None."""
    for rank, seg in enumerate(results, start=1):
        if overlap(seg, q["answer"]):
            return rank
    return None


def eval_avs(q, results, k=50):
    """Precision@k trên danh sách đáp án đã biết (chấp nhận không đầy đủ)."""
    if not results:
        return 0.0
    hits = sum(1 for seg in results[:k] if any(overlap(seg, a) for a in q["answers"]))
    return hits / min(k, len(results))


def run(api, queries, top_k=200):
    kis_ranks, avs_precisions, latencies, failures = [], [], [], []
    for q in queries:
        if q["type"] not in ("kis", "avs"):
            continue  # vqa/kisc: chấm ở pipeline agent (M4)
        payload = {
            "query_text": q["query_vi"],  # NOTE: khi N3 xong, đổi sang bản dịch EN
            "top_k": top_k,
            "mode": q["type"],
        }
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{api}/search", json=payload, timeout=60)
            r.raise_for_status()
            results = r.json()["results"]
        except Exception as e:  # noqa: BLE001 — benchmark phải chạy hết bộ câu
            failures.append((q["qid"], str(e)))
            continue
        latencies.append((time.perf_counter() - t0) * 1000)

        if q["type"] == "kis":
            kis_ranks.append((q, eval_kis(q, results)))
        else:
            avs_precisions.append((q, eval_avs(q, results)))
    return kis_ranks, avs_precisions, latencies, failures


def summarize(tag, kis_ranks, avs_precisions, latencies, failures):
    n_kis = len(kis_ranks)
    row = {"tag": tag, "time": datetime.now().isoformat(timespec="seconds"),
           "n_kis": n_kis, "n_avs": len(avs_precisions)}
    for k in RECALL_KS:
        hit = sum(1 for _, rank in kis_ranks if rank is not None and rank <= k)
        row[f"recall@{k}"] = round(hit / n_kis, 3) if n_kis else ""
    if avs_precisions:
        row["avs_p@50"] = round(statistics.mean(p for _, p in avs_precisions), 3)
    if latencies:
        row["p50_ms"] = round(statistics.median(latencies))
        row["p95_ms"] = round(sorted(latencies)[int(len(latencies) * 0.95) - 1])
    row["failures"] = len(failures)

    # Bảng markdown để dán vào PR
    cols = list(row.keys())
    print("\n| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    print("| " + " | ".join(str(row[c]) for c in cols) + " |\n")

    # Câu trượt hoàn toàn (không có trong top-K nào) — dữ liệu vàng cho N3 tune
    misses = [q["qid"] for q, rank in kis_ranks if rank is None]
    if misses:
        print("KIS trượt hẳn (không có trong top-200):", ", ".join(misses))
    for qid, err in failures:
        print(f"LỖI {qid}: {err}")

    # Ghi lịch sử
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    hist = out / "history.csv"
    write_header = not hist.exists()
    with open(hist, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Đã ghi {hist}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--queries", default="eval/queries.jsonl")
    ap.add_argument("--tag", required=True, help="tên cấu hình, vd: clip_only, with_rrf")
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    queries = load_queries(args.queries)
    print(f"{len(queries)} truy vấn từ {args.queries} → {args.api} (tag={args.tag})")
    summarize(args.tag, *run(args.api, queries, args.top_k))
