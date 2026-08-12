"""Chạy Textual KIS xuyên suốt và sinh file bài nộp (dạng 1, docs/onboarding/10).

    python eval/run_kis.py --queries eval/queries_kis_L21.jsonl \
        --index D:/AIC2026/index_L21_day --out eval/results/bai_nop_kis.jsonl

Rồi chấm:
    python eval/run_benchmark.py --submissions eval/results/bai_nop_kis.jsonl \
        --queries eval/queries_kis_L21.jsonl --tag kis_v1

Đường ống: truy vấn tiếng Việt -> understand() dịch sang mô tả thị giác tiếng
Anh (+ biến thể) -> tìm trên index CLIP -> RRF gộp các biến thể -> nộp 100 câu.

LUÔN nộp đủ 100 (docs/onboarding/11): câu đúng ở hạng 100 vẫn được 0.2 điểm,
và không có hình phạt nào cho câu sai.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.clip_encoder import ClipEncoder  # noqa: E402
from agent.understand import understand  # noqa: E402

MAX_NOP = 100
RRF_K = 60  # hằng số chuẩn của Reciprocal Rank Fusion, không cần tune


def rrf(danh_sach_hits):
    """Gộp nhiều lượt tìm bằng THỨ HẠNG, miễn nhiễm việc thang điểm khác nhau."""
    diem = {}
    for hits in danh_sach_hits:
        for hang, (kf_id, _) in enumerate(hits, start=1):
            diem[kf_id] = diem.get(kf_id, 0.0) + 1.0 / (RRF_K + hang)
    return sorted(diem.items(), key=lambda x: -x[1])


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", default="eval/results/bai_nop_kis.jsonl")
    ap.add_argument("--max-per-video", type=int, default=0,
                    help="0 = không giới hạn. Đặt >0 để ép đa dạng video "
                         "(đánh đổi: bớt cơ hội trúng khi đã đúng video)")
    ap.add_argument("--no-variants", action="store_true",
                    help="chỉ dùng search_text_en, bỏ biến thể — để đo xem "
                         "biến thể có giúp gì không")
    args = ap.parse_args()

    idir = Path(args.index)
    ten = next(idir.glob("*.faiss")).stem
    index = faiss.read_index(str(idir / f"{ten}.faiss"))
    ids = json.load(open(idir / f"{ten}_ids.json"))
    print(f"Index {ten}: {index.ntotal} vector")

    encoder = ClipEncoder()

    def tim(text, k=400):
        tv = encoder.encode_text([text]).astype("float32")
        D, I = index.search(tv, min(k, index.ntotal))
        return [(ids[i], float(d)) for d, i in zip(D[0], I[0]) if i != -1]

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8")
               if l.strip() and not l.startswith("//")]
    bai_nop = []

    for q in queries:
        if q.get("type") != "kis":
            continue
        t0 = time.time()
        r = understand(q["query_vi"])
        cau = [r["search_text_en"]]
        if not args.no_variants:
            cau += r.get("variants", [])[:3]
        hits = rrf([tim(c) for c in cau])

        answers, dem_video = [], {}
        for kf_id, _ in hits:
            v, f = kf_id.split("#")
            if args.max_per_video and dem_video.get(v, 0) >= args.max_per_video:
                continue
            dem_video[v] = dem_video.get(v, 0) + 1
            answers.append({"video_id": v, "frame_id": int(f)})
            if len(answers) >= MAX_NOP:
                break

        bai_nop.append({"qid": q["qid"], "answers": answers})
        dung = q["gt"]["video_id"]
        hang_v = next((i for i, a in enumerate(answers, 1)
                       if a["video_id"] == dung), None)
        hang_dung = next((i for i, a in enumerate(answers, 1)
                          if a["video_id"] == dung
                          and q["gt"]["s"] <= a["frame_id"] <= q["gt"]["e"]), None)
        print(f"{q['qid']}  \"{r['search_text_en'][:52]}\"")
        print(f"    đúng video hạng {hang_v}   ĐÚNG CẢ FRAME hạng {hang_dung}"
              f"   ({len(answers)} câu, {time.time()-t0:.1f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in bai_nop:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"\n{len(bai_nop)} bài nộp -> {out}")


if __name__ == "__main__":
    main()
