"""Chạy TRAKE xuyên suốt và sinh file bài nộp (issue #19).

    python eval/run_trake.py --queries eval/queries_trake_L21.jsonl \
        --videos D:/AIC2026/videos/Videos_L21_a/video \
        --index  D:/AIC2026/index_L21 \
        --out    eval/results/bai_nop_trake.jsonl

Rồi chấm:
    python eval/run_benchmark.py --submissions eval/results/bai_nop_trake.jsonl \
        --queries eval/queries_trake_L21.jsonl --tag trake_v1

Thư mục --index cần 3 file do indexing/build_index_video.py sinh ra:
    <ten>.faiss, <ten>_ids.json (id dạng "<video_id>#<frame_idx>"), <ten>_info.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.clip_encoder import ClipEncoder  # noqa: E402
from agent.trake import (decompose, rank_videos_by_events,  # noqa: E402
                         solve_two_pass)


def tach_meta(kf_id):
    """id dạng 'L21_V001#12345' — số sau # CHÍNH LÀ frame_idx của video gốc."""
    v, f = kf_id.split("#")
    return {"video_id": v, "frame_idx": int(f)}


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--videos", required=True, help="thư mục chứa <video_id>.mp4")
    ap.add_argument("--index", required=True, help="thư mục chứa .faiss + _ids.json")
    ap.add_argument("--out", default="eval/results/bai_nop_trake.jsonl")
    ap.add_argument("--top-video", type=int, default=3,
                    help="số video ứng viên chạy căn chỉnh (mỗi cái ~1-3 phút)")
    ap.add_argument("--coarse-step", type=int, default=30)
    ap.add_argument("--fine-step", type=int, default=2)
    ap.add_argument("--fine-radius", type=int, default=45)
    args = ap.parse_args()

    idir = Path(args.index)
    ten = next(idir.glob("*.faiss")).stem
    index = faiss.read_index(str(idir / f"{ten}.faiss"))
    ids = json.load(open(idir / f"{ten}_ids.json"))
    info = json.load(open(idir / f"{ten}_info.json"))
    print(f"Index {ten}: {index.ntotal} vector, {len(info)} video")

    encoder = ClipEncoder()

    def tim(text, k=2000):
        tv = encoder.encode_text([text]).astype("float32")
        D, I = index.search(tv, min(k, index.ntotal))
        return [(ids[i], float(d)) for d, i in zip(D[0], I[0]) if i != -1]

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8-sig")
               if l.strip() and not l.startswith("//")]
    bai_nop = []

    for q in queries:
        if q.get("type") != "trake":
            continue
        print(f"\n{'='*66}\n{q['qid']}")
        t0 = time.time()
        r = decompose(q["query_vi"])
        for i, ev in enumerate(r["events"], 1):
            print(f"  {i}. {ev['search_en']}")

        # Giai đoạn 1 — hỏi RIÊNG từng khoảnh khắc. Đừng dùng video_search_en
        # tổng hợp: bối cảnh chung (bản tin/trường quay) đúng với mọi video và
        # sẽ lấn át nội dung riêng. Đo thật: top-3 5/5 vs 2/5.
        hits_ev = [tim(ev["search_en"]) for ev in r["events"]]
        ung_vien = rank_videos_by_events(hits_ev, tach_meta, args.top_video)
        print("  video ứng viên: " + ", ".join(
            f"{v['video_id']}({v['score_cuoi']:.2f})" for v in ung_vien))

        # Giai đoạn 2 — căn chỉnh trên từng video ứng viên, xếp theo độ tin cậy
        answers = []
        for v in ung_vien:
            vp = Path(args.videos) / f"{v['video_id']}.mp4"
            if not vp.exists():
                print(f"    (bỏ qua {v['video_id']}: không thấy file)")
                continue
            chot, diem, _ = solve_two_pass(
                vp, info[v["video_id"]]["n_frames"], r["events"], encoder,
                coarse_step=args.coarse_step, fine_radius=args.fine_radius,
                fine_step=args.fine_step)
            if chot:
                answers.append({"video_id": v["video_id"], "frame_ids": chot})
                print(f"    {v['video_id']}: {chot}  ({diem:.3f})")
        bai_nop.append({"qid": q["qid"], "answers": answers})
        print(f"  ({time.time()-t0:.0f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in bai_nop:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"\n{len(bai_nop)} bài nộp -> {out}")


if __name__ == "__main__":
    main()
