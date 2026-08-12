"""Dựng index CLIP trực tiếp TỪ VIDEO GỐC (không dùng keyframe của BTC) — TRAKE.

Vì sao không dùng CLIPFeatures sẵn: chúng đánh theo keyframe I-frame, mà TRAKE
cần định vị theo `frame_idx` của video gốc. Index này lưu thẳng frame_idx nên
giai đoạn 1 chỉ ra được vị trí thô để giai đoạn 2 quét dày quanh đó.

    python -m indexing.build_index_video --videos D:/AIC2026/videos/Videos_L21_a/video \
        --out D:/AIC2026/index_L21 --name L21 --step 90

Sinh ra <out>/<name>.faiss, <name>_ids.json ("<video_id>#<frame_idx>"),
<name>_info.json (fps, số frame, thời lượng từng video).

Tốc độ đo thật: ~30 giây/video 20 phút ở 720p — nút thắt là GIẢI MÃ, không phải
model. Chạy trên GPU cũng không nhanh hơn nhiều.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.clip_encoder import ClipEncoder  # noqa: E402
from indexing.extract_frames import iter_frames, video_info  # noqa: E402


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="index")
    ap.add_argument("--step", type=int, default=90,
                    help="lấy 1 frame mỗi N frame (90 ~ 3 giây ở 30fps)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.videos).glob("*.mp4"))
    if not videos:
        sys.exit(f"Không thấy file .mp4 nào trong {args.videos}")
    print(f"{len(videos)} video, step={args.step}\n")

    encoder = ClipEncoder()
    vecs, ids, info = [], [], {}
    t_dau = time.time()

    for k, vid in enumerate(videos, 1):
        t0 = time.time()
        meta = video_info(vid)
        n = meta["n_frames_uoc_tinh"]
        idxs, imgs = [], []
        for fid, arr in iter_frames(vid, 0, n, args.step, size=(398, 224)):
            idxs.append(fid)
            imgs.append(arr)
        if not idxs:
            print(f"[{k}/{len(videos)}] {vid.stem}: KHÔNG đọc được frame nào")
            continue
        vecs.append(encoder.encode_images(imgs))
        ids += [f"{vid.stem}#{f}" for f in idxs]
        info[vid.stem] = {"fps": meta["fps"], "n_frames": n,
                          "duration": meta["duration"]}
        print(f"[{k:2}/{len(videos)}] {vid.stem}  {len(idxs):5} frame  "
              f"{time.time()-t0:5.1f}s  (tổng {(time.time()-t_dau)/60:.1f} phút)")

    X = np.ascontiguousarray(np.concatenate(vecs).astype("float32"))
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)  # vector đã chuẩn hoá L2 trong ClipEncoder
    faiss.write_index(index, str(out / f"{args.name}.faiss"))
    json.dump(ids, open(out / f"{args.name}_ids.json", "w"))
    json.dump(info, open(out / f"{args.name}_info.json", "w"), indent=2)
    print(f"\nXONG: {index.ntotal} vector, {X.shape[1]} chiều -> {out}")
    print(f"Tổng thời gian: {(time.time()-t_dau)/60:.1f} phút")


if __name__ == "__main__":
    main()
