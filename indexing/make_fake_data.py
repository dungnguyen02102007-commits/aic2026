"""Sinh dữ liệu GIẢ mô phỏng cấu trúc dataset AIC 2026 — để phát triển/test
inventory.py và pipeline TRƯỚC khi BTC phát dữ liệu thật (25/07).

Tạo cấu trúc (dựa trên định dạng các mùa AIC HCMC trước — CHỈNH theo dữ liệu thật khi có):
    data/keyframes/<video_id>/<nnn>.jpg         # I-frame keyframes
    data/clip-features/<video_id>.npy           # CLIP feature, 1 dòng / keyframe
    data/map-keyframes/<video_id>.csv           # n, frame_idx, pts_time, fps
    data/objects/<video_id>/<nnn>.json          # object detection / keyframe
    data/metadata/<video_id>.json               # metadata YouTube

CỐ TÌNH gài vài lỗi (video thiếu features, số dòng lệch, thiếu metadata) để
kiểm tra xem inventory.py có bắt được không.

Chạy:  python -m indexing.make_fake_data
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("data")
CLIP_DIM = 512          # giả lập CLIP ViT-B/32 (thật có thể 512/768 — inventory sẽ tự đo)
N_VIDEOS = 6
random.seed(7)

_CHANNELS = ["60 Giây Official", "HTV Tin Tức", "Đời sống 24h"]


def make():
    for v in range(1, N_VIDEOS + 1):
        vid = f"L01_V{v:03d}"
        n_kf = random.randint(8, 16)
        fps = 25.0

        # keyframes (ảnh nhỏ) + map-keyframes csv
        (ROOT / "keyframes" / vid).mkdir(parents=True, exist_ok=True)
        (ROOT / "map-keyframes").mkdir(parents=True, exist_ok=True)
        (ROOT / "objects" / vid).mkdir(parents=True, exist_ok=True)
        rows = []
        for i in range(1, n_kf + 1):
            Image.new("RGB", (64, 36),
                      (random.randint(0, 255),) * 3).save(
                ROOT / "keyframes" / vid / f"{i:03d}.jpg")
            frame_idx = i * random.randint(40, 80)
            rows.append([i, frame_idx, round(frame_idx / fps, 3), fps])
            # objects giả: vài nhãn Open Images
            (ROOT / "objects" / vid / f"{i:03d}.json").write_text(json.dumps({
                "detection_class_entities": random.sample(
                    ["Person", "Car", "Dog", "Table", "Food", "Tree"], k=3),
                "detection_scores": [round(random.uniform(.5, .95), 2) for _ in range(3)],
            }), encoding="utf-8")
        with open(ROOT / "map-keyframes" / f"{vid}.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["n", "frame_idx", "pts_time", "fps"])
            w.writerows(rows)

        # clip-features — CỐ TÌNH gài lỗi:
        #  - V003: thiếu hẳn file features
        #  - V005: số dòng features LỆCH số keyframe
        (ROOT / "clip-features").mkdir(parents=True, exist_ok=True)
        if vid != "L01_V003":
            n_feat = n_kf - 2 if vid == "L01_V005" else n_kf
            feats = np.random.randn(n_feat, CLIP_DIM).astype("float32")
            np.save(ROOT / "clip-features" / f"{vid}.npy", feats)

        # metadata — CỐ TÌNH gài lỗi: V004 thiếu metadata
        (ROOT / "metadata").mkdir(parents=True, exist_ok=True)
        if vid != "L01_V004":
            (ROOT / "metadata" / f"{vid}.json").write_text(json.dumps({
                "author": random.choice(_CHANNELS),
                "channel_id": "UC" + "".join(random.choices("abcdef0123", k=10)),
                "title": f"Bản tin ngày {v:02d}/05/2022 — Tin tức mới nhất",
                "description": "Kênh tổng hợp tin tức thời sự.",
                "keywords": ["tin tức", "thời sự", "HTV"],
                "length": random.randint(600, 1800),
                "publish_date": "2022-05-11",
                "watch_url": f"https://youtube.com/watch?v={vid}",
            }, ensure_ascii=False), encoding="utf-8")

    print(f"Đã sinh {N_VIDEOS} video giả trong {ROOT.resolve()}")
    print("Lỗi cố tình gài: V003 thiếu features, V005 lệch số dòng, V004 thiếu metadata")


if __name__ == "__main__":
    make()
