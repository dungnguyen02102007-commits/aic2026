"""Mock store — dữ liệu giả để N3/N4 dev song song trước khi có dữ liệu BTC (25/07).

8 video giả × 12 keyframe = 96 keyframe, mỗi cái có caption tiếng Anh.
- search_text: chấm điểm theo số từ trùng giữa query và caption (giả lập vector search).
- search_image: xếp hạng deterministic theo hash ảnh (giả lập image similarity).
- Thumbnail JPEG sinh tự động bằng PIL vào backend/mockdata/thumbs/ (đã gitignore).

Khi có dữ liệu thật: thay module này bằng indexing/stores.py (cùng chữ ký hàm) —
backend/main.py không phải đổi dòng nào.
"""
import hashlib
import os
import random
from pathlib import Path

MOCKDATA_DIR = Path(__file__).parent / "mockdata"
THUMB_DIR = MOCKDATA_DIR / "thumbs"

_SCENES = [
    "a person drinking coffee in a cafe with red walls",
    "a yellow dog running in a park",
    "driving on a highway at night with city lights",
    "a man eating a hamburger and fries in a fast food restaurant",
    "a person eating ice cream near the sea on a rocky beach",
    "a birthday party with a boy blowing candles on a cake",
    "a woman looking at purses in a shop, purple and white bags",
    "walking through a fruit market with ripe oranges and bananas",
    "a toy soldier in red coat displayed in a mall next to clothes",
    "people planting trees and watering flowers in a garden",
    "a news anchor reading headlines in a tv studio",
    "riding a motorbike through a busy street in the rain",
]


class MockStore:
    def __init__(self):
        self.keyframes = []  # list[dict]
        random.seed(42)
        for v in range(1, 9):
            video_id = f"L01_V{v:03d}"
            for i in range(12):
                caption = _SCENES[(v * 5 + i) % len(_SCENES)]
                self.keyframes.append({
                    "keyframe_id": f"{video_id}#{i * 25:06d}",
                    "video_id": video_id,
                    "timestamp": round(i * 10.0 + random.uniform(0, 3), 1),
                    "caption": caption,
                })
        self.by_id = {k["keyframe_id"]: k for k in self.keyframes}

    # --- API giống indexing/stores.py thật ---

    def search_text(self, query: str, top_k: int = 100, filter_ids=None):
        qwords = set(query.lower().split())
        scored = []
        for kf in self._pool(filter_ids):
            overlap = len(qwords & set(kf["caption"].split()))
            noise = int(hashlib.md5((query + kf["keyframe_id"]).encode()).hexdigest(), 16) % 100
            score = overlap + noise / 1000.0  # deterministic theo (query, kf)
            scored.append((kf["keyframe_id"], round(score, 4)))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def search_image(self, image_bytes: bytes, top_k: int = 100, filter_ids=None):
        h = int(hashlib.md5(image_bytes).hexdigest(), 16)
        scored = [(kf["keyframe_id"],
                   round((int(hashlib.md5(f"{h}{kf['keyframe_id']}".encode()).hexdigest(), 16) % 1000) / 1000, 4))
                  for kf in self._pool(filter_ids)]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def timeline(self, video_id: str, around: str | None = None, window: int = 30):
        kfs = sorted((k for k in self.keyframes if k["video_id"] == video_id),
                     key=lambda k: k["timestamp"])
        if around and around in self.by_id:
            center = next(i for i, k in enumerate(kfs) if k["keyframe_id"] == around)
            kfs = kfs[max(0, center - window):center + window + 1]
        return kfs

    def get_metadata(self, keyframe_id: str):
        return self.by_id.get(keyframe_id)

    def _pool(self, filter_ids):
        if filter_ids is None:
            return self.keyframes
        allowed = set(filter_ids)
        return [k for k in self.keyframes if k["keyframe_id"] in allowed]


def generate_thumbnails(store: MockStore):
    """Sinh thumbnail 320x180 cho từng keyframe giả (chạy 1 lần lúc khởi động)."""
    from PIL import Image, ImageDraw

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    for kf in store.keyframes:
        out = THUMB_DIR / f"{kf['keyframe_id'].replace('#', '_')}.jpg"
        if out.exists():
            continue
        h = int(hashlib.md5(kf["keyframe_id"].encode()).hexdigest(), 16)
        bg = ((h >> 16) % 160 + 40, (h >> 8) % 160 + 40, h % 160 + 40)
        img = Image.new("RGB", (320, 180), bg)
        d = ImageDraw.Draw(img)
        d.text((10, 10), kf["keyframe_id"], fill="white")
        words = kf["caption"].split()
        d.text((10, 80), " ".join(words[:4]), fill="white")
        d.text((10, 100), " ".join(words[4:8]), fill="white")
        d.text((10, 150), f"t={kf['timestamp']}s", fill="yellow")
        img.save(out, quality=80)
