"""Dựng index CLIP GIẢ để test backend thật khi chưa có dữ liệu BTC trên máy.

Khác make_fake_data.py (sinh vector ngẫu nhiên): script này encode caption của
MockStore bằng CHÍNH model CLIP thật, rồi dùng vector đó làm "vector ảnh". Nhờ
vậy tìm kiếm ngữ nghĩa hoạt động THẬT trên máy — gõ "news anchor" ra đúng
keyframe người dẫn chương trình — nên chạy được toàn bộ đường ống
FaissStore -> /search -> benchmark mà không cần tải 346 MiB index về.

Chạy:  python -m indexing.make_fake_index

Sinh ra data/fake_index/{clip_flat.faiss, ids.json, metadata.db} và ảnh
keyframe giả trong data/fake_index/keyframes/<video_id>/<n:03d>.jpg
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.stores_faiss import MODEL_NAME, PRETRAINED  # noqa: E402
from backend.stores_mock import MockStore  # noqa: E402
from indexing.build_metadata_db import SCHEMA  # noqa: E402

OUT = Path("data/fake_index")
KF_DIR = OUT / "keyframes"


def sinh_anh(store):
    """Ảnh keyframe giả, đặt tên đúng quy ước BTC: <video_id>/<n:03d>.jpg"""
    import hashlib

    from PIL import Image, ImageDraw

    for kf in store.keyframes:
        folder = KF_DIR / kf["video_id"]
        folder.mkdir(parents=True, exist_ok=True)
        out = folder / f"{kf['n']:03d}.jpg"
        if out.exists():
            continue
        h = int(hashlib.md5(kf["keyframe_id"].encode()).hexdigest(), 16)
        img = Image.new("RGB", (320, 180),
                        ((h >> 16) % 160 + 40, (h >> 8) % 160 + 40, h % 160 + 40))
        d = ImageDraw.Draw(img)
        d.text((10, 10), kf["keyframe_id"], fill="white")
        d.text((10, 40), f"frame_idx={kf['frame_idx']}", fill="yellow")
        words = kf["caption"].split()
        d.text((10, 90), " ".join(words[:5]), fill="white")
        d.text((10, 110), " ".join(words[5:]), fill="white")
        img.save(out, quality=80)


def main():
    import faiss
    import open_clip
    import torch

    store = MockStore()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(store.keyframes)} keyframe giả từ MockStore")

    print(f"Nạp {MODEL_NAME}/{PRETRAINED} (lần đầu phải tải ~600MB)...")
    model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME,
                                                        pretrained=PRETRAINED)
    model.eval()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    # Encode caption làm vector "ảnh". Cache theo caption vì chỉ có 12 cảnh khác nhau.
    cache = {}
    vecs = []
    with torch.no_grad():
        for kf in store.keyframes:
            cap = kf["caption"]
            if cap not in cache:
                cache[cap] = model.encode_text(tokenizer([cap])).float().numpy()[0]
            vecs.append(cache[cap])
    X = np.ascontiguousarray(np.stack(vecs).astype("float32"))
    print(f"Encode xong: {X.shape} ({len(cache)} caption khác nhau)")

    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(OUT / "clip_flat.faiss"))

    ids = [kf["keyframe_id"] for kf in store.keyframes]
    (OUT / "ids.json").write_text(json.dumps(ids), encoding="utf-8")

    # metadata.db dùng ĐÚNG schema của indexing/build_metadata_db.py
    db_path = OUT / "metadata.db"
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    for vid in sorted({k["video_id"] for k in store.keyframes}):
        con.execute("INSERT INTO videos (video_id, title) VALUES (?,?)",
                    (vid, f"Video gia {vid}"))
    for kf in store.keyframes:
        con.execute("INSERT INTO keyframes VALUES (?,?,?,?,?,?)",
                    (kf["keyframe_id"], kf["video_id"], kf["n"],
                     kf["frame_idx"], kf["timestamp"], None))
    con.commit()
    con.close()

    sinh_anh(store)

    # Bộ câu hỏi để chạy thử eval/run_benchmark.py end-to-end.
    # Lưu ý: 8 video giả đều chứa CÙNG 12 cảnh, nên mỗi truy vấn có 8 đáp án
    # "đúng như nhau" mà gt chỉ nhận 1 -> điểm sẽ ở mức trung bình, đó là
    # ĐÚNG CHỦ Ý. Đây là bài test đường ống, không phải đo chất lượng model.
    seen, queries = set(), []
    for kf in store.keyframes:
        if kf["caption"] in seen:
            continue
        seen.add(kf["caption"])
        queries.append({
            "qid": f"fake_{len(queries):02d}", "type": "kis",
            "query_vi": kf["caption"],  # dữ liệu giả dùng thẳng tiếng Anh
            "gt": {"video_id": kf["video_id"],
                   "s": kf["frame_idx"] - 5, "e": kf["frame_idx"] + 5},
        })
    with open(OUT / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"\n✓ {OUT}/clip_flat.faiss  ({index.ntotal} vector, {X.shape[1]} chiều)")
    print(f"✓ {OUT}/queries.jsonl     ({len(queries)} truy vấn thử)")
    print(f"✓ {OUT}/ids.json          ({len(ids)} id, đầu: {ids[0]})")
    print(f"✓ {OUT}/metadata.db")
    print(f"✓ {KF_DIR}/  (ảnh keyframe giả)")
    print("\nChạy backend với index này:")
    print('  $env:AIC_STORE="faiss"')
    print(f'  $env:AIC_INDEX_PATH="{OUT}/clip_flat.faiss"')
    print(f'  $env:AIC_IDS_PATH="{OUT}/ids.json"')
    print(f'  $env:AIC_DB_PATH="{OUT}/metadata.db"')
    print(f'  $env:AIC_KEYFRAMES_DIR="{KF_DIR}"')
    print("  python -m uvicorn backend.main:app --port 8000")


if __name__ == "__main__":
    main()
