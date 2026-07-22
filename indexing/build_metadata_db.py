"""Dựng metadata DB (SQLite) từ dữ liệu BTC — N1 (Tuấn Anh), issue #2.

Gom 3 nguồn rời rạc vào MỘT file data/metadata.db để backend tra cứu nhanh:
  - map-keyframes/<video_id>.csv  -> bảng keyframes (id, timestamp, frame_idx)
  - objects/<video_id>/<nnn>.json -> cột objects_json của keyframe
  - metadata/<video_id>.json      -> bảng videos (title, author, publish_date...)

Chuẩn ID (hợp đồng API mục 3):  keyframe_id = "<video_id>#<n:06d>"
  với n là số thứ tự keyframe trong map-keyframes (khớp tên file ảnh).

Chạy:  python -m indexing.build_metadata_db          # sau khi có data/
Kiểm tra nhanh sau khi build:
       python -m indexing.build_metadata_db --check
"""
import csv
import json
import sqlite3
import sys
from pathlib import Path

# ── CONFIG — khớp với inventory.py, chỉnh khi có layout thật (25/07) ────────
ROOT = Path("data")
DIR_MAP      = ROOT / "map-keyframes"
DIR_OBJECTS  = ROOT / "objects"
DIR_METADATA = ROOT / "metadata"
DB_PATH      = ROOT / "metadata.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id     TEXT PRIMARY KEY,
    title        TEXT, author TEXT, channel_id TEXT,
    description  TEXT, keywords TEXT,
    length_sec   REAL, publish_date TEXT, watch_url TEXT
);
CREATE TABLE IF NOT EXISTS keyframes (
    keyframe_id  TEXT PRIMARY KEY,
    video_id     TEXT NOT NULL,
    n            INTEGER,          -- số thứ tự trong map-keyframes (khớp tên ảnh)
    frame_idx    INTEGER,
    timestamp    REAL,             -- giây (pts_time)
    objects_json TEXT              -- danh sách object của keyframe (JSON string)
);
CREATE INDEX IF NOT EXISTS idx_kf_video ON keyframes(video_id);
CREATE INDEX IF NOT EXISTS idx_kf_time  ON keyframes(video_id, timestamp);
"""


def build():
    if not DIR_MAP.is_dir():
        sys.exit(f"⚠ Không thấy {DIR_MAP} — chạy make_fake_data trước, "
                 "hoặc chỉnh CONFIG theo layout BTC.")
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)

    n_videos = n_kf = 0
    for csv_file in sorted(DIR_MAP.glob("*.csv")):
        video_id = csv_file.stem

        # 1) bảng videos từ metadata json (thiếu thì vẫn ghi dòng rỗng)
        meta = {}
        meta_file = DIR_METADATA / f"{video_id}.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — metadata hỏng đã có inventory cảnh báo
                meta = {}
        con.execute(
            "INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?,?,?)",
            (video_id, meta.get("title"), meta.get("author"),
             meta.get("channel_id"), meta.get("description"),
             json.dumps(meta.get("keywords", []), ensure_ascii=False),
             meta.get("length"), meta.get("publish_date"), meta.get("watch_url")))
        n_videos += 1

        # 2) bảng keyframes từ map-keyframes + objects
        with open(csv_file, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                n = int(row["n"])
                kf_id = f"{video_id}#{n:06d}"
                obj_file = DIR_OBJECTS / video_id / f"{n:03d}.json"
                objects = obj_file.read_text(encoding="utf-8") if obj_file.exists() else None
                con.execute(
                    "INSERT OR REPLACE INTO keyframes VALUES (?,?,?,?,?,?)",
                    (kf_id, video_id, n, int(row["frame_idx"]),
                     float(row["pts_time"]), objects))
                n_kf += 1

    con.commit()
    con.close()
    print(f"✓ {DB_PATH}: {n_videos} video, {n_kf} keyframe")


def check():
    """Vài truy vấn mẫu để xác nhận DB dùng được (và làm ví dụ cho N2)."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    print("Số dòng:", dict(con.execute(
        "SELECT (SELECT COUNT(*) FROM videos) AS videos,"
        " (SELECT COUNT(*) FROM keyframes) AS keyframes").fetchone()))
    row = con.execute("SELECT * FROM keyframes LIMIT 1").fetchone()
    print("Keyframe mẫu:", dict(row) if row else None)
    # ví dụ lọc theo object — kiểu truy vấn N2 sẽ dùng cho filters.objects
    hits = con.execute(
        "SELECT keyframe_id FROM keyframes WHERE objects_json LIKE ? LIMIT 5",
        ('%"Dog"%',)).fetchall()
    print('Keyframe có "Dog":', [r["keyframe_id"] for r in hits])
    con.close()


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        build()
