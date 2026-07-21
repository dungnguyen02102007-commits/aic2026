"""Kiểm kê dữ liệu BTC — N1 (Tuấn Anh), issue #1.

Trả lời 4 câu hỏi trước khi làm bất cứ gì khác với dữ liệu:
  1. Có bao nhiêu video / keyframe? Tổng dung lượng?
  2. Dữ liệu có KHỚP nhau không? (mỗi keyframe có đúng 1 dòng CLIP feature,
     có map-keyframes, có objects, có metadata) — đây là nguồn bug số 1.
  3. CLIP features là model nào? (nhìn số chiều vector: 512=ViT-B/32, 768=ViT-L/14)
  4. Dữ liệu là tin tức hay lifelog? (đọc mẫu metadata)

Xuất báo cáo data/inventory_report.md để gửi cả đội.

Chạy thử NGAY (trước 25/07) trên dữ liệu giả:
    python -m indexing.make_fake_data      # sinh data/ giả
    python -m indexing.inventory           # kiểm kê

Khi có dữ liệu thật 25/07: chỉnh CONFIG cho khớp layout BTC rồi chạy lại.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

# ── CONFIG — CHỈNH theo layout thật của BTC khi có dữ liệu (25/07) ──────────
ROOT = Path("data")
DIR_KEYFRAMES = ROOT / "keyframes"        # keyframes/<video_id>/<nnn>.jpg
DIR_CLIP      = ROOT / "clip-features"    # clip-features/<video_id>.npy
DIR_MAP       = ROOT / "map-keyframes"    # map-keyframes/<video_id>.csv
DIR_OBJECTS   = ROOT / "objects"          # objects/<video_id>/*.json
DIR_METADATA  = ROOT / "metadata"         # metadata/<video_id>.json
REPORT = ROOT / "inventory_report.md"


def count_keyframes(video_id: str) -> int:
    d = DIR_KEYFRAMES / video_id
    return len(list(d.glob("*.jpg"))) if d.is_dir() else 0


def clip_info(video_id: str):
    """Trả (số dòng, số chiều) của file CLIP feature, hoặc (None, None) nếu thiếu."""
    f = DIR_CLIP / f"{video_id}.npy"
    if not f.exists():
        return None, None
    arr = np.load(f, mmap_mode="r")   # mmap: không nạp cả file vào RAM
    if arr.ndim == 1:                 # 1 keyframe -> có thể là vector phẳng
        return 1, arr.shape[0]
    return arr.shape[0], arr.shape[1]


def map_rows(video_id: str):
    f = DIR_MAP / f"{video_id}.csv"
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1   # trừ dòng header


def has_objects(video_id: str) -> int:
    d = DIR_OBJECTS / video_id
    return len(list(d.glob("*.json"))) if d.is_dir() else 0


def load_metadata(video_id: str):
    f = DIR_METADATA / f"{video_id}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 — báo file hỏng thay vì crash
        return {"_error": str(e)}


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main():
    if not DIR_KEYFRAMES.is_dir():
        print(f"⚠ Không thấy {DIR_KEYFRAMES}. Chạy `python -m indexing.make_fake_data` "
              "để tạo dữ liệu giả, hoặc chỉnh CONFIG cho khớp layout BTC.")
        return

    # Danh sách video = các thư mục con trong keyframes (nguồn sự thật)
    videos = sorted(d.name for d in DIR_KEYFRAMES.iterdir() if d.is_dir())
    rows, warnings, dims, channels = [], [], Counter(), Counter()
    total_kf = 0

    for vid in videos:
        n_kf = count_keyframes(vid)
        n_feat, dim = clip_info(vid)
        n_map = map_rows(vid)
        n_obj = has_objects(vid)
        meta = load_metadata(vid)
        total_kf += n_kf
        if dim:
            dims[dim] += 1

        # --- Đối chiếu chéo: bắt mọi thứ lệch ---
        flags = []
        if n_feat is None:
            flags.append("THIẾU clip-features")
        elif n_feat != n_kf:
            flags.append(f"features {n_feat}≠{n_kf} keyframe")
        if n_map is None:
            flags.append("thiếu map-keyframes")
        elif n_map != n_kf:
            flags.append(f"map {n_map}≠{n_kf}")
        if n_obj == 0:
            flags.append("thiếu objects")
        if meta is None:
            flags.append("THIẾU metadata")
        elif "_error" in meta:
            flags.append("metadata HỎNG")
        else:
            channels[meta.get("author", "?")] += 1
        if flags:
            warnings.append(f"{vid}: " + "; ".join(flags))

        rows.append((vid, n_kf, n_feat, dim, n_map, n_obj,
                     "ok" if not flags else "⚠"))

    _write_report(videos, rows, warnings, dims, channels, total_kf)
    _print_summary(videos, warnings, dims, channels, total_kf)


def _write_report(videos, rows, warnings, dims, channels, total_kf):
    lines = ["# Báo cáo kiểm kê dữ liệu AIC 2026", ""]
    lines.append(f"- Tổng video: **{len(videos)}**")
    lines.append(f"- Tổng keyframe: **{total_kf:,}**")
    lines.append(f"- Dung lượng keyframes: {dir_size_mb(DIR_KEYFRAMES):,.0f} MB · "
                 f"clip-features: {dir_size_mb(DIR_CLIP):,.0f} MB")
    dim_str = ", ".join(f"{d} chiều ({n} video)" for d, n in dims.items()) or "—"
    lines.append(f"- Số chiều CLIP: **{dim_str}**  "
                 f"→ {_guess_model(dims)}")
    lines.append(f"- Kênh/nguồn (mẫu): {dict(channels.most_common(5))}")
    lines.append("")
    lines.append(f"## Cảnh báo ({len(warnings)})")
    lines += [f"- {w}" for w in warnings] or ["- (không có — dữ liệu sạch)"]
    lines.append("")
    lines.append("## Chi tiết từng video")
    lines.append("| video | keyframe | feat rows | dim | map | objects | trạng thái |")
    lines.append("|---|---|---|---|---|---|---|")
    for vid, n_kf, n_feat, dim, n_map, n_obj, st in rows:
        lines.append(f"| {vid} | {n_kf} | {n_feat} | {dim} | {n_map} | {n_obj} | {st} |")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def _guess_model(dims: Counter) -> str:
    if not dims:
        return "chưa xác định (thiếu features)"
    guesses = {512: "CLIP ViT-B/32 hoặc B/16", 768: "CLIP ViT-L/14",
               1024: "CLIP ViT-H hoặc SigLIP-L", 1152: "SigLIP-SO400M"}
    d = dims.most_common(1)[0][0]
    return f"khả năng {guesses.get(d, 'model lạ — hỏi BTC')} — XÁC NHẬN trước khi encode text query!"


def _print_summary(videos, warnings, dims, channels, total_kf):
    print(f"\n{'='*54}")
    print(f"  {len(videos)} video · {total_kf:,} keyframe")
    print(f"  CLIP dim: {dict(dims)} → {_guess_model(dims)}")
    print(f"  Nguồn (mẫu): {dict(channels.most_common(3))}")
    print(f"  Cảnh báo: {len(warnings)}")
    for w in warnings:
        print(f"    ⚠ {w}")
    print(f"{'='*54}")
    print(f"Báo cáo đầy đủ: {REPORT.resolve()}")


if __name__ == "__main__":
    main()
