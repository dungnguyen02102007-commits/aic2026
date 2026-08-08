"""Kiểm kê dữ liệu BTC — N1 (Tuấn Anh), issue #1.

Bản HỢP NHẤT hai phiên bản:
  - Phần A (Tuấn Anh viết): đếm file từng loại + dung lượng thư mục.
  - Phần B (khung Dũng chuẩn bị): đối chiếu chéo từng video + đo chiều CLIP.

Trả lời 4 câu hỏi trước khi làm gì tiếp với dữ liệu:
  1. Có bao nhiêu video / keyframe? Nặng bao nhiêu? Đã tải đủ chưa?
  2. Dữ liệu có KHỚP nhau không? (mỗi keyframe đúng 1 dòng CLIP feature,
     có map-keyframes, objects, metadata) — nguồn bug số 1 của cả pipeline.
  3. CLIP features là model nào? (512 chiều = ViT-B; 768 = ViT-L/14...)
  4. Dữ liệu là tin tức hay lifelog? (đọc mẫu metadata)

Chạy:
    python -m indexing.inventory
Tập dượt khi chưa có dữ liệu thật:
    python -m indexing.make_fake_data     # sinh data/ giả
    python -m indexing.inventory

Báo cáo xuất ra <dataset_root>/inventory_report.md (KHÔNG commit — đã gitignore).
Dataset đang tải dở vẫn chạy được: phần thiếu sẽ hiện trong mục Cảnh báo.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

# ── CONFIG — chỉnh TÊN THƯ MỤC cho khớp layout thật của BTC ─────────────────
# Tự dò gốc dữ liệu: ưu tiên "dataset/" (Tuấn Anh dùng), rồi "data/" (bản cũ).
DATASET_ROOT = next((p for p in [Path("dataset"), Path("data")] if p.is_dir()),
                    Path("dataset"))

DIR_VIDEOS   = DATASET_ROOT / "videos"
DIR_KEYFRAMES = DATASET_ROOT / "keyframes"      # keyframes/<video_id>/<nnn>.jpg
# BTC có thể đặt "clip-features" hoặc "clip_features" — dò cả hai:
DIR_CLIP = next((DATASET_ROOT / n for n in ["clip-features", "clip_features"]
                 if (DATASET_ROOT / n).is_dir()), DATASET_ROOT / "clip-features")
DIR_MAP      = DATASET_ROOT / "map-keyframes"   # map-keyframes/<video_id>.csv
DIR_OBJECTS  = DATASET_ROOT / "objects"
DIR_METADATA = DATASET_ROOT / "metadata"
REPORT = DATASET_ROOT / "inventory_report.md"

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


# ═══ PHẦN A — đếm file & dung lượng (Tuấn Anh) ═══════════════════════════════

def count_files(folder: Path, extensions=None) -> int:
    if not folder.exists():
        return 0
    return sum(1 for f in folder.rglob("*") if f.is_file()
               and (extensions is None or f.suffix.lower() in extensions))


def folder_size(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())


def human_size(size: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def summary_section() -> list[str]:
    rows = [
        ("Videos", count_files(DIR_VIDEOS, VIDEO_EXT), folder_size(DIR_VIDEOS)),
        ("Keyframes", count_files(DIR_KEYFRAMES, IMAGE_EXT), folder_size(DIR_KEYFRAMES)),
        ("CLIP features", count_files(DIR_CLIP), folder_size(DIR_CLIP)),
        ("Objects", count_files(DIR_OBJECTS), folder_size(DIR_OBJECTS)),
        ("Metadata", count_files(DIR_METADATA), folder_size(DIR_METADATA)),
        ("Map-keyframes", count_files(DIR_MAP), folder_size(DIR_MAP)),
    ]
    lines = ["## A. Tổng quan tải về", "",
             "| Loại | Số file | Dung lượng |", "|---|---:|---:|"]
    lines += [f"| {name} | {n:,} | {human_size(sz)} |" for name, n, sz in rows]
    missing = [name for name, n, _ in rows if n == 0]
    if missing:
        lines += ["", f"⚠ **Chưa có/chưa tải xong:** {', '.join(missing)}"]
    return lines


# ═══ PHẦN B — đối chiếu chéo từng video + đo chiều CLIP ══════════════════════

def count_keyframes(video_id: str) -> int:
    d = DIR_KEYFRAMES / video_id
    return sum(1 for f in d.iterdir() if f.suffix.lower() in IMAGE_EXT) if d.is_dir() else 0


def clip_info(video_id: str):
    f = DIR_CLIP / f"{video_id}.npy"
    if not f.exists():
        return None, None
    arr = np.load(f, mmap_mode="r")     # chỉ đọc header, không nạp cả file vào RAM
    if arr.ndim == 1:
        return 1, arr.shape[0]
    return arr.shape[0], arr.shape[1]


def map_rows(video_id: str):
    f = DIR_MAP / f"{video_id}.csv"
    if not f.exists():
        return None
    with open(f, encoding="utf-8") as fh:
        return sum(1 for _ in csv.reader(fh)) - 1


def objects_count(video_id: str) -> int:
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


def guess_model(dims: Counter) -> str:
    if not dims:
        return "chưa xác định (chưa có features)"
    guesses = {512: "CLIP ViT-B/32 hoặc B/16", 768: "CLIP ViT-L/14",
               1024: "CLIP ViT-H hoặc SigLIP-L", 1152: "SigLIP-SO400M"}
    d = dims.most_common(1)[0][0]
    return (f"khả năng {guesses.get(d, 'model lạ — hỏi BTC')} — "
            "XÁC NHẬN trước khi encode text query!")


def crosscheck_section():
    if not DIR_KEYFRAMES.is_dir():
        return ["## B. Đối chiếu chéo", "",
                "⚠ Chưa có thư mục keyframes — tải xong hãy chạy lại."], []
    videos = sorted(d.name for d in DIR_KEYFRAMES.iterdir() if d.is_dir())
    rows, warnings, dims, channels = [], [], Counter(), Counter()
    total_kf = 0

    for vid in videos:
        n_kf = count_keyframes(vid)
        n_feat, dim = clip_info(vid)
        n_map = map_rows(vid)
        n_obj = objects_count(vid)
        meta = load_metadata(vid)
        total_kf += n_kf
        if dim:
            dims[dim] += 1

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

    lines = ["## B. Đối chiếu chéo từng video", "",
             f"- Tổng video (theo keyframes): **{len(videos)}** · "
             f"tổng keyframe: **{total_kf:,}**"]
    dim_str = ", ".join(f"{d} chiều ({n} video)" for d, n in dims.items()) or "—"
    lines.append(f"- **Số chiều CLIP: {dim_str}** → {guess_model(dims)}")
    lines.append(f"- Kênh/nguồn (mẫu metadata): {dict(channels.most_common(5))}")
    lines += ["", f"### Cảnh báo ({len(warnings)})"]
    lines += [f"- {w}" for w in warnings[:50]] or ["- (không có — dữ liệu sạch)"]
    if len(warnings) > 50:
        lines.append(f"- ... và {len(warnings) - 50} cảnh báo nữa")
    lines += ["", "### Chi tiết từng video (tối đa 200 dòng đầu)",
              "| video | keyframe | feat rows | dim | map | objects | trạng thái |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {v} | {k} | {f} | {d} | {m} | {o} | {s} |"
              for v, k, f, d, m, o, s in rows[:200]]
    return lines, warnings


# ═══ Ghép báo cáo ════════════════════════════════════════════════════════════

def main():
    print(f"Gốc dữ liệu: {DATASET_ROOT.resolve()}")
    lines = ["# Báo cáo kiểm kê dữ liệu AIC 2026", ""]
    lines += summary_section()
    lines.append("")
    cross, warnings = crosscheck_section()
    lines += cross

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[:30]))
    print("...")
    print(f"\n⚠ Tổng cảnh báo: {len(warnings)}")
    print(f"Báo cáo đầy đủ: {REPORT.resolve()}")
    print("→ Gửi nội dung báo cáo vào nhóm chat cho cả đội.")


if __name__ == "__main__":
    main()
