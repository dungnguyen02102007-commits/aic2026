from pathlib import Path

# ===== ĐƯỜNG DẪN DATASET =====
DATASET_ROOT = Path("dataset")

VIDEO_DIR = DATASET_ROOT / "videos"
KEYFRAME_DIR = DATASET_ROOT / "keyframes"
CLIP_DIR = DATASET_ROOT / "clip_features"
OBJECT_DIR = DATASET_ROOT / "objects"
METADATA_DIR = DATASET_ROOT / "metadata"

REPORT_FILE = Path("docs/inventory_report.md")


def count_files(folder, extensions=None):
    if not folder.exists():
        return 0

    count = 0
    for file in folder.rglob("*"):
        if file.is_file():
            if extensions is None:
                count += 1
            else:
                if file.suffix.lower() in extensions:
                    count += 1
    return count


def folder_size(folder):
    if not folder.exists():
        return 0

    total = 0
    for file in folder.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def human_size(size):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


video_count = count_files(VIDEO_DIR, {".mp4", ".avi", ".mov", ".mkv"})
keyframe_count = count_files(KEYFRAME_DIR, {".jpg", ".jpeg", ".png"})
clip_count = count_files(CLIP_DIR)
object_count = count_files(OBJECT_DIR)
metadata_count = count_files(METADATA_DIR)

video_size = folder_size(VIDEO_DIR)
keyframe_size = folder_size(KEYFRAME_DIR)
clip_size = folder_size(CLIP_DIR)

report = f"""# Inventory Report

## Dataset Summary

| Item | Count |
|------|------:|
| Videos | {video_count} |
| Keyframes | {keyframe_count} |
| CLIP Features | {clip_count} |
| Object Files | {object_count} |
| Metadata Files | {metadata_count} |

## Storage

| Folder | Size |
|--------|------:|
| Videos | {human_size(video_size)} |
| Keyframes | {human_size(keyframe_size)} |
| CLIP Features | {human_size(clip_size)} |

## Validation

- Video folder exists: {VIDEO_DIR.exists()}
- Keyframe folder exists: {KEYFRAME_DIR.exists()}
- CLIP folder exists: {CLIP_DIR.exists()}
- Object folder exists: {OBJECT_DIR.exists()}
- Metadata folder exists: {METADATA_DIR.exists()}
"""

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.write_text(report, encoding="utf-8")

print(report)
print(f"Report saved to: {REPORT_FILE}")