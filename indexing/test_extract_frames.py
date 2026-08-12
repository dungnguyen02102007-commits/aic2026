"""Kiểm chứng chỉ số frame trả về ĐÚNG TUYỆT ĐỐI (issue #19).

Chạy:  python -m indexing.test_extract_frames

Cách kiểm: dựng một video mà **mỗi frame mang chính chỉ số của nó trong màu
sắc** (kênh R = i). Trích frame i ra rồi đọc kênh R — phải bằng i.

Vì sao phải làm căng thế: `frame_idx` là con số nộp cho BTC. Lệch một đơn vị
là mất điểm mà không có lỗi nào báo ra, và không cách nào phát hiện bằng mắt.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from indexing.extract_frames import (extract_window,  # noqa: E402
                                     extract_windows, iter_frames, video_info)

N_FRAME, FPS, W, H = 200, 25, 128, 64
dat = truot = 0


def check(ten, ok, chi_tiet=""):
    global dat, truot
    print(f"{'✓' if ok else '✗'} {ten}" + (f"  — {chi_tiet}" if chi_tiet else ""))
    dat, truot = dat + bool(ok), truot + (not ok)


def dung_video_danh_dau(path):
    """Video N_FRAME frame, frame thứ i tô kín màu (R=i, G=100, B=200)."""
    import imageio_ffmpeg as ff

    writer = ff.write_frames(str(path), (W, H), fps=FPS, macro_block_size=1,
                             codec="libx264", quality=None,
                             output_params=["-crf", "0", "-pix_fmt", "yuv444p"])
    writer.send(None)
    for i in range(N_FRAME):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :, 0] = i          # kênh R mang chính chỉ số frame
        frame[:, :, 1] = 100
        frame[:, :, 2] = 200
        writer.send(frame.tobytes())
    writer.close()


def r_trung_binh(arr):
    return float(arr[:, :, 0].mean())


tmp = Path(tempfile.mkdtemp(prefix="aic_trake_"))
try:
    video = tmp / "danh_dau.mp4"
    dung_video_danh_dau(video)
    check("dựng được video thử", video.exists() and video.stat().st_size > 1000,
          f"{video.stat().st_size} byte")

    info = video_info(video)
    check("đọc được metadata", abs(info["fps"] - FPS) < 0.01
          and tuple(info["size"]) == (W, H),
          f"{info['size']} @ {info['fps']} fps, {info['duration']:.2f}s")

    # --- Phép thử cốt lõi: chỉ số trả về có khớp nội dung frame không ---
    lech = []
    for fid, arr in iter_frames(video, 0, N_FRAME - 1, 1):
        r = r_trung_binh(arr)
        if abs(r - fid) > 2:  # h264 lossless vẫn có thể lệch vài đơn vị khi đổi hệ màu
            lech.append((fid, round(r, 1)))
    check("MỌI frame: chỉ số trả về khớp nội dung frame", not lech,
          f"{len(lech)} frame lệch: {lech[:5]}" if lech else f"{N_FRAME} frame đúng hết")

    # --- Đếm đủ frame, không thiếu không thừa ---
    ids = [fid for fid, _ in iter_frames(video, 0, N_FRAME - 1, 1)]
    check("đếm đủ số frame", len(ids) == N_FRAME, f"{len(ids)}/{N_FRAME}")
    check("chỉ số bắt đầu từ 0", ids[0] == 0, f"đầu tiên = {ids[0]}")
    check("chỉ số liên tục, không nhảy cóc",
          ids == list(range(N_FRAME)))

    # --- Khoảng [start, end] lấy đúng hai đầu mút ---
    ids = [fid for fid, _ in iter_frames(video, 50, 59, 1)]
    check("khoảng [50, 59] lấy đủ cả hai đầu mút", ids == list(range(50, 60)),
          f"{ids}")

    # --- step ---
    ids = [fid for fid, _ in iter_frames(video, 100, 120, 5)]
    check("step=5 lấy đúng nhịp", ids == [100, 105, 110, 115, 120], f"{ids}")

    # --- extract_window: đây là hàm TRAKE thật sự gọi ---
    idxs, imgs = extract_window(video, center_frame=100, radius=20, step=4)
    check("extract_window trả đúng khoảng quanh tâm",
          idxs[0] == 80 and idxs[-1] == 120 and len(idxs) == len(imgs),
          f"{len(idxs)} frame, {idxs[0]}..{idxs[-1]}")
    sai = [(i, round(r_trung_binh(a), 1))
           for i, a in zip(idxs, imgs) if abs(r_trung_binh(a) - i) > 2]
    check("extract_window: nội dung khớp chỉ số", not sai, f"lệch: {sai[:3]}")

    # --- Biên: tâm sát đầu video thì không được trả chỉ số âm ---
    idxs, _ = extract_window(video, center_frame=5, radius=50, step=1)
    check("tâm sát đầu video -> không có chỉ số âm", idxs[0] == 0, f"đầu = {idxs[0]}")

    # --- extract_windows: nhiều cửa sổ trong MỘT lượt giải mã ---
    idxs, imgs = extract_windows(video, [30, 120], radius=10, step=2)
    check("extract_windows lấy đúng cả hai cửa sổ",
          idxs == [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40,
                   110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130],
          f"{idxs}")
    sai = [(i, round(r_trung_binh(a), 1))
           for i, a in zip(idxs, imgs) if abs(r_trung_binh(a) - i) > 2]
    check("extract_windows: nội dung khớp chỉ số", not sai, f"lệch: {sai[:3]}")

    # Cửa sổ chồng nhau -> không được trả trùng, phải giữ thứ tự tăng dần
    idxs, _ = extract_windows(video, [50, 60], radius=20, step=1)
    check("cửa sổ chồng nhau: không trùng lặp, vẫn tăng dần",
          idxs == sorted(set(idxs)) and idxs[0] == 30 and idxs[-1] == 80,
          f"{len(idxs)} frame, {idxs[0]}..{idxs[-1]}")

    # --- ffmpeg thu nhỏ sẵn: chỉ số vẫn phải đúng ---
    idxs, imgs = extract_window(video, 100, 10, 2, size=(64, 32))
    check("thu nhỏ bằng ffmpeg vẫn giữ đúng chỉ số",
          idxs[0] == 90 and imgs[0].shape[:2] == (32, 64),
          f"{idxs[0]}..{idxs[-1]}, ảnh {imgs[0].shape}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nKẾT QUẢ: {dat}/{dat + truot}")
sys.exit(1 if truot else 0)
