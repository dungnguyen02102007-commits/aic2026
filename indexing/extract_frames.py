"""Trích frame DÀY từ video gốc — phục vụ TRAKE (issue #19).

Vì sao cần: keyframe BTC phát là I-frame, cách nhau hàng trăm frame. Đoạn đáp
án của một khoảnh khắc ngữ nghĩa TRAKE thường **dưới 10 frame**. Nên chỉ dùng
keyframe có sẵn thì giai đoạn căn chỉnh gần như chắc chắn trượt.

BTC cho phép: "Dữ liệu thi chính thức là Video; các thành phần còn lại
(Keyframes, Objects, CLIP features, Metadata) chỉ nhằm mục đích cung cấp thêm
thông tin hoặc hỗ trợ xây dựng giải pháp mẫu."

Không cần cài ffmpeg vào máy — `imageio-ffmpeg` mang sẵn bản dựng tĩnh, chạy
được cả trên Windows lẫn Colab.

    pip install imageio-ffmpeg

⚠️ ĐIỀU QUAN TRỌNG NHẤT: chỉ số frame trả về phải TRÙNG KHÍT với cách BTC đánh
số (frame thứ 0 của video là 0). Lệch một đơn vị là nộp sai, mà không có lỗi
nào báo ra. Vì vậy module này giải mã TUẦN TỰ từ đầu và đếm — chậm hơn nhảy
cóc bằng `-ss` nhưng đúng tuyệt đối. Xem `indexing/test_extract_frames.py`.
"""
import sys
from pathlib import Path

import numpy as np


def video_info(path) -> dict:
    """fps, kích thước, thời lượng. Không giải mã toàn bộ video."""
    import imageio_ffmpeg as ff

    reader = ff.read_frames(str(path))
    meta = next(reader)  # yield đầu tiên luôn là metadata
    reader.close()
    fps = meta.get("fps") or 25.0
    dur = meta.get("duration") or 0.0
    return {"fps": fps, "size": meta.get("size"), "duration": dur,
            "n_frames_uoc_tinh": int(round(fps * dur))}


def iter_frames(path, start_frame: int = 0, end_frame: int | None = None,
                step: int = 1, size=None):
    """Sinh (frame_idx, ảnh RGB dạng ndarray) — frame_idx đếm từ 0, chính xác tuyệt đối.

    start_frame / end_frame: khoảng [start, end] tính theo chỉ số frame gốc.
    step: lấy 1 frame mỗi `step` frame. Đoạn đáp án TRAKE thường dưới 10 frame
          nên step <= 5 mới đủ an toàn.
    size: (w, h) — để ffmpeg thu nhỏ sẵn, nhanh hơn nhiều so với giải mã full HD
          rồi mới thu nhỏ trong Python. CLIP chỉ cần ~224px.
    """
    import imageio_ffmpeg as ff

    out = ["-vsync", "0"]
    if size:
        out += ["-vf", f"scale={size[0]}:{size[1]}"]
    reader = ff.read_frames(str(path), output_params=out)
    meta = next(reader)
    # meta['size'] là kích thước GỐC (lấy từ ffprobe đầu vào). Khi đã scale thì
    # phải reshape theo kích thước yêu cầu, không phải kích thước gốc.
    w, h = size if size else meta["size"]
    try:
        for i, raw in enumerate(reader):
            if i < start_frame:
                continue
            if end_frame is not None and i > end_frame:
                break
            if (i - start_frame) % step:
                continue
            yield i, np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        reader.close()


def extract_window(path, center_frame: int, radius: int = 150, step: int = 3,
                   size=None):
    """Lấy frame quanh một vị trí ứng viên. Trả về (danh sách frame_idx, danh sách ảnh).

    Dùng sau khi CLIP đã chỉ ra keyframe ứng viên: nới ra hai bên `radius` frame
    rồi quét dày, vì keyframe I-frame gần như không bao giờ rơi trúng khoảnh khắc.
    """
    lo = max(0, center_frame - radius)
    hi = center_frame + radius
    idxs, imgs = [], []
    for fid, arr in iter_frames(path, lo, hi, step, size):
        idxs.append(fid)
        imgs.append(arr)
    return idxs, imgs


def iter_frames_chon(path, start_frame: int, end_frame: int, step: int = 1,
                     size=None):
    """Như iter_frames nhưng để FFMPEG LỌC, chỉ truyền về frame thực sự cần.

    `iter_frames` truyền MỌI frame từ đầu video qua ống rồi mới lọc trong
    Python. Muốn lấy frame 22000 ở 720p thì phải truyền ~60 GB dữ liệu —
    chậm không chịu nổi. Bộ lọc `select` của ffmpeg giải mã như cũ nhưng
    chỉ xuất ra frame nằm trong khoảng, nên phần truyền gần như bằng 0.

    Chỉ số frame vẫn CHÍNH XÁC TUYỆT ĐỐI: `n` trong bộ lọc select là bộ đếm
    frame của bộ giải mã, đếm từ 0 — cùng quy ước với iter_frames. Với
    `-vsync 0` ffmpeg xuất đúng các frame được chọn, đúng thứ tự, nên chỉ số
    tính được chứ không phải đếm mò. Xem indexing/test_extract_frames.py.
    """
    import imageio_ffmpeg as ff

    lo, hi = max(0, start_frame), end_frame
    # dấu phẩy trong hàm của bộ lọc phải escape, vì ffmpeg dùng phẩy để nối bộ lọc
    chon = f"between(n\\,{lo}\\,{hi})"
    if step > 1:
        chon += f"*not(mod(n-{lo}\\,{step}))"
    vf = [f"select='{chon}'"]
    if size:
        vf.append(f"scale={size[0]}:{size[1]}")

    reader = ff.read_frames(str(path),
                            output_params=["-vf", ",".join(vf), "-vsync", "0"])
    meta = next(reader)
    w, h = size if size else meta["size"]
    try:
        for i, raw in enumerate(reader):
            yield lo + i * step, np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    finally:
        reader.close()


def extract_window_nhanh(path, center_frame: int, radius: int = 90,
                         step: int = 45, size=None):
    """extract_window nhưng dùng bộ lọc của ffmpeg — nhanh hơn nhiều ở full-res."""
    lo = max(0, center_frame - radius)
    idxs, imgs = [], []
    for fid, arr in iter_frames_chon(path, lo, center_frame + radius, step, size):
        idxs.append(fid)
        imgs.append(arr)
    return idxs, imgs


def extract_windows(path, centers, radius: int = 45, step: int = 2, size=None):
    """Lấy NHIỀU cửa sổ trong MỘT lượt giải mã.

    Dùng ở lượt tinh chỉnh của TRAKE: sau lượt quét thô ta có N vị trí thô,
    cần quét dày quanh từng vị trí. Gọi extract_window() N lần sẽ giải mã lại
    video N lần — hàm này chỉ giải mã một lần.

    Trả về (frame_idx tăng dần, ảnh) — không trùng lặp kể cả khi các cửa sổ
    chồng lên nhau.
    """
    cua_so = [(max(0, c - radius), c + radius) for c in sorted(centers)]
    het = max(h for _, h in cua_so)
    idxs, imgs = [], []
    for fid, arr in iter_frames(path, 0, het, 1, size):
        for lo, hi in cua_so:
            if lo <= fid <= hi:
                if (fid - lo) % step == 0:
                    idxs.append(fid)
                    imgs.append(arr)
                break  # chặn thêm hai lần khi các cửa sổ chồng nhau
    return idxs, imgs


def to_pil(arr):
    from PIL import Image
    return Image.fromarray(arr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Dùng: python -m indexing.extract_frames <video> [center] [radius] [step]")
    p = Path(sys.argv[1])
    info = video_info(p)
    print(f"{p.name}: {info['size']}  {info['fps']:.2f} fps  "
          f"{info['duration']:.1f}s  ~{info['n_frames_uoc_tinh']} frame")
    if len(sys.argv) > 2:
        center = int(sys.argv[2])
        radius = int(sys.argv[3]) if len(sys.argv) > 3 else 150
        step = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        idxs, imgs = extract_window(p, center, radius, step)
        print(f"Trích {len(idxs)} frame: {idxs[:5]} ... {idxs[-3:]}")
