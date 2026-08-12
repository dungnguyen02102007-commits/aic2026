"""Test TRAKE xuyên suốt: video -> trích frame -> CLIP -> căn chỉnh -> frame_id.

Chạy:  python -m agent.test_trake_e2e

Dựng một video có 3 đoạn màu rõ rệt (đỏ / xanh lá / xanh dương), rồi hỏi bằng
3 mô tả tiếng Anh tương ứng. Hệ thống phải chọn ra 3 frame nằm ĐÚNG ba đoạn đó,
theo đúng thứ tự thời gian.

Đây là bài test ngữ nghĩa thật (CLIP phải phân biệt được màu), không phải chỉ
kiểm tra đường ống chạy không lỗi.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.trake import align_events, solve_alignment  # noqa: E402

FPS, W, H = 25, 256, 144
DOAN = [((220, 30, 30), "a solid bright red screen"),
        ((30, 200, 30), "a solid bright green screen"),
        ((30, 40, 220), "a solid bright blue screen")]
MOI_DOAN = 100  # frame  ->  đỏ 0-99, lục 100-199, lam 200-299

dat = truot = 0


def check(ten, ok, chi_tiet=""):
    global dat, truot
    print(f"{'✓' if ok else '✗'} {ten}" + (f"  — {chi_tiet}" if chi_tiet else ""))
    dat, truot = dat + bool(ok), truot + (not ok)


def dung_video(path):
    import imageio_ffmpeg as ff

    writer = ff.write_frames(str(path), (W, H), fps=FPS, macro_block_size=1,
                             quality=8)
    writer.send(None)
    for (mau, _) in DOAN:
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        frame[:, :] = mau
        for _ in range(MOI_DOAN):
            writer.send(frame.tobytes())
    writer.close()


tmp = Path(tempfile.mkdtemp(prefix="aic_trake_e2e_"))
try:
    video = tmp / "ba_doan_mau.mp4"
    dung_video(video)
    print(f"Video thử: 3 đoạn × {MOI_DOAN} frame — đỏ 0-99, lục 100-199, lam 200-299\n")

    from agent.clip_encoder import ClipEncoder
    encoder = ClipEncoder()

    events = [{"name": f"đoạn {i + 1}", "search_en": mo_ta}
              for i, (_, mo_ta) in enumerate(DOAN)]

    # Quét cả video: tâm frame 150, bán kính 150 -> phủ 0..300
    chosen, total, scores = solve_alignment(video, center_frame=150,
                                            events=events, encoder=encoder,
                                            radius=150, step=5)
    check("chạy xuyên suốt, trả đủ 3 frame_id",
          chosen is not None and len(chosen) == 3, f"{chosen}")

    if chosen:
        khoang = [(0, 99), (100, 199), (200, 299)]
        ten = ["đỏ", "lục", "lam"]
        for i, (f, (lo, hi)) in enumerate(zip(chosen, khoang)):
            check(f"khoảnh khắc {i + 1} ({ten[i]}) rơi vào đúng đoạn [{lo},{hi}]",
                  lo <= f <= hi, f"frame {f}")

        check("thứ tự thời gian tăng dần",
              all(chosen[i] < chosen[i + 1] for i in range(len(chosen) - 1)),
              f"{chosen}")

        # frame_id phải là chỉ số THẬT trong video, không phải vị trí trong
        # danh sách ứng viên (danh sách chỉ có ~60 phần tử vì step=5)
        check("frame_id là chỉ số thật trong video, không phải vị trí mảng",
              max(chosen) > len(scores[0]),
              f"max={max(chosen)}, số ứng viên={len(scores[0])}")

        print(f"\n  điểm tổng: {total:.3f}")
        print(f"  frame nộp: {chosen}")

    # --- Ràng buộc thứ tự có thật sự thay đổi kết quả không? ---
    # Đảo ngược thứ tự sự kiện: hỏi lam trước, đỏ sau. Vì lam nằm CUỐI video,
    # ràng buộc tăng dần buộc hệ thống phải nhượng bộ.
    events_dao = list(reversed(events))
    chosen2, _, scores2 = solve_alignment(video, 150, events_dao, encoder,
                                          radius=150, step=5)
    argmax_tu_do = [max(range(len(row)), key=lambda i: row[i]) for row in scores2]
    check("truy vấn đảo thứ tự: argmax tự do vi phạm thời gian",
          argmax_tu_do != sorted(argmax_tu_do), f"argmax={argmax_tu_do}")
    check("nhưng kết quả cuối vẫn tăng dần (ràng buộc có hiệu lực)",
          chosen2 is not None
          and all(chosen2[i] < chosen2[i + 1] for i in range(len(chosen2) - 1)),
          f"{chosen2}")

    # --- Không đủ frame ---
    ket_qua = align_events([[0.5, 0.5]] * 5)
    check("nhiều sự kiện hơn frame -> không trả bừa", ket_qua[0] is None)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nKẾT QUẢ: {dat}/{dat + truot}")
sys.exit(1 if truot else 0)
