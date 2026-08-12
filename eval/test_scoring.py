"""Kiểm chứng công thức chấm điểm bằng CHÍNH các ví dụ BTC đưa trong PDF.

Chạy:  python eval/test_scoring.py

Nếu file này fail, nghĩa là run_benchmark.py đang chấm sai luật -> mọi con số
đo được đều vô nghĩa. Chạy lại sau mỗi lần sửa công thức.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.run_benchmark import (final_score, rscore_kis, rscore_qa,  # noqa: E402
                                rscore_trake)

dat, truot = 0, 0


def check(ten, thuc_te, mong_doi):
    global dat, truot
    ok = abs(thuc_te - mong_doi) < 1e-9
    print(f"{'✓' if ok else '✗'} {ten}: {thuc_te} (mong đợi {mong_doi})")
    dat, truot = dat + ok, truot + (not ok)


# --- PDF muc 2.1.1: KIS, đáp án L01_V001 khung 500-510 ---
gt_kis = {"video_id": "L01_V001", "s": 500, "e": 510}
check("KIS L01_V001,505 -> đúng",
      rscore_kis({"video_id": "L01_V001", "frame_id": 505}, gt_kis), 1.0)
check("KIS L01_V001,600 -> ngoài khoảng",
      rscore_kis({"video_id": "L01_V001", "frame_id": 600}, gt_kis), 0.0)
check("KIS L02_V003,505 -> sai video",
      rscore_kis({"video_id": "L02_V003", "frame_id": 505}, gt_kis), 0.0)
check("KIS biên s=500", rscore_kis({"video_id": "L01_V001", "frame_id": 500}, gt_kis), 1.0)
check("KIS biên e=510", rscore_kis({"video_id": "L01_V001", "frame_id": 510}, gt_kis), 1.0)

# --- PDF muc 2.1.2: Q&A, L05_V005 khung 800-900, đáp án "màu xanh" ---
gt_qa = {"video_id": "L05_V005", "s": 800, "e": 900, "answer": "màu xanh"}
check("QA đúng hết",
      rscore_qa({"video_id": "L05_V005", "frame_id": 888, "answer": "màu xanh"}, gt_qa), 1.0)
check("QA sai answer",
      rscore_qa({"video_id": "L05_V005", "frame_id": 888, "answer": "màu trắng"}, gt_qa), 0.0)
check("QA sai video",
      rscore_qa({"video_id": "L06_V007", "frame_id": 888, "answer": "màu xanh"}, gt_qa), 0.0)
check("QA thiếu answer",
      rscore_qa({"video_id": "L05_V005", "frame_id": 888}, gt_qa), 0.0)

# --- PDF muc 2.1.3: TRAKE, L10_V010, 4 khoảnh khắc, nộp 101/156/203/251 ---
gt_trake = {"video_id": "L10_V010",
            "spans": [[95, 105], [145, 155], [195, 205], [245, 255]]}
check("TRAKE khớp 3/4 (ví dụ PDF)",
      rscore_trake({"video_id": "L10_V010",
                    "frame_ids": [101, 156, 203, 251]}, gt_trake), 0.75)
check("TRAKE sai video -> 0 ngay",
      rscore_trake({"video_id": "L99_V999",
                    "frame_ids": [101, 151, 203, 251]}, gt_trake), 0.0)
check("TRAKE khớp cả 4",
      rscore_trake({"video_id": "L10_V010",
                    "frame_ids": [100, 150, 200, 250]}, gt_trake), 1.0)
check("TRAKE nộp thiếu frame -> tính là trượt",
      rscore_trake({"video_id": "L10_V010", "frame_ids": [100, 150]}, gt_trake), 0.5)

# --- PDF muc 2.2: Final Score ---
# "câu đầu tiên 0.5; câu số 3 là 0.8 (cao nhất); câu số 15 là 0.6"
vi_du = [0.5, 0.0, 0.8] + [0.0] * 11 + [0.6] + [0.0] * 85
fs, r_at_k = final_score(vi_du)
check("Final Score ví dụ PDF (0.5/0.8/0.6)", round(fs, 4), 0.74)
check("  R@1", r_at_k[0], 0.5)
check("  R@5", r_at_k[1], 0.8)
check("  R@100", r_at_k[4], 0.8)

# --- Bảng hệ quả: đúng ở hạng r thì được bao nhiêu điểm ---
print("\nHạng của câu đúng đầu tiên -> Final Score:")
for hang, mong_doi in [(1, 1.0), (2, 0.8), (5, 0.8), (6, 0.6), (20, 0.6),
                       (21, 0.4), (50, 0.4), (51, 0.2), (100, 0.2)]:
    diem = [0.0] * (hang - 1) + [1.0] + [0.0] * (100 - hang)
    check(f"  hạng {hang}", round(final_score(diem)[0], 4), mong_doi)

print(f"\nKẾT QUẢ: {dat}/{dat + truot}")
sys.exit(1 if truot else 0)
