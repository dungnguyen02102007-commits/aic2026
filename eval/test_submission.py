"""Kiểm chứng file nộp bài đúng định dạng BTC.

Chạy:  python -m eval.test_submission

Sai định dạng là MẤT TRẮNG công cả tuần, mà lỗi thì không hiện ra ở đâu cả —
nên mọi ràng buộc BTC ghi đều phải có một phép thử canh giữ.
"""
import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from eval.write_submission import SO_DAP_AN, o_dap_an, ten_video, viet  # noqa: E402

dat = truot = 0


def check(ten, ok, chi_tiet=""):
    global dat, truot
    print(f"{'✓' if ok else '✗'} {ten}" + (f"  — {chi_tiet}" if chi_tiet else ""))
    dat, truot = dat + bool(ok), truot + (not ok)


def doc(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


tmp = Path(tempfile.mkdtemp(prefix="aic_nop_"))
try:
    # ---------- Định dạng ô ----------
    print("=== Định dạng ô câu trả lời ===")
    a = {"video_id": "L21_V001", "frame_id": 275}
    check("thêm đuôi .mp4 (hệ thống nội bộ không có đuôi)",
          o_dap_an(a, "kis") == "L21_V001.mp4,275", o_dap_an(a, "kis"))
    check("đã có .mp4 thì không thêm lần nữa",
          ten_video("video1234.mp4") == "video1234.mp4")
    check("khớp đúng ví dụ trong tài liệu BTC",
          o_dap_an({"video_id": "video1234", "frame_id": 275}, "kis")
          == "video1234.mp4,275")

    ans_qa = {"video_id": "L21_V001", "frame_id": 888, "answer": "màu xanh"}
    check("Q&A gắn thêm đáp án chữ",
          o_dap_an(ans_qa, "qa") == "L21_V001.mp4,888,màu xanh")
    # Dấu phẩy trong đáp án sẽ bị hiểu thành trường mới -> phải bỏ
    ans_phay = {"video_id": "V", "frame_id": 1, "answer": "5 người, áo đỏ"}
    check("đáp án có dấu phẩy phải được làm sạch",
          "," not in o_dap_an(ans_phay, "qa").split(",", 2)[2],
          o_dap_an(ans_phay, "qa"))

    tr = {"video_id": "L21_V010", "frame_ids": [101, 156, 203, 251]}
    check("TRAKE ghi nhiều khung hình",
          o_dap_an(tr, "trake") == "L21_V010.mp4,101,156,203,251")

    # ---------- Bù cho đủ 100 ----------
    print("\n=== Đủ 100 câu trả lời ===")
    bn = [{"qid": "Q_0001", "answers": [{"video_id": "V1", "frame_id": i}
                                        for i in range(3)]}]
    out = tmp / "thieu.csv"
    cb = viet(bn, out, "kis", so_query_mong_doi=1)
    rows = doc(out)
    check("dòng tiêu đề đúng 101 cột", len(rows[0]) == SO_DAP_AN + 1,
          f"{len(rows[0])} cột")
    check("tiêu đề đặt tên đúng chuẩn",
          rows[0][:3] == ["query_id", "answer_001", "answer_002"]
          and rows[0][-1] == "answer_100")
    check("nộp thiếu -> bù cho đủ 100 ô", len(rows[1]) == SO_DAP_AN + 1,
          f"{len(rows[1])} cột")
    check("có cảnh báo khi phải bù",
          any("bù thêm" in c for c in cb), f"{cb}")
    check("3 ô đầu giữ nguyên thứ hạng gốc",
          rows[1][1:4] == ["V1.mp4,0", "V1.mp4,1", "V1.mp4,2"])

    bn = [{"qid": "Q_0001", "answers": [{"video_id": "V1", "frame_id": i}
                                        for i in range(150)]}]
    out = tmp / "thua.csv"
    cb = viet(bn, out, "kis", so_query_mong_doi=1)
    rows = doc(out)
    check("nộp thừa -> cắt còn đúng 100", len(rows[1]) == SO_DAP_AN + 1)
    check("cắt phần ĐUÔI, giữ thứ hạng đầu",
          rows[1][1] == "V1.mp4,0" and rows[1][100] == "V1.mp4,99")

    # ---------- Bọc ngoặc kép ----------
    print("\n=== Bọc ngoặc kép (ô chứa dấu phẩy) ===")
    raw = (tmp / "thua.csv").read_text(encoding="utf-8").splitlines()[1]
    check("ô có dấu phẩy được bọc ngoặc kép trong file thô",
          raw.startswith('Q_0001,"V1.mp4,0"'), raw[:40])
    check("đọc lại bằng csv vẫn tách đúng ô",
          doc(tmp / "thua.csv")[1][1] == "V1.mp4,0")

    # ---------- Kiểm tra theo corpus.csv ----------
    print("\n=== Đối chiếu corpus.csv ===")
    corpus = tmp / "corpus.csv"
    corpus.write_text("video_filename,frame_count\nV1.mp4,1000\nV2.mp4,50\n",
                      encoding="utf-8")

    bn = [{"qid": "Q_0001", "answers": [{"video_id": "V1", "frame_id": 999}]}]
    cb = viet(bn, tmp / "ok.csv", "kis", corpus, 1)
    check("frame_idx trong khoảng -> không cảnh báo về frame",
          not any("ngoài khoảng" in c for c in cb), f"{cb}")

    bn = [{"qid": "Q_0001", "answers": [{"video_id": "V1", "frame_id": 1000}]}]
    cb = viet(bn, tmp / "qua.csv", "kis", corpus, 1)
    check("frame_idx = frame_count -> BÁO LỖI (chỉ số đếm từ 0)",
          any("ngoài khoảng" in c for c in cb), f"{cb[:1]}")

    bn = [{"qid": "Q_0001", "answers": [{"video_id": "V9", "frame_id": 5}]}]
    cb = viet(bn, tmp / "la.csv", "kis", corpus, 1)
    check("video không có trong corpus -> BÁO LỖI",
          any("KHÔNG có trong corpus" in c for c in cb), f"{cb[:1]}")

    bn = [{"qid": "Q_0001", "answers": [{"video_id": "V1", "frame_id": -1}]}]
    cb = viet(bn, tmp / "am.csv", "kis", corpus, 1)
    check("frame_idx âm -> BÁO LỖI", any("ngoài khoảng" in c for c in cb))

    # TRAKE: mọi khung hình đều phải nằm trong khoảng
    bn = [{"qid": "Q_0001", "answers": [{"video_id": "V2",
                                         "frame_ids": [10, 20, 60]}]}]
    cb = viet(bn, tmp / "trake.csv", "trake", corpus, 1)
    check("TRAKE: kiểm TỪNG khung hình, không chỉ khung đầu",
          any("frame 60" in c for c in cb), f"{cb[:1]}")

    # ---------- Số lượng truy vấn ----------
    print("\n=== Số lượng truy vấn ===")
    bn = [{"qid": f"Q_{i:04d}", "answers": [{"video_id": "V1", "frame_id": 1}]}
          for i in range(3)]
    cb = viet(bn, tmp / "it.csv", "kis", None, 100)
    check("thiếu truy vấn so với yêu cầu -> BÁO LỖI",
          any("BTC yêu cầu đúng 100" in c for c in cb), f"{cb[-1:]}")

    bn = [{"qid": "Q_0001", "answers": []}]
    cb = viet(bn, tmp / "rong.csv", "kis", None, 1)
    check("truy vấn rỗng -> báo lỗi, không ghi dòng hỏng",
          any("KHÔNG có câu trả lời" in c for c in cb)
          and len(doc(tmp / "rong.csv")) == 1)
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print(f"\nKẾT QUẢ: {dat}/{dat + truot}")
sys.exit(1 if truot else 0)
