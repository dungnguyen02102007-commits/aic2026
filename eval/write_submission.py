"""Sinh file nộp bài đúng định dạng BTC + kiểm tra hợp lệ (issue #33).

    python eval/write_submission.py --submissions eval/results/bai_nop_kis.jsonl \
        --out nop.csv --corpus corpus.csv

Định dạng BTC:

    query_id,answer_001,answer_002,...,answer_100
    Q_0001,"video1234.mp4,275","video0198.mp4,120",...

- Đúng **100 dòng**, mỗi dòng một `query_id` trong `questions.csv`
- Mỗi truy vấn đúng **100 câu trả lời đã xếp hạng**; thứ hạng = thứ tự cột
- Mỗi ô: `video_filename.mp4,frame_idx` (có dấu phẩy nên phải bọc ngoặc kép)
- `frame_idx` là số nguyên không âm và **nhỏ hơn `frame_count`** của video đó

BỐN CHỖ SAI LÀ MẤT TRẮNG, không có lỗi nào báo ra:

  1. Thiếu đuôi `.mp4`. Hệ thống nội bộ dùng `L21_V001`, bài nộp cần
     `L21_V001.mp4`.
  2. Nộp chưa đủ 100 câu mỗi truy vấn -> BTC coi là không hợp lệ.
  3. `frame_idx >= frame_count` -> khung hình không tồn tại.
  4. Nhầm `n` (số thứ tự keyframe) với `frame_idx` (chỉ số khung hình trong
     video gốc). Xem docs/onboarding/00 mục 3.

⚠️ Định dạng ô ở trên là thứ BTC ghi rõ. Với Q&A và TRAKE, đáp án còn có
thêm phần chữ / nhiều khung hình mà tài liệu chưa nói cách ghi. Tham số
--task để chọn, mặc định `kis` là bản chắc chắn đúng. PHẢI HỎI BTC trước khi
dùng hai bản kia.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SO_DAP_AN = 100


def doc_corpus(path):
    """corpus.csv -> {ten_file_video: frame_count}. Chấp nhận vài cách đặt tên cột."""
    bang = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            khoa = {k.lower().strip(): v for k, v in row.items() if k}
            ten = (khoa.get("video_filename") or khoa.get("filename")
                   or khoa.get("video") or khoa.get("video_id"))
            fc = (khoa.get("frame_count") or khoa.get("n_frames")
                  or khoa.get("num_frames"))
            if ten:
                bang[ten.strip()] = int(fc) if fc and str(fc).strip().isdigit() else None
    return bang


def ten_video(v: str) -> str:
    """L21_V001 -> L21_V001.mp4. Đã có đuôi thì giữ nguyên."""
    return v if v.lower().endswith(".mp4") else f"{v}.mp4"


def o_dap_an(a: dict, task: str) -> str:
    v = ten_video(a["video_id"])
    if task == "trake":
        fs = a.get("frame_ids") or [a.get("frame_id")]
        return ",".join([v] + [str(int(f)) for f in fs])
    if task == "qa":
        # Đáp án chữ không được chứa dấu phẩy — nó sẽ bị hiểu là một trường mới
        ans = str(a.get("answer", "")).replace(",", " ").strip()
        return f"{v},{int(a['frame_id'])},{ans}"
    return f"{v},{int(a['frame_id'])}"


def viet(bai_nop, out, task="kis", corpus=None, so_query_mong_doi=None):
    """bai_nop: [{"qid": str, "answers": [...]}]. Trả về danh sách cảnh báo."""
    canh_bao = []
    fc = doc_corpus(corpus) if corpus else {}

    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(["query_id"] + [f"answer_{i:03d}" for i in range(1, SO_DAP_AN + 1)])

        for b in bai_nop:
            ans = list(b["answers"])

            for a in ans:
                v = ten_video(a["video_id"])
                if fc:
                    if v not in fc:
                        canh_bao.append(f"{b['qid']}: video '{v}' KHÔNG có trong corpus.csv")
                    elif fc[v] is not None:
                        khung = (a.get("frame_ids") or [a.get("frame_id")])
                        for k in khung:
                            if k is None or int(k) < 0 or int(k) >= fc[v]:
                                canh_bao.append(
                                    f"{b['qid']}: frame {k} ngoài khoảng hợp lệ "
                                    f"[0, {fc[v]-1}] của {v}")

            if len(ans) > SO_DAP_AN:
                canh_bao.append(f"{b['qid']}: có {len(ans)} câu, cắt còn {SO_DAP_AN}")
                ans = ans[:SO_DAP_AN]
            elif len(ans) < SO_DAP_AN:
                if not ans:
                    canh_bao.append(f"{b['qid']}: KHÔNG có câu trả lời nào — bỏ dòng")
                    continue
                # Bù bằng cách lặp lại câu cuối: BTC bắt buộc đủ 100 ô, mà lặp
                # lại một cặp (video, frame) HỢP LỆ thì không làm tụt điểm —
                # R@k chỉ lấy max, câu trùng không kéo điểm xuống.
                canh_bao.append(f"{b['qid']}: chỉ có {len(ans)} câu, bù thêm "
                                f"{SO_DAP_AN - len(ans)} bằng câu cuối")
                ans = ans + [ans[-1]] * (SO_DAP_AN - len(ans))

            w.writerow([b["qid"]] + [o_dap_an(a, task) for a in ans])

    n = len([b for b in bai_nop if b["answers"]])
    mong_doi = so_query_mong_doi or SO_DAP_AN
    if n != mong_doi:
        canh_bao.append(f"File có {n} truy vấn, BTC yêu cầu đúng {mong_doi}")
    return canh_bao


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", required=True, help="file .jsonl nội bộ")
    ap.add_argument("--out", required=True, help="file .csv nộp cho BTC")
    ap.add_argument("--task", default="kis", choices=["kis", "qa", "trake"])
    ap.add_argument("--corpus", help="corpus.csv của BTC để kiểm frame_count")
    ap.add_argument("--n-query", type=int, default=SO_DAP_AN,
                    help="số truy vấn mong đợi (BTC: 100)")
    args = ap.parse_args()

    bai_nop = [json.loads(l) for l in open(args.submissions, encoding="utf-8-sig")
               if l.strip()]
    canh_bao = viet(bai_nop, args.out, args.task, args.corpus, args.n_query)

    print(f"Đã ghi {args.out}  ({len(bai_nop)} truy vấn, dạng {args.task})")
    if canh_bao:
        print(f"\n⚠️ {len(canh_bao)} CẢNH BÁO:")
        for c in canh_bao[:30]:
            print(f"   {c}")
        if len(canh_bao) > 30:
            print(f"   ... và {len(canh_bao)-30} cảnh báo nữa")
        sys.exit(1)
    print("✓ Không có cảnh báo nào — file hợp lệ")


if __name__ == "__main__":
    main()
