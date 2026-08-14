"""DIỄN TẬP NGÀY THI: chạy trọn chuỗi từ questions.csv tới file nộp, có bấm giờ.

    python -m eval.dien_tap --questions eval/questions_dien_tap.csv \
        --index D:/AIC2026/full/index --map D:/AIC2026/full/map-keyframes-.../map-keyframes \
        --keyframes D:/AIC2026/keyframes --videos D:/AIC2026/videos/... \
        --out eval/results/dien_tap.csv

Chỉ chạy KIS thôi (nhanh, xem chuỗi có thông không):
    python -m eval.dien_tap ... --chi-dang kis

VÌ SAO CẦN: ngày 21 sẽ có 100 câu và một giới hạn thời gian. Ai đó phải nhận
đề, chạy, sinh CSV, nộp. Chuỗi đó chưa chạy thử lần nào. Nếu nó gãy hoặc quá
chậm thì điểm số bao nhiêu cũng vô nghĩa.

Script này KHÔNG đo điểm (đáp án đề diễn tập là giả). Nó đo:
  - toàn bộ mất bao lâu, có kịp giờ thi không
  - khâu nào là nút cổ chai
  - file nộp có hợp lệ không
  - có câu nào làm chương trình chết giữa chừng không
"""
import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MAX_NOP = 100
RRF_K = 60


def rrf(ds):
    diem = {}
    for hits in ds:
        for hang, (kf, _) in enumerate(hits, 1):
            diem[kf] = diem.get(kf, 0.0) + 1.0 / (RRF_K + hang)
    return sorted(diem.items(), key=lambda x: -x[1])


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--keyframes", help="để Q&A xem ảnh; thiếu thì Q&A chỉ đoán mù")
    ap.add_argument("--videos", help="để TRAKE trích frame; thiếu thì TRAKE bỏ qua")
    ap.add_argument("--out", default="eval/results/dien_tap.csv")
    ap.add_argument("--chi-dang", choices=["kis", "qa", "trake"],
                    help="chỉ chạy một dạng")
    ap.add_argument("--gioi-han-phut", type=float, default=0,
                    help="dừng và báo cáo khi quá ngần này phút (0 = không giới hạn)")
    ap.add_argument("--khong-llm", action="store_true",
                    help="KHÔNG gọi understand(), dùng thẳng câu tiếng Việt làm "
                         "truy vấn. Đo được thời gian khâu tìm kiếm + sinh file "
                         "mà KHÔNG TỐN API. Thời gian LLM cộng riêng: đo thực tế "
                         "~4-5 giây/câu, tức ~8 phút cho 100 câu.")
    args = ap.parse_args()

    from agent.clip_encoder import ClipEncoder
    from agent.understand import understand
    from eval.write_submission import viet
    from indexing.map_keyframes import MapKeyframes

    idir = Path(args.index)
    ten = next(idir.glob("*.faiss")).stem
    index = faiss.read_index(str(idir / f"{ten}.faiss"))
    ids = json.load(open(idir / f"{ten}_ids.json"))
    mk = MapKeyframes(args.map, args.keyframes)
    enc = ClipEncoder()
    print(f"Index {ten}: {index.ntotal:,} vector | map {len(mk.cac_video())} video\n")

    def tim(text, k=400):
        tv = enc.encode_text([text]).astype("float32")
        D, I = index.search(tv, min(k, index.ntotal))
        return [(ids[i], float(d)) for d, i in zip(D[0], I[0]) if i != -1]

    with open(args.questions, encoding="utf-8-sig", newline="") as f:
        de = list(csv.DictReader(f))
    if args.chi_dang:
        de = [d for d in de if d.get("task_type") == args.chi_dang]
    print(f"{len(de)} câu hỏi\n")

    bai_nop, thoi_gian, loi = [], defaultdict(list), []
    t_dau = time.time()

    for k, d in enumerate(de, 1):
        qid = d["query_id"]
        dang = d.get("task_type", "kis")
        t0 = time.time()
        try:
            vao = d["query_vi"] + (". " + d["question_vi"]
                                   if d.get("question_vi") else "")
            if args.khong_llm:
                cau = [vao]  # CLIP không hiểu tiếng Việt — chỉ để bấm giờ
            else:
                r = understand(vao)
                cau = [r["search_text_en"]] + r.get("variants", [])[:3]
            hits = rrf([tim(c) for c in cau])[:MAX_NOP]

            answers = []
            for kf, _ in hits:
                v, n = kf.split("#")
                try:
                    answers.append({"video_id": v, "frame_id": mk.frame_idx(v, int(n))})
                except (KeyError, FileNotFoundError):
                    continue
            # Q&A/TRAKE: phần trả lời và căn chỉnh chạy ở bước riêng; ở đây đo
            # khâu TÌM KIẾM là khâu mọi dạng đều phải qua.
            bai_nop.append({"qid": qid, "answers": answers})
        except Exception as e:  # noqa: BLE001 — diễn tập phải chạy hết, không dừng
            loi.append((qid, dang, str(e)[:100]))
            bai_nop.append({"qid": qid, "answers": []})

        dt = time.time() - t0
        thoi_gian[dang].append(dt)
        troi = time.time() - t_dau
        if k % 10 == 0 or k == len(de):
            print(f"  {k}/{len(de)}  {troi/60:.1f} phút  "
                  f"(ước tổng {troi/k*len(de)/60:.1f} phút)")
        if args.gioi_han_phut and troi > args.gioi_han_phut * 60:
            print(f"\n*** QUÁ {args.gioi_han_phut} PHÚT — dừng ở câu {k}/{len(de)} ***")
            break

    tong = time.time() - t_dau
    canh_bao = viet(bai_nop, args.out, "kis", None, len(de))

    print(f"\n{'='*58}\nBÁO CÁO DIỄN TẬP")
    print(f"{'='*58}")
    print(f"Tổng thời gian : {tong/60:.1f} phút cho {len(bai_nop)} câu")
    print(f"Trung bình     : {tong/max(1,len(bai_nop)):.1f} giây/câu")
    print("\nTheo dạng đề:")
    for dang, ds in sorted(thoi_gian.items()):
        print(f"  {dang:6} {len(ds):3} câu   trung vị {statistics.median(ds):5.1f}s   "
              f"chậm nhất {max(ds):5.1f}s   tổng {sum(ds)/60:5.1f} phút")

    thieu = [b["qid"] for b in bai_nop if len(b["answers"]) < MAX_NOP]
    print(f"\nCâu nộp chưa đủ 100 đáp án: {len(thieu)}")
    if thieu:
        print(f"  {', '.join(thieu[:8])}{' ...' if len(thieu) > 8 else ''}")
    print(f"Câu lỗi: {len(loi)}")
    for qid, dang, e in loi[:5]:
        print(f"  {qid} [{dang}]: {e}")
    print(f"\nFile nộp: {args.out}")
    print(f"Cảnh báo hợp lệ: {len(canh_bao)}")
    for c in canh_bao[:5]:
        print(f"  {c}")
    if not canh_bao and not loi:
        print("\n✓ Chuỗi thông suốt, file nộp hợp lệ.")


if __name__ == "__main__":
    main()
