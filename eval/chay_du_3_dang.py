"""Chạy ĐỦ 3 DẠNG ĐỀ từ questions.csv tới file nộp BTC — bản dùng ngày thi.

    python -m eval.chay_du_3_dang \
        --questions eval/questions_dien_tap.csv \
        --index D:/AIC2026/full/index D:/AIC2026/full/index_l14 \
        --map D:/AIC2026/full/map-keyframes-.../map-keyframes \
        --keyframes D:/AIC2026/keyframes \
        --out eval/results/nop_day_du.csv

Khác `dien_tap.py` (chỉ chạy khâu tìm kiếm): script này chạy trọn cả ba dạng —
KIS, Q&A (Claude nhìn ảnh trả lời), TRAKE (căn chỉnh chuỗi khoảnh khắc).

TRAKE CHẠY BẰNG KEYFRAME, KHÔNG CẦN VIDEO. Đội chỉ có 29/873 video trên máy,
nếu TRAKE bắt buộc phải giải mã video thì hầu hết truy vấn sẽ không chạy được
vào ngày thi. Căn chỉnh trên chuỗi keyframe của video ứng viên vẫn cho ra
frame_idx hợp lệ và tăng dần — độ chính xác bị giới hạn bởi khoảng cách
keyframe (trung vị 53 frame) nhưng CHẠY ĐƯỢC TRÊN MỌI VIDEO và nhanh gấp bội.
Có file video thì tinh chỉnh thêm bằng eval/run_trake.py.
"""
import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

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
    ap.add_argument("--index", required=True, nargs="+")
    ap.add_argument("--map", required=True)
    ap.add_argument("--keyframes", help="ảnh keyframe, cần cho Q&A")
    ap.add_argument("--out", default="eval/results/nop_day_du.csv")
    ap.add_argument("--n-ung-vien-qa", type=int, default=8)
    ap.add_argument("--n-video-trake", type=int, default=3)
    ap.add_argument("--gioi-han-phut", type=float, default=0)
    args = ap.parse_args()

    from agent.clip_encoder import ClipEncoder
    from agent.trake import align_events, decompose, rank_videos_by_events
    from agent.understand import understand
    from agent.vqa import chon_va_tra_loi
    from eval.write_submission import viet
    from indexing.map_keyframes import MapKeyframes

    THEO_CHIEU = {512: ("ViT-B-32-quickgelu", "openai"),
                  768: ("ViT-L-14", "laion2b_s32b_b82k")}
    kho = []
    for d in args.index:
        idir = Path(d)
        ten = next(idir.glob("*.faiss")).stem
        ix = faiss.read_index(str(idir / f"{ten}.faiss"))
        iids = json.load(open(idir / f"{ten}_ids.json"))
        m, p = THEO_CHIEU[ix.d]
        kho.append({"ten": ten, "ix": ix, "ids": iids,
                    "vt": {k: i for i, k in enumerate(iids)},
                    "enc": ClipEncoder(m, p)})
        print(f"Index {ten}: {ix.ntotal:,} × {ix.d} chiều -> {m}")
    print(f"FUSION {len(kho)} index\n")

    mk = MapKeyframes(args.map, args.keyframes)

    def tim(text, k=400):
        """Trả về danh sách hits của TỪNG index (RRF gộp ở ngoài)."""
        ra = []
        for h in kho:
            tv = h["enc"].encode_text([text]).astype("float32")
            D, I = h["ix"].search(tv, min(k, h["ix"].ntotal))
            # D, I có dạng (1, k) — phải lấy hàng đầu, nếu zip thẳng thì d và i
            # là MẢNG chứ không phải số, và mọi so sánh sẽ nổ lỗi mơ hồ
            ra.append([(h["ids"][int(i)], float(d))
                       for d, i in zip(D[0], I[0]) if i != -1])
        return ra

    def diem_tren_video(video_id, texts):
        """Điểm của TỪNG keyframe trong một video, với TỪNG mô tả. Gộp 2 model.

        Dùng cho TRAKE: cần chấm toàn bộ chuỗi keyframe của video ứng viên rồi
        chọn bộ tăng dần tối ưu — không cần đụng tới file video.
        """
        ns = [n for n, _, _ in mk.danh_sach(video_id)]
        tong = np.zeros((len(texts), len(ns)))
        for h in kho:
            vi = [h["vt"].get(f"{video_id}#{n:06d}") for n in ns]
            co = [(j, i) for j, i in enumerate(vi) if i is not None]
            if not co:
                continue
            M = np.stack([h["ix"].reconstruct(i) for _, i in co])
            T = h["enc"].encode_text(texts)
            S = T @ M.T
            for k, (j, _) in enumerate(co):
                tong[:, j] += S[:, k]
        return ns, tong / max(1, len(kho))

    with open(args.questions, encoding="utf-8-sig", newline="") as f:
        de = list(csv.DictReader(f))
    print(f"{len(de)} câu hỏi\n")

    bai_nop, tg, loi = [], defaultdict(list), []
    t_dau = time.time()

    for k, d in enumerate(de, 1):
        qid, dang = d["query_id"], d.get("task_type", "kis")
        t0 = time.time()
        answers = []
        try:
            if dang == "trake":
                r = decompose(d["query_vi"] + (
                    ". Các khoảnh khắc: " + d["events_vi"].replace("|", ",")
                    if d.get("events_vi") else ""))
                texts = [e["search_en"] for e in r["events"]]
                hits_ev = [rrf(tim(t)) for t in texts]
                uv = rank_videos_by_events(
                    hits_ev,
                    lambda kf: {"video_id": kf.split("#")[0], "frame_idx": 0},
                    args.n_video_trake)
                for v in uv:
                    ns, S = diem_tren_video(v["video_id"], texts)
                    duong, _ = align_events(S.tolist())
                    if duong is None:
                        continue
                    fids = [mk.frame_idx(v["video_id"], ns[i]) for i in duong]
                    answers.append({"video_id": v["video_id"], "frame_ids": fids})
            else:
                vao = d["query_vi"] + (". " + d["question_vi"]
                                       if d.get("question_vi") else "")
                r = understand(vao)
                cau = [r["search_text_en"]] + r.get("variants", [])[:3]
                hits = rrf([x for c in cau for x in tim(c)])[:MAX_NOP]

                dap_an = None
                if dang == "qa" and args.keyframes:
                    uv, thay = [], set()
                    for kf, _ in hits:
                        v, n = kf.split("#")
                        if v in thay:
                            continue
                        thay.add(v)
                        imgs, fids = [], []
                        for dn in (-1, 0, 1):
                            p = mk.duong_dan_anh(v, int(n) + dn)
                            if p:
                                from PIL import Image
                                imgs.append(Image.open(p).convert("RGB"))
                                fids.append(mk.frame_idx(v, int(n) + dn))
                        if imgs:
                            uv.append({"video_id": v, "images": imgs,
                                       "frame_ids": fids})
                        if len(uv) >= args.n_ung_vien_qa:
                            break
                    if uv:
                        kq = chon_va_tra_loi(uv, d["question_vi"], d["query_vi"])
                        dap_an = kq["answer"]
                        chon = uv[kq["ung_vien"]]["video_id"]
                        hits = ([h for h in hits if h[0].startswith(chon + "#")]
                                + [h for h in hits if not h[0].startswith(chon + "#")])

                for kf, _ in hits:
                    v, n = kf.split("#")
                    try:
                        a = {"video_id": v, "frame_id": mk.frame_idx(v, int(n))}
                    except (KeyError, FileNotFoundError):
                        continue
                    if dap_an is not None:
                        a["answer"] = dap_an
                    answers.append(a)
        except Exception as e:  # noqa: BLE001 — diễn tập phải chạy hết
            loi.append((qid, dang, str(e)[:90]))

        bai_nop.append({"qid": qid, "answers": answers, "_dang": dang})
        tg[dang].append(time.time() - t0)
        troi = time.time() - t_dau
        if k % 5 == 0 or k == len(de):
            print(f"  {k}/{len(de)}  {troi/60:.1f} phút  "
                  f"(ước tổng {troi/k*len(de)/60:.1f} phút)")
        if args.gioi_han_phut and troi > args.gioi_han_phut * 60:
            print(f"\n*** Dừng ở câu {k}/{len(de)} do quá giới hạn ***")
            break

    tong = time.time() - t_dau

    # BTC gộp cả 3 dạng vào MỘT file; ô câu trả lời khác nhau theo dạng
    for b in bai_nop:
        b["answers"] = b["answers"] or []
    from eval.write_submission import o_dap_an  # noqa: F401  (dùng qua viet)
    canh_bao = []
    for dg in ("kis", "qa", "trake"):
        phan = [b for b in bai_nop if b["_dang"] == dg]
        if phan:
            canh_bao += viet(phan, str(Path(args.out).with_suffix(f".{dg}.csv")),
                             dg, None, len(phan))

    print(f"\n{'='*58}\nBÁO CÁO DIỄN TẬP ĐẦY ĐỦ\n{'='*58}")
    print(f"Tổng: {tong/60:.1f} phút cho {len(bai_nop)} câu")
    for dg, ds in sorted(tg.items()):
        print(f"  {dg:6} {len(ds):3} câu  trung vị {statistics.median(ds):6.1f}s"
              f"  chậm nhất {max(ds):6.1f}s  tổng {sum(ds)/60:5.1f} phút")
    thieu = [b["qid"] for b in bai_nop if len(b["answers"]) < MAX_NOP]
    print(f"\nCâu nộp chưa đủ 100 đáp án: {len(thieu)}"
          + (f"  ({', '.join(thieu[:6])}...)" if thieu else ""))
    print(f"Câu lỗi: {len(loi)}")
    for qid, dg, e in loi[:6]:
        print(f"  {qid} [{dg}]: {e}")
    print(f"Cảnh báo hợp lệ: {len(canh_bao)}")
    for c in canh_bao[:6]:
        print(f"  {c}")


if __name__ == "__main__":
    main()
