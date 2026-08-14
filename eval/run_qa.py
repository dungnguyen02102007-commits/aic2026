"""Chạy Q&A xuyên suốt và sinh file bài nộp (dạng 2, docs/onboarding/10).

    python eval/run_qa.py --queries eval/queries_qa_L21.jsonl \
        --index D:/AIC2026/index_L21 --videos D:/AIC2026/videos/Videos_L21_a/video

Đường ống:
  1. Tìm khoảnh khắc y như KIS (understand -> CLIP -> RRF -> top 100)
  2. Trích vài frame quanh ứng viên số 1, đưa cho Claude XEM
  3. Claude trả lời câu hỏi
  4. Nộp 100 dòng, TẤT CẢ dùng chung một đáp án

Vì sao chung một đáp án: đáp án là của SỰ KIỆN chứ không của từng khung hình.
Nếu mỗi dòng một đáp án khác nhau thì dòng may mắn trúng đúng frame lại có thể
mang đáp án sai — tự bắn vào chân mình.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.clip_encoder import ClipEncoder  # noqa: E402
from agent.understand import understand  # noqa: E402
from agent.vqa import chon_va_tra_loi, doc_ky  # noqa: E402
from indexing.extract_frames import extract_window_nhanh  # noqa: E402

MAX_NOP = 100
RRF_K = 60


def rrf(danh_sach_hits):
    diem = {}
    for hits in danh_sach_hits:
        for hang, (kf_id, _) in enumerate(hits, start=1):
            diem[kf_id] = diem.get(kf_id, 0.0) + 1.0 / (RRF_K + hang)
    return sorted(diem.items(), key=lambda x: -x[1])


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--videos", help="thư mục video; chỉ cần khi KHÔNG có --keyframes")
    ap.add_argument("--keyframes", help="thư mục ảnh keyframe BTC. Có nó thì đọc "
                                        "ảnh TỨC THÌ thay vì giải mã video "
                                        "55-70 giây mỗi ứng viên — đây là nút "
                                        "thắt của Q&A")
    ap.add_argument("--map", help="thư mục map-keyframes. BẮT BUỘC khi index "
                                  "dùng ID theo `n` (index từ CLIPFeatures BTC)")
    ap.add_argument("--out", default="eval/results/bai_nop_qa.jsonl")
    ap.add_argument("--n-frame", type=int, default=3,
                    help="số khung hình mỗi ứng viên")
    ap.add_argument("--n-ung-vien", type=int, default=4,
                    help="số khoảnh khắc đưa cho Claude tự chọn")
    ap.add_argument("--gan-nhau", type=int, default=300,
                    help="hai hit cùng video cách dưới ngần này frame coi là "
                         "cùng một khoảnh khắc")
    ap.add_argument("--ban-kinh", type=int, default=90,
                    help="nửa bề rộng cửa sổ trích frame (90 ~ 3 giây ở 30fps)")
    ap.add_argument("--n-luoi", type=int, default=2,
                    help="lưới phóng to n×n ở lượt đọc kỹ (2 = 4 vùng)")
    ap.add_argument("--khong-doc-ky", action="store_true",
                    help="bỏ lượt đọc kỹ có phóng to — để đo xem nó giúp gì")
    ap.add_argument("--cache", default="eval/results/qa_cache.json",
                    help="lưu/dùng lại đáp án. Q&A là dạng TỐN TIỀN NHẤT "
                         "(mỗi câu 2 lượt gọi kèm ~25 ảnh) nên chạy lại để so "
                         "cấu hình mà không cache là đốt tiền vô ích")
    ap.add_argument("--refresh-cache", action="store_true")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    qa_cache = {}
    if cache_path.exists() and not args.refresh_cache:
        qa_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Dùng lại {len(qa_cache)} đáp án đã lưu từ {cache_path}")

    idir = Path(args.index)
    ten = next(idir.glob("*.faiss")).stem
    index = faiss.read_index(str(idir / f"{ten}.faiss"))
    ids = json.load(open(idir / f"{ten}_ids.json"))
    print(f"Index {ten}: {index.ntotal} vector")

    # ID trong index có hai kiểu: "<video>#<frame_idx>" (index tự dựng từ video)
    # hoặc "<video>#<n>" (index từ CLIPFeatures BTC). Có --map thì là kiểu thứ
    # hai, phải đổi n -> frame_idx mới nộp bài được.
    mk_anh = None
    if args.map:
        from indexing.map_keyframes import MapKeyframes
        mk_anh = MapKeyframes(args.map, args.keyframes)
        print(f"map-keyframes: {len(mk_anh.cac_video())} video"
              + (f" | ảnh keyframe: {args.keyframes}" if args.keyframes else ""))
    elif args.keyframes:
        sys.exit("--keyframes cần đi kèm --map (để đổi n sang frame_idx)")
    elif not args.videos:
        sys.exit("Cần --keyframes + --map, hoặc --videos")

    encoder = ClipEncoder()

    def tim(text, k=400):
        tv = encoder.encode_text([text]).astype("float32")
        D, I = index.search(tv, min(k, index.ntotal))
        return [(ids[i], float(d)) for d, i in zip(D[0], I[0]) if i != -1]

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8")
               if l.strip() and not l.startswith("//")]
    bai_nop = []
    dung_vi_tri = dung_dap_an = 0

    for q in queries:
        if q.get("type") != "qa":
            continue
        t0 = time.time()

        # --- Bước 1: tìm khoảnh khắc ---
        # Ghép CÂU HỎI vào mô tả: mô tả của BTC cố tình bỏ đi thứ đang được hỏi
        # nên kém phân biệt, còn câu hỏi lại lộ ra chi tiết nhìn thấy được
        # ("tên trên biển hiệu" -> cảnh có biển hiệu rõ). Đo thật trên 10 câu:
        # top-5 từ 1/10 lên 3/10.
        r = understand(f"{q['query_vi']}. {q['question_vi']}")
        cau = [r["search_text_en"]] + r.get("variants", [])[:3]
        hits = rrf([tim(c) for c in cau])[:MAX_NOP]
        if not hits:
            print(f"{q['qid']}: không tìm được gì")
            continue
        # --- Bước 2: gom thành các KHOẢNH KHẮC riêng biệt ---
        # Hai hit cùng video cách nhau dưới GAN_NHAU frame là một khoảnh khắc;
        # đưa cho model xem 5 khung gần trùng nhau thì phí, phải đa dạng.
        khoanh_khac = []
        for kf, _ in hits:
            v, so = kf.split("#")
            so = int(so)
            # so = `n` khi index từ CLIPFeatures BTC, = frame_idx khi index tự
            # dựng từ video. Quy về frame_idx để so khoảng cách cho đúng.
            if mk_anh is not None:
                n = so
                try:
                    f = mk_anh.frame_idx(v, n)
                except (KeyError, FileNotFoundError):
                    continue
            else:
                n, f = None, so
            if any(k["video_id"] == v and abs(k["frame"] - f) < args.gan_nhau
                   for k in khoanh_khac):
                continue
            khoanh_khac.append({"video_id": v, "n": n, "frame": f})
            if len(khoanh_khac) >= args.n_ung_vien:
                break

        # --- Bước 3: lấy ảnh cho từng ứng viên ---
        ung_vien = []
        for k in khoanh_khac:
            if mk_anh is not None:
                # ĐỌC THẲNG ẢNH KEYFRAME của BTC — tức thì. Trước đây phải giải
                # mã video 55-70 giây mỗi ứng viên nên chỉ dám đưa model xem 6
                # ứng viên, trong khi đáp án nằm ở hạng 8-46. Đó chính là nút
                # thắt của Q&A. Đọc file JPEG gỡ được hoàn toàn.
                fids, imgs = [], []
                for dn in range(-(args.n_frame // 2), args.n_frame // 2 + 1):
                    p = mk_anh.duong_dan_anh(k["video_id"], k["n"] + dn)
                    if p:
                        from PIL import Image
                        fids.append(mk_anh.frame_idx(k["video_id"], k["n"] + dn))
                        imgs.append(Image.open(p).convert("RGB"))
            else:
                vp = Path(args.videos) / f"{k['video_id']}.mp4"
                if not vp.exists():
                    continue
                buoc = max(1, (2 * args.ban_kinh) // max(1, args.n_frame - 1))
                fids, imgs = extract_window_nhanh(vp, k["frame"], args.ban_kinh,
                                                  buoc)
            if imgs:
                ung_vien.append({"video_id": k["video_id"], "frame": k["frame"],
                                 "frame_ids": fids, "images": imgs})
        if not ung_vien:
            print(f"{q['qid']}: không trích được frame nào")
            continue

        # --- Bước 4a: Claude chọn ứng viên (ảnh vừa phải, nhiều ứng viên) ---
        khoa = f"{q['qid']}|{args.n_ung_vien}|{args.khong_doc_ky}"
        if khoa in qa_cache:
            kq = qa_cache[khoa]
        else:
            kq = chon_va_tra_loi(ung_vien, q["question_vi"], q["query_vi"])
            # --- Bước 4b: đọc kỹ ứng viên đã chọn, có phóng to từng vùng ---
            # Chỉ còn MỘT ứng viên nên gửi được ảnh đầy đủ + các ô phóng to.
            # Đây là nhóm công cụ "không gian" trong khung STAR (tập huấn buổi 3).
            if not args.khong_doc_ky:
                c = ung_vien[kq["ung_vien"]]
                ky = doc_ky(c["images"], q["question_vi"], q["query_vi"],
                            c["frame_ids"], args.n_luoi)
                if ky["chac_chan"] >= kq.get("chac_chan", 0.5):
                    kq["answer"] = ky["answer"]
                kq["_ky"] = ky
            qa_cache[khoa] = kq
        chon = ung_vien[kq["ung_vien"]]
        dap_an = kq["answer"]

        # --- Bước 5: ĐẨY ứng viên model chọn lên đầu danh sách nộp ---
        # Model vừa đóng vai trò xếp hạng lại: nó nhìn được ảnh, CLIP thì không.
        uu_tien = [kf for kf, _ in hits if kf.startswith(chon["video_id"] + "#")]
        con_lai = [kf for kf, _ in hits if not kf.startswith(chon["video_id"] + "#")]
        thu_tu = uu_tien + con_lai
        answers = []
        for kf in thu_tu:
            v, so = kf.split("#")
            so = int(so)
            if mk_anh is not None:
                try:
                    so = mk_anh.frame_idx(v, so)  # n -> frame_idx để nộp
                except (KeyError, FileNotFoundError):
                    continue
            answers.append({"video_id": v, "frame_id": so, "answer": dap_an})
        bai_nop.append({"qid": q["qid"], "answers": answers})

        gt = q["gt"]
        ok_vt = next((i for i, a in enumerate(answers, 1)
                      if a["video_id"] == gt["video_id"]
                      and gt["s"] <= a["frame_id"] <= gt["e"]), None)
        chap_nhan = gt["answer"] if isinstance(gt["answer"], list) else [gt["answer"]]
        ok_da = any(a.lower() in dap_an.lower() or dap_an.lower() in a.lower()
                    for a in chap_nhan)
        dung_vi_tri += ok_vt is not None
        dung_dap_an += ok_da
        chon_dung = chon["video_id"] == gt["video_id"]
        print(f"{q['qid']}  chọn ứng viên {kq['ung_vien']+1}/{len(ung_vien)} "
              f"({chon['video_id']} {'ĐÚNG' if chon_dung else 'SAI'} video)  "
              f"vị trí hạng {ok_vt}  "
              f"đáp án {'ĐÚNG' if ok_da else 'SAI'}: \"{dap_an}\"  "
              f"{time.time()-t0:.0f}s")
        if not ok_da:
            print(f"      mong đợi: {chap_nhan}")
            print(f"      lý lẽ   : {kq['giai_thich'][:90]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in bai_nop:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(qa_cache, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    n = len(bai_nop)
    print(f"\nTìm đúng vị trí: {dung_vi_tri}/{n}   "
          f"Trả lời đúng: {dung_dap_an}/{n}")
    print(f"{n} bài nộp -> {out}")


if __name__ == "__main__":
    main()
