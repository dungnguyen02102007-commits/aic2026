"""Chạy Textual KIS xuyên suốt và sinh file bài nộp (dạng 1, docs/onboarding/10).

    python eval/run_kis.py --queries eval/queries_kis_L21.jsonl \
        --index D:/AIC2026/index_L21_day --out eval/results/bai_nop_kis.jsonl

Rồi chấm:
    python eval/run_benchmark.py --submissions eval/results/bai_nop_kis.jsonl \
        --queries eval/queries_kis_L21.jsonl --tag kis_v1

Đường ống: truy vấn tiếng Việt -> understand() dịch sang mô tả thị giác tiếng
Anh (+ biến thể) -> tìm trên index CLIP -> RRF gộp các biến thể -> nộp 100 câu.

LUÔN nộp đủ 100 (docs/onboarding/11): câu đúng ở hạng 100 vẫn được 0.2 điểm,
và không có hình phạt nào cho câu sai.
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

MAX_NOP = 100
RRF_K = 60  # hằng số chuẩn của Reciprocal Rank Fusion, không cần tune


def rrf(danh_sach_hits):
    """Gộp nhiều lượt tìm bằng THỨ HẠNG, miễn nhiễm việc thang điểm khác nhau."""
    diem = {}
    for hits in danh_sach_hits:
        for hang, (kf_id, _) in enumerate(hits, start=1):
            diem[kf_id] = diem.get(kf_id, 0.0) + 1.0 / (RRF_K + hang)
    return sorted(diem.items(), key=lambda x: -x[1])


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--index", required=True, nargs="+",
                    help="một hoặc NHIỀU thư mục index. Đưa nhiều thì tự FUSION "
                         "bằng RRF — đo thật trên 873 video: B/32 0.553, "
                         "L/14 0.529, nhưng GỘP CẢ HAI 0.635. Không model nào "
                         "một mình thắng, gộp lại mới ăn.")
    ap.add_argument("--out", default="eval/results/bai_nop_kis.jsonl")
    ap.add_argument("--max-per-video", type=int, default=0,
                    help="0 = không giới hạn. Đặt >0 để ép đa dạng video "
                         "(đánh đổi: bớt cơ hội trúng khi đã đúng video)")
    ap.add_argument("--uu-tien-video", type=int, default=0,
                    help="0 = tắt. ĐÃ ĐO LÀ TỆ HƠN — giữ lại để khỏi thử lại. "
                         "Xếp lại theo video (gom điểm keyframe theo video rồi "
                         "nộp theo thứ tự video) cho kết quả kém hẳn: "
                         "gốc 0.553 | 20 video x 5kf 0.400 | 10x10 0.318 | "
                         "5x20 0.247. R@1 tụt từ 0.294 xuống 0.059 ở mọi biến "
                         "thể. Lý do: tổng điểm RRF theo video thiên vị video "
                         "DÀI có nhiều keyframe khớp lờ mờ, hơn video ngắn có "
                         "một chỗ khớp mạnh.")
    ap.add_argument("--no-variants", action="store_true",
                    help="chỉ dùng search_text_en, bỏ biến thể — để đo xem "
                         "biến thể có giúp gì không")
    ap.add_argument("--cache", default="eval/results/understand_cache.json",
                    help="lưu/dùng lại bản dịch của understand(). BẮT BUỘC khi "
                         "so sánh hai cấu hình: understand() gọi LLM nên mỗi "
                         "lần dịch một khác, không khoá lại thì chênh lệch đo "
                         "được là nhiễu dịch thuật chứ không phải hiệu quả thật")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="dịch lại từ đầu, ghi đè cache (dùng khi vừa sửa prompt)")
    ap.add_argument("--map", help="thư mục map-keyframes. BẮT BUỘC khi index "
                                  "dùng ID theo `n` (index từ CLIPFeatures của "
                                  "BTC) — phải đổi n sang frame_idx mới nộp được")
    ap.add_argument("--model", help="model encode CÂU TRUY VẤN. PHẢI TRÙNG với "
                                    "model đã dựng index, sai là hai không gian "
                                    "vector khác nhau, điểm tụt mà không báo lỗi. "
                                    "Mặc định ViT-B-32-quickgelu (khớp CLIPFeatures BTC)")
    ap.add_argument("--pretrained")
    args = ap.parse_args()

    # ID trong index có hai kiểu:
    #   "<video>#<frame_idx>"  — index tự dựng từ video (build_index_video.py)
    #   "<video>#<n>"          — index từ CLIPFeatures BTC (build_index_clipfeatures)
    # Kiểu thứ hai PHẢI đổi sang frame_idx, nếu không nộp sai toàn bộ mà
    # không có lỗi nào báo ra.
    mk = None
    if args.map:
        from indexing.map_keyframes import MapKeyframes
        mk = MapKeyframes(args.map)
        print(f"Đổi n -> frame_idx bằng map-keyframes ({len(mk.cac_video())} video)")

    cache_path = Path(args.cache)
    cache = {}
    if cache_path.exists() and not args.refresh_cache:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"Dùng lại {len(cache)} bản dịch từ {cache_path}")

    kho = []  # [(index, ids, encoder)]
    for d in args.index:
        idir = Path(d)
        ten = next(idir.glob("*.faiss")).stem
        ix = faiss.read_index(str(idir / f"{ten}.faiss"))
        iids = json.load(open(idir / f"{ten}_ids.json"))
        kho.append([ten, ix, iids, None])
        print(f"Index {ten}: {ix.ntotal:,} vector × {ix.d} chiều")
    index, ids = kho[0][1], kho[0][2]

    # Tự nhận model theo SỐ CHIỀU của index. Ghép nhầm model với index là hai
    # không gian vector khác nhau — điểm tụt mà không có lỗi nào báo ra, nên
    # thà suy ra tự động còn hơn để người dùng gõ tay.
    THEO_CHIEU = {512: ("ViT-B-32-quickgelu", "openai"),
                  768: ("ViT-L-14", "laion2b_s32b_b82k")}
    for k in kho:
        if args.model:
            ten_m, pre = args.model, args.pretrained or "openai"
        elif k[1].d in THEO_CHIEU:
            ten_m, pre = THEO_CHIEU[k[1].d]
        else:
            sys.exit(f"Index {k[0]} có {k[1].d} chiều — không đoán được model. "
                     f"Chỉ rõ bằng --model / --pretrained.")
        k[3] = ClipEncoder(ten_m, pre)
        print(f"  {k[0]} ({k[1].d} chiều) -> {ten_m}")
        if k[3].encode_text(["x"]).shape[1] != k[1].d:
            sys.exit(f"LỆCH SỐ CHIỀU ở {k[0]} — dừng thay vì chạy ra kết quả vô nghĩa.")
    if len(kho) > 1:
        print(f"FUSION {len(kho)} index bằng RRF")

    def tim(text, k=400):
        """Trả về danh sách hits của TỪNG index (RRF gộp ở ngoài)."""
        ra = []
        for _, ix, iids, e in kho:
            tv = e.encode_text([text]).astype("float32")
            D, I = ix.search(tv, min(k, ix.ntotal))
            ra.append([(iids[i], float(d)) for d, i in zip(D[0], I[0]) if i != -1])
        return ra

    queries = [json.loads(l) for l in open(args.queries, encoding="utf-8-sig")
               if l.strip() and not l.startswith("//")]
    bai_nop = []

    for q in queries:
        if q.get("type") != "kis":
            continue
        t0 = time.time()
        if q["qid"] in cache:
            r = cache[q["qid"]]
        else:
            r = understand(q["query_vi"])
            cache[q["qid"]] = r
        cau = [r["search_text_en"]]
        if not args.no_variants:
            cau += r.get("variants", [])[:3]
        hits = rrf([ds for c in cau for ds in tim(c)])

        if args.uu_tien_video:
            # Gom điểm theo video: điểm video = tổng điểm RRF của các keyframe
            # thuộc nó. Video có NHIỀU keyframe cùng khớp thì mạnh hơn video
            # chỉ khớp tình cờ một chỗ.
            diem_video = {}
            for kf, d in hits:
                v = kf.split("#")[0]
                diem_video[v] = diem_video.get(v, 0.0) + d
            thu_tu_video = {v: i for i, (v, _) in enumerate(
                sorted(diem_video.items(), key=lambda x: -x[1]))}
            # Trong N video đầu: xếp theo thứ hạng video rồi tới điểm keyframe.
            # Ngoài N video đó: giữ nguyên thứ tự RRF, xếp sau.
            hits = sorted(
                hits,
                key=lambda x: (0 if thu_tu_video[x[0].split("#")[0]] < args.uu_tien_video
                               else 1,
                               thu_tu_video[x[0].split("#")[0]], -x[1]))

        answers, dem_video = [], {}
        for kf_id, _ in hits:
            v, f = kf_id.split("#")
            if args.max_per_video and dem_video.get(v, 0) >= args.max_per_video:
                continue
            dem_video[v] = dem_video.get(v, 0) + 1
            fid = int(f)
            if mk is not None:
                try:
                    fid = mk.frame_idx(v, fid)  # số sau # là `n`, đổi sang frame_idx
                except (KeyError, FileNotFoundError):
                    continue  # không tra được thì bỏ, đừng nộp số bịa
            answers.append({"video_id": v, "frame_id": fid})
            if len(answers) >= MAX_NOP:
                break

        bai_nop.append({"qid": q["qid"], "answers": answers})
        dung = q["gt"]["video_id"]
        hang_v = next((i for i, a in enumerate(answers, 1)
                       if a["video_id"] == dung), None)
        hang_dung = next((i for i, a in enumerate(answers, 1)
                          if a["video_id"] == dung
                          and q["gt"]["s"] <= a["frame_id"] <= q["gt"]["e"]), None)
        print(f"{q['qid']}  \"{r['search_text_en'][:52]}\"")
        print(f"    đúng video hạng {hang_v}   ĐÚNG CẢ FRAME hạng {hang_dung}"
              f"   ({len(answers)} câu, {time.time()-t0:.1f}s)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for b in bai_nop:
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n{len(bai_nop)} bài nộp -> {out}")
    print(f"{len(cache)} bản dịch -> {cache_path}")


if __name__ == "__main__":
    main()
