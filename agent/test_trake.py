"""Test TRAKE (issue #19).

Chạy:  python -m agent.test_trake

Phần A (align_events) chạy offline, không tốn tiền API.
Phần B (decompose) cần AIC_LLM_PROVIDER thật, tự bỏ qua nếu chưa cấu hình.
Phần C đo độ tương phản giữa các khoảnh khắc bằng CLIP — bỏ qua nếu chưa cài open_clip.
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agent.trake import (NEG_INF, align_events, decompose,  # noqa: E402
                         pick_frames, rank_videos, rank_videos_by_events)

dat = truot = 0


def check(ten, ok, chi_tiet=""):
    global dat, truot
    print(f"{'✓' if ok else '✗'} {ten}" + (f"  — {chi_tiet}" if chi_tiet else ""))
    dat, truot = dat + bool(ok), truot + (not ok)


# ===========================================================================
# A. align_events — quy hoạch động chọn frame tăng dần
# ===========================================================================
print("=== A. Căn chỉnh thời gian (offline) ===")

# A1. Trường hợp thuận: argmax độc lập đã đúng thứ tự sẵn
S = [[0.9, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.9, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.9]]
path, total = align_events(S)
check("3 sự kiện, đỉnh điểm đã đúng thứ tự", path == [0, 2, 4] and abs(total - 2.7) < 1e-9,
      f"path={path} total={total:.2f}")

# A2. BẪY CHÍNH: argmax độc lập cho ra thứ tự NGƯỢC
#     sự kiện 0 khớp nhất frame 2, sự kiện 1 khớp nhất frame 0 -> 2 > 0, vô lý.
#     DP phải hy sinh điểm để giữ thứ tự đúng.
S = [[0.1, 0.2, 0.9],
     [0.8, 0.3, 0.1]]
path, total = align_events(S)
argmax_doc_lap = [max(range(3), key=lambda i: row[i]) for row in S]
check("argmax độc lập vi phạm thứ tự (bằng chứng cần DP)",
      argmax_doc_lap == [2, 0], f"argmax={argmax_doc_lap} -> ngược thời gian")
check("DP giữ đúng thứ tự tăng dần", path == [0, 1] and abs(total - 0.4) < 1e-9,
      f"path={path} total={total:.2f} (các lựa chọn: (0,1)=0.4 (1,2)=0.3 (0,2)=0.2)")

# A3. Không được dùng lại cùng một frame cho hai sự kiện
S = [[0.9, 0.1], [0.9, 0.1]]
path, _ = align_events(S)
check("hai sự kiện không dùng chung một frame", path == [0, 1], f"path={path}")

# A4. min_gap — bắt hai khoảnh khắc cách nhau tối thiểu
S = [[0.9, 0.1, 0.1], [0.1, 0.9, 0.2]]
p1, _ = align_events(S, min_gap=1)
p2, _ = align_events(S, min_gap=2)
check("min_gap=1 chọn frame liền kề", p1 == [0, 1], f"path={p1}")
check("min_gap=2 buộc giãn ra", p2 == [0, 2], f"path={p2}")

# A5. Không đủ frame -> báo không xếp được, KHÔNG được trả bừa
path, total = align_events([[0.5, 0.5]] * 3)
check("3 sự kiện / 2 frame -> không xếp được", path is None and total == NEG_INF)
path, _ = align_events([[0.5] * 4] * 3, min_gap=2)
check("min_gap=2, 3 sự kiện cần >=5 frame", path is None)

# A6. Biên
path, total = align_events([[0.1, 0.9, 0.2]])
check("1 sự kiện -> lấy đỉnh", path == [1] and abs(total - 0.9) < 1e-9)
check("0 sự kiện -> None", align_events([])[0] is None)

# A7. pick_frames đổi vị trí -> frame_id THẬT (con số nộp cho BTC)
S = [[0.9, 0.1, 0.1, 0.1, 0.1],
     [0.1, 0.1, 0.9, 0.1, 0.1],
     [0.1, 0.1, 0.1, 0.1, 0.9]]
frame_ids = [100, 250, 400, 550, 700]
fids, _ = pick_frames(S, frame_ids)
check("pick_frames trả frame_id thật, không phải vị trí", fids == [100, 400, 700],
      f"{fids}")

# ===========================================================================
# A2. rank_videos — giai đoạn 1: chọn VIDEO ứng viên
# ===========================================================================
print("\n=== A2. Xếp hạng video ứng viên (offline) ===")

META = {
    "V1#000001": {"video_id": "V1", "frame_idx": 100},
    "V1#000002": {"video_id": "V1", "frame_idx": 200},
    "V1#000003": {"video_id": "V1", "frame_idx": 300},
    "V2#000001": {"video_id": "V2", "frame_idx": 500},
    "V3#000001": {"video_id": "V3", "frame_idx": 900},
}

# V1 khớp vừa phải ở 3 chỗ; V2 khớp mạnh hơn nhưng chỉ 1 chỗ
hits = [("V2#000001", 0.40), ("V1#000002", 0.38), ("V1#000001", 0.36),
        ("V1#000003", 0.35), ("V3#000001", 0.20)]
ds = rank_videos(hits, META.get)
check("gộp keyframe theo video", [v["video_id"] for v in ds] == ["V2", "V1", "V3"]
      or [v["video_id"] for v in ds][:2] == ["V1", "V2"],
      f"{[(v['video_id'], round(v['score_cuoi'], 3)) for v in ds]}")
check("đếm đúng số keyframe khớp mỗi video",
      {v["video_id"]: v["n_hits"] for v in ds} == {"V1": 3, "V2": 1, "V3": 1})
v1 = next(v for v in ds if v["video_id"] == "V1")
check("center_frame lấy của keyframe điểm cao nhất trong video",
      v1["center_frame"] == 200 and v1["best_keyframe_id"] == "V1#000002",
      f"center={v1['center_frame']} kf={v1['best_keyframe_id']}")

# Thưởng nhiều-hit phải ĐỦ NHỎ để không lật ngược một video khớp mạnh rõ rệt
hits_manh = [("V2#000001", 0.90), ("V1#000001", 0.30), ("V1#000002", 0.30),
             ("V1#000003", 0.30)]
ds2 = rank_videos(hits_manh, META.get)
check("thưởng nhiều-hit không lấn át video khớp mạnh",
      ds2[0]["video_id"] == "V2",
      f"{[(v['video_id'], round(v['score_cuoi'], 3)) for v in ds2]}")

check("top_videos cắt đúng số lượng", len(rank_videos(hits, META.get, 2)) == 2)
check("keyframe không có metadata thì bỏ qua, không sập",
      len(rank_videos([("KHONG_CO#000001", 0.9)] + hits, META.get)) == 3)


# ---------------------------------------------------------------------------
# rank_videos_by_events — bản dùng khi thi
# ---------------------------------------------------------------------------
print("\n=== A3. Xếp hạng video theo TỪNG khoảnh khắc ===")

M2 = {f"{v}#{f}": {"video_id": v, "frame_idx": f}
      for v in ("V1", "V2", "V3") for f in (10, 20, 30)}

# V1 khớp đều cả 3 khoảnh khắc; V2 chỉ khớp mạnh 1 cái; V3 không khớp gì
hits_ev = [
    [("V1#10", 0.50), ("V2#10", 0.48), ("V3#10", 0.40)],
    [("V1#20", 0.30), ("V2#20", 0.10), ("V3#20", 0.12)],
    [("V1#30", 0.30), ("V2#30", 0.10), ("V3#30", 0.11)],
]
ds = rank_videos_by_events(hits_ev, M2.get)
check("video khớp đủ 3 khoảnh khắc lên đầu", ds[0]["video_id"] == "V1",
      f"{[(v['video_id'], round(v['score_cuoi'], 2)) for v in ds]}")
check("center_frame lấy của khoảnh khắc khớp mạnh nhất",
      ds[0]["center_frame"] in (10, 20, 30), f"{ds[0]['center_frame']}")
check("giữ điểm từng khoảnh khắc để gỡ lỗi",
      len(ds[0]["diem_tung_su_kien"]) == 3 and len(ds[0]["z_tung_su_kien"]) == 3)

# Chuẩn hoá z: một khoảnh khắc có THANG ĐIỂM lớn bất thường không được
# quyết định tất cả. Cộng thô thì V2 thắng, chuẩn hoá z thì V1 thắng.
hits_lech = [
    [("V1#10", 0.50), ("V2#10", 2.00), ("V3#10", 0.40)],  # thang điểm to
    [("V1#20", 0.30), ("V2#20", 0.05), ("V3#20", 0.10)],
    [("V1#30", 0.30), ("V2#30", 0.05), ("V3#30", 0.10)],
]
tong_tho = {v: sum(h[i][1] for h in hits_lech)
            for i, v in enumerate(("V1", "V2", "V3"))}
check("cộng thô sẽ chọn NHẦM (bằng chứng cần chuẩn hoá z)",
      max(tong_tho, key=tong_tho.get) == "V2",
      f"tổng thô: {  {k: round(x, 2) for k, x in tong_tho.items()} }")
ds = rank_videos_by_events(hits_lech, M2.get)
check("chuẩn hoá z chọn ĐÚNG video khớp đều", ds[0]["video_id"] == "V1",
      f"{[(v['video_id'], round(v['score_cuoi'], 2)) for v in ds]}")

# Video vắng mặt ở một khoảnh khắc -> lấy điểm sàn, không được sập
thieu = [
    [("V1#10", 0.50), ("V2#10", 0.48)],
    [("V1#20", 0.30)],                    # V2 không xuất hiện
    [("V1#30", 0.30), ("V2#30", 0.10)],
]
ds = rank_videos_by_events(thieu, M2.get)
check("video vắng ở một khoảnh khắc vẫn xếp được, không sập",
      len(ds) == 2 and ds[0]["video_id"] == "V1",
      f"{[v['video_id'] for v in ds]}")
check("danh sách hits rỗng -> trả rỗng", rank_videos_by_events([], M2.get) == [])


# ===========================================================================
# B + C. decompose: tách truy vấn, và đo độ tương phản giữa các khoảnh khắc
# ===========================================================================

_clip = {}


def do_tuong_phan(texts):
    """Ma trận tương đồng CLIP giữa các mô tả khoảnh khắc.

    Đây là thước đo QUAN TRỌNG NHẤT của chất lượng prompt TRAKE: nếu CLIP thấy
    hai khoảnh khắc gần như y hệt nhau thì bước căn chỉnh thành ra đoán mò, dù
    câu chữ tiếng Anh đọc rất hay.
    """
    import numpy as np
    import open_clip
    import torch

    from backend.stores_faiss import MODEL_NAME, PRETRAINED

    if not _clip:
        model, _, _ = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED)
        model.eval()
        _clip["model"] = model
        _clip["tok"] = open_clip.get_tokenizer(MODEL_NAME)
    with torch.no_grad():
        V = _clip["model"].encode_text(_clip["tok"](texts)).float().numpy()
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    sim = V @ V.T
    np.fill_diagonal(sim, 0.0)
    return sim


def kiem_tra(ten, query, n_mong_doi):
    r = decompose(query)
    check(f"[{ten}] có video_search_en (quyết định 0 điểm hay không)",
          bool(r.get("video_search_en")))
    check(f"[{ten}] tách đúng {n_mong_doi} khoảnh khắc",
          len(r["events"]) == n_mong_doi, f"{len(r['events'])} sự kiện")
    check(f"[{ten}] mỗi khoảnh khắc có search_en",
          all(ev.get("search_en") for ev in r["events"]))

    if provider == "mock":
        return

    print(f"\n  video: {r['video_search_en']}")
    for i, ev in enumerate(r["events"], 1):
        print(f"  {i}. [{ev['name']}] {ev['search_en']}")

    texts = [ev["search_en"] for ev in r["events"]]
    check(f"[{ten}] các mô tả khác nhau", len(set(texts)) == len(texts))
    check(f"[{ten}] mô tả bằng tiếng Anh", all(t.isascii() for t in texts))
    check(f"[{ten}] caption ngắn (<= 25 từ, CLIP không đọc văn xuôi dài)",
          all(len(t.split()) <= 25 for t in texts),
          f"dài nhất {max(len(t.split()) for t in texts)} từ")

    try:
        sim = do_tuong_phan(texts)
    except ImportError as e:
        print(f"  (bỏ qua đo tương phản — thiếu thư viện: {e})")
        return
    print("  ma trận tương đồng CLIP:")
    for i, row in enumerate(sim):
        print(f"    {i + 1}: " + " ".join(f"{v:5.2f}" for v in row))
    cao_nhat = float(sim.max())
    i, j = divmod(int(sim.argmax()), sim.shape[1])
    # >0.95 nghĩa là CLIP nhìn hai khoảnh khắc gần như y hệt -> căn chỉnh vô nghĩa
    check(f"[{ten}] không có hai khoảnh khắc CLIP thấy giống hệt (< 0.95)",
          cao_nhat < 0.95, f"cao nhất {cao_nhat:.3f} giữa #{i + 1} và #{j + 1}")


provider = os.environ.get("AIC_LLM_PROVIDER", "mock")
print(f"\n=== B. Tách truy vấn (provider={provider}) ===")

# Câu 1: CHÍNH LÀ ví dụ mẫu trong prompt -> chỉ kiểm tra định dạng, không phải
# bằng chứng model hiểu bài.
kiem_tra("nhảy cao (có trong prompt mẫu)",
         "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy cao: "
         "chạy đà, giậm nhảy, bay qua xà, tiếp đất.", 4)

# Câu 2: HOÀN TOÀN MỚI, đúng lĩnh vực dữ liệu thật (bản tin tiếng Việt).
# Đây mới là phép thử tổng quát hóa thật sự.
kiem_tra("khánh thành cầu (câu mới, đúng lĩnh vực tin tức)",
         "Tìm đoạn bản tin về lễ khánh thành cầu gồm 4 khoảnh khắc: cắt băng "
         "khánh thành, lãnh đạo phát biểu trên bục, đoàn xe đầu tiên chạy qua "
         "cầu, người dân vẫy tay ăn mừng.", 4)

if provider == "mock":
    print("  (provider=mock — bỏ qua kiểm tra chất lượng, cần .env để chạy thật)")

print(f"\nKẾT QUẢ: {dat}/{dat + truot}")
sys.exit(1 if truot else 0)
