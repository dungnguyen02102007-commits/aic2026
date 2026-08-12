"""Test tích hợp backend — chạy trên CẢ hai store (mock và faiss thật).

Chạy:
    python -m indexing.make_fake_index     # dựng index giả (1 lần)
    python backend/test_api.py

Kiểm tra hai thứ dễ hỏng âm thầm:
  - /search có trả frame_idx không (thiếu là không nộp bài được — issue #20)
  - keyframe_id -> frame_idx -> ảnh có khớp nhau không (lệch 1 là sai hết)
"""
import os
import sys
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAKE = ROOT / "data" / "fake_index"
dat = truot = 0


def check(ten, ok, chi_tiet=""):
    global dat, truot
    print(f"{'✓' if ok else '✗'} {ten}" + (f"  — {chi_tiet}" if chi_tiet else ""))
    dat, truot = dat + bool(ok), truot + (not ok)


def chay_bo_test(client, ten_store, store):
    print(f"\n=== STORE: {ten_store} ===")

    r = client.get("/health")
    check("/health", r.status_code == 200 and r.json()["status"] == "ok")

    r = client.get("/stats")
    st = r.json()
    check("/stats", r.status_code == 200 and st["n_keyframes"] > 0,
          f"{st.get('n_keyframes')} keyframe, {st.get('n_videos')} video, "
          f"{st.get('index_version')}")

    # --- Tìm kiếm text ---
    q = "a news anchor reading headlines in a tv studio"
    r = client.post("/search", json={"query_text": q, "top_k": 10})
    check("/search 200", r.status_code == 200, r.text[:120])
    if r.status_code != 200:
        return
    res = r.json()["results"]
    check("/search có kết quả", len(res) > 0, f"{len(res)} segment")

    # issue #20: THIẾU frame_idx = không nộp bài được
    co_frame = all(isinstance(s.get("frame_idx"), int) for s in res)
    check("mọi kết quả có frame_idx (issue #20)", co_frame,
          f"vd frame_idx={res[0]['frame_idx']}" if res else "")

    # frame_idx phải khớp đúng keyframe được trả về, không phải số bịa
    top = res[0]
    meta = store.get_metadata(top["keyframe_id"])
    check("frame_idx khớp metadata của keyframe đó",
          meta is not None and meta["frame_idx"] == top["frame_idx"],
          f"{top['keyframe_id']} -> frame_idx {top['frame_idx']}")

    # ID phải đếm từ 1 (chuẩn cột n của BTC), không phải từ 0
    so = int(top["keyframe_id"].split("#")[1])
    check("keyframe_id đếm từ 1, không phải 0", so >= 1, top["keyframe_id"])

    check("kết quả xếp giảm dần theo điểm",
          all(res[i]["score"] >= res[i + 1]["score"] for i in range(len(res) - 1)))

    # --- Tìm kiếm có LỌC theo video ---
    vid = res[0]["video_id"]
    r = client.post("/search", json={"query_text": q, "top_k": 10,
                                     "filters": {"video_ids": [vid]}})
    loc = r.json()["results"]
    check("lọc theo video_ids", loc and all(s["video_id"] == vid for s in loc),
          f"{len(loc)} kết quả, tất cả thuộc {vid}")

    # --- Timeline ---
    r = client.get(f"/video/{vid}/timeline")
    tl = r.json()["entries"]
    check("/timeline có frame_idx",
          len(tl) > 0 and all("frame_idx" in e for e in tl), f"{len(tl)} mục")
    check("/timeline sắp theo thời gian tăng dần",
          all(tl[i]["timestamp"] <= tl[i + 1]["timestamp"] for i in range(len(tl) - 1)))

    # --- Ảnh keyframe ---
    # keyframe_id chứa '#' -> PHẢI mã hoá thành %23, nếu không trình duyệt cắt
    # nó thành fragment và server nhận thiếu đường dẫn. UI đã dùng
    # encodeURIComponent, client nào khác cũng phải làm vậy.
    kf_url = quote(top["keyframe_id"], safe="")
    r = client.get(f"/keyframe/{kf_url}/image")
    check("/keyframe ảnh mở được", r.status_code == 200 and len(r.content) > 500,
          f"{len(r.content)} byte" if r.status_code == 200 else r.text[:80])

    r = client.get("/keyframe/KHONG_TON_TAI%23000001/image")
    check("/keyframe id sai -> 404", r.status_code == 404)

    # --- Truy vấn rỗng phải báo lỗi rõ ràng ---
    r = client.post("/search", json={"top_k": 5})
    check("/search thiếu query -> 422", r.status_code == 422)

    return res


def main():
    from fastapi.testclient import TestClient

    # --- 1. Store mock ---
    os.environ["AIC_STORE"] = "mock"
    for m in [k for k in sys.modules if k.startswith("backend.")]:
        del sys.modules[m]
    from backend.main import app, store  # noqa: PLC0415

    chay_bo_test(TestClient(app), "mock", store)

    # --- 2. Store faiss thật ---
    if not (FAKE / "clip_flat.faiss").exists():
        print("\n(bỏ qua store faiss — chưa chạy `python -m indexing.make_fake_index`)")
    else:
        os.environ.update({
            "AIC_STORE": "faiss",
            "AIC_INDEX_PATH": str(FAKE / "clip_flat.faiss"),
            "AIC_IDS_PATH": str(FAKE / "ids.json"),
            "AIC_DB_PATH": str(FAKE / "metadata.db"),
            "AIC_KEYFRAMES_DIR": str(FAKE / "keyframes"),
        })
        for m in [k for k in sys.modules if k.startswith("backend.")]:
            del sys.modules[m]
        from backend.main import app as app2, store as store2  # noqa: PLC0415

        res = chay_bo_test(TestClient(app2), "faiss (index CLIP thật)", store2)

        # Kiểm tra CLIP có thực sự hiểu ngữ nghĩa không, hay chỉ trả bừa
        from backend.stores_mock import MockStore  # noqa: PLC0415
        caption = {k["keyframe_id"]: k["caption"] for k in MockStore().keyframes}
        dung = sum("news anchor" in caption.get(s["keyframe_id"], "")
                   for s in res[:3])
        check("CLIP hiểu ngữ nghĩa: top-3 của 'news anchor' đúng cảnh", dung >= 2,
              f"{dung}/3 — top-1: \"{caption.get(res[0]['keyframe_id'], '?')}\"")

    print(f"\nKẾT QUẢ: {dat}/{dat + truot}")
    sys.exit(1 if truot else 0)


if __name__ == "__main__":
    main()
