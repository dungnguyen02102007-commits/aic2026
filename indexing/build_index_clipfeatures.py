"""Dựng index FAISS từ CLIPFeatures của BTC — bản dùng khi thi.

    python -m indexing.build_index_clipfeatures \
        --features D:/AIC2026/full/clip_features/clip_features \
        --map      D:/AIC2026/full/map-keyframes-.../map-keyframes \
        --out      D:/AIC2026/full/index --name clip_full

Sinh ra <out>/<name>.faiss + <name>_ids.json + <name>_bao_cao.json

Khác build_index_video.py (tự trích frame từ video): script này dùng THẲNG
vector BTC đã tính sẵn, nên nhanh và khớp đúng keyframe BTC phát.

BA CHỖ SAI LÀ HỎNG ÂM THẦM, script đều chặn:

  1. ID lấy từ `range(len(vecs))` thay vì cột `n` của map-keyframes.
     Cột `n` đếm TỪ 1 -> dùng range() là lệch một đơn vị trên MỌI kết quả,
     mà số dòng vẫn khớp nên không phép kiểm nào phát hiện được.
  2. Số dòng .npy khác số dòng CSV -> hai file không cùng đợt phát. Script
     bỏ qua video đó và báo tên, thay vì ghép bừa.
  3. Video có map nhưng thiếu features -> không bao giờ tìm thấy được.
     Script liệt kê rõ để biết mà xử lý.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from indexing.map_keyframes import MapKeyframes  # noqa: E402


def main():
    import faiss

    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="thư mục chứa <video>.npy")
    ap.add_argument("--map", required=True, help="thư mục chứa <video>.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="clip_full")
    args = ap.parse_args()

    fdir, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mk = MapKeyframes(args.map)

    co_feat = {p.stem for p in fdir.glob("*.npy")}
    co_map = set(mk.cac_video())
    dung_chung = sorted(co_feat & co_map)
    thieu_feat = sorted(co_map - co_feat)
    thieu_map = sorted(co_feat - co_map)

    print(f"features: {len(co_feat)} video   map: {len(co_map)} video   "
          f"dùng được: {len(dung_chung)}")
    if thieu_feat:
        print(f"\n⚠ {len(thieu_feat)} video CÓ map nhưng THIẾU features "
              f"— sẽ KHÔNG BAO GIỜ tìm thấy được:")
        print("  " + ", ".join(thieu_feat[:20])
              + (" ..." if len(thieu_feat) > 20 else ""))
    if thieu_map:
        print(f"\n⚠ {len(thieu_map)} video có features nhưng THIẾU map "
              f"— không đổi được sang frame_idx để nộp:")
        print("  " + ", ".join(thieu_map[:20]))

    vecs, ids, lech, t0 = [], [], [], time.time()
    for k, v in enumerate(dung_chung, 1):
        X = np.load(fdir / f"{v}.npy").astype("float32")
        if X.ndim == 1:
            X = X.reshape(1, -1)
        ns = [n for n, _, _ in mk.danh_sach(v)]
        if len(ns) != len(X):
            lech.append({"video": v, "n_vector": len(X), "n_dong_csv": len(ns)})
            continue
        vecs.append(X)
        # ID theo ĐÚNG cột n của BTC (đếm từ 1), không phải range()
        ids += [f"{v}#{n:06d}" for n in ns]
        if k % 200 == 0:
            print(f"  ...{k}/{len(dung_chung)} video ({time.time()-t0:.0f}s)")

    if lech:
        print(f"\n⚠ {len(lech)} video LỆCH số dòng giữa .npy và .csv — đã bỏ qua:")
        for d in lech[:10]:
            print(f"    {d['video']}: {d['n_vector']} vector vs "
                  f"{d['n_dong_csv']} dòng csv")

    X = np.ascontiguousarray(np.concatenate(vecs).astype("float32"))
    faiss.normalize_L2(X)  # BẮT BUỘC: inner product = cosine
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)

    faiss.write_index(index, str(out / f"{args.name}.faiss"))
    (out / f"{args.name}_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    bao_cao = {"n_vector": int(index.ntotal), "n_chieu": int(X.shape[1]),
               "n_video": len({i.split("#")[0] for i in ids}),
               "thieu_features": thieu_feat, "thieu_map": thieu_map,
               "lech_so_dong": lech}
    (out / f"{args.name}_bao_cao.json").write_text(
        json.dumps(bao_cao, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✓ {index.ntotal:,} vector × {X.shape[1]} chiều, "
          f"{bao_cao['n_video']} video   ({time.time()-t0:.0f}s)")
    print(f"✓ {out / (args.name + '.faiss')}  "
          f"({(out / (args.name + '.faiss')).stat().st_size/1e6:.0f} MB)")
    print(f"  ID đầu: {ids[0]}   ID cuối: {ids[-1]}")
    if not ids[0].endswith("#000001"):
        print("  ⚠ ID đầu KHÔNG kết thúc bằng #000001 — kiểm lại cột n!")


if __name__ == "__main__":
    main()
