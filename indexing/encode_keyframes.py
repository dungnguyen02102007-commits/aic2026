"""Encode toàn bộ keyframe bằng model thị giác mạnh hơn — chạy tại máy có GPU.

    python -m indexing.encode_keyframes \
        --keyframes D:/AIC2026/keyframes \
        --map       D:/AIC2026/full/map-keyframes-.../map-keyframes \
        --out       D:/AIC2026/full/index_l14 --name l14

Vì sao cần: BTC cấp sẵn vector **ViT-B/32** — model yếu nhất họ CLIP. Đo thật
trên 17 truy vấn KIS (kho L21, 7.720 keyframe), cùng bản dịch đã cache:

    ViT-B/32 (BTC cấp)   Final 0.612   R@1 0.294   R@5 0.588
    ViT-L/14             Final 0.694   R@1 0.412   R@5 0.706   <- hơn 13%
    Fusion cả hai        Final 0.682   R@1 0.471   R@5 0.647

Đây là cách giải bế tắc đo được trước đó: khoảng cách điểm 0.06–0.11 giữa
keyframe đúng và top-1 quá lớn để tinh chỉnh câu chữ. Cần TÍN HIỆU KHÁC —
model thị giác mạnh hơn chính là tín hiệu đó.

LƯU TỪNG PHẦN: đứt giữa chừng thì chạy lại là tiếp tục chỗ dở, không mất công.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MAC_DINH = ("ViT-L-14", "laion2b_s32b_b82k")


class TapAnh:
    """Dataset đọc + tiền xử lý ảnh.

    PHẢI ở cấp module, không được lồng trong hàm: trên Windows, DataLoader
    nhiều luồng dùng cơ chế `spawn` và phải pickle lớp này sang tiến trình con —
    lớp lồng trong hàm là đối tượng cục bộ, không pickle được.
    """

    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.preprocess(Image.open(self.paths[i]).convert("RGB"))


def liet_ke(kf_dir):
    """Trả về [(keyframe_id, đường dẫn ảnh)] sắp xếp ổn định.

    keyframe_id = "<video>#<n:06d>" với n LẤY TỪ TÊN FILE (đếm từ 1) — khớp
    cột n của map-keyframes. Đây là quy ước chung của cả hệ thống.
    """
    ds = {}
    for p in Path(kf_dir).rglob("*.jpg"):
        try:
            n = int(p.stem)
        except ValueError:
            continue  # bỏ file tên lạ thay vì đoán bừa
        ds[f"{p.parent.name}#{n:06d}"] = p
    return sorted(ds.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--map", help="map-keyframes, để đối chiếu đủ/thiếu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="l14")
    ap.add_argument("--model", default=MAC_DINH[0])
    ap.add_argument("--pretrained", default=MAC_DINH[1])
    ap.add_argument("--batch", type=int, default=64,
                    help="giảm xuống nếu báo hết VRAM (3050 4GB: 64 là vừa)")
    ap.add_argument("--shard", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    import faiss
    import open_clip
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = liet_ke(args.keyframes)
    ids = [k for k, _ in ds]
    paths = [p for _, p in ds]
    print(f"{len(ids):,} ảnh   {len({k.split('#')[0] for k in ids})} video")

    if args.map:
        from indexing.map_keyframes import MapKeyframes
        mk = MapKeyframes(args.map)
        can = sum(len(mk.danh_sach(v)) for v in mk.cac_video())
        print(f"map-keyframes yêu cầu {can:,} keyframe "
              f"-> {'ĐỦ' if can == len(ids) else f'LỆCH {can - len(ids):+,}'}")
        if can != len(ids):
            print("  ⚠ Thiếu ảnh thì index sẽ không phủ hết kho. Cân nhắc tải nốt.")

    gpu = torch.cuda.is_available()
    print(f"\nThiết bị: {torch.cuda.get_device_name(0) if gpu else 'CPU'}")
    if not gpu:
        print("  ⚠ Không có GPU — sẽ mất khoảng 23 TIẾNG thay vì ~30 phút.")
        print("  Cài torch bản CUDA: pip install torch --index-url "
              "https://download.pytorch.org/whl/cu126")

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained)
    model = model.eval()
    if gpu:
        model = model.cuda().half()  # half: nhanh gấp đôi, sai số không đáng kể

    t0 = time.time()
    for s in range(0, len(paths), args.shard):
        f = out / f"phan_{s:07d}.npy"
        if f.exists():
            print(f"bỏ qua {f.name} (đã có)")
            continue
        lo = paths[s:s + args.shard]
        dl = DataLoader(TapAnh(lo, preprocess), batch_size=args.batch,
                        num_workers=args.workers, pin_memory=gpu)
        ra = []
        with torch.no_grad():
            for b in dl:
                if gpu:
                    b = b.cuda().half()
                ra.append(model.encode_image(b).float().cpu().numpy())
        np.save(f, np.concatenate(ra).astype("float32"))
        xong = min(s + args.shard, len(paths))
        troi = time.time() - t0
        print(f"{xong:,}/{len(paths):,}  {troi:.0f}s  "
              f"({xong/max(1,troi):.0f} ảnh/giây, "
              f"còn ~{(len(paths)-xong)/max(1,xong)*troi/60:.0f} phút)")

    # --- Gộp các phần, dựng index ---
    phan = sorted(out.glob("phan_*.npy"))
    X = np.concatenate([np.load(p) for p in phan]).astype("float32")
    if len(X) != len(ids):
        sys.exit(f"LỆCH: {len(X)} vector nhưng {len(ids)} id. "
                 f"Xoá file phan_*.npy cuối rồi chạy lại.")

    X = np.ascontiguousarray(X)
    faiss.normalize_L2(X)  # BẮT BUỘC: inner product = cosine
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    faiss.write_index(index, str(out / f"{args.name}.faiss"))
    (out / f"{args.name}_ids.json").write_text(json.dumps(ids), encoding="utf-8")

    print(f"\n✓ {index.ntotal:,} vector × {X.shape[1]} chiều  "
          f"({(time.time()-t0)/60:.1f} phút)")
    print(f"✓ {out / (args.name + '.faiss')}  "
          f"({(out / (args.name + '.faiss')).stat().st_size/1e6:.0f} MB)")
    print(f"  ID đầu: {ids[0]}   ID cuối: {ids[-1]}")
    if not ids[0].endswith("#000001"):
        print("  ⚠ ID đầu KHÔNG kết thúc bằng #000001 — kiểm lại tên file ảnh!")


if __name__ == "__main__":
    main()
