"""Đọc map-keyframes của BTC — cầu nối giữa `n` và `frame_idx`.

Hai con số này RẤT DỄ NHẦM và nhầm là mất trắng, vì không có lỗi nào báo ra:

    n          số thứ tự keyframe trong video, ĐẾM TỪ 1, khớp TÊN FILE ẢNH
               -> keyframes/L21_V001/007.jpg
    frame_idx  chỉ số khung hình trong VIDEO GỐC, đếm từ 0
               -> CON SỐ NỘP CHO BTC

Keyframe thứ 7 của một video có n=7 nhưng frame_idx có thể là 1508.

Dùng:
    from indexing.map_keyframes import MapKeyframes
    mk = MapKeyframes("D:/AIC2026/full/map-keyframes")
    mk.frame_idx("L21_V001", 7)        # 1508 — số nộp cho BTC
    mk.gan_nhat("L21_V001", 1500)      # (7, 1508) — keyframe sát frame 1500 nhất
    mk.duong_dan_anh("L21_V001", 7)    # .../keyframes/L21_V001/007.jpg
"""
import bisect
import csv
from pathlib import Path

# BTC các mùa đặt tên ảnh không đồng nhất — thử lần lượt
MAU_TEN_ANH = ("{n:03d}.jpg", "{n:04d}.jpg", "{n:06d}.jpg", "{n}.jpg",
               "{n:03d}.jpeg", "{n:03d}.png")


class MapKeyframes:
    def __init__(self, thu_muc_map, thu_muc_keyframes=None):
        self.dir = Path(thu_muc_map)
        self.kf_dir = Path(thu_muc_keyframes) if thu_muc_keyframes else None
        self._cache = {}
        self._mau = None  # nhớ mẫu tên ảnh đã dò ra, khỏi stat lại mỗi lần

    def _nap(self, video_id):
        if video_id in self._cache:
            return self._cache[video_id]
        p = self.dir / f"{video_id}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Không có map-keyframes cho {video_id}: {p}")
        ns, frames, times = [], [], []
        with open(p, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                k = {c.lower().strip(): v for c, v in row.items() if c}
                ns.append(int(k["n"]))
                frames.append(int(k["frame_idx"]))
                times.append(float(k.get("pts_time", 0) or 0))
        # sắp theo frame_idx để tra nhị phân được
        thu_tu = sorted(range(len(ns)), key=lambda i: frames[i])
        d = {"n": [ns[i] for i in thu_tu],
             "frame": [frames[i] for i in thu_tu],
             "time": [times[i] for i in thu_tu],
             "theo_n": {ns[i]: (frames[i], times[i]) for i in range(len(ns))}}
        self._cache[video_id] = d
        return d

    def frame_idx(self, video_id: str, n: int) -> int:
        """n (số thứ tự keyframe) -> frame_idx (số nộp cho BTC)."""
        d = self._nap(video_id)
        if n not in d["theo_n"]:
            raise KeyError(f"{video_id} không có keyframe n={n} "
                           f"(n hợp lệ: {min(d['theo_n'])}..{max(d['theo_n'])})")
        return d["theo_n"][n][0]

    def pts_time(self, video_id: str, n: int) -> float:
        return self._nap(video_id)["theo_n"][n][1]

    def gan_nhat(self, video_id: str, frame_idx: int):
        """frame_idx bất kỳ -> (n, frame_idx) của keyframe gần nhất.

        Dùng khi hệ thống chỉ ra một vị trí trong video gốc nhưng ta cần
        keyframe tương ứng để lấy ảnh.
        """
        d = self._nap(video_id)
        fs = d["frame"]
        i = bisect.bisect_left(fs, frame_idx)
        ung_vien = [j for j in (i - 1, i) if 0 <= j < len(fs)]
        if not ung_vien:
            raise ValueError(f"{video_id} không có keyframe nào")
        j = min(ung_vien, key=lambda j: abs(fs[j] - frame_idx))
        return d["n"][j], fs[j]

    def danh_sach(self, video_id: str):
        """[(n, frame_idx, pts_time)] theo thứ tự thời gian."""
        d = self._nap(video_id)
        return list(zip(d["n"], d["frame"], d["time"]))

    def duong_dan_anh(self, video_id: str, n: int):
        """Đường dẫn file ảnh keyframe, None nếu không tìm thấy."""
        if not self.kf_dir:
            return None
        folder = self.kf_dir / video_id
        mau = (self._mau,) if self._mau else MAU_TEN_ANH
        for m in mau:
            p = folder / m.format(n=n)
            if p.exists():
                self._mau = m
                return p
        self._mau = None
        return None

    def cac_video(self):
        return sorted(p.stem for p in self.dir.glob("*.csv"))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) < 2:
        sys.exit("Dùng: python -m indexing.map_keyframes <thu-muc-map-keyframes>")
    mk = MapKeyframes(sys.argv[1])
    vs = mk.cac_video()
    print(f"{len(vs)} video")
    v = vs[0]
    ds = mk.danh_sach(v)
    print(f"\n{v}: {len(ds)} keyframe")
    print(f"  n đầu = {ds[0][0]}  (phải là 1, không phải 0)")
    print(f"  5 keyframe đầu (n, frame_idx, giây):")
    for row in ds[:5]:
        print(f"     n={row[0]:4}  frame_idx={row[1]:7}  {row[2]:7.2f}s")
    print(f"\n  Khoảng cách trung bình giữa 2 keyframe: "
          f"{(ds[-1][1] - ds[0][1]) / max(1, len(ds) - 1):.0f} frame")
