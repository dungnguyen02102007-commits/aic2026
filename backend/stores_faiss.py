"""Store thật: FAISS + SQLite (N2 - Tiến Khang), issue #5 và #20.

Cùng chữ ký hàm với MockStore nên backend/main.py không phải đổi gì —
chỉ cần đặt biến môi trường:

    AIC_STORE=faiss
    AIC_INDEX_PATH=.../outputs/clip_flat.faiss
    AIC_IDS_PATH=.../outputs/ids.json
    AIC_DB_PATH=.../data/metadata.db
    AIC_KEYFRAMES_DIR=.../data/keyframes      # tùy chọn, để serve ảnh

Ba chỗ SAI LÀ HỎNG ÂM THẦM, không có lỗi nào báo ra (docs/onboarding/10):

  1. Model PHẢI là ViT-B-32-quickgelu. Bỏ hậu tố -quickgelu hoặc dùng
     ViT-B-16 đều cho vector 512 chiều nên code chạy trơn, nhưng điểm khớp
     tụt từ ~0.33 xuống ~0.10 (mức nhiễu).
  2. PHẢI normalize_L2 vector truy vấn, nếu không inner product != cosine.
  3. Thứ tự ids.json PHẢI khớp thứ tự vector trong index. Lệch một dòng là
     mọi kết quả trả về sai ảnh và sai frame_idx.
"""
import json
import os
import sqlite3
from pathlib import Path

import numpy as np

# Model BTC dùng để trích CLIPFeatures — BTC ghi rõ "clip-ViT-B-32" trong tài
# liệu sơ tuyển, và đo thực nghiệm xác nhận biến thể quickgelu (0.329 vs 0.099).
MODEL_NAME = "ViT-B-32-quickgelu"
PRETRAINED = "openai"

# Thứ tự thử khi dò tên file ảnh keyframe (BTC các mùa đặt tên không đồng nhất)
_IMG_PATTERNS = ("{n:03d}.jpg", "{n:04d}.jpg", "{n:06d}.jpg", "{n}.jpg",
                 "{n:03d}.jpeg", "{n:03d}.png")


class FaissStore:
    def __init__(self, index_path, ids_path, db_path, keyframes_dir=None):
        import faiss

        self._faiss = faiss
        self.index = faiss.read_index(str(index_path))
        with open(ids_path, encoding="utf-8") as f:
            self.ids = json.load(f)

        if self.index.ntotal != len(self.ids):
            raise RuntimeError(
                f"Index có {self.index.ntotal} vector nhưng ids.json có "
                f"{len(self.ids)} id. Lệch nhau nghĩa là mọi kết quả sẽ trỏ sai "
                f"keyframe — dựng lại index (indexing/notebooks/01).")

        self.pos = {kf: i for i, kf in enumerate(self.ids)}
        self.dim = self.index.d

        self.db = sqlite3.connect(str(db_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.keyframes_dir = Path(keyframes_dir) if keyframes_dir else None
        self._img_pattern = None  # nhớ pattern đã dò được, khỏi stat lại mỗi lần

        self._model = None  # nạp lười: khởi động backend không cần chờ tải model

    # --- Encoder ---------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        import open_clip
        import torch

        self._torch = torch
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        got = self._model.text_projection.shape[1] if hasattr(
            self._model, "text_projection") else None
        if got and got != self.dim:
            raise RuntimeError(
                f"Model {MODEL_NAME} cho vector {got} chiều nhưng index là "
                f"{self.dim} chiều — sai model, kết quả sẽ vô nghĩa.")

    # --- Tìm kiếm --------------------------------------------------------

    def _search(self, vec: np.ndarray, top_k: int, filter_ids=None):
        vec = np.ascontiguousarray(vec.astype("float32"))
        self._faiss.normalize_L2(vec)  # BẮT BUỘC: inner product = cosine

        params = None
        if filter_ids is not None:
            rows = [self.pos[k] for k in filter_ids if k in self.pos]
            if not rows:
                return []
            try:
                sel = self._faiss.IDSelectorBatch(
                    np.array(rows, dtype="int64"))
                params = self._faiss.SearchParameters(sel=sel)
            except Exception:  # noqa: BLE001 — bản faiss cũ không có SearchParameters
                params = None
                top_k = min(self.index.ntotal, max(top_k * 20, 5000))

        if params is not None:
            D, I = self.index.search(vec, top_k, params=params)
        else:
            D, I = self.index.search(vec, top_k)

        hits = [(self.ids[i], float(d)) for d, i in zip(D[0], I[0]) if i != -1]
        if filter_ids is not None and params is None:
            allowed = set(filter_ids)
            hits = [h for h in hits if h[0] in allowed]
        return hits

    def search_text(self, query: str, top_k: int = 100, filter_ids=None):
        """query PHẢI là tiếng Anh — CLIP không hiểu tiếng Việt.
        Dùng agent.understand() để dịch trước."""
        self._ensure_model()
        with self._torch.no_grad():
            v = self._model.encode_text(self._tokenizer([query])).float().numpy()
        return self._search(v, top_k, filter_ids)

    def search_image(self, image_bytes: bytes, top_k: int = 100, filter_ids=None):
        import io

        from PIL import Image

        self._ensure_model()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = self._preprocess(img).unsqueeze(0)
        with self._torch.no_grad():
            v = self._model.encode_image(tensor).float().numpy()
        return self._search(v, top_k, filter_ids)

    # --- Tra cứu metadata ------------------------------------------------

    @staticmethod
    def _row_to_meta(row):
        return {"keyframe_id": row["keyframe_id"], "video_id": row["video_id"],
                "n": row["n"], "frame_idx": row["frame_idx"],
                "timestamp": row["timestamp"]}

    def get_metadata(self, keyframe_id: str):
        row = self.db.execute(
            "SELECT keyframe_id, video_id, n, frame_idx, timestamp "
            "FROM keyframes WHERE keyframe_id = ?", (keyframe_id,)).fetchone()
        return self._row_to_meta(row) if row else None

    def timeline(self, video_id: str, around: str | None = None, window: int = 30):
        rows = self.db.execute(
            "SELECT keyframe_id, video_id, n, frame_idx, timestamp "
            "FROM keyframes WHERE video_id = ? ORDER BY n", (video_id,)).fetchall()
        kfs = [self._row_to_meta(r) for r in rows]
        if around:
            center = next((i for i, k in enumerate(kfs)
                           if k["keyframe_id"] == around), None)
            if center is not None:
                kfs = kfs[max(0, center - window):center + window + 1]
        return kfs

    def keyframe_ids_for_videos(self, video_ids):
        if not video_ids:
            return []
        marks = ",".join("?" * len(video_ids))
        rows = self.db.execute(
            f"SELECT keyframe_id FROM keyframes WHERE video_id IN ({marks})",
            list(video_ids)).fetchall()
        return [r["keyframe_id"] for r in rows]

    def stats(self):
        n_videos = self.db.execute(
            "SELECT COUNT(DISTINCT video_id) c FROM keyframes").fetchone()["c"]
        return {"index_version": f"faiss_flat_ip_{MODEL_NAME}_{self.dim}d",
                "n_keyframes": self.index.ntotal, "n_videos": n_videos}

    # --- Ảnh keyframe ----------------------------------------------------

    def keyframe_image_path(self, keyframe_id: str):
        if not self.keyframes_dir:
            return None
        meta = self.get_metadata(keyframe_id)
        if not meta:
            return None
        folder = self.keyframes_dir / meta["video_id"]
        n = meta["n"]
        patterns = ((self._img_pattern,) if self._img_pattern
                    else _IMG_PATTERNS)
        for pat in patterns:
            p = folder / pat.format(n=n)
            if p.exists():
                self._img_pattern = pat
                return p
        self._img_pattern = None  # pattern cũ hết đúng -> dò lại lần sau
        return None


def from_env():
    """Dựng FaissStore từ biến môi trường. Báo lỗi rõ ràng nếu thiếu file."""
    need = {"AIC_INDEX_PATH": "clip_flat.faiss",
            "AIC_IDS_PATH": "ids.json",
            "AIC_DB_PATH": "metadata.db"}
    paths = {}
    for var, ten in need.items():
        val = os.environ.get(var)
        if not val:
            raise RuntimeError(
                f"Thiếu biến môi trường {var} (đường dẫn tới {ten}). "
                f"Xem hướng dẫn ở đầu backend/stores_faiss.py")
        if not Path(val).exists():
            raise RuntimeError(f"{var} trỏ tới file không tồn tại: {val}")
        paths[var] = val

    return FaissStore(paths["AIC_INDEX_PATH"], paths["AIC_IDS_PATH"],
                      paths["AIC_DB_PATH"],
                      os.environ.get("AIC_KEYFRAMES_DIR"))
