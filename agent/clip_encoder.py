"""Bọc CLIP để dùng ngoài backend (TRAKE cần encode frame trích từ video).

Dùng chung hằng số model với backend/stores_faiss.py — chỉ có MỘT nơi khai báo
`ViT-B-32-quickgelu`, tránh cảnh hai chỗ dùng hai model khác nhau mà không ai
biết (điểm tụt âm thầm, không có lỗi nào báo ra).
"""
import numpy as np

from backend.stores_faiss import MODEL_NAME, PRETRAINED


class ClipEncoder:
    def __init__(self, model_name=MODEL_NAME, pretrained=PRETRAINED):
        import open_clip
        import torch

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @staticmethod
    def _chuan_hoa(V):
        return V / np.linalg.norm(V, axis=1, keepdims=True)

    def encode_text(self, texts: list[str]) -> np.ndarray:
        with self._torch.no_grad():
            V = self.model.encode_text(self.tokenizer(texts)).float().numpy()
        return self._chuan_hoa(V)

    def encode_images(self, images, batch: int = 64) -> np.ndarray:
        """images: danh sách ndarray RGB (từ extract_frames) hoặc PIL.Image."""
        from PIL import Image

        out = []
        for i in range(0, len(images), batch):
            chunk = [im if isinstance(im, Image.Image) else Image.fromarray(im)
                     for im in images[i:i + batch]]
            tensors = self._torch.stack([self.preprocess(im) for im in chunk])
            with self._torch.no_grad():
                out.append(self.model.encode_image(tensors).float().numpy())
        return self._chuan_hoa(np.concatenate(out)) if out else np.zeros((0, 512))
