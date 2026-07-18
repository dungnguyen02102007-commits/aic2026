# 07 — Hướng dẫn Google Colab cho đội (hạ tầng GPU miễn phí)

> Chủ trì: Quang Dũng (N5). Người dùng chính: Tuấn Anh (N1). Cả đội đều có thể nhận ca chạy.

## 1. Kiến trúc làm việc trên Colab

```
GitHub (repo aic2026)            Google Drive chung (AIC2026/)
  indexing/notebooks/*.ipynb  →    data/      ← dữ liệu BTC (upload 1 lần)
  (mở trực tiếp vào Colab)         outputs/   ← kết quả các notebook ghi về
```

- **Code sống ở GitHub** (sửa qua PR như mọi code khác). **Dữ liệu và kết quả sống ở Drive chung** — không bao giờ nằm trong git.
- Notebook nào cũng theo pattern **resume theo shard**: kết quả ghi thành từng mảnh lên Drive, chạy lại tự bỏ qua mảnh đã xong → Colab ngắt phiên (chuyện chắc chắn xảy ra) không mất gì.

## 2. Thiết lập một lần (Dũng làm, ~15 phút)

1. **Tạo thư mục Drive chung**: Google Drive → New folder `AIC2026` → tạo con: `data/keyframes`, `data/clip_features`, `data/videos`, `data/objects`, `data/metadata`, `outputs/`.
2. **Chia sẻ** thư mục `AIC2026` cho email Google của cả 5 thành viên, quyền **Editor**.
3. Mỗi thành viên mở [drive.google.com](https://drive.google.com) → Shared with me → chuột phải `AIC2026` → **"Add shortcut to Drive"** (bắt buộc — notebook mount qua đường dẫn `MyDrive/AIC2026`).
4. Khi BTC phát dữ liệu: upload vào đúng các thư mục `data/…` (dùng Drive for Desktop upload ổn định hơn trình duyệt với file lớn).

## 3. Cách mở notebook (mỗi thành viên)

1. Vào [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub**.
2. Đăng nhập GitHub (repo private nên Colab sẽ xin quyền) → gõ `dungnguyen02102007-commits/aic2026` → chọn notebook trong `indexing/notebooks/`.
3. Cần GPU: **Runtime → Change runtime type → T4 GPU** (notebook nào cần đã ghi rõ ở đầu).
4. Chạy cell đầu → cấp quyền mount Drive → sửa cell CONFIG (nhất là `SHARD_START/END`) → Run all.
5. **Lưu thay đổi notebook**: File → Save a copy in GitHub → chọn repo + branch của mình → mở PR như thường lệ. (Đừng Save copy in Drive — code phải về GitHub.)

## 4. Chia ca nhiều tài khoản (chiến thuật vượt giới hạn Colab free)

Colab free: ~3–4 giờ GPU/phiên, có thể bị ngắt bất kỳ lúc nào, giới hạn theo ngày theo tài khoản. Cách đội mình dùng 5 tài khoản:

1. Dũng lập **bảng chia ca** trong nhóm chat, ví dụ SigLIP encode 1 triệu ảnh = 500 shard:

| Ai | SHARD_START | SHARD_END |
|---|---|---|
| Tuấn Anh | 0 | 99 |
| Tiến Khang | 100 | 199 |
| Gia Thoại | 200 | 299 |
| Đình Trung | 300 | 399 |
| Quang Dũng | 400 | 499 |

2. Mỗi người mở cùng notebook, sửa đúng dải của mình, Run all, để máy chạy (tab phải mở, thi thoảng ngó — Colab hỏi "Are you still there?").
3. Kết quả đổ chung về `outputs/` trên Drive — vì shard đánh số toàn cục nên không giẫm nhau. Ai xong dải thì nhận dải tiếp.
4. Việc nhẹ ưu tiên chạy trước: **OCR (M2) → ASR (M2, nền dài hạn) → SigLIP (M3)**.

## 5. Quy tắc an toàn

- **KHÔNG dán API key vào notebook** (notebook nằm trong git!). Cần key → dùng Colab Secrets (biểu tượng chìa khóa 🔑 sidebar trái) rồi `from google.colab import userdata; userdata.get('TEN_KEY')`.
- Không sửa trực tiếp file trong `data/` từ notebook — chỉ đọc `data/`, chỉ ghi `outputs/`.
- File ghi Drive dùng pattern ghi `.tmp` rồi rename (notebook 04 có mẫu) — tránh file hỏng khi ngắt phiên.
- Trước khi chạy dải lớn, chạy thử `SHARD_START=0, SHARD_END=0` (1 shard) xem output đúng format đã.

## 6. Bốn notebook hiện có

| Notebook | Việc | GPU | Mốc |
|---|---|---|---|
| `01_faiss_baseline_clipfeatures.ipynb` | FAISS từ CLIPFeatures BTC + dò model CLIP | Không | M1 |
| `02_siglip_encode.ipynb` | Encode SigLIP 1M keyframe → index v2 | Có | M3 |
| `03_ocr_batch.ipynb` | PaddleOCR chữ trên màn hình | Nên | M2 |
| `04_asr_batch.ipynb` | faster-whisper tiếng Việt theo video | Có | M2+ |

Cần notebook mới (caption VLM, benchmark model…) → mở issue gắn label `N1-du-lieu`.
