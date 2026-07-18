# N1 — Onboarding: Dữ liệu & Chỉ mục (Data Engineer)

> Đọc `00-tong-quan-va-hop-dong-api.md` trước. Tài liệu này là của riêng bạn.

## Sứ mệnh của bạn

Biến đống dữ liệu thô BTC phát thành các **kho chỉ mục tra cứu trong mili-giây**. Cả đội đứng trên vai bạn: N2 không có index thì không có API, N4 không có ảnh thì không có UI. Bạn là người có deliverable sớm nhất và quan trọng nhất ở mốc M1.

Nguyên tắc: **indexing chạy batch một lần bằng GPU thuê/mượn (Colab, Kaggle, máy trường); máy thi chỉ load file.** Tách hẳn code indexing khỏi code serving.

## Phần 1 — Việc cần làm ngay (tuần đầu, mốc M1)

### 1.1 Kiểm kê dữ liệu BTC
Ngay khi BTC phát dữ liệu, viết script `indexing/inventory.py`:
- Đếm: bao nhiêu video, bao nhiêu keyframe/video, tổng dung lượng.
- Đối chiếu chéo: video nào thiếu keyframe? CLIPFeatures có đúng số dòng bằng số keyframe không? Objects có đủ file không? Metadata thiếu trường gì?
- Xuất `data/inventory_report.md` gửi cả đội. Phát hiện lệch dữ liệu sớm tiết kiệm cả tuần debug về sau.
- XÁC NHẬN với cả đội: dữ liệu 2026 là lifelog egocentric, video tin tức YouTube, hay cả hai (ảnh hưởng chiến lược OCR/ASR).

### 1.2 Chuẩn hóa ID và metadata DB
- Chốt quy ước `keyframe_id = "{video_id}#{frame_index}"` (khớp hợp đồng API mục 3).
- Dựng metadata DB bằng **SQLite** (hoặc parquet + DuckDB): bảng `keyframes(keyframe_id, video_id, frame_index, timestamp_sec, objects_json)`, bảng `videos(video_id, title, description, keywords, publish_date, length_sec, watch_url, channel)`.
- Timestamp của keyframe: nếu BTC không phát kèm, tính từ frame_index / fps (đọc fps bằng ffprobe).

### 1.3 FAISS baseline từ CLIPFeatures sẵn
- Load toàn bộ CLIPFeatures → chuẩn hóa L2 → `faiss.IndexFlatIP` → lưu `data/faiss_clip.index` + mảng `keyframe_ids.npy` cùng thứ tự.
- ~1 triệu vector float16 chỉ vài GB, vừa RAM, tìm ~50ms trên CPU. KHÔNG dùng IVF/HNSW vội — exact search cho recall tối đa, tốc độ tính sau.
- QUAN TRỌNG: hỏi BTC/đọc kỹ tài liệu xem CLIPFeatures dùng model CLIP nào (ViT-B/32? ViT-L/14?). Text encoder phía truy vấn PHẢI cùng model thì mới so được. Ghi rõ vào README.
- Bàn giao cho N2: module `indexing/stores.py` với hàm `search_vector(vec, top_k, filter_ids=None) -> [(keyframe_id, score)]`.

## Phần 2 — Làm giàu tín hiệu (mốc M2, chạy nền liên tục)

Thứ tự ưu tiên theo giá trị/chi phí (đã học từ slide BTC — họ liệt kê rõ OCR, ASR trong pipeline chuẩn):

1. **OCR** (PaddleOCR, multilingual): chạy trên TẤT CẢ keyframe. Với video tin tức, headline/phụ đề trên màn hình là mỏ vàng; với lifelog là biển hiệu, tên đường. Lưu `ocr(keyframe_id, text)`.
2. **ASR** (faster-whisper; cân nhắc PhoWhisper cho tiếng Việt): chạy trên audio toàn bộ video, lưu theo đoạn `asr(video_id, start_sec, end_sec, text)`.
3. **SigLIP re-encode** (google/siglip-so400m-patch14-384): encode lại toàn bộ keyframe → `faiss_siglip.index`. SigLIP retrieval tốt hơn CLIP đáng kể; giữ CẢ HAI index để N2 fusion.
4. **Caption** (nếu còn GPU budget): BLIP-2 hoặc Qwen2-VL sinh 1 câu mô tả/keyframe → nuôi BM25 + làm `reason` hiển thị.

### 2.1 BM25 index
- Thư viện `bm25s` (nhanh, thuần Python, không cần Elasticsearch).
- Mỗi document = 1 keyframe, text = OCR + caption + ASR (đoạn chứa timestamp keyframe) + title/keywords của video.
- Bàn giao: `search_text(query, top_k, filter_ids=None) -> [(keyframe_id, score)]`.
- Lưu ý tiếng Việt: giữ nguyên có dấu + thêm bản không dấu (người thi gõ nhanh thường bỏ dấu).

## Phần 3 — Cạm bẫy đã biết trước (từ slide BTC)

- **Keyframe là I-frame nén, không phải keyframe ngữ nghĩa** — phân bố không đều theo nội dung. Đừng cố "chọn lọc bớt" keyframe; index hết, vì BTC cam kết đoạn đáp án chứa ≥1 keyframe.
- **Video lifelog 5 tiếng liên tục**: rung lắc, ánh sáng đổi. Nếu chất lượng embedding kém trên frame mờ → cân nhắc lọc/đánh dấu frame mờ (variance of Laplacian) nhưng CHỈ đánh dấu, không xóa khỏi index.
- Objects của BTC từ model 2018 (FasterRCNN/OpenImages) — dùng làm bộ lọc thô, đừng tin confidence tuyệt đối.
- Đừng phí thời gian dựng Milvus — FAISS + SQLite là đủ cho 1M vector, thời gian đó dành cho OCR/ASR có giá trị hơn nhiều.

## Công cụ & tài liệu cần học

- FAISS: `IndexFlatIP`, `IDSelector` (lọc theo tập id) — đọc wiki FAISS phần "Guidelines to choose an index".
- HuggingFace: cách batch-encode ảnh hiệu quả (dataloader + fp16 + `torch.no_grad`).
- ffmpeg/ffprobe: trích frame, đọc fps, cắt audio.
- bm25s, PaddleOCR, faster-whisper.

## Definition of Done của bạn

- [ ] `inventory_report.md` — cả đội đã đọc, mọi lệch dữ liệu có ghi chú.
- [ ] SQLite metadata DB + tài liệu schema 1 trang.
- [ ] `faiss_clip.index` + `search_vector()` chạy <100ms/truy vấn trên máy thi (M1).
- [ ] OCR + ASR phủ 100% dữ liệu; BM25 `search_text()` hoạt động (M2).
- [ ] `faiss_siglip.index` + script encode text query SigLIP cho N2 (M2–M3).
- [ ] Script tải/tái tạo toàn bộ `data/` từ đầu bằng 1 lệnh (phòng mất máy).
- [ ] Mọi index có version (`v1_clip`, `v2_siglip`...) — N5 cần để so benchmark giữa các phiên bản.

## Giao tiếp với người khác

- **N2** là khách hàng trực tiếp: thống nhất chữ ký hàm `search_vector/search_text/get_metadata` ngay tuần đầu, đổi gì báo trước.
- **N3** cần biết model encode text nào khớp với index nào (CLIP text encoder cho index CLIP, SigLIP cho SigLIP).
- **N5** cần bạn cung cấp thời gian build và version của từng index.
