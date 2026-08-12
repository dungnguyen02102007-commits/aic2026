# N2 — Onboarding: Lõi truy hồi Backend (Backend Engineer)

> Đọc `00-tong-quan-va-hop-dong-api.md` trước. Bạn là CHỦ SỞ HỮU của hợp đồng API trong đó — mọi thay đổi schema do bạn quyết và thông báo.

## Sứ mệnh của bạn

Biến các index của N1 thành một **API tìm kiếm nhanh, ổn định, đúng format cuộc thi**. N4 (UI) và N3 (agent) đều gọi vào bạn. Chất lượng xếp hạng thô (trước rerank) và độ trễ là KPI của bạn: mục tiêu **< 300ms** cho một truy vấn /search đầy đủ trên máy thi.

## Phần 1 — Dựng khung (mốc M1)

### 1.1 FastAPI skeleton
- `backend/main.py`: dựng đủ các endpoint trong hợp đồng API (mục 3 file 00) — `/search`, `/search/feedback`, `/video/{id}/timeline`, và proxy `/agent/*` (M1 chỉ cần stub trả lỗi 501 cho phần agent).
- Load index của N1 MỘT LẦN lúc khởi động (singleton), không load lại mỗi request.
- CORS mở cho localhost để N4 dev.
- Ngay khi xong skeleton với dữ liệu giả (mock 20 keyframe), đưa cho N4 và N3 — họ cần nó để làm việc song song, đừng đợi index thật.

### 1.2 /search phiên bản 1 (chỉ vector)
- Nhận `query_text` → encode bằng text encoder KHỚP với index (hỏi N1: CLIP model nào) → `search_vector` → map keyframe_id sang metadata → trả kết quả.
- Nhận `query_image_b64` → decode → encode bằng image encoder cùng model → tìm. Đây là nền của **Video KIS** — tính năng thắng thua về tốc độ, làm sớm.

## Phần 2 — Hybrid retrieval + xử lý thời gian (mốc M2)

### 2.1 Reciprocal Rank Fusion (RRF)
Chạy song song (asyncio/thread) vector search và BM25, hợp nhất:

```
score(kf) = Σ_channels 1 / (60 + rank_channel(kf))
```

- k=60 là chuẩn, không cần tune. RRF chỉ dùng THỨ HẠNG nên miễn nhiễm việc thang điểm CLIP/BM25 khác nhau — đây là lý do chọn nó thay vì weighted sum.
- Khi N1 có thêm index SigLIP: thêm làm kênh thứ 3, cũng qua RRF.
- Nhiều biến thể truy vấn (`query_variants` từ N3): mỗi biến thể là một lượt tìm, RRF gộp tất cả.

### 2.2 Gộp keyframe → segment (temporal grouping)
- Sau fusion lấy top ~1000 keyframe. Nhóm các keyframe **cùng video và cách nhau < T giây** (bắt đầu với T=10, N5 sẽ tune) thành 1 segment.
- Điểm segment = max điểm keyframe thành viên (+ bonus nhỏ nếu nhiều keyframe cùng khớp, vd `max + 0.1*log(count)`).
- `start_time/end_time` = min/max timestamp thành viên, nới mỗi bên 2–3 giây. **Chỉ để UI tua video, KHÔNG dùng nộp bài.**
- `frame_idx` của segment = `frame_idx` của keyframe **điểm cao nhất** — đó là khung hình nộp cho BTC (`docs/onboarding/10`).
- Đây là bước biến "frame khớp" thành "khoảnh khắc" đúng yêu cầu đề bài.

### 2.3 ~~Chế độ AVS~~ (hoãn — không có ở vòng sơ tuyển)
Vòng sơ tuyển chỉ có **Textual KIS / Q&A / TRAKE** (`docs/onboarding/10`). Code `mode="avs"` giữ nguyên trong `main.py` phòng vòng sau, nhưng **ngừng đầu tư**.

### 2.4 Filters
- `filters.objects`: tra bảng objects trong SQLite → tập keyframe_id hợp lệ → truyền `filter_ids` xuống `search_vector/search_text` (FAISS `IDSelector`).
- `filters.publish_date/channel/video_ids`: lọc từ metadata DB trước, đưa tập id vào search. LUÔN lọc-trước-tìm-sau (pre-filter), không tìm xong mới lọc (mất recall).

### 2.5 Chạy backend với index thật

`backend/main.py` chọn store bằng biến môi trường, không phải sửa code:

```powershell
$env:AIC_STORE="faiss"
$env:AIC_INDEX_PATH="D:\AIC2026\outputs\clip_flat.faiss"
$env:AIC_IDS_PATH="D:\AIC2026\outputs\ids.json"
$env:AIC_DB_PATH="D:\AIC2026\data\metadata.db"
$env:AIC_KEYFRAMES_DIR="D:\AIC2026\data\keyframes"
python -m uvicorn backend.main:app --port 8000
```

Bỏ hết biến này thì chạy `MockStore` như cũ (dev UI không cần index).

**Dev khi chưa tải index thật về máy:**
```powershell
python -m indexing.make_fake_index    # dựng index CLIP nhỏ từ caption mock
python backend/test_api.py            # 29 phép thử trên cả 2 store
```
`make_fake_index.py` encode caption của MockStore bằng **chính model CLIP thật**, nên tìm kiếm ngữ nghĩa hoạt động thật trên máy — gõ "news anchor" ra đúng cảnh người dẫn chương trình.

## Phần 3 — Phục vụ vòng lặp KISC (mốc M3–M4)

### 3.1 /search/feedback — average query expansion
Kỹ thuật cổ điển từ BoVW (CVPR 2007) vẫn ăn điểm năm 2026:
```
new_vec = normalize( α·text_vec + β·mean(positive_vecs) − γ·mean(negative_vecs) )
```
Bắt đầu α=0.5, β=0.5, γ=0.25 — N5 tune bằng benchmark. Trả kết quả như /search.

### 3.2 /video/{id}/timeline
- Trả `window` keyframe trước/sau một keyframe trong cùng video (id, timestamp, thumbnail path). N4 dùng cho sequence view — endpoint này phải < 50ms vì được gọi liên tục khi duyệt.

### 3.3 Hỗ trợ N3
- Thêm param nội bộ `return_raw=true` trả cả top keyframe chưa gộp — N3 cần frame lẻ để đưa vào VLM verify.
- Endpoint `/keyframe/{id}/image` trả ảnh JPEG (N3 gửi cho VLM, N4 hiển thị).

## Phần 4 — Cạm bẫy đã biết trước

- **Điểm số chỉ để xếp hạng.** Đừng bao giờ đặt ngưỡng tuyệt đối trên điểm CLIP (Case Study 3 của BTC: đúng/sai chênh nhau 0.02). Ngưỡng nếu cần thì đặt trên thứ hạng.
- **Độ trễ chết người ở serialize ảnh**: đừng nhét base64 thumbnail vào response /search — trả đường dẫn, N4 lấy ảnh qua static file server (FastAPI `StaticFiles` mount thư mục keyframes).
- Encode text query cần model chạy được trên CPU máy thi hoặc GPU nhỏ — load 1 lần, warm-up lúc khởi động.
- Log MỌI truy vấn + kết quả top-10 ra file (jsonl) — N5 cần để phân tích sau tập dượt, và chính bạn cần để debug "tại sao lúc nãy ra mà giờ không ra".

## Công cụ & tài liệu cần học

- FastAPI + uvicorn, asyncio cơ bản (chạy 2 kênh search song song).
- FAISS search + IDSelector (phối hợp N1).
- RRF: đọc paper gốc Cormack 2009 (2 trang) hoặc bất kỳ blog RRF nào.
- pytest: mỗi endpoint ít nhất 1 test với dữ liệu giả.

## Definition of Done của bạn

- [ ] Skeleton API + mock data cho N3/N4 (tuần đầu, M1).
- [ ] /search vector-only với text + image query (M1).
- [ ] Hybrid RRF (CLIP + SigLIP + BM25) + temporal grouping + filters (M2).
- [ ] mode=avs khử trùng lặp; /search/feedback; /timeline (M3).
- [ ] Toàn pipeline /search < 300ms p95 trên máy thi; /timeline < 50ms (M4–M5).
- [ ] Query log jsonl + README cách chạy server bằng 1 lệnh.

## Giao tiếp với người khác

- **N1**: thống nhất chữ ký `search_vector/search_text/get_metadata` tuần đầu; bạn là người phát hiện index lỗi sớm nhất — báo ngay.
- **N3**: agent gọi /search nhiều lần mỗi lượt hội thoại — giữ API idempotent, không state ẩn; cung cấp `return_raw` và ảnh keyframe.
- **N4**: mọi thay đổi schema response phải versioned và báo trước 1 ngày.
- **N5**: expose endpoint `/health` + `/stats` (số vector, version index đang load) để benchmark script kiểm tra môi trường.
