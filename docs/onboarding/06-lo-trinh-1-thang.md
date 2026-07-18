# Lộ trình 1 tháng — Đội AIC 2026 (20/7 → 16/8/2026)

> Tuần 1 = M1, Tuần 2 = M2, Tuần 3 = M3, Tuần 4 = M4 + tập dượt đầu tiên.
> Nếu BTC chưa phát dữ liệu 2026: dùng dữ liệu mùa cũ (keyframes + CLIP features các năm trước lan truyền trong cộng đồng AIC) — toàn bộ pipeline không đổi, chỉ swap dữ liệu khi BTC phát.

## Nhịp làm việc cố định (cả tháng)

- **Standup 3 dòng mỗi tối** (nhóm chat): hôm nay làm gì / mai làm gì / đang kẹt gì. Quang Dũng nhắc.
- **Demo nội bộ tối Chủ nhật hằng tuần** (60–90 phút, online): mỗi người demo phần mình chạy thật, Dũng ghi biên bản + chốt việc tuần sau.
- Luật chung: PR phải có người review; từ Tuần 2 thêm luật **không có số benchmark → không merge** (với thay đổi ảnh hưởng chất lượng).

---

## TUẦN 1 (20–26/7) — Mốc M1: KIS text chạy end-to-end

**Mục tiêu tuần:** gõ một câu tiếng Anh → thấy lưới keyframe kết quả trên UI. Xấu cũng được, phải CHẠY.

| Ai | Việc trong tuần | Deliverable cuối tuần (demo CN) |
|---|---|---|
| **Tuấn Anh** | T2–T3: tải dữ liệu, chạy `inventory.py` kiểm kê. T4–T5: metadata DB SQLite + chuẩn ID `video_id#frame`. T6–CN: FAISS `IndexFlatIP` từ CLIPFeatures sẵn + xác định đúng model CLIP của BTC | `inventory_report.md` + `faiss_clip.index` + hàm `search_vector()` chạy demo được |
| **Tiến Khang** | T2–T3: skeleton FastAPI đủ endpoint (stub) + mock 50 keyframe → **giao N3/N4 ngay T4**. T5–CN: `/search` vector-only nối index thật của Tuấn Anh, static server ảnh thumbnail | `/search` trả kết quả thật < 1s; N3/N4 đã dev được trên mock từ giữa tuần |
| **Gia Thoại** | T2–T3: đăng ký API key (Claude/GPT), đọc lại file 00 + 3 case study. T4–CN: prompt `understand()` v1 (dịch + 3 biến thể + positives/negatives) chạy trên mock; tự soạn 20 truy vấn test prompt | `understand()` demo: nhập câu tiếng Việt → JSON đúng cấu trúc trên ≥ 15/20 câu |
| **Đình Trung** | T2: chốt stack với cả đội (khuyến nghị Vite+React). T3–T5: lưới keyframe ảo hóa (react-window) + ô tìm kiếm trên mock. T6–CN: nối API thật, phím tắt cơ bản (`/`, mũi tên, Enter, Space) | UI gõ truy vấn → lưới 100+ ảnh thật, cuộn mượt |
| **Quang Dũng** | T2: lập repo, branch rule, mời cả đội, dựng kênh chat + lịch standup. T3–T4: tải **VBS-Archive**, đọc format đề các mùa. T5–CN: soạn khung `queries.jsonl` + phân bổ mỗi người viết 10 truy vấn test (deadline T4 tuần 2); bảng theo dõi GPU/Colab khả dụng | Repo hoạt động có PR đầu tiên được merge; khung benchmark + mẫu 5 truy vấn do Dũng tự viết |

**Điều kiện qua tuần (gate):** demo CN phải thấy flow text → kết quả end-to-end. Nếu tắc ở dữ liệu BTC → cả đội dồn giúp Tuấn Anh, các phần khác chạy mock thêm 2 ngày.

---

## TUẦN 2 (27/7–2/8) — Mốc M2: Video KIS + hybrid + benchmark

**Mục tiêu tuần:** dán ảnh tìm được ngay; BM25 + RRF chạy; có bảng số đầu tiên.

| Ai | Việc trong tuần | Deliverable cuối tuần |
|---|---|---|
| **Tuấn Anh** | T2–T4: chạy OCR (PaddleOCR) toàn bộ keyframe trên Colab/Kaggle (chia ca với Dũng). T5–CN: chạy ASR (faster-whisper) toàn bộ video; build BM25 index (`bm25s`) từ OCR+ASR+metadata | `search_text()` hoạt động; báo cáo độ phủ OCR/ASR (bao nhiêu % xong, chạy nền tiếp) |
| **Tiến Khang** | T2–T3: `query_image_b64` cho `/search` (encode ảnh cùng model). T4–T5: RRF hợp nhất vector + BM25, hỗ trợ `query_variants`. T6–CN: gộp keyframe→segment (T=10s) + pre-filter metadata | `/search` hybrid trả segment (start/end) < 500ms; endpoint image-search sẵn cho UI |
| **Gia Thoại** | T2–T3: hoàn thiện `understand()`: ràng buộc đồng thời + thứ tự thời gian, few-shot từ ví dụ BTC. T4–CN: khảo sát VLM rerank — thử Qwen2.5-VL (API hoặc local) chấm checklist trên 10 ảnh mẫu, đo giá + thời gian/lượt | `understand()` đạt ≥ 18/20; bảng so sánh 2–3 lựa chọn VLM (chất lượng/giá/tốc độ) để chốt |
| **Đình Trung** | T2–T3: **flow dán ảnh** (paste toàn trang → search ngay), mục tiêu < 5s. T4–T5: panel filters (thời gian/kênh/objects). T6–CN: hover phóng to + badge score + test 1000 kết quả | Demo Video KIS giả lập: chiếu 1 clip, chụp màn hình, dán, ra đáp án |
| **Quang Dũng** | T2–T3: gom đủ 50 truy vấn từ cả đội, chọn 15 câu giữ kín. T4–T6: viết `run_benchmark.py` (recall@1/5/20/100 + latency), chạy cấu hình `clip_only` vs `+bm25_rrf`. CN: công bố **bảng số đầu tiên** + áp luật "không số không merge" | Bảng benchmark v1 gửi cả đội; hỗ trợ Tuấn Anh chạy OCR/ASR song song |

**Gate:** benchmark chạy được và cho thấy hybrid ≥ clip_only. Nếu không hơn → họp bất thường tìm nguyên nhân (thường do BM25 text bẩn hoặc RRF nối sai id).

---

## TUẦN 3 (3–9/8) — Mốc M3: Reasoning + timeline

**Mục tiêu tuần:** kết quả có `reason`, có sequence view, có tìm-tương-tự; SigLIP vào trận.

| Ai | Việc trong tuần | Deliverable cuối tuần |
|---|---|---|
| **Tuấn Anh** | T2–T5: encode SigLIP toàn bộ keyframe (GPU Colab/máy trường) → `faiss_siglip.index` + text encoder cho Khang. T6–CN: hoàn tất phần OCR/ASR còn sót; versioning index (v1_clip, v2_siglip) | Index SigLIP bàn giao; benchmark của Dũng đo được kênh SigLIP riêng |
| **Tiến Khang** | T2: thêm kênh SigLIP vào RRF. T3–T4: `/video/{id}/timeline` (< 50ms) + `/keyframe/{id}/image`. T5–CN: `/search/feedback` (average query expansion α/β/γ) + `return_raw` cho N3 | 3 endpoint mới chạy; benchmark 3 kênh RRF |
| **Gia Thoại** | T2–T4: **VLM rerank checklist** trên top-50 (song song hóa, < 10s/lượt), điền `reason`. T5–CN: xác minh thứ tự thời gian qua `/timeline` (dải frame đánh số → hỏi VLM A trước B?); đọc paper NII-UIT 2025, thử nghiệm dynamic temporal search 2 chiều | Benchmark có cột `+rerank`: recall@5 KIS khó phải tăng rõ; demo 1 câu có ràng buộc thứ tự |
| **Đình Trung** | T2–T4: **sequence view/timeline panel** (dải phim ngang, phím ←→, video player seek `#t=`). T5–T6: tìm-tương-tự (phím `f`) + giỏ feedback ✓/✗ nối `/search/feedback`. CN: chọn biên `[start`/`end]` để nộp | Demo Case Study 2 giả lập: tìm gần đúng → tua timeline → xác minh sự kiện trước đó |
| **Quang Dũng** | T2–T3: máy thi chạy `run_all.sh` 1 lệnh, đo p95 các endpoint. T4–T5: tìm hiểu thể thức nộp bài BTC (DRES?) → báo format cho Trung. T6–CN: benchmark từng-câu gửi Thoại (câu nào trượt, tầng nào); bản sao index ra ổ rời | Báo cáo hạ tầng + format nộp bài; backlog tune prompt cho Thoại |

**Gate:** rerank phải cải thiện KIS khó trên benchmark. Nếu VLM quá chậm/đắt → demo CN quyết: giảm ứng viên rerank hoặc đổi model.

---

## TUẦN 4 (10–16/8) — Mốc M4: KISC + VQA + tập dượt số 1

**Mục tiêu tuần:** chạy kịch bản KISC 4 lượt của BTC; VQA trả lời được; cả đội thi thử lần đầu.

| Ai | Việc trong tuần | Deliverable cuối tuần |
|---|---|---|
| **Tuấn Anh** | T2–T3: tối ưu load index (mmap, thời gian khởi động < 1 phút). T4–CN: hỗ trợ Dũng dựng máy dự phòng; caption VLM cho tập keyframe "khó" nếu dư GPU | Máy chính + dự phòng đều tự chạy từ số 0 bằng script |
| **Tiến Khang** | T2–T3: `mode=avs` khử trùng lặp (cosine > 0.92 cùng video). T4–T5: tối ưu latency về đích < 300ms p95. T6–CN: sửa lỗi phát sinh từ tập dượt | AVS trả ranked list dài + số đo p95 đạt đích |
| **Gia Thoại** | T2–T4: **`/agent/query` vòng lặp KISC đầy đủ** (clarify khi phân tán, thu hẹp trên tập cũ, chọn câu hỏi phân tách 50/50). T5–T6: `/agent/vqa` (định vị → cửa sổ frame kéo dài về sau → VLM; đếm bằng object tags). CN: fallback khi API chết | Chạy trơn kịch bản KISC 4 lượt của BTC + 5 câu VQA nội bộ |
| **Đình Trung** | T2–T3: panel chat KISC + chip structured query sửa được + nút trả lời nhanh. T4–T5: màn nộp bài đúng format BTC (giỏ nộp, Ctrl+Enter, hiển thị to trước khi nộp). T6: đóng băng UI trước tập dượt | UI đầy đủ 100% flow không chuột |
| **Quang Dũng** | T2–T4: chuẩn bị **tập dượt số 1 (thứ Bảy 15/8)**: 12 câu từ tập kín (4 KIS, 3 AVS, 3 VQA, 2 KISC), bấm giờ theo format VBS (KIS-V 5', KIS-T 7', AVS 5', QA 5'), phân vai thi thử 2 đội hình. CN: biên bản rút kinh nghiệm → backlog tuần 5 cho từng người | Biên bản tập dượt + bảng điểm + danh sách issue ưu tiên |

**Gate cuối tháng:** tập dượt giải được ≥ 50% số câu trong thời gian quy định. Trên mức đó → tháng sau dồn vào tối ưu + luyện superuser (bài học vibro: 1/3 model, 2/3 con người và UI).

---

## Rủi ro & phương án trong tháng

| Rủi ro | Phát hiện | Phương án |
|---|---|---|
| BTC chưa phát dữ liệu 2026 | Ngay T2 tuần 1 | Dùng dữ liệu mùa cũ, pipeline giữ nguyên; Dũng hỏi BTC lịch phát |
| OCR/ASR chạy không kịp (nghìn giờ video) | Giữa tuần 2 | Ưu tiên OCR trước (giá trị cao hơn với tin tức); ASR chạy nền các tuần sau; chia nhiều tài khoản Colab |
| VLM rerank đắt/chậm | Cuối tuần 2 (bảng so sánh của Thoại) | Giảm còn 20 ứng viên × 1 frame; hoặc dùng SigLIP large re-score làm rerank rẻ tầng giữa |
| Thành viên bận (lịch học/thi) | Standup | Dũng là người trám chỗ theo thiết kế — ưu tiên trám N1 (tuần 1–2) và N3 (tuần 3–4) |
| CLIPFeatures BTC không rõ model | Tuần 1 | Thử lần lượt ViT-B/32, ViT-B/16, ViT-L/14: encode 1 câu, xem chiều vector + điểm khớp; hỏi thẳng BTC |
