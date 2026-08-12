# AIC 2026 — Tổng quan hệ thống & Hợp đồng API chung

> Tài liệu này BẮT BUỘC đọc trước khi đọc tài liệu riêng của từng người (01→05).
> Mục đích: cả đội dùng chung một bức tranh và một hợp đồng dữ liệu, để 5 người làm song song không giẫm chân nhau.

## 1. Cuộc thi trong 10 dòng

- AI Challenge HCMC 2026, bảng A (Sinh viên). Bài toán: tìm kiếm khoảnh khắc trong kho video lớn (hàng nghìn giờ, thiên hướng lifelog/egocentric + có thể lẫn video YouTube tiếng Việt).
- 4 dạng bài:
  - **KIS** (Known-Item Search): tìm MỘT khoảnh khắc duy nhất. Có 2 dạng nhỏ: **Textual KIS** (mô tả văn bản) và **Video KIS** (BTC chiếu 1 clip, phải tìm đúng clip đó). Chấm Top-1/Top-5, điều kiện khớp NGHIÊM NGẶT (mọi yếu tố phải đồng thời xuất hiện).
  - **AVS** (Ad-hoc Video Search): tìm TẤT CẢ phân đoạn khớp mô tả, trả danh sách xếp hạng. Truy vấn có thể chứa điều kiện loại trừ ("ăn ở nhà KHÔNG tính").
  - **VQA**: câu hỏi tự nhiên về video, trả lời TEXT ngắn có suy luận (đếm, thứ tự sự kiện, "sau đó ai...").
  - **KISC** (Conversational KIS): hội thoại nhiều lượt, trợ lý CHỦ ĐỘNG hỏi lại để thu hẹp. Đây là ĐIỂM NHẤN CÔNG NGHỆ 2026 theo BTC.
- Dữ liệu BTC phát sẵn: **Videos, Keyframes (I-frame), Objects (FasterRCNN/OpenImages, 600 lớp), CLIPFeatures (embedding tính sẵn), Metadata YouTube** (title, description, publish_date, length, watch_url...). BTC cam kết: đoạn cần tìm chứa ít nhất 1 keyframe.
- Ba thách thức BTC nêu ("Big Three"): semantic gap → dùng foundation models; data sparsity/quy mô → BẮT BUỘC lọc thô rẻ trước, model đắt chỉ chạy trên top nhỏ; temporal logic → "cởi mũ TRƯỚC KHI vào phòng" ≠ "vào phòng RỒI cởi mũ".

## 2. Kiến trúc: 4 tầng + 1 mảng xuyên suốt

```
[N4 UI]  ⇄  [N3 Agent/Reasoning]  ⇄  [N2 Retrieval Core]  ⇄  [N1 Index Stores]
                    (N5 Đánh giá & Vận hành bao trùm cả 4 tầng)
```

- **N1 — Dữ liệu & chỉ mục (offline)**: FAISS (CLIP sẵn → SigLIP), BM25 (caption/OCR/ASR/metadata), metadata DB.
- **N2 — Lõi truy hồi (backend)**: hybrid search + RRF, gộp keyframe→segment, query-by-image, filter, API.
- **N3 — Agent & suy luận**: LLM hiểu truy vấn (dịch, mở rộng, tách ràng buộc ±/thứ tự), VLM rerank-verify theo checklist, VQA đa bước, KISC clarification.
- **N4 — Giao diện**: lưới keyframe + phím tắt, timeline browsing, tìm bằng ảnh chụp màn hình, chat KISC, màn nộp bài.
- **N5 — Đánh giá & vận hành**: benchmark nội bộ 50 truy vấn, đo recall@K + latency, hạ tầng GPU, tập dượt, báo cáo kỹ thuật.

Nguyên tắc vàng: **mọi thứ nặng làm offline; online chỉ tra cứu + rerank nhẹ. Điểm số CLIP tuyệt đối không đáng tin — thứ hạng + verify + mắt người mới quyết định.**

## 3. Hợp đồng API (v1 — N2 là chủ sở hữu, muốn đổi phải báo cả đội)

Mọi ID và thời gian thống nhất:
- `video_id`: string, đúng theo tên BTC phát (vd `"L01_V001"`).
- `keyframe_id`: string, `"{video_id}#{n:06d}"` — `n` là **cột `n` của map-keyframes CSV, ĐẾM TỪ 1**, khớp tên file ảnh keyframe. Không phải chỉ số khung hình.
- `frame_idx`: int — **chỉ số khung hình trong video gốc. ĐÂY là con số nộp cho BTC.** Lấy từ cột `frame_idx` của map-keyframes CSV.
- Thời gian: giây (float), không dùng chuỗi "hh:mm:ss" trong API (UI tự format khi hiển thị). **Giây chỉ để tua video, KHÔNG dùng để nộp bài.**

> ⚠️ Phân biệt `n` và `frame_idx`: keyframe thứ 3 của một video có `n = 3` nhưng `frame_idx` có thể là 750. Nhầm hai cái này là nộp sai toàn bộ. Xem `docs/onboarding/10`.

> ⚠️ `keyframe_id` chứa ký tự `#` — client gọi API **phải** mã hoá thành `%23` (`encodeURIComponent` trong JS, `urllib.parse.quote` trong Python). Không mã hoá thì trình duyệt cắt phần sau `#` thành fragment và server nhận thiếu đường dẫn.

### 3.1 POST /search — tìm kiếm chính (KIS/AVS)

```jsonc
// Request
{
  "query_text": "a man dropping keys near a fruit stall",  // đã là tiếng Anh thị giác (N3 xử lý), null nếu tìm bằng ảnh
  "query_variants": ["...", "..."],          // các biến thể mở rộng, optional
  "query_image_b64": null,                    // base64 JPEG khi tìm bằng ảnh (Video KIS / query-by-example)
  "filters": {
    "video_ids": null,                        // giới hạn trong tập video, dùng cho KISC thu hẹp dần
    "publish_date_from": null, "publish_date_to": null,
    "objects": ["dog"],                       // phải chứa object tag (Open Images label)
    "channel": null
  },
  "top_k": 100,
  "mode": "kis"                               // vòng sơ tuyển: "kis" | "qa" | "trake" (docs 10)
}
// Response
{
  "results": [
    {
      "video_id": "L01_V001",
      "keyframe_id": "L01_V001#000042",         // n đếm từ 1, khớp tên file ảnh
      "frame_idx": 4521,                          // CON SỐ NỘP CHO BTC
      "start_time": 751.2, "end_time": 765.4,   // biên segment sau khi gộp, chỉ để tua video
      "score": 0.83,                              // điểm đã fusion, chỉ dùng để XẾP HẠNG
      "signals": {"clip": 0.31, "bm25": 12.4},   // debug
      "reason": null                              // N3 điền sau khi rerank, N2 để null
    }
  ],
  "latency_ms": 87
}
```

Khi segment gộp nhiều keyframe, `keyframe_id` và `frame_idx` là của keyframe **điểm cao nhất** trong segment — đó là khung hình sẽ nộp.

### 3.2 POST /search/feedback — relevance feedback (KISC / average query expansion)

```jsonc
{ "positive_keyframe_ids": ["L01_V001#004521", "..."],
  "negative_keyframe_ids": [],
  "query_text": "...",        // giữ nguyên truy vấn gốc để trộn
  "top_k": 100 }
// Response: giống /search
```

### 3.3 GET /video/{video_id}/timeline?around={keyframe_id}&window=30
Trả dải keyframe trước/sau một keyframe (phục vụ sequence view / temporal navigation).

### 3.4 POST /agent/query — cổng vào của N3 (UI gọi cái này, không gọi /search trực tiếp khi ở chế độ trợ lý)

```jsonc
// Request
{ "session_id": "abc", "message": "tôi gặp bạn cũ tuần trước ở quán cà phê", "mode": "kisc" }
// Response
{
  "reply": "Bạn có nhớ quán trong nhà hay ngoài trời không?",   // câu trả lời/hỏi lại cho user
  "action": "clarify" | "results" | "answer",
  "structured_query": { "positives": [...], "negatives": [...], "temporal": [...] }, // để debug/hiển thị
  "results": [ /* như /search, đã có reason */ ],
  "answer": null                                                  // dùng cho VQA
}
```

### 3.5 POST /agent/vqa
```jsonc
{ "question": "cậu bé áo đỏ thổi bao nhiêu cây nến, sau đó ai tặng quà?", "video_id": null }
// → { "answer": "5 cây nến; người phụ nữ áo vàng tặng quà", "evidence": [ {keyframe_id, start_time, end_time} ] }
```

## 4. Quy ước làm việc chung

- Repo chung (GitHub private): thư mục `indexing/` (N1), `backend/` (N2), `agent/` (N3), `ui/` (N4), `eval/` (N5). Không ai commit thẳng vào `main` — dùng branch + PR, người khác review.
- Dữ liệu lớn (index, embedding) KHÔNG đưa vào git — để trong thư mục `data/` (gitignore), N1 cung cấp link tải + script tải.
- Mỗi thay đổi ảnh hưởng chất lượng tìm kiếm PHẢI chạy qua benchmark của N5 trước khi merge. Không có số → không merge.
- Kênh chat đội: mỗi người báo cáo 3 dòng mỗi ngày (hôm qua làm gì / hôm nay làm gì / đang kẹt gì).

## 5. Mốc chung (điều chỉnh theo lịch BTC phát dữ liệu)

| Mốc | Nội dung | Ai |
|---|---|---|
| M1 | KIS text chạy end-to-end trên CLIPFeatures sẵn + UI lưới ảnh | N1, N2, N4 |
| M2 | Query-by-image (Video KIS) + BM25 (OCR/ASR/metadata) + RRF + benchmark chạy | N1, N2, N5 |
| M3 | LLM query understanding + VLM rerank/verify + timeline UI | N3, N4 |
| M4 | KISC agent + VQA + feedback loop + tập dượt bấm giờ | N3, N4, N5 |
| M5 | Đóng băng tính năng, tối ưu tốc độ, báo cáo kỹ thuật | Cả đội |
