# N3 — Onboarding: Agent & Suy luận (AI Engineer)

> Đọc `00-tong-quan-va-hop-dong-api.md` trước. Mảng của bạn là mảng NẶNG NHẤT và là thứ BTC chấm là điểm nhấn 2026 — KISC và khả năng "lý luận" thay vì chỉ "so khớp".

## Sứ mệnh của bạn

Xây tầng **Reasoning** trong khung Perception–Reasoning–Action mà BTC dùng: hiểu truy vấn tiếng Việt mơ hồ → dịch thành ràng buộc có cấu trúc → điều phối tìm kiếm → xác minh kết quả từng điều kiện → hỏi lại người dùng khi cần → trả lời câu hỏi VQA đa bước.

Bạn làm việc được ĐỘC LẬP từ ngày một: N2 cung cấp mock API tuần đầu, bạn dev toàn bộ logic agent trên đó.

## Phần 1 — LLM Query Understanding (mốc M3, bắt đầu ngay)

### 1.1 Vì sao đây là module ăn điểm nhất
CLIP/SigLIP hiểu tiếng Anh tốt hơn nhiều tiếng Việt. Bước "dịch + viết lại thành mô tả thị giác tiếng Anh" thường tăng điểm nhiều hơn mọi tinh chỉnh model khác. BTC gọi đây là Generative Query Expansion.

### 1.2 Spec: hàm `understand(query_vi, history) -> StructuredQuery`
LLM (Claude/GPT API; dự phòng Qwen local) trả JSON:

```jsonc
{
  "search_text_en": "a person dropping keys near a ripe fruit stall",
  "variants": [        // 2-3 cách diễn đạt khác — nhớ Case Study 1: khái niệm hiếm ("lead soldier") phải đổi thành mô tả thuộc tính ("man in red coat, tall hat")
    "hand dropping a keychain on the ground at a market",
    "pink teddy bear keychain falling, fruit stand in background"
  ],
  "positives": ["keys with pink teddy bear keychain", "fruit stall", "dropping action"],
  "negatives": [],       // "ăn hamburger Ở NHÀ không tính" → ["at home", "home kitchen"]
  "temporal": [],        // [{"before": "takes off hat", "after": "enters room"}]
  "cooccur": true,       // các positives phải xuất hiện ĐỒNG THỜI (KIS nghiêm ngặt: kem + biển cùng khung hình)
  "filters": {"objects": null, "publish_date_from": null},
  "task_type": "kis"     // kis | avs | vqa
}
```

Prompt engineering ở đây quyết định tất cả — viết prompt có ví dụ few-shot lấy thẳng từ các ví dụ BTC (đánh rơi chìa khóa, ăn kem bờ biển, fast food xa nhà). Test bộ 20 truy vấn mẫu, xem JSON ra có đúng ý không.

## Phần 2 — VLM Rerank & Verify (mốc M3)

### 2.1 Tư tưởng: checklist, không phải điểm khớp chung
KIS chấm nghiêm ngặt: "ăn kem bên bờ biển" = phải thấy CẢ kem CẢ biển. CLIP mù thuộc tính chi tiết (Case Study 3: điểm đúng/sai chênh 0.02). Giải pháp: với top ~50 segment từ N2 (`return_raw=true` lấy frame lẻ):

1. Lấy 1–3 frame đại diện mỗi segment (qua `/keyframe/{id}/image`).
2. Gửi VLM (Qwen2.5-VL local hoặc GPT-4o/Claude API) prompt dạng:
   > Đây là các điều kiện: [1] người cầm kem, [2] biển ở phía sau, [3] KHÔNG ở trong nhà. Với ảnh này, trả JSON {"c1": true/false, "c2": ..., "c3": ..., "explain": "..."}
3. Điểm rerank = số điều kiện đạt (điều kiện âm vi phạm → loại thẳng). `explain` điền vào trường `reason` của response.
4. Ràng buộc thứ tự thời gian (`temporal`): lấy dải frame theo thứ tự từ `/timeline`, hỏi VLM "hành động A có xảy ra TRƯỚC hành động B không?".

### 2.2 Ngân sách suy luận
Big Three #2: model đắt chỉ chạy trên top nhỏ. Định mức: rerank ≤ 50 ứng viên, ≤ 3 frame/ứng viên, chạy song song (asyncio + batch). Đo thời gian: mục tiêu < 10 giây cho một lượt rerank đầy đủ. Có nút "rerank sâu hơn" thay vì luôn chạy max.

## Phần 3 — KISC Clarification Agent (mốc M4) — ĐIỂM NHẤN 2026

Kịch bản chuẩn BTC (học thuộc): user hỏi mơ hồ "gặp bạn cũ tuần trước" → agent hỏi "trong nhà hay ngoài trời? nam hay nữ?" → user bổ sung "cà phê ngoài trời, sơ mi xanh" → agent lọc metadata + không gian + vật thể/màu → kết quả.

### 3.1 Vòng lặp agent (implement trong `/agent/query`)
```
1. understand(message, history) → StructuredQuery
2. Gọi /search (hoặc /search/feedback nếu có positive từ lượt trước,
   hoặc giới hạn filters.video_ids = kết quả lượt trước nếu user đang THU HẸP)
3. Đánh giá độ mơ hồ của kết quả:
   - Top kết quả phân tán nhiều video/bối cảnh khác nhau? → action="clarify"
   - Kết quả tập trung? → rerank + action="results"
4. Nếu clarify: chọn CÂU HỎI PHÂN TÁCH MẠNH NHẤT
```

### 3.2 Chọn câu hỏi thông minh (đây là chỗ ăn điểm demo)
Heuristic đủ dùng: nhìn top-50 kết quả hiện tại, tìm thuộc tính mà tập kết quả chia ~50/50 (trong nhà/ngoài trời, ngày/đêm, có người/không) — hỏi thuộc tính đó thì câu trả lời nào cũng loại được nửa không gian tìm kiếm. Cách làm rẻ: đưa 20 caption/OCR của top kết quả cho LLM, yêu cầu "đề xuất 1 câu hỏi yes/no phân tách tập này đều nhất".
- Mỗi lượt user trả lời → dịch thành filter/điều kiện mới, KHÔNG tìm lại từ đầu — thu hẹp trên tập cũ (đúng tinh thần "thu hẹp dần theo độ chắc chắn" của BTC).
- Giữ `session_id` → history trong RAM (dict), không cần DB.

## Phần 4 — VQA đa bước (mốc M4)

Ví dụ chuẩn BTC: "cậu bé áo siêu nhân đỏ thổi bao nhiêu cây nến, sau đó ai tặng quà?" — chain kỳ vọng: định vị nhân vật → định vị cảnh thổi nến → ĐẾM → theo dõi hành động SAU ĐÓ → nhận diện người.

Pipeline `/agent/vqa`:
1. `understand()` tách câu hỏi thành: mô tả định vị ("boy in red superman shirt blowing candles") + các câu hỏi con (đếm nến; ai tặng quà sau đó).
2. Retrieval định vị segment (qua /search).
3. Lấy **cửa sổ frame theo thứ tự thời gian, kéo dài VỀ PHÍA SAU** segment (câu "sau đó ai..." cần 30–60s sau sự kiện) qua /timeline — 8–16 frame đều nhau.
4. Gửi VLM toàn bộ dải frame (đánh số thứ tự trong prompt) + câu hỏi → trả lời text ngắn + evidence (keyframe nào chứng minh).
5. Câu hỏi đếm/tần suất kiểu "xe xuất hiện bao nhiêu lần": ưu tiên đếm bằng object tags trong metadata DB (SQL) thay vì bắt VLM đếm — VLM đếm rất tệ.

## Phần 5 — Cạm bẫy đã biết trước

- LLM trả JSON hỏng → luôn parse trong try/except + retry 1 lần với reminder "JSON only". Dùng structured output nếu API hỗ trợ.
- Đừng để agent hỏi lại QUÁ nhiều — mỗi câu hỏi tốn 1 lượt của user; chỉ clarify khi kết quả thực sự phân tán. Đo bằng benchmark KISC của N5 (số lượt đến khi tìm ra).
- API ngoài có thể chết giữa buổi thi → mọi lời gọi LLM/VLM có fallback: local Qwen hoặc degrade về pipeline không rerank. Phối hợp N5 chuẩn bị.
- Ghi log toàn bộ prompt/response (jsonl) để tune prompt sau tập dượt.

## Công cụ & tài liệu cần học

- API Claude/OpenAI: structured outputs, vision input, batch/parallel calls.
- Qwen2.5-VL chạy local (vLLM hoặc transformers) làm phương án dự phòng.
- Đọc lại 3 case study + kịch bản KISC 4 lượt trong file 00 — đó là đề thi mẫu của bạn.

## Definition of Done của bạn

- [ ] `understand()` với bộ test 20 truy vấn mẫu (gồm đủ: ràng buộc âm, đồng thời, thứ tự) — JSON đúng ≥ 18/20 (M3).
- [ ] VLM rerank checklist, < 10s/lượt, điền `reason` cho top kết quả (M3).
- [ ] `/agent/query` KISC đầy đủ vòng lặp clarify/thu hẹp, chạy được kịch bản 4 lượt của BTC (M4).
- [ ] `/agent/vqa` trả lời đúng bộ câu hỏi VQA nội bộ của N5 ≥ mức thống nhất (M4).
- [ ] Fallback local khi API chết; log prompt đầy đủ.

## Giao tiếp với người khác

- **N2**: bạn là khách hàng lớn nhất của /search — cần `return_raw`, `/timeline`, ảnh keyframe; báo sớm nếu thiếu gì.
- **N4**: thống nhất cách hiển thị câu clarify + structured_query (hiện để user sửa tay được là điểm cộng demo).
- **N5**: cung cấp cho N5 bảng chi phí API/lượt để tính budget; nhận từ N5 kết quả benchmark để tune prompt.
