# N5 — Onboarding: Đánh giá, Hạ tầng & Báo cáo (Team Lead / Ops)

> Đọc `00-tong-quan-va-hop-dong-api.md` trước. Vai này phù hợp team lead kiêm nhiệm: ít code sâu nhất nhưng nhìn thấy toàn cục nhiều nhất, và là người quyết định "có merge hay không".

## Sứ mệnh của bạn

Ba việc: (1) làm cho MỌI cải tiến đo được bằng số — không có số thì mọi "cải tiến" chỉ là cảm giác; (2) đảm bảo hệ thống sống sót ngày thi; (3) biến công sức cả đội thành báo cáo kỹ thuật + màn demo ăn điểm.

## Phần 1 — Benchmark nội bộ (mốc M1–M2, ưu tiên số 1)

### 1.1 Bộ truy vấn test (`eval/queries.jsonl`)
Tự xây ~50 truy vấn có đáp án trên chính dữ liệu BTC phát. Cách làm: mở video ngẫu nhiên, chọn một khoảnh khắc, tự viết mô tả như đề thi. PHẢI phủ đủ độ khó (bám các ví dụ chính thức của BTC):

| Nhóm | Số lượng | Ví dụ mẫu theo phong cách BTC |
|---|---|---|
| KIS dễ (cảnh đặc trưng) | 10 | "ăn kem bên bờ biển" |
| KIS khó (vật nhỏ + hành động 1-2s) | 10 | "đánh rơi chìa khóa móc gấu bông hồng ở quầy hoa quả" |
| KIS có điều kiện ĐỒNG THỜI | 5 | cả kem CẢ biển cùng khung hình |
| AVS (nhiều đáp án + điều kiện loại trừ) | 10 | "ăn fast food ở nhà hàng, KHÔNG tính ở nhà" |
| VQA (đếm, thứ tự, "sau đó ai") | 10 | "thổi bao nhiêu nến, sau đó ai tặng quà" |
| KISC kịch bản nhiều lượt | 5 | mơ hồ dần rõ, kèm sẵn câu trả lời clarify |

Format mỗi dòng:
```jsonc
{"qid": "kis_07", "type": "kis", "query_vi": "...",
 "answer": {"video_id": "L01_V001", "start": 751.0, "end": 766.0},
 "notes": "vật nhỏ, hành động nhanh"}
```
Kéo cả đội cùng đóng góp truy vấn (mỗi người 10 câu) — vừa nhanh vừa giúp mọi người hiểu dữ liệu.

### 1.2 Script chấm (`eval/run_benchmark.py`)
- Với KIS: gọi /search (và pipeline agent đầy đủ), tính **recall@1/@5/@20/@100** (đúng nếu segment trả về giao với khoảng đáp án) + latency p50/p95.
- Với AVS: precision@10/@50 trên các đáp án đã đánh dấu (chấp nhận không đầy đủ).
- Với VQA: so khớp đáp án bằng tay hoặc LLM-judge đơn giản.
- Với KISC: **số lượt hội thoại đến khi tìm ra** (metric quan trọng nhất cho điểm nhấn 2026).
- Xuất bảng markdown so sánh giữa các cấu hình: `clip_only` vs `+siglip` vs `+bm25_rrf` vs `+rerank` — mỗi PR ảnh hưởng chất lượng phải đính kèm bảng này. **Luật của đội: không có số → không merge.**

## Phần 2 — Hạ tầng & vận hành (mốc M2–M4)

### 2.1 Ngân sách tài nguyên
- Lập bảng GPU khả dụng (Colab/Kaggle/máy trường/máy cá nhân) và phân bổ cho N1 (indexing cần nhiều nhất, một đợt) + N3 (VLM, liên tục).
- Theo dõi chi phí API LLM/VLM của N3: đặt trần $/tuần, có dashboard đơn giản (log jsonl của N3 → script tổng hợp).

### 2.2 Máy thi & kịch bản sự cố
- Dựng môi trường thi: script `run_all.sh` bật backend + UI bằng 1 lệnh; index load từ đĩa; KHÔNG phụ thuộc mạng cho phần lõi (chỉ agent gọi API ngoài).
- Diễn tập sự cố: API LLM chết → N3 degrade về không-rerank; máy chính chết → máy dự phòng có sẵn toàn bộ `data/` (bạn giữ bản sao index trên ổ cứng rời).
- Tìm hiểu **hệ thống nộp bài chính thức của BTC** (thường kiểu DRES: nộp video_id + timestamp qua API/web) ngay khi BTC công bố — báo format cho N4 làm màn nộp bài, và viết sẵn client nộp nếu có API.

### 2.3 Tập dượt (mốc M4–M5, ít nhất 3 buổi)
- Mô phỏng đúng thể thức: chiếu truy vấn, bấm giờ (KIS thường 5–7 phút/câu), dùng bộ test giữ kín một phần mà cả đội chưa thấy.
- Phân vai thi đấu và thử nhiều đội hình: ai gõ truy vấn, ai duyệt lưới (2 tab song song — BTC gợi ý crowdsourcing), ai thao tác nộp.
- Ngồi sau quan sát, ghi từng chỗ lúng túng: mỗi lần lúng túng = 1 issue cho N3/N4.
- Sau mỗi buổi: họp 30 phút rút kinh nghiệm, cập nhật chiến thuật "thu hẹp dần theo độ chắc chắn" (thời gian → địa điểm → đối tượng → cảnh vật).

## Phần 3 — Báo cáo kỹ thuật & demo (mốc M5)

- Viết theo khung **Perception – Reasoning – Action** (đúng ngôn ngữ BTC dạy trong tập huấn): Perception = index đa kênh (SigLIP/OCR/ASR/objects); Reasoning = LLM understanding + VLM verify + clarification; Action = search/filter/feedback/submit.
- Nhấn các điểm khớp trực tiếp thông điệp BTC: giải Big Three thế nào (funnel cho sparsity, checklist verify cho semantic gap + điều kiện đồng thời, timeline reasoning cho temporal logic), KISC clarification (điểm nhấn 2026), và trả lời "dấu hỏi 2026" trong slide timeline kỹ thuật của họ.
- **Mục đạo đức & quyền riêng tư** (slide đa ngành của BTC nhắc từ tutorial Gurrin): dữ liệu lifelog là dữ liệu cá nhân nhạy cảm — nêu biện pháp (xử lý cục bộ phần lõi, cân nhắc khi gửi frame lên API ngoài, không lưu dữ liệu ngoài phạm vi thi). Điểm cộng rẻ, rất ít đội làm.
- Kịch bản demo 5 phút: 1 Video KIS (dán ảnh, 5 giây ra đáp án — mở màn ấn tượng), 1 KIS khó qua KISC 3 lượt, 1 VQA đa bước. Tập demo như tập văn nghệ.

## Phần 4 — Cạm bẫy đã biết trước

- Benchmark làm muộn = cả đội bay mù 1 tháng. Đây là việc ưu tiên 1 của bạn, trước cả hạ tầng.
- Đáp án bộ test phải là KHOẢNG thời gian (không phải 1 điểm) — người chấm giao-nhau (overlap) mới công bằng với biên segment.
- Đừng tune trên toàn bộ 50 câu — giữ 15 câu làm tập kín (held-out), chỉ chạy trước tập dượt, tránh tự lừa mình overfit.
- Lịch BTC (vòng loại/chung kết, ngày phát dữ liệu, thể thức nộp) là input số 1 của kế hoạch — bám website/nhóm thông báo chính thức hằng tuần, cập nhật mốc M1–M5 tương ứng.

## Công cụ & tài liệu cần học

- Python scripting (jsonl, pandas, tính metric), shell script cơ bản.
- Đọc quy chế AI Challenge HCMC + tìm hiểu DRES (Distributed Retrieval Evaluation Server — hệ nộp bài chuẩn của VBS/LSC) nếu BTC dùng.
- Cách các đội VBS/LSC tổ chức thi đấu (search "VBS team strategy paper" — các đội top như vibro, SOMHunter, VISIONE đều có paper mô tả chiến thuật).

## Definition of Done của bạn

- [ ] 50 truy vấn test (35 mở + 15 kín) đủ 6 nhóm độ khó (M2).
- [ ] `run_benchmark.py` + bảng so sánh cấu hình đầu tiên công bố cho đội (M2).
- [ ] Bảng phân bổ GPU/API budget; máy thi chạy `run_all.sh` 1 lệnh (M3).
- [ ] Nắm rõ thể thức nộp bài BTC, client nộp sẵn sàng (M3–M4).
- [ ] ≥ 3 buổi tập dượt có biên bản rút kinh nghiệm (M4–M5).
- [ ] Báo cáo kỹ thuật hoàn chỉnh + kịch bản demo đã tập (M5).

## Giao tiếp với người khác

- Với cả đội: bạn giữ nhịp — điều phối standup 3 dòng/ngày, giữ luật "không có số không merge", cập nhật mốc theo lịch BTC.
- **N1**: nhận version index để gắn vào từng dòng kết quả benchmark.
- **N2**: dùng `/health`, `/stats` kiểm môi trường trước mỗi lần đo.
- **N3**: đưa kết quả benchmark chi tiết từng câu (câu nào trượt, trượt ở tầng nào) — đó là dữ liệu tune prompt của N3.
- **N4**: biên bản quan sát tập dượt là backlog UX của N4.
