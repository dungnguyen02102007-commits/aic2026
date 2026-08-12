# 10 — Luật vòng sơ tuyển và cách chấm điểm

> Nguồn: `Thong tin vong So tuyen AIC2026.pdf` (BTC phát 12/08/2026).
> Tài liệu này **ghi đè** phần mô tả 4 dạng truy vấn ở tài liệu 00 — vòng sơ tuyển chỉ có 3 dạng.

---

## 1. Ba dạng truy vấn (KHÔNG có AVS, KHÔNG có Conversational KIS)

| Dạng | Đề bài | Nộp gì |
|---|---|---|
| **Textual KIS** | Mô tả sự kiện bằng văn bản, cung cấp trọn vẹn | `<video_id>, <frame_id>` |
| **Q&A** | Mô tả sự kiện + một câu hỏi về nội dung | `<video_id>, <frame_id>, <answer>` |
| **TRAKE** | Chuỗi sự kiện có cấu trúc, gồm N khoảnh khắc | `<video_id>, <frame_id₁>, ..., <frame_idₙ>` |

Đáp án Q&A **tiếng Việt hay tiếng Anh đều được**.

### Việc bị hủy khỏi kế hoạch

- Toàn bộ nhánh **AVS** (`mode="avs"` trong backend, khử trùng lặp trong video).
- Toàn bộ nhánh **Conversational KIS / KISC** (mốc M4 cũ của N3).

Chưa xóa code, chỉ ngừng đầu tư. Nếu vòng chung kết có lại thì bật lên.

---

## 2. ⚠️ Nộp bài bằng `frame_id`, KHÔNG phải giây

Đáp án đúng là khoảng `[s, e]` **theo chỉ số khung hình của video gốc**.

```
map-keyframes/<video_id>.csv  →  cột frame_idx  ←  ĐÂY là con số phải nộp
                                 cột pts_time   ←  giây, chỉ dùng để hiển thị
```

`indexing/build_metadata_db.py` đã lưu `frame_index` sẵn trong SQLite. Backend phải trả nó ra trong `SearchResult` — xem issue #20.

---

## 3. Công thức tính điểm

Mỗi truy vấn nộp **tối đa 100 câu trả lời**, xếp theo độ tin cậy giảm dần.

```
R-Score(rᵢ)  ∈ [0, 1]   độ chính xác của MỘT câu trả lời
R@k          = max R-Score trong k câu ĐẦU TIÊN,  k ∈ {1, 5, 20, 50, 100}
Final Score  = trung bình cộng của 5 giá trị R@k
```

### R-Score từng dạng

| Dạng | Công thức |
|---|---|
| KIS | `1` nếu đúng video **và** `frame_id ∈ [s,e]`, ngược lại `0` |
| Q&A | `1` nếu đúng video **và** `frame_id ∈ [s,e]` **và** answer khớp ngữ nghĩa |
| TRAKE | Sai video → `0` ngay. Đúng video → `(số khoảnh khắc khớp) / N` |

TRAKE là dạng duy nhất có **điểm thành phần**: khớp 3/4 khoảnh khắc vẫn được 0.75.

### Bảng quy đổi — hạng của câu đúng đầu tiên

| Hạng | 1 | 2–5 | 6–20 | 21–50 | 51–100 | >100 |
|---|---|---|---|---|---|---|
| **Điểm** | **1.00** | 0.80 | 0.60 | 0.40 | 0.20 | 0 |

---

## 4. Hai luật rút ra từ công thức — cả đội phải thuộc

**Luật 1 — LUÔN nộp đủ 100 câu, không bao giờ ít hơn.**
Câu ở hạng 100 vẫn mang về 0.2 điểm. Nộp 20 câu rồi dừng là tự vứt điểm miễn phí. Không có hình phạt nào cho câu sai.

> 📄 Chi tiết cách làm cho từng dạng đề, kèm ví dụ thật từ đội mình cho thấy chênh **gần gấp đôi** điểm (0.36 → 0.68): [`11-luon-nop-du-100-dap-an.md`](11-luon-nop-du-100-dap-an.md)

**Luật 2 — Rerank đáng giá ngang retrieval.**
Đẩy một đáp án từ hạng 6 lên hạng 1 được thêm 0.4 điểm — bằng đúng việc tìm mới được cả một câu đúng ở hạng 21. Nên khi hệ thống "đã tìm ra nhưng xếp thấp", đó là chỗ ăn điểm rẻ nhất.

`eval/run_benchmark.py` in riêng danh sách này sau mỗi lần chạy:
```
ĐÃ tìm ra nhưng xếp hạng > 5 — đẩy lên top-1 là điểm tăng ngay:
  kis_07: hạng 12
```

---

## 5. ⚠️ TRAKE đòi độ chính xác mà keyframe BTC không đáp ứng nổi

PDF ghi rõ: đoạn `[sⱼ, eⱼ]` của mỗi khoảnh khắc ngữ nghĩa **thường dưới 10 frame**.

Keyframe BTC phát là **I-frame** — khung hình kỹ thuật của thuật toán nén, cách nhau hàng trăm frame. PDF cũng nói thẳng chúng khác với "khung hình ngữ nghĩa". Nên xác suất một I-frame rơi trúng cửa sổ 10 frame là rất thấp.

May là BTC cho phép:

> *"Dữ liệu thi chính thức là Video; các thành phần còn lại (Keyframes, Objects, CLIP features, Metadata) chỉ nhằm mục đích cung cấp thêm thông tin hoặc hỗ trợ xây dựng giải pháp mẫu."*

→ TRAKE bắt buộc có tầng **trích frame dày từ video gốc** quanh vùng ứng viên. Xem issue #19.

---

## 6. Dữ liệu đợt 1 chính là batch 1 của AIC 2025

> *"Đây cũng là dữ liệu batch 1 của AIC 2025."*

Nghĩa là **đề thi và đáp án AIC 2025 dùng được làm bộ benchmark thật** — thay vì 50 câu tự nghĩ. Đo trên đề thật mới biết hệ thống mạnh yếu ở đâu. Xem issue #10.

Batch 2 sẽ phát thêm, chưa có lịch.

---

## 7. Cách chạy benchmark

```bash
# Gọi API thật, tự dựng bài nộp (kis + qa)
python eval/run_benchmark.py --api http://127.0.0.1:8000 --queries eval/queries.jsonl --tag clip_only

# Chấm file bài nộp có sẵn (đủ 3 dạng, kể cả trake)
python eval/run_benchmark.py --submissions eval/sub.jsonl --queries eval/queries.jsonl --tag trake_v1

# Kiểm chứng công thức chấm bằng ví dụ trong PDF BTC
python eval/test_scoring.py
```

`test_scoring.py` phải luôn ra **26/26**. Fail nghĩa là công thức chấm sai — mọi số đo trước đó đều vô nghĩa.

Định dạng câu hỏi và bài nộp: xem `eval/queries.template.jsonl`.
