# Ghi chú VBS-Archive & bài học nhà vô địch (N5 — issue #11)

Nguồn: github.com/lucaro/VBS-Archive (kho đề + kết quả VBS 2012–2026) và paper các đội thắng.
Mục đích: chuẩn văn phong đề + format tập dượt + hướng nâng cấp đã được kiểm chứng.

## 1. Format thi VBS (chuẩn tham chiếu cho tập dượt)

| Dạng | Thời gian | Ghi chú |
|---|---|---|
| KIS-Visual (chiếu clip) | ~5 phút | Giải nhanh nhất bằng chụp màn hình → tìm ảnh |
| KIS-Textual | ~7 phút | Mô tả text, **hint lộ dần theo thời gian** — càng chờ càng rõ nhưng càng ít điểm |
| AVS | ~5 phút | Nộp càng nhiều segment đúng càng tốt, giám khảo chấm live |
| QA | ~5 phút | Trả lời text |

→ Tập dượt của đội bấm giờ đúng các mốc này. Điểm VBS giảm dần theo thời gian nộp
và trừ điểm nộp sai → chiến thuật: **chắc mới nộp, nhưng đừng cầu toàn quá 50% thời gian**.

## 2. Văn phong đề thật (mẫu để viết 50 câu benchmark)

- KIS-T: 2–4 "mỏ neo" cụ thể, viết như người nhớ lại (không phải liệt kê camera):
  bối cảnh + đối tượng có thuộc tính + hành động. Hint bổ sung xuất hiện sau
  (thêm chi tiết màu sắc / vị trí / âm thanh).
- AVS: mô tả ngữ nghĩa tổng quát, đôi khi kèm điều kiện loại trừ.
- Thư mục `avsjudgements/` (2021–2024) có phán quyết đúng/sai chuẩn TRECVID —
  đọc để hiểu ranh giới "đúng" của giám khảo (vd: thấy một phần vật thể có tính không).

## 3. Nhà vô địch các năm & bài học chính

| Năm | Đội | Bài học rút ra cho mình |
|---|---|---|
| 2025 | NII-UIT (có UIT VN) | LLM query expansion; **Stable Diffusion sinh ảnh từ text rồi tìm bằng ảnh**; **dynamic temporal search 2 chiều** (chấm frame trước VÀ sau kết quả) |
| 2024 | VISIONE (mã nguồn mở: github.com/aimh-lab/visione) | **Fusion 3 model** (OpenCLIP ViT-L + CLIP2Video + ALADIN); object+màu+vị trí không gian; temporal "first this then that" |
| 2022–23 | vibro | 3 yếu tố thắng: **superuser thao tác nhanh** + browsing mạnh + image-similarity tổng quát; **chọn model bằng cách replay đề cũ** |
| 2021 | vitrivr | Hệ đa phương thức lâu đời, ổn định |
| 2020 | SOMHunter | Self-organizing map browsing + relevance feedback |

Điểm chung mọi đội top: (1) không tin một embedding duy nhất — fusion;
(2) image-similarity là công cụ hạng nhất; (3) 2/3 chiến thắng đến từ con người + UI,
không phải model → luyện tập dượt quan trọng ngang code.

## 4. Việc áp dụng cụ thể

- [ ] N5: viết 50 câu benchmark theo văn phong mục 2; tập dượt theo giờ mục 1.
- [ ] N1: khi phân vân model, replay bộ benchmark trên từng model (cách của vibro).
- [ ] N2: cân nhắc endpoint temporal search 2 chiều (kiểu NII-UIT) ở M3.
- [ ] N3: đọc paper NII-UIT 2025 (Springer, MMM 2025) — mẫu LLM integration gần nhất;
      thử Stable Diffusion sinh ảnh làm biến thể truy vấn khi CLIP score thấp.
- [ ] N4: tham khảo UI VISIONE (có bản "for newbies") + vibro để lấy pattern browsing.
