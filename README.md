# AIC 2026 — Hệ thống truy hồi video đa phương thức

Hệ thống dự thi AI Challenge HCMC 2026 (bảng A — Sinh viên): KIS / AVS / VQA / Conversational KIS trên kho video lớn.

## Cấu trúc repo

| Thư mục | Mảng | Phụ trách |
|---|---|---|
| `indexing/` | N1 — Dữ liệu & chỉ mục (FAISS, SigLIP, OCR, ASR, BM25) | Tuấn Anh |
| `backend/` | N2 — Lõi truy hồi (FastAPI, RRF, temporal grouping) | Tiến Khang |
| `agent/` | N3 — Agent & suy luận (LLM understanding, VLM rerank, KISC, VQA) | Gia Thoại |
| `ui/` | N4 — Giao diện (lưới keyframe, timeline, chat KISC) | Đình Trung |
| `eval/` | N5 — Benchmark & vận hành | Quang Dũng |
| `docs/` | Tài liệu onboarding + hợp đồng API | Cả đội |
| `data/` | Dữ liệu & index — **KHÔNG commit** (gitignore) | — |

## Bắt đầu

1. Đọc `docs/onboarding/00-tong-quan-va-hop-dong-api.md` (bắt buộc, tất cả thành viên).
2. Đọc tiếp file onboarding của mảng mình (`01`→`05`).
3. Lộ trình tháng đầu: `docs/onboarding/06-lo-trinh-1-thang.md`.

## Quy ước làm việc

- Không commit thẳng vào `main` — tạo branch `ten-minh/mo-ta-ngan`, mở Pull Request, cần 1 người review.
- Thay đổi ảnh hưởng chất lượng tìm kiếm phải kèm số benchmark (`eval/`) — **không có số, không merge**.
- Dữ liệu lớn (video, keyframe, embedding, index) để trong `data/`, chia sẻ qua link — không đưa vào git.
- Standup 3 dòng mỗi tối trên nhóm chat; demo nội bộ tối Chủ nhật hằng tuần.

## Hợp đồng API

Xem mục 3 của `docs/onboarding/00-tong-quan-va-hop-dong-api.md`. Chủ sở hữu: Tiến Khang (N2) — đổi schema phải thông báo cả đội.
