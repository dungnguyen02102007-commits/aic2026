# eval/ — Benchmark & đánh giá (N5 - Quang Dũng)

Luật của đội: **thay đổi ảnh hưởng chất lượng tìm kiếm phải kèm số từ thư mục này — không có số, không merge.**

## Files

- `queries.jsonl` — bộ truy vấn test CÔNG KHAI (cả đội thấy, được tune trên đây).
- `queries_held_out.jsonl` — 15 câu GIỮ KÍN (chỉ Dũng giữ, KHÔNG commit — chỉ chạy trước tập dượt).
- `queries.template.jsonl` — mẫu schema + ví dụ để mọi người viết truy vấn đóng góp.
- `run_benchmark.py` — script chấm: recall@K + latency, xuất bảng markdown.
- `vbs-archive-notes.md` — ghi chú văn phong đề từ VBS-Archive (issue #11).

## Cách viết một truy vấn test (mỗi người 10 câu — deadline T4 tuần 2)

1. Mở một video ngẫu nhiên trong dataset, chọn một khoảnh khắc cụ thể.
2. Viết mô tả theo đúng văn phong đề BTC (xem ví dụ trong template — có đủ 6 nhóm độ khó).
3. Ghi đáp án là **KHOẢNG thời gian** `[start, end]` (giây), không phải 1 điểm.
4. Thêm dòng vào `queries.jsonl` qua PR, hoặc gửi Dũng gom.

Phân bố cần đạt (tổng 50): 10 KIS dễ · 10 KIS khó (vật nhỏ/hành động 1-2s) · 5 KIS điều kiện đồng thời · 10 AVS (có câu chứa điều kiện loại trừ) · 10 VQA (đếm/thứ tự/"sau đó") · 5 KISC nhiều lượt.

## Chạy benchmark

```bash
# backend phải đang chạy (mock hoặc thật)
python eval/run_benchmark.py --api http://localhost:8000 --queries eval/queries.jsonl --tag clip_only
# so sánh 2 cấu hình:
python eval/run_benchmark.py --api http://localhost:8000 --queries eval/queries.jsonl --tag with_rrf
```

Mỗi lần chạy ghi thêm 1 dòng vào `eval/results/history.csv` và in bảng markdown để dán vào PR.
