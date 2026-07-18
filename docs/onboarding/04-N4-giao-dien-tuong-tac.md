# N4 — Onboarding: Giao diện & Tương tác (Frontend Engineer)

> Đọc `00-tong-quan-va-hop-dong-api.md` trước. Bài học lớn nhất từ tập huấn BTC: ở cuộc thi interactive search, **tốc độ thao tác của UI quan trọng ngang chất lượng model** — người thi phải duyệt hàng trăm keyframe trong vài giây.

## Sứ mệnh của bạn

Xây giao diện để 1–2 người thi thao tác NHANH NHẤT có thể: gõ truy vấn, quét lưới ảnh, nhảy timeline, tìm bằng ảnh, hội thoại với agent, và nộp bài — tất cả tối ưu bằng phím tắt. Người dùng của bạn chính là đồng đội trong phòng thi, deadline tính bằng giây.

Stack đề xuất: **Next.js/React + Tailwind** (hoặc Vite + React). Nếu cả đội thiên backend, Gradio cũng chấp nhận được cho M1 nhưng sẽ đụng trần ở phím tắt/hiệu năng — quyết định sớm trong tuần đầu.

## Phần 1 — Màn hình tìm kiếm chính (mốc M1)

### 1.1 Layout
```
+--------------------------------------------------------------+
| [ô truy vấn................] [🔍] [📷 dán ảnh] [chế độ: KIS ▾]|
| filters: [thời gian] [kênh] [objects]                        |
+--------------------------------------------------------------+
|  LƯỚI KEYFRAME (6-8 cột, ảnh nhỏ, dày đặc)                    |
|  mỗi ô: thumbnail + video_id + timestamp + badge score        |
+--------------------------------------------------------------+
| thanh trạng thái: số kết quả, latency, index version          |
+--------------------------------------------------------------+
```

- Lưới ảnh DÀY ĐẶC là vua: 6–8 cột, 40–60 ảnh/màn hình, lazy-load + virtualization (react-window) để cuộn 1000 kết quả không giật.
- Ảnh lấy từ static server của N2 (`/keyframe/{id}/image?size=thumb`) — yêu cầu N2 hỗ trợ param size.
- Hover một ô → phóng to ảnh + hiện `reason` (nếu có).

### 1.2 Phím tắt (làm từ M1, không để sau)
- `/` focus ô tìm kiếm; `Esc` quay lại lưới.
- Mũi tên di chuyển selection trong lưới; `Enter` mở panel chi tiết; `Space` phóng to nhanh ảnh đang chọn.
- `v` mở video tại timestamp; `s` đánh dấu ✓ đúng hướng; `x` đánh dấu ✗ sai; `f` tìm-tương-tự từ ảnh đang chọn; `Ctrl+Enter` NỘP kết quả đang chọn.
- In danh sách phím tắt ngay trong UI (nhấn `?`).

## Phần 2 — Ba tính năng thắng thua (mốc M2–M3)

### 2.1 Tìm bằng ảnh — Video KIS (M2)
BTC sẽ TRÌNH CHIẾU một clip, đội phải tìm đúng đoạn đó. Flow phải đạt < 5 giây:
- Người thi chụp màn hình clip đang chiếu (Win+Shift+S) → `Ctrl+V` dán thẳng vào UI → tự động gọi `/search` với `query_image_b64` → lưới kết quả.
- Bắt sự kiện paste ở mức toàn trang (window paste listener), không bắt phải click vào ô nào trước.
- Vì frame query lấy từ chính dataset, kết quả top-1 gần như chắc chắn đúng — tốc độ dán-và-tìm là tất cả.

### 2.2 Timeline / Sequence view (M3)
Nhấn `Enter` trên một keyframe → panel chi tiết:
- **Dải phim ngang**: keyframe trước/sau trong cùng video (gọi `/video/{id}/timeline`), cuộn ngang bằng phím `←→`, click frame nào nhảy tới frame đó.
- **Video player** seek sẵn tới `start_time` (HTML5 video + `#t=`), phím `,` `.` tua ±2s.
- Nút/phím chọn biên `[start` và `end]` để nộp chính xác khoảng thời gian.
- Đây là tính năng giải Case Study 2 (xác minh sự kiện xảy ra TRƯỚC đó — người thi tua ngược timeline) — làm cho mượt.

### 2.3 Tìm tương tự + feedback (M3)
- Phím `f` trên bất kỳ ảnh nào → `/search` bằng ảnh đó (query-by-example, giải Case Study 1: tìm ảnh minh họa gần đúng rồi lần ra đáp án).
- Đánh dấu ✓/✗ tích lũy vào giỏ feedback → nút "Tìm lại theo feedback" gọi `/search/feedback` (average query expansion). Hiện giỏ ở sidebar để người thi thấy đang có gì.

## Phần 3 — Panel KISC + Nộp bài (mốc M4)

### 3.1 Chat KISC
- Panel chat bên phải (toggle bằng `c`): gọi `/agent/query`, hiển thị `reply` của agent; khi `action="clarify"` hiện câu hỏi nổi bật + các nút trả lời nhanh nếu đoán được (Trong nhà / Ngoài trời).
- Khi `action="results"` → đổ kết quả vào lưới chính (KHÔNG hiển thị kết quả trong khung chat — lưới luôn là nơi duyệt).
- Hiển thị `structured_query` (positives/negatives/temporal) dưới dạng chip có thể xóa/sửa — người thi chỉnh tay nhanh hơn gõ lại câu, và là điểm cộng lớn khi demo trước giám khảo.

### 3.2 Màn nộp bài
- Hỏi N5 format nộp chính thức của BTC (thường qua hệ thống kiểu DRES: video_id + thời điểm ms).
- Giỏ nộp: `Ctrl+Enter` thêm kết quả đang chọn; với AVS cho phép chọn nhiều + "nộp cả trang".
- Hiển thị to & rõ những gì sắp nộp (video_id, mm:ss) — nộp nhầm là mất lượt, double-check rẻ hơn hối hận.

## Phần 4 — Cạm bẫy đã biết trước

- Đừng làm đẹp trước khi làm nhanh. Không animation, không skeleton loading màu mè — mỗi 100ms là kẻ thù.
- Ảnh thumbnail phải có kích thước cố định trong CSS (tránh layout shift khi cuộn nhanh).
- Test với 1000 kết quả thật từ ngày đầu, đừng test với 20 mock rồi ngã ngửa.
- Mọi tính năng phải dùng được KHÔNG CHẠM CHUỘT. Người thi giữ tay trên bàn phím 100% thời gian.
- Trạng thái truy vấn nằm trên URL (query params) — F5 không mất phiên, mở 2 tab để 2 người cùng duyệt (crowdsourcing trong đội, BTC gợi ý hẳn trong Best Practice).

## Công cụ & tài liệu cần học

- React + react-window (virtualized grid), TanStack Query (cache + retry API).
- Clipboard API (paste ảnh), HTML5 video seeking.
- Đọc phần Best Practice + 3 Case Study trong file 00 — mỗi case study là một user flow bạn phải hỗ trợ.
- Tham khảo giao diện các hệ VBS nổi tiếng (vibro, SOMHunter — search "VBS video browser showdown interface" xem ảnh chụp màn hình để lấy cảm hứng layout).

## Definition of Done của bạn

- [ ] M1: lưới kết quả + ô tìm text + phím tắt cơ bản, chạy trên API thật của N2.
- [ ] M2: dán-ảnh-tìm-ngay < 5 giây end-to-end; filters hoạt động.
- [ ] M3: timeline/sequence view mượt; tìm-tương-tự; giỏ feedback.
- [ ] M4: chat KISC + chip structured query sửa được; màn nộp bài đúng format BTC.
- [ ] M5: cuộn 1000 kết quả 60fps; toàn bộ flow không cần chuột; 2 người dùng song song 2 tab không đụng nhau.

## Giao tiếp với người khác

- **N2**: bạn là người ép SLA — /search < 300ms, /timeline < 50ms, thumbnail có size param; đo và báo hằng tuần.
- **N3**: thống nhất UX câu clarify + chip structured_query; đề xuất nút trả lời nhanh.
- **N5**: N5 tổ chức tập dượt — ngồi cạnh xem đồng đội dùng UI, ghi lại mọi lần họ lúng túng; đó là backlog của bạn.
