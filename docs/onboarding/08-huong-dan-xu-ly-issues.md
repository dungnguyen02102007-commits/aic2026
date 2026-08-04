# 08 — Hướng dẫn xử lý các issue đang mở (cập nhật 04/08)

> Dành cho cả đội. Viết kiểu Bước 1-2-3, không cần biết lập trình sâu.
> **Bối cảnh mới:** BTC phát dataset ngày 03/08 (trễ 9 ngày so với kế hoạch).
> Mọi mốc trong file 06 lùi tương ứng: tuần "48 giờ dữ liệu thật" = TUẦN NÀY.

## Việc chung TRƯỚC TIÊN (ai cũng làm, 5 phút)

1. Mở GitHub → repo `aic2026` → tab **Issues** → bấm vào issue có tên mình
   → bên phải mục **Assignees** → gán tên mình. (Để Dũng biết ai đã nhận việc.)
2. Mở PowerShell:
   ```
   cd Documents\aic2026
   git pull
   ```
3. Tối nay nhớ nhắn 3 dòng standup vào nhóm.

## Việc của DŨNG trước khi mọi người bắt đầu (đầu mối review)

- [ ] Review + merge PR #13 (backend + agent + UI) — mọi người cần code này trên main.
- [ ] Review + merge PR #15 (ghi chú VBS + sửa benchmark).
- [ ] Review + merge PR hợp nhất inventory (mới, thay PR #14 đã đóng).
- [ ] Chia link tải dataset + tiến độ tải vào nhóm (ai tải phần nào để không trùng).

---

## TUẤN ANH (N1)

### Issue #1 — Kiểm kê dữ liệu ✅ gần xong, còn bước cuối
Script của bạn đã được ghép với phần đối chiếu chéo thành MỘT file (vẫn tên
`indexing/inventory.py`). Việc còn lại:

1. `git pull` để lấy bản hợp nhất.
2. Kiểm tra dữ liệu đang tải nằm ở thư mục `dataset/` trong `Documents\aic2026`
   (script tự nhận cả `dataset/` lẫn `data/`).
3. Đợi tải xong phần keyframes + clip features (chưa cần videos), chạy:
   ```
   python -m indexing.inventory
   ```
4. Mở file `dataset\inventory_report.md`, copy toàn bộ gửi vào nhóm.
   Cả đội đang chờ nhất dòng: **"Số chiều CLIP: ..."** — nó cho biết BTC dùng
   model gì (Khang cần con số này).
5. QUAN TRỌNG: **không bao giờ gõ `git add dataset`** — thư mục dữ liệu đã được
   chặn tự động, nhưng cẩn thận vẫn hơn. Dữ liệu chia sẻ qua Drive, không qua GitHub.

### Issue #2 — Xếp dữ liệu vào "tủ tra cứu" (SQLite)
1. Sau khi issue #1 xong: mở `indexing/build_metadata_db.py` bằng Notepad,
   nhìn phần CONFIG đầu file — sửa tên thư mục giống như đã làm ở inventory.
2. Chạy:
   ```
   python -m indexing.build_metadata_db
   python -m indexing.build_metadata_db --check
   ```
3. Dòng cuối in ra vài keyframe có object "Dog" → nghĩa là tủ tra cứu hoạt động.
   Chụp màn hình gửi nhóm, kéo issue sang Done.

### Issue #3 — Xây "mục lục tìm kiếm" FAISS (việc lớn nhất tuần này)
1. Upload thư mục clip features lên Drive chung `AIC2026/data/clip_features`
   (dùng Drive for Desktop cho ổn định).
2. Mở Colab → File → Open notebook → GitHub → repo mình →
   `indexing/notebooks/01_faiss_baseline_clipfeatures.ipynb`.
3. Sửa cell CONFIG cho khớp tên thư mục trên Drive → Run all.
4. Cell cuối in ra 10 kết quả thử — nếu chiều vector khớp và kết quả "có vẻ
   liên quan" là thành công. Báo nhóm: xong index v1.

---

## TIẾN KHANG (N2)

### Issue #4 — Làm chủ "tổng đài" (backend mock)
1. Sau khi Dũng merge PR #13: `git pull`.
2. Cài + chạy:
   ```
   python -m pip install -r backend/requirements.txt
   python -m uvicorn backend.main:app --port 8000
   ```
3. Mở `http://localhost:8000` (giao diện) và `http://localhost:8000/docs`
   (danh sách API, bấm Try it out nghịch từng cái).
4. Đọc `backend/schemas.py` — hợp đồng dữ liệu bạn sở hữu. Có gì muốn đổi,
   nêu trong nhóm TRƯỚC khi sửa.
5. Xong 4 bước trên → comment vào issue #4 "đã chạy được + đã đọc schemas" → Done.

### Issue #5 — Nối dữ liệu THẬT vào tổng đài (làm cùng Dũng + Tuấn Anh)
Chờ Tuấn Anh xong issue #3. Sau đó (Dũng sẽ ngồi cùng):
1. Nhận file `faiss_clip.index` + `keyframe_ids.npy` + `metadata.db` từ Tuấn Anh.
2. Nhiệm vụ code: viết `indexing/stores.py` có 3 hàm giống hệt tên trong
   `backend/stores_mock.py` (search_text, search_image cần model encode — Dũng
   hỗ trợ phần này) rồi đổi 1 dòng import trong `backend/main.py`.
3. Bật server, gõ thử một câu tiếng Anh — thấy keyframe THẬT là xong M1 thật sự.

---

## GIA THOẠI (N3)

### Issue #7 — Chìa khóa API (làm trước)
1. Vào `console.anthropic.com` (hoặc `platform.openai.com`), đăng ký, tạo API key.
2. Trong `Documents\aic2026` tạo file `.env` (Notepad → Save As → gõ tên
   `".env"` có ngoặc kép) gồm:
   ```
   AIC_LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=dán-key-vào-đây
   ```
3. KHÔNG gửi key cho ai, không chụp màn hình có key. Ghi vào nhóm: giá tiền
   ước tính mỗi 1 triệu token (xem trang pricing) để Dũng tính ngân sách. → Done.

### Issue #6 — Dạy "người phiên dịch" đạt 18/20
1. `git pull` (sau khi Dũng merge PR #13). Cài:
   ```
   python -m pip install anthropic python-dotenv
   ```
2. Chạy bài chấm: `python -m agent.test_understand`
3. Nhìn các câu bị ✗ → mở `agent/prompts.py` bằng Notepad → sửa/bổ sung phần
   hướng dẫn (viết tiếng Việt được) → chạy lại bài chấm.
4. Lặp đến khi ≥ 18/20. Dán kết quả cuối vào issue #6 → Done.
   Mẹo: câu sai thường thuộc 3 nhóm — thiếu negatives ("không tính..."),
   thiếu temporal ("trước khi..."), sai task_type. Thêm ví dụ mẫu cho đúng nhóm đó.

---

## ĐÌNH TRUNG (N4)

### Issue #8 — Chốt công nghệ giao diện + chạy thử
1. `git pull` (sau khi PR #13 merge). Bật backend như bước của Khang.
2. Mở `http://localhost:8000`, bấm `?` xem phím tắt, thử TỪNG phím.
3. Thử vũ khí bí mật: `Win+Shift+S` chụp một góc màn hình → `Ctrl+V` vào trang.
4. Ghi 5 điều thấy vướng/chậm → dán vào issue #8.
5. Ra quyết định: giữ giao diện 1-file (ui/index.html) hay chuyển React.
   Gợi ý của Dũng: GIỮ 1-file cho đến sau thi thử 15/8 — đang chạy tốt, đổi
   công nghệ giữa chừng là rủi ro lớn nhất của mảng UI. Ghi quyết định → Done.

### Issue #9 — Phím tắt + nối API thật
1. Sau khi Khang xong issue #5 (dữ liệu thật): mở UI, tìm thử 10 câu khác nhau
   trên dữ liệu THẬT, đo bằng đồng hồ xem từ lúc gõ đến lúc thấy ảnh mất mấy giây.
2. Ảnh thật sẽ NHIỀU và NẶNG hơn mock — nếu cuộn lag, ghi vào issue #9 kèm số
   lượng kết quả đang hiển thị (Dũng + Trung sẽ tối ưu sau).

---

## QUANG DŨNG (N5)

### Issue #10 — Kích hoạt "bảng điểm"
1. Sau khi có dữ liệu thật + issue #5 xong: nhắn nhóm phân công mỗi người viết
   10 câu đề thử theo mẫu `eval/queries.template.jsonl` (hướng dẫn chi tiết cách
   viết đã gửi trong nhóm trước đó). Deadline: 4 ngày sau khi index thật chạy.
2. Gom về `eval/queries.jsonl`, giữ riêng 15 câu làm tập kín.
3. Chạy: `python eval/run_benchmark.py --tag baseline_v1` → công bố bảng số
   đầu tiên. Từ đây áp luật "không có số, không merge".

### Issue #11 — ✅ XONG (chờ merge PR #15). Đóng issue sau khi merge.

### Issue #12 — Lịch + tài nguyên
1. Cập nhật file 06: mọi mốc +9 ngày (dataset về 03/08). Thi thử lần 1 dời sang ~24/08.
2. Lập bảng ai có tài khoản Colab/GPU gì, dán vào nhóm (mẫu ở file 07 mục 4).
3. Hỏi BTC: đợt phát dữ liệu tiếp theo? thể thức nộp bài vòng loại? → dán trả lời vào issue.

---

## Thứ tự phụ thuộc (nhìn nhanh)

```
Dũng merge PR  →  Tuấn Anh #1 → #2 → #3  →  Khang #5  →  Trung #9, Dũng #10
                  Khang #4  ──────────────────↗
                  Thoại #7 → #6   (độc lập, làm song song)
                  Trung #8        (độc lập, làm song song)
```
