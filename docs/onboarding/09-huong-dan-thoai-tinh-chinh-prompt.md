# 09 — Hướng dẫn Gia Thoại: tinh chỉnh "người phiên dịch" (N3)

> Phân công mới (08/08): **Dũng viết code** phần N3, **Thoại tinh chỉnh prompt** trên Colab.
> Việc của Thoại không cần biết lập trình — chỉ cần sửa chữ tiếng Việt và đọc điểm số.

---

## 1. Hiểu đúng việc mình làm (đọc kỹ phần này)

Có ba hiểu lầm phổ biến, làm rõ trước khi bắt đầu:

**❌ "Train AI cho hiểu tiếng Việt."**
✅ Claude **không train được** — Anthropic không cho phép fine-tune. Và Claude **đã hiểu tiếng Việt xuất sắc** rồi. Bài kiểm tra thực tế ngày 08/08:

| Đầu vào tiếng Việt | Claude dịch ra |
|---|---|
| `tim canh nguoi dan ong doi mu di qua duong` (không dấu) | *a man wearing a hat crossing the street* ✅ |
| "cầm **ly** cà phê cạnh **xe hơi**" (miền Nam) | *a girl holding a coffee cup next to a red car* |
| "cầm **cốc** cà phê cạnh **ô tô**" (miền Bắc) | *a girl holding a coffee cup next to a red car* ← **giống hệt** ✅ |
| "cho tui coi đoạn mà **ổng** đang sửa xe máy" (văn nói) | *a man repairing a motorcycle with tools* ✅ |
| "**không khí ấm cúng** của gia đình buổi tối" (trừu tượng) | *a family in a cozy living room with warm lighting* ✅ |

Không cần dạy nó tiếng Việt. Nó giỏi sẵn.

**❌ "Phải train trên Colab bằng GPU."**
✅ Việc của bạn **không cần GPU**. Colab ở đây chỉ là chỗ chạy thử tiện lợi (không phải cài đặt gì trên máy). Bấm Run, đọc điểm, sửa chữ, bấm Run lại.

**✅ Vậy việc thật của bạn là gì?**
Thứ **KHÔNG** hiểu tiếng Việt là **CLIP** — model tìm ảnh. CLIP chỉ hiểu tiếng Anh, và nó "thích" một số cách diễn đạt hơn cách khác. Bạn là người bắc cầu:

```
Người thi gõ tiếng Việt  →  [BẠN TINH CHỈNH CHỖ NÀY]  →  Mô tả tiếng Anh  →  CLIP tìm ảnh
     (mơ hồ, vội vàng)         Claude + prompt của bạn      (CLIP hiểu được)
```

Câu hỏi bạn phải trả lời mỗi ngày: **"Mô tả tiếng Anh này có làm CLIP tìm ra đúng ảnh không?"** — chứ không phải "dịch có đúng ngữ pháp không".

---

## 2. Bắt đầu (10 phút, làm 1 lần)

**Bước 1 — Tạo API key riêng của bạn.** Không dùng chung key với Dũng (lộ key nào thì thu hồi đúng key đó).
1. Vào `console.anthropic.com` → đăng nhập → **API Keys** → **Create Key** → đặt tên `aic2026-thoai`.
2. Copy chuỗi `sk-ant-api03-...` — **nó chỉ hiện một lần duy nhất**.
3. Nếu tài khoản chưa có tiền: nhờ Dũng nạp, hoặc dùng chung tài khoản của Dũng nhưng key riêng.

**Bước 2 — Mở notebook trên Colab.**
1. Vào `colab.research.google.com` → **File → Open notebook → GitHub**.
2. Tích ô **"Bao gồm các kho lưu trữ riêng tư"** (bắt buộc, repo của đội là private) → cho phép popup GitHub hiện ra và bấm Authorize.
3. Gõ `dungnguyen02102007-commits/aic2026` → chọn nhánh `dung/tuan1-khoi-dong-3-mang` → mở
   **`agent/notebooks/05_prompt_tuning_tieng_viet.ipynb`**.

**Bước 3 — Cắm key vào Colab AN TOÀN.**
Nhìn thanh bên trái Colab → biểu tượng **chìa khóa 🔑** → **+ Add new secret**:
- Name: `ANTHROPIC_API_KEY`
- Value: dán key của bạn
- Bật công tắc **Notebook access**

⚠️ **Tuyệt đối không gõ key thẳng vào ô code** — notebook nằm trong GitHub, gõ key vào là công khai cho cả thế giới.

**Bước 4 — Chạy.** Runtime → Run all. Cell cuối in ra điểm dạng `KẾT QUẢ: 20/20`.

---

## 3. Vòng lặp công việc hằng ngày

```
1. Chạy BƯỚC 3 trong notebook  →  ghi lại điểm hiện tại (vd 20/20)
2. Chạy BƯỚC 5 (bẫy tiếng Việt) →  đọc từng bản dịch tiếng Anh
3. Thấy bản dịch nào tệ         →  sửa/thêm luật ở BƯỚC 4
4. Chạy lại BƯỚC 3              →  điểm có tụt không?
5. Điểm giữ nguyên hoặc tăng    →  giữ. Điểm tụt → hoàn tác.
6. Cuối ngày: copy prompt tốt nhất, gửi Dũng đưa vào repo
```

**Luật vàng: không bao giờ sửa prompt mà không chạy lại điểm.** Prompt hay dở là chuyện cảm giác; điểm số mới là sự thật.

---

## 4. Cách sửa prompt cho hiệu quả

Toàn bộ "bộ não" nằm trong `agent/prompts.py`, hai phần:

**`UNDERSTAND_SYSTEM`** — bản hướng dẫn, viết tiếng Việt được. Ví dụ luật hữu ích:
> *"Ưu tiên danh từ cụ thể nhìn thấy được. Thay 'phương tiện giao thông' bằng 'xe máy' hoặc 'ô tô'."*

**`UNDERSTAND_FEWSHOT`** — các ví dụ mẫu. **Đây là công cụ mạnh nhất.** Dạy bằng ví dụ hiệu quả hơn nhiều so với dạy bằng lời:

```
Người dùng hỏi:  "Tìm cảnh người lính chì trong trung tâm thương mại"
Trả lời mẫu:     "toy soldier figurine in red coat and tall black hat in a shopping mall"
```

Vì sao ví dụ này quý: "lính chì" là khái niệm hiếm, CLIP không biết. Nhưng CLIP biết "red coat", "tall black hat" — mô tả **bề ngoài** thay vì gọi **tên**. Đây chính là bài học Case Study 1 của BTC.

**Ba nguyên tắc viết mô tả cho CLIP:**
1. **Mô tả cái nhìn thấy, không phải cái suy ra.** "vui vẻ" → "smiling, laughing". "giàu có" → "expensive suit, luxury car".
2. **Cụ thể hơn trừu tượng.** "đồ ăn" → "bowl of pho with beef and herbs".
3. **Ngắn gọn như caption ảnh.** CLIP được huấn luyện trên caption ngắn, không phải văn xuôi dài.

---

## 5. Việc theo mốc

| Mốc | Việc của bạn |
|---|---|
| **Bây giờ** | Chạy notebook, làm quen vòng lặp; mở rộng bộ test từ 20 → 50 câu (thêm bẫy tiếng Việt bạn nghĩ ra) |
| Khi có dữ liệu thật | Kiểm tra bản dịch có tìm ra ảnh đúng không — đây là lúc biết prompt tốt thật hay chỉ đẹp trên giấy |
| **M3** | Tinh chỉnh prompt cho **VLM rerank**: dạy Claude chấm từng điều kiện ("có kem chưa? có biển chưa?") |
| **M4** | Tinh chỉnh prompt **KISC**: dạy trợ lý biết hỏi ngược câu nào để thu hẹp nhanh nhất |

---

## 6. Gặp lỗi

| Hiện tượng | Cách xử lý |
|---|---|
| `KeyError: ANTHROPIC_API_KEY` | Chưa thêm secret, hoặc chưa bật "Notebook access" ở biểu tượng 🔑 |
| `AuthenticationError` | Key sai/đã bị thu hồi → tạo key mới |
| `credit balance is too low` | Tài khoản hết tiền → báo Dũng nạp |
| Notebook không hiện trong Colab | Chưa tích ô "Bao gồm các kho lưu trữ riêng tư" |
| Điểm tụt sau khi sửa | Hoàn tác luật vừa thêm. Thêm từng luật một, đừng thêm 5 luật rồi mới chạy |

Kẹt quá 30 phút → nhắn nhóm kèm ảnh chụp màn hình lỗi.
