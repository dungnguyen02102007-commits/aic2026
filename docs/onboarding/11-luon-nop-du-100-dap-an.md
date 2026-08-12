# 11 — Luôn nộp đủ 100 đáp án

> Đây là chỗ **mất điểm nhiều nhất mà sửa dễ nhất** của cả hệ thống.
> Không cần model tốt hơn, không cần code thông minh hơn — chỉ cần nộp dài hơn.

---

## 1. Luật của BTC

Mỗi truy vấn được nộp **tối đa 100 câu trả lời**, xếp theo độ tin cậy giảm dần.

```
R@k         = câu trả lời ĐÚNG NHẤT trong k câu ĐẦU TIÊN,   k ∈ {1, 5, 20, 50, 100}
Final Score = trung bình cộng của 5 giá trị R@k
```

**Không có hình phạt nào cho câu trả lời sai.** Nộp sai 99 câu và đúng 1 câu vẫn được điểm.

Quy ra bảng — giả sử câu đúng đầu tiên nằm ở hạng *r*:

| Hạng của câu đúng | 1 | 2–5 | 6–20 | 21–50 | 51–100 | >100 hoặc không nộp |
|---|---|---|---|---|---|---|
| **Điểm** | **1.00** | 0.80 | 0.60 | 0.40 | 0.20 | **0** |

---

## 2. Vì sao nộp thiếu là tự vứt điểm

Nhìn cột cuối: câu đúng nằm ở hạng 87 vẫn được **0.2 điểm**. Nhưng nếu bạn chỉ nộp 20 câu thì nó **không nằm trong danh sách** → **0 điểm**.

Cùng một hệ thống, cùng một kết quả tìm kiếm, chỉ khác ở chỗ danh sách nộp dài hay ngắn.

### Ví dụ thật từ đội mình

Khi đo TRAKE, lúc đầu tôi dùng cách chọn video kém. Video đúng nằm ở các hạng:

```
Truy vấn:   01    02    03    04    05
Hạng:        1    24    14    13     2
```

Cùng kết quả đó, chỉ khác số câu nộp:

| Truy vấn | Hạng | Nộp 3 câu | Nộp 100 câu |
|---|---|---|---|
| Q01 | 1 | **1.00** | **1.00** |
| Q02 | 24 | **0** — ngoài danh sách | **0.40** |
| Q03 | 14 | **0** — ngoài danh sách | **0.60** |
| Q04 | 13 | **0** — ngoài danh sách | **0.60** |
| Q05 | 2 | **0.80** — vẫn lọt | **0.80** |
| | | **Final = 0.36** | **Final = 0.68** |

**Gần gấp đôi.** Không sửa một dòng model nào, không tìm thêm được đáp án nào — chỉ nộp danh sách dài hơn.

Ba truy vấn Q02–Q04 là ví dụ điển hình: hệ thống **đã tìm ra đáp án đúng rồi**, nó nằm ở hạng 13–24, nhưng vì danh sách nộp chỉ có 3 câu nên công sức đó đổ sông đổ biển.

---

## 3. ⚠️ Quy tắc vàng: THÊM VÀO CUỐI, không chèn lên đầu

Đây là chỗ dễ làm hỏng nhất.

Thêm câu trả lời vào **cuối** danh sách thì **không bao giờ mất điểm** — nó chỉ có thể làm R@k lớn tốt lên, không đụng tới R@1.

Nhưng nếu bạn chèn câu mới vào **đầu** và đẩy câu đúng từ hạng 1 xuống hạng 2, bạn **mất 0.2 điểm** ngay lập tức.

```
Trước:  [ĐÚNG, sai, sai]                    -> 1.00
Thêm cuối:  [ĐÚNG, sai, sai, sai, ..., sai] -> 1.00  (an toàn tuyệt đối)
Chèn đầu:   [sai, ĐÚNG, sai, sai, ..., sai] -> 0.80  (mất điểm)
```

→ Nộp thêm cho đủ 100 là việc **an toàn tuyệt đối**. Còn thay đổi thứ tự top-5 thì phải cân nhắc kỹ.

---

## 4. Lấy đâu ra 100 đáp án?

### Textual KIS — dễ nhất

`/search` đã trả về danh sách xếp hạng sẵn. Chỉ cần lấy `top_k=100` thay vì 10–20.

```python
payload = {"query_text": query_en, "top_k": 100}
answers = [{"video_id": s["video_id"], "frame_id": s["frame_idx"]}
           for s in results]           # đủ 100, giữ nguyên thứ tự
```

### Q&A — giống KIS, thêm phần trả lời

Cùng một `answer` áp cho cả 100 vị trí. Nếu VLM cho ra vài phương án khác nhau thì xếp phương án tin nhất lên trước.

```python
answers = [{"video_id": s["video_id"], "frame_id": s["frame_idx"],
            "answer": dap_an} for s in results]
```

### TRAKE — phải tự sinh ra, hiện đang thiếu nhiều nhất

Hiện `eval/run_trake.py` chỉ nộp **3 câu** (3 video ứng viên × 1 tổ hợp frame). Cần nhân lên theo hai chiều:

**Chiều 1 — nhiều video ứng viên hơn.** Đang lấy top-3, nên lấy top-20 hoặc hơn. Mỗi video tốn ~1 phút căn chỉnh nên đây là đánh đổi thời gian, không phải đánh đổi điểm.

**Chiều 2 — nhiều tổ hợp frame cho mỗi video.** `align_events()` trả về tổ hợp tốt nhất; có thể lấy thêm tổ hợp tốt thứ 2, thứ 3 (vd dịch một khoảnh khắc sang đỉnh điểm cao nhì).

```
20 video × 5 tổ hợp = 100 đáp án
```

Thứ tự nộp: mọi tổ hợp của video hạng 1 trước, rồi tới video hạng 2... Vì sai video là 0 điểm ngay, đặt cược vào video tin nhất trước là đúng.

---

## 5. Cách tự kiểm tra

`eval/run_benchmark.py` tự cảnh báo sau mỗi lần chạy:

```
Nộp CHƯA ĐỦ 100 câu (5 truy vấn): trake_L21_01, trake_L21_02, ...
  -> Hạng 100 vẫn được 0.2 điểm. Luôn nộp đủ 100.
```

**Thấy dòng này là còn điểm đang bị vứt đi.** Chạy tới khi nó biến mất.

---

## 6. Tóm tắt cho người vội

1. Luôn nộp **đủ 100** câu, mọi dạng đề, mọi lúc.
2. Câu sai **không bị trừ điểm** — không có lý do gì nộp ít hơn.
3. Thêm vào **cuối** danh sách là an toàn tuyệt đối; đừng chèn lên đầu.
4. Thứ tự 5 câu đầu quyết định nhiều nhất — đó là chỗ đáng bỏ công tinh chỉnh.
