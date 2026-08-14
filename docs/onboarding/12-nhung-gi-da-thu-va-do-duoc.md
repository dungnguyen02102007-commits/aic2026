# 12 — Những gì đã thử và đo được

> Sổ ghi các thí nghiệm **đã chạy trên dữ liệu thật**, kèm số đo. Mục đích: đội
> không thử lại thứ đã biết là không ăn, và không bỏ qua thứ đã biết là ăn.
>
> Mọi con số dưới đây đo trên **17 truy vấn KIS**, kho **873 video / 177.321
> keyframe**, cùng một bản dịch đã khoá cache. Chênh 0.06 tương đương ~1 câu —
> đọc số với cỡ mẫu đó trong đầu.

---

## ✅ ĂN — nên giữ

### Fusion nhiều model bằng RRF

| Cấu hình | Final | R@1 |
|---|---|---|
| ViT-B/32 (BTC cấp) | 0.553 | 0.294 |
| ViT-L/14 | 0.529 | 0.176 |
| **Fusion cả hai** | **0.635** | **0.412** |

**Không model nào một mình thắng.** L/14 còn kém hơn B/32. Nhưng gộp lại hơn 15%, và R@1 nhảy từ 0.294 lên 0.412 — vì hai model **sai khác nhau** nên bù được cho nhau.

```bash
python eval/run_kis.py --index <thư-mục-b32> <thư-mục-l14> --map <map-keyframes>
```

### Biến thể truy vấn (query expansion)

`understand()` sinh thêm 2–3 cách diễn đạt, gộp bằng RRF. Đo được **0.800 vs 0.788** trên bộ 29 video — nhỏ nhưng dương, chủ yếu ở hạng 1.

### Ghép CÂU HỎI vào truy vấn tìm kiếm (dạng Q&A)

Mô tả của BTC cố tình bỏ đi thứ đang được hỏi nên kém phân biệt; câu hỏi lại lộ ra chi tiết nhìn thấy được (*"tên trên biển hiệu"* → cảnh phải có biển hiệu rõ). Top-5 từ **1/10 lên 3/10**.

Đội vô địch VBS2025 làm y hệt: *"What is the license plate number?"* → *"A red Volkswagen Golf GTI with a license plate"*.

---

## ❌ KHÔNG ĂN — đã thử, đừng thử lại

### Xếp tầng model (lớp 1 lọc, lớp 2 xếp lại)

| Cấu hình | Final | R@1 |
|---|---|---|
| Fusion song song | **0.635** | **0.412** |
| Xếp tầng, bể 200 | 0.588 | 0.176 |
| Xếp tầng, bể 1.000 | 0.553 | 0.176 |
| Xếp tầng, bể 5.000 | 0.529 | 0.176 |
| Xếp tầng, bể 20.000 | 0.529 | 0.176 |

**Bể càng lớn càng tệ**, hội tụ đúng về 0.529 — điểm của L/14 chạy một mình. Hợp lý: bể đủ lớn thì lớp 1 không lọc gì nữa, hệ thống thành "chỉ dùng L/14".

**Vì sao thua fusion:** xếp tầng để lớp 2 **ghi đè** phán đoán của lớp 1, nên thừa hưởng luôn điểm yếu của lớp 2, mà lại **mất khả năng cứu** những gì lớp 1 bỏ sót. Fusion cho hai model **bỏ phiếu**. Khi không model nào rõ ràng mạnh hơn (0.553 vs 0.529), bỏ phiếu thắng ghi đè.

*Xếp tầng thường được dùng để đổi lấy TỐC ĐỘ. Ở đây không có tốc độ nào để đổi — cả hai index đều `IndexFlatIP` 177 nghìn vector, mỗi lượt tìm vài mili giây.*

### Xếp lại kết quả theo video

| Cấu hình | Final | R@1 |
|---|---|---|
| RRF thuần | **0.553** | **0.294** |
| 20 video × 5 keyframe | 0.400 | 0.059 |
| 5 video × 20 keyframe | 0.247 | 0.059 |

Tổng điểm RRF theo video **thiên vị video dài** — video 20 phút có 400 keyframe khớp lờ mờ thắng video ngắn có một chỗ khớp mạnh.

### Index dày hơn

| Index | Số vector | Final | R@1 |
|---|---|---|---|
| 1 frame/3 giây | 10.021 | **0.800** | **0.529** |
| 1 frame/1 giây | 30.043 | 0.777 | 0.412 |

Dày gấp 3, tốn 22 phút thay vì 15, **không tốt hơn**. Ngân sách nộp cố định 100 câu — index càng dày thì mỗi khoảnh khắc chiếm càng nhiều ô, danh sách phủ được **ít khoảnh khắc khác nhau** hơn. KIS cần phủ rộng hơn phủ dày.

→ **Keyframe I-frame BTC phát sẵn (trung vị 53 frame) dùng thẳng được cho KIS/Q&A.** Không cần tự trích.

### Giới hạn số frame mỗi video

Ép đa dạng làm R@1 tốt hơn (0.588 vs 0.529) nhưng **mất hẳn 2 câu** ở hạng sâu. Tổng lại lỗ (0.776 vs 0.800).

→ Frame gần trùng nhau trong cùng video **không phải rác** — chúng là thêm vé số để trúng khoảng đáp án.

### Tinh chỉnh prompt cho các câu hỏng

Đọc lại bản dịch của 8 câu hỏng: **đều tốt**, chính xác và thị giác. Khoảng cách điểm giữa keyframe đúng và top-1 là **0.06–0.11** trong không gian CLIP — quá lớn để câu chữ đóng được. Cần tín hiệu khác, không phải cách diễn đạt khác.

---

## ⚠️ Bài học phương pháp

**Kết quả trên mẫu nhỏ không suy ra được kết quả ở quy mô thật.**

Trên bộ L21 (7.720 keyframe) đo được ViT-L/14 **hơn** B/32 13%. Trên toàn bộ 873 video, L/14 **kém hơn**. Nếu chốt theo bộ nhỏ thì đã làm tệ hơn hiện trạng.

Mọi quyết định model/tham số phải đo trên **toàn bộ kho**, không đo trên lô con.

**Luôn khoá bản dịch khi so cấu hình.** `understand()` gọi LLM nên mỗi lần dịch một khác; không khoá thì chênh lệch đo được là nhiễu dịch thuật chứ không phải hiệu quả thật.

```bash
python eval/run_kis.py ... --cache eval/results/understand_cache.json
```
