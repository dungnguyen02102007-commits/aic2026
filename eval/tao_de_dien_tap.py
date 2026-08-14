"""Sinh 100 câu hỏi giả đúng định dạng BTC, để DIỄN TẬP ngày thi.

    python -m eval.tao_de_dien_tap --out eval/questions_dien_tap.csv

Mục đích KHÔNG phải đo điểm — đáp án là giả. Mục đích là **bấm giờ toàn
chuỗi**: 100 câu chạy hết bao lâu, khâu nào là nút cổ chai, có kịp giờ thi
không. Ngày 21 mới phát hiện chậm thì không sửa kịp.

Nội dung câu hỏi mô phỏng bản tin thời sự Việt Nam (giao thông, thời tiết,
nông nghiệp, công nghiệp, y tế, quốc tế, thể thao, văn hoá) — đúng loại nội
dung trong kho dữ liệu.
"""
import argparse
import csv
import random
from pathlib import Path

# Cảnh nền, viết theo văn phong mô tả của BTC
CANH = [
    ("một vụ tai nạn giao thông trên đường phố đông xe cộ", "xe máy"),
    ("cảnh ngập lụt, nước dâng cao trên đường làng", "nhà dân"),
    ("công nhân đang làm việc trên dây chuyền trong nhà máy", "công nhân"),
    ("một đám cháy lớn bốc khói đen tại khu nhà xưởng", "xe cứu hỏa"),
    ("người dân thu hoạch lúa trên cánh đồng", "người nông dân"),
    ("cảnh họp báo có nhiều phóng viên cầm máy quay", "micro"),
    ("một khu chợ đông người mua bán vào buổi sáng", "quầy hàng"),
    ("bác sĩ đang khám bệnh cho bệnh nhân trong phòng khám", "bác sĩ"),
    ("học sinh mặc đồng phục trắng trong sân trường", "học sinh"),
    ("cảnh thi công công trình cầu đường có máy xúc", "máy xúc"),
    ("người dẫn chương trình nam trong trường quay tin tức", "người dẫn"),
    ("cảnh biển với tàu cá neo đậu ở cảng", "tàu cá"),
    ("một buổi lễ khai trương có cắt băng khánh thành", "người dự lễ"),
    ("cảnh giao thông ùn tắc giờ cao điểm ở thành phố", "ô tô"),
    ("công nhân vệ sinh môi trường thu gom rác trên phố", "xe rác"),
    ("một trận đấu bóng đá trên sân cỏ có khán giả", "cầu thủ"),
    ("cảnh nuôi trồng thủy sản, ao cá ở vùng quê", "ao cá"),
    ("người dân xếp hàng tiêm chủng tại trạm y tế", "người dân"),
    ("cảnh núi lửa phun trào với cột tro bụi", "cột tro"),
    ("một cuộc diễu hành có nhiều người cầm cờ", "lá cờ"),
    ("cảnh sạt lở đất chắn ngang đường đèo", "đất đá"),
    ("gian hàng triển lãm công nghệ tại hội chợ", "gian hàng"),
    ("cảnh trẻ em vui chơi trong công viên", "trẻ em"),
    ("người thợ hàn đang làm việc với tia lửa bắn ra", "thợ hàn"),
    ("cảnh sân bay với máy bay đang lăn bánh", "máy bay"),
    ("một phiên tòa xét xử có bị cáo đứng trước vành móng ngựa", "bị cáo"),
    ("cảnh trang trại chăn nuôi bò sữa", "con bò"),
    ("người dân đi xe máy dưới trời mưa lớn", "áo mưa"),
    ("cảnh cứu hộ người bị nạn sau bão lũ", "lực lượng cứu hộ"),
    ("một buổi biểu diễn văn nghệ trên sân khấu có đèn màu", "diễn viên"),
    ("cảnh khai thác khoáng sản có xe tải lớn", "xe tải"),
    ("phòng thí nghiệm có nhà nghiên cứu mặc áo blouse trắng", "nhà nghiên cứu"),
    ("cảnh trồng rừng, người dân cầm cây giống", "cây giống"),
    ("một cửa hàng bán lẻ có khách hàng đang chọn đồ", "khách hàng"),
    ("cảnh đường sắt với đoàn tàu đang chạy qua", "đoàn tàu"),
]

HOI = [
    ("Trong cảnh có bao nhiêu {vat}?", "đếm"),
    ("{Vat} trong cảnh có màu gì?", "màu"),
    ("Cảnh này quay vào ban ngày hay ban đêm?", "thời điểm"),
    ("Chữ hiển thị trên màn hình ghi gì?", "đọc chữ"),
    ("{Vat} đang ở vị trí nào trong khung hình?", "vị trí"),
]

GIAI_DOAN = [
    ["lúc mới bắt đầu", "khi đang diễn ra", "lúc kết thúc"],
    ["cảnh toàn cảnh từ xa", "cảnh cận nhân vật chính", "cảnh người dân phản ứng"],
    ["trước khi lực lượng chức năng đến", "khi lực lượng chức năng làm việc",
     "sau khi xử lý xong"],
    ["cảnh hiện trường", "cảnh phỏng vấn nhân chứng", "cảnh người dẫn kết luận"],
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/questions_dien_tap.csv")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    # Tỉ lệ 3 dạng: chưa biết BTC chia thế nào, tạm chia đều để đo được cả ba
    dang = (["kis"] * ((args.n + 2) // 3) + ["qa"] * ((args.n + 1) // 3)
            + ["trake"] * (args.n // 3))[:args.n]
    rnd.shuffle(dang)

    rows = []
    for i, d in enumerate(dang, 1):
        canh, vat = CANH[rnd.randrange(len(CANH))]
        mo_ta = canh if canh.startswith("cảnh") else f"Cảnh {canh}"
        r = {"query_id": f"Q_{i:04d}", "task_type": d,
             "query_vi": mo_ta[0].upper() + mo_ta[1:],
             "question_vi": "", "events_vi": ""}
        if d == "qa":
            mau, _ = HOI[rnd.randrange(len(HOI))]
            r["question_vi"] = mau.format(vat=vat, Vat=vat.capitalize())
        elif d == "trake":
            gd = GIAI_DOAN[rnd.randrange(len(GIAI_DOAN))]
            r["events_vi"] = " | ".join(gd)
            r["query_vi"] += f", gồm {len(gd)} khoảnh khắc"
        rows.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "task_type", "query_vi",
                                          "question_vi", "events_vi"])
        w.writeheader()
        w.writerows(rows)

    dem = {d: dang.count(d) for d in ("kis", "qa", "trake")}
    print(f"{len(rows)} câu -> {out}")
    print(f"  kis {dem['kis']}   qa {dem['qa']}   trake {dem['trake']}")
    print("\nĐáp án là GIẢ — bộ này để BẤM GIỜ, không phải để đo điểm.")


if __name__ == "__main__":
    main()
