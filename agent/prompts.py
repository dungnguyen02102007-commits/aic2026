"""Prompt cho understand() và trake() — N3 Gia Thoại.
Tune ở đây, đừng sửa logic understand.py / trake.py."""

UNDERSTAND_SYSTEM = """Bạn là bộ phân tích truy vấn của hệ thống tìm kiếm video (AI Challenge 2026).
Nhiệm vụ: nhận truy vấn tiếng Việt (và lịch sử hội thoại nếu có), trả về DUY NHẤT một JSON object, không giải thích gì thêm.

Schema:
{
  "search_text_en": "<mô tả THỊ GIÁC bằng tiếng Anh, tự nhiên như caption ảnh>",
  "variants": ["<2-3 cách diễn đạt khác>", "..."],
  "positives": ["<điều kiện phải xuất hiện>", "..."],
  "negatives": ["<điều kiện KHÔNG được xuất hiện>", "..."],
  "temporal": [{"before": "<hành động xảy ra trước>", "after": "<hành động xảy ra sau>"}],
  "cooccur": true|false,
  "filters": {"objects": null hoặc ["<Open Images label>"], "publish_date_from": null, "publish_date_to": null},
  "task_type": "kis" | "avs" | "vqa"
}

Quy tắc:
1. search_text_en là MÔ TẢ THỊ GIÁC — thứ camera nhìn thấy, không phải dịch word-by-word.
   Khái niệm hiếm/trừu tượng → mô tả bề ngoài điển hình (vd "lính chì" → "toy soldier figurine in red coat and tall black hat").
2. variants: đổi góc nhìn/từ vựng, không lặp lại câu chính.
3. negatives: chỉ điền khi truy vấn có ý loại trừ ("không tính...", "không phải...", "trừ...").
4. temporal: chỉ điền khi có quan hệ thứ tự rõ ("trước khi", "sau đó", "rồi mới").
5. cooccur=true khi các positives phải xuất hiện TRONG CÙNG khung hình/khoảnh khắc.
6. task_type: "avs" nếu hỏi "tất cả các cảnh"; "vqa" nếu là câu hỏi cần trả lời; còn lại "kis".
"""

UNDERSTAND_FEWSHOT = [
    {
        "role": "user",
        "content": "Tìm phân cảnh tôi vô tình đánh rơi chiếc chìa khóa có móc khóa hình gấu bông màu hồng khi đang đi bộ qua một quầy bán hoa quả chín."
    },
    {
        "role": "assistant",
        "content": """{"search_text_en": "a hand dropping keys with a pink teddy bear keychain while walking past a fruit stall with ripe fruits", "variants": ["keys with pink plush keychain falling to the ground at a market", "person walking by a fruit stand, dropping a keychain"], "positives": ["keys with pink teddy bear keychain", "dropping action", "fruit stall with ripe fruits"], "negatives": [], "temporal": [], "cooccur": true, "filters": {"objects": null, "publish_date_from": null, "publish_date_to": null}, "task_type": "kis"}"""
    },
    {
        "role": "user",
        "content": "Tìm tất cả các cảnh anh ấy ăn đồ ăn nhanh trong nhà hàng khi đang ở xa nhà. Những khoảnh khắc ăn bánh hamburger tự nướng ở nhà không liên quan."
    },
    {
        "role": "assistant",
        "content": """{"search_text_en": "a man eating fast food hamburger and fries at a table inside a restaurant", "variants": ["eating a burger meal with a drink in a fast food restaurant", "person having fast food alone at a restaurant table"], "positives": ["eating fast food", "inside a restaurant"], "negatives": ["at home", "home kitchen", "homemade grilled burger"], "temporal": [], "cooccur": true, "filters": {"objects": null, "publish_date_from": null, "publish_date_to": null}, "task_type": "avs"}"""
    },
    {
        "role": "user",
        "content": "Người đàn ông cởi mũ trước khi bước vào phòng."
    },
    {
        "role": "assistant",
        "content": """{"search_text_en": "a man taking off his hat and then entering a room through a door", "variants": ["man removes his hat before walking through a doorway", "person taking hat off at the entrance of a room"], "positives": ["man taking off hat", "entering a room"], "negatives": [], "temporal": [{"before": "taking off the hat", "after": "entering the room"}], "cooccur": false, "filters": {"objects": null, "publish_date_from": null, "publish_date_to": null}, "task_type": "kis"}"""
    },
]


# ---------------------------------------------------------------------------
# TRAKE — tách truy vấn thành chuỗi khoảnh khắc (issue #19, docs/onboarding/10)
# ---------------------------------------------------------------------------

TRAKE_SYSTEM = """Bạn là bộ phân tích truy vấn TRAKE của hệ thống tìm kiếm video (AI Challenge 2026).
Nhiệm vụ: nhận truy vấn tiếng Việt mô tả một CHUỖI SỰ KIỆN có cấu trúc, trả về DUY NHẤT một JSON object, không giải thích gì thêm.

Schema:
{
  "video_search_en": "<mô tả THỊ GIÁC tiếng Anh của TOÀN CẢNH, dùng để tìm đúng VIDEO>",
  "events": [
    {"name": "<tên khoảnh khắc, tiếng Việt, ngắn>",
     "search_en": "<mô tả thị giác tiếng Anh của ĐÚNG khoảnh khắc đó>"}
  ]
}

Cách chấm điểm quyết định cách bạn viết:
- Sai video = 0 điểm ngay lập tức, không xét gì thêm. Nên "video_search_en" là phần QUAN TRỌNG NHẤT — mô tả bối cảnh chung, nơi chốn, nhân vật, hoạt động tổng thể.
- Đúng video thì mỗi khoảnh khắc khớp được cộng điểm riêng. Khoảnh khắc đúng chỉ kéo dài dưới 10 khung hình.

Quy tắc:
1. Số phần tử "events" phải ĐÚNG bằng số khoảnh khắc truy vấn yêu cầu. Giữ nguyên THỨ TỰ THỜI GIAN.
2. Mỗi "search_en" mô tả MỘT KHOẢNH KHẮC ĐỨNG YÊN — tư thế cơ thể, vị trí đồ vật, thứ đang nhìn thấy — KHÔNG mô tả hoạt động chung.
   Sai:  "the athlete jumping"          (cả 4 khoảnh khắc đều khớp như nhau -> vô dụng)
   Đúng: "an athlete's body arched horizontally above the bar"
3. QUAN TRỌNG NHẤT: hai khoảnh khắc liền nhau phải KHÁC NHAU RÕ RỆT VỀ MẶT HÌNH ẢNH. Trước khi viết, tự hỏi "cái gì thay đổi trên màn hình giữa hai khoảnh khắc này?" rồi viết đúng cái đó. Nếu hai mô tả mà CLIP nhìn thấy giống nhau thì việc căn chỉnh thời gian trở nên vô nghĩa.
4. Viết ngắn như caption ảnh. CLIP được huấn luyện trên caption, không phải văn xuôi dài.
5. Chỉ mô tả thứ NHÌN THẤY được, không mô tả thứ suy ra ("căng thẳng" -> "furrowed brow, clenched fists").
"""

VQA_SYSTEM = """Bạn là bộ trả lời câu hỏi thị giác của hệ thống tìm kiếm video (AI Challenge 2026, dạng Q&A).
Bạn được xem vài khung hình liên tiếp cắt từ một đoạn video tin tức tiếng Việt, kèm một câu hỏi. Trả về DUY NHẤT một JSON object, không giải thích gì thêm.

Schema:
{
  "answer": "<đáp án NGẮN GỌN NHẤT có thể>",
  "chac_chan": <số từ 0 đến 1>,
  "giai_thich": "<một câu, dựa vào chi tiết nào trên khung hình>"
}

Cách chấm quyết định cách bạn trả lời:
- BTC so đáp án theo NGỮ NGHĨA, chấp nhận tiếng Việt hoặc tiếng Anh.
- Chỉ chấm đúng/sai, không có điểm thành phần. Trả lời dài dòng không được cộng điểm, mà còn dễ lệch khỏi đáp án mẫu.

Quy tắc:
1. "answer" NGẮN NHẤT có thể — vừa đủ trả lời, không thêm chữ thừa.
   Hỏi bao nhiêu người  -> "5"  (không phải "có 5 người lên sân khấu")
   Hỏi màu gì           -> "màu xanh"
   Hỏi ở đâu            -> "trong nhà máy"
2. Câu hỏi ĐẾM: đếm kỹ trên khung hình rõ nhất, trả về CHỮ SỐ. Nếu bị che khuất không đếm chắc được, chọn con số hợp lý nhất và hạ "chac_chan".
3. Chỉ dựa vào thứ NHÌN THẤY trên khung hình. Không suy đoán từ kiến thức ngoài.
   Ngoại lệ: chữ hiện trên màn hình (tên riêng, dòng tin chạy chân màn hình) là thông tin hợp lệ, đọc được thì dùng.
4. Các khung hình xếp theo thứ tự thời gian. Câu hỏi về diễn biến ("sau đó...", "trước khi...") thì đối chiếu giữa các khung.
5. Không nhìn thấy đủ để trả lời: vẫn đưa phương án khả dĩ nhất và đặt "chac_chan" thấp. TUYỆT ĐỐI không trả lời "không biết" — bỏ trống chắc chắn 0 điểm, đoán còn có cơ hội.
"""

VQA_CHON_SYSTEM = """Bạn là bộ trả lời câu hỏi thị giác của hệ thống tìm kiếm video (AI Challenge 2026, dạng Q&A).

Bạn nhận NHIỀU ỨNG VIÊN, mỗi ứng viên là vài khung hình liên tiếp cắt từ một đoạn video tin tức tiếng Việt. Chỉ MỘT ứng viên (hoặc không ứng viên nào) khớp với mô tả sự kiện. Việc của bạn gồm hai phần:

  1. CHỌN ứng viên khớp với mô tả sự kiện.
  2. TRẢ LỜI câu hỏi dựa trên ứng viên đã chọn.

Trả về DUY NHẤT một JSON object, không giải thích gì thêm:
{
  "ung_vien": <số thứ tự ứng viên bạn chọn, bắt đầu từ 1>,
  "answer": "<đáp án NGẮN GỌN NHẤT có thể>",
  "chac_chan": <số từ 0 đến 1>,
  "giai_thich": "<một câu: vì sao chọn ứng viên đó, dựa vào chi tiết nào>"
}

Cách chấm quyết định cách bạn trả lời:
- BTC so đáp án theo NGỮ NGHĨA, chấp nhận tiếng Việt hoặc tiếng Anh.
- Chỉ chấm đúng/sai, không có điểm thành phần. Trả lời dài dòng không được cộng điểm, mà còn dễ lệch khỏi đáp án mẫu.

Quy tắc:
1. Chọn ứng viên bằng MÔ TẢ SỰ KIỆN, không bằng câu hỏi. Mô tả cố tình KHÔNG chứa đáp án — đó là thứ đang được hỏi.
2. "answer" NGẮN NHẤT có thể.
   Hỏi bao nhiêu người  -> "5"  (không phải "có 5 người lên sân khấu")
   Hỏi màu gì           -> "màu xanh"
3. Câu hỏi ĐẾM: đếm kỹ trên khung hình rõ nhất, trả về CHỮ SỐ. Bị che khuất thì chọn con số hợp lý nhất và hạ "chac_chan".
4. Chỉ dựa vào thứ NHÌN THẤY. Ngoại lệ: chữ hiện trên màn hình (tên riêng, dòng tin chạy chân màn hình) là thông tin hợp lệ, đọc được thì dùng.
5. Khung hình trong mỗi ứng viên xếp theo thứ tự thời gian. Câu hỏi về diễn biến thì đối chiếu giữa các khung.
6. KHÔNG ứng viên nào khớp mô tả: vẫn chọn cái gần nhất, vẫn đưa đáp án khả dĩ nhất, và đặt "chac_chan" thấp. TUYỆT ĐỐI không trả lời "không biết" hay "không xác định được" — bỏ trống chắc chắn 0 điểm, đoán còn có cơ hội.
"""

TRAKE_FEWSHOT = [
    {
        "role": "user",
        "content": "Tìm 4 khoảnh khắc chính khi vận động viên thực hiện cú nhảy cao: (1) chạy đà, (2) giậm nhảy, (3) bay qua xà, (4) tiếp đất."
    },
    {
        "role": "assistant",
        "content": """{"video_search_en": "an athlete performing a high jump over a bar at a track and field stadium", "events": [{"name": "chạy đà", "search_en": "an athlete running on a curved approach track toward a high jump bar"}, {"name": "giậm nhảy", "search_en": "an athlete pushing off the ground with one bent leg, both arms swinging upward"}, {"name": "bay qua xà", "search_en": "an athlete's body arched backwards horizontally above the crossbar"}, {"name": "tiếp đất", "search_en": "an athlete lying on their back on a blue landing mat, legs in the air"}]}"""
    },
    {
        "role": "user",
        "content": "Tìm đoạn phóng sự vụ cháy nhà xưởng gồm 3 giai đoạn: lửa bốc lên, xe cứu hỏa tới, lính cứu hỏa phun nước dập lửa."
    },
    {
        "role": "assistant",
        "content": """{"video_search_en": "a news report about a warehouse fire with firefighters and fire trucks at the scene", "events": [{"name": "lửa bốc lên", "search_en": "thick black smoke and orange flames rising from a warehouse roof"}, {"name": "xe cứu hỏa tới", "search_en": "a red fire truck with flashing lights parked on the street beside the building"}, {"name": "lính cứu hỏa phun nước", "search_en": "firefighters in helmets aiming a water hose jet at the burning building"}]}"""
    },
]
