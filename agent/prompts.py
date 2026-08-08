"""Prompt cho understand() — N3 Gia Thoại. Tune ở đây, đừng sửa logic understand.py."""

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
