"""Q&A — trả lời câu hỏi về khoảnh khắc trong video (dạng 2, docs/onboarding/10).

    Nộp:  <video_id>, <frame_id>, <answer>
    Chấm: đúng video VÀ frame trong [s,e] VÀ answer khớp NGỮ NGHĨA.

Đường ống: mô tả sự kiện -> tìm khoảnh khắc như KIS -> đưa vài frame quanh đó
cho Claude xem -> Claude trả lời câu hỏi.

Vì sao đưa NHIỀU frame chứ không phải một: câu hỏi Q&A hay hỏi về diễn biến
("sau đó ai lên nhận giải"), một khung hình tĩnh không trả lời được. Vài frame
trải theo thời gian cho thấy chuyện gì xảy ra.

Vì sao dùng CHUNG một đáp án cho cả 100 câu nộp: đáp án là của SỰ KIỆN, không
phải của từng khung hình. Nếu để mỗi dòng một đáp án khác nhau thì dòng may mắn
trúng đúng frame lại có thể mang đáp án sai — tự bắn vào chân mình.
"""
import base64
import io
import os

from agent.llm import parse_json
from agent.prompts import VQA_CHON_SYSTEM, VQA_SYSTEM

# 1568 là cạnh dài tối ưu của Anthropic — to hơn nữa thì API tự thu nhỏ lại,
# tốn token mà không thêm chi tiết. Trước đây để 1024 nhưng ảnh đưa vào chỉ
# 640px nên chữ trên biển hiệu không đọc nổi — đó là nguyên nhân 4/10 câu Q&A sai.
MAX_CANH = 1568
# Lượt CHỌN ứng viên gửi nhiều ảnh (vd 6 ứng viên × 3 khung = 18 ảnh) nên thu
# nhỏ mạnh — chỉ cần nhận ra cảnh, chưa cần đọc chữ. Lượt ĐỌC KỸ mới dùng
# MAX_CANH vì lúc đó chỉ còn một ứng viên.
CANH_CHON = 672


def _ma_hoa_anh(img, max_canh: int = MAX_CANH) -> dict:
    """PIL.Image hoặc ndarray -> block image của Anthropic API."""
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    if max(img.size) > max_canh:
        ti_le = max_canh / max(img.size)
        img = img.resize((int(img.width * ti_le), int(img.height * ti_le)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg",
                       "data": base64.b64encode(buf.getvalue()).decode()}}


def cat_luoi(img, n: int = 2, chong: float = 0.15):
    """Cắt ảnh thành lưới n×n CÓ CHỒNG MÉP, trả về các ô đã phóng to.

    Vì sao chồng mép: biển hiệu hay dòng chữ nằm vắt ngang đường cắt sẽ bị xẻ
    đôi, mỗi nửa một ô, đọc không ra. Cho các ô lấn nhau ~30% thì mọi vùng đều
    nằm trọn trong ít nhất một ô.

    Đây là công cụ "phóng to" của nhóm không gian trong khung STAR mà BTC giới
    thiệu ở buổi tập huấn 3 — CLIP và cả VLM đều đọc chữ nhỏ rất kém, phải cắt
    ra rồi mới đọc được.
    """
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    W, H = img.size
    w, h = int(W * (1 / n + chong)), int(H * (1 / n + chong))
    o = []
    for r in range(n):
        for c in range(n):
            x = int(c * (W - w) / max(1, n - 1))
            y = int(r * (H - h) / max(1, n - 1))
            o.append(img.crop((x, y, x + w, y + h)))
    return o


def doc_ky(images, question_vi: str, mo_ta_vi: str, frame_ids=None,
           n_luoi: int = 2) -> dict:
    """Lượt 2: đọc kỹ MỘT ứng viên đã chọn, có phóng to từng vùng.

    Lượt 1 (chon_va_tra_loi) chọn đúng cảnh nhưng xem ảnh nhỏ để đỡ token.
    Lượt này chỉ còn một ứng viên nên gửi được ảnh đầy đủ độ phân giải, kèm
    các ô phóng to — dành cho câu hỏi phải đọc chữ hoặc đếm vật nhỏ.
    """
    provider = os.environ.get("AIC_LLM_PROVIDER", "mock")
    if provider == "mock":
        return {"answer": f"(mock ky: {question_vi[:30]})", "chac_chan": 0.0,
                "giai_thich": "chế độ mock"}
    if not images:
        raise ValueError("Không có khung hình nào")

    import anthropic

    giua = len(images) // 2
    noi_dung = [{"type": "text", "text": f"BỐI CẢNH: {mo_ta_vi}"},
                {"type": "text", "text": "\n--- Toàn cảnh theo thứ tự thời gian ---"}]
    for i, img in enumerate(images):
        nhan = f"khung {i + 1}" + (f" (frame {frame_ids[i]})" if frame_ids else "")
        noi_dung.append({"type": "text", "text": nhan})
        noi_dung.append(_ma_hoa_anh(img))

    noi_dung.append({"type": "text",
                     "text": f"\n--- Khung {giua + 1} phóng to từng vùng "
                             f"(các vùng có lấn nhau) ---"})
    for k, o in enumerate(cat_luoi(images[giua], n_luoi), 1):
        noi_dung.append({"type": "text", "text": f"vùng {k}"})
        noi_dung.append(_ma_hoa_anh(o))

    noi_dung.append({"type": "text", "text": f"\nCÂU HỎI: {question_vi}"})

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("AIC_VQA_MODEL", "claude-sonnet-5"),
        max_tokens=1024, system=VQA_SYSTEM,
        messages=[{"role": "user", "content": noi_dung}])
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Model từ chối trả lời: {resp.stop_details}")
    raw = next(b.text for b in resp.content if b.type == "text")

    kq = parse_json(raw)
    if "answer" not in kq:
        raise ValueError(f"Thiếu key 'answer' — raw: {raw[:200]}")
    kq.setdefault("chac_chan", 0.5)
    kq.setdefault("giai_thich", "")
    return kq


def answer_question(images, question_vi: str, mo_ta_vi: str | None = None,
                    frame_ids=None) -> dict:
    """Trả về {"answer": "<ngắn gọn>", "chac_chan": 0..1, "giai_thich": "..."}.

    images    : danh sách ảnh (PIL hoặc ndarray), xếp theo THỨ TỰ THỜI GIAN
    question_vi: câu hỏi của BTC
    mo_ta_vi  : mô tả sự kiện kèm theo (giúp model biết đang nhìn cảnh gì)
    """
    provider = os.environ.get("AIC_LLM_PROVIDER", "mock")
    if provider == "mock":
        return {"answer": f"(mock answer: {question_vi[:40]})",
                "chac_chan": 0.0, "giai_thich": "chế độ mock"}
    if provider != "anthropic":
        raise ValueError("Q&A cần model nhìn được ảnh — đặt "
                         "AIC_LLM_PROVIDER=anthropic")
    if not images:
        raise ValueError("Không có khung hình nào để nhìn")

    import anthropic

    noi_dung = []
    for i, img in enumerate(images):
        nhan = (f"Khung hình {i + 1}"
                + (f" (frame {frame_ids[i]})" if frame_ids else ""))
        noi_dung.append({"type": "text", "text": nhan})
        noi_dung.append(_ma_hoa_anh(img))
    if mo_ta_vi:
        noi_dung.append({"type": "text", "text": f"Bối cảnh: {mo_ta_vi}"})
    noi_dung.append({"type": "text", "text": f"CÂU HỎI: {question_vi}"})

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("AIC_VQA_MODEL", "claude-sonnet-5"),
        max_tokens=1024, system=VQA_SYSTEM,
        messages=[{"role": "user", "content": noi_dung}])
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Model từ chối trả lời: {resp.stop_details}")
    # content là danh sách block; model đời mới bật thinking mặc định nên
    # block[0] thường KHÔNG phải text.
    raw = next(b.text for b in resp.content if b.type == "text")

    kq = parse_json(raw)
    if "answer" not in kq:
        raise ValueError(f"Thiếu key 'answer' — raw: {raw[:200]}")
    kq.setdefault("chac_chan", 0.5)
    kq.setdefault("giai_thich", "")
    return kq


def chon_va_tra_loi(ung_vien, question_vi: str, mo_ta_vi: str) -> dict:
    """Cho model xem NHIỀU ứng viên, tự chọn cái khớp mô tả rồi trả lời.

    Vì sao cần: đo thật cho thấy 5/10 câu Q&A sai KHÔNG phải vì model nhìn kém,
    mà vì ta chỉ đưa nó xem ứng viên hạng 1 — trong khi đáp án nằm ở hạng 26–57.
    Truy vấn Q&A khó tìm hơn KIS vì mô tả CỐ TÌNH không chứa thứ đang được hỏi.

    Model đóng luôn vai trò xếp hạng lại: ứng viên nó chọn được đẩy lên đầu
    danh sách nộp.

    ung_vien: [{"images": [...], "frame_ids": [...], "video_id": str}, ...]
    Trả về {"ung_vien": <chỉ số 0-based>, "answer", "chac_chan", "giai_thich"}.
    """
    provider = os.environ.get("AIC_LLM_PROVIDER", "mock")
    if provider == "mock":
        return {"ung_vien": 0, "answer": f"(mock: {question_vi[:30]})",
                "chac_chan": 0.0, "giai_thich": "chế độ mock"}
    if not ung_vien:
        raise ValueError("Không có ứng viên nào")

    import anthropic

    noi_dung = [{"type": "text", "text": f"MÔ TẢ SỰ KIỆN: {mo_ta_vi}"}]
    for k, uv in enumerate(ung_vien, 1):
        noi_dung.append({"type": "text",
                         "text": f"\n=== ỨNG VIÊN {k} (video {uv['video_id']}) ==="})
        for i, img in enumerate(uv["images"]):
            noi_dung.append({"type": "text",
                             "text": f"khung {i + 1}/{len(uv['images'])}"})
            noi_dung.append(_ma_hoa_anh(img, CANH_CHON))
    noi_dung.append({"type": "text", "text": f"\nCÂU HỎI: {question_vi}"})

    # Lượt CHỌN dùng model rẻ: "cảnh nào khớp mô tả" dễ hơn nhiều so với
    # "trả lời câu hỏi", mà lượt này lại gửi nhiều ảnh nhất (6 ứng viên × 3
    # khung = 18 ảnh) nên là chỗ tốn tiền nhất. Lượt ĐỌC KỸ mới cần model mạnh.
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.environ.get("AIC_VQA_CHON_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=1024, system=VQA_CHON_SYSTEM,
        messages=[{"role": "user", "content": noi_dung}])
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Model từ chối trả lời: {resp.stop_details}")
    raw = next(b.text for b in resp.content if b.type == "text")

    kq = parse_json(raw)
    if "answer" not in kq:
        raise ValueError(f"Thiếu key 'answer' — raw: {raw[:200]}")
    # model đánh số từ 1, code dùng từ 0
    chon = int(kq.get("ung_vien", 1)) - 1
    kq["ung_vien"] = max(0, min(chon, len(ung_vien) - 1))
    kq.setdefault("chac_chan", 0.5)
    kq.setdefault("giai_thich", "")
    return kq
