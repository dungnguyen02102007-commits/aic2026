"""understand(): truy vấn tiếng Việt -> StructuredQuery JSON (N3 - Gia Thoại).

Provider chọn qua biến môi trường (file .env — KHÔNG commit):
    AIC_LLM_PROVIDER = anthropic | openai | mock   (mặc định: mock)
    ANTHROPIC_API_KEY / OPENAI_API_KEY

Chế độ mock trả kết quả rule-based đơn giản — đủ để N2/N4 dev và test schema
mà không tốn tiền API. Chất lượng thật nằm ở provider LLM + prompts.py.
"""
import json
import os
import re
from pathlib import Path

from agent.prompts import UNDERSTAND_FEWSHOT, UNDERSTAND_SYSTEM

# Nạp .env ở gốc repo nếu có (pip install python-dotenv). Không có cũng chạy —
# khi đó biến môi trường phải được đặt sẵn trong shell.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

REQUIRED_KEYS = {"search_text_en", "variants", "positives", "negatives",
                 "temporal", "cooccur", "filters", "task_type"}


def understand(query_vi: str, history: list[dict] | None = None) -> dict:
    provider = os.environ.get("AIC_LLM_PROVIDER", "mock")
    raw = _call_llm(provider, query_vi, history or [])
    result = _parse_json(raw)
    missing = REQUIRED_KEYS - set(result)
    if missing:
        raise ValueError(f"LLM thiếu key: {missing} — raw: {raw[:200]}")
    return result


def _call_llm(provider: str, query_vi: str, history: list[dict]) -> str:
    messages = UNDERSTAND_FEWSHOT + history + [{"role": "user", "content": query_vi}]

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=os.environ.get("AIC_LLM_MODEL", "claude-sonnet-5"),
            max_tokens=2048, system=UNDERSTAND_SYSTEM, messages=messages)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"Model từ chối trả lời: {resp.stop_details}")
        # content là danh sách block (ThinkingBlock, TextBlock...) — các model
        # đời mới bật thinking mặc định nên block[0] thường KHÔNG phải text.
        return next(b.text for b in resp.content if b.type == "text")

    if provider == "openai":
        import openai
        client = openai.OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("AIC_LLM_MODEL", "gpt-4o"),
            messages=[{"role": "system", "content": UNDERSTAND_SYSTEM}] + messages)
        return resp.choices[0].message.content

    if provider == "mock":
        return _mock_understand(query_vi)

    raise ValueError(f"AIC_LLM_PROVIDER không hợp lệ: {provider}")


def _mock_understand(query_vi: str) -> str:
    """Rule-based tối giản cho dev offline. KHÔNG dùng khi thi."""
    q = query_vi.lower()
    negatives = []
    if any(w in q for w in ["không tính", "không liên quan", "không phải", "trừ "]):
        negatives = ["excluded condition (mock)"]
    temporal = []
    if any(w in q for w in ["trước khi", "sau đó", "rồi mới", "sau khi"]):
        temporal = [{"before": "first action (mock)", "after": "second action (mock)"}]
    task = "avs" if "tất cả" in q else ("vqa" if any(
        w in q for w in ["bao nhiêu", "ai ", "cái gì", "?"]) else "kis")
    return json.dumps({
        "search_text_en": f"(mock translation of: {query_vi[:60]})",
        "variants": ["(mock variant 1)", "(mock variant 2)"],
        "positives": ["(mock positive)"],
        "negatives": negatives,
        "temporal": temporal,
        "cooccur": True,
        "filters": {"objects": None, "publish_date_from": None,
                    "publish_date_to": None},
        "task_type": task,
    }, ensure_ascii=False)


def _parse_json(raw: str) -> dict:
    """Chịu được LLM bọc ```json ... ``` hoặc thêm chữ thừa quanh JSON."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    return json.loads(text)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Tìm cảnh tôi ăn kem bên bờ biển"
    print(json.dumps(understand(q), ensure_ascii=False, indent=2))
