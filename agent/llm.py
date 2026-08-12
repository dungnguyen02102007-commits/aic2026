"""Phần gọi LLM dùng chung cho understand() và trake() — N3.

Tách ra khỏi understand.py để trake.py dùng lại mà không phải chép code.
Provider chọn qua biến môi trường (file .env — KHÔNG commit):
    AIC_LLM_PROVIDER = anthropic | openai | mock   (mặc định: mock)
    ANTHROPIC_API_KEY / OPENAI_API_KEY
"""
import json
import os
import re
from pathlib import Path

# Nạp .env ở gốc repo nếu có (pip install python-dotenv). Không có cũng chạy —
# khi đó biến môi trường phải được đặt sẵn trong shell.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass


def call_llm(system: str, fewshot: list[dict], user_text: str,
             history: list[dict] | None = None, max_tokens: int = 2048) -> str:
    """Gọi LLM, trả về text thô. Provider lấy từ AIC_LLM_PROVIDER."""
    provider = os.environ.get("AIC_LLM_PROVIDER", "mock")
    messages = fewshot + (history or []) + [{"role": "user", "content": user_text}]

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=os.environ.get("AIC_LLM_MODEL", "claude-sonnet-5"),
            max_tokens=max_tokens, system=system, messages=messages)
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
            messages=[{"role": "system", "content": system}] + messages)
        return resp.choices[0].message.content

    if provider == "mock":
        raise NotImplementedError(
            "Provider 'mock' phải được xử lý ở module gọi (understand/trake)")

    raise ValueError(f"AIC_LLM_PROVIDER không hợp lệ: {provider}")


def parse_json(raw: str) -> dict:
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


def provider() -> str:
    return os.environ.get("AIC_LLM_PROVIDER", "mock")
