"""Chấm understand() trên bộ 20 truy vấn test (issue #6). DoD: >= 18/20 với provider thật.

Chạy:  python -m agent.test_understand           (provider theo env, mặc định mock)
       AIC_LLM_PROVIDER=anthropic python -m agent.test_understand
Chế độ mock chỉ kiểm tra SCHEMA + vài heuristic — con số đạt/trượt chỉ có ý nghĩa
khi chạy provider thật.
"""
import json
import sys
from pathlib import Path

from agent.understand import REQUIRED_KEYS, understand

# Console Windows mặc định cp1258 không in được ✓/✗ — ép UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def check(expect: dict, result: dict) -> list[str]:
    errs = []
    if not REQUIRED_KEYS <= set(result):
        errs.append(f"thiếu key: {REQUIRED_KEYS - set(result)}")
        return errs
    if not result["search_text_en"].strip():
        errs.append("search_text_en rỗng")
    exp_task = expect.get("task_type")
    if exp_task and exp_task != "avs_or_kis" and result["task_type"] != exp_task:
        errs.append(f"task_type={result['task_type']}, kỳ vọng {exp_task}")
    if expect.get("has_negatives") and not result["negatives"]:
        errs.append("kỳ vọng có negatives")
    if expect.get("has_temporal") and not result["temporal"]:
        errs.append("kỳ vọng có temporal")
    if expect.get("cooccur") and not result["cooccur"]:
        errs.append("kỳ vọng cooccur=true")
    return errs


def main():
    path = Path(__file__).parent / "test_queries.jsonl"
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    passed = 0
    for i, case in enumerate(cases, 1):
        try:
            result = understand(case["q"])
            errs = check(case["expect"], result)
        except Exception as e:  # noqa: BLE001 — chấm hết bộ, không dừng giữa chừng
            errs = [f"exception: {e}"]
        if errs:
            print(f"✗ [{i:02d}] {case['q'][:50]}...")
            for e in errs:
                print(f"      {e}")
        else:
            passed += 1
            print(f"✓ [{i:02d}] {case['q'][:50]}")
    print(f"\nKẾT QUẢ: {passed}/{len(cases)} (DoD issue #6: >=18 với provider thật)")


if __name__ == "__main__":
    main()
