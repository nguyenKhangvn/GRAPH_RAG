import json
import re
from typing import Any, Dict, List


def estimate_tokens(text: Any) -> int:
    if text is None:
        return 0
    if isinstance(text, (list, tuple)):
        return sum(estimate_tokens(item) for item in text)
    if isinstance(text, dict):
        return estimate_tokens(json.dumps(text, ensure_ascii=False))
    value = str(text or "")
    if not value.strip():
        return 0
    words = re.findall(r"\w+|[^\w\s]", value, flags=re.UNICODE)
    return max(1, int(len(words) * 1.15))


class LLMUsageTracker:
    def __init__(self) -> None:
        self.reset(question_id="")

    def reset(self, question_id: str) -> None:
        self.question_id = str(question_id or "")
        self.calls: List[Dict[str, Any]] = []

    def record(
        self,
        call_type: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        latency_ms: int,
        exact: bool,
        provider: str,
    ) -> None:
        self.calls.append(
            {
                "call_type": call_type,
                "model": model,
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or 0),
                "latency_ms": int(latency_ms or 0),
                "exact": bool(exact),
                "provider": provider,
            }
        )

    def snapshot(self) -> Dict[str, Any]:
        total_prompt = sum(int(c.get("prompt_tokens") or 0) for c in self.calls)
        total_completion = sum(int(c.get("completion_tokens") or 0) for c in self.calls)
        total_tokens = sum(int(c.get("total_tokens") or 0) for c in self.calls)
        exact = all(bool(c.get("exact")) for c in self.calls) if self.calls else False
        return {
            "question_id": self.question_id,
            "exact": exact,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "calls": list(self.calls),
        }
