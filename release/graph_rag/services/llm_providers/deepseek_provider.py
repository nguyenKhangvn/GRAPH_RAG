import json
import re

from openai import OpenAI

from .base import BaseLLMProvider


def _strip_code_fences(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _extract_json(text: str) -> dict:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


class DeepSeekProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name, base_url=None):
        self.model = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com/v1",
            timeout=120,
        )
        self.last_usage = None

    @staticmethod
    def _extract_usage(response) -> dict:
        usage = getattr(response, "usage", None)
        if not usage:
            return {}
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def generate_text(self, system_prompt, user_prompt, max_tokens=2048):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        self.last_usage = self._extract_usage(response)
        return response.choices[0].message.content or ""

    def generate_json(self, system_prompt, user_prompt):
        if "json" not in system_prompt.lower():
            system_prompt += "\nIMPORTANT: Output strictly in JSON format."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        self.last_usage = self._extract_usage(response)
        return _extract_json(response.choices[0].message.content or "")

    def generate_text_stream(self, system_prompt, user_prompt, on_token=None):
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )
        collected = []
        for chunk in stream:
            if not chunk.choices:
                continue
            token = chunk.choices[0].delta.content or ""
            if token:
                collected.append(token)
                if on_token:
                    on_token(token)
            if hasattr(chunk, "usage") and chunk.usage:
                self.last_usage = self._extract_usage(chunk)
        return "".join(collected)
