import logging

logger = logging.getLogger(__name__)

import json
import re
from typing import Any

from openai import OpenAI

from .base import BaseLLMProvider
from graph_rag.core.intents import IntentType


class XAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.model_name = model_name
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

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()

        chunks = []
        for item in getattr(response, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    chunks.append(part_text)

        return "".join(chunks).strip()

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
        return cleaned

    @classmethod
    def _load_json(cls, text: str) -> dict:
        cleaned = cls._strip_code_fences(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {}

    def _call_responses_api(self, system_prompt: str, user_prompt: str, temperature: float, max_output_tokens: int) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=system_prompt,
            input=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        self.last_usage = self._extract_usage(response)
        return self._extract_text(response)

    def _call_chat_completion(self, system_prompt: str, user_prompt: str, temperature: float, max_tokens: int):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.last_usage = self._extract_usage(response)
        return response.choices[0].message.content or ""

    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 1024) -> str:
        try:
            result = self._call_responses_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.3,
                max_output_tokens=1024,
            )
            if result:
                return result
            return self._call_chat_completion(system_prompt, user_prompt, temperature=0.3, max_tokens=1024)
        except (ValueError, RuntimeError, OSError) as exc:
            logger.error("[XAI Error] %s", exc)
            return "Xin lỗi, tôi không thể trả lời lúc này."

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        if "json" not in system_prompt.lower():
            system_prompt += "\nIMPORTANT: Output strictly in JSON format."

        try:
            result_text = self._call_responses_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.1,
                max_output_tokens=1024,
            )
            parsed = self._load_json(result_text)
            if parsed:
                return parsed

            fallback_text = self._call_chat_completion(
                system_prompt,
                user_prompt,
                temperature=0.1,
                max_tokens=1024,
            )
            parsed = self._load_json(fallback_text)
            if parsed:
                return parsed
        except (ValueError, RuntimeError, OSError) as exc:
            logger.error("[XAI JSON Error] %s", exc)

        return {"search_keywords": [user_prompt], "intent": IntentType.DISCOVERY}