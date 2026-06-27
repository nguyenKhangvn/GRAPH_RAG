import logging

logger = logging.getLogger(__name__)

import json
import time
import random
from groq import Groq, RateLimitError
from .base import BaseLLMProvider
from graph_rag.core.intents import IntentType

def _groq_call_with_retry(fn, max_retries=4):
    """Retry với exponential backoff khi gặp rate limit."""
    for attempt in range(max_retries):
        try:
            return fn()
        except RateLimitError:
            # Exponential backoff + jitter để giảm dồn cụm retry.
            base_wait = min(30, 2 ** (attempt + 1))
            jitter = random.uniform(0.2, 1.8)
            wait = round(base_wait + jitter, 2)
            logger.info("  [Rate Limit] Retry %s/%s after %ss...", attempt+1, max_retries, wait)
            time.sleep(wait)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("  [Groq Error] %s", e)
            return None
    return None


class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name):
        self.client = Groq(api_key=api_key)
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

    def generate_text(self, system_prompt, user_prompt, max_tokens=1024):
        def call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            self.last_usage = self._extract_usage(response)
            return response.choices[0].message.content

        result = _groq_call_with_retry(call)
        if result is None:
            return "Xin lỗi, tôi không thể trả lời lúc này."
        return result

    def generate_json(self, system_prompt, user_prompt):
        """Sử dụng tính năng 'JSON Mode' native của Llama 3 trên Groq"""
        if "json" not in system_prompt.lower():
            system_prompt += " \nIMPORTANT: Output strictly in JSON format."

        def call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            self.last_usage = self._extract_usage(response)
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            return {}

        result = _groq_call_with_retry(call)
        if result is None:
            return {"search_keywords": [user_prompt], "intent": IntentType.DISCOVERY}
        return result
