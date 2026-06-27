import logging

logger = logging.getLogger(__name__)

import json
import re
import time
import random
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from .base import BaseLLMProvider
from graph_rag.core.intents import IntentType


def _mimo_call_with_retry(fn, max_retries=3):
    for attempt in range(max_retries):
        try:
            return fn()
        except (RateLimitError, APITimeoutError):
            base_wait = min(30, 2 ** (attempt + 1))
            jitter = random.uniform(0.5, 1.5)
            wait = round(base_wait + jitter, 2)
            logger.info("  [MiMo Rate Limit] Retry %s/%s after %ss...", attempt+1, max_retries, wait)
            time.sleep(wait)
        except APIError as e:
            status = getattr(e, "status_code", 0)
            if status == 401:
                logger.info("  [MiMo 401 Unauthorized] Invalid API key. FAIL-FAST, no retry.")
                return None
            if status in (502, 503, 504):
                wait = round(5 + random.uniform(0, 3), 2)
                logger.error("  [MiMo Server Error %s] Retry %s/%s after %ss...", status, attempt+1, max_retries, wait)
                time.sleep(wait)
            else:
                logger.error("  [MiMo API Error %s] %s", status, e)
                return None
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("  [MiMo Error] %s", e)
            return None
    return None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
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


class MiMoProvider(BaseLLMProvider):
    """Stateless LLM provider — no caching.

    Caching context-sensitive prompts (containing history, location,
    conversation state) inside a generic provider causes stale results
    when the same query text is sent with different contexts.
    Caching, if needed, should live in the orchestration layer with
    a context-aware cache key.
    """

    def __init__(self, api_key, model_name, base_url=None):
        self.model_name = model_name
        self.base_url = base_url or "https://openrouter.ai/api/v1"
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
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
        def call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=max_tokens,
            )
            self.last_usage = self._extract_usage(response)
            content = response.choices[0].message.content or ""
            # MiMo reasoning model may put the answer in reasoning_content
            # instead of content when the prompt is long/complex.
            if not content:
                reasoning = (response.choices[0].message.model_extra or {}).get("reasoning_content") or ""
                if reasoning:
                    logger.info("[MiMo] content empty, using reasoning_content (%d chars)", len(reasoning))
                    content = reasoning
            return content

        result = _mimo_call_with_retry(call)
        if result is None:
            return "Xin lỗi, tôi không thể trả lời lúc này."
        return result

    def generate_text_stream(self, system_prompt, user_prompt, on_token=None):
        def call():
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=2048,
                stream=True,
            )
            collected = []
            reasoning_collected = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                token = delta.content or ""
                if token:
                    collected.append(token)
                    if on_token:
                        on_token(token)
                reasoning_token = getattr(delta, "reasoning_content", None) or ""
                if reasoning_token:
                    reasoning_collected.append(reasoning_token)
                if hasattr(chunk, "usage") and chunk.usage:
                    self.last_usage = self._extract_usage(chunk)
            result = "".join(collected)
            if not result and reasoning_collected:
                reasoning_text = "".join(reasoning_collected)
                logger.info("[MiMo stream] content empty, using reasoning_content (%d chars)", len(reasoning_text))
                if on_token:
                    on_token(reasoning_text)
                result = reasoning_text
            return result

        result = _mimo_call_with_retry(call)
        if result is None:
            return "Xin lỗi, tôi không thể trả lời lúc này."
        return result

    def generate_json(self, system_prompt, user_prompt):
        if "json" not in system_prompt.lower():
            system_prompt += "\nIMPORTANT: Output strictly in JSON format."

        def call():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
            )
            self.last_usage = self._extract_usage(response)
            content = response.choices[0].message.content or ""
            if not content:
                reasoning = (response.choices[0].message.model_extra or {}).get("reasoning_content") or ""
                if reasoning:
                    logger.info("[MiMo JSON] content empty, using reasoning_content (%d chars)", len(reasoning))
                    content = reasoning
            return _extract_json(content)

        result = _mimo_call_with_retry(call)
        if result is None:
            return {"search_keywords": [user_prompt], "intent": IntentType.DISCOVERY}
        return result
