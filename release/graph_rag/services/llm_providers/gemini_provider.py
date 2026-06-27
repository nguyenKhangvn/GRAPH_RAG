import logging

logger = logging.getLogger(__name__)

from google import genai
from google.genai import types
from .base import BaseLLMProvider
from graph_rag.core.intents import IntentType
import json
import time


class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name):
        self.model_name = model_name
        # Key rotation
        from graph_rag.utils.key_rotation import get_rotator
        self._rotator = get_rotator()
        # Initialize with primary key
        self.client = genai.Client(api_key=api_key)
        self.last_usage = None

    @staticmethod
    def _extract_usage(response) -> dict:
        usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
        if not usage:
            return {}
        return {
            "prompt_tokens": getattr(usage, "prompt_token_count", None) or getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "candidates_token_count", None) or getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_token_count", None) or getattr(usage, "total_tokens", None),
        }

    def _new_client(self, key: str):
        return genai.Client(api_key=key)

    def _call_with_rotation(self, contents, config, max_retries=3):
        """Call Gemini API with automatic key rotation on failure."""
        last_err = None
        for attempt in range(max_retries):
            key = self._rotator.get_key()
            try:
                client = self._new_client(key)
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                return response
            except (ValueError, RuntimeError, OSError) as e:
                last_err = e
                err_str = str(e)
                is_retryable = any(k in err_str for k in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"])
                if is_retryable and attempt < max_retries - 1:
                    self._rotator.mark_failed(key)
                    wait = 10 * (attempt + 1)
                    logger.info("    Key rotation: retry %s/%s after %ss (key ...%s)", attempt+1, max_retries, wait, key[-6:])
                    time.sleep(wait)
                else:
                    break
        raise last_err

    def _call_with_rotation_stream(self, contents, config, max_retries=3):
        """Stream Gemini API with automatic key rotation on failure."""
        last_err = None
        for attempt in range(max_retries):
            key = self._rotator.get_key()
            try:
                client = self._new_client(key)
                for chunk in client.models.generate_content_stream(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                ):
                    yield chunk
                return
            except (ValueError, RuntimeError, OSError) as e:
                last_err = e
                err_str = str(e)
                is_retryable = any(k in err_str for k in ["429", "503", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"])
                if is_retryable and attempt < max_retries - 1:
                    self._rotator.mark_failed(key)
                    wait = 10 * (attempt + 1)
                    logger.info("    Key rotation (stream): retry %s/%s after %ss (key ...%s)", attempt+1, max_retries, wait, key[-6:])
                    time.sleep(wait)
                else:
                    break
        raise last_err

    def generate_text(self, system_prompt, user_prompt, max_tokens=2048):
        try:
            response = self._call_with_rotation(
                contents=f"{system_prompt}\n\nUser Question: {user_prompt}",
                config=types.GenerateContentConfig(temperature=0.3),
            )
            self.last_usage = self._extract_usage(response)
            return response.text
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(" Gemini Text Error: %s", e)
            return "Xin lỗi, tôi không thể trả lời lúc này."

    def generate_json(self, system_prompt, user_prompt):
        try:
            response = self._call_with_rotation(
                contents=f"{system_prompt}\n\nUser Query: {user_prompt}",
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1,
                ),
            )
            self.last_usage = self._extract_usage(response)
            if response.text:
                return json.loads(response.text)
            return {}
        except (OSError, FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(" Gemini JSON Error: %s", e)
            return {"search_keywords": [user_prompt], "intent": IntentType.DISCOVERY}

    def generate_text_stream(self, system_prompt, user_prompt, on_token=None):
        try:
            collected = []
            for chunk in self._call_with_rotation_stream(
                contents=f"{system_prompt}\n\nUser Question: {user_prompt}",
                config=types.GenerateContentConfig(temperature=0.3),
            ):
                text = chunk.text or ""
                if text:
                    collected.append(text)
                    if on_token:
                        on_token(text)
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    self.last_usage = self._extract_usage(chunk)
            return "".join(collected)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error(" Gemini Stream Error: %s, falling back to non-stream", e)
            return self.generate_text(system_prompt, user_prompt)