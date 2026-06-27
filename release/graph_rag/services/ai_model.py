# graph_rag/service/ai_model.py
import json
import logging
import time

from graph_rag.services.llm_usage import LLMUsageTracker, estimate_tokens

from graph_rag.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    ENABLE_LLM_FALLBACKS,
    GEMINI_API_KEY,
    GROQ_API_KEY,
    LLM_FALLBACK_MODELS,
    LLM_MODEL_NAME,
    MIMO_API_KEY,
    MIMO_BASE_URL,
    OPENAI_API_KEY,
    XAI_API_KEY,
)
from .llm_providers.gemini_provider import GeminiProvider
from .llm_providers.openai_provider import OpenAIProvider
from .llm_providers.groq_provider import GroqProvider
from .llm_providers.xai_provider import XAIProvider
from .llm_providers.mimo_provider import MiMoProvider
from .llm_providers.deepseek_provider import DeepSeekProvider

logger = logging.getLogger("graph_rag.llm_service")

# Map model name substring -> env key name (to check availability)
_MODEL_KEY_MAP = {
    "gemini": "GEMINI_API_KEY",
    "gpt": "OPENAI_API_KEY",
    "llama": "GROQ_API_KEY",
    "mixtral": "GROQ_API_KEY",
    "groq": "GROQ_API_KEY",
    "grok": "XAI_API_KEY",
    "xai": "XAI_API_KEY",
    "mimo": "MIMO_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Resolve env key name -> actual value
_KEY_VALUES = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "GROQ_API_KEY": GROQ_API_KEY,
    "XAI_API_KEY": XAI_API_KEY,
    "MIMO_API_KEY": MIMO_API_KEY,
    "DEEPSEEK_API_KEY": DEEPSEEK_API_KEY,
}


def _has_api_key(model_name: str) -> bool:
    """Check if the env API key for this model name is available."""
    normalized = (model_name or "").lower()
    for substr, key_name in _MODEL_KEY_MAP.items():
        if substr in normalized:
            return bool(_KEY_VALUES.get(key_name))
    return False


class LLMService:
    """
    Facade/Factory for LLM providers.

    Primary provider is selected by model_name. Fallback providers are disabled
    by default for reproducible DeepSeek-only evaluation, and can be enabled
    explicitly with ENABLE_LLM_FALLBACKS=true.
    """

    def __init__(self, api_key=None, model_name=None):
        self.model_name = model_name or LLM_MODEL_NAME
        self.provider = self._get_provider(self.model_name, api_key)
        self.fallback_providers = self._build_fallback_providers()
        self.usage_tracker = LLMUsageTracker()

    def _get_provider(self, model_name, api_key_override):
        """Factory Method: Chỉ sửa ở đây khi thêm hãng mới"""
        normalized_model = (model_name or "").lower()

        if "gemini" in normalized_model:
            key = api_key_override or GEMINI_API_KEY
            return GeminiProvider(key, model_name)

        elif "gpt" in normalized_model:
            key = api_key_override or OPENAI_API_KEY
            return OpenAIProvider(key, model_name)

        elif any(x in normalized_model for x in ["llama", "mixtral", "groq"]):
            key = api_key_override or GROQ_API_KEY
            return GroqProvider(key, model_name)

        elif any(x in normalized_model for x in ["grok", "xai"]):
            key = api_key_override or XAI_API_KEY
            return XAIProvider(key, model_name)

        elif "mimo" in normalized_model:
            key = api_key_override or MIMO_API_KEY
            return MiMoProvider(key, model_name, base_url=MIMO_BASE_URL)

        elif "deepseek" in normalized_model:
            key = api_key_override or DEEPSEEK_API_KEY
            if not key:
                raise RuntimeError("Missing DEEPSEEK_API_KEY for DeepSeek provider.")
            return DeepSeekProvider(key, model_name, base_url=DEEPSEEK_BASE_URL)

        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _build_fallback_providers(self):
        """Instantiate fallback providers from config, skipping unavailable keys."""
        if not ENABLE_LLM_FALLBACKS:
            logger.info("LLM fallback providers disabled.")
            return []
        providers = []
        primary_normalized = (self.model_name or "").lower()
        for fb_model in LLM_FALLBACK_MODELS:
            fb_normalized = (fb_model or "").lower()
            # Skip if fallback is same model as primary
            if fb_normalized == primary_normalized:
                continue
            if not _has_api_key(fb_model):
                logger.info("Fallback '%s' skipped: no API key", fb_model)
                continue
            try:
                providers.append(self._get_provider(fb_model, None))
                logger.info("Fallback provider registered: %s", fb_model)
            except ValueError:
                logger.warning("Fallback '%s' skipped: unknown model", fb_model)
        return providers

    def _call_with_fallback(self, method: str, system: str, user: str, **kwargs):
        """Try primary provider, then fallbacks in order."""
        providers = [self.provider] + self.fallback_providers
        last_err = None
        for p in providers:
            try:
                start = time.time()
                result = getattr(p, method)(system, user, **kwargs)
                latency_ms = int((time.time() - start) * 1000)
                self._record_usage(p, method, system, user, result, latency_ms)
                if p is not self.provider:
                    logger.warning(
                        "Fallback provider %s succeeded for %s",
                        type(p).__name__, method,
                    )
                return result
            except (ValueError, RuntimeError, OSError) as e:
                last_err = e
                logger.warning(
                    "Provider %s failed on %s: %s",
                    type(p).__name__, method, e,
                )
        raise last_err

    def _record_usage(
        self,
        provider,
        call_type: str,
        system: str,
        user: str,
        result,
        latency_ms: int,
    ) -> None:
        try:
            usage = getattr(provider, "last_usage", None) or {}
            prompt_tokens = (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or usage.get("prompt_token_count")
                or 0
            )
            completion_tokens = (
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or usage.get("candidates_token_count")
                or 0
            )
            total_tokens = usage.get("total_tokens") or 0
            exact = bool(usage)

            if not exact:
                prompt_tokens = estimate_tokens([system, user])
                completion_tokens = estimate_tokens(result)
                total_tokens = prompt_tokens + completion_tokens

            if not total_tokens:
                total_tokens = int(prompt_tokens or 0) + int(completion_tokens or 0)

            self.usage_tracker.record(
                call_type=call_type,
                model=str(getattr(provider, "model", None) or getattr(provider, "model_name", None) or self.model_name),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                exact=exact,
                provider=type(provider).__name__,
            )
        except (ValueError, RuntimeError, OSError):
            logger.debug("LLM usage tracking skipped due to tracker error.")

    def generate_text(self, system, user, max_tokens=2048):
        return self._call_with_fallback("generate_text", system, user, max_tokens=max_tokens)

    def generate_json(self, system, user):
        return self._call_with_fallback("generate_json", system, user)

    def generate_text_stream(self, system, user, on_token=None):
        """Try primary provider streaming, then fallbacks.

        Nếu fail TRƯỚC token đầu → thử provider khác.
        Nếu fail SAU khi đã stream → raise exception (không silent fail).
        """
        providers = [self.provider] + self.fallback_providers
        last_err = None
        tokens_started = False

        def tracking_callback(token):
            nonlocal tokens_started
            tokens_started = True
            if on_token:
                on_token(token)

        for p in providers:
            try:
                start = time.time()
                result = p.generate_text_stream(system, user, on_token=tracking_callback)
                latency_ms = int((time.time() - start) * 1000)
                self._record_usage(p, "generate_text_stream", system, user, result, latency_ms)
                if p is not self.provider:
                    logger.warning(
                        "Fallback provider %s succeeded for generate_text_stream",
                        type(p).__name__,
                    )
                return result
            except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
                last_err = e
                if tokens_started:
                    logger.error(
                        "Provider %s failed mid-stream: %s",
                        type(p).__name__, e,
                    )
                    raise
                logger.warning(
                    "Provider %s failed on generate_text_stream (before first token): %s",
                    type(p).__name__, e,
                )
        raise last_err
