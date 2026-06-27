# graph_rag/service/llm_providers/openai_provider.py
import json
from openai import OpenAI
from .base import BaseLLMProvider

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key, model_name):
        self.client = OpenAI(api_key=api_key)
        self.model = model_name
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
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens
            )
            self.last_usage = self._extract_usage(response)
            return response.choices[0].message.content or ""
        except (ValueError, RuntimeError, OSError) as e:
            return f"OpenAI Error: {e}"

    def generate_json(self, system_prompt, user_prompt):
        # OpenAI hỗ trợ JSON mode native
        if "json" not in system_prompt.lower():
            system_prompt += " \nIMPORTANT: Output strictly in JSON format."
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        self.last_usage = self._extract_usage(response)
        return json.loads(response.choices[0].message.content)

    def generate_text_stream(self, system_prompt, user_prompt, on_token=None):
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=1024,
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
        except (ValueError, RuntimeError, OSError) as e:
            raise RuntimeError(f"OpenAI streaming error: {e}")