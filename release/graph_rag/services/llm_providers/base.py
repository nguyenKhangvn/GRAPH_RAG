# graph_rag/service/llm_providers/base.py
from abc import ABC, abstractmethod
from typing import Optional, Callable


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_text(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> str:
        pass

    @abstractmethod
    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        pass

    def generate_text_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Default: gọi generate_text(), KHÔNG gọi on_token.
        Subclass override để stream thật từng token."""
        return self.generate_text(system_prompt, user_prompt)