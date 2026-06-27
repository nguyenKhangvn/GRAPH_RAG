from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Sequence


@dataclass
class RerankResult:
    texts: List[str]
    debug: dict[str, Any] = field(default_factory=dict)


class CrossEncoderTextualReranker:
    """Optional cross-encoder reranker for textual context.

    The class is deliberately fail-open. If the dependency/model cannot be
    loaded, or scoring fails, the caller receives the original MMR output.
    """

    _model_cache: dict[str, Any] = {}

    def __init__(
        self,
        enabled: bool,
        model_name: str,
        top_n: int,
        timeout_sec: float,
        scorer: Callable[[str, Sequence[str]], Sequence[float]] | None = None,
    ):
        self.enabled = bool(enabled)
        self.model_name = str(model_name or "").strip()
        self.top_n = max(1, int(top_n or 1))
        self.timeout_sec = max(0.1, float(timeout_sec or 0.1))
        self._scorer = scorer

    def rerank(self, query_text: str, texts: Sequence[str]) -> RerankResult:
        started = time.time()
        original = [str(text or "").strip() for text in texts or [] if str(text or "").strip()]
        debug: dict[str, Any] = {
            "reranker_enabled": self.enabled,
            "reranker_applied": False,
            "reranker_model": self.model_name,
            "reranker_input_count": len(original),
            "reranker_output_count": len(original),
            "reranker_latency_ms": 0.0,
            "reranker_error": "",
        }
        if not self.enabled:
            debug["reranker_error"] = "disabled"
            return RerankResult(texts=original, debug=debug)
        if not query_text or len(original) <= 1:
            debug["reranker_error"] = "insufficient_input"
            return RerankResult(texts=original, debug=debug)

        candidates = original[: self.top_n]
        tail = original[self.top_n :]
        try:
            scores = list(self._score(query_text, candidates))
            if len(scores) != len(candidates):
                raise ValueError("score_count_mismatch")
            ranked = sorted(
                zip(scores, range(len(candidates)), candidates),
                key=lambda row: (float(row[0]), -row[1]),
                reverse=True,
            )
            reranked = [text for _score, _idx, text in ranked] + tail
            debug.update(
                {
                    "reranker_applied": True,
                    "reranker_output_count": len(reranked),
                    "reranker_latency_ms": round((time.time() - started) * 1000, 2),
                    "reranker_scores": [
                        round(float(score), 4) for score, _idx, _text in ranked[: min(5, len(ranked))]
                    ],
                }
            )
            if (time.time() - started) > self.timeout_sec:
                debug["reranker_applied"] = False
                debug["reranker_error"] = "timeout_after_scoring"
                return RerankResult(texts=original, debug=debug)
            return RerankResult(texts=reranked, debug=debug)
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            debug["reranker_latency_ms"] = round((time.time() - started) * 1000, 2)
            debug["reranker_error"] = str(exc)[:200]
            return RerankResult(texts=original, debug=debug)

    def _score(self, query_text: str, texts: Sequence[str]) -> Sequence[float]:
        if self._scorer is not None:
            return self._scorer(query_text, texts)
        model = self._load_model()
        pairs = [(query_text, text) for text in texts]
        return model.predict(pairs)

    def _load_model(self) -> Any:
        if not self.model_name:
            raise RuntimeError("missing_model_name")
        cached = self._model_cache.get(self.model_name)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import CrossEncoder
        except (ImportError, ValueError, OSError, RuntimeError) as exc:
            raise RuntimeError(f"missing_sentence_transformers: {exc}") from exc
        model = CrossEncoder(self.model_name)
        self._model_cache[self.model_name] = model
        return model
