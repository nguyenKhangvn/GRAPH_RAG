"""ExclusionContext: single-pass exclusion state for follow-up deduplication.

Consolidates the fragmented exclusion/previous-entities contract into one
dataclass.  Built once in Step 4, consumed by PolicyRanker,
PipelineResponseMixin, and Step5GenerationMixin.

Normalization happens exactly once at build time — downstream consumers
receive pre-normalized entity names.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Set

from graph_rag.utils.text import normalize_text


@dataclass
class ExclusionContext:
    """Carries exclusion state through the pipeline.

    Attributes:
        entity_names: Normalized entity names to exclude from results.
        should_force_deterministic: Whether downstream should force
            deterministic rendering (insufficient context for LLM).
    """

    entity_names: Set[str] = field(default_factory=set)
    should_force_deterministic: bool = False

    @property
    def has_entities(self) -> bool:
        """Return True when there are entities to exclude."""
        return len(self.entity_names) > 0

    @classmethod
    def build_from_conversation_state(
        cls,
        conversation_state: Dict[str, Any],
        is_follow_up: bool,
        raw_context_len: int,
        threshold: int,
    ) -> "ExclusionContext":
        """Build an ExclusionContext from conversation state.

        Args:
            conversation_state: Pipeline conversation state dict
                (must contain ``previously_answered_entities`` key).
            is_follow_up: Whether the current query is a follow-up.
            raw_context_len: Number of raw context facts available.
            threshold: Fact-count threshold below which deterministic
                rendering is forced.

        Returns:
            ExclusionContext with normalized entity names and
            ``should_force_deterministic`` flag.
        """
        if not is_follow_up:
            return cls(entity_names=set(), should_force_deterministic=False)

        # should_force_deterministic depends on context length, not entity presence
        should_force_deterministic = raw_context_len <= threshold

        prev_entities = conversation_state.get("previously_answered_entities") or []
        if not prev_entities:
            return cls(entity_names=set(), should_force_deterministic=should_force_deterministic)

        # Single normalize_text call for the entire pipeline
        entity_names = {normalize_text(n, strip_punct=True) for n in prev_entities if n}

        return cls(
            entity_names=entity_names,
            should_force_deterministic=should_force_deterministic,
        )
