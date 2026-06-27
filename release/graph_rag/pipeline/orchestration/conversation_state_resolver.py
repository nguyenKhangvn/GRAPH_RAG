from __future__ import annotations
"""ConversationStateResolver — Resolve follow-up query context from conversation state.

Solves the problem of follow-up queries losing context between turns:
- region_focus drifts when current_location (GPS) overrides inherited location
- target_class (e.g. Dish) is forgotten, so retrieval returns wrong label types
- answered_entities exclusion only works on exact pattern match
- search_query is too vague for vector search

Usage:
    resolver = ConversationStateResolver()
    resolved = resolver.resolve(
        current_query=state.user_query,
        metadata=metadata,
        conversation_state=p.location_grounding_service.conversation_state,
    )
    # resolved.effective_location, resolved.region_focus, etc.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.utils.text import normalize_text


# Domain-switch keywords: if the follow-up query contains these,
# the user is switching topic → don't inherit target_class.
_DOMAIN_SWITCH_MARKERS = {
    "khach san": "Accommodation",
    "nha nghi": "Accommodation",
    "homestay": "Accommodation",
    "resort": "Accommodation",
    "villa": "Accommodation",
    "le hoi": "Event",
    "su kien": "Event",
    "tour ": "Tour",
    "lich trinh": "Tour",
    "lo trinh": "Tour",
    "may bay": "Transport",
    "xe buyt": "Transport",
    "thue xe": "Transport",
}

# "Other/more" markers for follow-up exclusion
_OTHER_MARKERS = [
    "con", "khac", "nua", "them",
]

# Complex signal markers: queries with these words need LLM re-analysis
_COMPLEX_SIGNAL_MARKERS = {
    "o dau", "nhu the nao", "bao nhieu", "the nao",
    "tai sao", "vi sao", "so sanh", "khac biet",
}


@dataclass
class ResolvedQueryFrame:
    """Resolved context for follow-up queries.

    Produced by ConversationStateResolver. Downstream modules read from this
    instead of re-detecting intent/location/target_class from scratch.
    """

    raw_query: str
    is_follow_up: bool = False

    # Inherited or detected values
    intent: str = ""
    target_class: Optional[str] = None
    effective_location: str = ""
    region_focus: str = "all"
    answer_mode: str = ""
    semantic_category: str = ""

    # Exclusion
    exclude_entities: List[str] = field(default_factory=list)
    previous_entities: List[str] = field(default_factory=list)  # raw names for logging

    # Search enhancement
    enhanced_search_query: Optional[str] = None

    # Debug
    inheritance_source: str = ""  # "inherited" | "detected" | "mixed"


class ConversationStateResolver:
    """Resolves follow-up query context from conversation state.

    Runs early in Step 1, before intent/location override logic.
    Produces a ResolvedQueryFrame that downstream modules read from.

    Priority:
    1. Explicit location/intent in current query → always wins (handled by existing Step 1 logic)
    2. Inherited from previous turn → for follow-ups without new signals
    3. Detected from current query → fallback
    """

    def resolve(
        self,
        current_query: str,
        metadata: Dict[str, Any],
        conversation_state: Dict[str, Any],
    ) -> ResolvedQueryFrame:
        """Resolve follow-up context from conversation state.

        Args:
            current_query: Raw user query text
            metadata: Analyzer output metadata (already populated by LLM)
            conversation_state: Pipeline's persistent conversation state dict

        Returns:
            ResolvedQueryFrame with inherited/detected context
        """
        query_norm = normalize_text(current_query, strip_punct=True)
        is_follow_up = bool(metadata.get("is_follow_up", False))

        # Quick exit: not a follow-up
        if not is_follow_up:
            return ResolvedQueryFrame(
                raw_query=current_query,
                is_follow_up=False,
                inheritance_source="detected",
            )

        # --- Read previous turn state ---
        last_intent = conversation_state.get("last_intent") or ""
        last_target_class = conversation_state.get("last_target_class") or ""
        last_answer_mode = conversation_state.get("last_answer_mode") or ""
        last_region_focus = conversation_state.get("last_region_focus") or "all"
        last_semantic_category = conversation_state.get("last_semantic_category") or ""
        # Fallback: try current_location if last_active_location not set
        last_location = (
            conversation_state.get("last_active_location")
            or conversation_state.get("current_location")
            or ""
        )
        prev_answered = list(conversation_state.get("previously_answered_entities") or [])

        # --- Detect domain switch ---
        detected_switch_class = self._detect_domain_switch(query_norm)
        if detected_switch_class:
            # User is switching topic — don't inherit target_class/intent/semantic_category
            return ResolvedQueryFrame(
                raw_query=current_query,
                is_follow_up=True,
                intent=metadata.get("intent") or last_intent,
                target_class=detected_switch_class,
                effective_location=last_location,
                region_focus=last_region_focus,
                semantic_category="",
                exclude_entities=[],
                previous_entities=prev_answered,
                inheritance_source="mixed",
            )

        # --- Check if current query has explicit location ---
        has_explicit_location = bool(metadata.get("has_explicit_location"))
        explicit_loc = metadata.get("detected_location") or ""

        # --- Resolve effective location ---
        if has_explicit_location and explicit_loc:
            effective_location = explicit_loc
        elif last_location:
            effective_location = last_location
        else:
            effective_location = ""

        # --- Resolve region_focus ---
        if has_explicit_location:
            # Explicit location → let Step 1 detect region_focus normally
            region_focus = metadata.get("region_focus") or "all"
        elif last_region_focus and last_region_focus != "all":
            region_focus = last_region_focus
        else:
            region_focus = metadata.get("region_focus") or "all"

        # --- Resolve intent ---
        current_intent = metadata.get("intent") or ""
        if current_intent:
            intent = current_intent
        elif last_intent:
            intent = last_intent
        else:
            intent = "DISCOVERY"

        # --- Resolve target_class ---
        current_target_class = metadata.get("target_class") or ""
        if current_target_class:
            target_class = current_target_class
        elif last_target_class:
            target_class = last_target_class
        else:
            target_class = None

        # --- Resolve answer_mode ---
        current_answer_mode = metadata.get("answer_mode") or ""
        if current_answer_mode:
            answer_mode = current_answer_mode
        elif last_answer_mode:
            answer_mode = last_answer_mode
        else:
            answer_mode = ""

        # --- Resolve semantic_category ---
        current_semantic_category = metadata.get("semantic_category") or ""
        if current_semantic_category:
            semantic_category = current_semantic_category
        elif last_semantic_category:
            semantic_category = last_semantic_category
        else:
            semantic_category = ""

        # --- Build exclude entities ---
        # NOTE: This normalize call is intentionally kept separate from ExclusionContext.
        # ConversationStateResolver runs in Step 1 (before Step 4) and populates
        # resolved.exclude_entities as a fallback source for ExclusionContext in Step 4.
        # The duplication is acceptable — two entry points feeding one consolidation point.
        exclude_entities = []
        has_other_marker = self._has_other_marker(query_norm)
        if has_other_marker and prev_answered:
            exclude_entities = [normalize_text(n, strip_punct=True) for n in prev_answered if n]

        source = "inherited" if (last_location or last_intent) else "detected"

        return ResolvedQueryFrame(
            raw_query=current_query,
            is_follow_up=True,
            intent=intent,
            target_class=target_class,
            effective_location=effective_location,
            region_focus=region_focus,
            answer_mode=answer_mode,
            semantic_category=semantic_category,
            exclude_entities=exclude_entities,
            previous_entities=prev_answered,
            inheritance_source=source,
        )

    @staticmethod
    def _detect_domain_switch(query_norm: str) -> Optional[str]:
        """Detect if user is switching to a different domain.

        Returns the new target_class if domain switch detected, else None.
        """
        for marker, target_class in _DOMAIN_SWITCH_MARKERS.items():
            if marker in query_norm:
                return target_class
        return None

    @staticmethod
    def _has_other_marker(query_norm: str) -> bool:
        """Check if query asks for 'other/more' items."""
        for marker in _OTHER_MARKERS:
            if marker in query_norm:
                return True
        return False

    @staticmethod
    def _has_new_entity_signal(query_norm: str, metadata: Dict[str, Any]) -> bool:
        """Detect if query introduces signals that require LLM re-analysis.

        Returns True if the query has new entities, domain switch, explicit location,
        or complex question signals. Returns False for pure "other/more" patterns.
        """
        # a. Non-empty entities list in metadata
        if metadata.get("entities"):
            return True

        # b. Domain switch detected
        if ConversationStateResolver._detect_domain_switch(query_norm) is not None:
            return True

        # c. Explicit location detected in metadata
        if metadata.get("has_explicit_location"):
            return True

        # d. Complex question signals
        for marker in _COMPLEX_SIGNAL_MARKERS:
            if marker in query_norm:
                return True

        return False

    @staticmethod
    def _build_follow_up_metadata(query: str, conversation_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Synthesize analyzer-equivalent metadata from conversation_state for simple follow-ups.

        Returns None if conversation_state has no last_intent (no previous turn context).
        Returns a metadata dict matching analyzer._normalize_output return shape otherwise.
        """
        last_intent = conversation_state.get("last_intent")
        if not last_intent:
            return None

        query_norm = normalize_text(query, strip_punct=True)

        return {
            "intents": [last_intent],
            "rewritten_query": query,
            "entities": [],
            "resolved_entities": [],
            "has_explicit_location": False,
            "detected_location": (
                conversation_state.get("last_active_location")
                or conversation_state.get("current_location")
                or None
            ),
            "search_keywords": [query],
            "constraints": {"optimize_distance": False},
            "coreference_confidence": 0.9,
            "needs_clarification": False,
            "requested_attributes": [],
            "requested_relations": [],
            "is_follow_up": True,
            "dialog_act": "REQUEST_MORE",
            # Extra keys for downstream use
            "_target_class": conversation_state.get("last_target_class") or "",
            "_semantic_category": conversation_state.get("last_semantic_category") or "",
            "_answer_mode": conversation_state.get("last_answer_mode") or "",
            "_region_focus": conversation_state.get("last_region_focus") or "all",
            "_fast_path": True,
        }
