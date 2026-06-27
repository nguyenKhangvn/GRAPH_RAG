from __future__ import annotations

import dataclasses
import logging
import re
import time

_logger = logging.getLogger(__name__)
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from graph_rag.config import (
    QUERY_FRAME_MIN_CONFIDENCE,
)
from graph_rag.config import cfg as _cfg
from graph_rag.config.deictic_patterns import (
    DEICTIC_QUERY_PATTERNS,
    is_deictic_query,
    is_deictic_entity_phrase,
)
from graph_rag.core.intents import IntentType
from graph_rag.core.state import QueryOperation
from graph_rag.core.answer_mode import AnswerMode
from graph_rag.utils.text import normalize_text
from .dto import PipelineRunState
from .query_frame_stage import QueryFrameStage
from .relation_verification_mixin import RelationVerificationMixin
from .query_pattern_mixin import QueryPatternMixin
from .deterministic_answer_mixin import DeterministicAnswerMixin
from .entity_processor_mixin import EntityProcessorMixin
from .answer_validators_mixin import AnswerValidatorsMixin
from .context_processor_mixin import ContextProcessorMixin
from .comparison_engine_mixin import ComparisonEngineMixin
from .step_dispatch import StepDispatchMixin
from .closed_form_dispatch_mixin import ClosedFormDispatchMixin

if TYPE_CHECKING:
    from graph_rag.pipeline.graph_rag_pipeline import RAGPipeline


class PipelineApplicationService(
    EntityProcessorMixin,
    AnswerValidatorsMixin,
    ContextProcessorMixin,
    ComparisonEngineMixin,
    StepDispatchMixin,
    ClosedFormDispatchMixin,
    RelationVerificationMixin,
    QueryPatternMixin,
    DeterministicAnswerMixin,
):
    # DEICTIC_QUERY_PATTERNS imported from graph_rag.config.deictic_patterns
    # Xem file đó để debug/thêm pattern mới
    DEICTIC_QUERY_PATTERNS = DEICTIC_QUERY_PATTERNS

    REQUESTED_ATTRIBUTE_LABELS = _cfg.requested_attribute_labels()
    REQUESTED_ATTRIBUTE_QUERY_HINTS = _cfg.requested_attribute_query_hints()
    ANALYTICAL_LOCATION_HINTS = _cfg.analytical_location_hints()
    DISTANCE_TAIL_PATTERNS = [
        r"\s+la\s+bao\s+nhieu\s+km\s*$",
        r"\s+là\s+bao\s+nhiêu\s+km\s*$",
        r"\s+la\s+bao\s+nhieu\s*$",
        r"\s+là\s+bao\s+nhiêu\s*$",
        r"\s+bao\s+nhieu\s+km\s*$",
        r"\s+bao\s+nhiêu\s+km\s*$",
        r"\s+bao\s+nhieu\s*$",
        r"\s+bao\s+nhiêu\s*$",
        r"\s+bao\s+xa\s*$",
        r"\s+bao\s+xa\s*$",
    ]

    def __init__(self, pipeline: "RAGPipeline"):
        self.pipeline = pipeline
        self.query_frame_stage = QueryFrameStage(
            normalizer=lambda t: normalize_text(t, strip_punct=True),
            min_confidence=QUERY_FRAME_MIN_CONFIDENCE,
        )

    # ------------------------------------------------------------------
    # Conflict Resolver + Dispatch Table (Phase 5, Plan 05-01)
    # ------------------------------------------------------------------

    # QueryOperation → forced answer_mode mapping
    _OPERATION_MODE_OVERRIDES: Dict[QueryOperation, str] = {
        QueryOperation.ATTRIBUTE_LOOKUP: AnswerMode.FACT_ANSWER,
        QueryOperation.AVAILABILITY_SEARCH: AnswerMode.TOUR_LIST,
        QueryOperation.ITINERARY_BUILD: AnswerMode.TOUR_PLAN,
        QueryOperation.COMPARISON: "comparison",
        QueryOperation.CONSTRAINED_NEARBY: AnswerMode.TOUR_LIST,
        QueryOperation.FACT_VERIFY: AnswerMode.FACT_ANSWER,
        QueryOperation.DISCOVERY: AnswerMode.DISCOVERY_LIST,
    }

    def _resolve_answer_mode(self, plan: "QueryPlan") -> str:
        """Resolve answer_mode conflicts based on QueryOperation.

        Operation takes precedence over LLM-predicted answer_mode.
        This is the single source of truth for routing decisions.
        """
        op = plan.operation
        current_mode = plan.answer_mode

        forced = self._OPERATION_MODE_OVERRIDES.get(op)
        if forced and forced != current_mode:
            # Don't force discovery_list for FOOD/ACCOMMODATION intents —
            # LLM synthesis produces richer answers than deterministic templates.
            if forced == AnswerMode.DISCOVERY_LIST and plan.intent:
                intent_upper = plan.intent.upper()
                if "FOOD" in intent_upper or "ACCOMMODATION" in intent_upper:
                    _logger.info(
                        "[ConflictResolver] op=%s would force %s, but intent=%s → keeping %s for LLM synthesis",
                        op.value, forced, plan.intent, current_mode,
                    )
                    return current_mode
            _logger.info(
                "[ConflictResolver] op=%s overriding %s -> %s",
                op.value, current_mode, forced,
            )
            return forced

        return current_mode

    def _update_conversation_state_from_result(
        self, state: "PipelineRunState", result: Dict[str, Any]
    ) -> None:
        """Single point for conversation_state updates after short-circuit handlers."""
        p = self.pipeline
        p._update_conversation_state(
            history=state.history,
            user_query=state.user_query,
            answer=result.get("answer") or "",
            location=(result.get("metadata") or {}).get("detected_location") or state.location,
            entities=state.entities,
            last_grounded_anchor=(state.metadata or {}).get("last_grounded_anchor")
            or (state.metadata or {}).get("active_grounded_anchor"),
        )

    def _try_handler(
        self,
        state: "PipelineRunState",
        handler: Callable[[], Optional[Dict[str, Any]]],
        label: str,
        request_id: str,
        start_time: float,
        step_timings: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        """Execute a short-circuit handler; update state + log if it fires."""
        result = handler()
        if result is not None:
            self._update_conversation_state_from_result(state, result)
            _logger.info("%s request_id=%s", label, request_id)
            step_timings["total"] = int((time.time() - start_time) * 1000)
        return result

    # --- Individual short-circuit handlers ---

    def _handle_website_lookup(
        self, state: "PipelineRunState", answer_mode: str
    ) -> Optional[Dict[str, Any]]:
        if AnswerMode.is_closed_form(answer_mode):
            return None
        return self._answer_website_lookup_if_possible(state)

    def _handle_tour_plan(
        self, state: "PipelineRunState", answer_mode: str
    ) -> Optional[Dict[str, Any]]:
        if answer_mode != AnswerMode.TOUR_PLAN:
            return None
        return self._answer_strict_tour_itinerary_if_possible(state)

    def _handle_address_lookup(
        self, state: "PipelineRunState", answer_mode: str
    ) -> Optional[Dict[str, Any]]:
        if AnswerMode.is_closed_form(answer_mode):
            return None
        multi_entity_guard = (
            len(state.grounded_nodes or []) > 1
            or ((state.metadata or {}).get("query_frame") or {}).get("query_operator") == "comparison"
            or str((state.metadata or {}).get("intent") or "") in ("comparison", "DISCOVERY")
        )
        if multi_entity_guard:
            return None
        return self._answer_address_lookup_if_possible(state)

    def _handle_nearby_accommodation(
        self, state: "PipelineRunState", answer_mode: str
    ) -> Optional[Dict[str, Any]]:
        if AnswerMode.is_closed_form(answer_mode):
            return None
        return self._answer_nearby_accommodation_if_possible(state)

    def _handle_constrained_nearby(
        self, state: "PipelineRunState", answer_mode: str
    ) -> Optional[Dict[str, Any]]:
        if AnswerMode.is_closed_form(answer_mode):
            return None
        return self._answer_constrained_nearby_search_if_possible(state)

    def _handle_nearby_cultural(
        self, state: "PipelineRunState", answer_mode: str
    ) -> Optional[Dict[str, Any]]:
        if AnswerMode.is_closed_form(answer_mode):
            return None
        skip = bool(
            (
                (state.metadata or {}).get("retrieval_plan_mode") == "tour_plan"
                and (state.metadata or {}).get("query_frame_traversal_intent")
            )
            or (state.metadata or {}).get("retrieval_plan_mode") == "comparison"
            or ((state.metadata or {}).get("query_frame") or {}).get("query_operator") == "comparison"
            or (state.metadata or {}).get("retrieval_plan_mode") == "constrained_nearby_search"
        )
        if skip:
            return None
        return self._answer_nearby_cultural_categories_if_possible(state)

    def _handle_distance(
        self, state: "PipelineRunState", answer_mode: str, request_id: str
    ) -> Optional[Dict[str, Any]]:
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        if answer_mode != AnswerMode.DISTANCE and intent != IntentType.DISTANCE:
            return None
        _logger.info("distance_short_circuit request_id=%s", request_id)
        p = self.pipeline
        distance_source_location = (
            state.runtime.metadata.get("user_gps")
            or state.runtime.metadata.get("user_provided_location")
            or state.location
        )
        preflight_result = p.distance_intent_service.preflight_followup_direction(
            user_query=state.user_query,
            metadata=state.metadata,
            entities=state.entities,
            detected_location=distance_source_location,
            conversation_state=p.location_grounding_service.conversation_state,
        )
        if preflight_result is not None:
            _logger.info("distance_preflight_stopped request_id=%s", request_id)
            return preflight_result
        result = p._run_distance_intent(
            user_query=state.user_query,
            metadata=state.metadata,
            grounded_nodes=state.grounded_nodes,
            entities=state.entities,
            detected_location=distance_source_location,
        )
        _logger.info("distance_branch_completed request_id=%s", request_id)
        return result

    def _handle_shared_location(
        self, state: "PipelineRunState", answer_mode: str
    ) -> Optional[Dict[str, Any]]:
        return self._answer_shared_location_fill_blank_if_possible(state)

    def _dispatch_short_circuits(
        self,
        state: "PipelineRunState",
        answer_mode: str,
        request_id: str,
        start_time: float,
        step_timings: Dict[str, int],
    ) -> Optional[Dict[str, Any]]:
        """Dispatch table replacing the 9 duplicate short-circuits.

        Handlers are tried in priority order. Each handler is a method that
        returns a result dict if it fires, or None to fall through.
        _update_conversation_state() is called exactly ONCE per short-circuit
        via _try_handler().
        """
        # Ordered handler list: (handler_fn, log_label)
        handlers: list[tuple[Callable[[], Optional[Dict[str, Any]]], str]] = [
            (lambda: self._handle_website_lookup(state, answer_mode), "website_lookup_short_circuit"),
            (lambda: self._handle_tour_plan(state, answer_mode), "strict_tour_short_circuit"),
            (lambda: self._handle_address_lookup(state, answer_mode), "address_lookup_short_circuit"),
            (lambda: self._handle_nearby_accommodation(state, answer_mode), "nearby_accommodation_short_circuit"),
            (lambda: self._handle_constrained_nearby(state, answer_mode), "constrained_nearby_short_circuit"),
            (lambda: self._handle_nearby_cultural(state, answer_mode), "nearby_cultural_short_circuit"),
            (lambda: self._handle_distance(state, answer_mode, request_id), "distance_short_circuit"),
            (lambda: self._handle_shared_location(state, answer_mode), "shared_location_short_circuit"),
        ]

        for handler_fn, label in handlers:
            result = self._try_handler(state, handler_fn, label, request_id, start_time, step_timings)
            if result is not None:
                return result

        return None

    def _elapsed(self, step_start: float) -> str:
        return f"{time.time() - step_start:.2f}s"

    def _preview_list(self, values: List[Any], limit: int = 8) -> str:
        if not values:
            return "[]"
        shown = values[:limit]
        suffix = " ..." if len(values) > limit else ""
        return f"{shown}{suffix}"

    def _count_bulleted_lines(self, text: str) -> int:
        if not text:
            return 0
        return sum(1 for line in text.splitlines() if line.strip().startswith("- "))

    def _canonicalize_entity_name(self, name: str) -> str:
        text = str(name or "").strip()
        if not text:
            return ""

        cleaned = re.sub(r"_{2,}", "", text).strip(" ,.;:!?")
        cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip(" ,.;:!?")
        prefix_patterns = [
            r"^(?:nhà\s+hàng|nha\s+hang)\s+",
            r"^(?:quán|quan)\s+",
            r"^(?:khách\s+sạn|khach\s+san)\s+",
            r"^(?:nhà\s+nghỉ|nha\s+nghi)\s+",
            r"^(?:khu\s+du\s+lịch|khu\s+du\s+lich)\s+",
        ]
        for pattern in prefix_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" ,.;:!?")

        suffix_patterns = [
            r"\s+(?:được\s+đặt|duoc\s+dat|đặt|dat)\s+(?:tại|tai|ở|o)\b.*$",
            r"\s+(?:tọa\s+lạc|toa\s+lac)\s+(?:tại|tai|ở|o)\b.*$",
            r"\s+(?:n\u1eb1m|nam)\s+(?:t\u1ea1i|tai|\u1edf|o)\b.*$",
            r"\s+(?:thu\u1ed9c|thuoc)\b.*$",
            r"\s+(?:theo)\b.*$",
            r"\s+(?:\u1edf|o)\b.*$",
            r"\s+(?:đối\s+với|doi\s+voi).*$",
            r"\s+(?:và|va)\s+(?:du\s+lich|thanh\s+pho).*$",
            r"\s+(?:hãy|har)\s+.*$",
            r"\s+(?:t\u1ea1i|tai)\b.*$",
        ]
        for _ in range(2):
            previous = cleaned
            for pattern in suffix_patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" ,.;:!?")
            if cleaned == previous:
                break
        cleaned = self._strip_entity_tail_noise(cleaned).strip(" ,.;:!?")
        # Phase 3: Truncate overly long entity names (>15 words likely malformed)
        words = cleaned.split()
        if len(words) > 15:
            # Try truncating at first comma
            comma_idx = cleaned.find(',')
            if comma_idx > 10:
                cleaned = cleaned[:comma_idx].strip()
            else:
                # Keep first 12 words
                cleaned = ' '.join(words[:12]).strip()
        return cleaned or text

    def _canonicalize_entities_for_grounding(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        canonicalized: List[Dict[str, Any]] = []
        for entity in entities or []:
            if not isinstance(entity, dict):
                canonicalized.append(entity)
                continue
            raw_name = str(entity.get("name") or "").strip()
            raw_norm = normalize_text(raw_name, strip_punct=True)
            e_type = str(entity.get("type") or "").strip().lower()
            is_lodging = e_type in {"accommodation", "hotel", "lodging"} or raw_norm.startswith(
                ("nha nghi ", "khach san ", "homestay ", "resort ")
            )
            is_short_named_quan = (
                e_type in {"restaurant", "food", "eatery"}
                and raw_norm.startswith("quan ")
                and len([part for part in raw_norm.split() if part]) <= 3
            )
            is_named_establishment = raw_norm.startswith(
                ("quan ", "nha hang ", "nha nghi ", "khach san ", "homestay ", "resort ", "tour ")
            )
            preserved_name = re.sub(r"\s*\([^)]*\)\s*$", "", raw_name).strip(" ,.;:!?")
            canonicalized.append({
                **entity,
                "name": preserved_name if (is_lodging or is_short_named_quan or is_named_establishment) else self._canonicalize_entity_name(raw_name),
            })
        return canonicalized

    def _infer_entity_type_from_hint(self, hint: str) -> str:
        norm = normalize_text(hint, strip_punct=True)
        if norm.startswith(("quan ", "nha hang ")):
            return "Restaurant"
        if norm.startswith(("nha nghi ", "khach san ", "homestay ", "resort ")):
            return "Accommodation"
        return "Place"

    def _is_deictic_reference_query(self, query: str) -> bool:
        normalized = normalize_text(query, strip_punct=True)
        if not normalized:
            return False
        return is_deictic_query(normalized)

    def _is_deictic_entity_phrase(self, entity_name: str) -> bool:
        """Kiểm tra entity name có phải deictic phrase (cấm semantic search)."""
        normalized = normalize_text(entity_name or "", strip_punct=True)
        return is_deictic_entity_phrase(normalized)

    def _build_grounded_anchor(self, node: Any, location_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if not node:
            return {}

        metadata = getattr(node, "metadata", {}) or {}
        labels = metadata.get("labels") or []
        anchor_location = ""
        if isinstance(location_context, dict):
            anchor_location = str(location_context.get("name") or "").strip()

        return {
            "id": str(getattr(node, "id", "") or ""),
            "name": str(metadata.get("name") or getattr(node, "content", "") or "").strip(),
            "labels": [str(label) for label in labels if str(label).strip()],
            "address": str(metadata.get("address") or "").strip(),
            "lat": metadata.get("lat"),
            "lng": metadata.get("lng"),
            "location": anchor_location or str(metadata.get("address") or "").strip(),
        }

    def _build_anchor_node(self, anchor: Dict[str, Any]) -> Any | None:
        if not isinstance(anchor, dict):
            return None
        name = str(anchor.get("name") or "").strip()
        if not name:
            return None
        metadata = {
            "name": name,
            "labels": [str(label) for label in (anchor.get("labels") or []) if str(label).strip()],
            "address": str(anchor.get("address") or anchor.get("location") or "").strip(),
            "lat": anchor.get("lat"),
            "lng": anchor.get("lng"),
        }
        return SimpleNamespace(
            id=str(anchor.get("id") or name),
            content=name,
            metadata=metadata,
        )

    def _is_groundable_entity(self, entity: Dict[str, Any] | None) -> bool:
        if not isinstance(entity, dict):
            return False
        e_type = str(entity.get("type") or "").strip().lower()
        if e_type in self.NON_GROUNDABLE_ENTITY_TYPES:
            return False
        e_name = str(entity.get("name") or "").strip()
        if not e_name:
            return False
        if self._is_generic_category_phrase(e_name):
            return False
        # Ignore pure numeric placeholders (e.g., group size "2").
        if e_name.isdigit():
            return False
        return True

    def _is_generic_category_phrase(self, text: str) -> bool:
        norm = normalize_text(text, strip_punct=True)
        if not norm:
            return False

        # Exact generic category words
        generic_words = {
            "khach san", "nha nghi", "homestay", "resort", "dia diem du lich",
            "diem du lich", "dia diem tham quan", "diem tham quan", "mon an dac san",
            "mon dac san", "dac san", "quan an", "nha hang", "le hoi", "su kien",
            "tour", "mon an", "mon ngon", "dac san dia phuong", "am thuc", "cac khach san",
            "cac nha nghi", "cac homestay", "cac resort", "nhung khach san", "nhung nha nghi",
            "nhung homestay", "nhung resort", "cac dia diem", "nhung dia diem", "cac nha hang",
            "nhung nha hang", "cac quan an", "nhung quan an", "cac tour", "nhung tour",
            "tour nao", "tour gi", "le hoi nao"
        }
        if norm in generic_words:
            return True

        generic_patterns = [
            r"\bkhach\s+san\s+(?:nao|khac\s+khong|trung\s+tam|phu\s+hop|gan\s+day)\b",
            r"\bnha\s+nghi\s+(?:nao|khac\s+khong|trung\s+tam|phu\s+hop|gan\s+day)\b",
            r"\bhomestay\s+(?:nao|khac\s+khong|trung\s+tam|phu\s+hop|gan\s+day)\b",
            r"\b(?:khach\s+san|nha\s+nghi|homestay)\s+.+\bbao\s+gom\b",
            r"\b(?:khach\s+san|nha\s+nghi|homestay)\s+.+\bco\s+nhung\b",
            r"\b(?:khach\s+san|nha\s+nghi|homestay)\s+.+\bnhung\b.+\bnao\b",
            r"\bkhach\s+san\s+khac\s+khong\b",
            r"\b(?:dia\s+diem|diem)\s+du\s+lich\s+(?:nao|gan\s+day)\b",
            r"\bmon\s+an\s+dac\s+san\s+(?:nao|gi)\b",
        ]
        return any(re.search(pattern, norm) for pattern in generic_patterns)

    def _is_category_listing_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        if not q:
            return False
        category_terms = ["khach san", "nha nghi", "homestay", "resort", "luu tru", "nghi dem"]
        listing_terms = ["nhung", "nao", "khac khong", "them", "bao gom", "danh sach", "liet ke"]
        return any(term in q for term in category_terms) and any(term in q for term in listing_terms)

    def _score_option_against_context(self, option_text: str, context_text: str) -> int:
        option_norm = normalize_text(option_text, strip_punct=True)
        context_norm = normalize_text(context_text, strip_punct=True)
        if not option_norm or not context_norm:
            return 0
        if option_norm in context_norm:
            return 100
        score = 0
        fragments = re.split(r"[,;]|(?:\s+v\S*\s+)|(?:\s+va\s+)|(?:\s+g\S*n\s+)|(?:\s+gan\s+)", option_text)
        for fragment in fragments:
            fragment_norm = normalize_text(fragment, strip_punct=True)
            if len(fragment_norm) >= 4 and fragment_norm in context_norm:
                score += 12
        tokens = self._content_tokens(option_text, min_len=3)
        if tokens:
            context_tokens = self._content_tokens(context_text, min_len=3)
            overlap = tokens & context_tokens
            score += len(overlap) * 3
            if len(overlap) / len(tokens) >= 0.8:
                score += 10
        return score

    def _resolve_true_false_from_context(self, state: PipelineRunState, context_text: str) -> str | None:
        query_norm = normalize_text(state.user_query, strip_punct=True)
        context_norm = normalize_text(context_text, strip_punct=True)
        if not query_norm or not context_norm:
            return None

        main_entity = self._primary_specific_entity_name(state)
        direct_context = "\n".join(self._direct_context_lines(state, main_entity)) or context_text
        direct_norm = normalize_text(direct_context, strip_punct=True)
        negative_claim = self._question_has_negative_claim(state.user_query)

        object_supported = False
        for pattern in [
            r"\b(?:la|thuoc|nam tai|nam o|o|tai|gan|bao gom|to chuc tai)\s+(.+?)(?:\.|$)",
            r"\b(?:voi|ve)\s+(.+?)(?:\.|$)",
        ]:
            for match in re.finditer(pattern, query_norm):
                phrase = match.group(1).strip(" ,.;:!?")
                phrase_tokens = self._content_tokens(phrase)
                if phrase_tokens and len(phrase_tokens & self._content_tokens(direct_norm)) / len(phrase_tokens) >= 0.75:
                    object_supported = True
                    break
            if object_supported:
                break

        overlap = self._context_overlap_score(state.user_query, direct_norm)
        entity_supported = True
        if main_entity:
            entity_supported = self._retrieval_evidence_contains_entity(
                main_entity, state.all_seeds or [], state.raw_context or []
            )
        relation_supported = any(marker in direct_norm for marker in [
            "located_in", "belongs_to", "near", "has", "held_at", "includes", "offers",
            "address", "description", "type", "month", "activities",
            "dia chi", "nam tai", "nam o", "thuoc", "gan",
        ])

        if entity_supported and relation_supported and (object_supported or overlap >= 0.25):
            state.metadata["true_false_deterministic_score"] = overlap
            if negative_claim:
                return "Sai. Dữ liệu ngữ cảnh có fact liên quan, nên mệnh đề phủ định không được hỗ trợ."
            return "Đúng. Dữ liệu ngữ cảnh có fact trực tiếp hỗ trợ mệnh đề này."

        if negative_claim and entity_supported and relation_supported and overlap < 0.35:
            state.metadata["true_false_deterministic_score"] = overlap
            return "Đúng. Dữ liệu ngữ cảnh không có fact hỗ trợ nội dung bị phủ định trong mệnh đề."

        return None

    def _choice_content_tokens(self, text: str, min_len: int = 3) -> set[str]:
        norm = normalize_text(text or "", strip_punct=True)
        return {
            token
            for token in re.findall(r"\w+", norm)
            if len(token) >= min_len and token not in self._OPTION_FRAGMENT_STOPWORDS
        }

    def _option_fragments(self, option_text: str) -> list[str]:
        text = re.sub(r"^\s*[A-D]\s*[\).:-]\s*", "", str(option_text or "").strip(), flags=re.IGNORECASE)
        text = re.sub(
            r"(?i)\b(?:la|co|thuoc|nam|duoc|dua tren|nha hang nay|cong trinh nay|dia diem nay)\b",
            " ",
            text,
        )
        parts = re.split(r"[,;]|(?:\s+va\s+)|(?:\s+v\S*\s+)|(?:\s+bao\s+gom\s+)|(?:\s+gom\s+)", text)
        fragments = []
        for part in parts:
            fragment = part.strip(" .;:,()")
            if fragment and self._choice_content_tokens(fragment):
                fragments.append(fragment)
        return fragments

    def _fragment_supported_by_context(self, fragment_norm: str, context_norm: str, targets: set[str]) -> bool:
        if not fragment_norm:
            return False
        if fragment_norm in context_norm:
            return True
        if any(fragment_norm == target or fragment_norm in target or target in fragment_norm for target in targets):
            return True

        if "di tich" in fragment_norm:
            category_aliases = {
                "di tich lich su van hoa": ["di tich lich su van hoa", "lich su van hoa"],
                "di tich kien truc co": ["di tich kien truc co"],
                "di tich kien truc nghe thuat": ["di tich kien truc nghe thuat", "kien truc nghe thuat"],
                "di tich khao co hoc": ["di tich khao co", "khao co hoc"],
                "di tich danh thang": ["di tich danh thang", "danh thang"],
            }
            for key, aliases in category_aliases.items():
                if key in fragment_norm or fragment_norm in key:
                    return any(alias in context_norm for alias in aliases)
            return False

        if "phuong " in fragment_norm:
            return fragment_norm in context_norm or any(fragment_norm in target for target in targets)

        strict_place_fragments = [
            "nha tho go", "ho tay", "nui ham rong", "bai bien nha trang",
            "cho pleiku", "san bay pleiku", "ben xe gia lai",
            "nha hat", "san van dong", "trung tam mua sam", "cho dem",
            "khu bao ton thien nhien", "den tho",
        ]
        if any(marker in fragment_norm for marker in strict_place_fragments):
            return fragment_norm in context_norm or any(fragment_norm in target for target in targets)

        strict_named_markers = [
            "bai xep", "hon kho", "eo gio", "flc", "kayak", "thuyen kayak",
            "bai dua", "ky co", "bien ky co",
        ]
        present_markers = [marker for marker in strict_named_markers if marker in fragment_norm]
        if present_markers and not all(marker in context_norm for marker in present_markers):
            return False

        if "cong ty" in fragment_norm:
            company_aliases = {
                "cong ty co phan quy nhon tourist": ["quy nhon tourist", "cong ty co phan quy nhon tourist"],
                "cong ty du lich binh dinh": ["cong ty du lich binh dinh"],
                "cong ty du lich flc": ["cong ty du lich flc", "flc"],
            }
            for key, values in company_aliases.items():
                if key in fragment_norm or fragment_norm in key:
                    return any(value in context_norm for value in values)
            return False

        fragment_tokens = {
            token for token in re.findall(r"\w+", fragment_norm)
            if len(token) >= 3 and token not in self._OPTION_FRAGMENT_STOPWORDS
        }
        if fragment_tokens:
            context_tokens = set(re.findall(r"\w+", context_norm))
            if len(fragment_tokens & context_tokens) / len(fragment_tokens) >= 0.67:
                return True
            for target in targets:
                target_tokens = set(re.findall(r"\w+", target))
                if target_tokens and len(fragment_tokens & target_tokens) / len(fragment_tokens) >= 0.67:
                    return True

        semantic_aliases = {
            "bao tang": ["bao tang"],
            "chua chien": ["chua"],
            "chua": ["chua"],
            "lang nghe thu cong": ["lang nghe", "det tho cam", "nhac cu dan toc", "non ngua"],
            "lang nghe": ["lang nghe", "det tho cam", "nhac cu dan toc", "non ngua"],
            "cong vien": ["cong vien"],
            "canh quan bien": ["bien", "ghenh", "hai dang", "eo gio", "ky co", "quy hoa"],
            "di tich lich su van hoa": ["lich su van hoa", "di tich", "tuong dai", "den tho"],
            "di tich lich su": ["lich su", "di tich", "tuong dai", "den tho"],
            "di tich kien truc nghe thuat": ["kien truc nghe thuat"],
            "lang nghe truyen thong": ["lang nghe", "nghe truyen thong"],
            "quy nhon tourist": ["quy nhon tourist", "cong ty co phan quy nhon tourist"],
            "lan ngam san ho": ["lan ngam san ho", "lan san ho"],
            "bai dua": ["bai dua"],
            "ky co": ["ky co"],
            "trung op la": ["trung op la", "op la"],
            "bo ne": ["bo ne"],
        }
        aliases: list[str] = []
        for key, values in semantic_aliases.items():
            if fragment_norm == key or fragment_norm in key or key in fragment_norm:
                aliases.extend(values)
        return any(alias in context_norm or any(alias in target for target in targets) for alias in aliases)

    def _option_category_compatible(self, question: str, option_text: str) -> bool:
        question = re.split(r"(?im)^\s*A\s*[\).:-]\s+", str(question or ""), maxsplit=1)[0]
        q_raw = str(question or "").lower()
        opt_raw = str(option_text or "").lower()
        q = normalize_text(question, strip_punct=True)
        opt = normalize_text(option_text, strip_punct=True)
        if ("van hoa" in q or "văn hóa" in q_raw) and ("tam linh" in q or "tâm linh" in q_raw):
            positive = [
                "bao tang", "chua", "lang nghe", "det tho cam", "nha tho",
                "di tich", "den", "thap", "bao tang tinh gia lai",
            ]
            negative = ["cong vien", "san van dong", "trung tam mua sam", "nha hat"]
            if any(marker in opt for marker in negative) and not any(marker in opt for marker in positive):
                return False
            return any(marker in opt for marker in positive)
        if "canh quan bien" in q or "cảnh quan biển" in q_raw or (("di tich lich su" in q or "di tích lịch sử" in q_raw) and ("bien" in q or "biển" in q_raw)):
            positive = ["bien", "ghenh", "hai dang", "eo gio", "ky co", "quy hoa", "tuong", "di tich", "den", "thap"]
            negative = ["cho", "cong vien", "trung tam mua sam"]
            if any(marker in opt for marker in negative) and not any(marker in opt for marker in positive):
                return False
            return any(marker in opt for marker in positive)
        if "di tich lich su" in q or "di tích lịch sử" in q_raw:
            return any(marker in opt for marker in ["di tich", "thap", "den", "tay son", "khao co", "lich su", "tuong"])
        if ("van hoa" in q or "văn hóa" in q_raw) and any(marker in opt for marker in ["trung tam mua sam", "san van dong"]):
            return False
        return True

    def _direct_option_scores(
        self,
        state: PipelineRunState,
        choices: list[tuple[str, str]],
        context_text: str,
    ) -> list[tuple[str, str, int, float]]:
        targets = self._relation_targets_from_context(context_text)
        context_norm = normalize_text(context_text, strip_punct=True)
        question_norm = normalize_text(state.user_query, strip_punct=True)
        asks_nearby_reason = any(token in question_norm for token in [
            "diem dung chan thuan tien",
            "tham quan",
            "gan ca hai",
            "gan cac dia diem",
        ])
        near_target_hits = 0
        if asks_nearby_reason:
            for name in (state.metadata or {}).get("evidence_names") or []:
                name_norm = normalize_text(name, strip_punct=True)
                if name_norm and name_norm in question_norm and name_norm in context_norm:
                    near_target_hits += 1
            for target in targets:
                if target and target in question_norm and re.search(r"\bnear\b|\bnam gan\b", context_norm):
                    near_target_hits += 1
        scored: list[tuple[str, str, int, float]] = []
        for letter, text in choices:
            fragments = self._option_fragments(text)
            if not fragments:
                scored.append((letter, text, 0, 0.0))
                continue
            matched = 0
            for fragment in fragments:
                frag_norm = normalize_text(fragment, strip_punct=True)
                if self._fragment_supported_by_context(frag_norm, context_norm, targets):
                    matched += 1
            ratio = matched / len(fragments)
            option_tokens = self._choice_content_tokens(text)
            context_tokens = self._choice_content_tokens(context_text)
            token_ratio = (len(option_tokens & context_tokens) / len(option_tokens)) if option_tokens else 0.0
            score = matched * 100 - (len(fragments) - matched) * 35
            option_norm = normalize_text(text, strip_punct=True)
            option_digits = set(re.findall(r"\d+", option_norm))
            if option_digits:
                context_digits = set(re.findall(r"\d+", context_norm))
                if option_digits.issubset(context_digits):
                    score += 45
                    ratio = max(ratio, 0.75)
                else:
                    score -= 160
                    ratio = min(ratio, 0.25)
                    token_ratio = min(token_ratio, 0.25)
            if asks_nearby_reason:
                if any(token in option_norm for token in ["gan ca hai", "gan hai dia diem", "gan cac dia diem", "nam gan ca hai"]):
                    score += 220 if near_target_hits >= 2 else 80
                    ratio = max(ratio, 1.0 if near_target_hits >= 2 else 0.75)
                    token_ratio = max(token_ratio, ratio)
                elif any(token in option_norm for token in ["dia chi chinh xac", "toa do", "wgs84", "loai hinh"]):
                    score -= 90
            if not self._option_category_compatible(state.user_query, text):
                score -= 120
            scored.append((letter, text, score, max(ratio, token_ratio)))
        return scored

    def _resolve_options_from_context(self, state: PipelineRunState, multi: bool = False) -> str | None:
        choices = self._choice_lines_from_state(state)
        if not choices:
            return None
        context_text = self._closed_form_context_text(state)
        if not context_text.strip():
            return None

        type_answer = self._resolve_type_option_from_context(state, choices, context_text)
        if type_answer:
            return type_answer

        frame = (state.metadata or {}).get("query_frame") or {}
        plan = frame.get("retrieval_plan") or {}
        dish_constraints = [
            str(item or "").strip()
            for item in ((plan.get("context_policy") or {}).get("dish_constraints") or [])
            if str(item or "").strip()
        ]
        if dish_constraints and "HAS" in (plan.get("required_relations") or []):
            context_norm = normalize_text(context_text, strip_punct=True)
            matched_letters = []
            for letter, text in choices:
                option_norm = normalize_text(text, strip_punct=True)
                if not option_norm or option_norm.startswith(("ca hai", "tat ca", "khong ")):
                    continue
                for dish in dish_constraints:
                    dish_norm = normalize_text(dish, strip_punct=True)
                    if (
                        option_norm in context_norm
                        and dish_norm in context_norm
                        and any(
                            marker in context_norm
                            for marker in [
                                f"{option_norm} phuc vu mon {dish_norm}",
                                f"{dish_norm} phuc vu mon {option_norm}",
                                f"{option_norm} has {dish_norm}",
                                f"{dish_norm} has {option_norm}",
                            ]
                        )
                    ):
                        matched_letters.append((letter, text, dish))
                        break
            if matched_letters:
                state.runtime.metadata["option_resolver_relation_match"] = {
                    "relation": "HAS",
                    "dish_constraints": dish_constraints,
                    "matched": [
                        {"letter": letter, "text": text, "dish": dish}
                        for letter, text, dish in matched_letters
                    ],
                }
                if multi:
                    return ", ".join(letter for letter, _, _ in matched_letters)
                letter, text, dish = matched_letters[0]
                return f"{letter}: {text}."

        scored = self._direct_option_scores(state, choices, context_text)
        state.runtime.metadata["option_scores"] = [
            {"letter": letter, "score": score, "match_ratio": round(ratio, 3)}
            for letter, _, score, ratio in scored
        ]

        if self._is_negative_option_question(state.user_query):
            ranked_low = sorted(scored, key=lambda item: item[2])
            best_negative = ranked_low[0]
            second_score = ranked_low[1][2] if len(ranked_low) > 1 else 999
            if best_negative[2] < 45 and second_score - best_negative[2] >= 25:
                state.runtime.metadata["negative_option_eliminator"] = True
                return f"{best_negative[0]}. Du lieu ngu canh khong ho tro phuong an {best_negative[0]}: {best_negative[1]}."
            return None

        if multi:
            selected = [
                (letter, text, score, ratio)
                for letter, text, score, ratio in scored
                if score >= 45 and ratio >= 0.45
            ]
            if not selected:
                return None
            selected = sorted(selected, key=lambda item: item[0])
            letters = ", ".join(letter for letter, _, _, _ in selected)
            evidence = "; ".join(f"{letter}: {text}" for letter, text, _, _ in selected[:4])
            return f"{letters}. Du lieu ngu canh khop voi: {evidence}."

        ranked = sorted(scored, key=lambda item: item[2], reverse=True)
        best = ranked[0]
        second_score = ranked[1][2] if len(ranked) > 1 else 0
        if best[2] < 45 or best[3] < 0.35 or best[2] - second_score < 15:
            return None
        return f"{best[0]}: {best[1]}."

    def execute(
        self,
        user_query: str,
        chat_history: List[Dict] = None,
        current_location: str = "",
        user_gps: str = "",
        eval_metadata: Dict[str, Any] | None = None,
        on_token=None,
        request_id: str = "",
        step_timings: Dict[str, int] | None = None,
    ) -> Dict[str, Any]:
        p = self.pipeline
        start_time = time.time()
        if step_timings is None:
            step_timings: Dict[str, int] = {}
        normalized_query = p._normalize_known_location_typos(user_query)
        _logger.info("pipeline_start request_id=%s query='%s'", request_id, normalized_query)

        history = chat_history if chat_history is not None else (p.conversation_state.get("history") or [])
        _logger.debug(
            "pipeline_input request_id=%s history_turns=%d current_location='%s' query_len=%d",
            request_id, len(history), current_location or '', len(normalized_query or '')
        )

        state = PipelineRunState(user_query=normalized_query, history=history)
        if eval_metadata:
            state.runtime.metadata.update({k: v for k, v in eval_metadata.items() if v is not None})

        step_start = time.time()
        self._run_step_1_query_understanding(state, current_location)
        step_timings["query_understanding"] = int((time.time() - step_start) * 1000)
        _logger.info("step_complete step=query_understanding request_id=%s duration_ms=%d", request_id, step_timings["query_understanding"])

        # Preserve user's original location and GPS before grounding may override.
        state.runtime.metadata["user_provided_location"] = current_location or state.location
        if user_gps:
            state.runtime.metadata["user_gps"] = user_gps
        early_guard_result = self._early_scope_and_clarification_guard(state)
        if early_guard_result is not None:
            p._update_conversation_state(
                history=state.history,
                user_query=state.user_query,
                answer=early_guard_result.get("answer") or "",
                location=(early_guard_result.get("metadata") or {}).get("detected_location") or state.location,
                entities=state.entities,
                last_grounded_anchor=(state.metadata or {}).get("last_grounded_anchor") or (state.metadata or {}).get("active_grounded_anchor"),
            )
            _logger.info("early_guard reason=%s request_id=%s", (early_guard_result.get('metadata') or {}).get('early_guard'), request_id)
            step_timings["total"] = int((time.time() - start_time) * 1000)
            return early_guard_result
        step_start = time.time()
        self._run_step_2_grounding(state)
        step_timings["grounding"] = int((time.time() - step_start) * 1000)
        _logger.info("step_complete step=grounding request_id=%s duration_ms=%d", request_id, step_timings["grounding"])

        # Skip proximity anchor guard for constrained_nearby_search:
        # chain reasoning uses Cypher, not grounded anchors
        is_constrained_mode = (state.metadata or {}).get("retrieval_plan_mode") == "constrained_nearby_search"
        proximity_guard_result = (
            None if is_constrained_mode
            else self._proximity_anchor_grounding_guard(state)
        )
        if proximity_guard_result is not None:
            p._update_conversation_state(
                history=state.history,
                user_query=state.user_query,
                answer=proximity_guard_result.get("answer") or "",
                location=(proximity_guard_result.get("metadata") or {}).get("detected_location") or state.location,
                entities=state.entities,
                last_grounded_anchor=(state.metadata or {}).get("last_grounded_anchor") or (state.metadata or {}).get("active_grounded_anchor"),
            )
            _logger.info("proximity_anchor_guard request_id=%s", request_id)
            step_timings["total"] = int((time.time() - start_time) * 1000)
            return proximity_guard_result

        answer_mode = (state.metadata or {}).get("answer_mode", AnswerMode.FACT_ANSWER)

        # --- Phase 5: Conflict Resolver ---
        # Resolve answer_mode conflicts based on QueryOperation.
        # Operation takes precedence over LLM-predicted answer_mode.
        resolved_mode = self._resolve_answer_mode(state.query_plan)
        if resolved_mode != answer_mode:
            state.query_plan = dataclasses.replace(state.query_plan, answer_mode=resolved_mode)
            answer_mode = resolved_mode

        # [DEBUG] Trace state before short-circuit chain
        _logger.info(
            "[DEBUG-PRE-SHORTCIRCUIT] request_id=%s answer_mode=%s intent=%s grounded_nodes=%d location=%s region_focus=%s",
            request_id, answer_mode,
            (state.query_plan.intent if state.query_plan else "N/A"),
            len(state.grounded_nodes or []),
            state.location,
            state.region_focus,
        )

        # --- Phase 5: Dispatch table replaces 9 duplicate short-circuits ---
        dispatch_result = self._dispatch_short_circuits(
            state, answer_mode, request_id, start_time, step_timings,
        )
        if dispatch_result is not None:
            return dispatch_result

        step_start = time.time()
        self._run_step_3_query_expansion(state)
        step_timings["query_expansion"] = int((time.time() - step_start) * 1000)
        _logger.info("step_complete step=query_expansion request_id=%s duration_ms=%d", request_id, step_timings["query_expansion"])

        step_start = time.time()
        step_4_result = self._run_step_4_retrieve_and_prune(state)
        step_timings["retrieve_and_prune"] = int((time.time() - step_start) * 1000)
        _logger.info("step_complete step=retrieve_and_prune request_id=%s duration_ms=%d", request_id, step_timings["retrieve_and_prune"])

        if step_4_result.get("early_result"):
            # Update conversation state before early return to prevent state drift
            plan = state.query_plan
            intent = plan.intent if plan else state.primary_intent

            # Ensure intent is set in early_result metadata
            early_metadata = step_4_result["early_result"].get("metadata") or {}
            if not early_metadata.get("intent") and intent:
                early_metadata["intent"] = intent
                step_4_result["early_result"]["metadata"] = early_metadata
                _logger.info("   -> [DIAGNOSTIC-EARLY] Set intent='%s' in early_result metadata", intent)

            p._update_conversation_state(
                history=state.history,
                user_query=state.user_query,
                answer=(step_4_result["early_result"].get("answer") or ""),
                location=state.location,
                entities=state.entities,
                last_grounded_anchor=(state.metadata or {}).get("last_grounded_anchor") or (state.metadata or {}).get("active_grounded_anchor"),
                intent=intent,
                target_class=(plan.target_class if plan else "") or "",
                answer_mode=(plan.answer_mode if plan else "") or "",
                region_focus=state.region_focus or (plan.region_focus if plan else "") or "",
                semantic_category=(plan.semantic_category if plan else "") or "",
            )
            step_timings["total"] = int((time.time() - start_time) * 1000)
            return step_4_result["early_result"]

        step_start = time.time()
        self._run_step_5_generate_answer(state, on_token=on_token)
        step_timings["generate_answer"] = int((time.time() - step_start) * 1000)
        _logger.info("step_complete step=generate_answer request_id=%s duration_ms=%d", request_id, step_timings["generate_answer"])

        result = self._finalize_pipeline_response(state)

        # Extract plan fields for ConversationStateResolver (follow-up inheritance)
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        # Resolve region_focus from multiple sources (priority: state > metadata > plan)
        final_region_focus = (
            state.region_focus
            or (state.metadata or {}).get("region_focus")
            or (plan.region_focus if plan else "")
            or ""
        )
        p._update_conversation_state(
            history=state.history,
            user_query=state.user_query,
            answer=state.answer,
            location=state.location,
            entities=state.entities,
            last_grounded_anchor=(state.metadata or {}).get("last_grounded_anchor") or (state.metadata or {}).get("active_grounded_anchor"),
            intent=intent,
            target_class=(plan.target_class if plan else "") or "",
            answer_mode=(plan.answer_mode if plan else "") or "",
            region_focus=final_region_focus,
            semantic_category=(plan.semantic_category if plan else "") or "",
        )

        step_timings["total"] = int((time.time() - start_time) * 1000)
        _logger.info("pipeline_complete request_id=%s total_ms=%d intent=%s", request_id, step_timings["total"], intent)
        return result
