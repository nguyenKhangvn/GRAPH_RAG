"""Answer Mode Router — determines HOW to answer based on question_type and query signals.

Separates answer_mode (output format/strategy) from intent (retrieval strategy).
Injected after Step 1, consumed by Step 5 dispatch.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional
from graph_rag.core import keywords
from graph_rag.core.keywords import TIME_RANGE_KEYWORDS, TRANSPORT_NEGATIVE_SIGNALS
from graph_rag.core.intents import IntentType
from graph_rag.utils.text import normalize_text


class AnswerMode:
    # Closed-form deterministic
    FILL_BLANK_SHORT = "fill_blank_short"
    TRUE_FALSE_VERIFIER = "true_false_verifier"
    SINGLE_OPTION_RESOLVER = "single_option_resolver"
    MULTI_OPTION_RESOLVER = "multi_option_resolver"
    NEGATIVE_ABSTAIN_GUARD = "negative_abstain_guard"

    # Structured answers
    DISTANCE = "distance"
    TOUR_PLAN = "tour_plan"
    TOUR_LIST = "tour_list"
    DISCOVERY_LIST = "discovery_list"
    AIRPORT_INFO = "airport_info"  # Airport/flight info from TravelInfo

    # Open-ended
    OPEN_ANALYSIS = "open_analysis"
    FACT_ANSWER = "fact_answer"
    PARTIAL_FACT_ANSWER = "partial_fact_answer"
    CURATED_RECOMMENDATION = "curated_recommendation"

    CLOSED_FORM_MODES = {
        FILL_BLANK_SHORT,
        TRUE_FALSE_VERIFIER,
        SINGLE_OPTION_RESOLVER,
        MULTI_OPTION_RESOLVER,
        NEGATIVE_ABSTAIN_GUARD,
    }

    @classmethod
    def is_closed_form(cls, mode: str) -> bool:
        return mode in cls.CLOSED_FORM_MODES


# ---------------------------------------------------------------------------
# Negative-text signal words that must NOT trigger distance mode
# ---------------------------------------------------------------------------
_ANALYSIS_NEGATIVE_SIGNALS = keywords.ANALYSIS_SIGNALS

# Distance positive signals — only these should trigger distance mode
_DISTANCE_SIGNALS = keywords.DISTANCE_SIGNALS

# Tour plan positive signals
_TOUR_PLAN_SIGNALS = keywords.TOUR_PLAN_SIGNALS



def _is_tour_availability_query(question: str) -> bool:
    """Return True when the user asks 'what tours exist?' rather than 'build me an itinerary'."""
    q = normalize_text(question)
    availability_markers = [
        "co tour nao", "co tour", "tour nao", "tour gi",
        "goi tour nao", "goi tour gi", "ben nao cung cap tour",
        "hien tai co tour", "co goi tour", "tour nao phu hop",
        "tour nao hay", "tour nao dep", "tim tour",
    ]
    has_availability = any(sig in q for sig in availability_markers)
    if not has_availability:
        return False
    # Exclude only when the user clearly commands us to build an itinerary.
    # "co tour nao cung cap lich trinh..." is still an availability/list query.
    build_patterns = [
        r"\b(lap|xay dung|thiet ke|tao|goi y)\s+(?:mot\s+)?(lich trinh|lo trinh|ke hoach)\b",
        r"\b(lich trinh|lo trinh|ke hoach)\s+(?:giup|minh|cho toi|cho minh)\b",
    ]
    if any(re.search(pattern, q) for pattern in build_patterns):
        return False
    return True


# Airport-specific signals for AIRPORT_INFO answer mode
_AIRPORT_SIGNALS = {
    "san bay", "san bay pleiku", "san bay phu cat",
    "pxu", "uih", "may bay", "ve may bay", "chuyen bay",
    "bay den", "bay tu", "ha noi", "ho chi minh",
    "tan son nhat", "noi bai", "hang hang khong",
}


def _is_airport_query(question: str) -> bool:
    """Return True when the question is specifically about airport/flight info."""
    q = normalize_text(question)
    return any(sig in q for sig in _AIRPORT_SIGNALS)


def _is_real_distance_query(question: str) -> bool:
    """Return True only when the question genuinely asks for distance/route."""
    q = normalize_text(question)
    # Reject transport/airport questions early
    if any(sig in q for sig in TRANSPORT_NEGATIVE_SIGNALS):
        return False
    # Must have a positive distance signal
    has_signal = any(sig in q for sig in _DISTANCE_SIGNALS)
    if not has_signal:
        # Check regex pattern: "tu X toi/den Y"
        m = re.search(r"\btu\s+.+\s+(toi|den)\s+.+", q)
        if m:
            # Reject if the matched segment contains time-range keywords
            segment = m.group(0)
            if any(kw in segment for kw in TIME_RANGE_KEYWORDS):
                has_signal = False
            else:
                has_signal = True
    if not has_signal:
        return False
    # Reject if analysis signals dominate
    has_analysis = any(sig in q for sig in _ANALYSIS_NEGATIVE_SIGNALS)
    if has_analysis:
        return False
    # Reject if tour plan signals present — "tu X den Y" is transit, not distance
    has_tour_plan = any(sig in q for sig in _TOUR_PLAN_SIGNALS)
    if not has_tour_plan:
        has_tour_plan = bool(re.search(r"\b\d+\s*(?:ngay|nay|ngy)\b", q)) or bool(re.search(r"\b\d+\s*n\s*\d+\s*d\b", q))
    if has_tour_plan:
        return False
    return True


def _is_real_tour_plan_query(question: str) -> bool:
    """Return True only when the question requests an itinerary/plan."""
    q = normalize_text(question)
    if any(sig in q for sig in _ANALYSIS_NEGATIVE_SIGNALS):
        return False
    # Must have explicit tour plan signal
    has_signal = any(sig in q for sig in _TOUR_PLAN_SIGNALS)
    if not has_signal:
        # Check duration pattern: "X ngay Y dem"
        has_signal = bool(re.search(r"\b\d+\s*(?:ngay|nay|ngy)\b", q)) or bool(re.search(r"\b\d+\s*n\s*\d+\s*d\b", q))
    if not has_signal:
        # Additional itinerary signals: feasibility + time arrangement
        _ITINERARY_EXTRA_SIGNALS = [
            "sap xep thoi gian", "thoi gian hop ly",
            "co kha thi khong", "kha thi",
            "trong mot ngay", "trong 1 ngay",
            "lich trinh 1 ngay", "lich trinh mot ngay",
        ]
        has_signal = any(sig in q for sig in _ITINERARY_EXTRA_SIGNALS)
    if not has_signal:
        # Pattern: "đi A và B trong ... ngày" (multi-destination + duration)
        if re.search(r"\bdi\s+\w+\s+va\s+\w+.*ngay\b", q):
            has_signal = True
    return has_signal


def _has_choice_markers(question: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Detect choices embedded in the user text, e.g. A) ..., B) ... or JSON format."""
    text = str(question or "")
    # Inline choices: A. ..., B) ..., etc.
    if re.search(r"(?im)^\s*[A-D]\s*[\).:-]\s+\S+", text):
        return True
    # JSON-format choices: "A. ..." or 'A. ...' in JSON arrays
    if re.search(r'["\']\s*[A-D]\s*[.)]\s*[^"\']+["\']', text):
        return True
    # Choices passed via metadata (from API request)
    if metadata and (metadata.get("choices") or metadata.get("question_type") in {"Multi-Choice", "Multi-Select"}):
        return True
    return False


def _is_multi_select_query(question: str) -> bool:
    q = normalize_text(question)
    return any(
        signal in q
        for signal in [
            "nhung cai nao",
            "nhung dia diem nao",
            "cac dia diem nao",
            "chon cac",
            "chon nhung",
            "dau la cac",
            "cai nao duoi day",
            "nhung lua chon nao",
        ]
    )


def _starts_with_analysis_command(question: str) -> bool:
    q = normalize_text(question).lstrip()
    command_prefixes = [
        "phan tich",
        "hay phan tich",
        "giai thich",
        "hay giai thich",
        "danh gia",
        "hay danh gia",
        "so sanh",
        "hay so sanh",
        "nhan xet",
        "hay nhan xet",
        "tong hop",
        "hay tong hop",
    ]
    return any(q.startswith(prefix) for prefix in command_prefixes)


def _looks_like_true_false_statement(question: str) -> bool:
    """Heuristic for production/eval statements without explicit 'đúng/sai' wording."""
    text = str(question or "").strip()
    q = normalize_text(text)
    if not text or "?" in text or _has_choice_markers(text):
        return False
    question_markers = [
        " nao",
        " nhung ",
        " co nhung",
        " bao gom co nhung",
        " gom co nhung",
        " o dau",
        " bao nhieu",
        " nhu the nao",
        " the nao",
        " khach san nao",
        " khach san khac khong",
    ]
    if any(marker in f" {q} " for marker in question_markers):
        return False
    if re.search(r"\bnhung\b.+\bnao\b", q):
        return False
    if re.search(r"\bbao\s+gom\s+co\s+nhung\b", q):
        return False
    # Open-ended analysis prompts are often imperative sentences without "?";
    # keep them out of the true/false route.
    if _starts_with_analysis_command(text) or _is_real_distance_query(text):
        return False
    if any(token in q for token in ["phan tich", "giai thich", "tai sao", "danh gia", "so sanh"]):
        return False
    # A verification statement can mention "tour" or "moi quan he" without
    # asking for an itinerary/analysis. These patterns are assertions.
    statement_patterns = [
        r"\bla\s+mot\b",
        r"\bco\s+moi\s+quan\s+he\b",
        r"\bmoi\s+quan\s+he\b.+\bla\b",
        r"\bduoc\s+(to\s+chuc|dat|tham\s+quan|cong\s+nhan|xep\s+hang)\b",
        r"\bnam\s+(tai|trong|o|gan)\b",
        r"\bthuoc\s+(loai|danh\s+muc|khu\s+vuc)\b",
        r"\bto\s+chuc\s+tai\b",
        r"\bbao\s+gom\b",
    ]
    return any(re.search(pattern, q) for pattern in statement_patterns)


def _is_analysis_query(question: str) -> bool:
    """Return True when the question asks for analysis/strategy/evaluation."""
    q = normalize_text(question)
    analysis_signals = [
        "phan tich",
        "vi tri chien luoc",
        "moi quan he",
        "giai thich",
        "tai sao",
        "loi the",
        "tiem nang",
        "danh gia",
        "so sanh",
        "tong hop",
        "nhan xet",
    ]
    return any(sig in q for sig in analysis_signals)


def _is_discovery_list_query(question: str, metadata: Dict[str, Any]) -> bool:
    """Return True when the query is a topic-based discovery/list request."""
    # Check query_frame operator first (highest confidence)
    qf = metadata.get("query_frame") or {}
    if qf.get("query_operator") == "global_discovery":
        return True
    q = normalize_text(question)
    discovery_markers = [
        "nen di", "nen den", "nen tham quan",
        "tim hieu ve", "tim hieu",
        "liet ke", "nhung dia diem nao",
        "dia diem nao", "diem du lich nao",
        "khong the bo qua",
        # NOTE: "co nhung gi" / "co nhung" removed — too broad, matches
        # attribute questions like "Cù Lao Xanh có những hoạt động gì?"
    ]
    if any(m in q for m in discovery_markers):
        return True

    # Event/festival/cultural discovery: "lễ hội", "sự kiện văn hóa", "festival"
    event_discovery_markers = [
        "le hoi", "su kien van hoa", "su kien", "festival",
        "le hoi va su kien", "van hoa", "hoat dong van hoa",
    ]
    if any(m in q for m in event_discovery_markers):
        return True

    return False


def infer_answer_mode(
    question: str,
    question_type: str = "",
    intent: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Determine answer_mode from question_type, intent, and query signals.

    Priority: question_type > query signals > intent fallback.
    This ensures closed-form types NEVER fall through to open-ended modes.
    """
    meta = metadata or {}
    q_norm = normalize_text(question)

    # --- Priority 1: question_type (from eval pipeline or inferred) ---
    qt = str(question_type or "").strip()
    qt_norm = qt.lower()

    if qt == "Fill-in-Blank":
        return AnswerMode.FILL_BLANK_SHORT

    if qt == "True-or-False":
        return AnswerMode.TRUE_FALSE_VERIFIER

    if qt == "Multi-Choice":
        return AnswerMode.SINGLE_OPTION_RESOLVER

    if qt == "Multi-Select":
        return AnswerMode.MULTI_OPTION_RESOLVER

    if qt_norm in {"negative", "negative-sample", "negative_sample"}:
        return AnswerMode.NEGATIVE_ABSTAIN_GUARD

    # --- Priority 2: query signals (for Open-Ended and untyped questions) ---
    if _has_choice_markers(question, meta):
        return AnswerMode.MULTI_OPTION_RESOLVER if _is_multi_select_query(question) else AnswerMode.SINGLE_OPTION_RESOLVER

    if _looks_like_true_false_statement(question):
        return AnswerMode.TRUE_FALSE_VERIFIER

    if _is_real_tour_plan_query(question):
        return AnswerMode.TOUR_PLAN

    if _is_tour_availability_query(question):
        return AnswerMode.TOUR_LIST

    if _is_airport_query(question):
        return AnswerMode.AIRPORT_INFO

    if _is_real_distance_query(question):
        return AnswerMode.DISTANCE

    if _is_analysis_query(question):
        return AnswerMode.OPEN_ANALYSIS

    # Discovery list: recommendation markers or global_discovery operator
    # Skip generic discovery markers when intent is a specific recommendation
    # (FOOD, ACCOMMODATION, EVENT) — "nên đi ăn hải sản" is not a generic discovery.
    _SPECIFIC_RECOMMENDATION_INTENTS = {
        IntentType.FOOD, IntentType.ACCOMMODATION, IntentType.EVENT, IntentType.TOURISM,
    }
    if _is_discovery_list_query(question, meta):
        if intent not in _SPECIFIC_RECOMMENDATION_INTENTS:
            # Guard: when concrete entities are extracted, the query is about
            # those entities (e.g. "Cù Lao Xanh có những hoạt động gì?"),
            # NOT a broad discovery. Only trust query_frame operator override.
            _has_concrete_entities = any(
                str(e.get("type") or "").strip() in {"TouristAttraction", "Restaurant", "Accommodation", "Event", "Tour"}
                for e in (meta.get("entities") or meta.get("resolved_entities") or [])
                if isinstance(e, dict)
            )
            if _has_concrete_entities:
                # Only allow high-confidence query_frame override, not generic markers
                qf = meta.get("query_frame") or {}
                if qf.get("query_operator") == "global_discovery":
                    return AnswerMode.DISCOVERY_LIST
                # Otherwise skip — let it fall through to FACT_ANSWER
            else:
                return AnswerMode.DISCOVERY_LIST

    # --- Priority 3: intent fallback ---
    if intent == IntentType.DISTANCE:
        # Double-check: only if query really looks like distance
        if _is_real_distance_query(question):
            return AnswerMode.DISTANCE

    if intent == IntentType.TOUR_PLAN:
        if _is_real_tour_plan_query(question):
            return AnswerMode.TOUR_PLAN

    # Discovery/list queries → DISCOVERY_LIST (not FACT_ANSWER)
    if intent == IntentType.DISCOVERY:
        return AnswerMode.DISCOVERY_LIST

    # Recommendation intents (EVENT, FOOD, ACCOMMODATION, TOURISM) with
    # location/region target → DISCOVERY_LIST (user wants a list, not a fact).
    # Example: "Lễ hội và sự kiện văn hóa ở Gia Lai?" → list of events
    # Example: "Nhà hàng nào ngon ở Quy Nhon?" → list of restaurants
    _RECOMMENDATION_INTENTS = {
        IntentType.EVENT, IntentType.FOOD,
        IntentType.ACCOMMODATION, IntentType.TOURISM,
    }
    if intent in _RECOMMENDATION_INTENTS:
        # Check if target is a broad location (not a specific entity)
        _entities = meta.get("entities") or meta.get("resolved_entities") or []
        _has_specific_entity = any(
            str(e.get("type") or "").strip() in {
                "TouristAttraction", "Restaurant", "Accommodation",
                "Event", "Tour", "Dish", "Specialty",
            }
            for e in _entities if isinstance(e, dict)
        )
        if not _has_specific_entity:
            return AnswerMode.DISCOVERY_LIST

    return AnswerMode.FACT_ANSWER
