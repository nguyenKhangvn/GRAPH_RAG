"""Standalone query field inference — extracted from QueryState.from_metadata().

This module provides ``build_query_fields()`` which infers query attributes
(query_norm, question_shape, target_class, constraints, operation, etc.)
from raw metadata without depending on QueryState.

QueryPlanBuilder consumes the returned ``QueryFields`` dataclass.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.core.state import ConstraintSpec, QueryOperation, QuestionShape
from graph_rag.utils.duration import infer_duration as _infer_duration_util
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relation trigger table — normalized markers -> relation + label hints
# All markers are in normalized (no diacritics) form.
# ---------------------------------------------------------------------------

RELATION_TRIGGERS: Dict[str, Dict[str, Any]] = {
    "SPECIALTY_OF": {
        "markers": ["dac san", "mon ngon", "am thuc", "co gi ngon", "nen an gi", "mon gi"],
        "target_labels": ["Dish", "Specialty"],
        "support_labels": ["Restaurant", "Location"],
    },
    "HAS": {
        "markers": ["an o dau", "quan nao", "nha hang nao", "co ban o dau", "quan an"],
        "target_labels": ["Restaurant"],
        "support_labels": ["Dish"],
    },
    "NEAR": {
        "markers": ["gan", "gan day", "quanh", "cach", "ben canh", "o gan"],
        "target_labels": ["Restaurant", "Accommodation", "TouristAttraction"],
    },
    "LOCATED_IN": {
        "markers": ["o dau", "nam o", "thuoc", "dia chi", "vi tri"],
        "target_labels": ["Location", "TouristAttraction", "Accommodation", "Restaurant"],
    },
    "HELD_AT": {
        "markers": ["le hoi", "su kien", "dien ra", "to chuc"],
        "target_labels": ["Event"],
    },
    "Guide_for": {
        "markers": ["thong tin", "huong dan", "kinh nghiem", "can biet", "meo", "luu y",
                     "thoi tiet", "san bay", "taxi", "khan cap", "so dien thoai"],
        "target_labels": ["TravelInfo"],
        "support_labels": ["Location"],
    },
}


# ---------------------------------------------------------------------------
# QueryFields — plain dataclass holding all inferred values
# ---------------------------------------------------------------------------

@dataclass
class QueryFields:
    """All fields inferred from a user query + metadata.

    Attribute names match QueryState for drop-in compatibility with
    QueryPlanBuilder's getattr() calls.
    """
    query: str = ""
    query_norm: str = ""
    question_shape: QuestionShape = QuestionShape.UNKNOWN
    target_class: Optional[str] = None
    target_dish: Optional[str] = None
    target_entity: Optional[str] = None
    requested_attributes: List[str] = field(default_factory=list)
    requested_relations: List[str] = field(default_factory=list)
    matched_markers: List[str] = field(default_factory=list)
    is_follow_up: bool = False

    # Constraint-driven tour plan requirements
    constraints: List[ConstraintSpec] = field(default_factory=list)

    # Legacy booleans — derived from constraints for backward compat
    coastal_required: bool = False
    sunset_required: bool = False
    island_required: bool = False
    walking_required: bool = False
    low_mobility_required: bool = False
    family_friendly_required: bool = False
    budget_required: bool = False
    duration_days: int = 0
    duration_nights: int = 0

    # Operation: what the user wants to DO (separate from target_class)
    operation: QueryOperation = QueryOperation.DISCOVERY
    operation_source: Optional[str] = None
    operation_confidence: float = 0.0

    # Metadata and tracking fields
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source and confidence tracking for Phase 2 ranker/context
    target_class_source: Optional[str] = None
    target_class_confidence: float = 0.0
    target_dish_source: Optional[str] = None
    target_dish_confidence: float = 0.0
    question_shape_source: Optional[str] = None
    question_shape_confidence: float = 0.0

    # Semantic category: narrows retrieval within target_class
    semantic_category: Optional[str] = None
    semantic_category_confidence: float = 0.0


# ---------------------------------------------------------------------------
# Standalone inference functions (extracted from QueryState static methods)
# ---------------------------------------------------------------------------

def infer_semantic_category(query_norm: str) -> tuple:
    """Detect semantic category from query keywords.

    Returns (category, confidence) or (None, 0.0).
    Categories: cultural_village, heritage, natural_landmark, spiritual, craft, public_space.

    Order matters: check specific categories BEFORE broad ones
    to avoid false positives (e.g., "lang van hoa" matching "van hoa" -> heritage).
    """
    q = query_norm or ""

    # Cultural village markers — check FIRST (most specific)
    cultural_village_markers = [
        "lang van hoa", "lang dan toc", "van hoa ban dia",
        "van hoa dan toc", "cong dong dan toc", "nguoi dan toc",
        "van hoa cong dong", "hoat dong van hoa", "trai nghiem van hoa",
        "lang jrai", "lang bahnar", "lang jarai", "lang ba na",
        "nha rong", "cong chieng", "co chieng",
        "le hoi van hoa", "le hoi dan gian", "le hoi truyen thong",
    ]
    if any(m in q for m in cultural_village_markers):
        return "cultural_village", 0.9

    # Event markers — check BEFORE nature to avoid "le hoi ... bien" misclassification
    event_markers = ["le hoi", "su kien", "dien ra", "to chuc", "le ky niem"]
    if any(m in q for m in event_markers):
        return None, 0.0

    # Heritage markers
    heritage_markers = [
        "di tich", "lich su", "bao tang", "khao co", "di san",
        "cham pa", "champa", "thap cham", "van hoa cham",
        "co do", "kinh do", "hoang de", "vuong trieu",
    ]
    if any(m in q for m in heritage_markers):
        return "heritage", 0.85

    # Spiritual markers
    spiritual_markers = ["chua", "tinh xa", "nha tho", "tu vien", "tam linh"]
    if any(m in q for m in spiritual_markers):
        return "spiritual", 0.8

    # Craft markers — specific to artisan/craft work, NOT general ethnic culture
    craft_markers = ["lang nghe", "det", "tho cam", "nhac cu", "thu cong my nghe", "gom su", "ren"]
    if any(m in q for m in craft_markers):
        return "craft", 0.8

    # Food markers — check BEFORE nature to avoid false positives
    food_markers = [
        "mon an", "an gi", "quan an", "nha hang", "dac san", "am thuc",
        "banh xeo", "banh mi", "banh cuon", "banh canh", "pho", "bun",
        "com", "nem", "che", "lau", "nuong", "hai san", "tom nhay",
        "an sang", "an trua", "an toi", "an vat", "mon ngon",
    ]
    if any(m in q for m in food_markers):
        return None, 0.0

    # Natural landmark markers (high confidence — explicit nature phrases)
    nature_markers = [
        "tu nhien", "dia danh tu nhien", "thien nhien", "phong canh",
        "dia danh noi tieng", "dia diem tu nhien", "winh canh dep",
        "noi dep", "kham pha thien nhien",
    ]
    if any(m in q for m in nature_markers):
        return "natural_landmark", 0.9

    # Nature keywords (medium confidence — require word boundary)
    nature_kws = [
        "bien ", " bien", " ho ", "thac ", "thac_", "nui ", "suoi ",
        "khu du lich sinh thai", " dao ", " dam ", "eo gio", "nui lua",
        "ho nuoc", "bai bien", "hon dao", "rung nguyen sinh",
    ]
    if any(f" {kw.strip()} " in f" {q} " for kw in nature_kws):
        return "natural_landmark", 0.7

    return None, 0.0


def infer_target_class(query: str, meta: Dict[str, Any]) -> Optional[str]:
    """Infer target class from LLM analyzer intent.

    Primary source: LLM intent -> target_class mapping.
    Fallback: keyword markers for when LLM intent is missing/ambiguous.
    """
    q = normalize_text(query)

    # 1. LLM intent -> target_class (PRIMARY — highest confidence)
    intent = str(meta.get("intent") or "").upper()
    intents = list(dict.fromkeys(
        [intent] + [str(i).upper() for i in (meta.get("intents") or []) if i]
    ))

    for intent_item in intents:
        if "FOOD" in intent_item:
            if "nha hang" in q:
                return "Restaurant"
            has_standalone_quan = (
                q.startswith("quan ") or " quan " in q or q.endswith(" quan")
            ) and "tham quan" not in q
            if has_standalone_quan:
                return "Restaurant"
            return "Dish"
        elif "ACCOMMODATION" in intent_item:
            return "Accommodation"
        elif "EVENT" in intent_item:
            return "Event"
        elif "TOURISM" in intent_item:
            return "TouristAttraction"

    # 2. Keyword fallback (when LLM intent is missing or DISCOVERY_SEARCH)
    has_dac_san = "dac san" in q
    has_am_thuc = "am thuc" in q
    has_eating_place = "nha hang" in q or (
        ("quan" in q) and (q.startswith("quan ") or " quan " in q or q.endswith(" quan")) and "tham quan" not in q
    )
    if has_dac_san and not has_eating_place:
        return "Specialty"
    if has_am_thuc and not has_eating_place and not has_dac_san:
        return "Dish"

    food_terms = {
        "pho", "bun", "com", "banh", "nem", "hai san", "ca phe", "cafe",
        "lau", "nuong", "chao", "mi", "sinh to", "kem", "che", "oc",
        "banh canh", "banh cuon", "banh mi",
    }
    if has_eating_place:
        for term in food_terms:
            if term in q:
                return "Restaurant"

    accom_terms = {"khach san", "resort", "homestay", "nha nghi", "villa", "hotel"}
    if any(t in q for t in accom_terms):
        return "Accommodation"

    event_terms = {"le hoi", "su kien", "giai chay", "marathon", "cuoc thi"}
    if any(t in q for t in event_terms):
        return "Event"

    tour_terms = {"co tour", "tour nao", "tour gi", "goi tour", "tim tour"}
    if any(t in q for t in tour_terms):
        return "Tour"

    attraction_terms = {"dia diem", "tham quan", "check in", "danh lam", "thang canh"}
    if any(t in q for t in attraction_terms):
        return "TouristAttraction"

    for intent_item in intents:
        if "DISCOVERY" in intent_item:
            return "TouristAttraction"

    return None


def infer_target_dish(query: str, meta: Dict[str, Any]) -> Optional[str]:
    """Infer target dish from entities or query patterns."""

    _NOT_DISH_PHRASES = [
        "quan an", "nha hang", "dia diem", "loai hinh",
        "nao sau day", "nao sau", "sau day", "trong hai", "trong ba",
        "loai nao", "mon nao", "mon an nao",
    ]

    entities = meta.get("entities") or []
    for ent in entities:
        if isinstance(ent, dict):
            if str(ent.get("type")).strip().lower() == "dish":
                ent_name = str(ent.get("name") or "").strip()
                if not ent_name:
                    continue
                ent_norm = normalize_text(ent_name, strip_punct=True)
                if any(p in ent_norm for p in _NOT_DISH_PHRASES):
                    continue
                return ent_name

    target_category = meta.get("target_category")
    if target_category:
        cat_norm = normalize_text(str(target_category)).lower()
        if any(kw in cat_norm for kw in ["pho", "bun", "com", "banh", "nem", "hai san", "ca phe"]):
            return str(target_category)

    q = normalize_text(query).lower()

    DISH_MAP = {
        "pho kho": "pho kho",
        "banh canh": "banh canh",
        "banh cuon": "banh cuon",
        "banh my": "banh my",
        "banh mi": "banh mi",
        "pho": "pho",
        "bun": "bun",
        "com": "com",
        "banh": "banh",
        "nem": "nem",
        "hai san": "hai san",
        "ca phe": "ca phe",
        "cafe": "ca phe",
    }
    for pat, accented in DISH_MAP.items():
        if " " in pat and pat in q:
            return accented

    _FALSE_POSITIVE_CONTEXTS = {
        "pho": ["thanh pho", "pho dong", "pho tay", "pho bien", "pho co", "pho di bo", "pho phuong"],
        "com": ["com phai", "com bang"],
        "banh": ["banh rang", "banh tay"],
    }
    q_bounded = f" {q} "
    for pat, accented in DISH_MAP.items():
        if " " not in pat and f" {pat} " in q_bounded:
            false_positive = False
            for fp_ctx in _FALSE_POSITIVE_CONTEXTS.get(pat, []):
                if fp_ctx in q:
                    false_positive = True
                    break
            if not false_positive:
                return accented

    intent = meta.get("intent")
    v3_intent_data = meta.get("v3_intent_data")
    if v3_intent_data and intent and "FOOD" in str(intent).upper():
        if hasattr(v3_intent_data, "get"):
            anchors = v3_intent_data.get("anchors") or []
            for anchor in anchors:
                if isinstance(anchor, str):
                    anchor_norm = normalize_text(anchor)
                    for pat in DISH_MAP:
                        if pat in anchor_norm:
                            return anchor

    return None


def infer_requested_attributes(query: str, meta: Dict[str, Any]) -> List[str]:
    """Read requested_attributes from LLM analyzer output only."""
    return list(meta.get("requested_attributes") or [])


def infer_question_shape(query: str, meta: Dict[str, Any]) -> QuestionShape:
    """Infer question shape from LLM analyzer intent."""
    intent = str(meta.get("intent") or "").upper()

    _INTENT_TO_SHAPE = {
        "ENTITY_FACT_QUERY": QuestionShape.SINGLE_FACT,
        "DISTANCE_QUERY": QuestionShape.SINGLE_FACT,
        "FOOD_RECOMMENDATION": QuestionShape.RECOMMENDATION_LIST,
        "TOURISM_RECOMMENDATION": QuestionShape.RECOMMENDATION_LIST,
        "ACCOMMODATION_RECOMMENDATION": QuestionShape.RECOMMENDATION_LIST,
        "EVENT_RECOMMENDATION": QuestionShape.RECOMMENDATION_LIST,
        "TRAVEL_ADVICE": QuestionShape.LIST,
        "TOUR_AVAILABILITY": QuestionShape.TOUR_AVAILABILITY,
        "TOUR_PLAN": QuestionShape.ITINERARY,
        "COMPARISON": QuestionShape.COMPARISON,
        "TRUE_FALSE": QuestionShape.YES_NO,
    }
    if intent in _INTENT_TO_SHAPE:
        return _INTENT_TO_SHAPE[intent]

    plan_mode = str(meta.get("retrieval_plan_mode") or "").strip()
    _PLAN_MODE_TO_SHAPE = {
        "comparison": QuestionShape.COMPARISON,
        "dish_to_restaurant": QuestionShape.RECOMMENDATION_LIST,
        "itinerary": QuestionShape.ITINERARY,
    }
    if plan_mode in _PLAN_MODE_TO_SHAPE:
        return _PLAN_MODE_TO_SHAPE[plan_mode]

    return QuestionShape.LIST


def infer_operation(
    query: str,
    query_norm: str,
    metadata: Dict[str, Any],
    question_shape: QuestionShape,
    target_class: Optional[str],
) -> tuple:
    """Infer what the user wants to DO with the target entity.

    Returns (operation, source, confidence).
    Priority: query_frame operator > answer_mode > keyword patterns > question_shape.
    """
    query_frame = metadata.get("query_frame") or {}
    operator = str(query_frame.get("query_operator") or "").strip()
    plan_mode = str(metadata.get("retrieval_plan_mode") or "").strip()

    _OPERATOR_TO_OPERATION = {
        "tour_availability": QueryOperation.AVAILABILITY_SEARCH,
        "constrained_nearby_search": QueryOperation.CONSTRAINED_NEARBY,
        "comparison": QueryOperation.COMPARISON,
        "dish_to_restaurant": QueryOperation.RECOMMENDATION,
        "lodging_near_anchor": QueryOperation.RECOMMENDATION,
        "itinerary_recommendation": QueryOperation.ITINERARY_BUILD,
        "choice_selection": QueryOperation.COMPARISON,
        "global_discovery": QueryOperation.DISCOVERY,
    }
    if operator in _OPERATOR_TO_OPERATION:
        return _OPERATOR_TO_OPERATION[operator], "query_frame", 0.95

    answer_mode = str(metadata.get("answer_mode") or "").strip()
    from graph_rag.core.answer_mode import AnswerMode
    _ANSWER_MODE_TO_OPERATION = {
        AnswerMode.TOUR_PLAN: QueryOperation.ITINERARY_BUILD,
        getattr(AnswerMode, "TOUR_LIST", "tour_list"): QueryOperation.AVAILABILITY_SEARCH,
        AnswerMode.TRUE_FALSE_VERIFIER: QueryOperation.FACT_VERIFY,
        AnswerMode.DISTANCE: QueryOperation.ATTRIBUTE_LOOKUP,
        getattr(AnswerMode, "DISCOVERY_LIST", "discovery_list"): QueryOperation.DISCOVERY,
    }
    if answer_mode in _ANSWER_MODE_TO_OPERATION:
        return _ANSWER_MODE_TO_OPERATION[answer_mode], "answer_mode", 0.9

    target_entity = metadata.get("target_entity") or ""
    is_broad_loc = False
    if target_entity:
        from graph_rag.core import keywords
        target_norm = normalize_text(target_entity, strip_punct=True)
        if target_norm in keywords.BROAD_LOCATION_NAMES or target_norm.startswith(
            ("phuong ", "xa ", "thi tran ", "thi xa ", "huyen ", "thanh pho ", "tp ", "tinh ")
        ):
            is_broad_loc = True
        else:
            for entity in metadata.get("entities") or []:
                if isinstance(entity, dict):
                    e_name = str(entity.get("name") or "").strip()
                    e_type = str(entity.get("type") or "").strip().lower()
                    if normalize_text(e_name, strip_punct=True) == target_norm:
                        if e_type in {"location", "city", "province", "district", "ward", "region"}:
                            is_broad_loc = True
                            break

    _ATTR_LOOKUP_MARKERS = [
        "gia bao nhieu", "gia bao", "bao nhieu tien", "chi phi",
        "bao gom gi", "bao gom nhung gi", "co nhung gi",
        "thoi gian", "thoi luong", "lich trinh cua tour",
        "don vi to chuc", "cong ty", "lien he", "so dien thoai",
        "website", "gio mo cua", "gia ve",
    ]
    if not is_broad_loc and any(m in query_norm for m in _ATTR_LOOKUP_MARKERS):
        return QueryOperation.ATTRIBUTE_LOOKUP, "keyword", 0.75

    _AVAILABILITY_MARKERS = [
        "co tour nao", "co tour", "tour nao", "tour gi",
        "goi tour nao", "khach san nao", "nha nghi nao",
        "co khach san", "co nha nghi", "tim tour", "tim khach san",
    ]
    if any(m in query_norm for m in _AVAILABILITY_MARKERS):
        return QueryOperation.AVAILABILITY_SEARCH, "keyword", 0.75

    _ITINERARY_MARKERS = [
        "lap lich trinh", "thiet ke chuyen di", "di nhu the nao",
        "lo trinh", "ke hoach du lich", "xay dung lich",
    ]
    if any(m in query_norm for m in _ITINERARY_MARKERS):
        return QueryOperation.ITINERARY_BUILD, "keyword", 0.75

    _REC_MARKERS = [
        "goi y", "de xuat", "nen di", "nen an", "nen o",
        "khuyen dung", "phu hop",
    ]
    if any(m in query_norm for m in _REC_MARKERS):
        return QueryOperation.RECOMMENDATION, "keyword", 0.65

    _SHAPE_TO_OPERATION = {
        QuestionShape.TOUR_AVAILABILITY: QueryOperation.AVAILABILITY_SEARCH,
        QuestionShape.ITINERARY: QueryOperation.ITINERARY_BUILD,
        QuestionShape.COMPARISON: QueryOperation.COMPARISON,
        QuestionShape.RECOMMENDATION_LIST: QueryOperation.RECOMMENDATION,
        QuestionShape.SINGLE_FACT: QueryOperation.ATTRIBUTE_LOOKUP,
        QuestionShape.YES_NO: QueryOperation.FACT_VERIFY,
    }
    if question_shape in _SHAPE_TO_OPERATION:
        return _SHAPE_TO_OPERATION[question_shape], "question_shape", 0.7

    return QueryOperation.DISCOVERY, "default", 0.5


def resolve_operation_conflicts(
    operation: QueryOperation,
    question_shape: QuestionShape,
    target_class: Optional[str],
    metadata: Dict[str, Any],
) -> tuple:
    """Resolve conflicts between operation and question_shape."""
    if operation == QueryOperation.ATTRIBUTE_LOOKUP and question_shape == QuestionShape.ITINERARY:
        question_shape = QuestionShape.SINGLE_FACT

    if operation == QueryOperation.AVAILABILITY_SEARCH and question_shape == QuestionShape.ITINERARY:
        question_shape = QuestionShape.TOUR_AVAILABILITY

    if operation == QueryOperation.CONSTRAINED_NEARBY and question_shape not in {
        QuestionShape.RECOMMENDATION_LIST, QuestionShape.LIST
    }:
        question_shape = QuestionShape.RECOMMENDATION_LIST

    return operation, question_shape


def infer_duration(q_norm: str) -> tuple:
    """Extract duration (days, nights) from normalized query.

    Delegates to graph_rag.utils.duration.infer_duration.
    """
    return _infer_duration_util(q_norm)


def extract_negative_activity(query_norm: str, matched_terms: List[str]) -> List[str]:
    """Extract the negated activity from a 'khong + [activity]' pattern."""
    _NEGATABLE_ACTIVITIES = [
        "canoe", "di bo", "lan", "boi", "chay bo", "leo nui",
        "choi team building", "tham quan bao tang", "lan",
    ]
    negated: List[str] = []
    for activity in _NEGATABLE_ACTIVITIES:
        pattern = rf"\bkhong\s+{re.escape(activity)}\b"
        if re.search(pattern, query_norm):
            negated.append(activity)
        pattern2 = rf"\btranh\s+{re.escape(activity)}\b"
        if re.search(pattern2, query_norm):
            negated.append(activity)
    if negated:
        return list(set(matched_terms + negated))
    return matched_terms


def extract_constraints(query_norm: str) -> List[ConstraintSpec]:
    """Extract semantic constraints from the normalized query.

    Uses FEATURE_REGISTRY to identify keyword matches and
    returns a list of ConstraintSpec objects.
    """
    from graph_rag.core.feature_registry import FEATURE_REGISTRY

    constraints: List[ConstraintSpec] = []
    for feature, spec in FEATURE_REGISTRY.items():
        matched = [kw for kw in spec["keywords"] if kw in query_norm]
        if matched:
            if spec.get("requires_activity_extraction"):
                matched = extract_negative_activity(query_norm, matched)
            constraints.append(ConstraintSpec(
                feature=feature,
                weight=spec.get("weight", 1.0),
                is_hard=spec.get("is_hard", True),
                source="keyword",
                matched_terms=matched,
                label_filter=spec.get("labels", []),
            ))
    return constraints


# ---------------------------------------------------------------------------
# build_query_fields — the main entry point
# ---------------------------------------------------------------------------

def build_query_fields(
    query: str,
    metadata: Dict[str, Any],
    answer_mode: str = "",
    query_frame: Optional[Dict[str, Any]] = None,
) -> QueryFields:
    """Canonical builder to parse metadata and infer missing query attributes.

    This is the standalone replacement for ``QueryState.from_metadata()``.
    Returns a plain ``QueryFields`` dataclass instead of a ``QueryState``.
    All inference logic is preserved verbatim.
    """

    # 0. Normalize query (two-tier: raw for display, normalized for matching)
    query_norm = normalize_text(query, strip_punct=True)

    # 1. Infer follow up
    is_follow_up = bool(metadata.get("is_follow_up", False))

    # 2. Infer target_class
    target_class = metadata.get("target_class")
    target_class_source = "metadata" if target_class else None
    target_class_confidence = 1.0 if target_class else 0.0

    if not target_class:
        inferred_class = infer_target_class(query, metadata)
        if inferred_class:
            target_class = inferred_class
            target_class_source = "query_pattern"
            target_class_confidence = 0.9

    # 2b. Infer semantic category
    semantic_category, semantic_category_confidence = infer_semantic_category(query_norm)

    # Safeguard: if target_class is Dish, semantic_category should NOT be nature/heritage
    if target_class == "Dish" and semantic_category in {"natural_landmark", "heritage"}:
        semantic_category = None
        semantic_category_confidence = 0.0

    # 3. Infer target_dish
    target_dish = None
    target_dish_source = None
    target_dish_confidence = 0.0

    inferred_dish = infer_target_dish(query, metadata)
    if inferred_dish:
        target_dish = inferred_dish
        target_dish_source = "query_pattern"
        target_dish_confidence = 0.9
        _ADVICE_SIGNALS = {"kinh nghiem", "meo", "luu y", "nen chuan bi", "can biet", "tiet kiem"}
        q_norm_advice = normalize_text(query, strip_punct=True)
        is_advice = any(s in q_norm_advice for s in _ADVICE_SIGNALS)
        if target_class and target_class not in {"Dish", "Specialty"} and not is_advice:
            target_class = "Dish"
            target_class_source = "target_dish_override"
            target_class_confidence = 0.85

    # 4. Infer requested attributes
    requested_attributes = infer_requested_attributes(query, metadata)

    # 5. Infer question shape
    question_shape = None
    question_shape_source = None
    question_shape_confidence = 0.0

    # Read from query_frame or answer_mode first if present.
    if query_frame and query_frame.get("mode"):
        mode = query_frame.get("mode")
        operator = (
            query_frame.get("query_operator")
            or (query_frame.get("query_frame") or {}).get("query_operator")
            or metadata.get("query_operator")
            or (metadata.get("query_frame") or {}).get("query_operator")
        )
        if operator == "tour_availability" or (
            mode == "class_search" and (
                query_frame.get("target_class") == "Tour"
                or metadata.get("target_class") == "Tour"
                or ((query_frame.get("query_frame") or {}).get("retrieval_plan") or {}).get("context_policy", {}).get("target_class") == "Tour"
                or ((metadata.get("query_frame") or {}).get("retrieval_plan") or {}).get("context_policy", {}).get("target_class") == "Tour"
            )
        ):
            question_shape = QuestionShape.TOUR_AVAILABILITY
            question_shape_source = "query_frame"
            question_shape_confidence = 1.0
            target_class = "Tour"
            target_class_source = "query_frame"
            target_class_confidence = 1.0
        elif mode == "comparison":
            question_shape = QuestionShape.COMPARISON
            question_shape_source = "query_frame"
            question_shape_confidence = 1.0
        elif mode in {"dish_to_restaurant", "lodging_near_anchor", "constrained_nearby_search"}:
            question_shape = QuestionShape.RECOMMENDATION_LIST
            question_shape_source = "query_frame"
            question_shape_confidence = 1.0

    if not question_shape and answer_mode:
        try:
            from graph_rag.core.answer_mode import AnswerMode
            if answer_mode in {
                AnswerMode.FILL_BLANK_SHORT,
                AnswerMode.NEGATIVE_ABSTAIN_GUARD,
            }:
                question_shape = QuestionShape.SINGLE_FACT
                question_shape_source = "answer_mode"
                question_shape_confidence = 1.0
            elif answer_mode == AnswerMode.TOUR_PLAN:
                question_shape = QuestionShape.ITINERARY
                question_shape_source = "answer_mode"
                question_shape_confidence = 1.0
            elif getattr(AnswerMode, "TOUR_LIST", None) and answer_mode == AnswerMode.TOUR_LIST:
                question_shape = QuestionShape.TOUR_AVAILABILITY
                question_shape_source = "answer_mode"
                question_shape_confidence = 1.0
                target_class = "Tour"
                target_class_source = "answer_mode"
                target_class_confidence = 1.0
            elif answer_mode == AnswerMode.TRUE_FALSE_VERIFIER:
                question_shape = QuestionShape.YES_NO
                question_shape_source = "answer_mode"
                question_shape_confidence = 1.0
            elif answer_mode == AnswerMode.DISTANCE:
                question_shape = QuestionShape.SINGLE_FACT
                question_shape_source = "answer_mode"
                question_shape_confidence = 1.0
            elif answer_mode in {
                AnswerMode.SINGLE_OPTION_RESOLVER,
                AnswerMode.MULTI_OPTION_RESOLVER,
            }:
                question_shape = QuestionShape.COMPARISON
                question_shape_source = "answer_mode"
                question_shape_confidence = 1.0
        except (ImportError, AttributeError):
            pass

    if not question_shape:
        inferred_shape = infer_question_shape(query, metadata)
        question_shape = inferred_shape
        question_shape_source = "keyword"
        question_shape_confidence = 0.8

    # 6. target_entity
    target_entity = metadata.get("target_entity")

    # 7. Infer requested_relations from RELATION_TRIGGERS
    requested_relations: List[str] = list(metadata.get("requested_relations") or [])
    matched_markers: List[str] = []
    for relation, trigger in RELATION_TRIGGERS.items():
        for marker in trigger["markers"]:
            if marker in query_norm:
                if relation not in requested_relations:
                    requested_relations.append(relation)
                if marker not in matched_markers:
                    matched_markers.append(marker)
                break

    # 8. Constraint extraction
    constraints = extract_constraints(query_norm)
    duration_days, duration_nights = infer_duration(query_norm)

    # Backward-compat booleans derived from constraints
    coastal_required = any(c.feature == "coastal" for c in constraints)
    sunset_required = any(c.feature == "sunset" for c in constraints)
    island_required = any(c.feature == "island" for c in constraints)
    walking_required = any(c.feature == "walking" for c in constraints)
    low_mobility_required = any(c.feature == "low_mobility" for c in constraints)
    family_friendly_required = any(c.feature == "family_friendly" for c in constraints)
    budget_required = any(c.feature == "budget" for c in constraints)

    if constraints:
        constraint_desc = ", ".join(
            f"{c.feature}({'hard' if c.is_hard else 'soft'})" for c in constraints
        )
        logger.info("   -> [ConstraintExtraction] constraints=[%s], duration=%sD%sN", constraint_desc, duration_days, duration_nights)

    # 9. Infer operation
    operation, operation_source, operation_confidence = infer_operation(
        query=query,
        query_norm=query_norm,
        metadata=metadata,
        question_shape=question_shape,
        target_class=target_class,
    )
    logger.info("   -> [Operation] %s (source=%s, conf=%.2f)", operation.value, operation_source, operation_confidence)

    # 10. Conflict resolution
    operation, question_shape = resolve_operation_conflicts(
        operation, question_shape, target_class, metadata
    )

    return QueryFields(
        query=query,
        query_norm=query_norm,
        question_shape=question_shape,
        target_class=target_class,
        target_dish=target_dish,
        target_entity=target_entity,
        requested_attributes=requested_attributes,
        requested_relations=requested_relations,
        matched_markers=matched_markers,
        is_follow_up=is_follow_up,
        constraints=constraints,
        coastal_required=coastal_required,
        sunset_required=sunset_required,
        island_required=island_required,
        walking_required=walking_required,
        low_mobility_required=low_mobility_required,
        family_friendly_required=family_friendly_required,
        budget_required=budget_required,
        duration_days=duration_days,
        duration_nights=duration_nights,
        operation=operation,
        operation_source=operation_source,
        operation_confidence=operation_confidence,
        metadata=metadata,
        target_class_source=target_class_source,
        target_class_confidence=target_class_confidence,
        target_dish_source=target_dish_source,
        target_dish_confidence=target_dish_confidence,
        question_shape_source=question_shape_source,
        question_shape_confidence=question_shape_confidence,
        semantic_category=semantic_category,
        semantic_category_confidence=semantic_category_confidence,
    )
