"""Centralized Vietnamese keyword/hint definitions for the GraphRAG pipeline.

All keyword lists are now loaded from graph_rag/config/domain_keywords.json
via the ConfigLoader. This module provides backward-compatible constant names
so existing code continues to work without changes.

To modify keywords: edit domain_keywords.json, NOT this file.
"""

from graph_rag.config import cfg as _cfg
from graph_rag.core.intents import IntentType
from graph_rag.utils.text import normalize_text

_kw = _cfg.keywords()

# === Intent Signals ===
ACCOMMODATION_SIGNALS = set(_kw.get("intent_signals", {}).get(IntentType.ACCOMMODATION, []))
FOOD_SIGNALS = set(_kw.get("intent_signals", {}).get(IntentType.FOOD, []))
TOURISM_SIGNALS = set(_kw.get("intent_signals", {}).get(IntentType.TOURISM, []))
HERITAGE_SIGNALS = set(_kw.get("heritage_signals", []))
ANALYSIS_SIGNALS = set(_kw.get("analysis_signals", []))
DISTANCE_SIGNALS = set(_kw.get("intent_signals", {}).get(IntentType.DISTANCE, []))
TOUR_PLAN_SIGNALS = set(_kw.get("intent_signals", {}).get("TOUR_PLAN", []))
SHOPPING_SIGNALS = set(_kw.get("intent_signals", {}).get("SHOPPING_RECOMMENDATION", []))
PROXIMITY_SIGNALS = set(_kw.get("proximity_signals", []))
RELATION_MARKER_NAMES = set(_kw.get("relation_marker_names", []))
LOCATION_ROUTE_SIGNALS = set(_kw.get("location_route_signals", []))

# === Category Phrases ===
CATEGORY_PHRASES = set(_kw.get("category_phrases", []))

# === Attribute Hints ===
ATTRIBUTE_HINTS = {k: set(v) for k, v in _kw.get("attribute_hints", {}).items()}

# === Relation Hints ===
RELATION_HINTS = {k: set(v) for k, v in _kw.get("relation_hints", {}).items()}

# === Entity Prefixes ===
ENTITY_PREFIXES = _kw.get("entity_prefixes", [])

# === Region Keywords ===
COASTAL_KEYWORDS = set(_kw.get("region", {}).get("coastal_keywords", []))
INLAND_KEYWORDS = set(_kw.get("region", {}).get("inland_keywords", []))
OUT_OF_REGION_TERMS = set(_kw.get("region", {}).get("out_of_region_terms", []))
IN_SCOPE_REGION_TERMS = set(_kw.get("region", {}).get("in_scope_terms", []))

# === Question Particles ===
# Normalize to match against normalized entity names (no diacritics)
QUESTION_PARTICLES = {normalize_text(p) for p in _kw.get("question_particles", []) if p}
QUESTION_BIGRAMS = {normalize_text(b) for b in _kw.get("question_bigrams", []) if b}

# === Tourism Analysis Hints ===
TOURISM_ANALYSIS_HINTS = set(_kw.get("tourism_analysis_hints", []))
REAL_FOOD_HINTS = set(_kw.get("real_food_hints", []))
ADMIN_CENTER_HINTS = set(_kw.get("admin_center_hints", []))
LOCATION_LEAKAGE_HINTS = set(_kw.get("location_leakage_hints", []))

# === Mode Signals ===
CONSTRAINT_SIGNALS = set(_kw.get("mode_signals", {}).get("constraint", []))
FILTER_SIGNALS = set(_kw.get("mode_signals", {}).get("filter", []))
COMPARISON_SIGNALS = set(_kw.get("mode_signals", {}).get("comparison", []))
COMBINE_SIGNALS = set(_kw.get("mode_signals", {}).get("combine", []))
NEGATIVE_SIGNALS = set(_kw.get("mode_signals", {}).get("negative", []))

# === Labels ===
LODGING_LABELS = set(_kw.get("lodging_labels", []))
HERITAGE_LABELS = set(_kw.get("heritage_labels", []))

# === Service Signals ===
SERVICE_SIGNALS = set(_kw.get("service_signals", []))

# === Analytical Location Hints ===
ANALYTICAL_LOCATION_HINTS = set(_kw.get("analytical_location_hints", []))

# === Distance Tail Patterns ===
DISTANCE_TAIL_PATTERNS = _kw.get("distance_tail_patterns", [])

# === Typo Normalization ===
TYPO_NORMALIZATION = _kw.get("typo_normalization", {})

# === Category Keywords ===
CATEGORY_KEYWORDS = _kw.get("category_keywords", {})

# === Category Aliases ===
CATEGORY_ALIASES = [
    (item["display"], item["keys"])
    for item in _kw.get("category_aliases", [])
]

# === Category Markers ===
CATEGORY_MARKERS = _kw.get("category_markers", [])

# === Region Address Aliases ===
REGION_ADDRESS_ALIASES = _kw.get("region", {}).get("region_address_aliases", {})

# === Fact Verification Signals ===
FACT_VERIFICATION_SIGNALS = set(_kw.get("mode_signals", {}).get("fact_verification", []))

# === Transfer Route Signals ===
TRANSFER_ROUTE_SIGNALS = set(_kw.get("mode_signals", {}).get("transfer_route", []))

# === Classify Keywords ===
_classify = _kw.get("classify_keywords", {})
CLASSIFY_HERITAGE_KEYWORDS = _classify.get("heritage", [])
CLASSIFY_SPIRITUAL_KEYWORDS = _classify.get("spiritual", [])
CLASSIFY_CRAFT_KEYWORDS = _classify.get("craft", [])
CLASSIFY_PUBLIC_SPACE_KEYWORDS = _classify.get("public_space", [])
CLASSIFY_NATURE_KEYWORDS = _classify.get("nature", [])

# === Specific Entity Types ===
SPECIFIC_ENTITY_TYPES = set(_kw.get("specific_entity_types", []))

# === Constrained Nearby Patterns ===
CONSTRAINED_NEARBY_PATTERNS = _kw.get("constrained_nearby_patterns", [])

# === Non-Groundable Phrases ===
NON_GROUNDABLE_GENERIC_PHRASES = set(_kw.get("non_groundable_generic_phrases", []))
NON_GROUNDABLE_STARTS = _kw.get("non_groundable_starts", [])

# === Constraint Terms ===
CONSTRAINT_TERMS = _kw.get("constraint_terms", {})

# === Broad Location Anchor Names ===
BROAD_LOCATION_NAMES = set(_kw.get("broad_location_names", []))

# === Time-Range Keywords ===
# Temporal expressions that indicate "từ X đến Y" is time-based, not spatial.
TIME_RANGE_KEYWORDS = {
    "thang", "ngay", "nam", "tuan", "gio", "phut", "mua", "dem",
    "sang", "chieu", "toi", "trua", "hom nay", "ngay mai",
    "hom qua", "dau nam", "cuoi nam", "dau thang", "cuoi thang",
}

# === Transport Negative Signals ===
# Transportation-related terms that disqualify a "từ X đến Y" pattern as distance query.
TRANSPORT_NEGATIVE_SIGNALS = {
    "san bay", "bay thang", "bay den", "bay tu", "chuyen bay",
    "hang hang khong", "ve may bay", "may bay", "check in",
    "tau", "ga tau", "tau hoa", "xe buyt", "nha xe",
    "co the bay", "co tau", "co xe", "di bang", "di may bay",
    "di tau", "di xe",
}

# === Entity Processor Mixin Keywords ===
ACCOMMODATION_HINT_TOKENS = _kw.get("intent_signals", {}).get(IntentType.ACCOMMODATION, [])
HERITAGE_HINT_TOKENS = _kw.get("heritage_signals", [])
TOURISM_HINT_TOKENS = _kw.get("intent_signals", {}).get(IntentType.TOURISM, [])
QUERY_CONTEXT_TOURISM_SIGNALS = set(_kw.get("query_context_tourism_signals", []))
GENERIC_ANCHOR_TERMS = set(_kw.get("generic_anchor_terms", []))
GROUNDABLE_SHORT_NAMES = set(_kw.get("groundable_short_names", []))
ADDITIONAL_CATEGORY_TOKENS = _kw.get("additional_category_tokens", [])
DISTANCE_CONNECTORS = set(_kw.get("distance_connectors", []))
