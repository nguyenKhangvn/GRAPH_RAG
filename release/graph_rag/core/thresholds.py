"""Configurable thresholds for the GraphRAG pipeline.

All thresholds are now loaded from graph_rag/config/thresholds.json
via the ConfigLoader. Environment variable overrides still work
(same semantics as before).

To modify thresholds: edit thresholds.json or set env vars, NOT this file.
"""

from graph_rag.config import cfg as _cfg
_t = _cfg.thresholds()

# === Grounding ===
GROUNDING_CONFIDENCE_THRESHOLD = _t["grounding"]["confidence_threshold"]
EXACT_MATCH_SCORE = _t["grounding"]["exact_match_score"]
SUBSTRING_MATCH_SCORE = _t["grounding"]["substring_match_score"]
CONTAINS_MATCH_SCORE = _t["grounding"]["contains_match_score"]

# === Semantic Grounding ===
SEMANTIC_GROUNDING_THRESHOLD = _t["semantic_grounding"]["threshold"]
SEMANTIC_GROUNDING_TOP_K = _t["semantic_grounding"]["top_k"]

# === Location ===
LOCATION_SOURCE_CONFIDENCE = _t["location"]["source_confidence"]

# === Distance ===
TOUR_PLAN_MAX_HOP_KM = _t["distance"]["tour_plan_max_hop_km"]
WALKING_MAX_HOP_KM = _t["distance"]["walking_max_hop_km"]
SENIOR_FAMILY_MAX_HOP_KM = _t["distance"]["senior_family_max_hop_km"]
PROXIMITY_SEARCH_MAX_M = _t["distance"]["proximity_search_max_m"]
LOCATION_FILTER_MAX_KM = _t["distance"]["location_filter_max_km"]

# === Restaurant filter ===
MIN_RESTAURANT_RATING = _t["restaurant"]["min_rating"]

# === Tour plan clustering ===
CITY_CENTER_DISTANCE_KM = _t["tour_plan_clustering"]["city_center_distance_km"]
CITY_CENTER_FALLBACK_DISTANCE_KM = _t["tour_plan_clustering"]["city_center_fallback_distance_km"]
NEAR_SUBURBAN_DISTANCE_KM = _t["tour_plan_clustering"]["near_suburban_distance_km"]

# === Geographic bounds ===
_inland = _t["geographic_bounds"]["inland"]
_coastal = _t["geographic_bounds"]["coastal"]
INLAND_LAT_MIN = _inland["lat_min"]
INLAND_LAT_MAX = _inland["lat_max"]
INLAND_LNG_MIN = _inland["lng_min"]
INLAND_LNG_MAX = _inland["lng_max"]
COASTAL_LAT_MIN = _coastal["lat_min"]
COASTAL_LAT_MAX = _coastal["lat_max"]
COASTAL_LNG_MIN = _coastal["lng_min"]
COASTAL_LNG_MAX = _coastal["lng_max"]

# === Tour plan adaptive hop distances ===
ADAPTIVE_HOP_KM = _t["adaptive_hop_km"]

# === Tour plan seed quotas ===
TOUR_PLAN_SEED_QUOTAS = _t["tour_plan_seed_quotas"]

# === Pruning / Context ===
_cl = _t["context_limits"]
RAW_CONTEXT_DEFAULT_MAX_ITEMS = _cl["default_max_items"]
RAW_CONTEXT_MAX_ITEMS_BY_INTENT = {
    intent: spec["default"] if isinstance(spec, dict) else spec
    for intent, spec in _cl["max_items_by_intent"].items()
}

# === Answer contract ===
_ac = _t["answer_contract"]
MIN_CONTEXT_LENGTH = _ac["min_context_length"]

# === Closed-form dispatch ===
_cf = _t["closed_form"]
OPTION_SCORE_THRESHOLD = _cf["option_score_threshold"]

# === Pipeline thresholds ===
_pl = _t["pipeline"]
INSUFFICIENT_FACT_THRESHOLD = _pl["insufficient_fact_threshold"]["normal"]
INSUFFICIENT_FACT_THRESHOLD_FOLLOWUP = _pl["insufficient_fact_threshold"]["follow_up"]
RICH_CONTEXT_MIN_FACTS = _pl["rich_context_min_facts"]
LOCATION_FILTER_EXEMPT_INTENTS = set(_pl["location_filter_exempt_intents"])

