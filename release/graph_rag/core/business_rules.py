"""Externalized business rules for the GraphRAG pipeline.

All domain-specific rules are now loaded from graph_rag/config/business_rules.json
via the ConfigLoader.

To modify rules: edit business_rules.json, NOT this file.
"""

from graph_rag.config import cfg as _cfg

# === Critical System Facts (OVERRIDE PRE-TRAINED KNOWLEDGE) ===
SYSTEM_FACTS = _cfg.system_facts()

# === Geographic Region Aliases ===
REGION_ALIASES = _cfg.region_aliases()

# === Location Display Names ===
LOCATION_DISPLAY_NAMES = _cfg.location_display_names()

# === Restaurant Exclusion Rules ===
RESTAURANT_EXCLUDE_NAMES = _cfg.restaurant_exclude_names()

# === Tour Plan Cost Template ===
TOUR_COST_TEMPLATE = _cfg.business_rules().get("tour_cost_template", "")

# === Few-shot Examples ===
FEW_SHOT_EXAMPLES = _cfg.business_rules().get("few_shot_examples", "")
