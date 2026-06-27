"""Centralized configuration loader.

Reads all JSON config files once (lazy-loaded) and exposes them as typed
accessor methods. Supports env-var overrides for thresholds (same pattern
as the original thresholds.py).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONFIG_DIR = Path(__file__).resolve().parent


def _load_json(filename: str) -> dict:
    path = _CONFIG_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_env_value(spec: Any) -> Any:
    """If spec is a dict with 'env'/'default', resolve via env var."""
    if isinstance(spec, dict) and "env" in spec and "default" in spec:
        raw = os.getenv(spec["env"])
        if raw is None:
            return spec["default"]
        # coerce to same type as default
        default = spec["default"]
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, int):
            return int(raw)
        return raw
    if isinstance(spec, dict) and "env" in spec:
        return os.getenv(spec["env"])
    return spec


def _resolve_env_recursive(obj: Any) -> Any:
    """Walk a dict/list and resolve all env-value specs."""
    if isinstance(obj, dict):
        if "env" in obj and "default" in obj:
            return _resolve_env_value(obj)
        return {k: _resolve_env_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_resolve_env_recursive(v) for v in obj]
    return obj


class ConfigLoader:
    """Lazy-loading configuration facade."""

    # ── Schema Metadata ─────────────────────────────────────────────
    @lru_cache(maxsize=1)
    def schema_metadata(self) -> dict:
        return _load_json("schema_metadata.json")

    def node_type_meta(self, node_type: str) -> dict | None:
        return self.schema_metadata().get("node_types", {}).get(node_type)

    def relation_meta(self, relation: str) -> dict | None:
        return self.schema_metadata().get("relations", {}).get(relation)

    def all_relation_names(self) -> list[str]:
        return list(self.schema_metadata().get("relations", {}).keys())

    def relations_by_semantic_role(self, role: str) -> list[dict]:
        """Return all relations whose semantic_role matches."""
        results = []
        for name, meta in self.schema_metadata().get("relations", {}).items():
            if meta.get("semantic_role") == role:
                results.append({"name": name, **meta})
        return results

    def relations_for_evidence_types(self, evidence_types: list[str]) -> list[dict]:
        """Return relations that support any of the given evidence types."""
        results = []
        for name, meta in self.schema_metadata().get("relations", {}).items():
            supported = meta.get("supports_evidence_types", [])
            if any(et in supported for et in evidence_types):
                results.append({"name": name, **meta})
        return sorted(results, key=lambda r: r.get("confidence_prior", 0), reverse=True)

    def relationship_map(self) -> dict[str, str]:
        """Build RELATIONSHIP_MAP {name: display_vi} from schema metadata."""
        return {
            name: meta.get("display_vi", name)
            for name, meta in self.schema_metadata().get("relations", {}).items()
        }

    # ── Domain Keywords ─────────────────────────────────────────────
    @lru_cache(maxsize=1)
    def keywords(self) -> dict:
        return _load_json("domain_keywords.json")

    def intent_signals(self, intent: str) -> list[str]:
        return self.keywords().get("intent_signals", {}).get(intent, [])

    def relation_hints(self) -> dict[str, list[str]]:
        return self.keywords().get("relation_hints", {})

    def region_keywords(self, region_type: str) -> list[str]:
        return self.keywords().get("region", {}).get(region_type, [])

    def mode_signals(self, mode: str) -> list[str]:
        return self.keywords().get("mode_signals", {}).get(mode, [])

    # ── Thresholds ──────────────────────────────────────────────────
    @lru_cache(maxsize=1)
    def _thresholds_raw(self) -> dict:
        return _load_json("thresholds.json")

    def thresholds(self) -> dict:
        """Return fully resolved thresholds (env overrides applied)."""
        return _resolve_env_recursive(self._thresholds_raw())

    def grounding_threshold(self) -> float:
        return self.thresholds()["grounding"]["confidence_threshold"]

    def geographic_bounds(self, region: str) -> dict:
        return self.thresholds()["geographic_bounds"][region]

    def adaptive_hop_km(self, key: str) -> float:
        return self.thresholds()["adaptive_hop_km"].get(key,
            self.thresholds()["adaptive_hop_km"]["DEFAULT"])

    def evidence_sufficiency_config(self) -> dict:
        return self.thresholds()["evidence_sufficiency"]

    # ── Scoring Weights ─────────────────────────────────────────────
    @lru_cache(maxsize=1)
    def scoring_weights(self) -> dict:
        return _load_json("scoring_weights.json")

    def context_organizer_weights(self) -> dict:
        return self.scoring_weights()["context_organizer"]

    def reranker_weights(self) -> dict:
        return self.scoring_weights()["fact_reranker"]

    def fusion_weights(self) -> dict:
        return self.scoring_weights()["hybrid_fusion"]

    def relationship_confidence(self) -> dict[str, float]:
        return self.scoring_weights()["relationship_confidence"]

    def relation_priority_for_intent(self, intent: str) -> list[str]:
        return self.scoring_weights().get("relation_priority_by_intent", {}).get(intent, [])

    def structural_budget_for_intent(self, intent: str) -> int:
        return self.scoring_weights().get("structural_budget_by_intent", {}).get(intent, 12)

    def grounded_topk_for_intent(self, intent: str) -> int:
        return self.scoring_weights().get("grounded_topk_by_intent", {}).get(intent, 8)

    # ── Feature Flags ────────────────────────────────────────────────
    def is_bge_candidate_scoring_enabled(self) -> bool:
        return os.getenv("ENABLE_BGE_CANDIDATE_SCORING", "false").lower() == "true"

    # ── Admin Location Keywords ──────────────────────────────────────
    def province_keywords(self) -> set:
        return set(self.keywords().get("region", {}).get("province_keywords", []))

    def district_keywords(self) -> set:
        return set(self.keywords().get("region", {}).get("district_keywords", []))

    def ward_keywords(self) -> set:
        return set(self.keywords().get("region", {}).get("ward_keywords", []))

    # ── Attribute Labels (for answer validators) ─────────────────────
    def requested_attribute_labels(self) -> dict:
        return self.keywords().get("requested_attribute_labels", {})

    def requested_attribute_query_hints(self) -> dict:
        return self.keywords().get("requested_attribute_query_hints", {})

    def analytical_location_hints(self) -> list:
        return self.keywords().get("analytical_location_hints", [])

    # ── Intent Policy ───────────────────────────────────────────────
    @lru_cache(maxsize=1)
    def intent_policy(self) -> dict:
        return _load_json("intent_policy.json")

    def intent_config(self, intent: str) -> dict | None:
        return self.intent_policy().get("intents", {}).get(intent)

    def intent_evidence_types(self, intent: str) -> list[str]:
        config = self.intent_config(intent)
        return config.get("evidence_types", []) if config else []

    def all_intent_names(self) -> list[str]:
        return list(self.intent_policy().get("intents", {}).keys())

    def traversal_whitelist(self) -> list[str]:
        return self.intent_policy().get("traversal_whitelist", [])

    def attribute_policy(self, intent: str) -> list[dict]:
        return self.intent_policy().get("attribute_policy", {}).get(intent, [])

    # ── Business Rules ──────────────────────────────────────────────
    @lru_cache(maxsize=1)
    def business_rules(self) -> dict:
        return _load_json("business_rules.json")

    def system_facts(self) -> str:
        return self.business_rules().get("system_facts", "")

    def region_aliases(self) -> dict:
        return self.business_rules().get("region_aliases", {})

    def location_display_names(self) -> dict:
        return self.business_rules().get("location_display_names", {})

    def restaurant_exclude_names(self) -> set:
        return set(self.business_rules().get("restaurant_exclude_names", []))

    # ── Cache Management ────────────────────────────────────────────
    def reload(self):
        """Clear all cached configs. Useful for hot-reload or testing."""
        self.schema_metadata.cache_clear()
        self.keywords.cache_clear()
        self._thresholds_raw.cache_clear()
        self.scoring_weights.cache_clear()
        self.intent_policy.cache_clear()
        self.business_rules.cache_clear()
