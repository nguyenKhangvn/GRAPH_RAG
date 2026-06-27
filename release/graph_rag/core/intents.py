"""Intent constants and utilities used across GraphRAG pipeline.

Intent taxonomy is now loaded from graph_rag/config/intent_policy.json.
This module provides backward-compatible class interface.

To modify intents: edit intent_policy.json, NOT this file.
"""

from __future__ import annotations

from typing import Iterable, Optional, Union

from enum import Enum

import json
from pathlib import Path

try:
    _policy_path = Path(__file__).resolve().parent.parent / "config" / "intent_policy.json"
    with open(_policy_path, "r", encoding="utf-8") as _f:
        _policy = json.load(_f)
except (FileNotFoundError, json.JSONDecodeError, OSError):
    _policy = {}


class IntentType(str, Enum):
    # Recommendation intents
    ACCOMMODATION = "ACCOMMODATION_RECOMMENDATION"
    FOOD = "FOOD_RECOMMENDATION"
    TOURISM = "TOURISM_RECOMMENDATION"
    EVENT = "EVENT_RECOMMENDATION"

    # Plan/search intents
    TOUR_PLAN = "TOUR_PLAN"
    ENTITY_FACT = "ENTITY_FACT_QUERY"
    DISCOVERY = "DISCOVERY_SEARCH"
    DISTANCE = "DISTANCE_QUERY"
    TRAVEL_ADVICE = "TRAVEL_ADVICE"
    TRANSPORT_INFO = "TRANSPORT_INFO"
    EMERGENCY_SUPPORT = "EMERGENCY_SUPPORT"
    CASHLESS_PAYMENT = "CASHLESS_PAYMENT"
    WEATHER_ADVICE = "WEATHER_ADVICE"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def all(cls) -> set[str]:
        return set(cls.ALL_INTENTS)

    @classmethod
    def is_valid(cls, intent: Optional[Union[str, "IntentType"]]) -> bool:
        if isinstance(intent, IntentType):
            intent = intent.value
        return bool(intent) and str(intent) in cls.ALL_INTENTS

    @classmethod
    def normalize(
        cls,
        intent: Optional[Union[str, "IntentType"]],
        default: Optional[Union[str, "IntentType"]] = None,
    ) -> str:
        if isinstance(default, IntentType):
            default = default.value
        if isinstance(intent, IntentType):
            return intent.value
        if not intent:
            return default or cls.DISCOVERY.value
        normalized = str(intent).strip().upper()
        return normalized if normalized in cls.ALL_INTENTS else (default or cls.DISCOVERY.value)

    @classmethod
    def first_valid(
        cls,
        intents: Iterable[Union[str, "IntentType"]],
        default: Optional[Union[str, "IntentType"]] = None,
    ) -> str:
        if isinstance(default, IntentType):
            default = default.value
        for intent in intents:
            normalized = cls.normalize(intent, default="")
            if normalized:
                return normalized
        return default or cls.DISCOVERY.value

    @classmethod
    def from_value(
        cls,
        intent: Optional[Union[str, "IntentType"]],
        default: Optional["IntentType"] = None,
    ) -> "IntentType":
        if isinstance(intent, IntentType):
            return intent
        if not intent:
            return default or cls.DISCOVERY
        try:
            return cls(str(intent).strip().upper())
        except ValueError:
            return default or cls.DISCOVERY


class IntentMode:
    """Query processing modes — HOW to retrieve, independent of intent."""
    _modes = _policy.get("modes", {})
    SINGLE_ANCHOR = _modes.get("SINGLE_ANCHOR", {}).get("value", "single_anchor")
    COMPARISON = _modes.get("COMPARISON", {}).get("value", "comparison")
    CONSTRAINT_MATCHING = _modes.get("CONSTRAINT_MATCHING", {}).get("value", "constraint_matching")
    MULTI_ENTITY_NEARBY = _modes.get("MULTI_ENTITY_NEARBY", {}).get("value", "multi_entity_nearby")
    DISH_TO_RESTAURANT = _modes.get("DISH_TO_RESTAURANT", {}).get("value", "dish_to_restaurant")
    TOUR_PLAN = _modes.get("TOUR_PLAN", {}).get("value", "tour_plan")
    NEGATIVE = _modes.get("NEGATIVE", {}).get("value", "negative")


class RegionFocus:
    """Geographic region scope for retrieval."""
    COASTAL = "coastal_quy_nhon"
    INLAND = "inland_gia_lai"
    ALL = "all"


# Build sets from intent_policy.json groups (strings only for compatibility)
_groups = _policy.get("intent_groups", {})
_DEFAULT_RECOMMENDATION = [
    IntentType.ACCOMMODATION.value,
    IntentType.FOOD.value,
    IntentType.TOURISM.value,
    IntentType.EVENT.value,
]
_DEFAULT_SEARCH = [
    IntentType.TOUR_PLAN.value,
    IntentType.ENTITY_FACT.value,
    IntentType.DISCOVERY.value,
    IntentType.DISTANCE.value,
]
_RECOMMENDATION_INTENTS = set(_groups.get("recommendation", _DEFAULT_RECOMMENDATION))
_SEARCH_INTENTS = set(_groups.get("search", _DEFAULT_SEARCH))
_RECOMMENDATION_INTENTS = {str(i).strip().upper() for i in _RECOMMENDATION_INTENTS}
_SEARCH_INTENTS = {str(i).strip().upper() for i in _SEARCH_INTENTS}

IntentType.RECOMMENDATION_INTENTS = _RECOMMENDATION_INTENTS
IntentType.SEARCH_INTENTS = _SEARCH_INTENTS
IntentType.ALL_INTENTS = IntentType.RECOMMENDATION_INTENTS | IntentType.SEARCH_INTENTS
