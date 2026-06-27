"""Core data objects used by retrieval and pipeline modules.

- ``NodeItem`` and ``RelationshipItem`` are used by production retrieval and
  graph modules.
- ``QueryOperation``, ``QuestionShape``, ``ConstraintSpec`` are shared enums
  and dataclasses used by QueryPlan, query_plan_builder, and policy_ranker.

The legacy ``QueryState`` and ``RagState`` classes have been removed.
Query field inference now lives in ``graph_rag.core.query_fields``.
The active orchestration state is ``QueryPlan`` + ``PipelineRuntime`` in
``graph_rag.pipeline.orchestration.query_plan``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


@dataclass
class NodeItem:
    """Represents a graph node returned by retrieval or grounding."""

    id: str
    content: str
    score: float
    source_type: str  # 'vector', 'fulltext', 'graph', ...
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"NodeItem(id={self.id[-10:]}..., content={self.content[:20:]}..., score={self.score:.3f})"


@dataclass
class RelationshipItem:
    """Represents a graph relationship in legacy state/test code."""

    start_node: str
    end_node: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"{self.start_node} --[{self.relationship_type}]--> {self.end_node}"


class QueryOperation(str, Enum):
    """What the user wants to DO with the target entity.

    Separates operation (action) from target_class (object) and constraints (conditions).
    This prevents misrouting like "Tour 1 ngay gia bao nhieu" -> itinerary_build.
    """

    AVAILABILITY_SEARCH = "availability_search"
    ATTRIBUTE_LOOKUP = "attribute_lookup"
    ITINERARY_BUILD = "itinerary_build"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    CONSTRAINED_NEARBY = "constrained_nearby"
    FACT_VERIFY = "fact_verify"
    DISCOVERY = "discovery"


class QuestionShape(str, Enum):
    LIST = "list"
    LIST_RANKING = "list_ranking"
    RECOMMENDATION_LIST = "recommendation_list"
    ITINERARY = "itinerary"
    TOUR_AVAILABILITY = "tour_availability"
    SINGLE_FACT = "single_fact"
    COMPARISON = "comparison"
    YES_NO = "yes_no"
    DISCOVERY = "discovery"
    ADVICE = "advice"
    UNKNOWN = "unknown"


@dataclass
class ConstraintSpec:
    """A single semantic constraint extracted from the user query.

    Used by PolicyRanker and RouteGate to enforce feature requirements
    (e.g. coastal, sunset, island) without hardcoding each feature as
    a boolean field.
    """

    feature: str                                     # coastal, sunset, island, ...
    weight: float = 1.0
    is_hard: bool = False
    source: str = "keyword"                          # keyword | user_explicit | llm_inferred
    matched_terms: List[str] = field(default_factory=list)
    label_filter: List[str] = field(default_factory=list)  # expected candidate labels
