from __future__ import annotations
"""
QueryFrame Contract - Semantic query contract giữa các modules.

QueryFrame chứa các field có ý nghĩa ổn định cho retrieval/generation.
Không thay thế metadata dict, mà tách phần "semantic contract" ra khỏi metadata.

Metadata vẫn giữ cho: debug_trace, retrieval_scores, raw_llm_output,
fallback_reason, latency_ms, provider_error, candidate_count, etc.

Usage:
    frame = QueryFrame(
        intent=IntentType.FOOD,
        geo_scope="coastal_quy_nhon",
        legacy_province="Bình Định",
        target_labels=["Restaurant", "Dish"],
    )
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional



@dataclass
class QueryFrame:
    """Semantic query contract - ý nghĩa ổn định cho retrieval/generation.

    Không chứa debug info, scores, raw output -这些东西属于 metadata dict.
    """

    # === Intent & Operator ===
    intent: str = "UNKNOWN"                    # FOOD_RECOMMENDATION, DISCOVERY, etc.
    original_intent: str = "UNKNOWN"           # Intent trước khi query_frame override
    operator: str = "default"                  # global_discovery, comparison, etc.
    answer_mode: str = "fact_answer"           # discovery_list, fact_answer, etc.

    # === Geographic Scope ===
    geo_scope: str = "all"                     # coastal_quy_nhon, inland_gia_lai, all
    province: str = ""                         # Tỉnh hiện hành (unified)
    legacy_province: Optional[str] = None      # Bình Định, Kon Tum (alias)
    region: Optional[str] = None               # Quy Nhơn, Pleiku, Măng Đen
    region_focus: str = "all"                  # Focus filter cho retrieval
    region_group: Optional[str] = None         # binh_dinh_legacy, gia_lai_core

    # === Admin Level (graph-based) ===
    admin_level: Optional[str] = None          # province | area | ward
    admin_status: Optional[str] = None         # current | merged | legacy

    # === Retrieval Targets ===
    target_labels: List[str] = field(default_factory=list)  # ["Restaurant", "Dish"]
    required_evidence: List[str] = field(default_factory=list)  # ["price_range", "description"]
    forbidden_fallbacks: List[str] = field(default_factory=list)  # Không dùng nếu thiếu evidence
    forbidden_labels: List[str] = field(default_factory=list)  # Labels explicitly forbidden by validator or intent


    # === Grounding ===
    anchors: List[str] = field(default_factory=list)  # Grounded entity names
    anchor_types: Dict[str, str] = field(default_factory=dict)  # {name: type}

    # === Constraints ===
    constraints: Dict[str, Any] = field(default_factory=dict)  # max_distance, mobility, etc.
    max_results: int = 10
    month_constraint: Optional[List[int]] = None  # For events

    # === Confidence & Source ===
    confidence: float = 0.0
    source: str = ""  # Who created this frame (v3_router, llm_analyzer, etc.)

    # === Debug Trace (lightweight) ===
    trace: List[str] = field(default_factory=list)  # Short audit trail

    def add_trace(self, msg: str) -> None:
        """Add a trace message for debugging."""
        self.trace.append(msg)

    def is_discovery(self) -> bool:
        """Check if this is a discovery query."""
        return self.operator == "global_discovery" or self.answer_mode == "discovery_list"

    def is_food_query(self) -> bool:
        """Check if this is a food/restaurant query."""
        return "FOOD" in self.intent.upper()

    def has_location(self) -> bool:
        """Check if query has geographic context."""
        return bool(self.legacy_province or self.region or self.geo_scope != "all")

    def get_location_display(self) -> str:
        """Get display string for location."""
        if self.region and self.legacy_province:
            return f"{self.region}, {self.legacy_province}"
        if self.region:
            return self.region
        if self.legacy_province:
            return self.legacy_province
        return self.province or ""

    def __repr__(self) -> str:
        return (
            f"QueryFrame(intent='{self.intent}', "
            f"geo_scope='{self.geo_scope}', "
            f"legacy_province='{self.legacy_province}', "
            f"target_labels={self.target_labels}, "
            f"anchors={self.anchors})"
        )
