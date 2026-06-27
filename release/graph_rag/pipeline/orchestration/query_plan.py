from __future__ import annotations
"""
QueryPlan + PipelineRuntime — Tách business intent khỏi runtime state.

QueryPlan (frozen): Ý định nghiệp vụ, ổn định sau Step 1.
PipelineRuntime (mutable): Kết quả chạy thực tế, cập nhật qua Step 2-5.

Thay thế việc đọc metadata dict trực tiếp ở Step 4/5.
"""


from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from graph_rag.core.state import ConstraintSpec, QueryOperation, QuestionShape


@dataclass(frozen=True)
class QueryPlan:
    """Frozen business intent — ổn định sau Step 1, không thay đổi qua Step 2-5.

    Hợp nhất 3 lớp cũ:
    - QueryState:    operation, question_shape, target_class, constraints
    - QueryFrame:    intent, geo_scope, target_labels, anchors
    - RetrievalPlan: retrieval_mode, required_relations, context_policy

    Tất cả collection fields dùng tuple/frozenset để đảm bảo immutable.
    """

    # === Query identity ===
    query: str = ""
    query_norm: str = ""

    # === From QueryState ===
    question_shape: QuestionShape = QuestionShape.UNKNOWN
    operation: QueryOperation = QueryOperation.DISCOVERY
    target_class: Optional[str] = None
    target_dish: Optional[str] = None
    target_entity: Optional[str] = None
    requested_attributes: tuple[str, ...] = ()
    requested_relations: tuple[str, ...] = ()
    constraints: tuple[ConstraintSpec, ...] = ()
    semantic_category: Optional[str] = None
    is_follow_up: bool = False
    duration_days: int = 0
    duration_nights: int = 0

    # === From QueryFrame contract ===
    intent: str = "UNKNOWN"
    original_intent: str = "UNKNOWN"
    topic: Optional[str] = None
    operator: str = "default"
    answer_mode: str = "fact_answer"


    # Geographic scope
    geo_scope: str = "all"
    province: str = ""
    legacy_province: Optional[str] = None
    region: Optional[str] = None
    region_focus: str = "all"
    region_group: Optional[str] = None

    # Admin level (graph-based)
    admin_level: Optional[str] = None          # province | area | ward
    admin_status: Optional[str] = None         # current | merged | legacy

    # Retrieval targets
    target_labels: tuple[str, ...] = ()
    forbidden_labels: tuple[str, ...] = ()

    # Grounding anchors
    anchors: tuple[str, ...] = ()
    anchor_types: tuple[tuple[str, str], ...] = ()  # ((name, type), ...)

    # Constraints
    max_results: int = 10
    month_constraint: Optional[tuple[int, ...]] = None

    # === From RetrievalPlan (routing) ===
    retrieval_mode: str = "single_anchor"
    required_relations_for_retrieval: tuple[str, ...] = ()
    context_policy: tuple[tuple[str, Any], ...] = ()  # frozen key-value pairs

    # === Source tracking ===
    target_class_source: Optional[str] = None
    target_class_confidence: float = 0.0
    target_dish_source: Optional[str] = None
    target_dish_confidence: float = 0.0
    question_shape_source: Optional[str] = None
    question_shape_confidence: float = 0.0
    operation_source: Optional[str] = None
    operation_confidence: float = 0.0
    semantic_category_confidence: float = 0.0

    # === Contract flags ===
    hard_label_contract: bool = False
    disable_agentic_retrieval: bool = False
    disable_generic_discovery: bool = False
    disable_discovery_expansion: bool = False
    disable_food_keywords: bool = False
    disable_entity_grounding: bool = False
    disable_non_location_grounding: bool = False
    skip_realtime_booking_guard: bool = False
    fallback_policy: Optional[str] = None

    # === From QueryFrame (evidence constraints) ===
    required_evidence: tuple[str, ...] = ()
    forbidden_fallbacks: tuple[str, ...] = ()

    # === Region policy (for Step 4) ===
    region_lock_mode: Optional[str] = None  # "disabled_multi_anchor_comparison", etc.
    exempt_location_from_grounding_filter: bool = False

    # === Renderer hint (for Step 5 dispatch) ===
    renderer: Optional[str] = None  # "comparison", "dish_to_restaurant", "tour_plan", etc.

    # === Audit ===
    built_from: str = "query_plan_builder_v1"
    contract_name: Optional[str] = None  # ticket_price, emergency, etc.

    # --- Convenience methods ---

    def is_food_query(self) -> bool:
        return "FOOD" in self.intent.upper()

    def is_discovery(self) -> bool:
        return (
            self.operator == "global_discovery"
            or self.answer_mode == "discovery_list"
            or self.operation == QueryOperation.DISCOVERY
        )

    def has_location(self) -> bool:
        return bool(self.legacy_province or self.region or self.geo_scope != "all")

    def get_location_display(self) -> str:
        if self.region and self.legacy_province:
            return f"{self.region}, {self.legacy_province}"
        if self.region:
            return self.region
        if self.legacy_province:
            return self.legacy_province
        return self.province or ""

    def to_debug_dict(self) -> Dict[str, Any]:
        """Debug representation for logging."""
        return {
            "query": self.query,
            "intent": self.intent,
            "operation": self.operation.value,
            "question_shape": self.question_shape.value,
            "target_class": self.target_class,
            "target_dish": self.target_dish,
            "target_labels": list(self.target_labels),
            "forbidden_labels": list(self.forbidden_labels),
            "retrieval_mode": self.retrieval_mode,
            "geo_scope": self.geo_scope,
            "anchors": list(self.anchors),
            "answer_mode": self.answer_mode,
            "contract_name": self.contract_name,
            "hard_label_contract": self.hard_label_contract,
            "semantic_category": self.semantic_category,
        }

    @property
    def comparison_subjects(self) -> List[str]:
        return list(self.anchors)

    @property
    def metadata(self) -> Dict[str, Any]:
        # Return a dict containing target_entity and entities for compatibility with from_query_state
        return {
            "target_entity": self.target_entity,
            "entities": [{"name": a} for a in self.anchors],
        }

    @property
    def coastal_required(self) -> bool:
        return any(getattr(c, "feature", c) == "coastal" for c in self.constraints)

    @property
    def sunset_required(self) -> bool:
        return any(getattr(c, "feature", c) == "sunset" for c in self.constraints)

    @property
    def island_required(self) -> bool:
        return any(getattr(c, "feature", c) == "island" for c in self.constraints)




@dataclass
class PipelineRuntime:
    """Mutable runtime state — được cập nhật qua Step 2/3/4/5.

    Chứa kết quả thực tế của pipeline execution.
    Tách ra khỏi metadata dict để rõ ràng về lifecycle.
    """

    # === Grounding (Step 2) ===
    grounded_nodes: List[Any] = field(default_factory=list)
    subject_grounding_status: Dict[str, str] = field(default_factory=dict)
    # {entity_name: "grounded" | "not_found" | "ambiguous"}

    # === Retrieval (Step 4) ===
    seed_nodes: List[Any] = field(default_factory=list)
    retrieval_confidence: float = 0.0
    missing_evidence: List[str] = field(default_factory=list)

    # === Context (Step 4-5) ===
    raw_context: List[str] = field(default_factory=list)
    clean_context: str = ""
    context_state: str = "empty"
    # "empty" | "partial" | "complete" | "hallucination_risk"

    # === Region (Step 2) ===
    region_resolved: bool = False
    resolved_location: str = ""
    region_focus: str = "all"


    # === Generation (Step 5) ===
    answer: str = ""
    answer_validation: Optional[Any] = None  # ValidationResult

    # === Debug ===
    step_timings: Dict[str, float] = field(default_factory=dict)

    # === Post-Step 1 Mutable Metadata ===
    metadata: Dict[str, Any] = field(default_factory=dict)

