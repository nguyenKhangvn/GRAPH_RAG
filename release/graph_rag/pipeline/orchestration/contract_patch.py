from __future__ import annotations
"""
ContractPatch — Immutable data structure returned by ContractValidator.

Thay vì ContractValidator mutate trực tiếp metadata/QueryState/QueryFrame,
nó trả về ContractPatch. QueryPlanBuilder.apply_contract_patch() sẽ áp dụng.

Ưu điểm:
- ContractValidator là pure function (input → output, no side effects)
- Dễ test: chỉ cần kiểm tra patch fields
- Dễ log: patch là data, không phải mutation
- QueryPlanBuilder là nơi duy nhất apply overrides
"""


from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class ContractPatch:
    """Immutable patch returned by ContractValidator.

    All fields are Optional — only set fields will be applied.
    None means "no override from contract".
    """

    # === Identity ===
    contract_name: Optional[str] = None  # ticket_price, emergency, etc.

    # === Intent & Mode ===
    intent: Optional[str] = None
    answer_mode: Optional[str] = None
    operator: Optional[str] = None  # QueryFrame operator override

    # === Labels ===
    target_labels: Optional[Tuple[str, ...]] = None
    forbidden_labels: Optional[Tuple[str, ...]] = None

    # === Target class ===
    target_class: Optional[str] = None
    semantic_category: Optional[str] = None

    # === Operation ===
    operation: Optional[str] = None  # QueryOperation value

    # === Attributes ===
    requested_attributes: Optional[Tuple[str, ...]] = None

    # === Flags ===
    hard_label_contract: Optional[bool] = None
    disable_agentic_retrieval: Optional[bool] = None
    disable_generic_discovery: Optional[bool] = None
    disable_discovery_expansion: Optional[bool] = None
    disable_food_keywords: Optional[bool] = None
    disable_entity_grounding: Optional[bool] = None
    disable_non_location_grounding: Optional[bool] = None
    exempt_location_from_grounding_filter: Optional[bool] = None
    skip_realtime_booking_guard: Optional[bool] = None

    # === Policy ===
    fallback_policy: Optional[str] = None

    # === Geo/Time constraints ===
    geo_scope: Optional[str] = None  # "multi_region", etc.
    month_constraint: Optional[Tuple[int, ...]] = None

    # === Query shape override ===
    question_shape: Optional[str] = None  # "advice", etc.

    # === Relations ===
    requested_relations: Optional[Tuple[str, ...]] = None

    # === Intent tracking ===
    original_intent: Optional[str] = None
    topic: Optional[str] = None  # "community", "event", etc.

    # === Target class priority ===
    target_class_priority: Optional[str] = None  # "Dish" for food specialty

    # === Entity corrections ===
    # List of (entity_index, new_type, source_suffix) tuples
    entity_corrections: Tuple[Tuple[int, str, str], ...] = ()

    # === Extra metadata overrides ===
    # For fields that don't fit standard categories
    extra_metadata: Tuple[Tuple[str, Any], ...] = ()

    def has_overrides(self) -> bool:
        """Check if this patch has any overrides."""
        return self.contract_name is not None

    def to_debug_dict(self) -> Dict[str, Any]:
        """Debug representation."""
        result = {}
        for k, v in self.__dict__.items():
            if v is not None and v != () and v != ():
                result[k] = v
        return result
