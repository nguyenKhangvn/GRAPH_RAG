from __future__ import annotations
"""
PlanDiffLogger — Log differences between old metadata-based logic and new QueryPlan.

Dùng trong giai đoạn chuyển đổi để biết chỗ nào plan mới khác logic cũ
trước khi cắt fallback.

Usage:
    PlanDiffLogger.log(plan, metadata)
"""


import logging
from typing import Any, Dict, List

from graph_rag.pipeline.orchestration.query_plan import QueryPlan

_logger = logging.getLogger(__name__)


class PlanDiffLogger:
    """Log differences between QueryPlan and metadata-based values.

    Mỗi request log:
        metadata.intent = FOOD_RECOMMENDATION
        query_plan.intent = FOOD_RECOMMENDATION
        agreement = true/false
    """

    # Fields to compare: (plan_field, metadata_key, label)
    _COMPARE_FIELDS = [
        ("intent", "intent", "Intent"),
        ("target_class", "target_class", "Target Class"),
        ("answer_mode", "answer_mode", "Answer Mode"),
        ("geo_scope", "region_focus", "Geo Scope"),
        ("semantic_category", "semantic_category", "Semantic Category"),
        ("fallback_policy", "fallback_policy", "Fallback Policy"),
    ]

    # Fields that need set comparison
    _SET_FIELDS = [
        ("target_labels", "retrieval_allowed_labels", "Target Labels"),
        ("forbidden_labels", "forbidden_labels", "Forbidden Labels"),
    ]

    @classmethod
    def log(cls, plan: QueryPlan, metadata: Dict[str, Any]) -> None:
        """Compare QueryPlan with metadata and log differences.

        Args:
            plan: The newly built QueryPlan
            metadata: The metadata dict (old source of truth)
        """
        diffs: List[str] = []
        agreements = 0
        total = 0

        # Compare scalar fields
        for plan_field, meta_key, label in cls._COMPARE_FIELDS:
            plan_val = getattr(plan, plan_field, None)
            meta_val = metadata.get(meta_key)

            # Normalize for comparison
            plan_str = str(plan_val or "").upper()
            meta_str = str(meta_val or "").upper()

            total += 1
            if plan_str == meta_str:
                agreements += 1
            else:
                diffs.append(
                    f"   {label}: metadata={meta_val!r} | plan={plan_val!r}"
                )

        # Compare set fields
        for plan_field, meta_key, label in cls._SET_FIELDS:
            plan_val = set(getattr(plan, plan_field, ()) or ())
            meta_val = set(metadata.get(meta_key) or [])

            total += 1
            if plan_val == meta_val:
                agreements += 1
            else:
                diffs.append(
                    f"   {label}: metadata={sorted(meta_val)} | plan={sorted(plan_val)}"
                )

        # Compare operation
        plan_op = plan.operation.value if hasattr(plan.operation, "value") else str(plan.operation)
        meta_op = str(metadata.get("operation") or "").upper()
        total += 1
        if plan_op.upper() == meta_op or not meta_op:
            agreements += 1
        else:
            diffs.append(f"   Operation: metadata={meta_op!r} | plan={plan_op!r}")

        # Compare retrieval_mode
        plan_mode = plan.retrieval_mode
        meta_mode = str(metadata.get("retrieval_plan_mode") or "single_anchor")
        total += 1
        if plan_mode == meta_mode:
            agreements += 1
        else:
            diffs.append(f"   Retrieval Mode: metadata={meta_mode!r} | plan={plan_mode!r}")

        # Compare hard_label_contract
        plan_hlc = plan.hard_label_contract
        meta_hlc = bool(metadata.get("hard_label_contract", False))
        total += 1
        if plan_hlc == meta_hlc:
            agreements += 1
        else:
            diffs.append(f"   Hard Label Contract: metadata={meta_hlc} | plan={plan_hlc}")

        # Summary
        agreement_rate = agreements / total if total > 0 else 1.0

        if diffs:
            _logger.info(
                "[PlanDiffLogger] Query: '%s' | Agreement: %d/%d (%.0f%%)\n%s",
                plan.query[:50],
                agreements,
                total,
                agreement_rate * 100,
                "\n".join(diffs),
            )
            # Also print for visibility during development
            _logger.info(
                f"   -> [PlanDiffLogger] Agreement: {agreements}/{total} ({agreement_rate:.0%})"
            )
            for diff in diffs:
                _logger.info(diff)
        else:
            _logger.debug(
                "[PlanDiffLogger] Full agreement for: '%s'", plan.query[:50]
            )
