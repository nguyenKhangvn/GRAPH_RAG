from __future__ import annotations
"""
QueryPlanBuilder — Single source of truth for building QueryPlan.

Priority: ContractPatch > QueryFrame > QueryState > V3/LLM

Chỉ class này được quyền quyết định priority giữa các nguồn.
"""


import logging
from typing import Any, Dict, Optional

from graph_rag.core.state import QueryOperation, QuestionShape
from graph_rag.pipeline.orchestration.contract_patch import ContractPatch
from graph_rag.pipeline.orchestration.query_frame_contract import QueryFrame
from graph_rag.pipeline.orchestration.query_plan import QueryPlan
from graph_rag.utils.text import normalize_text

_logger = logging.getLogger(__name__)


class QueryPlanBuilder:
    """Build QueryPlan from metadata + QueryState + QueryFrame + ContractPatch.

    Đây là nơi DUY NHẤT được quyền quyết định priority:
    ContractPatch > QueryFrame > QueryState > V3/LLM

    Usage:
        plan = QueryPlanBuilder().build(
            query=state.user_query,
            metadata=metadata,
            query_state=state.query_state,
            query_frame=state.query_frame,
            contract_patch=contract_patch,  # NEW
        )
    """

    def build(
        self,
        *,
        query: str,
        metadata: Dict[str, Any],
        query_state: Any = None,  # QueryState
        query_frame: Optional[QueryFrame] = None,
        contract_patch: Optional[ContractPatch] = None,
        contract_name: Optional[str] = None,
    ) -> QueryPlan:
        """Build a frozen QueryPlan from all sources.

        Args:
            query: Raw user query
            metadata: Runtime metadata dict
            query_state: Canonical QueryState built at end of Step 1
            query_frame: Semantic contract (QueryFrame from query_frame_contract.py)
            contract_patch: Optional ContractPatch from ContractValidator
            contract_name: Name of active contract (deprecated, use contract_patch.contract_name)
        """
        # Apply ContractPatch to metadata first (for backward compat)
        if contract_patch and contract_patch.has_overrides():
            self._apply_contract_patch_to_metadata(contract_patch, metadata)

        if query_state is None:
            from graph_rag.core.query_fields import build_query_fields
            query_state = build_query_fields(
                query=query,
                metadata=metadata,
                answer_mode=metadata.get("answer_mode", ""),
                query_frame=metadata if metadata.get("query_frame_applied") else None,
            )

        # === 1. Base layer from QueryState ===
        qs = query_state
        qf = query_frame

        # Question shape
        question_shape = self._resolve_question_shape(qs, metadata)
        question_shape_source = getattr(qs, "question_shape_source", None) or "metadata"
        question_shape_confidence = float(getattr(qs, "question_shape_confidence", 0.0) or 0.0)

        # Operation
        operation = self._resolve_operation(qs, metadata)
        operation_source = getattr(qs, "operation_source", None) or "metadata"
        operation_confidence = float(getattr(qs, "operation_confidence", 0.0) or 0.0)

        # Target class — contract override > QueryState
        target_class = self._resolve_target_class(qs, metadata)
        target_class_source = getattr(qs, "target_class_source", None) or "metadata"
        target_class_confidence = float(getattr(qs, "target_class_confidence", 0.0) or 0.0)

        # Target dish
        target_dish = getattr(qs, "target_dish", None) or metadata.get("target_dish")
        target_dish_source = getattr(qs, "target_dish_source", None) or metadata.get("target_dish_source")
        target_dish_confidence = float(getattr(qs, "target_dish_confidence", 0.0) or 0.0)

        # Target entity
        target_entity = metadata.get("target_entity") or getattr(qs, "target_entity", None)

        # Requested attributes — contract overrides
        requested_attributes = self._resolve_requested_attributes(qs, metadata)

        # Requested relations
        requested_relations = tuple(getattr(qs, "requested_relations", ()) or ())
        if not requested_relations:
            requested_relations = tuple(metadata.get("requested_relations") or ())

        # Constraints
        constraints = self._resolve_constraints(qs, metadata)

        # Semantic category
        semantic_category = getattr(qs, "semantic_category", None) or metadata.get("semantic_category")
        semantic_category_confidence = float(getattr(qs, "semantic_category_confidence", 0.0) or 0.0)

        # Follow-up — also check resolved frame from ConversationStateResolver
        resolved = metadata.get("resolved_query_frame")
        is_follow_up = bool(
            getattr(qs, "is_follow_up", False)
            or metadata.get("is_follow_up", False)
            or (resolved and resolved.is_follow_up)
        )

        # Duration
        duration_days = int(getattr(qs, "duration_days", 0) or 0)
        duration_nights = int(getattr(qs, "duration_nights", 0) or 0)
        # === 2. QueryFrame layer ===
        intent = self._resolve_intent(qf, metadata)
        original_intent = str(metadata.get("original_intent") or intent).upper()

        # Fix: if target_dish detected but intent is not food-related → override
        # e.g. "Cà phê Pleiku có hương vị như thế nào" → TRAVEL_ADVICE nhưng có target_dish
        if target_dish and "FOOD" not in intent.upper() and intent.upper() not in {
            "TOUR_PLAN", "ACCOMMODATION_RECOMMENDATION", "EVENT_RECOMMENDATION",
            "DISTANCE_QUERY", "ENTITY_FACT_QUERY",
        }:
            _logger.info("   -> [IntentOverride] target_dish='%s' overrides %s → FOOD_RECOMMENDATION", target_dish, intent)
            intent = "FOOD_RECOMMENDATION"

        operator = self._resolve_operator(qf, metadata)
        answer_mode = str(metadata.get("answer_mode") or "fact_answer")
        topic = (contract_patch.topic if contract_patch else None) or metadata.get("topic")
        # Geographic scope
        geo_scope = str(metadata.get("region_focus") or "all")
        province = str(metadata.get("current_province") or "")
        legacy_province = metadata.get("legacy_province") or None
        region = metadata.get("display_region") or None

        # Region focus — respect ConversationStateResolver for follow-ups
        resolved = metadata.get("resolved_query_frame")
        if resolved and resolved.is_follow_up and resolved.region_focus != "all":
            region_focus = resolved.region_focus
        else:
            region_focus = str(metadata.get("region_focus") or "all")
        region_group = metadata.get("region_group") or None

        # Target labels — contract override > QueryFrame > metadata
        target_labels = self._resolve_target_labels(qf, metadata)
        forbidden_labels = tuple(metadata.get("forbidden_labels") or ())

        # Evidence constraints from QueryFrame
        required_evidence = tuple(metadata.get("required_evidence") or ())
        forbidden_fallbacks = tuple(metadata.get("forbidden_fallbacks") or ())

        # Anchors
        anchors, anchor_types = self._resolve_anchors(qf, metadata)

        # Max results
        max_results = int(metadata.get("max_results") or 10)

        # Month constraint
        month_constraint_raw = metadata.get("month_constraint")
        month_constraint = tuple(month_constraint_raw) if month_constraint_raw else None

        # === 3. RetrievalPlan layer ===
        retrieval_mode = str(metadata.get("retrieval_plan_mode") or "single_anchor")
        required_relations_for_retrieval = tuple(
            metadata.get("query_frame_traversal_relations") or ()
        )
        context_policy_raw = metadata.get("query_frame_context_policy") or {}
        context_policy = tuple(sorted(context_policy_raw.items())) if isinstance(context_policy_raw, dict) else ()

        # === 4. Contract flags ===
        hard_label_contract = bool(metadata.get("hard_label_contract", False))
        disable_agentic_retrieval = bool(metadata.get("disable_agentic_retrieval", False))
        disable_generic_discovery = bool(metadata.get("disable_generic_discovery", False))
        disable_discovery_expansion = bool(metadata.get("disable_discovery_expansion", False))
        disable_food_keywords = bool(metadata.get("disable_food_keywords", False))
        disable_entity_grounding = bool(metadata.get("disable_entity_grounding", False))
        disable_non_location_grounding = bool(metadata.get("disable_non_location_grounding", False))
        skip_realtime_booking_guard = bool(metadata.get("skip_realtime_booking_guard", False))
        fallback_policy = metadata.get("fallback_policy")

        # === 4b. Region policy ===
        region_lock_mode = metadata.get("region_lock_mode")
        exempt_location_from_grounding_filter = bool(metadata.get("exempt_location_from_grounding_filter", False))

        # === 4c. Renderer hint ===
        renderer = self._resolve_renderer(retrieval_mode, operator, metadata)

        # === 5. Build frozen QueryPlan ===
        plan = QueryPlan(
            # Query identity
            query=query,
            query_norm=normalize_text(query, strip_punct=True),
            # From QueryState
            question_shape=question_shape,
            operation=operation,
            target_class=target_class,
            target_dish=target_dish,
            target_entity=target_entity,
            requested_attributes=requested_attributes,
            requested_relations=requested_relations,
            constraints=constraints,
            semantic_category=semantic_category,
            is_follow_up=is_follow_up,
            duration_days=duration_days,
            duration_nights=duration_nights,
            # From QueryFrame
            intent=intent,
            original_intent=original_intent,
            topic=topic,
            operator=operator,
            answer_mode=answer_mode,

            # Geographic
            geo_scope=geo_scope,
            province=province,
            legacy_province=legacy_province,
            region=region,
            region_focus=region_focus,
            region_group=region_group,
            # Retrieval targets
            target_labels=target_labels,
            forbidden_labels=forbidden_labels,
            # Anchors
            anchors=anchors,
            anchor_types=anchor_types,
            # Constraints
            max_results=max_results,
            month_constraint=month_constraint,
            # RetrievalPlan
            retrieval_mode=retrieval_mode,
            required_relations_for_retrieval=required_relations_for_retrieval,
            context_policy=context_policy,
            # Source tracking
            target_class_source=target_class_source,
            target_class_confidence=target_class_confidence,
            target_dish_source=target_dish_source,
            target_dish_confidence=target_dish_confidence,
            question_shape_source=question_shape_source,
            question_shape_confidence=question_shape_confidence,
            operation_source=operation_source,
            operation_confidence=operation_confidence,
            semantic_category_confidence=semantic_category_confidence,
            # Contract flags
            hard_label_contract=hard_label_contract,
            disable_agentic_retrieval=disable_agentic_retrieval,
            disable_generic_discovery=disable_generic_discovery,
            disable_discovery_expansion=disable_discovery_expansion,
            disable_food_keywords=disable_food_keywords,
            disable_entity_grounding=disable_entity_grounding,
            disable_non_location_grounding=disable_non_location_grounding,
            skip_realtime_booking_guard=skip_realtime_booking_guard,
            fallback_policy=fallback_policy,
            # Evidence constraints
            required_evidence=required_evidence,
            forbidden_fallbacks=forbidden_fallbacks,
            # Region policy
            region_lock_mode=region_lock_mode,
            exempt_location_from_grounding_filter=exempt_location_from_grounding_filter,
            # Renderer
            renderer=renderer,
            # Audit
            built_from="query_plan_builder_v1",
            contract_name=(
                (contract_patch.contract_name if contract_patch else None)
                or contract_name
                or metadata.get("_active_contract")
            ),
        )

        _logger.debug("QueryPlan built: %s", plan.to_debug_dict())
        return plan

    # --- Resolution helpers ---

    def _resolve_question_shape(self, qs: Any, metadata: Dict[str, Any]) -> QuestionShape:
        """Resolve question_shape: QueryState > metadata."""
        shape = getattr(qs, "question_shape", None)
        if shape and shape != QuestionShape.UNKNOWN:
            return shape
        raw = metadata.get("question_shape")
        if raw:
            try:
                return QuestionShape(raw)
            except (ValueError, KeyError):
                pass
        return QuestionShape.UNKNOWN

    def _resolve_operation(self, qs: Any, metadata: Dict[str, Any]) -> QueryOperation:
        """Resolve operation: QueryState > metadata."""
        op = getattr(qs, "operation", None)
        if op and op != QueryOperation.DISCOVERY:
            return op
        raw = metadata.get("operation")
        if raw:
            try:
                return QueryOperation(raw)
            except (ValueError, KeyError):
                pass
        return QueryOperation.DISCOVERY

    def _resolve_target_class(self, qs: Any, metadata: Dict[str, Any]) -> Optional[str]:
        """Resolve target_class: metadata (contract may override) > QueryState."""
        # Contract overrides are already in metadata at this point
        tc = metadata.get("target_class")
        if tc:
            return tc
        return getattr(qs, "target_class", None)

    def _resolve_requested_attributes(self, qs: Any, metadata: Dict[str, Any]) -> tuple[str, ...]:
        """Resolve requested_attributes: metadata (contract) > QueryState."""
        # Contract may have set metadata["requested_attributes"]
        from_metadata = metadata.get("requested_attributes")
        if from_metadata:
            return tuple(from_metadata)
        from_qs = getattr(qs, "requested_attributes", None)
        if from_qs:
            return tuple(from_qs)
        return ()

    def _resolve_constraints(self, qs: Any, metadata: Dict[str, Any]) -> tuple[Any, ...]:
        """Resolve constraints as ConstraintSpec tuple."""
        from graph_rag.core.state import ConstraintSpec

        specs = getattr(qs, "constraints", None)
        if specs:
            resolved = []
            for s in specs:
                if isinstance(s, ConstraintSpec):
                    resolved.append(s)
                elif isinstance(s, str):
                    resolved.append(ConstraintSpec(feature=s))
                elif isinstance(s, dict):
                    resolved.append(ConstraintSpec(
                        feature=s.get("feature", ""),
                        weight=s.get("weight", 1.0),
                        is_hard=s.get("is_hard", False),
                    ))
            return tuple(resolved)

        raw = metadata.get("constraints")
        if isinstance(raw, dict):
            resolved = []
            for k, v in raw.items():
                if isinstance(v, dict):
                    resolved.append(ConstraintSpec(
                        feature=k,
                        weight=v.get("weight", 1.0),
                        is_hard=v.get("is_hard", False),
                    ))
                elif isinstance(v, bool):
                    resolved.append(ConstraintSpec(feature=k, is_hard=v))
                else:
                    resolved.append(ConstraintSpec(feature=k))
            return tuple(resolved)

        if isinstance(raw, (list, tuple)):
            resolved = []
            for s in raw:
                if isinstance(s, ConstraintSpec):
                    resolved.append(s)
                elif isinstance(s, str):
                    resolved.append(ConstraintSpec(feature=s))
                elif isinstance(s, dict):
                    resolved.append(ConstraintSpec(
                        feature=s.get("feature", ""),
                        weight=s.get("weight", 1.0),
                        is_hard=s.get("is_hard", False),
                    ))
            return tuple(resolved)

        return ()

    def _resolve_intent(self, qf: Optional[QueryFrame], metadata: Dict[str, Any]) -> str:
        """Resolve intent: metadata (contract override) > QueryFrame > resolved frame > metadata fallback."""
        # Contract overrides are already in metadata["intent"]
        intent = metadata.get("intent")
        if intent:
            return str(intent).upper()
        if qf and qf.intent != "UNKNOWN":
            return qf.intent
        # Fallback: check resolved frame from ConversationStateResolver
        resolved = metadata.get("resolved_query_frame")
        if resolved and resolved.is_follow_up and resolved.intent:
            return str(resolved.intent).upper()
        return "UNKNOWN"

    def _resolve_operator(self, qf: Optional[QueryFrame], metadata: Dict[str, Any]) -> str:
        """Resolve operator: QueryFrame > metadata."""
        if qf and qf.operator != "default":
            return qf.operator
        frame_dict = metadata.get("query_frame") or {}
        op = frame_dict.get("query_operator")
        if op:
            return op
        return "default"

    def _resolve_target_labels(
        self, qf: Optional[QueryFrame], metadata: Dict[str, Any]
    ) -> tuple[str, ...]:
        """Resolve target_labels: metadata (contract) > QueryFrame > metadata fallback."""
        # Contract sets metadata["retrieval_allowed_labels"]
        from_metadata = metadata.get("retrieval_allowed_labels")
        if from_metadata:
            return tuple(from_metadata)
        if qf and qf.target_labels:
            return tuple(qf.target_labels)
        return ()

    def _resolve_anchors(
        self, qf: Optional[QueryFrame], metadata: Dict[str, Any]
    ) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
        """Resolve anchors: metadata > QueryFrame."""
        # metadata may have query_frame_anchor_names from QueryFrameStage
        anchor_names = metadata.get("query_frame_anchor_names") or []
        if not anchor_names and qf:
            anchor_names = list(qf.anchors)

        anchors = tuple(str(a) for a in anchor_names if a)

        # Anchor types from QueryFrame
        anchor_types_raw: Dict[str, str] = {}
        if qf and qf.anchor_types:
            anchor_types_raw.update(qf.anchor_types)
        # Also from metadata entities
        for ent in (metadata.get("entities") or []):
            if isinstance(ent, dict):
                name = str(ent.get("name") or "").strip()
                etype = str(ent.get("type") or "Unknown")
                if name and name not in anchor_types_raw:
                    anchor_types_raw[name] = etype

        anchor_types = tuple(
            (k, v) for k, v in anchor_types_raw.items() if k in set(anchors)
        )

        return anchors, anchor_types

    def _apply_contract_patch_to_metadata(
        self, patch: ContractPatch, metadata: Dict[str, Any]
    ) -> None:
        """Apply ContractPatch overrides to metadata dict.

        This ensures backward compatibility — downstream code that still reads
        from metadata will see the same values as QueryPlan.
        """
        if patch.intent is not None:
            metadata["intent"] = patch.intent
        if patch.operator is not None:
            if not isinstance(metadata.get("query_frame"), dict):
                metadata["query_frame"] = {}
            metadata["query_frame"]["query_operator"] = patch.operator
        if patch.target_labels is not None:
            metadata["retrieval_allowed_labels"] = list(patch.target_labels)
        if patch.forbidden_labels is not None:
            metadata["forbidden_labels"] = list(patch.forbidden_labels)
        if patch.fallback_policy is not None:
            metadata["fallback_policy"] = patch.fallback_policy
        if patch.hard_label_contract is not None:
            metadata["hard_label_contract"] = patch.hard_label_contract
        if patch.disable_agentic_retrieval is not None:
            metadata["disable_agentic_retrieval"] = patch.disable_agentic_retrieval
        if patch.disable_generic_discovery is not None:
            metadata["disable_generic_discovery"] = patch.disable_generic_discovery
        if patch.disable_discovery_expansion is not None:
            metadata["disable_discovery_expansion"] = patch.disable_discovery_expansion
        if patch.disable_food_keywords is not None:
            metadata["disable_food_keywords"] = patch.disable_food_keywords
        if patch.exempt_location_from_grounding_filter is not None:
            metadata["exempt_location_from_grounding_filter"] = patch.exempt_location_from_grounding_filter
        if patch.disable_entity_grounding is not None:
            metadata["disable_entity_grounding"] = patch.disable_entity_grounding
        if patch.disable_non_location_grounding is not None:
            metadata["disable_non_location_grounding"] = patch.disable_non_location_grounding
        if patch.skip_realtime_booking_guard is not None:
            metadata["skip_realtime_booking_guard"] = patch.skip_realtime_booking_guard
        if patch.geo_scope is not None:
            metadata["geo_scope"] = patch.geo_scope
        if patch.month_constraint is not None:
            metadata["month_constraint"] = list(patch.month_constraint)
        if patch.question_shape is not None:
            metadata["question_shape"] = patch.question_shape
        if patch.requested_relations is not None:
            metadata["requested_relations"] = list(patch.requested_relations)
        if patch.original_intent is not None:
            metadata["original_intent"] = patch.original_intent
        if patch.topic is not None:
            metadata["topic"] = patch.topic
        if patch.target_class_priority is not None:
            metadata["target_class_priority"] = patch.target_class_priority
        if patch.answer_mode is not None:
            metadata["answer_mode"] = patch.answer_mode
        if patch.target_class is not None:
            metadata["target_class"] = patch.target_class
        if patch.semantic_category is not None:
            metadata["semantic_category"] = patch.semantic_category
        if patch.requested_attributes is not None:
            metadata["requested_attributes"] = list(patch.requested_attributes)
        if patch.operation is not None:
            metadata["operation"] = patch.operation

        # Contract flag for backward compat
        if patch.contract_name:
            metadata["_active_contract"] = patch.contract_name
            metadata[f"{patch.contract_name}_contract_active"] = True

        # Extra metadata overrides
        for key, value in patch.extra_metadata:
            metadata[key] = value

    def _resolve_renderer(
        self, retrieval_mode: str, operator: str, metadata: Dict[str, Any]
    ) -> Optional[str]:
        """Resolve renderer hint for Step 5 dispatch.

        Renderer is a hint that tells Step 5 which rendering strategy to use.
        It combines retrieval_mode and operator into a single dispatch key.
        """
        # Comparison takes priority
        if retrieval_mode == "comparison" or operator == "comparison":
            return "comparison"
        # Dish-to-restaurant
        if retrieval_mode == "dish_to_restaurant" or operator == "dish_to_restaurant":
            return "dish_to_restaurant"
        # Tour plan
        if retrieval_mode == "tour_plan":
            return "tour_plan"
        # Lodging near anchor
        if retrieval_mode == "lodging_near_anchor":
            return "lodging_near_anchor"
        # Constrained nearby
        if retrieval_mode == "constrained_nearby_search":
            return "constrained_nearby"
        # Global discovery
        if retrieval_mode == "global_discovery" or operator == "global_discovery":
            return "global_discovery"
        # Multi candidate (food recommendation)
        if retrieval_mode == "multi_candidate":
            return "multi_candidate"
        # Class search (tour availability)
        if retrieval_mode == "class_search":
            return "class_search"
        # Ticket price
        if operator == "ticket_price_lookup":
            return "ticket_price"
        # No specific renderer
        return None
