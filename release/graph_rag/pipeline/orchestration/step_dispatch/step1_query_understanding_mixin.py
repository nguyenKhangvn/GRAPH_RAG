from __future__ import annotations
"""Step 1: Query understanding, entity extraction, and intent classification."""

import logging
import re
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

from graph_rag.config import ENABLE_QUERY_FRAME_V2, QUERY_FRAME_DEBUG_LOG, ENABLE_ROLE_AWARE_GROUNDING, GRAPH_RAG_V3_ENABLED
from graph_rag.core.answer_mode import AnswerMode, infer_answer_mode
from graph_rag.core.intents import IntentType, RegionFocus
from graph_rag.config.deictic_patterns import is_deictic_entity_phrase
from graph_rag.config.region_patterns import (
    MULTI_REGION_NAMES,
)
from graph_rag.utils.text import normalize_text
from ..conversation_state_resolver import ConversationStateResolver
from ..dto import PipelineRunState
from graph_rag.modules.pipeline_support.distance_intent_service import DistanceQueryParser


class Step1QueryUnderstandingMixin:
    """Mixin providing Step 1 query understanding."""

    @staticmethod
    def _inject_entity_hint(
        entities: list,
        metadata: dict,
        hint: str,
        entity_type: str,
        *,
        metadata_key: str | None = None,
        set_target: bool = False,
        log_label: str = "",
    ) -> list:
        """Normalize, deduplicate, and prepend an entity hint into the entities list.

        Args:
            entities: Current entities list (may be None).
            metadata: Metadata dict to update.
            hint: The extracted hint string.
            entity_type: Entity type to assign (e.g. "Restaurant", "Place").
            metadata_key: If set, metadata[key] = hint (always, even if dupe).
            set_target: If True, metadata["target_entity"] = hint.
            log_label: Label for the info log message.

        Returns:
            The (possibly updated) entities list.
        """
        hint_norm = normalize_text(hint, strip_punct=True)
        existing_norms = {
            normalize_text(str(e.get("name") or ""), strip_punct=True)
            for e in (entities or [])
            if isinstance(e, dict)
        }
        if hint_norm and hint_norm not in existing_norms:
            entities = [{"name": hint, "type": entity_type}] + list(entities or [])
            metadata["entities"] = entities
            logger.info("   -> %s entity hint extracted: '%s'", log_label, hint)
        if metadata_key:
            metadata[metadata_key] = hint
        if set_target:
            metadata["target_entity"] = hint
        return entities

    # ------------------------------------------------------------------
    # Sub-method 1: Location detection, region focus, admin mapping
    # ------------------------------------------------------------------
    def _resolve_location_context(
        self,
        state: PipelineRunState,
        current_location: str,
    ) -> Dict[str, Any]:
        """Fast-path check, LLM analyzer call, initial location context, ConversationStateResolver."""
        p = self.pipeline

        # --- Fast-path: skip LLM for simple follow-ups ---
        conversation_state = p.location_grounding_service.conversation_state
        query_norm = normalize_text(state.user_query, strip_punct=True)
        resolver_for_fast = ConversationStateResolver()

        fast_path_metadata = None
        if resolver_for_fast._has_other_marker(query_norm):
            # Quick pre-check: is this a simple "other/more" pattern?
            if not resolver_for_fast._has_new_entity_signal(query_norm, {}):
                fast_path_metadata = resolver_for_fast._build_follow_up_metadata(
                    state.user_query, conversation_state
                )

        if fast_path_metadata is not None:
            metadata = fast_path_metadata
            logger.info("   -> FastPath: skipped LLM analyzer for simple follow-up (~52s saved)")
        else:
            metadata = p.query_analyzer.analyze(
                query=state.user_query,
                history=state.history,
                current_location=current_location,
            )
        # Preserve evaluator-provided metadata (question_type, choices,
        # negative flags, evidence ids) because analyzer output is rebuilt
        # from scratch for production queries.
        if state.metadata:
            metadata.update({k: v for k, v in state.metadata.items() if v is not None})

        location_context = p._build_initial_location_context(current_location, state.history, analyzer_output=metadata)
        loc = location_context.get("name") or ""
        # Flag GPS-sourced location so downstream can avoid search bias
        if location_context.get("source") == "user":
            metadata["location_from_gps"] = True
        search_query = metadata.get("rewritten_query", state.user_query)
        current_intents = metadata.get("intents", [IntentType.DISCOVERY])
        entities = metadata.get("entities", [])
        # Unified Contract: save analyzer intent as immutable reference.
        # Downstream overrides (QueryFrame, V3 router) may only upgrade
        # DISCOVERY → specific intent, never downgrade specific → DISCOVERY.
        metadata["_analyzer_primary_intent"] = (
            current_intents[0] if current_intents else IntentType.DISCOVERY
        )

        # ConversationStateResolver: resolve follow-up context early
        resolver = ConversationStateResolver()
        resolved = resolver.resolve(
            current_query=state.user_query,
            metadata=metadata,
            conversation_state=p.location_grounding_service.conversation_state,
        )
        if resolved.is_follow_up:
            metadata["resolved_query_frame"] = resolved
            state.resolved_query_frame = resolved
            logger.debug("FollowUp Resolved: region='%s', intent='%s', target_class='%s', source='%s'",
                        resolved.region_focus, resolved.intent, resolved.target_class, resolved.inheritance_source)
            # Apply inherited effective_location if query does not have explicit location
            if resolved.effective_location and not metadata.get("has_explicit_location"):
                loc = resolved.effective_location
                location_context = p._build_initial_location_context(loc, state.history, analyzer_output=metadata)
                metadata["detected_location"] = loc
                logger.info("   -> [FollowUp] Inherited location: '%s'", loc)
            # Apply inherited region_focus
            if resolved.region_focus and resolved.region_focus != "all":
                metadata["region_focus"] = resolved.region_focus
                metadata["region_focus_source"] = "conversation_state_resolver"
                # Mark on resolved frame so admin mapping guard can check later
                resolved._region_inherited = True
                logger.info("   -> [FollowUp] Inherited region_focus: '%s'", resolved.region_focus)

        return {
            "metadata": metadata,
            "location_context": location_context,
            "loc": loc,
            "search_query": search_query,
            "current_intents": current_intents,
            "entities": entities,
        }

    # ------------------------------------------------------------------
    # Sub-method 2: Entity extraction, hint injection, merge/dedup, type correction, pruning
    # ------------------------------------------------------------------
    def _extract_and_classify_entities(
        self,
        state: PipelineRunState,
        metadata: dict,
        location_context: dict,
        loc: str,
        search_query: str,
        current_intents: list,
        entities: list,
    ) -> Dict[str, Any]:
        """Entity inheritance, deictic injection, history filter, hint extraction,
        location finalization from entities, intent overrides, policy, region focus,
        coreference resolution, V3 router, merge/dedup, type correction, pruning."""
        p = self.pipeline

        # Entity Inheritance: inject main entity from previous query for follow-up
        is_follow_up = bool(metadata.get("is_follow_up", False))
        if is_follow_up and not entities:
            prev_entity = p.location_grounding_service.conversation_state.get("last_active_entity") or {}
            if prev_entity.get("name"):
                entities = [prev_entity]
                metadata["entities"] = entities
                metadata["entity_inherited"] = True
                logger.info("   -> Entity inherited (follow-up, no entities): '%s'", prev_entity.get('name'))

        # Deictic Entity Injection: "quán này", "chỗ này" → dùng last_active_entity
        # Chạy TRƯỚC history filter để deictic entities không bị semantic search sai
        has_deictic_entity_in_entities = any(
            isinstance(e, dict) and is_deictic_entity_phrase(
                normalize_text(str(e.get("name") or ""), strip_punct=True)
            )
            for e in (entities or [])
        )
        if has_deictic_entity_in_entities:
            prev_entity = p.location_grounding_service.conversation_state.get("last_active_entity") or {}
            if prev_entity.get("name"):
                new_entities = []
                for e in (entities or []):
                    if isinstance(e, dict) and is_deictic_entity_phrase(
                        normalize_text(str(e.get("name") or ""), strip_punct=True)
                    ):
                        new_entities.append(prev_entity)
                    else:
                        new_entities.append(e)
                entities = new_entities
                metadata["entities"] = entities
                metadata["entity_inherited"] = True
                logger.info("   -> Deictic entity injection: replaced with '%s'", prev_entity.get('name'))
            else:
                logger.info("   -> Deictic entity detected but no last_active_entity available", )

        current_query_norm = normalize_text(state.user_query, strip_punct=True)
        has_deictic_reference = any(pattern in current_query_norm for pattern in self.DEICTIC_QUERY_PATTERNS)
        explicit_query_location = ""
        admin_region_service = getattr(p, "admin_region_mapping_service", None)
        if admin_region_service is not None:
            resolved = admin_region_service.resolve(state.user_query, entities)
            if resolved and resolved.get("matched_alias"):
                admin_level = resolved.get("admin_level", "")
                if admin_level in ("province", "area"):
                    explicit_query_location = resolved["matched_alias"]
        if explicit_query_location and not has_deictic_reference:
            if metadata.get("is_follow_up"):
                metadata["is_follow_up"] = False
                metadata["dialog_act"] = "NEW_QUERY"
                metadata["follow_up_overridden_by_explicit_location"] = explicit_query_location
            loc = explicit_query_location
            metadata["detected_location"] = explicit_query_location
            metadata["has_explicit_location"] = True
            # Fix search_query: if rewritten_query hallucinated a different location,
            # revert search_query to the original user query so vector/fulltext search
            # targets the correct region.
            rewritten_norm = normalize_text(search_query, strip_punct=True)
            if explicit_query_location.lower().replace(" ", " ") not in rewritten_norm:
                search_query = state.user_query
                logger.info(
                    "   -> Explicit location override: search_query reverted to original "
                    "(rewritten_query did not contain the explicit location)."
                )
            logger.info(
                "   -> Explicit location override: query location "
                f"'{explicit_query_location}' takes priority over current_location."
            )
        if not has_deictic_reference:
            filtered_entities = []
            for entity in entities or []:
                if not isinstance(entity, dict):
                    filtered_entities.append(entity)
                    continue
                source = str(entity.get("source") or "").strip().lower()
                name = str(entity.get("name") or "").strip()
                name_norm = normalize_text(name, strip_punct=True)
                if source == "history" and name_norm and name_norm not in current_query_norm:
                    continue
                filtered_entities.append(entity)
            if len(filtered_entities) != len(entities or []):
                metadata["history_entity_filter_applied"] = True
                metadata["entities"] = filtered_entities
                entities = filtered_entities
        detected_from_analyzer = metadata.get("detected_location") or ""
        has_explicit_location = bool(metadata.get("has_explicit_location")) or p._has_explicit_location(state.user_query, entities)
        if has_explicit_location and not detected_from_analyzer:
            for ent in entities or []:
                if isinstance(ent, dict):
                    etype = str(ent.get("type") or "").strip().lower()
                    if etype in {"location", "province", "city", "district", "ward", "commune"}:
                        e_name = str(ent.get("name") or "").strip()
                        if e_name:
                            detected_from_analyzer = e_name
                            metadata["detected_location"] = e_name
                            break
        disable_coreference = False
        address_lookup_entity_hint = self._extract_address_lookup_entity_hint(state.user_query)
        phone_lookup_entity_hint = self._extract_phone_lookup_entity_hint(state.user_query)
        opening_hours_entity_hint = self._extract_opening_hours_entity_hint(state.user_query)
        proximity_anchor_hint = self._extract_proximity_anchor_hint(state.user_query)
        analysis_subject_entity_hint = self._extract_analysis_subject_entity_hint(state.user_query)
        fill_blank_subject_entity_hint = self._extract_fill_blank_subject_entity_hint(state.user_query)
        statement_subject_entity_hint = self._extract_statement_subject_entity_hint(state.user_query)
        service_subject_entity_hint = self._extract_service_subject_entity_hint(state.user_query)
        if service_subject_entity_hint:
            statement_subject_entity_hint = ""
        if address_lookup_entity_hint:
            entities = self._inject_entity_hint(
                entities, metadata, address_lookup_entity_hint, "Restaurant",
                metadata_key="address_lookup_entity_hint", log_label="Address lookup",
            )
        if phone_lookup_entity_hint:
            entities = self._inject_entity_hint(
                entities, metadata, phone_lookup_entity_hint, "Place",
                metadata_key="phone_lookup_entity_hint", set_target=True,
                log_label="Phone lookup",
            )
        if opening_hours_entity_hint:
            entities = self._inject_entity_hint(
                entities, metadata, opening_hours_entity_hint, "TouristAttraction",
                metadata_key="opening_hours_entity_hint", set_target=True,
                log_label="Opening-hours",
            )
        if service_subject_entity_hint:
            entities = self._inject_entity_hint(
                entities, metadata, service_subject_entity_hint,
                self._infer_entity_type_from_hint(service_subject_entity_hint),
                metadata_key="service_subject_entity_hint", set_target=True,
                log_label="Service subject",
            )
        if proximity_anchor_hint:
            entities = self._inject_entity_hint(
                entities, metadata, proximity_anchor_hint, "Place",
                log_label="Proximity anchor",
            )
            # Use LLM-classified anchor type (from analyzer response)
            _llm_anchor_type = metadata.get("proximity_anchor_type")  # "generic_feature" | "named_entity" | None
            _grounding_required = (_llm_anchor_type or "named_entity") != "generic_feature"
            anchor_classified = {"text": proximity_anchor_hint, "type": _llm_anchor_type or "named_entity", "grounding_required": _grounding_required}
            metadata["proximity_anchor_required"] = True
            metadata["proximity_anchor"] = anchor_classified
            logger.info("   -> Proximity anchor classified: %s", anchor_classified)
        if analysis_subject_entity_hint:
            entities = self._inject_entity_hint(
                entities, metadata, analysis_subject_entity_hint,
                self._infer_entity_type_from_hint(analysis_subject_entity_hint),
                metadata_key="analysis_subject_entity_hint", set_target=True,
                log_label="Analysis subject",
            )
        if fill_blank_subject_entity_hint:
            entities = self._inject_entity_hint(
                entities, metadata, fill_blank_subject_entity_hint, "TouristAttraction",
                metadata_key="fill_blank_subject_entity_hint", set_target=True,
                log_label="Fill-blank",
            )
        if statement_subject_entity_hint:
            hint_norm = normalize_text(statement_subject_entity_hint, strip_punct=True)
            # Skip if hint is a multi-choice question prefix (not a real entity)
            _MC_PREFIXES = ["nhung cai nao", "cai nao duoi", "dau la", "nhung dia diem nao", "cac dia diem nao"]
            is_mc_prefix = any(hint_norm.startswith(pfx) for pfx in _MC_PREFIXES)
            if not is_mc_prefix:
                entities = self._inject_entity_hint(
                    entities, metadata, statement_subject_entity_hint,
                    self._infer_entity_type_from_hint(statement_subject_entity_hint),
                    metadata_key="statement_subject_entity_hint", set_target=True,
                    log_label="Statement subject",
                )

        # Multi-Select/Choice entity extraction: strip question prefix, extract anchor + category.
        # Keep this route scoped to actual option questions; open analysis prompts
        # often contain words such as "địa chỉ" or "lân cận" that are not anchors.
        is_option_question = (
            str(metadata.get("question_type") or "") in {"Multi-Choice", "Multi-Select"}
            or bool(re.search(r"(?im)^\s*[A-D]\s*[\).:-]\s+\S+", state.user_query or ""))
        )
        multi_choice_anchor = self._extract_multi_choice_anchor_hint(state.user_query) if is_option_question else ""
        multi_choice_category = self._extract_multi_choice_target_category(state.user_query) if is_option_question else ""
        if multi_choice_anchor:
            entities = self._inject_entity_hint(
                entities, metadata, multi_choice_anchor, "Place",
                metadata_key="multi_choice_anchor_hint", log_label="Multi-choice anchor",
            )
            # Override proximity_anchor: multi-choice extraction is more specific
            # Multi-choice anchors are always named entities (e.g., "như Eo Gió hay Kỳ Co")
            metadata["proximity_anchor_required"] = True
            metadata["proximity_anchor"] = {"text": multi_choice_anchor, "type": "named_entity", "grounding_required": True}
            # Override target_entity if multi-choice anchor is more specific
            # than statement_subject_entity_hint (which often captures question prefixes)
            if metadata.get("target_entity"):
                target_norm = normalize_text(metadata["target_entity"], strip_punct=True)
                anchor_norm = normalize_text(multi_choice_anchor, strip_punct=True)
                target_is_bad_anchor = (
                    any(target_norm.startswith(pfx) for pfx in ["nhung cai nao", "cai nao duoi", "dau la", "nhung dia diem nao"])
                    or any(marker in target_norm for marker in ["located_in", "belongs_to", "near", "has", "loai hinh", "moi quan he"])
                    or (anchor_norm and len(anchor_norm.split()) >= 3 and anchor_norm not in target_norm and len(target_norm.split()) > len(anchor_norm.split()) + 2)
                )
                if target_is_bad_anchor:
                    metadata["target_entity"] = multi_choice_anchor
            else:
                metadata["target_entity"] = multi_choice_anchor
        if multi_choice_category:
            metadata["multi_choice_target_category"] = multi_choice_category
            logger.info("   -> Multi-choice target category extracted: '%s'", multi_choice_category)

        return {
            "entities": entities,
            "metadata": metadata,
            "has_explicit_location": has_explicit_location,
            "detected_from_analyzer": detected_from_analyzer,
            "disable_coreference": disable_coreference,
            "loc": loc,
            "search_query": search_query,
            "address_lookup_entity_hint": address_lookup_entity_hint,
            "phone_lookup_entity_hint": phone_lookup_entity_hint,
            "opening_hours_entity_hint": opening_hours_entity_hint,
            "analysis_subject_entity_hint": analysis_subject_entity_hint,
        }

    # ------------------------------------------------------------------
    # Sub-method 3: Intent selection, override paths, retrieval policy resolution
    # ------------------------------------------------------------------
    def _resolve_intent_and_policy(
        self,
        state: PipelineRunState,
        metadata: dict,
        entities: list,
        loc: str,
        location_context: dict,
        search_query: str,
        current_intents: list,
        has_explicit_location: bool,
        detected_from_analyzer: str,
        disable_coreference: bool,
        address_lookup_entity_hint: str,
        phone_lookup_entity_hint: str,
        opening_hours_entity_hint: str,
        analysis_subject_entity_hint: str,
        step_1_start: float,
    ) -> Dict[str, Any]:
        """Location finalization from entities, intent selection/overrides,
        retrieval policy, admin region mapping, region focus, coreference resolution,
        V3 router, entity merge/dedup/type correction/pruning."""
        p = self.pipeline

        if has_explicit_location and detected_from_analyzer:
            location_context = p._build_location_context(
                name=detected_from_analyzer,
                source="user",
                reason="explicit_location_from_user",
                confidence=0.7,
            )
            loc = detected_from_analyzer
            disable_coreference = True
            p._clear_conversation_context(new_location=loc)
            metadata["disable_coreference"] = True
            metadata["mode"] = "new_context"
            logger.info(
                "   -> Mode switch NEW_CONTEXT: explicit location detected, "
                f"accept='{loc}', coreference disabled."
            )

        query_region_signal = p._query_region_signal(state.user_query, entities)
        anchor_region = p._location_to_region_focus(loc)
        detected_region = p._location_to_region_focus(detected_from_analyzer)
        if detected_from_analyzer and not has_explicit_location:
            analyzer_ctx = p._build_location_context(
                name=detected_from_analyzer,
                source="history",
                reason="analyzer_detected_location",
                confidence=0.45,
            )
            if loc and query_region_signal == "all" and anchor_region != "all" and detected_region not in {"all", anchor_region}:
                logger.info(
                    "   -> Hard geo anchor retained: "
                    f"anchor='{loc}', analyzer_detected='{detected_from_analyzer}'."
                )
            else:
                location_context, _ = p._choose_location_context(location_context, analyzer_ctx)
                loc = location_context.get("name") or loc

        primary_intent = p._select_primary_intent(current_intents, state.user_query)
        if self._is_nearby_accommodation_query(state.user_query):
            primary_intent = IntentType.ACCOMMODATION
            metadata["intent_override_reason"] = "nearby_accommodation_signal_detected"

        # Emergency queries (khẩn cấp, đường dây nóng, sự cố...) → route to DISCOVERY
        # Must be BEFORE phone lookup to prevent "liên hệ" false positive
        elif self._is_emergency_query(state.user_query):
            primary_intent = IntentType.DISCOVERY
            metadata["intent_override_reason"] = "emergency_query_detected"
            # Clear entity extraction — emergency queries have no specific entity
            phone_lookup_entity_hint = ""
            entities = []
            metadata["entities"] = []
            metadata.pop("target_entity", None)
            metadata.pop("phone_lookup_entity_hint", None)
            detected_from_analyzer = ""
            loc = ""
            location_context = p._build_location_context(
                name="",
                source="global",
                reason="emergency_no_location_filter",
                confidence=0.0,
            )
            has_explicit_location = False
            # Override retrieval labels to include TravelInfo
            metadata["retrieval_label_override"] = ["TravelInfo"]
            # Clear semantic_category to prevent policy resolver from blocking labels
            metadata["semantic_category"] = None
            logger.info("   -> Intent override: detected emergency query, forcing DISCOVERY + TravelInfo")

        # Override intent to ENTITY_FACT if query is address/location lookup (e.g., "ở đâu", "nằm ở đâu")
        elif self._is_phone_lookup_query(state.user_query):
            primary_intent = IntentType.ENTITY_FACT
            metadata["intent_override_reason"] = "phone_lookup_signal_detected"
            if phone_lookup_entity_hint:
                metadata["target_entity"] = phone_lookup_entity_hint
            detected_from_analyzer = ""
            loc = ""
            location_context = p._build_location_context(
                name="",
                source="global",
                reason="phone_lookup_no_location_filter",
                confidence=0.0,
            )
            has_explicit_location = False
            logger.info("   -> Intent override: detected phone lookup query, forcing ENTITY_FACT")
        elif opening_hours_entity_hint:
            primary_intent = IntentType.ENTITY_FACT
            metadata["intent_override_reason"] = "opening_hours_lookup_signal_detected"
            metadata["target_entity"] = opening_hours_entity_hint
            detected_from_analyzer = ""
            loc = ""
            location_context = p._build_location_context(
                name="",
                source="global",
                reason="opening_hours_lookup_no_location_filter",
                confidence=0.0,
            )
            has_explicit_location = False
            logger.info("   -> Intent override: detected opening-hours lookup query, forcing ENTITY_FACT")
        elif self._is_address_lookup_query(state.user_query) and not self._is_mixed_address_and_description_query(state.user_query) and len(entities or []) <= 1:
            primary_intent = IntentType.ENTITY_FACT
            metadata["intent_override_reason"] = "address_lookup_signal_detected"
            if address_lookup_entity_hint:
                metadata["target_entity"] = address_lookup_entity_hint
                detected_from_analyzer = ""
                loc = ""
                location_context = p._build_location_context(
                    name="",
                    source="global",
                    reason="address_lookup_entity_hint_no_location_filter",
                    confidence=0.0,
                )
                has_explicit_location = False
            logger.info("   -> Intent override: detected address lookup query, forcing ENTITY_FACT")
        elif p._intent_equals(primary_intent, IntentType.DISTANCE) and (
            str(metadata.get("question_type") or "") in {"Multi-Choice", "Multi-Select", "True-or-False", "Fill-in-Blank"}
            or analysis_subject_entity_hint
        ):
            non_distance = [
                intent for intent in (current_intents or [])
                if not p._intent_equals(str(intent), IntentType.DISTANCE)
            ]
            primary_intent = non_distance[0] if non_distance else IntentType.DISCOVERY
            metadata["intent_override_reason"] = "non_distance_answer_mode"

        from graph_rag.core.retrieval_policy import RetrievalPolicy
        policy = RetrievalPolicy.resolve_policy(
            primary_intent,
            current_intents,
            state.user_query,
        )
        retrieval_allowed_labels = policy.allowed_labels
        # Merge any label overrides (e.g., emergency queries adding TravelInfo)
        label_override = metadata.pop("retrieval_label_override", None)
        if label_override:
            retrieval_allowed_labels = sorted(set(retrieval_allowed_labels) | set(label_override))
        # If transport_hint is set (query asks about di chuyen), include TravelInfo
        if metadata.get("transport_hint") and "TravelInfo" not in retrieval_allowed_labels:
            retrieval_allowed_labels = sorted(set(retrieval_allowed_labels) | {"TravelInfo"})
            logger.info("   -> [TransportHint] Added TravelInfo to retrieval labels")
        metadata["retrieval_allowed_labels"] = retrieval_allowed_labels
        metadata["retrieval_policy"] = policy.to_dict()
        metadata["is_multi_intent_travel"] = p._is_multi_intent_travel_query(current_intents, state.user_query)
        metadata["intent"] = primary_intent
        # Preserve original intent before query_frame may override it to DISCOVERY_SEARCH.
        # Used by seed_retriever for intent-aware label prioritization.
        metadata["original_intent"] = primary_intent

        # Diagnostic logging for empty intent
        if not primary_intent:
            logger.warning("   -> [DIAGNOSTIC] Empty primary_intent! current_intents=%s, query=%s", current_intents, state.user_query[:80])
        logger.info("   -> [DIAGNOSTIC] primary_intent='%s', current_intents=%s", primary_intent, current_intents)

        if p._intent_equals(primary_intent, IntentType.DISCOVERY):
            if not metadata.get("proximity_anchor_required"):
                metadata.pop("target_entity", None)
                metadata["target_entity_policy"] = "discovery_no_single_target"

        trip_days = p.tour_route_optimizer.extract_trip_days(state.user_query)
        metadata["constraints"] = p.tour_route_optimizer.extract_route_constraints(
            primary_intent,
            state.user_query,
            metadata.get("constraints") if isinstance(metadata, dict) else {},
            trip_days=trip_days,
        )
        admin_region_match: Dict[str, Any] = {}
        admin_region_service = getattr(p, "admin_region_mapping_service", None)
        if admin_region_service is not None:
            admin_region_match = admin_region_service.resolve(
                " ".join([state.user_query or "", detected_from_analyzer or "", loc or ""]),
                entities,
            )
            if admin_region_match:
                metadata["admin_region_match"] = admin_region_match
                metadata["matched_admin_alias"] = admin_region_match.get("matched_alias") or ""
                metadata["display_region"] = admin_region_match.get("display_region") or ""
                metadata["region_group"] = admin_region_match.get("region_group") or ""
                metadata["legacy_province"] = admin_region_match.get("legacy_province") or admin_region_match.get("old_province") or ""
                metadata["current_province"] = admin_region_match.get("current_province") or admin_region_match.get("new_province") or ""
                logger.info(
                    "   -> Admin region mapping matched: "
                    f"alias='{metadata['matched_admin_alias']}', "
                    f"region='{admin_region_match.get('region_focus')}', "
                    f"group='{metadata['region_group']}', "
                    f"display='{metadata['display_region']}'"
                )

        # Detect ward-level location from entity admin_level (set by LLM analyzer)
        # Used for post-filtering retrieval results via graph LOCATED_IN
        for ent in entities or []:
            if isinstance(ent, dict):
                al = str(ent.get("admin_level") or "").strip().lower()
                if al == "ward":
                    _ward_name = str(ent.get("name") or "").strip()
                    if _ward_name:
                        metadata["has_sub_province_location"] = True
                        metadata["sub_province_location_name"] = _ward_name
                        logger.info("   -> Ward-level location detected: '%s'", _ward_name)
                        break

        # Follow-up guard: check BEFORE _detect_region_focus to preserve inherited region
        resolved = metadata.get("resolved_query_frame")
        _resolver_set_region = bool(
            resolved and getattr(resolved, "_region_inherited", False)
        )

        # For follow-ups with inherited region, skip _detect_region_focus entirely
        # to prevent current_location (GPS) from overriding the inherited region
        if _resolver_set_region:
            region_focus = metadata.get("region_focus") or "all"
            logger.info("   -> [FollowUp] Preserving inherited region_focus: '%s'", region_focus)
        else:
            region_focus = p._detect_region_focus(state.user_query, loc, entities)

        state.primary_intent = primary_intent
        admin_region_focus = (admin_region_match or {}).get("region_focus")

        if admin_region_focus in {"binh_dinh_old", "gia_lai_core", "gia_lai_new"} and not _resolver_set_region:
            # For single-location EVENT queries, don't broaden to ALL —
            # the event is at a specific location, results should be filtered.
            if state.primary_intent == IntentType.EVENT:
                metadata["region_focus_source"] = "admin_region_mapping"
                # Keep region_focus as detected; will be refined below
            else:
                from graph_rag.modules.pipeline_support.admin_region_mapping_service import AdminRegionMappingService
                region_focus = AdminRegionMappingService.resolve_region_focus(admin_region_focus)
                metadata["region_focus_source"] = "admin_region_mapping"
                logger.info("   -> Admin region mapping: '%s' → region_focus='%s'", admin_region_focus, region_focus)

        # Multi-region detection: "từ A đến B", "A - B", "A rồi B"
        # If query mentions multiple regions, keep ALL — don't narrow to one region.
        q_norm_region = normalize_text(state.user_query, strip_punct=True)
        q_padded = f" {q_norm_region} "
        region_name_count = sum(1 for m in MULTI_REGION_NAMES if m in q_norm_region)

        # REQUIREMENT: At least 2 distinct region names to trigger multi-region.
        # Single region + connector is NOT enough (e.g., "Đến Bình Định" = destination).
        # "tôi" (I) normalizes to "toi" which falsely matches connector "tới".
        is_multi_region = region_name_count >= 2

        if is_multi_region:
            region_focus = RegionFocus.ALL
            metadata["region_focus_source"] = "multi_region_detection"
        elif loc and region_focus == "all" and admin_region_focus != "gia_lai_new":
            region_focus = p._location_to_region_focus(loc)
        metadata["region_focus"] = region_focus
        metadata["geo_anchor_location"] = loc

        coref_conf = float(metadata.get("coreference_confidence", 0.5))
        needs_clarification = bool(metadata.get("needs_clarification", False))
        has_history_coref = any(
            isinstance(e, dict) and str(e.get("source") or "").strip().lower() == "history"
            for e in metadata.get("resolved_entities", [])
        )
        if needs_clarification or (has_history_coref and coref_conf < 0.35):
            search_query = state.user_query
            logger.info(
                "   -> Coreference low-confidence fallback: "
                f"needs_clarification={needs_clarification}, confidence={coref_conf:.2f}."
            )

        if disable_coreference:
            search_query = metadata.get("rewritten_query") or state.user_query

        if not disable_coreference:
            entities = p._resolve_generic_entities_with_history(
                entities,
                state.history,
                primary_intent,
                entity_memory=p.conversation_state.get("entity_memory") or [],
            )

        if p._intent_equals(primary_intent, IntentType.DISTANCE):
            src, dst = DistanceQueryParser.parse(state.user_query)
            repaired_entities = []
            if src:
                repaired_entities.append({
                    "name": src,
                    "type": "Location",
                    "role": "origin",
                    "source": "distance_parser",
                    "confidence": 1.0,
                    "trusted": True
                })
            if dst:
                dst_type = "Location"
                if len(entities or []) >= 2 and isinstance(entities[1], dict):
                    hinted = str(entities[1].get("type") or "").strip()
                    if hinted:
                        dst_type = hinted
                elif len(entities or []) == 1 and isinstance(entities[0], dict):
                    hinted = str(entities[0].get("type") or "").strip()
                    if hinted:
                        dst_type = hinted
                repaired_entities.append({
                    "name": dst,
                    "type": dst_type,
                    "role": "destination",
                    "source": "distance_parser",
                    "confidence": 1.0,
                    "trusted": True
                })
            if repaired_entities:
                logger.info(
                    "   -> Distance query parsing applied (preserves raw case): "
                    f"{entities} -> {repaired_entities}"
                )
                entities = repaired_entities

        # ── Example Detection (MUST run BEFORE V3 router) ──
        # Mark example entities so V3 router doesn't use them as anchors
        example_text = self._extract_example_text(state.user_query or "")
        example_entity_norms = set()
        if example_text:
            example_norm = normalize_text(example_text, strip_punct=True)
            for entity in (entities or []):
                e_name = str(entity.get("name") or "").strip()
                if e_name and normalize_text(e_name, strip_punct=True) in example_norm:
                    entity["example_origin"] = True
                    entity["confidence"] = min(float(entity.get("confidence") or 0.5), 0.3)
                    example_entity_norms.add(normalize_text(e_name, strip_punct=True))
            if example_entity_norms:
                logger.info("   -> Example entities detected (BEFORE V3): %s", list(example_entity_norms))

        # ── Confidence Routing & Proactive Merge ──
        router_anchors = []
        if GRAPH_RAG_V3_ENABLED:
            # Strip example clauses before V3 anchor extraction
            # Pattern 1: "(Ví dụ: X, Y)" or "Ví dụ: X, Y"
            _EXAMPLE_RE = re.compile(
                r"(?i)\s*(?:\(|\s)(?:ví\s+dụ|vi\s+du|VD|vd|chẳng\s+hạn|chang\s+han|ví\s+dụ\s+như|vi\s+du\s+như)\s*[:：]?\s*[^)]*?(?:\)|$)",
                re.DOTALL,
            )
            # Pattern 2: "như X hay Y thì nên..." or "như X, Y, Z"
            _NHU_EXAMPLE_RE = re.compile(
                r"(?i),?\s*như\s+[A-ZÀ-Ỹ0-9][^\n,.;:!?]*(?:\s+(?:hay|hoặc|và|or|and)\s+[A-ZÀ-Ỹ0-9][^\n,.;:!?]*)*(?:\s*(?:thì|[,.!?])|\s*$)",
                re.DOTALL,
            )
            query_for_v3 = _EXAMPLE_RE.sub("", state.user_query or "").strip()
            query_for_v3 = _NHU_EXAMPLE_RE.sub("", query_for_v3).strip()
            query_for_v3 = re.sub(r"\(\s*\)", "", query_for_v3).strip() or state.user_query

            # Filter example entities from metadata before V3 router
            # so V3 doesn't extract them as anchors
            v3_metadata = {**metadata}
            if example_entity_norms:
                non_example_entities = [
                    e for e in (entities or [])
                    if not isinstance(e, dict) or normalize_text(str(e.get("name") or ""), strip_punct=True) not in example_entity_norms
                ]
                v3_metadata["entities"] = non_example_entities

            v3_intent_data = p.intent_router.parse(query_for_v3, metadata=v3_metadata)
            metadata["v3_enabled"] = True
            metadata["v3_intent_data"] = v3_intent_data
            router_anchors = v3_intent_data.get("anchors") or []

            # Filter out garbage anchors (too long, not entity-like, or example)
            _QUESTION_WORDS = {"khong", "không", "co", "có", "nah", "nha", "chua", "chưa", "nao", "gì", "gi", "sao", "the", "thế", "nhi", "nhỉ", "vay", "vậy"}
            _QUESTION_PHRASES = ["nao sau day", "nao sau đây", "sau day", "sau đây", "trong hai", "trong ba", "loai nao", "loại nào"]
            _cleaned_anchors = []

            # Check if this is a comparison query — skip word limit for comparison subjects
            q_norm_for_check = normalize_text(state.user_query or "", strip_punct=True)
            is_comparison_query = "so sanh" in q_norm_for_check or "khac biet" in q_norm_for_check

            for a in router_anchors:
                a_str = str(a or "").strip()
                a_norm = normalize_text(a_str, strip_punct=True)
                # Skip if too long (>50 chars = not an entity name)
                if len(a_str) > 50:
                    continue
                # Skip if too many words (>6 = likely a sentence, not entity)
                # But keep comparison subjects even if long (e.g., "Nhà hàng Lá Xanh (Công viên Đồng Xanh)")
                if len(a_str.split()) > 6 and not is_comparison_query:
                    continue
                # Skip if it's an example entity
                if example_entity_norms and a_norm in example_entity_norms:
                    continue
                # Skip if it looks like a sentence fragment
                if any(phrase in a_norm for phrase in ["dua tren", "theo du lieu", "xac dinh", "tim thay"]):
                    continue
                # Skip Vietnamese question words
                if a_norm in _QUESTION_WORDS:
                    continue
                # Skip if anchor ends with question word (e.g., "quan không")
                tokens = a_norm.split()
                if tokens and tokens[-1] in _QUESTION_WORDS:
                    continue
                # Skip question phrase patterns (e.g., "Quán ăn nào sau đây")
                if any(phrase in a_norm for phrase in _QUESTION_PHRASES):
                    continue
                # Skip if most tokens are question words (e.g., "nào sau đây")
                q_word_count = sum(1 for t in tokens if t in _QUESTION_WORDS)
                if len(tokens) >= 2 and q_word_count / len(tokens) >= 0.5:
                    continue
                _cleaned_anchors.append(a)
            if len(_cleaned_anchors) != len(router_anchors):
                logger.info("   -> V3 anchors filtered: %s -> %s", router_anchors, _cleaned_anchors)
                if isinstance(v3_intent_data, dict):
                    v3_intent_data["anchors"] = list(_cleaned_anchors)
                    metadata["v3_intent_data"] = v3_intent_data
            router_anchors = _cleaned_anchors
            if metadata.get("proximity_anchor"):
                _pa = metadata.get("proximity_anchor") or {}
                prox_text = _pa.get("text") if isinstance(_pa, dict) else str(_pa or "")
                prox_norm = normalize_text(prox_text, strip_punct=True)
                kept_norms = {
                    normalize_text(str(anchor or ""), strip_punct=True)
                    for anchor in router_anchors
                }
                if prox_norm and prox_norm not in kept_norms:
                    metadata.pop("proximity_anchor", None)
                    metadata["proximity_anchor_required"] = False
                    logger.info("   -> Proximity anchor cleared after V3 anchor filtering.")
            pruned_anchors = self._prune_generic_recovered_anchors(router_anchors, state.user_query)
            if pruned_anchors != router_anchors:
                logger.info("   -> V3 generic anchors pruned: %s -> %s", router_anchors, pruned_anchors)
                router_anchors = pruned_anchors
                if isinstance(v3_intent_data, dict):
                    v3_intent_data["anchors"] = list(pruned_anchors)
                    metadata["v3_intent_data"] = v3_intent_data

            logger.info(
                "   -> V3 intent router: "
                f"mode='{v3_intent_data.get('intent_mode')}', "
                f"anchors={self._preview_list(router_anchors, 5)}, "
                f"conditions={(v3_intent_data.get('constraints') or {}).get('required_conditions') or []}"
            )
            if v3_intent_data.get("label_hints"):
                logger.info("   -> V3 label hints: %s", v3_intent_data['label_hints'])

        if router_anchors or entities:
            # Filter example entities from merge input
            merge_entities = [
                e for e in (entities or [])
                if not isinstance(e, dict) or not e.get("example_origin")
            ]
            merged_entities = p.query_analyzer._merge_and_route_sources(state.user_query, merge_entities, router_anchors)
            if merged_entities != entities:
                logger.info(
                    "   -> Proactive Merge & Dedup Layer applied: "
                    f"{len(merge_entities)} LLM + {len(router_anchors)} V3 -> {len(merged_entities)} merged"
                )
                entities = merged_entities
                metadata["entities"] = entities

        canonicalized_entities = self._canonicalize_entities_for_grounding(entities)
        if canonicalized_entities != entities:
            logger.info(
                "   -> Entity canonicalization applied: "
                f"{entities} -> {canonicalized_entities}"
            )
        entities = canonicalized_entities

        corrected_entities = self._correct_entity_types_from_query_context(
            entities,
            state.user_query,
            metadata,
        )
        if corrected_entities != entities:
            logger.info(
                "   -> Entity type correction from query context applied: "
                f"{entities} -> {corrected_entities}"
            )
            entities = corrected_entities

        pruned_entities = self._prune_generic_entities_with_specific_siblings(entities, metadata)
        if pruned_entities != entities:
            logger.info(
                "   -> Generic entities pruned after merge: "
                f"{entities} -> {pruned_entities}"
            )
            entities = pruned_entities

        # --- Architectural fix: classify entities at source ---
        # Separate category hints (e.g., "nhà nghỉ", "di tích lịch sử") from
        # specific entities (e.g., "Nhà nghỉ 22", "Bảo tàng Quang Trung").
        # Category hints are NOT grounded — they expand retrieval labels instead.
        classified_entities = []
        category_label_hints: List[str] = []
        for entity in (entities or []):
            classified = self._classify_entity(entity)
            if classified.get("is_category_hint"):
                hint_label = str(classified.get("label_hint") or "").strip()
                if hint_label and hint_label not in category_label_hints:
                    category_label_hints.append(hint_label)
                logger.info("   -> Category hint detected: '%s' → %s", classified.get('name'), hint_label)
            classified_entities.append(classified)
        entities = classified_entities
        if category_label_hints:
            existing_hints = list(metadata.get("label_hints") or [])
            merged_hints = list(dict.fromkeys(existing_hints + category_label_hints))
            metadata["label_hints"] = merged_hints
            logger.info("   -> Label hints from entities: %s", merged_hints)
            # Expand retrieval_allowed_labels if needed
            current_labels = list(metadata.get("retrieval_allowed_labels") or [])
            current_query_norm = normalize_text(state.user_query, strip_punct=True)
            expanded = self._expand_labels_from_category_hints(current_labels, metadata, current_query_norm)
            if expanded != current_labels:
                metadata["retrieval_allowed_labels"] = expanded
                logger.info("   -> Retrieval labels expanded: %s", expanded)

        metadata["entities"] = entities
        if not metadata.get("target_entity"):
            if metadata.get("proximity_anchor_required") and metadata.get("proximity_anchor"):
                _pa = metadata["proximity_anchor"]
                metadata["target_entity"] = _pa.get("text") if isinstance(_pa, dict) else _pa
            for entity in entities or []:
                if metadata.get("target_entity"):
                    break
                if not self._is_groundable_entity(entity):
                    continue
                # Skip category hints — they are labels, not groundable entities
                if entity.get("is_category_hint"):
                    continue
                e_type = str(entity.get("type") or "").strip().lower()
                if e_type in {"province", "city", "district", "ward", "commune", "location"}:
                    continue
                e_name = str(entity.get("name") or "").strip()
                if e_name:
                    metadata["target_entity"] = e_name
                    break

        logger.info("   -> Metadata Keys: %s", sorted(list(metadata.keys())))
        logger.info("   -> Intents: %s | Loc: %s", current_intents, loc)
        logger.info("   -> Extracted Entities: %s", entities)
        logger.info("   -> Rewritten: '%s'", search_query)
        logger.info(
            "   -> Intent Routing: "
            f"primary='{primary_intent}', region_focus='{region_focus}', "
            f"constraints={metadata.get('constraints', {})}, "
            f"retrieval_labels={metadata.get('retrieval_allowed_labels', [])}, "
            f"multi_intent_travel={metadata.get('is_multi_intent_travel', False)}"
        )
        logger.info("   -> Coreference-resolved entities (%s): %s", len(entities), entities)
        logger.info("   -> Step 1 completed in %s", self._elapsed(step_1_start))

        return {
            "entities": entities,
            "metadata": metadata,
            "primary_intent": primary_intent,
            "location_context": location_context,
            "loc": loc,
            "region_focus": region_focus,
            "search_query": search_query,
            "admin_region_match": admin_region_match,
            "has_explicit_location": has_explicit_location,
        }

    # ------------------------------------------------------------------
    # Sub-method 4: Answer mode routing, V3 override, QueryFrame V2
    # ------------------------------------------------------------------
    def _apply_query_frame(
        self,
        state: PipelineRunState,
        metadata: dict,
        entities: list,
        primary_intent: str,
    ) -> Dict[str, Any]:
        """Answer mode inference, V3 intent override, QueryFrame V2 stage."""
        p = self.pipeline

        # --- Answer Mode Router (P0) ---
        # Determine HOW to answer, separate from intent (which determines WHAT to retrieve).
        question_type = metadata.get("question_type") or ""
        if not question_type:
            q_norm = normalize_text(state.user_query, strip_punct=True)
            if any(t in q_norm for t in ["dung hay sai", "true or false", "dung hay khong dung"]):
                question_type = "True-or-False"
            elif "___" in state.user_query:
                question_type = "Fill-in-Blank"
        answer_mode = infer_answer_mode(
            question=state.user_query,
            question_type=question_type,
            intent=primary_intent,
            metadata=metadata,
        )
        metadata["answer_mode"] = answer_mode
        metadata["question_type"] = question_type
        if AnswerMode.is_closed_form(answer_mode) and primary_intent in {IntentType.TOUR_PLAN, IntentType.DISTANCE}:
            primary_intent = IntentType.DISCOVERY
            metadata["intent"] = primary_intent
            metadata["intent_override_reason"] = "closed_form_answer_mode"
        logger.info("   -> Answer Mode Router: mode='%s', question_type='%s'", answer_mode, question_type)

        # V3 Intent Override: If V3 router detected a specific mode (e.g., tour_plan),
        # override primary_intent BEFORE QueryFrame can override it to DISCOVERY.
        # This fixes the mismatch where analyzer returns DISCOVERY_SEARCH but V3 correctly
        # detects TOUR_PLAN, causing Step 4 to run the wrong retrieval strategy.
        v3_mode = (metadata.get("v3_intent_data") or {}).get("intent_mode") or ""
        _V3_MODE_TO_INTENT = {
            "tour_plan": IntentType.TOUR_PLAN,
            "dish_to_restaurant": IntentType.FOOD,
            "lodging_near_anchor": IntentType.ACCOMMODATION,
        }
        v3_mapped_intent = _V3_MODE_TO_INTENT.get(v3_mode)
        if v3_mapped_intent and v3_mapped_intent != primary_intent:
            # Unified Contract: V3 may only UPGRADE DISCOVERY → specific intent.
            # If analyzer already returned a specific intent (TOURISM, FOOD, etc.),
            # V3 must NOT override it.
            _analyzer_intent = metadata.get("_analyzer_primary_intent") or ""
            if _analyzer_intent in ("", IntentType.DISCOVERY):
                logger.info(
                    "   -> V3 intent upgrade: '%s' -> '%s' (source: v3_router mode='%s')",
                    primary_intent, v3_mapped_intent, v3_mode,
                )
                primary_intent = v3_mapped_intent
                metadata["intent"] = primary_intent
                metadata["intent_override_reason"] = f"v3_router_{v3_mode}"
            else:
                logger.info(
                    "   -> V3 intent override BLOCKED: analyzer='%s' is specific, "
                    "V3 wanted '%s' (source: v3_router mode='%s')",
                    _analyzer_intent, v3_mapped_intent, v3_mode,
                )

        if ENABLE_QUERY_FRAME_V2:
            # Skip regex query frame stage if it's a follow-up query and the LLM extracted no new entities.
            # This avoids false positive regex entity extraction on functional/question phrases (e.g. "còn món nào khác nữa không").
            skip_query_frame = False
            if metadata.get("is_follow_up"):
                raw_entities = [e for e in (entities or []) if not metadata.get("entity_inherited")]
                if not raw_entities:
                    skip_query_frame = True
                    logger.info("   -> QueryFrameV2 skipped: follow-up query has no newly extracted entities.")

            if skip_query_frame:
                metadata["query_frame_applied"] = False
                metadata["query_frame_valid"] = False
                metadata["query_frame_skip_reason"] = "follow_up_without_new_entities"
            else:
                metadata, framed_entities, frame_debug = self.query_frame_stage.build_and_apply(
                    query=state.user_query,
                    metadata=metadata,
                    entities=entities,
                    primary_intent=primary_intent,
                    role_aware_grounding=ENABLE_ROLE_AWARE_GROUNDING,
                )
                if framed_entities != entities:
                    framed_entities = self._canonicalize_entities_for_grounding(framed_entities)
                    framed_entities = self._correct_entity_types_from_query_context(
                        framed_entities,
                        state.user_query,
                        metadata,
                    )
                    framed_entities = self._prune_generic_entities_with_specific_siblings(
                        framed_entities,
                        metadata,
                    )
                    metadata["entities"] = framed_entities
                    entities = framed_entities

                # Build QueryFrame contract — populates metadata for QueryPlanBuilder
                # The typed contract itself is no longer stored on state (QueryPlan is the single source)
                self.query_frame_stage.build_query_frame_contract(
                    metadata=metadata,
                    entities=entities,
                    primary_intent=primary_intent,
                )

                # Re-check answer_mode now that QueryFrame is available.
                # infer_answer_mode() runs before QueryFrame, so it can't see
                # query_frame.operator. Fix up if frame says global_discovery.
                if answer_mode == AnswerMode.FACT_ANSWER:
                    qf_op = str((metadata.get("query_frame") or {}).get("query_operator") or "")
                    if qf_op == "global_discovery":
                        answer_mode = AnswerMode.DISCOVERY_LIST
                        metadata["answer_mode"] = answer_mode
                        logger.info("   -> Answer Mode Router: re-checked after QueryFrame → mode='%s' (was fact_answer)", answer_mode)

                type_comparison_entities = [
                    str(entity.get("name") or "").strip()
                    for entity in entities
                    if isinstance(entity, dict)
                    and self._is_groundable_entity(entity)
                    and str(entity.get("name") or "").strip()
                ]
                q_norm_for_frame = normalize_text(state.user_query, strip_punct=True)
                if (
                    len(dict.fromkeys(type_comparison_entities)) >= 2
                    and any(marker in q_norm_for_frame for marker in ["deu thuoc loai", "thuoc loai hinh", "loai hinh du lich", "deu thuoc"])
                ):
                    metadata["retrieval_plan_mode"] = "comparison"
                    metadata["query_frame_anchor_names"] = list(dict.fromkeys(type_comparison_entities))
                    metadata["query_frame_multi_anchor_mode"] = True
                    metadata["query_frame_traversal_relations"] = ["BELONGS_TO"]
                    metadata["query_frame_target_policy"] = "forced_type_comparison_multi_anchor"
                    metadata["query_frame_applied"] = True
                    metadata["target_entity"] = ""
                if metadata.get("query_frame_global_discovery"):
                    # Unified Contract: QueryFrame may only set DISCOVERY when
                    # the analyzer ALSO returned DISCOVERY (or empty).  If the
                    # analyzer returned a specific intent (TOURISM, FOOD, etc.),
                    # QueryFrame must NOT downgrade it.
                    _analyzer_intent = metadata.get("_analyzer_primary_intent") or ""
                    if _analyzer_intent in ("", IntentType.DISCOVERY):
                        primary_intent = IntentType.DISCOVERY
                        metadata["intent"] = primary_intent
                        metadata["intent_override_reason"] = "query_frame_global_discovery"
                    else:
                        logger.info(
                            "   -> QueryFrame global_discovery BLOCKED: analyzer='%s' is specific",
                            _analyzer_intent,
                        )
                elif metadata.get("retrieval_plan_mode") in {"dish_to_restaurant", "lodging_near_anchor"}:
                    framed_intent = IntentType.normalize(metadata.get("intent"), default=primary_intent)
                    if framed_intent != primary_intent:
                        primary_intent = framed_intent
                        metadata["intent"] = primary_intent
                        metadata["intent_override_reason"] = f"query_frame_{metadata.get('retrieval_plan_mode')}"
                if QUERY_FRAME_DEBUG_LOG:
                    logger.info(
                        "   -> QueryFrameV2: "
                        f"operator='{frame_debug.get('query_operator')}', "
                        f"valid={frame_debug.get('valid')}, "
                        f"applied={metadata.get('query_frame_applied', False)}, "
                        f"mode='{(frame_debug.get('retrieval_plan') or {}).get('mode')}', "
                        f"target='{metadata.get('target_entity', '')}'"
                    )
                if (
                    metadata.get("retrieval_plan_mode") == "comparison"
                    and len(metadata.get("query_frame_anchor_names") or []) >= 2
                ):
                    metadata["region_lock_mode"] = "disabled_multi_anchor_comparison"
                    metadata["region_filter_disabled_reason"] = "comparison_multi_anchor_may_cross_region"
                    logger.info(
                        "   -> Region lock relaxed for comparison anchors: "
                        f"{metadata.get('query_frame_anchor_names')}"
                    )

        return {
            "entities": entities,
            "metadata": metadata,
            "primary_intent": primary_intent,
            "answer_mode": answer_mode,
        }

    # ------------------------------------------------------------------
    # Sub-method 5: QueryPlan building, refinement, plan diff, metadata freeze
    # ------------------------------------------------------------------
    def _build_query_plan(
        self,
        state: PipelineRunState,
        metadata: dict,
        entities: list,
        primary_intent: str,
        location_context: dict,
        loc: str,
        region_focus: str,
        search_query: str,
        has_explicit_location: bool,
        current_intents: list,
    ) -> None:
        """Label expansion, state finalization, follow-up context injection,
        contract patch, QueryPlan building, retrieval policy refinement, metadata freeze."""
        p = self.pipeline

        # Expand retrieval labels when query signals multiple entity types
        v3_data = metadata.get("v3_intent_data") or {}
        if metadata.get("disable_discovery_expansion"):
            expanded_labels = metadata.get("retrieval_allowed_labels") or []
        else:
            expanded_labels = self._expand_labels_from_category_hints(
                current_labels=metadata.get("retrieval_allowed_labels") or [],
                v3_intent_data=v3_data,
                query_norm=normalize_text(state.user_query, strip_punct=True),
            )
        # Protect forbidden_labels from being re-introduced by category hint expansion
        forbidden = set(metadata.get("forbidden_labels") or [])
        if forbidden:
            expanded_labels = [l for l in expanded_labels if l not in forbidden]
        if expanded_labels != metadata.get("retrieval_allowed_labels"):
            logger.info(
                "   -> Retrieval labels expanded by category hints: "
                f"{metadata.get('retrieval_allowed_labels')} -> {expanded_labels}"
            )
            metadata["retrieval_allowed_labels"] = expanded_labels

        state.metadata = metadata
        state.search_query = search_query

        # Follow-up search_query enhancement: inject inherited context for better vector search
        resolved = metadata.get("resolved_query_frame")
        if resolved and resolved.is_follow_up:
            search_norm = normalize_text(search_query, strip_punct=True)

            # Build context terms from conversation state
            context_parts = []

            # DON'T add broad location (Gia Lai, Quy Nhơn) to search query —
            # location filtering is handled by Step 4's location_filter parameter.
            # Adding broad location pollutes vector search and triggers entity-first guard.

            # Add target_class with domain-specific keywords
            if resolved.target_class:
                _TARGET_CLASS_KEYWORDS = {
                    "Specialty": ["đặc sản", "món ngon"],
                    "Dish": ["món ăn", "đặc sản"],
                    "Restaurant": ["nhà hàng", "quán ăn"],
                    "Accommodation": ["khách sạn", "lưu trú"],
                    "Event": ["lễ hội", "sự kiện"],
                    "TouristAttraction": ["địa điểm", "tham quan"],
                }
                keywords = _TARGET_CLASS_KEYWORDS.get(resolved.target_class, [])
                for kw in keywords:
                    kw_norm = normalize_text(kw, strip_punct=True)
                    if kw_norm not in search_norm:
                        context_parts.append(kw)

            # DON'T extract terms from previous answer — too noisy.
            # Domain keywords from _TARGET_CLASS_KEYWORDS are sufficient.

            if context_parts:
                # Limit to 2 context terms to avoid query pollution
                state.search_query = f"{search_query} {' '.join(context_parts[:2])}"
                logger.info("   -> [FollowUp] Enhanced search_query: '%s'", state.search_query)

        # Example entities: DO NOT append to search_query
        # They are examples (Ví dụ: X, Y), not the actual search target.
        # Appending them would make hybrid search return example-matching nodes
        # instead of topic-relevant nodes.
        example_ents = [e for e in (entities or []) if e.get("example_origin")]
        if example_ents:
            example_names = [str(e.get("name") or "") for e in example_ents if e.get("name")]
            metadata["example_entity_names"] = example_names
            logger.info("   -> Example entities noted (NOT appended to search): %s", example_names)

        state.primary_intent = primary_intent
        state.entities = entities
        state.location_context = location_context
        state.location = loc
        state.region_focus = region_focus
        state.has_explicit_location = has_explicit_location

        # Store is_follow_up in conversation_state for context expiry in _update_conversation_state
        p.location_grounding_service.conversation_state["current_is_follow_up"] = bool(metadata.get("is_follow_up", False))

        # Inject resolved follow-up context into metadata for QueryPlan
        resolved = metadata.get("resolved_query_frame")
        if resolved and resolved.is_follow_up:
            if resolved.target_class and not metadata.get("target_class"):
                metadata["target_class"] = resolved.target_class
                logger.info("   -> [FollowUp] Injected target_class: '%s'", resolved.target_class)
            if resolved.semantic_category and not metadata.get("semantic_category"):
                metadata["semantic_category"] = resolved.semantic_category
                logger.info("   -> [FollowUp] Injected semantic_category: '%s'", resolved.semantic_category)

                # Propagate contract labels based on inherited target_class
                _TARGET_CLASS_LABELS = {
                    "Specialty": ["Dish", "Specialty", "Restaurant", "Location"],
                    "Dish": ["Dish", "Restaurant", "Specialty", "Location"],
                    "Restaurant": ["Restaurant", "Dish", "Location"],
                    "Accommodation": ["Accommodation", "Location"],
                    "Event": ["Event", "TravelInfo"],
                    "TouristAttraction": ["TouristAttraction", "Location"],
                }
                inherited_labels = _TARGET_CLASS_LABELS.get(resolved.target_class)
                if inherited_labels:
                    current_labels = set(metadata.get("retrieval_allowed_labels") or [])
                    merged = sorted(current_labels | set(inherited_labels))
                    metadata["retrieval_allowed_labels"] = merged
                    logger.info("   -> [FollowUp] Propagated labels for target_class='%s': %s", resolved.target_class, merged)

            if resolved.intent and not metadata.get("intent"):
                metadata["intent"] = resolved.intent

        # Phase 3+4: Detect contract patch (immutable) + optional validate
        # ClosedFormGuard: detect() will block open-ended contracts when answer_mode is closed-form
        from graph_rag.pipeline.orchestration.contract_validator import ContractValidator
        from graph_rag.pipeline.orchestration.contract_patch import ContractPatch
        q_norm_for_contract = normalize_text(state.user_query or "", strip_punct=True)
        contract_patch: ContractPatch = ContractValidator.detect(q_norm_for_contract, metadata)
        if contract_patch.has_overrides():
            logger.info("   -> [ContractPatch] Detected: %s", contract_patch.contract_name)
            if contract_patch.entity_corrections:
                corrected_entities = list(state.entities) if state.entities else []
                for idx, new_type, suffix in contract_patch.entity_corrections:
                    if 0 <= idx < len(corrected_entities):
                        ent = dict(corrected_entities[idx])
                        old_type = ent.get("type", "Unknown")
                        ent["type"] = new_type
                        ent["type_source"] = f"{ent.get('type_source', 'extracted')}{suffix}"
                        corrected_entities[idx] = ent
                        logger.info("   -> [ContractPatch] Corrected entity at index %s: type %s -> %s", idx, old_type, new_type)
                state.entities = corrected_entities
                metadata["entities"] = corrected_entities

        elif AnswerMode.is_closed_form(metadata.get("answer_mode", "")):
            logger.info("   -> [ContractPatch] ClosedFormGuard: open-ended contracts blocked", )

        # Build QueryPlan — frozen business intent (Phase 3: with ContractPatch)
        from graph_rag.pipeline.orchestration.query_plan_builder import QueryPlanBuilder
        from graph_rag.pipeline.orchestration.plan_diff_logger import PlanDiffLogger

        state.query_plan = QueryPlanBuilder().build(
            query=state.user_query,
            metadata=metadata,
            query_state=None,
            contract_patch=contract_patch,
        )

        # Unified Contract: sync frozen query_plan.intent back to metadata.
        # metadata["intent"] may have been set by contract patch before build,
        # but this sync ensures they are identical even if contract patch path changes.
        if state.query_plan.intent:
            metadata["intent"] = state.query_plan.intent

        # Store query_plan properties in metadata so agentic retriever can access semantic_category
        metadata["query_state"] = {
            "semantic_category": state.query_plan.semantic_category,
            "target_class": state.query_plan.target_class,
            "question_shape": state.query_plan.question_shape.value if state.query_plan.question_shape else None,
            "duration_days": state.query_plan.duration_days,
            "duration_nights": state.query_plan.duration_nights,
        }
        # Also store at top level for easy access by seed_retriever
        metadata["semantic_category"] = state.query_plan.semantic_category
        logger.info(
            "   -> [QueryPlan] Standardized Plan Built:\n"
            f"      question_shape: {state.query_plan.question_shape.value} (source: {state.query_plan.question_shape_source}, confidence: {state.query_plan.question_shape_confidence:.2f})\n"
            f"      target_class: {state.query_plan.target_class} (source: {state.query_plan.target_class_source}, confidence: {state.query_plan.target_class_confidence:.2f})\n"
            f"      semantic_category: {state.query_plan.semantic_category} (confidence: {state.query_plan.semantic_category_confidence:.2f})\n"
            f"      target_dish: {state.query_plan.target_dish} (source: {state.query_plan.target_dish_source}, confidence: {state.query_plan.target_dish_confidence:.2f})\n"
            f"      requested_attributes: {state.query_plan.requested_attributes}\n"
            f"      is_follow_up: {state.query_plan.is_follow_up}"
        )

        # Phase 2: Refine retrieval policy using shape-aware QueryPlan
        from graph_rag.core.retrieval_policy import RetrievalPolicy
        refined_policy = RetrievalPolicy.resolve_policy_from_query_plan(state.query_plan, intents=current_intents)
        metadata["retrieval_policy"] = refined_policy.to_dict()
        # Only update allowed_labels if the refined policy is more specific
        # (has a shorter, more focused label list) to avoid over-expanding
        old_allowed = set(metadata.get("retrieval_allowed_labels") or [])
        new_allowed = set(refined_policy.allowed_labels)
        hard_label_contract = bool(metadata.get("hard_label_contract"))
        if hard_label_contract:
            logger.info(
                "   -> [Phase2] RetrievalPolicy refinement skipped: "
                f"hard label contract keeps labels={metadata.get('retrieval_allowed_labels')}"
            )
        elif old_allowed != new_allowed:
            metadata["retrieval_allowed_labels"] = refined_policy.allowed_labels
            # Keep QueryPlan in sync (single source of truth)
            import dataclasses
            state.query_plan = dataclasses.replace(
                state.query_plan,
                target_labels=tuple(refined_policy.allowed_labels),
            )
            logger.info(
                "   -> [Phase2] RetrievalPolicy refined by QueryPlan:\n"
                f"      shape={state.query_plan.question_shape.value}, "
                f"primary={refined_policy.primary_labels}, "
                f"budget={refined_policy.context_budget}"
            )
        state.metadata = metadata

        PlanDiffLogger.log(state.query_plan, metadata)

        # Phase 4: Freeze metadata after Step 1 is stable
        if hasattr(state.metadata, "freeze"):
            state.metadata.freeze()

    # ------------------------------------------------------------------
    # Orchestrator: calls sub-methods in sequence
    # ------------------------------------------------------------------
    def _run_step_1_query_understanding(self, state: PipelineRunState, current_location: str) -> None:
        p = self.pipeline
        logger.info("STEP 1: ANALYZING QUERY...")
        step_1_start = time.time()

        # Sub-method 1: Location detection, region focus, admin mapping
        loc_ctx = self._resolve_location_context(state, current_location)
        metadata = loc_ctx["metadata"]
        location_context = loc_ctx["location_context"]
        loc = loc_ctx["loc"]
        search_query = loc_ctx["search_query"]
        current_intents = loc_ctx["current_intents"]
        entities = loc_ctx["entities"]

        # Sub-method 2: Entity extraction, hint injection, merge/dedup, type correction, pruning
        ent_ctx = self._extract_and_classify_entities(
            state, metadata, location_context, loc, search_query, current_intents, entities,
        )
        entities = ent_ctx["entities"]
        metadata = ent_ctx["metadata"]
        has_explicit_location = ent_ctx["has_explicit_location"]
        detected_from_analyzer = ent_ctx["detected_from_analyzer"]
        disable_coreference = ent_ctx["disable_coreference"]
        loc = ent_ctx["loc"]
        search_query = ent_ctx["search_query"]
        address_lookup_entity_hint = ent_ctx["address_lookup_entity_hint"]
        phone_lookup_entity_hint = ent_ctx["phone_lookup_entity_hint"]
        opening_hours_entity_hint = ent_ctx["opening_hours_entity_hint"]
        analysis_subject_entity_hint = ent_ctx["analysis_subject_entity_hint"]

        # Sub-method 3: Intent selection, override paths, retrieval policy resolution
        intent_ctx = self._resolve_intent_and_policy(
            state, metadata, entities, loc, location_context, search_query,
            current_intents, has_explicit_location, detected_from_analyzer,
            disable_coreference, address_lookup_entity_hint,
            phone_lookup_entity_hint, opening_hours_entity_hint,
            analysis_subject_entity_hint, step_1_start,
        )
        entities = intent_ctx["entities"]
        metadata = intent_ctx["metadata"]
        primary_intent = intent_ctx["primary_intent"]
        location_context = intent_ctx["location_context"]
        loc = intent_ctx["loc"]
        region_focus = intent_ctx["region_focus"]
        search_query = intent_ctx["search_query"]
        has_explicit_location = intent_ctx["has_explicit_location"]

        # Sub-method 4: Answer mode routing, V3 override, QueryFrame V2
        frame_ctx = self._apply_query_frame(state, metadata, entities, primary_intent)
        entities = frame_ctx["entities"]
        metadata = frame_ctx["metadata"]
        primary_intent = frame_ctx["primary_intent"]

        # Sub-method 5: QueryPlan building, refinement, plan diff, metadata freeze
        self._build_query_plan(
            state, metadata, entities, primary_intent,
            location_context, loc, region_focus, search_query, has_explicit_location,
            current_intents,
        )
