from __future__ import annotations
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
"""Step 2: Entity grounding against the knowledge graph."""
import logging
from graph_rag.utils.node_utils import get_node_labels

logger = logging.getLogger(__name__)


import re
import time



from graph_rag.core.intents import IntentType, RegionFocus

from graph_rag.utils.text import normalize_text


from ..dto import PipelineRunState


class Step2GroundingMixin:
    """Mixin providing Step 2 entity grounding."""

    def _disambiguate_grounded_nodes(self, grounded_nodes, grounding_entities, state):
        """Post-extraction disambiguation: resolve ambiguous entity names.

        Applies a 3-tier cascade:
        1. Type-based filtering: filter by entity's expected type from Step 1
        2. Intent-based filtering: filter by query intent's allowed labels
        3. Score-based fallback: keep highest-scored node

        Args:
            grounded_nodes: List of NodeItem from ground_entities()
            grounding_entities: List of entity dicts (with 'name', 'type' keys)
            state: PipelineRunState with metadata.entities from Step 1

        Returns:
            Disambiguated list of NodeItem
        """
        from graph_rag.modules.retrieval.seed_retriever import SeedRetriever

        if not grounded_nodes or not grounding_entities:
            return grounded_nodes

        # Group grounded nodes by entity name (case-insensitive)
        entity_names = {}
        for ent in grounding_entities:
            name = str(ent.get("name") or "").strip()
            if name:
                key = name.lower()
                if key not in entity_names:
                    entity_names[key] = ent

        # If only one entity or all nodes have the same type, no disambiguation needed
        if len(entity_names) <= 1:
            return grounded_nodes

        # Build name->expected_type map from Step 1 extraction metadata
        step1_entities = (state.metadata or {}).get("entities") or []
        name_to_type = {}
        for ent in step1_entities:
            if isinstance(ent, dict):
                ent_name = str(ent.get("name") or "").strip().lower()
                ent_type = str(ent.get("type") or "").strip()
                if ent_name and ent_type:
                    name_to_type[ent_name] = ent_type

        # Get intent labels for intent-based filtering
        plan = state.query_plan
        intent_labels = SeedRetriever.get_intent_labels(plan.intent) if plan else []

        # Generic types that should use intent-based filtering instead
        _GENERIC_TYPES = {"place", "touristattraction", "location"}

        disambiguated = []
        for node in grounded_nodes:
            node_name = str(getattr(node, "content", "") or "").strip().lower()
            if not node_name:
                disambiguated.append(node)
                continue

            # Find the entity dict that matches this node
            matched_entity = None
            for ent_key, ent_dict in entity_names.items():
                if ent_key in node_name or node_name in ent_key:
                    matched_entity = ent_dict
                    break

            if not matched_entity:
                disambiguated.append(node)
                continue

            entity_name = str(matched_entity.get("name") or "").strip()
            expected_type = name_to_type.get(entity_name.lower(), "")
            node_labels = [str(l) for l in (node.metadata.get("labels") or [])]

            # Check if there are multiple candidates for this entity name
            candidates = [
                n for n in grounded_nodes
                if entity_name.lower() in str(getattr(n, "content", "") or "").lower()
                or str(getattr(n, "content", "") or "").lower() in entity_name.lower()
            ]
            if len(candidates) <= 1:
                disambiguated.append(node)
                continue

            # Tier 1: Type-based filtering (skip for generic types)
            if expected_type and expected_type.lower() not in _GENERIC_TYPES:
                type_filtered = SeedRetriever.filter_nodes_by_type(candidates, [expected_type])
                if len(type_filtered) == 1:
                    if type_filtered[0] not in disambiguated:
                        logger.info(
                            "   -> [Disambig] '%s': type_match '%s' selected '%s' from %d candidates",
                            entity_name, expected_type, type_filtered[0].content, len(candidates)
                        )
                        disambiguated.append(type_filtered[0])
                    continue
                elif len(type_filtered) > 1:
                    candidates = type_filtered

            # Tier 2: Intent-based filtering
            if intent_labels:
                intent_filtered = SeedRetriever.filter_nodes_by_type(candidates, intent_labels)
                if len(intent_filtered) == 1:
                    if intent_filtered[0] not in disambiguated:
                        logger.info(
                            "   -> [Disambig] '%s': intent_match '%s' selected '%s' from %d candidates",
                            entity_name, plan.intent, intent_filtered[0].content, len(candidates)
                        )
                        disambiguated.append(intent_filtered[0])
                    continue
                elif len(intent_filtered) > 1:
                    candidates = intent_filtered

            # Tier 3: Score-based fallback
            best = max(candidates, key=lambda n: getattr(n, "score", 0.0))
            if best not in disambiguated:
                logger.info(
                    "   -> [Disambig] '%s': score_fallback selected '%s' (score=%.3f) from %d candidates",
                    entity_name, best.content, best.score, len(candidates)
                )
                disambiguated.append(best)

        # Preserve order of nodes that weren't part of any disambiguation group
        result = []
        for node in grounded_nodes:
            if node in disambiguated:
                result.append(node)
        # Add any disambiguated nodes not in original order
        for node in disambiguated:
            if node not in result:
                result.append(node)

        if len(result) != len(grounded_nodes):
            logger.info(
                "   -> [Disambig] Disambiguation changed node count: %d -> %d",
                len(grounded_nodes), len(result)
            )

        return result if result else grounded_nodes

    def _run_step_2_grounding(self, state: PipelineRunState) -> None:
        p = self.pipeline
        logger.info("\n [STEP 2] GRAPH GROUNDING...")
        step_2_start = time.time()
        grounded_nodes = []
        metadata = state.metadata or {}
        logger.info("   -> Grounding Input: entity_count=%s", len(state.entities))

        # Read from QueryPlan — single source of truth (Milestone 2: no metadata fallback)
        plan = state.query_plan
        assert plan is not None, "QueryPlan must be initialized in Step 1"

        # Build example anchor norms by detecting entities in example clauses
        # This is more reliable than relying on example_origin flag from Step 1
        example_anchor_norms: set = set()
        _EXAMPLE_RE = re.compile(
            r"(?i)(?:ví\s+dụ|vi\s+du|VD|vd|chẳng\s+hạn|chang\s+han)\s*[:：]?\s*(.+?)(?:\)|$)",
            re.DOTALL,
        )
        _example_match = _EXAMPLE_RE.search(state.user_query or "")
        if _example_match:
            _example_text = normalize_text(str(_example_match.group(1) or ""), strip_punct=True)
            for e in ((state.metadata or {}).get("entities") or []):
                e_name = str(e.get("name") or "").strip()
                if e_name and normalize_text(e_name, strip_punct=True) in _example_text:
                    example_anchor_norms.add(normalize_text(e_name, strip_punct=True))
            if example_anchor_norms:
                logger.info("   -> Example anchors detected from query: %s", example_anchor_norms)

        grounding_entities = self._select_entities_for_grounding(state)
        if plan.disable_entity_grounding or plan.disable_non_location_grounding or plan.intent == IntentType.WEATHER_ADVICE:
            grounding_entities = [e for e in grounding_entities if e.get("type") == "Location"]
            logger.info("   -> [STEP 2] Grounding restricted to Location entities: %s", [e.get('name') for e in grounding_entities])
        # Separate example-origin entities — they're soft hints, not hard anchors
        example_entities = [e for e in grounding_entities if e.get("example_origin")]
        if example_entities:
            grounding_entities = [e for e in grounding_entities if not e.get("example_origin")]
            logger.info(
                "   -> Example entities separated from grounding: "
                f"{[e.get('name') for e in example_entities]} (soft hints only)"
            )
        if state.entities and len(grounding_entities) != len(state.entities):
            logger.info(
                "   -> Grounding entity filter applied: "
                f"{len(state.entities)} -> {len(grounding_entities)} (skip non-place entities)"
            )
        if grounding_entities:
            canonicalized_entities = self._canonicalize_entities_for_grounding(grounding_entities)
            if canonicalized_entities != grounding_entities:
                logger.info(
                    "   -> Entity canonicalization applied: "
                    f"{grounding_entities} -> {canonicalized_entities}"
                )
            grounding_entities = canonicalized_entities
        if grounding_entities:
            try:
                grounded_nodes = p.retriever.ground_entities(grounding_entities)
            except (Neo4jClientError, ServiceUnavailable) as e:
                logger.error("   -> [Grounding] Neo4j ground_entities failed: %s", e)
                grounded_nodes = []
            logger.info("   -> Grounded %s nodes exactly from Graph: %s", len(grounded_nodes), [n.content for n in grounded_nodes])
            logger.info("   -> Grounded Node IDs (preview): %s", self._preview_list([n.id for n in grounded_nodes if getattr(n, 'id', None)]))
        else:
            logger.info("   -> Grounding skipped: no entities extracted from query.")

        # Proximity Anchor Fallback: inject proximity anchor as grounding entity
        # when the regular entity extraction pipeline did not include it.
        if not grounding_entities:
            pa = (state.metadata or {}).get("proximity_anchor") or {}
            pa_text = str(pa.get("text") if isinstance(pa, dict) else pa or "").strip()
            if pa_text and (state.metadata or {}).get("proximity_anchor_required"):
                pa_type = str(pa.get("type") or "named_entity").strip()
                type_map = {"named_entity": "Place", "generic_feature": "Place"}
                grounding_entities = [{"name": pa_text, "type": type_map.get(pa_type, "Place")}]
                logger.info("   -> Proximity anchor fallback: injecting '%s' for grounding", pa_text)
                canonicalized_entities = self._canonicalize_entities_for_grounding(grounding_entities)
                grounding_entities = canonicalized_entities
                try:
                    grounded_nodes = p.retriever.ground_entities(grounding_entities)
                except (ValueError, RuntimeError, OSError) as e:
                    logger.error("   -> [Grounding] Proximity anchor fallback ground_entities failed: %s", e)
                    grounded_nodes = []
                logger.info("   -> Proximity anchor fallback: grounded %s nodes", len(grounded_nodes))

        # Post-extraction disambiguation: resolve ambiguous entity names
        # (e.g., 'Huế' as city vs dish) by expected type, then intent, then score
        grounded_nodes = self._disambiguate_grounded_nodes(grounded_nodes, grounding_entities, state)

        # V3 Anchor Fallback: only ground short, entity-like anchors
        # Skip for constrained_nearby_search — executor uses chain, not grounded anchors
        is_constrained_mode = plan.retrieval_mode == "constrained_nearby_search"
        if not grounded_nodes and state.metadata.get("v3_intent_data") and not is_constrained_mode:
            v3_anchors = state.metadata["v3_intent_data"].get("anchors") or []
            _NON_ENTITY_KEYWORDS = {"nửa ngày", "ngày", "đêm", "phù hợp", "địa điểm", "cái nào", "hay hơn", "tốt hơn"}
            # Skip example-origin anchors from V3 fallback (use pre-built norms)
            filtered_anchors = [
                a for a in v3_anchors
                if isinstance(a, str) and len(a.split()) <= 4
                and not any(kw in a.lower() for kw in _NON_ENTITY_KEYWORDS)
                and normalize_text(a, strip_punct=True) not in example_anchor_norms
            ]
            if filtered_anchors:
                logger.warning("   -> V3 anchor fallback triggered (grounded=0): anchors=%s", filtered_anchors)
                # Infer anchor type from intent instead of hardcoding TouristAttraction
                _INTENT_TO_ANCHOR_TYPE = {
                    IntentType.FOOD: "Restaurant",
                    IntentType.ACCOMMODATION: "Accommodation",
                    IntentType.EVENT: "Event",
                    IntentType.TRANSPORT_INFO: "TravelInfo",
                    IntentType.EMERGENCY_SUPPORT: "TravelInfo",
                    IntentType.CASHLESS_PAYMENT: "TravelInfo",
                    IntentType.WEATHER_ADVICE: "TravelInfo",
                }
                anchor_type = _INTENT_TO_ANCHOR_TYPE.get(IntentType.from_value(plan.intent), "TouristAttraction")
                fallback_entities = [{"name": a, "type": anchor_type} for a in filtered_anchors]
                canonical_fallback = self._canonicalize_entities_for_grounding(fallback_entities)
                grounded_nodes = p.retriever.ground_entities(canonical_fallback)
                logger.warning("   -> Grounded %s nodes via V3 fallback: %s", len(grounded_nodes), [n.content for n in grounded_nodes])

        # Fallback to inherited location grounding for follow-up queries when no entities were grounded
        if not grounded_nodes and plan.is_follow_up and state.location:
            logger.warning("   -> Follow-up location grounding fallback: attempting to ground inherited location '%s'", state.location)
            try:
                grounded_nodes = p.retriever.ground_entities([{"name": state.location, "type": "Location"}])
                if grounded_nodes:
                    logger.info("   -> Grounded %s nodes for inherited location: %s", len(grounded_nodes), [n.content for n in grounded_nodes])
            except (Neo4jClientError, ServiceUnavailable) as e:
                logger.error("   -> [Grounding] Neo4j grounding of inherited location failed: %s", e)

        is_multi_anchor = plan.retrieval_mode in {"comparison", "multi_candidate", "lodging_near_anchor", "tour_plan"}
        if is_multi_anchor or state.metadata.get("query_frame_multi_anchor_mode"):
            grounded_region = "all"
            state.runtime.metadata["grounding_region_consistency_skipped"] = "query_frame_multi_anchor"
        else:
            grounded_nodes, grounded_region = p._enforce_grounding_region_consistency(
                grounded_nodes,
                user_query=state.user_query,
                entities=state.entities,
            )

        relation_verification = self._verify_requested_relation_triples(state)
        if relation_verification.get("attempted"):
            state.runtime.metadata["relation_verification"] = {
                "matched": relation_verification.get("matched", False),
                "facts": relation_verification.get("facts", []),
                "failed_details": relation_verification.get("failed_details", []),
            }
            if relation_verification.get("matched"):
                relation_nodes = relation_verification.get("nodes") or []
                grounded_nodes = p.retriever._deduplicate_seeds(list(relation_nodes) + list(grounded_nodes))
                state.runtime.metadata["relation_verified"] = True
                state.runtime.metadata["relation_verified_facts"] = relation_verification.get("facts", [])
                logger.info(
                    "   -> Relation verifier matched triple(s), boosted anchors: "
                    f"{relation_verification.get('facts', [])}"
                )
            else:
                requested_rel = ((state.metadata.get("requested_relations") or ["quan hệ"])[0])
                requested_rel = relation_verification.get("relation") or requested_rel
                state.runtime.metadata["relation_verified"] = False
                state.runtime.metadata["relation_verification_failed"] = True
                state.runtime.metadata["relation_verification_relation"] = requested_rel
                state.runtime.metadata["relation_verification_entities"] = [
                    str(e.get("name") or "").strip()
                    for e in (state.entities or [])
                    if isinstance(e, dict) and str(e.get("name") or "").strip()
                ]
                state.runtime.metadata["relation_verification_is_hard"] = self._is_relation_verification_query(state.user_query)
                logger.info(
                    "   -> Relation verifier did not match requested triple; "
                    f"hard={state.runtime.metadata['relation_verification_is_hard']}."
                )

        if state.metadata.get("geo_scope") == "multi_region":
            logger.info("   -> [STEP 2] Multi-region query detected, bypassing single-region location overrides.")
            state.location = ""
            state.location_context = {"name": "all", "source": "contract", "confidence": 1.0}

        grounded_location_ctx = p._infer_grounded_location_context(grounded_nodes)
        grounded_loc = grounded_location_ctx.get("name") or ""

        # ── Explicit Query Region Override ──
        # Check if query has explicit region signal that should override current_location
        priority = p.location_grounding_service.resolve_location_priority(
            query=state.user_query,
            current_location=state.location,
            grounded_location=grounded_loc,
            grounded_reason=grounded_location_ctx.get("reason", ""),
            entities=metadata.get("entities", []),
        )
        if priority["source"] == "explicit_query_region":
            logger.info(
                f"   -> Explicit Query Region Override: "
                f"location='{priority['final_location']}', "
                f"region_focus='{priority['region_focus']}', "
                f"disable_filter={priority['disable_current_location_filter']}"
            )
            state.location = priority["final_location"]
            state.region_focus = priority["region_focus"]
            state.runtime.metadata["region_focus"] = priority["region_focus"]
            state.runtime.metadata["location_source"] = "explicit_query_region"
            state.runtime.metadata["disable_current_location_filter"] = priority["disable_current_location_filter"]
            # Filter grounded nodes to match the explicit region
            if priority["region_focus"] != "all":
                grounded_nodes = p._apply_region_focus_filter(grounded_nodes, priority["region_focus"])

        if grounded_loc and state.location and normalize_text(grounded_loc, strip_punct=True) != normalize_text(state.location):
            p.logger.warning(
                "Location conflict detected before retrieval: current='%s', grounded='%s', reason='%s'",
                state.location,
                grounded_loc,
                grounded_location_ctx.get("reason") or "",
            )
            if grounded_location_ctx.get("reason") == "grounded_entity_exact_match":
                # Admin level guard: don't let ward/commune override province/district
                user_loc_norm = normalize_text(state.location, strip_punct=True)
                grounded_loc_norm = normalize_text(grounded_loc, strip_punct=True)
                user_is_broad = p._is_broad_admin_location(user_loc_norm)
                grounded_is_narrow = p._is_narrow_admin_location(grounded_loc_norm)
                if user_is_broad and grounded_is_narrow:
                    # User asked about province/district, grounded entity is in a ward
                    # -> keep user's broader scope, don't override
                    state.runtime.metadata["location_override_reason"] = "blocked_narrow_over_broad"
                    state.runtime.metadata["location_source"] = state.location_context.get("source") or "user"
                    logger.info(
                        f"   -> Location override BLOCKED: user scope '{state.location}' "
                        f"(broad) vs grounded '{grounded_loc}' (narrow). Keeping user scope."
                    )
                else:
                    state.location_context = grounded_location_ctx
                    state.location = grounded_loc
                    state.runtime.metadata["location_override_reason"] = "grounded_entity_exact_match"
                    state.runtime.metadata["location_source"] = "ground_truth"
                    logger.info(
                        "   -> Ground-truth override applied: "
                        f"'{state.location}' from exact grounded entity."
                    )
            else:
                if state.has_explicit_location:
                    anchor_region = p._location_to_region_focus(state.location)
                    if anchor_region != "all":
                        grounded_nodes = p._apply_region_focus_filter(grounded_nodes, anchor_region)
                    state.runtime.metadata["location_override_reason"] = "soft_grounded_conflict_keep_explicit"
                    state.runtime.metadata["location_source"] = state.location_context.get("source") or "user"
                    logger.info(
                        "   -> Soft grounded location ignored: "
                        f"keep explicit location '{state.location}', region_focus='{anchor_region}'."
                    )
                elif not state.location:
                    state.location_context, overridden = p._choose_location_context(state.location_context, grounded_location_ctx)
                    if overridden:
                        state.location = state.location_context.get("name") or state.location
                        state.runtime.metadata["location_override_reason"] = grounded_location_ctx.get("reason")
                        state.runtime.metadata["location_source"] = grounded_location_ctx.get("source")
                else:
                    anchor_region = p._location_to_region_focus(state.location)
                    if anchor_region != "all":
                        grounded_nodes = p._apply_region_focus_filter(grounded_nodes, anchor_region)
                    state.runtime.metadata["location_override_reason"] = "soft_grounded_conflict_keep_user_anchor"
                    state.runtime.metadata["location_source"] = state.location_context.get("source") or "user"
                    logger.info(
                        "   -> Soft grounded location ignored: "
                        f"keep anchor '{state.location}', region_focus='{anchor_region}'."
                    )
        elif grounded_loc:
            state.location_context, overridden = p._choose_location_context(state.location_context, grounded_location_ctx)
            if overridden:
                state.location = state.location_context.get("name") or state.location

        active_anchor = None
        if grounded_nodes:
            active_anchor = self._build_grounded_anchor(grounded_nodes[0], state.location_context)
            state.runtime.metadata["last_grounded_anchor"] = active_anchor
        elif self._is_deictic_reference_query(state.user_query):
            active_anchor = state.metadata.get("active_grounded_anchor") or p.conversation_state.get("last_grounded_anchor") or {}
            if active_anchor:
                state.runtime.metadata["active_grounded_anchor"] = active_anchor
        # Graph signal: Union labels of all successfully grounded nodes into retrieval allowed labels
        # Skip Location label from broad location nodes (province/city anchors) to prevent
        # Location-type nodes from appearing in seed results for FOOD/ACCOMMODATION queries.
        if grounded_nodes:
            from graph_rag.modules.pipeline_support.admin_region_mapping_service import AdminRegionMappingService
            _admin_svc = AdminRegionMappingService()
            grounded_labels = set()
            for node in grounded_nodes:
                node_labels = [str(lbl) for lbl in get_node_labels(node)]
                is_broad_loc = (
                    "Location" in node_labels
                    and _admin_svc.is_broad_location(
                        str(getattr(node, "content", "") or (getattr(node, "metadata", {}) or {}).get("name") or "")
                    )
                )
                for lbl in node_labels:
                    if lbl and not (is_broad_loc and lbl == "Location"):
                        grounded_labels.add(lbl)
            if grounded_labels:
                forbidden_labels = set(plan.forbidden_labels or [])
                current_allowed = list(plan.target_labels or [])
                updated_allowed = sorted(list(set(current_allowed).union(grounded_labels) - forbidden_labels))
                
                # Update QueryPlan (single source of truth)
                import dataclasses
                state.query_plan = dataclasses.replace(state.query_plan, target_labels=tuple(updated_allowed))
                logger.info("   -> Grounded graph signal label expansion: %s -> %s", current_allowed, updated_allowed)

        pre_intent_filter_count = len(grounded_nodes)
        is_multi_anchor = plan.retrieval_mode in {"comparison", "multi_candidate", "lodging_near_anchor", "tour_plan"}
        if is_multi_anchor or state.metadata.get("query_frame_multi_anchor_mode"):
            state.runtime.metadata["grounding_intent_filter_skipped"] = "query_frame_multi_anchor"
            logger.info("   -> Grounding intent filter skipped for QueryFrame multi-anchor mode.")
        else:
            grounded_nodes = p._filter_grounded_nodes_for_intent(
                grounded_nodes,
                plan.intent,
                allowed_labels_override=list(plan.target_labels) or [],
                entities=state.entities,
            )
        if pre_intent_filter_count != len(grounded_nodes):
            logger.info(
                "   -> Grounding top-k by intent applied: "
                f"{pre_intent_filter_count} -> {len(grounded_nodes)} "
                f"for intent '{plan.intent}'"
            )

        if grounded_region != "all":
            state.runtime.metadata["region_focus"] = grounded_region
            state.region_focus = grounded_region
            logger.info("   -> Grounding consistency applied: region='%s', seeds=%s", grounded_region, len(grounded_nodes))

        state.runtime.metadata["location_context"] = state.location_context
        state.runtime.metadata["detected_location"] = state.location
        admin_rf = (state.metadata.get("admin_region_match") or {}).get("region_focus")
        if admin_rf in {"binh_dinh_old", "gia_lai_core", "gia_lai_new"}:
            from graph_rag.modules.pipeline_support.admin_region_mapping_service import AdminRegionMappingService
            state.region_focus = AdminRegionMappingService.resolve_region_focus(admin_rf)
            state.runtime.metadata["region_focus"] = state.region_focus
        elif state.location:
            resolved_region = p._location_to_region_focus(state.location)
            if resolved_region != "all":
                state.region_focus = resolved_region
                state.runtime.metadata["region_focus"] = resolved_region

        # If ground-truth override was applied, recalculate region from the grounded entity node itself
        if (state.metadata.get("location_override_reason") == "grounded_entity_exact_match" and
            grounded_nodes and hasattr(grounded_nodes[0], 'metadata')):
            entity_region = p.location_grounding_service._node_region(grounded_nodes[0])
            if entity_region != "unknown":
                state.region_focus = entity_region
                state.runtime.metadata["region_focus"] = entity_region
                logger.info("   -> Ground-truth override region recalculated from entity node: region='%s'", entity_region)

        # Multi-region guard: if query was detected as multi-region ("từ A đến B"),
        # restore region_focus=ALL after grounding overrides.
        # This prevents grounding from collapsing "Gia Lai + Bình Định" to just "Gia Lai".
        if state.metadata.get("region_focus_source") == "multi_region_detection" or plan.geo_scope == "multi_region":
            if state.region_focus != RegionFocus.ALL:
                logger.info("   -> Multi-region guard: restoring region_focus=ALL (was '%s')", state.region_focus)
                state.region_focus = RegionFocus.ALL
                state.runtime.metadata["region_focus"] = RegionFocus.ALL

        # Broad admin location filter removed — downstream RRF + MMR + BAAI reranker
        # handle ranking quality; pre-filtering broke ENTITY_FACT queries about locations.

        state.runtime.metadata["force_proximity_anchor"] = bool(
            grounded_location_ctx.get("reason") == "grounded_entity_exact_match"
        )
        if active_anchor and not state.metadata.get("last_grounded_anchor"):
            state.runtime.metadata["last_grounded_anchor"] = active_anchor

        # ── Comparison Subject Grounding Guard ──
        # Check if all comparison subjects are grounded
        comparison_subjects = [e for e in (state.entities or []) if isinstance(e, dict) and e.get("source") == "comparison_subject"]
        if comparison_subjects:
            grounded_names = {normalize_text(str(getattr(n, "content", "") or ""), strip_punct=True) for n in grounded_nodes}
            subject_status = []
            for subj in comparison_subjects:
                subj_name = str(subj.get("name") or "").strip()
                subj_norm = normalize_text(subj_name, strip_punct=True)
                is_grounded = any(subj_norm in gn or gn in subj_norm for gn in grounded_names if gn)
                subject_status.append({"name": subj_name, "grounded": is_grounded})
                if not is_grounded:
                    logger.info("   -> [ComparisonGuard] Subject NOT grounded: '%s'", subj_name)
            state.runtime.metadata["comparison_subject_grounding"] = subject_status
            all_grounded = all(s["grounded"] for s in subject_status)
            state.runtime.metadata["comparison_all_subjects_grounded"] = all_grounded
            if not all_grounded:
                logger.info("   -> [ComparisonGuard] Not all subjects grounded — will use partial/clarification in Step 5", )

        logger.info(
            "   -> Grounding Output: "
            f"grounded_nodes={len(grounded_nodes)}, final_location='{state.location}', "
            f"region_focus='{state.metadata.get('region_focus', 'all')}'"
        )
        logger.info("   -> Step 2 completed in %s", self._elapsed(step_2_start))

        state.grounded_nodes = grounded_nodes
