from __future__ import annotations
"""Step 4: Dual-stream retrieval, graph traversal, and pruning."""
import logging

logger = logging.getLogger(__name__)


import time


from typing import Any, Dict



from graph_rag.config import ENABLE_AGENTIC_RETRIEVAL, GRAPH_RAG_V3_ENABLED


from graph_rag.config.region_patterns import DISCOVERY_LOCATION_MIN_LENGTH

from graph_rag.core.thresholds import INSUFFICIENT_FACT_THRESHOLD, INSUFFICIENT_FACT_THRESHOLD_FOLLOWUP


from graph_rag.core.intents import IntentType


from graph_rag.utils.text import normalize_text


from ..context_stage import ContextStage


from ..dto import PipelineRunState


class Step4RetrievalMixin:
    """Mixin providing Step 4 retrieval and pruning."""

    def _run_step_4_retrieve_and_prune(self, state: PipelineRunState) -> Dict[str, Any]:
        """Step 4 orchestrator: seeds -> comparison/ranking -> guards -> web fallback/prune."""
        logger.info("\n [STEP 4] DUAL-STREAM RETRIEVAL...")
        step_4_start = time.time()

        state.runtime.metadata["current_location"] = state.location
        state.runtime.metadata["grounded_anchor_nodes"] = state.grounded_nodes
        if state.metadata.get("relation_verification_failed") and state.metadata.get("relation_verification_is_hard"):
            logger.error("   -> Relation verification guard: hard verification query failed; forcing abstain.")
            return self._relation_verification_abstain(state)
        # Read from QueryPlan -- single source of truth (Milestone 2: no metadata fallback)
        plan = state.query_plan
        assert plan is not None, "QueryPlan must be initialized in Step 1"

        logger.info(
            "   -> Retrieval Input: "
            f"intent='{plan.intent}', location='{state.location}', grounded_anchors={len(state.grounded_nodes)}"
        )

        # --- 1. Seed retrieval, agentic retrieval, region filtering ---
        all_seeds, n_discovered, n_grounded = self._retrieve_seeds_and_context(state, plan)

        # --- 2. Comparison detection, ranking, graph traversal ---
        comp_result = self._handle_comparison_and_ranking(state, all_seeds, step_4_start, n_grounded, n_discovered)
        if "early_result" in comp_result:
            return comp_result
        all_seeds = comp_result["all_seeds"]
        raw_context = comp_result["raw_context"]

        # --- 3. Context validation guards ---
        guard_result = self._validate_and_guard_context(state, all_seeds, raw_context)
        if "early_result" in guard_result:
            return guard_result
        raw_context = guard_result["raw_context"]

        # --- 4. Web fallback, context validation, pruning ---
        return self._apply_web_fallback_and_prune(
            state, all_seeds, raw_context,
            comp_result["context_stage"],
            comp_result["use_context_v2"],
            comp_result["v2_organizer"],
            step_4_start,
        )

    # ------------------------------------------------------------------
    # Sub-methods for _run_step_4_retrieve_and_prune
    # ------------------------------------------------------------------

    def _retrieve_seeds_and_context(self, state: PipelineRunState, plan) -> tuple:
        """Retrieve seeds: find_seeds, agentic retrieval, label filtering, region focus.

        Returns (all_seeds, n_discovered, n_grounded).
        """
        p = self.pipeline

        # Pass query_plan to find_seeds for typed semantic contract
        logger.info("   -> [DEBUG-STEP4] Calling find_seeds with search_query='%s'", str(state.search_query or "")[:80])
        _t4a = time.time()
        discovered_seeds = p.retriever.find_seeds(
            state.search_query,
            metadata=state.metadata,
            rank=False,
            query_plan=state.query_plan,
        )
        logger.info("   -> [DEBUG-STEP4] find_seeds returned %d seeds in %.2fs", len(discovered_seeds), time.time() - _t4a)

        agentic_intents = {IntentType.TOUR_PLAN, IntentType.DISCOVERY}
        allow_agentic_for_multi_intent = bool(state.metadata.get("is_multi_intent_travel"))
        has_explicit_destinations = self._has_two_or_more_explicit_destinations(state.entities)
        is_tour_list = plan.answer_mode == "tour_list"
        is_global_discovery = plan.is_discovery()
        disable_agentic_retrieval = plan.disable_agentic_retrieval
        should_run_agentic = (
            ENABLE_AGENTIC_RETRIEVAL
            and (IntentType.from_value(plan.intent) in agentic_intents or allow_agentic_for_multi_intent)
            and not has_explicit_destinations
            and not is_tour_list
            and not is_global_discovery
            and not disable_agentic_retrieval
        )
        if should_run_agentic:
            pre_agentic_count = len(discovered_seeds)
            discovered_seeds = p.agentic_retriever.retrieve_iterative(
                query=state.search_query,
                metadata=state.metadata,
                initial_results=state.grounded_nodes + discovered_seeds,
            )
            logger.info(
                "   -> Agentic retrieval applied: "
                f"{pre_agentic_count} -> {len(discovered_seeds)} discovered seeds, "
                f"reason={'multi_intent_travel' if allow_agentic_for_multi_intent and IntentType.from_value(plan.intent) not in agentic_intents else 'intent_whitelist'}"
            )
        else:
            if has_explicit_destinations:
                logger.info("   -> Agentic retrieval skipped: query already contains >=2 explicit destinations.")
            elif is_global_discovery:
                logger.info("   -> Agentic retrieval skipped: global_discovery query with clear topic.")
            elif is_tour_list:
                logger.info("   -> Agentic retrieval skipped: tour_list mode.")
            elif disable_agentic_retrieval:
                logger.info("   -> Agentic retrieval skipped: disabled by contract.")
            else:
                logger.info("   -> Agentic retrieval skipped for current intent/config.")

        # Filter grounded nodes by allowed labels to prevent irrelevant
        # categories (Dish, Accommodation) from polluting discovery results.
        # EXCEPTION: Always keep Location nodes -- they're used for geographic
        # expansion in the traverser (Location expansion finds nearby entities).
        grounded_nodes = list(state.grounded_nodes)
        allowed_labels = set(plan.target_labels) if plan.target_labels else set()
        if allowed_labels and grounded_nodes:
            pre_filter = len(grounded_nodes)
            grounded_nodes = [
                n for n in grounded_nodes
                if set(n.metadata.get("labels") or []) & allowed_labels
                or "Location" in set(n.metadata.get("labels") or [])
            ]
            if len(grounded_nodes) != pre_filter:
                logger.info(
                    "   -> Grounded label filter: %d -> %d nodes (allowed=%s)",
                    pre_filter, len(grounded_nodes), sorted(allowed_labels),
                )

        all_seeds = p.retriever._deduplicate_seeds(grounded_nodes + discovered_seeds)
        logger.info(
            "   -> Seed merge/dedup: "
            f"grounded={len(grounded_nodes)}, discovered={len(discovered_seeds)}, merged={len(all_seeds)}"
        )

        # --- Region focus filtering ---
        pre_focus_count = len(all_seeds)
        has_specific_target = bool((state.metadata or {}).get("target_entity"))
        has_query_frame_explicit_anchors = bool(plan.anchors)
        has_admin_region_lock = bool((state.metadata or {}).get("region_group")) and (
            (state.metadata or {}).get("region_lock_mode") != "disabled_multi_anchor_comparison"
        )
        # Skip region filter for global analysis queries that need data across regions
        # HARD RULE: When user asks to analyze all data or build routes from data,
        # do NOT filter by current_location -- use all regions.
        q_norm_region = normalize_text(state.user_query, strip_punct=True)
        is_global_analysis = any(
            token in q_norm_region
            for token in [
                "dua tren du lieu", "du lieu", "phan tich tiem nang", "toan bo", "tat ca",
                "cac nha nghi", "cac di tich", "xay dung tuyen", "ket hop",
                "các nhà nghỉ", "các di tích", "xây dựng tuyến", "kết hợp",
            ]
        ) and any(
            token in q_norm_region
            for token in ["phan tich", "tiem nang", "phat trien", "xay dung tuyen", "ket hop"]
        )
        if is_global_analysis and not has_admin_region_lock:
            state.runtime.metadata["region_filter_skipped"] = "global_analysis_query"
            logger.info("   -> Region focus filter skipped: global analysis query needs cross-region data.")
            region_filter_applied = False
        elif has_query_frame_explicit_anchors and not has_admin_region_lock:
            state.runtime.metadata["region_filter_skipped"] = "query_frame_explicit_anchors"
            logger.info("   -> Region focus filter skipped: QueryFrame explicit anchors/candidates.")
            region_filter_applied = False
        elif has_specific_target and IntentType.from_value(plan.intent) in {
            IntentType.ENTITY_FACT,
            IntentType.EVENT,
            IntentType.FOOD,
            IntentType.ACCOMMODATION,
            IntentType.TOURISM,
        } and not has_admin_region_lock:
            # ENFORCEMENT: If user explicitly mentioned a region (e.g., "Binh Dinh"),
            # do NOT skip region filter even with specific target
            user_mentioned_region = bool(state.location) and len(state.location.strip()) > 2
            if user_mentioned_region and state.region_focus not in ("", "all"):
                filtered_seeds = p._apply_region_focus_filter(all_seeds, state.region_focus)
                if filtered_seeds:
                    all_seeds = filtered_seeds
                    region_filter_applied = state.region_focus != "all"
                    logger.info("   -> Region focus '%s' ENFORCED (user mentioned region): %s -> %s seeds", state.region_focus, pre_focus_count, len(all_seeds))
                else:
                    logger.info("   -> Region focus filter skipped: specific target entity is grounded/requested.")
                    region_filter_applied = False
            else:
                logger.info("   -> Region focus filter skipped: specific target entity is grounded/requested.")
                region_filter_applied = False
        elif pre_focus_count == 0:
            # No seeds at all -- applying region filter is pointless, skip to allow fallback retrieval
            state.runtime.metadata["region_filter_skipped"] = "zero_seeds"
            logger.info("   -> Region focus filter skipped: no seeds to filter (would remain zero).")
            region_filter_applied = False
        elif IntentType.from_value(plan.intent) == IntentType.DISCOVERY and not has_admin_region_lock:
            # Discovery queries cast a wider net -- don't hard filter by region
            # But still filter if query has a specific location mentioned
            if state.location and len(state.location.strip()) > DISCOVERY_LOCATION_MIN_LENGTH:
                filtered_seeds = p._apply_region_focus_filter(all_seeds, state.region_focus)
                if filtered_seeds:
                    all_seeds = filtered_seeds
                    region_filter_applied = state.region_focus != "all"
                    logger.info("   -> Region focus '%s' applied for DISCOVERY with location: %s -> %s seeds", state.region_focus, pre_focus_count, len(all_seeds))
                else:
                    state.runtime.metadata["region_filter_skipped"] = "discovery_intent"
                    logger.warning("   -> Region focus filter skipped: DISCOVERY intent uses all regions (empty filtered fallback).")
                    region_filter_applied = False
            else:
                state.runtime.metadata["region_filter_skipped"] = "discovery_intent"
                logger.info("   -> Region focus filter skipped: DISCOVERY intent uses all regions.")
                region_filter_applied = False
        else:
            filtered_seeds = p._apply_region_focus_filter(all_seeds, state.region_focus)
            if filtered_seeds:
                all_seeds = filtered_seeds
                region_filter_applied = state.region_focus != "all"
            else:
                state.runtime.metadata["region_filter_skipped"] = "empty_filtered_seeds_fallback"
                logger.info("   -> Region focus filter '%s' returned 0 seeds. Falling back to unfiltered seeds.", state.region_focus)
                region_filter_applied = False
        if region_filter_applied:
            logger.info(
                f"   -> Region focus '{state.region_focus}' applied: {pre_focus_count} -> {len(all_seeds)} seeds"
            )
        elif not has_specific_target:
            logger.info("   -> Region focus filter skipped: using all regions.")

        return all_seeds, len(discovered_seeds), len(grounded_nodes)

    def _handle_comparison_and_ranking(self, state, all_seeds, step_4_start, n_grounded, n_discovered) -> Dict[str, Any]:
        """Comparison detection, CandidatePool/PolicyRanker ranking, graph traversal, ContextStage.

        Returns either {"early_result": ...} or
        {"all_seeds", "raw_context", "context_stage", "use_context_v2", "v2_organizer"}.
        """
        p = self.pipeline
        plan = state.query_plan

        # --- Comparison detection ---
        frame = (state.metadata or {}).get("query_frame") or {}
        is_comparison = (
            plan.retrieval_mode == "comparison"
            or frame.get("query_operator") == "comparison"
        )
        if not is_comparison:
            q_norm_for_comparison = normalize_text(state.user_query, strip_punct=True)
            suppress_auto_comparison = any(
                marker in q_norm_for_comparison
                for marker in [
                    "tham gia tour",
                    "tour ",
                    "tuyen tham quan",
                    "tuyến tham quan",
                    "tuyen",
                    "tuyến",
                    "ket hop",
                    "kết hợp",
                    "goi y",
                    "gợi ý",
                    "lich trinh",
                    "lịch trình",
                    "diem tham quan trong tour",
                    "diem tham quan",
                    "cong ty to chuc tour",
                    "công ty tổ chức tour",
                    # Listing/discovery markers -- not comparison
                    "co nhung",
                    "có những",
                    "nao noi tieng",
                    "nào nổi tiếng",
                    "khong the bo qua",
                    "không thể bỏ qua",
                    "dia danh",
                    "địa danh",
                    "dia diem noi tieng",
                    "địa điểm nổi tiếng",
                    "vi du",
                    "ví dụ",
                    # Recommendation / topic discovery markers
                    "nen di",
                    "nên đi",
                    "nen den",
                    "nên đến",
                    "nen tham quan",
                    "nên tham quan",
                    "tim hieu",
                    "tìm hiểu",
                ]
            )
            unique_ents = []
            if not suppress_auto_comparison:
                for entity in state.entities or []:
                    # Skip example-origin entities -- they're soft hints, not comparison subjects
                    if entity.get("example_origin"):
                        continue
                    if self._is_groundable_entity(entity):
                        name = str(entity.get("name") or "").strip()
                        if name and name not in unique_ents:
                            unique_ents.append(name)
            if not suppress_auto_comparison and len(unique_ents) >= 2:
                is_comparison = True

        # Final guard: suppress comparison when query has analysis signals.
        # "giua A va B" in analysis context means "combining", not "comparing".
        # This must run AFTER auto-detection to catch all paths.
        if is_comparison:
            q_norm_analysis = normalize_text(state.user_query, strip_punct=True)
            has_analysis_signal = any(
                token in q_norm_analysis
                for token in ["phan tich", "chien luoc", "tiem nang", "phat trien", "xay dung tuyen", "ket hop", "danh gia"]
            )
            if has_analysis_signal:
                is_comparison = False
                state.runtime.metadata["comparison_suppressed_by_analysis"] = True
                logger.info("   -> Comparison suppressed: query has analysis signals, deferring to analysis handler.")

        if is_comparison:
            subjects = self._comparison_subject_names(state)
            if len(subjects) < 2:
                fallback_subjects: list[str] = []
                for entity in state.entities or []:
                    if not self._is_groundable_entity(entity):
                        continue
                    name = str(entity.get("name") or "").strip()
                    if name:
                        fallback_subjects.append(name)
                if fallback_subjects:
                    subjects = list(dict.fromkeys(subjects + fallback_subjects))
                    if len(subjects) >= 2:
                        state.runtime.metadata["query_frame_anchor_names"] = subjects
                        state.runtime.metadata["query_frame_multi_anchor_mode"] = True
                        state.runtime.metadata["comparison_subjects_from_fallback"] = True
            # Early type lookup: fire before seed guard so factual category
            # answers are never overridden by LLM-generated traversal context.
            # Only short-circuit for simple category-only questions.
            if len(subjects) >= 2:
                q_norm_early = normalize_text(state.user_query, strip_punct=True)
                is_category_only = not any(
                    marker in q_norm_early
                    for marker in [
                        "nha nghi", "khach san", "gan", "luu tru", "phong",
                        "gia", "dat", "lich trinh", "an uong", "nha hang",
                        "mon an", "check in", "photo", "chup anh",
                        # Comparison/analysis markers -- don't short-circuit
                        "so sanh", "diem giong", "diem khac", "phan biet",
                        "dac diem", "khac biet", "giong nhau",
                    ]
                )
                # QueryFrameV2 comparison operator takes priority over marker check
                if is_category_only and (plan.retrieval_mode if plan else (state.metadata or {}).get("retrieval_plan_mode")) == "comparison":
                    is_category_only = False
                early_type = self._answer_comparison_type_lookup_if_possible(state, subjects)
                if early_type and is_category_only:
                    state.runtime.metadata["comparison_type_lookup_deterministic"] = True
                    state.runtime.metadata["comparison_subjects_expected"] = subjects
                    logger.info("   -> Comparison type lookup (early): returning deterministic category answer.")
                    return {
                        "early_result": {
                            "answer": early_type,
                            "metadata": state.runtime.metadata,
                        }
                    }
                elif early_type:
                    # Store for later use in LLM context, don't short-circuit
                    state.runtime.metadata["comparison_type_lookup_result"] = early_type
                    logger.info("   -> Comparison type lookup (early): stored for multi-part question context.")
            covered, missing = self._comparison_subject_seed_coverage(subjects, all_seeds)
            state.runtime.metadata["comparison_subjects_expected"] = subjects
            state.runtime.metadata["comparison_subjects_covered"] = covered
            state.runtime.metadata["comparison_subjects_missing"] = missing
            if subjects and missing and not covered:
                state.runtime.metadata["comparison_missing_seed_guard"] = True
                grouped = {subject: [] for subject in subjects}
                answer = (
                    self._answer_comparison_type_lookup_if_possible(state, subjects)
                    or self._build_comparison_deterministic_answer(state, subjects, grouped)
                )
                logger.info("   -> Comparison seed guard: missing anchors, returning deterministic answer.")
                return {
                    "early_result": {
                        "answer": answer,
                        "metadata": state.runtime.metadata,
                    }
                }

        # --- Phase 3: CandidatePool + PolicyRanker re-ranking ---
        if all_seeds:
            from graph_rag.core.candidate_pool import CandidatePool
            from graph_rag.core.policy_ranker import PolicyRanker
            from graph_rag.core.retrieval_policy import RetrievalPolicy

            # Build ExclusionContext for follow-up deduplication (single normalize pass)
            from graph_rag.pipeline.orchestration.exclusion_context import ExclusionContext
            conversation_state = self.pipeline.location_grounding_service.conversation_state
            exclusion_ctx = ExclusionContext.build_from_conversation_state(
                conversation_state=conversation_state,
                is_follow_up=plan.is_follow_up,
                raw_context_len=0,  # raw_context not yet computed; updated after graph expansion
                threshold=0,
            )
            # Merge resolved.exclude_entities from ConversationStateResolver as fallback
            resolved = state.resolved_query_frame
            if resolved and resolved.exclude_entities:
                merged = exclusion_ctx.entity_names | set(resolved.exclude_entities)
                if merged != exclusion_ctx.entity_names:
                    exclusion_ctx = ExclusionContext(
                        entity_names=merged,
                        should_force_deterministic=exclusion_ctx.should_force_deterministic,
                    )
            exclusion_set = exclusion_ctx.entity_names
            state.runtime.metadata["exclusion_context"] = exclusion_ctx
            if exclusion_set:
                logger.info("   -> [Phase3] Follow-up exclusion: %s previous entities to exclude", len(exclusion_set))

            policy = RetrievalPolicy.resolve_policy_from_query_plan(plan)
            pool = CandidatePool.from_nodes(all_seeds, plan, policy)
            logger.info(
                "   -> [Phase3] CandidatePool built:\n"
                f"      nodes={len(pool.nodes)}, "
                f"source_breakdown={pool.source_breakdown}, "
                f"label_distribution={pool.label_distribution}"
            )

            # BGE candidate scoring -- semantic relevance before PolicyRanker
            from graph_rag.config import (
                ENABLE_BGE_CANDIDATE_SCORING,
                BGE_CANDIDATE_SCORING_MODEL,
                BGE_CANDIDATE_SCORE_WEIGHT,
                BGE_CANDIDATE_SCORING_TIMEOUT_SEC,
            )
            if ENABLE_BGE_CANDIDATE_SCORING and pool.nodes:
                from graph_rag.modules.context.bge_scorer import score_candidates_bge
                score_candidates_bge(
                    query_text=state.search_query or plan.query,
                    nodes=pool.nodes,
                    model_name=BGE_CANDIDATE_SCORING_MODEL,
                    weight=BGE_CANDIDATE_SCORE_WEIGHT,
                    timeout_sec=BGE_CANDIDATE_SCORING_TIMEOUT_SEC,
                )

            ranker = PolicyRanker()
            ranked_pool = ranker.rank(
                pool,
                exclusion_set=exclusion_set,
                entities=state.entities,
                region_focus=state.region_focus,
                detected_location=state.location,
                grounded_anchor_nodes=state.grounded_nodes,
            )
            all_seeds = ranked_pool.nodes

            # Filter out already answered candidates if we have other choices remaining
            if plan.is_follow_up and exclusion_set:
                filtered_seeds = [
                    s for s in all_seeds
                    if normalize_text(s.metadata.get("name") or s.content or "", strip_punct=True) not in exclusion_set
                ]
                if filtered_seeds:
                    logger.info("   -> [Phase3] Filtered out %s already answered seeds.", len(all_seeds) - len(filtered_seeds))
                    all_seeds = filtered_seeds
                else:
                    logger.warning("   -> [Phase3] All candidates were already answered. Keeping them as fallback.")

            # If follow-up exclusion removed all candidates, log clearly
            if exclusion_set and not all_seeds:
                logger.info("   -> [Phase3] All candidates excluded by follow-up memory -- no new entities available")

            top_ranked_log = [
                f"{node.metadata.get('name') or node.content} (final={node.score:.3f}, orig={(ranked_pool.score_breakdown.get(node.id) or ranked_pool).original_score:.3f})"
                for node in all_seeds[:5]
            ]
            logger.info("   -> [Phase3] PolicyRanker finished ranking. Top candidates: %s", top_ranked_log)

        # --- Empty seeds guard ---
        if not all_seeds:
            logger.info("   -> [DEBUG-STEP4] Step 4 completed in %s with EMPTY seeds (grounded=%d, discovered=%d). Returning early_result.",
                        self._elapsed(step_4_start), n_grounded, n_discovered)
            return {
                "early_result": {
                    "answer": "Xin lỗi, tôi không tìm thấy thông tin nào liên quan tới khu vực này trong dữ liệu.",
                    "metadata": state.runtime.metadata,
                }
            }

        # --- Graph expansion ---
        logger.info("\n [STEP 4b] GRAPH EXPANSION (Traversing)...")
        step_4b_start = time.time()
        # Sanitize location_filter: use matched_alias from admin mapping if available,
        # otherwise strip parenthetical display strings like "Gia Lai (Khu vuc bien / Quy Nhon cu)"
        traversal_location = state.location
        if traversal_location:
            admin_alias = (state.metadata or {}).get("matched_admin_alias") or ""
            if admin_alias:
                traversal_location = admin_alias
            elif "(" in traversal_location:
                traversal_location = traversal_location.split("(")[0].strip()
        is_multi_anchor = plan.retrieval_mode in {"comparison", "multi_candidate", "lodging_near_anchor", "tour_plan"}
        if is_multi_anchor:
            traversal_location = ""
            state.runtime.metadata["traversal_location_filter_skipped"] = "query_frame_multi_anchor"
        has_specific_target = bool((state.metadata or {}).get("target_entity"))
        if has_specific_target and IntentType.from_value(plan.intent) in {
            IntentType.ENTITY_FACT,
            IntentType.EVENT,
            IntentType.FOOD,
            IntentType.ACCOMMODATION,
        }:
            traversal_location = ""
        logger.info(
            "   -> Traversal Input: "
            f"seed_count={len(all_seeds)}, location_filter='{traversal_location}', intent='{plan.intent}'"
        )
        # Read traversal policy from QueryPlan -- single source of truth (Milestone 2)
        traversal_intent = state.metadata.get("query_frame_traversal_intent") or plan.intent
        traversal_relations = (
            list(plan.required_relations_for_retrieval)
            or list(plan.requested_relations)
            or state.metadata.get("query_frame_traversal_relations")
            or []
        )
        if traversal_intent != plan.intent or traversal_relations != list(plan.requested_relations):
            logger.info(
                "   -> QueryFrame traversal policy: "
                f"intent='{traversal_intent}', rels={traversal_relations}"
            )
        raw_context = p.traverser.traverse(
            all_seeds,
            intent=traversal_intent,
            location_filter=traversal_location,
            requested_attributes=list(plan.requested_attributes),
            requested_relations=traversal_relations,
            allowed_labels=list(plan.target_labels),
        )
        if GRAPH_RAG_V3_ENABLED and (state.metadata or {}).get("v3_intent_data"):
            try:
                v3_intent_data = state.metadata["v3_intent_data"]
                v3_grouped_facts = p.multi_anchor_retriever.retrieve(
                    v3_intent_data,
                    metadata=state.metadata,
                )
                v3_validation = p.completeness_gate.validate(v3_intent_data, v3_grouped_facts)
                v3_structured_context = p.structured_context_builder.build(
                    v3_intent_data,
                    v3_grouped_facts,
                    v3_validation,
                )
                state.runtime.metadata["v3_grouped_facts"] = v3_grouped_facts
                state.runtime.metadata["v3_validation"] = v3_validation
                state.runtime.metadata["v3_structured_context"] = v3_structured_context
                v3_lines = [
                    line.strip()
                    for line in str(v3_structured_context or "").splitlines()
                    if line.strip()
                ]
                if v3_lines:
                    raw_context = v3_lines + list(raw_context or [])
                logger.info(
                    "   -> V3 multi-anchor retrieval: "
                    f"anchors={len(v3_grouped_facts)}, "
                    f"context_state='{v3_validation.get('context_state')}', "
                    f"structured_lines={len(v3_lines)}"
                )
            except (ValueError, RuntimeError, OSError) as exc:
                state.runtime.metadata["v3_error"] = f"{type(exc).__name__}: {exc}"
                logger.warning("   -> V3 warning: %s: %s", type(exc).__name__, exc)
        context_stage = ContextStage(self)
        raw_context, use_context_v2, v2_organizer = context_stage.prepare_raw_context(
            state=state,
            all_seeds=all_seeds,
            raw_context=raw_context,
        )
        logger.info("   -> Step 4b completed in %s", self._elapsed(step_4b_start))

        return {
            "all_seeds": all_seeds,
            "raw_context": raw_context,
            "context_stage": context_stage,
            "use_context_v2": use_context_v2,
            "v2_organizer": v2_organizer,
        }

    def _validate_and_guard_context(self, state, all_seeds, raw_context) -> Dict[str, Any]:
        """Context guards: no-context, entity validation, light abstain, comparison balancing, insufficient context.

        Returns either {"early_result": ...} or {"raw_context": ...}.
        """
        plan = state.query_plan

        # NO-CONTEXT GUARD: bat buoc abstain neu traversal khong tra ve fact nao
        # Ngan LLM hallucination khi co seeds nhung khong co context
        if not raw_context and IntentType.from_value(plan.intent) != IntentType.TOUR_PLAN:
            entity_names = [s.metadata.get("name") or s.content for s in all_seeds[:3]]
            logger.info(
                "   -> No-context guard: 0 facts from traversal despite "
                f"{len(all_seeds)} seeds {entity_names}. Forcing abstain."
            )
            return {
                "early_result": {
                    "answer": (
                        "Xin lỗi, mình chưa có đủ thông tin chi tiết về "
                        f"{', '.join(entity_names)} trong hệ thống dữ liệu du lịch. "
                        "Bạn có thể cung cấp thêm thông tin hoặc hỏi về địa điểm khác không?"
                    ),
                    "metadata": state.runtime.metadata,
                }
            }

        skip_entity_validation = bool(
            plan.retrieval_mode in {"comparison", "multi_candidate", "global_discovery"}
            or plan.is_discovery()
            or plan.is_follow_up
            or plan.hard_label_contract
        )
        if self._requires_entity_validation(state) and not skip_entity_validation:
            target_entity = self._primary_specific_entity_name(state)
            if not self._retrieval_evidence_contains_entity(target_entity, all_seeds, raw_context):
                logger.info(
                    "   -> Entity validation guard: retrieval evidence does not contain "
                    f"target entity '{target_entity}'."
                )
                return self._entity_validation_abstain(state, target_entity)

        light_abstain = self._light_abstain_after_context_filter(state, raw_context)
        if light_abstain:
            logger.info("   -> Light abstain gate: target not found in filtered context.")
            return light_abstain

        # Balance context for comparison queries
        raw_context = self.balance_comparison_context(state, raw_context)

        # Insufficient context guard: when facts are too few for LLM to generate
        # a reliable answer, force deterministic renderer to avoid hallucination.
        # Threshold is lower for follow-up queries because they are context-specific.
        # Values from thresholds.json -> pipeline.insufficient_fact_threshold
        _INSUFFICIENT_FACT_THRESHOLD = INSUFFICIENT_FACT_THRESHOLD_FOLLOWUP if plan.is_follow_up else INSUFFICIENT_FACT_THRESHOLD
        if (len(raw_context) <= _INSUFFICIENT_FACT_THRESHOLD
                and IntentType.from_value(plan.intent) in {IntentType.FOOD, IntentType.TOURISM, IntentType.ACCOMMODATION}
                and not state.runtime.metadata.get("answer_mode_override")):
            # Force deterministic answer mode to prevent LLM hallucination
            from graph_rag.core.answer_mode import AnswerMode
            if IntentType.from_value(plan.intent) == IntentType.FOOD:
                state.runtime.metadata["answer_mode_override"] = AnswerMode.FACT_ANSWER
                # Update ExclusionContext with should_force_deterministic
                from graph_rag.pipeline.orchestration.exclusion_context import ExclusionContext
                existing_ctx = state.runtime.metadata.get("exclusion_context")
                if existing_ctx:
                    existing_ctx.should_force_deterministic = True
                else:
                    state.runtime.metadata["exclusion_context"] = ExclusionContext(
                        entity_names=set(), should_force_deterministic=True,
                    )
                logger.info("   -> [InsufficientContext] %s facts < %s. Forcing deterministic renderer for %s.", len(raw_context), _INSUFFICIENT_FACT_THRESHOLD, plan.intent)

        return {"raw_context": raw_context}

    def _apply_web_fallback_and_prune(self, state, all_seeds, raw_context, context_stage, use_context_v2, v2_organizer, step_4_start) -> Dict[str, Any]:
        """Web search fallback, context validation, pruning, and state finalization."""
        plan = state.query_plan

        state.raw_context = raw_context
        context_validation = self._validate_requested_context(state, raw_context)
        state.runtime.metadata["context_validation"] = context_validation

        # --- Web Search Fallback for missing attributes ---
        if not context_validation.get("ok", True):
            missing_attrs = context_validation.get("missing_attributes") or []
            web_results = self._try_web_search_fallback(state, missing_attrs)
            if web_results:
                raw_context.extend(web_results)
                state.raw_context = raw_context
                # Re-validate after web search
                context_validation = self._validate_requested_context(state, raw_context)
                state.runtime.metadata["context_validation"] = context_validation
                state.runtime.metadata["web_search_used"] = True
                state.runtime.metadata["web_search_result_count"] = len(web_results)
                logger.info("   -> Web search added %d context lines. Re-validation ok=%s", len(web_results), context_validation.get("ok"))

        if self._should_hard_fail_context_validation(state, context_validation):
            logger.info(
                "   -> Context validation guard: missing requested scope "
                f"attrs={context_validation.get('missing_attributes')}, "
                f"rels={context_validation.get('missing_relations')}."
            )
            return self._requested_context_abstain(state, context_validation)
        if not context_validation.get("ok", True):
            logger.info(
                "   -> Context validation warning: missing requested scope "
                f"attrs={context_validation.get('missing_attributes')}, "
                f"rels={context_validation.get('missing_relations')}."
            )
            # Don't degrade itinerary_build to partial_fact_answer --
            # route optimizer should still run even with partial context.
            # Don't degrade discovery/list queries either -- they don't need
            # every attribute to produce a useful ranked list.
            from graph_rag.core.answer_mode import AnswerMode
            skip_degrade = False
            if plan and plan.operation.value == "itinerary_build":
                skip_degrade = True
            current_mode = plan.answer_mode
            if "discovery" in current_mode:
                skip_degrade = True
            if not skip_degrade:
                state.runtime.metadata["answer_mode"] = AnswerMode.PARTIAL_FACT_ANSWER

        clean_context = context_stage.prune_context(
            state=state,
            all_seeds=all_seeds,
            raw_context=raw_context,
            use_context_v2=use_context_v2,
            v2_organizer=v2_organizer,
        )
        logger.info("   -> Step 4 total elapsed: %s", self._elapsed(step_4_start))

        state.all_seeds = all_seeds
        state.ranked_candidates = all_seeds
        if state.metadata is None:
            state.metadata = {}
        state.runtime.metadata["ranked_candidate_nodes"] = all_seeds
        state.raw_context = raw_context
        state.clean_context = clean_context
        return {}

    def _try_web_search_fallback(self, state, missing_attributes: list) -> list[str]:
        """Attempt web search for missing attributes. Returns context lines or empty list."""
        try:
            from graph_rag.config import WEB_SEARCH_ENABLED
            if not WEB_SEARCH_ENABLED:
                return []
        except ImportError:
            return []

        if not missing_attributes:
            return []

        from graph_rag.services.web_search import WebSearchService

        # Find fetchable attributes
        service = WebSearchService()
        fetchable = [a for a in missing_attributes if service.is_fetchable(a)]
        if not fetchable:
            return []

        # Extract entity name from state
        entity_name = ""
        if state.all_seeds:
            # Use the top-ranked seed as the entity
            top_seed = state.all_seeds[0] if isinstance(state.all_seeds, list) else None
            if top_seed:
                entity_name = getattr(top_seed, "name", "") or str(top_seed)
        if not entity_name:
            entity_name = (state.metadata or {}).get("target_entity", "")
        if not entity_name:
            # Try to extract from query
            entity_name = state.user_query

        # Extract location
        location = state.location or ""

        all_lines = []
        for attr in fetchable[:2]:  # Limit to 2 attributes to avoid too many API calls
            logger.info("   -> [WebSearch] Searching for '%s' of '%s'...", attr, entity_name)
            results = service.search_entity_attribute(entity_name, attr, location)
            lines = service.format_as_context(results, entity_name, attr)
            all_lines.extend(lines)

        if all_lines:
            logger.info("   -> [WebSearch] Found %d context lines for %s", len(all_lines), fetchable[:2])

        return all_lines
