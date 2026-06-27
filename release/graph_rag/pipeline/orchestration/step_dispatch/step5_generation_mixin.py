from __future__ import annotations
"""Step 5: Answer generation dispatcher."""

import logging
import time

from graph_rag.config import GRAPH_RAG_V3_ENABLED
from graph_rag.core.answer_mode import AnswerMode
from graph_rag.core.intents import IntentType
from graph_rag.utils.text import normalize_text
from ..dto import PipelineRunState

logger = logging.getLogger(__name__)


class Step5GenerationMixin:
    """Mixin providing the main Step 5 answer generation dispatcher."""

    def _run_step_5_generate_answer(self, state: PipelineRunState, on_token=None) -> None:
        p = self.pipeline
        logger.info("\n [STEP 5] SYNTHESIS GENERATION...")
        step_5_start = time.time()
        generator_candidates = self._build_generator_candidates(state.all_seeds)
        full_generator_candidates = list(generator_candidates)
        # Read from QueryPlan — single source of truth (Milestone 2: no metadata fallback)
        plan = state.query_plan
        assert plan is not None, "QueryPlan must be initialized in Step 1"
        answer_mode = plan.answer_mode or AnswerMode.FACT_ANSWER
        question_type = (state.metadata or {}).get("question_type") or ""
        # Renderer hint from QueryPlan — used for dispatch shortcuts
        renderer = plan.renderer

        # This prevents misrouting like "Tour 1 ngày giá bao nhiêu" → itinerary_build.
        from graph_rag.core.state import QueryOperation
        op = plan.operation

        # attribute_lookup → fact_answer, NEVER route optimizer
        if op == QueryOperation.ATTRIBUTE_LOOKUP and answer_mode == AnswerMode.TOUR_PLAN:
            answer_mode = AnswerMode.FACT_ANSWER
            state.runtime.metadata["answer_mode"] = answer_mode
            state.runtime.metadata["operation_override"] = "attribute_lookup → fact_answer"
            logger.info("   -> [Operation] Override: attribute_lookup forces fact_answer (was tour_plan)", )

        # availability_search → tour_list, NEVER itinerary
        if op == QueryOperation.AVAILABILITY_SEARCH and answer_mode == AnswerMode.TOUR_PLAN:
            answer_mode = getattr(AnswerMode, "TOUR_LIST", "tour_list")
            state.runtime.metadata["answer_mode"] = answer_mode
            state.runtime.metadata["operation_override"] = "availability_search → tour_list"
            logger.info("   -> [Operation] Override: availability_search forces tour_list (was tour_plan)", )

        # Route optimization must only run for itinerary_build operation.
        # Closed-form or analysis questions may be misclassified as TOUR_PLAN by
        # retrieval intent, but should not inherit itinerary metadata/templates.
        is_itinerary_op = (op == QueryOperation.ITINERARY_BUILD)

        if answer_mode == AnswerMode.TOUR_PLAN and is_itinerary_op:
            self._build_tour_route_metadata(state, generator_candidates)
            generator_candidates = state.runtime.metadata.get("route_seed_nodes") or generator_candidates

        logger.info(
            "   -> Generation Input: candidate_nodes=%s, detected_location='%s', intent='%s', answer_mode='%s'",
            len(generator_candidates), state.location, plan.intent, answer_mode
        )

        # ============================================================
        # PHASE 0: ANSWER MODE DISPATCH — closed-form NEVER fall through
        # ============================================================

        if answer_mode == AnswerMode.FILL_BLANK_SHORT:
            answer = self._dispatch_fill_blank(state)
            if not answer or not answer.strip():
                answer = "Không đủ thông tin trong dữ liệu để điền vào chỗ trống."
            state.answer = self._sanitize_answer_text(answer)
            logger.info("   -> Step 5 completed in %s [mode=fill_blank_short]", self._elapsed(step_5_start))
            return

        if answer_mode == AnswerMode.TRUE_FALSE_VERIFIER:
            answer = self._dispatch_true_false(state)
            if not answer or not answer.strip():
                answer = "Không đủ thông tin trong dữ liệu để xác minh."
            state.answer = self._sanitize_answer_text(answer)
            logger.info("   -> Step 5 completed in %s [mode=true_false_verifier]", self._elapsed(step_5_start))
            return

        if answer_mode == AnswerMode.SINGLE_OPTION_RESOLVER:
            answer = self._dispatch_option_resolver(state, multi=False)
            if not answer or not answer.strip():
                answer = "Không đủ thông tin trong dữ liệu để xác minh."
            state.answer = self._sanitize_answer_text(answer)
            logger.info("   -> Step 5 completed in %s [mode=single_option_resolver]", self._elapsed(step_5_start))
            return

        if answer_mode == AnswerMode.MULTI_OPTION_RESOLVER:
            answer = self._dispatch_option_resolver(state, multi=True)
            if not answer or not answer.strip():
                answer = "Không đủ thông tin trong dữ liệu để xác minh."
            state.answer = self._sanitize_answer_text(answer)
            logger.info("   -> Step 5 completed in %s [mode=multi_option_resolver]", self._elapsed(step_5_start))
            return

        if answer_mode == AnswerMode.NEGATIVE_ABSTAIN_GUARD:
            answer = self._dispatch_negative_guard(state)
            if not answer or not answer.strip():
                answer = "Xin lỗi, hệ thống dữ liệu du lịch hiện chưa có đủ thông tin để trả lời câu hỏi này."
            state.answer = self._sanitize_answer_text(answer)
            logger.info("   -> Step 5 completed in %s [mode=negative_abstain_guard]", self._elapsed(step_5_start))
            return

        # BELONGS_TO classification: deterministic renderer (no LLM needed)
        # e.g. "Làng Du lịch cộng đồng Mơ Hra thuộc loại hình du lịch nào?"
        belongs_answer = self._answer_belongs_to_classification_if_possible(state)
        if belongs_answer:
            state.answer = self._sanitize_answer_text(belongs_answer)
            logger.info("   -> Step 5 completed in %s [mode=belongs_to_classification_deterministic]", self._elapsed(step_5_start))
            return

        # Food specialty: deterministic renderer (no LLM needed)
        # e.g. "Ở Gia Lai có đặc sản gì?"
        food_answer = self._answer_food_specialty_deterministic(state)
        if food_answer:
            # Curated food context available → let LLM curate instead of returning raw list
            curated_food_ctx = state.runtime.metadata.get("curated_food_context")
            if curated_food_ctx:
                logger.info("   -> [Curated] Skip deterministic, use LLM curation (%d chars)", len(curated_food_ctx))
            else:
                state.answer = self._sanitize_answer_text(food_answer)
                logger.info("   -> Step 5 completed in %s [mode=food_specialty_deterministic]", self._elapsed(step_5_start))
                return

        cultural_lodging_itinerary = (
            self._answer_lodging_cultural_itinerary_if_possible(state)
            if answer_mode != AnswerMode.TOUR_PLAN
            else None
        )
        if cultural_lodging_itinerary:
            state.runtime.metadata["lodging_cultural_itinerary_deterministic"] = True
            state.answer = self._sanitize_answer_text(cultural_lodging_itinerary)
            logger.info("   -> Step 5 completed in %s [mode=lodging_cultural_itinerary_deterministic]", self._elapsed(step_5_start))
            return

        # Early lodging+heritage analysis: must run before V3 structured generation
        # to handle "phân tích tiềm năng nhà nghỉ + di tích" queries.
        lodging_heritage_early = self._answer_lodging_heritage_strategy_if_possible(state)
        if lodging_heritage_early:
            state.runtime.metadata["lodging_heritage_strategy_short_circuit"] = True
            state.answer = self._sanitize_answer_text(lodging_heritage_early)
            logger.info("   -> Step 5 completed in %s [mode=lodging_heritage_early]", self._elapsed(step_5_start))
            return

        # TOUR_LIST: user asks "có tour nào?" — list/rank Tour nodes, NO route optimization
        if answer_mode == AnswerMode.TOUR_LIST:
            answer = self._dispatch_tour_list(state, generator_candidates, full_generator_candidates)
            state.answer = self._sanitize_answer_text(answer)
            logger.info("   -> Step 5 completed in %s [mode=tour_list]", self._elapsed(step_5_start))
            return

        # DISCOVERY_LIST: topic-based discovery — deterministic list from grounded candidates, NO LLM
        # FIX: Always try deterministic first regardless of rich context count.
        # Rich context was causing skip → LLM received only 4 pruned facts → apology.
        # Now: deterministic first, fall through to LLM only if deterministic returns empty.
        food_specialty_skip = (state.metadata or {}).get("food_specialty_skip_deterministic")
        raw_facts = state.raw_context or []
        rich_context_count = len([f for f in raw_facts if isinstance(f, str) and len(f.strip()) > 20])
        if answer_mode == AnswerMode.DISCOVERY_LIST:
            answer = self._dispatch_discovery_list(state, generator_candidates)
            if answer:
                state.answer = self._sanitize_answer_text(answer)
                logger.info("   -> Step 5 completed in %s [mode=discovery_list]", self._elapsed(step_5_start))
                return
            # Fall through to LLM — either curated context was prepared (ACCOMMODATION/TOURISM/FOOD),
            # or deterministic rendering truly had no output.
            _has_curated = any(
                state.runtime.metadata.get(k)
                for k in ("curated_accommodation_context", "curated_tourism_context",
                           "curated_food_context", "curated_event_context")
            )
            if _has_curated:
                logger.info("   -> [DiscoveryList] Curated context prepared, delegating to LLM for synthesis.")
            else:
                logger.info("   -> [DiscoveryList] No deterministic output (%d rich facts available), falling through to LLM.", rich_context_count)

        # TOUR_PLAN uses compose logic (from candidates), not v3 exact lookup.
        # Skip v3_structured for TOUR_PLAN — let _dispatch_tour_plan handle it.
        # Skip v3_structured for comparison — let comparison engine handle it.
        is_comparison_v3 = renderer == "comparison"
        if answer_mode != AnswerMode.TOUR_PLAN and not is_comparison_v3 and self._should_route_v3_structured_generation(state, answer_mode):
            answer = self._generate_v3_structured_answer(state)
            state.answer = self._sanitize_answer_text(answer)
            logger.info("   -> Step 5 completed in %s [mode=v3_structured_answer]", self._elapsed(step_5_start))
            return

        # AIRPORT_INFO: deterministic renderer from TravelInfo
        if answer_mode == AnswerMode.AIRPORT_INFO:
            answer = self._dispatch_airport_info(state, generator_candidates)
            if answer:
                state.answer = self._sanitize_answer_text(answer)
                logger.info("   -> Step 5 completed in %s [mode=airport_info]", self._elapsed(step_5_start))
                return

        # EMERGENCY_INFO_DETERMINISTIC: deterministic renderer for emergency support
        if answer_mode == "emergency_info_deterministic":
            emergency_answer = self._answer_emergency_info_if_possible(state)
            if emergency_answer:
                state.answer = self._sanitize_answer_text(emergency_answer)
                logger.info("   -> Step 5 completed in %s [mode=emergency_info_deterministic]", self._elapsed(step_5_start))
                return

        # travel_info_topic_deterministic: deterministic renderer for TravelInfo topics
        fallback_policy = plan.fallback_policy or ""
        is_travel_info_query = (
            answer_mode == "travel_info_topic_deterministic"
            or fallback_policy.endswith("_guided_fallback")
        )
        if is_travel_info_query:
            travel_info_ans = self._answer_travel_info_topic_deterministic(state)
            if not travel_info_ans:
                # Try fallback lookup directly to load from Neo4j if seeds are empty
                travel_info_ans = self._check_guided_fallbacks(state)
            if travel_info_ans:
                state.answer = self._sanitize_answer_text(travel_info_ans)
                logger.info("   -> Step 5 completed in %s [mode=travel_info_topic_deterministic]", self._elapsed(step_5_start))
                return

        if answer_mode == AnswerMode.TOUR_PLAN:
            # Skip tour plan for comparison questions — let comparison logic handle it
            q_norm_for_tp = normalize_text(state.user_query, strip_punct=True)
            is_comparison_tp = (
                "so sanh" in q_norm_for_tp
                or renderer == "comparison"
            )
            if not is_comparison_tp:
                answer = self._dispatch_tour_plan(state, generator_candidates, full_generator_candidates)
                state.answer = self._sanitize_answer_text(answer)
                logger.info("   -> Step 5 completed in %s [mode=tour_plan]", self._elapsed(step_5_start))
                return

        if answer_mode == AnswerMode.OPEN_ANALYSIS:
            is_comparison = renderer in {"comparison", "multi_candidate", "lodging_near_anchor", "tour_plan"}
            if not is_comparison:
                category_answer = self._answer_global_category_listing_if_possible(state)
                if category_answer:
                    state.runtime.metadata["global_category_listing_deterministic"] = True
                    state.answer = self._sanitize_answer_text(category_answer)
                    logger.info("   -> Step 5 completed in %s [mode=global_category_listing_deterministic]", self._elapsed(step_5_start))
                    return
            answer = self._dispatch_open_analysis(state, full_generator_candidates)
            state.answer = self._sanitize_answer_text(answer)
            logger.info("   -> Step 5 completed in %s [mode=open_analysis]", self._elapsed(step_5_start))
            return

        if answer_mode in {AnswerMode.FACT_ANSWER, AnswerMode.PARTIAL_FACT_ANSWER}:
            is_comparison = renderer in {"comparison", "multi_candidate", "lodging_near_anchor", "tour_plan"}
            if (state.metadata or {}).get("comparison_suppressed_by_analysis"):
                is_comparison = False
            if is_comparison:
                common_missing_answer = self._answer_comparison_common_missing_attributes_if_possible(state, self._comparison_subject_names(state))
                if common_missing_answer:
                    state.runtime.metadata["fact_answer_common_missing_attributes"] = True
                    state.answer = self._sanitize_answer_text(common_missing_answer)
                    logger.info("   -> Step 5 completed in %s [mode=fact_answer_common_missing_attributes]", self._elapsed(step_5_start))
                    return
                common_location_answer = self._answer_comparison_location_lookup_if_possible(state, self._comparison_subject_names(state))
                if common_location_answer:
                    state.runtime.metadata["fact_answer_comparison_location_lookup"] = True
                    state.answer = self._sanitize_answer_text(common_location_answer)
                    logger.info("   -> Step 5 completed in %s [mode=fact_answer_comparison_location_lookup]", self._elapsed(step_5_start))
                    return
                comparison_answer = self._dispatch_comparison_analysis(state)
                if comparison_answer:
                    # Validate that answer mentions both comparison subjects
                    missing_subjects = self.validate_comparison_answer(state, comparison_answer)
                    if missing_subjects:
                        logger.info("   -> [ComparisonValidator] Missing subjects: %s", missing_subjects)
                        state.runtime.metadata["comparison_answer_missing_subjects"] = missing_subjects
                        # Don't return — fall through to LLM generation for better answer
                    else:
                        state.runtime.metadata["fact_answer_comparison_renderer"] = True
                        state.answer = self._sanitize_answer_text(comparison_answer)
                        logger.info("   -> Step 5 completed in %s [mode=fact_answer_comparison_renderer]", self._elapsed(step_5_start))
                        return

            category_answer = self._answer_global_category_listing_if_possible(state)
            if category_answer:
                state.runtime.metadata["global_category_listing_deterministic"] = True
                state.answer = self._sanitize_answer_text(category_answer)
                logger.info("   -> Step 5 completed in %s [mode=global_category_listing_deterministic]", self._elapsed(step_5_start))
                return
            event_time_answer = self._answer_event_time_fact_if_possible(state)
            if event_time_answer:
                state.runtime.metadata["event_time_fact_deterministic"] = True
                state.answer = self._sanitize_answer_text(event_time_answer)
                logger.info("   -> Step 5 completed in %s [mode=event_time_fact_deterministic]", self._elapsed(step_5_start))
                return
            attribute_answer = self._answer_requested_attribute_from_context(state)
            if attribute_answer:
                # For recommendation intents, skip deterministic and let LLM
                # generate a more natural answer from the retrieved context.
                _RECOMMENDATION_INTENTS = {
                    "FOOD_RECOMMENDATION", "EVENT_RECOMMENDATION",
                    "ACCOMMODATION_RECOMMENDATION", "TOURISM_RECOMMENDATION",
                }
                _intent_upper = str(getattr(plan, 'intent', '') or '').upper()
                if _intent_upper in _RECOMMENDATION_INTENTS:
                    logger.info("   -> [requested_attribute] Skipping deterministic for recommendation intent='%s', falling through to LLM.", _intent_upper)
                else:
                    state.runtime.metadata["requested_attribute_deterministic"] = True
                    state.answer = self._sanitize_answer_text(attribute_answer)
                    logger.info("   -> Step 5 completed in %s [mode=requested_attribute_deterministic]", self._elapsed(step_5_start))
                    return
            # Deterministic abstain for ATTRIBUTE_LOOKUP: if we can't resolve
            # the attribute from context, don't fall through to LLM.
            # This prevents slow LLM calls for "SĐT của X", "giá vé Y", etc.
            # But if we HAVE context (facts), allow LLM to generate answer.
            if plan and plan.operation.value == "attribute_lookup":
                has_context = bool(state.all_seeds or getattr(state, 'context_facts', None))
                allow_llm = bool((state.metadata or {}).get("closed_form_allow_llm", False)) or has_context
                if not allow_llm:
                    # TravelInfo rescue: nếu có TravelInfo seed khớp intent,
                    # gọi guided fallback để lấy data từ Neo4j
                    _intent = str((state.metadata or {}).get("intent") or "").upper()
                    _TRAVEL_INFO_INTENTS = {"TRAVEL_ADVICE", "WEATHER_ADVICE", "TRANSPORT_INFO"}
                    _has_travelinfo_seed = any(
                        "TravelInfo" in str((s.metadata or {}).get("labels") or [])
                        for s in (state.all_seeds or [])
                    )
                    if _has_travelinfo_seed and _intent in _TRAVEL_INFO_INTENTS:
                        # Gọi guided fallback để inject TravelInfo description
                        travel_info_ans = self._check_guided_fallbacks(state)
                        if travel_info_ans:
                            state.answer = self._sanitize_answer_text(travel_info_ans)
                            state.runtime.metadata["travel_info_rescue"] = True
                            logger.warning("   -> [TravelInfo Rescue] Guided fallback returned data for intent='%s'.", _intent)
                            logger.info("   -> Step 5 completed in %s [mode=travel_info_rescue]", self._elapsed(step_5_start))
                            return
                        # Fallback không có data → dùng fact_answer với context hiện có
                        state.runtime.metadata["answer_mode"] = AnswerMode.FACT_ANSWER
                        state.runtime.metadata["travel_info_rescue"] = True
                        logger.warning("   -> [TravelInfo Rescue] No fallback data. Falling through to fact_answer.", )
                        # Không return — cho fallthrough xuống PHASE 1
                    else:
                        state.answer = self._sanitize_answer_text(
                            "Không đủ thông tin trong dữ liệu để trả lời."
                        )
                        state.runtime.metadata["attribute_lookup_abstain"] = True
                        logger.info("   -> Step 5 completed in %s [mode=attribute_lookup_abstain]", self._elapsed(step_5_start))
                        return
            dish_answer = self._answer_dish_to_restaurant_if_possible(state)
            if dish_answer:
                state.runtime.metadata["dish_to_restaurant_deterministic"] = True
                state.answer = self._sanitize_answer_text(dish_answer)
                logger.info("   -> Step 5 completed in %s [mode=dish_to_restaurant_deterministic]", self._elapsed(step_5_start))
                return
            main_entity = self._analysis_main_entity_name(state)
            # Skip slot builder for classification/type queries — let Phase 1 handle them
            # with _answer_location_type_if_possible which uses BELONGS_TO relationships
            q_norm_skip = normalize_text(state.user_query, strip_punct=True)
            is_classification_query = any(token in q_norm_skip for token in [
                "thuoc loai", "loai hinh", "la loai", "phan loai", "thuoc nhom", "danh muc",
            ])
            if not GRAPH_RAG_V3_ENABLED and not is_classification_query:
                slot_answer = self._build_slot_based_open_answer(state, main_entity)
                if slot_answer:
                    state.runtime.metadata["fact_answer_slot_builder"] = True
                    state.runtime.metadata["slot_answer_context"] = slot_answer
                    logger.info("   -> [SlotBuilder] Built factual context (%d chars), delegating to LLM for friendly rephrase.", len(slot_answer))

        # ============================================================
        # PHASE 1: WATERFALL (kept for Open-Ended / Fact / Analysis)
        # Preserves existing behavior to avoid regression.
        # ============================================================

        description_fill_blank_result = self._answer_description_fill_blank_if_possible(state)
        if description_fill_blank_result:
            state.runtime.metadata.update(description_fill_blank_result.get("metadata") or {})
            state.answer = self._sanitize_answer_text(description_fill_blank_result.get("answer") or "")
            logger.info("   -> Generation short-circuit: description fill-blank answer applied.")
            logger.info("   -> Step 5 completed in %s", self._elapsed(step_5_start))
            return

        shared_location_result = self._answer_shared_location_fill_blank_if_possible(state)
        if shared_location_result:
            state.runtime.metadata.update(shared_location_result.get("metadata") or {})
            state.answer = self._sanitize_answer_text(shared_location_result.get("answer") or "")
            logger.info("   -> Generation short-circuit: shared-location fill-blank answer applied.")
            logger.info("   -> Step 5 completed in %s", self._elapsed(step_5_start))
            return

        tour_offer_answer = self._answer_tour_offer_includes_if_possible(state)
        if tour_offer_answer:
            logger.info("   -> Generation short-circuit: deterministic tour offer/includes answer applied.")
            answer = tour_offer_answer
            state.runtime.metadata["tour_offer_includes_short_circuit"] = True
        else:
            answer = ""

        nearby_reason_answer = self._answer_nearby_reason_if_possible(state)
        if not answer and nearby_reason_answer:
            logger.info("   -> Generation short-circuit: deterministic nearby reason answer applied.")
            answer = nearby_reason_answer
            state.runtime.metadata["nearby_reason_short_circuit"] = True

        lodging_heritage_answer = self._answer_lodging_heritage_strategy_if_possible(state)
        if not answer and lodging_heritage_answer:
            logger.info("   -> Generation short-circuit: deterministic lodging heritage strategy applied.")
            answer = lodging_heritage_answer
            state.runtime.metadata["lodging_heritage_strategy_short_circuit"] = True

        spatial_strategy_answer = self._answer_spatial_strategy_analysis_from_context_if_possible(state)
        if not answer and spatial_strategy_answer:
            logger.info("   -> Generation short-circuit: deterministic spatial strategy analysis applied.")
            answer = spatial_strategy_answer
            state.runtime.metadata["spatial_strategy_analysis_short_circuit"] = True

        nearby_accommodation_context_answer = self._answer_nearby_accommodation_from_context_if_possible(state)
        if not answer and nearby_accommodation_context_answer:
            logger.info("   -> Generation short-circuit: deterministic nearby accommodation context answer applied.")
            answer = nearby_accommodation_context_answer
            state.runtime.metadata["nearby_accommodation_context_short_circuit"] = True

        nearby_cultural_context_answer = self._answer_nearby_cultural_from_context_if_possible(state)
        if not answer and nearby_cultural_context_answer:
            logger.info("   -> Generation short-circuit: deterministic nearby cultural context answer applied.")
            answer = nearby_cultural_context_answer
            state.runtime.metadata["nearby_cultural_context_short_circuit"] = True

        attraction_classification_answer = self._answer_attraction_classification_analysis_if_possible(state)
        if not answer and attraction_classification_answer:
            logger.info("   -> Generation short-circuit: deterministic attraction classification analysis applied.")
            answer = attraction_classification_answer
            state.runtime.metadata["attraction_classification_analysis_short_circuit"] = True

        location_type_answer = self._answer_location_type_if_possible(state)
        if not answer and location_type_answer:
            logger.info("   -> Generation short-circuit: deterministic location/type answer applied.")
            answer = location_type_answer
            state.runtime.metadata["location_type_short_circuit"] = True

        menu_items_answer = self._answer_menu_items_if_possible(state)
        if not answer and menu_items_answer:
            logger.info("   -> Generation short-circuit: deterministic menu items answer applied.")
            answer = menu_items_answer
            state.runtime.metadata["menu_items_short_circuit"] = True

        deterministic_answer = self._build_entity_fact_fallback_answer(
            state,
            full_generator_candidates,
        )

        guided_fallback = self._check_guided_fallbacks(state)
        if not answer and guided_fallback:
            logger.warning("   -> Generation short-circuit: guided fallback applied.")
            answer = guided_fallback
            state.runtime.metadata["guided_fallback_applied"] = True

        if not answer and deterministic_answer:
            logger.warning("   -> Generation short-circuit: deterministic ENTITY_FACT fallback applied.")
            answer = deterministic_answer
        elif not answer:
            # Check for slot builder context (factual info card for LLM rephrase)
            slot_answer_ctx = state.runtime.metadata.get("slot_answer_context")
            # Check for curated contexts (prepared by deterministic renderer)
            curated_food_ctx = state.runtime.metadata.get("curated_food_context")
            curated_event_ctx = state.runtime.metadata.get("curated_event_context")
            curated_tourism_ctx = state.runtime.metadata.get("curated_tourism_context")
            curated_accommodation_ctx = state.runtime.metadata.get("curated_accommodation_context")

            if slot_answer_ctx:
                anchored_context = slot_answer_ctx
                logger.info("   -> [SlotBuilder] Using slot answer context for LLM rephrase (%d chars)", len(anchored_context))
            elif curated_food_ctx:
                anchored_context = curated_food_ctx
                logger.info("   -> [Curated] Using curated food context (%d chars)", len(anchored_context))
            elif curated_event_ctx:
                anchored_context = curated_event_ctx
                logger.info("   -> [Curated] Using curated event context (%d chars)", len(anchored_context))
            elif curated_tourism_ctx:
                anchored_context = curated_tourism_ctx
                logger.info("   -> [Curated] Using curated tourism context (%d chars)", len(anchored_context))
            elif curated_accommodation_ctx:
                # Merge curated context with graph facts if curated context is too thin
                if len(curated_accommodation_ctx) < 200 and state.clean_context:
                    anchored_context = curated_accommodation_ctx + "\n\n---\nTHÔNG TIN CHI TIẾT:\n" + state.clean_context
                    logger.info("   -> [Curated] Enriched accommodation context: curated=%d + graph=%d = %d chars",
                                len(curated_accommodation_ctx), len(state.clean_context), len(anchored_context))
                else:
                    anchored_context = curated_accommodation_ctx
                    logger.info("   -> [Curated] Using curated accommodation context (%d chars)", len(anchored_context))
            else:
                # Build anchored context if grounded entity is TouristAttraction but asking about nearby food/accommodation
                anchored_context = state.clean_context
                grounded_attractions = [
                    n for n in (state.all_seeds or [])
                    if "TouristAttraction" in (n.metadata.get("labels") or [])
                ]
                is_nearby_query = any(
                    token in normalize_text(state.user_query, strip_punct=True)
                    for token in ["gan", "gần", "near", "o gan", "ở gần", "nam gan", "nằm gần"]
                )
                if grounded_attractions and is_nearby_query and IntentType.from_value(plan.intent) in {IntentType.FOOD, IntentType.ACCOMMODATION}:
                    anchor_name = grounded_attractions[0].metadata.get("name") or grounded_attractions[0].content
                    anchor_ctx = f"**ANCHOR POINT:** Người dùng đang hỏi về những địa điểm gần {anchor_name}. Khi trả lời, hãy LUÔN refer về địa điểm này bằng đúng tên: '{anchor_name}'.\n\n"
                    anchored_context = anchor_ctx + state.clean_context
                    logger.info("   -> Anchored context injection: anchor='%s' (type=TouristAttraction)", anchor_name)

            # Keep state.clean_context in sync with the actual anchored_context passed to the LLM generator
            # so downstream validations and retries use the correct/complete context.
            state.clean_context = anchored_context

            validation_can_replace_answer = bool(
                state.query_plan is not None
                and (
                    IntentType.from_value(plan.intent) not in {IntentType.DISCOVERY, IntentType.TOUR_PLAN, IntentType.DISTANCE}
                    or self._is_service_availability_query(state.user_query)
                )
            )
            generation_on_token = None if validation_can_replace_answer else on_token
            if on_token and validation_can_replace_answer:
                state.runtime.metadata["streaming_suppressed_for_validation"] = True
                logger.warning("   -> Streaming suppressed: answer may be replaced by validation/fallback.")

            answer = p.generator.generate(
                user_query=state.user_query,
                context_text=anchored_context,
                intent=plan.intent,
                detected_location=state.location,
                candidate_nodes=full_generator_candidates,
                strict_route_nodes=state.runtime.metadata.get("route_seed_nodes") if IntentType.from_value(plan.intent) == IntentType.TOUR_PLAN else None,
                dropped_route_points=state.runtime.metadata.get("dropped_route_points") if IntentType.from_value(plan.intent) == IntentType.TOUR_PLAN else None,
                daily_cluster_plan=state.runtime.metadata.get("daily_cluster_plan") if IntentType.from_value(plan.intent) == IntentType.TOUR_PLAN else None,
                route_optimizer_metrics=state.runtime.metadata.get("route_optimizer_metrics") if IntentType.from_value(plan.intent) == IntentType.TOUR_PLAN else None,
                context_validation=state.runtime.metadata.get("context_validation")
                if (state.runtime.metadata.get("context_validation") or {}).get("ok", True)
                else None,
                on_token=generation_on_token,
                query_state=state.query_plan,
                answer_mode=answer_mode,
            )

            # --- Empty Answer Guard ---
            if not answer or not answer.strip():
                logger.warning("   -> [Step5] LLM returned empty answer, using fallback.")
                answer = "Xin lỗi, tôi không tìm thấy thông tin đủ để trả lời câu hỏi này trong dữ liệu hiện có."

            # --- Anti-Apology Guard ---
            # FIX: Check clean_context (pruned, what LLM sees) first, fallback to raw_context.
            # Previously only checked raw_context — if raw had facts but pruned was too thin,
            # LLM would still apologize and guard wouldn't trigger properly.
            # _context_has_facts expects List[str], so split clean_context if it's a string.
            _guard_context = state.raw_context or []
            if state.clean_context:
                _guard_context = [line for line in state.clean_context.splitlines() if line.strip()] or _guard_context
            if (
                self._is_apology_answer(answer)
                and self._context_has_facts(_guard_context)
                and not self._is_service_availability_query(state.user_query)
            ):
                target_entity = self._primary_specific_entity_name(state)
                context_entity_matches = True
                if target_entity:
                    context_entity_matches = self._retrieval_evidence_contains_entity(
                        target_entity, state.all_seeds or [], state.raw_context or []
                    )
                if context_entity_matches and target_entity:
                    question_location = self._extract_location_from_query(state.user_query)
                    if question_location:
                        # Use clean_context (what LLM actually sees) for location check
                        context_text = state.clean_context or " ".join(str(item or "") for item in state.raw_context or [])
                        context_norm = normalize_text(context_text, strip_punct=True)
                        location_norm = normalize_text(question_location, strip_punct=True)
                        if location_norm and location_norm not in context_norm:
                            context_entity_matches = False
                            logger.info(
                                f"   -> Anti-apology guard: entity found but location '{question_location}' "
                                "not in context. Keeping abstain."
                            )
                if context_entity_matches:
                    target_category = (state.metadata or {}).get("multi_choice_target_category", "")
                    question_type_local = (state.metadata or {}).get("question_type") or ""
                    if not question_type_local:
                        q_norm = normalize_text(state.user_query, strip_punct=True)
                        if any(token in q_norm for token in ["dung hay sai", "true or false", "dung hay khong dung"]):
                            question_type_local = "True-or-False"
                    if question_type_local == "True-or-False":
                        answer = "Không đủ thông tin trong dữ liệu để xác minh."
                        state.runtime.metadata["apology_guard_triggered"] = True
                        state.runtime.metadata["apology_guard_target_entity"] = target_entity
                        logger.info("   -> Anti-apology guard: True-or-False question, using abstain format.")
                    elif question_type_local in {"Multi-Choice", "Multi-Select"}:
                        answer = "Không đủ thông tin trong dữ liệu để xác minh."
                        state.runtime.metadata["apology_guard_triggered"] = True
                        state.runtime.metadata["apology_guard_target_entity"] = target_entity
                        logger.info("   -> Anti-apology guard: %s question, using abstain format.", question_type_local)
                    else:
                        fallback = None
                        # Unified Contract: derive intent from query_plan only
                        is_food = plan and "FOOD" in str(plan.intent).upper()
                        if is_food:
                            # Read should_force_deterministic from ExclusionContext
                            exclusion_ctx = state.runtime.metadata.get("exclusion_context")
                            orig_force = exclusion_ctx.should_force_deterministic if exclusion_ctx else False
                            # Temporarily force deterministic for the anti-apology call
                            state.runtime.metadata["force_deterministic"] = True
                            fallback = self._answer_food_specialty_deterministic(state)
                            if not fallback:
                                # Restore original value (don't leave stale force_deterministic)
                                if not orig_force:
                                    state.runtime.metadata.pop("force_deterministic", None)
                        if not fallback:
                            fallback = self._build_context_based_answer(state, target_category=target_category)
                        if fallback:
                            logger.info(
                                "   -> Anti-apology guard: LLM returned apology but context has facts. "
                                f"Using deterministic fallback ({len(fallback)} chars)."
                            )
                            answer = fallback
                            state.runtime.metadata["apology_guard_triggered"] = True
                            state.runtime.metadata["apology_guard_target_entity"] = target_entity
                else:
                    logger.info(
                        "   -> Anti-apology guard SKIPPED: context entity does not match "
                        f"target '{target_entity}'. Keeping abstain."
                    )

            # --- Negative Hallucination Guard ---
            # DISABLED: Guard quá strict, chặn cả câu trả lời đúng khi entity name
            # không match exact với graph node. Anti-apology guard (Rule #9) trong
            # prompt đã đủ ngăn hallucination — LLM tự nói "chưa có thông tin"
            # khi context không có dữ liệu.

        # --- Closed-Form Answer Format Enforcement (safety net) ---
        if question_type:
            answer = self._enforce_closed_form_answer_format(answer, question_type, state.user_query)

        if self._is_service_availability_query(state.user_query) and self._is_apology_answer(answer):
            answer = self._build_missing_data_answer(state)
            state.runtime.metadata["service_missing_data_answer"] = True

        logger.info(
            "   -> Generation Output: "
            f"answer_chars={len(answer or '')}, answer_lines={len((answer or '').splitlines())}"
        )
        logger.info("   -> Step 5 completed in %s", self._elapsed(step_5_start))
        state.answer = self._sanitize_answer_text(answer)

        # Phase 5: Answer Contract & Validation
        if getattr(state, "query_plan", None) is not None:
            try:
                from graph_rag.core.answer_contract import AnswerContract, AnswerValidator

                context_validation_ok = (state.runtime.metadata.get("context_validation") or {}).get("ok", True)
                contract = AnswerContract.from_query_plan(
                    query_plan=state.query_plan,
                    clean_context=state.clean_context or "",
                    entities=state.entities,
                    context_validation_ok=context_validation_ok,
                    seed_nodes=state.all_seeds,
                )

                validator = AnswerValidator()
                val_result = validator.validate(state.answer, contract)

                # Fast-path: skip LLM retry for mass_ungrounded on discovery/list queries.
                # These queries have grounded candidates — retry with LLM just hallucinates more.
                has_mass_ungrounded = any(i.code == "mass_ungrounded_entities" for i in val_result.issues)
                is_list_mode = answer_mode in {
                    getattr(AnswerMode, "DISCOVERY_LIST", "discovery_list"),
                    getattr(AnswerMode, "TOUR_LIST", "tour_list"),
                }
                if has_mass_ungrounded and is_list_mode and not state.metadata.get("answer_validation_retry_triggered"):
                    # For DISCOVERY_LIST, use the richer deterministic renderer
                    # that parses context facts by relation type, instead of
                    # the generic fallback that only lists bare entity names.
                    is_discovery = answer_mode == getattr(AnswerMode, "DISCOVERY_LIST", "discovery_list")
                    if is_discovery:
                        fallback_ans = self._dispatch_discovery_list(state, full_generator_candidates)
                    if not is_discovery or not fallback_ans:
                        fallback_ans = self._build_grounded_fallback(state, full_generator_candidates)
                    if fallback_ans:
                        state.answer = fallback_ans
                        state.runtime.metadata["answer_validation_fallback_triggered"] = True
                        state.runtime.metadata["fallback_reason"] = ["mass_ungrounded_entities"]
                        state.runtime.metadata["answer_validation_retry_triggered"] = True
                        logger.warning("   -> [Phase5] Fast-path: mass_ungrounded on list mode -> grounded fallback (no LLM retry).")

                # Retry loop if validation fails
                if not val_result.passed and not state.metadata.get("answer_validation_retry_triggered"):
                    logger.error("   -> [Phase5] Answer validation failed. Triggering apology/grounding retry loop...")
                    feedback_msgs = [f"- {issue.message}" for issue in val_result.issues]
                    feedback_str = "\n".join(feedback_msgs)

                    state.runtime.metadata["answer_validation_retry_triggered"] = True
                    state.runtime.metadata["answer_validation_first_attempt"] = state.answer
                    state.runtime.metadata["answer_validation_first_issues"] = [i.code for i in val_result.issues]

                    # Prepare anchored context again
                    anchored_context = state.clean_context
                    grounded_attractions = [
                        n for n in (state.all_seeds or [])
                        if "TouristAttraction" in (n.metadata.get("labels") or [])
                    ]
                    is_nearby_query = any(
                        token in normalize_text(state.user_query, strip_punct=True)
                        for token in ["gan", "gần", "near", "o gan", "ở gần", "nam gan", "nằm gần"]
                    )
                    if grounded_attractions and is_nearby_query and IntentType.from_value(plan.intent) in {IntentType.FOOD, IntentType.ACCOMMODATION}:
                        anchor_name = grounded_attractions[0].metadata.get("name") or grounded_attractions[0].content
                        anchor_ctx = f"**ANCHOR POINT:** Người dùng đang hỏi về những địa điểm gần {anchor_name}. Khi trả lời, hãy LUÔN refer về địa điểm này bằng đúng tên: '{anchor_name}'.\n\n"
                        anchored_context = anchor_ctx + state.clean_context

                    # Re-generate answer with retry feedback
                    retry_answer = p.generator.generate(
                        user_query=state.user_query,
                        context_text=anchored_context,
                        intent=plan.intent,
                        detected_location=state.location,
                        candidate_nodes=full_generator_candidates,
                        strict_route_nodes=state.runtime.metadata.get("route_seed_nodes") if IntentType.from_value(plan.intent) == IntentType.TOUR_PLAN else None,
                        dropped_route_points=state.runtime.metadata.get("dropped_route_points") if IntentType.from_value(plan.intent) == IntentType.TOUR_PLAN else None,
                        daily_cluster_plan=state.runtime.metadata.get("daily_cluster_plan") if IntentType.from_value(plan.intent) == IntentType.TOUR_PLAN else None,
                        route_optimizer_metrics=state.runtime.metadata.get("route_optimizer_metrics") if IntentType.from_value(plan.intent) == IntentType.TOUR_PLAN else None,
                        context_validation=state.runtime.metadata.get("context_validation")
                        if (state.runtime.metadata.get("context_validation") or {}).get("ok", True)
                        else None,
                        on_token=None,
                        query_state=state.query_plan,
                        validation_feedback=feedback_str,
                        answer_mode=answer_mode,
                    )

                    state.answer = self._sanitize_answer_text(retry_answer)
                    val_result = validator.validate(state.answer, contract)
                    logger.info("   -> [Phase5] Answer validation after retry complete. passed=%s", val_result.passed)

                issues_list = []
                for issue in val_result.issues:
                    issues_list.append({
                        "code": issue.code,
                        "severity": issue.severity,
                        "message": issue.message
                    })

                state.runtime.metadata["answer_validation"] = {
                    "passed": val_result.passed,
                    "issues": issues_list,
                    "contract": {
                        "question_shape": contract.question_shape.value,
                        "target_class": contract.target_class,
                        "requested_attributes": contract.requested_attributes,
                        "comparison_subjects": contract.comparison_subjects,
                        "context_entity_names": contract.context_entity_names,
                        "context_has_rating_evidence": contract.context_has_rating_evidence,
                        "context_has_review_evidence": contract.context_has_review_evidence,
                        "context_sufficient": contract.context_sufficient,
                        "unsupported_attributes": contract.unsupported_attributes,
                        "allow_apology": contract.allow_apology
                    }
                }

                logger.info("   -> [Phase5] Answer validation complete. passed=%s, issues=%s", val_result.passed, len(issues_list))
                for issue in val_result.issues:
                    logger.info("      [%s] %s: %s", issue.severity.upper(), issue.code, issue.message)

                # Grounded fallback for fatal validation issues
                fatal_issue_codes = {
                    "mass_ungrounded_entities",
                    "rating_hallucination",
                    "review_hallucination",
                }
                has_fatal_issue = not val_result.passed and any(issue.code in fatal_issue_codes for issue in val_result.issues)
                if has_fatal_issue:
                    logger.warning("   -> [Phase5] Fatal validation issue detected. Substituting response with grounded fallback.")
                    original_answer = state.answer
                    fallback_reason = [i.code for i in val_result.issues if i.code in fatal_issue_codes]
                    fallback_ans = self._build_grounded_fallback(state, full_generator_candidates)
                    if fallback_ans:
                        state.answer = fallback_ans
                        state.runtime.metadata["answer_validation_fallback_triggered"] = True
                        state.runtime.metadata["fallback_reason"] = fallback_reason
                        state.runtime.metadata["original_answer_before_fallback"] = original_answer
                        logger.warning("   -> [Phase5] Grounded fallback successfully applied (%s chars).", len(fallback_ans))
            except (ValueError, RuntimeError, OSError) as e:
                import traceback
                logger.error("   -> [Phase5] Error during answer validation: %s", e)
                logger.error("   -> [Phase5] Traceback: %s", traceback.format_exc())
