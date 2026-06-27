from __future__ import annotations
import logging
import re as _re

logger = logging.getLogger(__name__)


import time


from typing import Any
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable



from graph_rag.config import (
    COMMUNITY_SUMMARY_MIN_SCORE,
    COMMUNITY_SUMMARY_PATH,
    COMMUNITY_SUMMARY_TOP_K,
    CONTEXT_BUILDER_VERSION,
    ENABLE_COMMUNITY_SUMMARY,
    ENABLE_CONTEXT_DEBUG_LOG,
    ENABLE_CONTEXT_ORGANIZER,
    RAW_CONTEXT_DEFAULT_MAX_ITEMS,
    RAW_CONTEXT_MAX_ITEMS_BY_INTENT,
    RELATIONSHIP_MAP,
)


from graph_rag.core.intents import IntentType


from graph_rag.modules.context import CommunitySummaryRetriever, ContextOrganizerV2


from graph_rag.utils.text import normalize_text


from graph_rag.modules.generation.pruner import ContextPruner



from .dto import PipelineRunState


class ContextStage:
    """Context processing stage extracted from PipelineApplicationService.

    This class owns raw-context preparation, Context V2 invocation, and final
    pruning. It deliberately delegates target-specific heuristics back to the
    application service so this refactor preserves behavior.
    """

    def __init__(self, app_service: Any):
        self.app = app_service
        self.pipeline = app_service.pipeline

    @staticmethod
    def _extract_entity_name(fact: str) -> str:
        """Extract entity name from various fact patterns."""
        # Pattern 1: "Thông tin {name}: ..." (legacy)
        m = _re.match(r"^Thông tin\s+(?:của\s+)?(.+?):\s", fact)
        if m:
            return normalize_text(m.group(1), strip_punct=True)
        # Pattern 2: "{name} thuộc loại ..."
        m = _re.match(r"^(.+?)\s+thuộc loại\s", fact)
        if m:
            return normalize_text(m.group(1), strip_punct=True)
        # Pattern 3: "Địa chỉ {name}: ..." (legacy)
        m = _re.match(r"^Địa chỉ\s+(.+?):\s", fact)
        if m:
            return normalize_text(m.group(1), strip_punct=True)
        # Pattern 4: "SĐT {name}: ..." (legacy)
        m = _re.match(r"^SĐT\s+(.+?):\s", fact)
        if m:
            return normalize_text(m.group(1), strip_punct=True)
        # Pattern 5: "{name}: {desc}" — new description format
        m = _re.match(r"^(.+?):\s+", fact)
        if m:
            return normalize_text(m.group(1), strip_punct=True)
        # Pattern 6: "{name} - Địa chỉ/SĐT: ..." — new attribute format
        m = _re.match(r"^(.+?)\s*-\s*(?:Địa chỉ|SĐT|Loại hình)\s*:", fact)
        if m:
            return normalize_text(m.group(1), strip_punct=True)
        # Pattern 7: "{name} {relation} {object}" — edge fact, extract subject
        m = _re.match(r"^(.+?)\s+(?:phục vụ|nằm gần|nằm tại|thuộc loại|có|bao gồm)\s", fact)
        if m:
            return normalize_text(m.group(1), strip_punct=True)
        return ""

    @staticmethod
    def _dedup_context_by_entity(raw_context: list[str]) -> list[str]:
        """Deduplicate context facts by entity name. Keep first (usually richest) occurrence per entity."""
        seen_entities: set[str] = set()
        result: list[str] = []
        for fact in raw_context:
            ent = ContextStage._extract_entity_name(fact)
            if ent:
                if ent in seen_entities:
                    continue
                seen_entities.add(ent)
            result.append(fact)
        return result

    def prepare_raw_context(
        self,
        state: PipelineRunState,
        all_seeds: list[Any],
        raw_context: list[str],
    ) -> tuple[list[str], bool, ContextOrganizerV2 | None]:
        p = self.pipeline
        seed_context = self.seed_attribute_context(all_seeds, state)
        if seed_context:
            # Dedup: skip seed facts whose entity already appears in raw_context
            raw_list = list(raw_context or [])
            seen_entities = set()
            for f in raw_list:
                ent = self._extract_entity_name(f)
                if ent:
                    seen_entities.add(ent)
            filtered_seed = []
            for f in seed_context:
                ent = self._extract_entity_name(f)
                if ent and ent in seen_entities:
                    continue
                filtered_seed.append(f)
            raw_context = filtered_seed + raw_list
        # Dedup entire raw_context by entity name (defense-in-depth)
        raw_context = self._dedup_context_by_entity(raw_context)
        if (state.metadata or {}).get("query_frame_multi_anchor_mode"):
            state.runtime.metadata["target_context_filter_skipped"] = "query_frame_multi_anchor"
        else:
            raw_context = self.app._prioritize_raw_context_for_target(state, raw_context)
            raw_context = self.app._filter_context_for_target(state, raw_context)

        use_context_v2 = ENABLE_CONTEXT_ORGANIZER or CONTEXT_BUILDER_VERSION == "v2"
        v2_organizer = ContextOrganizerV2(normalize_text=lambda t: normalize_text(t, strip_punct=True)) if use_context_v2 else None
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        if v2_organizer is not None:
            has_multiple_anchors = (
                len((state.metadata or {}).get("query_frame_anchor_names") or []) >= 2
                or len(((state.metadata or {}).get("v3_intent_data") or {}).get("anchors") or []) >= 2
            )
            _is_multi_anchor = bool(
                (state.metadata or {}).get("query_frame_multi_anchor_mode")
                or (state.metadata or {}).get("query_frame_global_discovery")
                or intent == IntentType.DISCOVERY
                or (state.metadata or {}).get("retrieval_plan_mode") in {
                    "comparison",
                    "multi_candidate",
                    "tour_plan",
                    "lodging_near_anchor",
                }
                or has_multiple_anchors
            )
            main_entity = v2_organizer.select_main_entity(
                all_seeds,
                state.entities,
                intent,
                query_text=state.search_query or state.user_query,
                is_multi_anchor=_is_multi_anchor,
                retrieval_policy=(state.metadata or {}).get("retrieval_policy"),
                metadata=state.metadata,
                query_state=state.query_plan,
            )
            # Disable hard-keep for multi-anchor queries (comparison, tour_plan, discovery)
            _is_multi_anchor = bool(
                (state.metadata or {}).get("query_frame_multi_anchor_mode")
                or (state.metadata or {}).get("query_frame_global_discovery")
                or intent == IntentType.DISCOVERY
                or (state.metadata or {}).get("retrieval_plan_mode") in {
                    "comparison",
                    "multi_candidate",
                    "tour_plan",
                    "lodging_near_anchor",
                }
                or has_multiple_anchors
            )
            if _is_multi_anchor and getattr(main_entity, "hard_keep_enabled", False):
                main_entity.hard_keep_enabled = False
                logger.info("   -> Hard-keep disabled for multi-anchor mode (intent=%s)", intent)
            
            # Disable hard-keep for TravelInfo contract mode to prevent traversing off-topic relations
            fallback_policy = (state.metadata or {}).get("fallback_policy") or ""
            is_travel_info_contract = (
                intent in {IntentType.CASHLESS_PAYMENT, IntentType.WEATHER_ADVICE, IntentType.TRANSPORT_INFO, IntentType.TRAVEL_ADVICE, IntentType.EMERGENCY_SUPPORT}
                or (state.metadata or {}).get("intent") in {IntentType.CASHLESS_PAYMENT, IntentType.WEATHER_ADVICE, IntentType.TRANSPORT_INFO, IntentType.TRAVEL_ADVICE, IntentType.EMERGENCY_SUPPORT}
                or fallback_policy.endswith("_guided_fallback")
            )
            if is_travel_info_contract and getattr(main_entity, "hard_keep_enabled", False):
                main_entity.hard_keep_enabled = False
                logger.info("   -> Hard-keep disabled for TravelInfo contract mode (intent=%s)", intent)

            direct_1hop_context = self.direct_1hop_context_for_main_entity(
                main_entity=main_entity,
                primary_intent=intent,
                organizer=v2_organizer,
            )
            if direct_1hop_context:
                raw_context = direct_1hop_context + list(raw_context or [])
                state.runtime.metadata["context_direct_1hop_count"] = len(direct_1hop_context)
                logger.info(
                    "   -> ContextOrganizerV2 prefetch: "
                    f"main_entity='{main_entity.name}', direct_1hop={len(direct_1hop_context)}"
                )

        pre_cap_count = len(raw_context)
        raw_context_cap = RAW_CONTEXT_MAX_ITEMS_BY_INTENT.get(
            intent,
            RAW_CONTEXT_DEFAULT_MAX_ITEMS,
        )
        if not use_context_v2 and pre_cap_count > raw_context_cap:
            raw_context = raw_context[:raw_context_cap]
            logger.info(
                f"   -> Guardrail applied: capped raw facts {pre_cap_count} -> {len(raw_context)} "
                f"for intent '{intent}'."
            )
        else:
            if use_context_v2 and pre_cap_count > raw_context_cap:
                logger.info(
                    f"   -> V2 raw cap deferred to ContextOrganizer: raw_facts={pre_cap_count}, "
                    f"legacy_cap={raw_context_cap}."
                )
            else:
                logger.info("   -> Found %s facts from edges.", pre_cap_count)
        return raw_context, use_context_v2, v2_organizer

    def prune_context(
        self,
        state: PipelineRunState,
        all_seeds: list[Any],
        raw_context: list[str],
        use_context_v2: bool,
        v2_organizer: ContextOrganizerV2 | None,
    ) -> str:
        p = self.pipeline
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        logger.info("\n [STEP 4] PRUNING (MMR)...")
        
        # Rule-based Fact Evidence Reranking
        try:
            from graph_rag.modules.context.fact_evidence_reranker import FactEvidenceReranker
            reranker = FactEvidenceReranker()
            relation_priority = ContextOrganizerV2.RELATION_PRIORITY_BY_INTENT.get(intent, [])
            raw_context = reranker.rerank(
                raw_context=raw_context,
                seeds=all_seeds,
                query_text=state.search_query or state.user_query or "",
                primary_intent=intent,
                relation_priority=relation_priority,
                metadata=state.metadata,
            )
            logger.info("   -> FactEvidenceReranker: sorted %s raw facts.", len(raw_context))
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("   -> FactEvidenceReranker warning: %s", e)


        step_4c_start = time.time()
        query_embedding = state.metadata.get("query_embedding")
        query_embedding_source = "metadata"
        if not query_embedding:
            try:
                query_embedding = p.embedding_service.embed_query(state.search_query)
                query_embedding_source = "recomputed"
            except (ValueError, TypeError, RuntimeError, OSError) as e:
                logger.error("   -> [ContextStage] Embedding computation failed: %s", e)
                query_embedding = None
                query_embedding_source = "unavailable"
        logger.info(
            "   -> Pruning Input: "
            f"raw_facts={len(raw_context)}, embedding_source='{query_embedding_source}'"
        )
 
        context_top_k = self.app._dynamic_context_top_k(state)
        if use_context_v2:
            organizer = v2_organizer or ContextOrganizerV2(normalize_text=lambda t: normalize_text(t, strip_punct=True))
            retrieval_plan_mode = (state.metadata or {}).get("retrieval_plan_mode")
            _is_global_discovery = bool(
                (state.metadata or {}).get("query_frame_global_discovery")
                or (
                    intent == IntentType.DISCOVERY
                    and retrieval_plan_mode not in {
                        "comparison",
                        "multi_candidate",
                        "tour_plan",
                        "lodging_near_anchor",
                        "dish_to_restaurant",
                    }
                )
            )
            organization = organizer.organize(
                raw_context=raw_context,
                seeds=all_seeds,
                entities=state.entities,
                primary_intent=intent,
                query_text=state.search_query,
                max_items=context_top_k,
                query_embedding=query_embedding,
                embedding_service=p.embedding_service,
                is_multi_anchor=bool(
                    (state.metadata or {}).get("query_frame_multi_anchor_mode")
                    or _is_global_discovery
                    or (state.metadata or {}).get("retrieval_plan_mode") in {
                        "comparison",
                        "multi_candidate",
                        "tour_plan",
                        "lodging_near_anchor",
                    }
                ),
                is_global_discovery=_is_global_discovery,
                retrieval_policy=(state.metadata or {}).get("retrieval_policy"),
                metadata=state.metadata,
                query_state=state.query_plan,
            )
            clean_context = organization.final_context
            community_result = self.community_summary_context(state, all_seeds)
            community_context = community_result.render()
            if community_context:
                clean_context = self.append_context_section(clean_context, community_context)
            if ENABLE_CONTEXT_DEBUG_LOG:
                state.runtime.metadata["context_debug"] = organization.debug
                state.runtime.metadata["context_debug"]["community_summary"] = community_result.debug
            if organization.main_entity and getattr(organization.main_entity, "name", None):
                state.runtime.metadata["context_organizer_output"] = {
                    "main_entity": organization.main_entity.name,
                }
            logger.info(
                "   -> ContextOrganizerV2 Output: "
                f"main_entity='{organization.main_entity.name}', "
                f"confidence={organization.main_entity.confidence:.2f}, "
                f"mode={organization.main_entity.query_mode}, "
                f"structural={len(organization.kept_structural_items)}/{len(organization.structural_items)}, "
                f"textual_candidates={len(organization.textual_items)}"
            )
        else:
            clean_context = ContextPruner.prune(
                raw_context,
                max_items=context_top_k,
                query_embedding=query_embedding,
                embedding_service=p.embedding_service,
                query_text=state.search_query,
            )
            if ENABLE_CONTEXT_DEBUG_LOG:
                state.runtime.metadata["context_debug"] = {
                    "context_builder_version": "v1",
                    "raw_context_items": len(raw_context),
                    "context_budget": context_top_k,
                }
 
        clean_context_facts = self.app._count_bulleted_lines(clean_context)
        logger.info(
            "   -> Pruning Output: "
            f"selected_facts={clean_context_facts}, context_chars={len(clean_context or '')}"
        )
        logger.info("   -> Step 4c completed in %s", self.app._elapsed(step_4c_start))
        return clean_context
 
    def community_summary_context(
        self,
        state: PipelineRunState,
        all_seeds: list[Any],
    ):
        retriever = CommunitySummaryRetriever(
            path=COMMUNITY_SUMMARY_PATH,
            enabled=ENABLE_COMMUNITY_SUMMARY,
            top_k=COMMUNITY_SUMMARY_TOP_K,
            min_score=COMMUNITY_SUMMARY_MIN_SCORE,
        )
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        return retriever.retrieve(
            query_text=state.search_query or state.user_query,
            primary_intent=intent,
            seeds=all_seeds,
            metadata=state.metadata,
        )

    def append_context_section(self, clean_context: str, section: str) -> str:
        base = str(clean_context or "").strip()
        addition = str(section or "").strip()
        if not addition:
            return base
        if not base:
            return addition
        return f"{base}\n\n{addition}"

    def direct_1hop_context_for_main_entity(
        self,
        main_entity: Any,
        primary_intent: str,
        organizer: ContextOrganizerV2,
    ) -> list[str]:
        """Fetch bounded direct relation facts for the selected main entity."""
        if not main_entity or not getattr(main_entity, "hard_keep_enabled", False):
            return []

        node_id = str(getattr(main_entity, "node_id", "") or "").strip()
        if not node_id:
            return []

        budget = organizer.structural_budget(
            primary_intent,
            float(getattr(main_entity, "confidence", 0.0) or 0.0),
        )
        if budget <= 0:
            return []

        allowed_rels = list(
            organizer.RELATION_PRIORITY_BY_INTENT.get(primary_intent)
            or RELATIONSHIP_MAP.keys()
        )
        if not allowed_rels:
            return []

        cypher = """
        MATCH (main {id: $node_id})-[r]-(neighbor)
        WHERE type(r) IN $allowed_rels
        RETURN
            coalesce(main.name, main.id) AS subject,
            type(r) AS rel_type,
            coalesce(neighbor.name, neighbor.id) AS object,
            neighbor.address AS object_addr
        LIMIT $limit
        """
        try:
            with self.pipeline.driver.session() as session:
                records = session.run(
                    cypher,
                    node_id=node_id,
                    allowed_rels=allowed_rels,
                    limit=budget,
                )
                facts: list[str] = []
                seen = set()
                for record in records:
                    subject = str(record.get("subject") or "").strip()
                    rel_type = str(record.get("rel_type") or "").strip()
                    obj = str(record.get("object") or "").strip()
                    if not subject or not rel_type or not obj:
                        continue
                    rel_label = RELATIONSHIP_MAP.get(rel_type, rel_type)
                    fact = f"{subject} {rel_label} {obj}"
                    object_addr = str(record.get("object_addr") or "").strip()
                    if object_addr:
                        fact += f" (Dia chi: {object_addr})"
                    if fact not in seen:
                        seen.add(fact)
                        facts.append(fact)
                return facts
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.warning("       Direct 1-hop context warning (non-fatal): %s", exc)
            return []

    def seed_attribute_context(
        self,
        seeds: list[Any],
        state: PipelineRunState | None = None,
    ) -> list[str]:
        lines: list[str] = []
        seen_names: set[str] = set()  # dedup by entity name
        target_norm = ""
        if state is not None:
            is_multi_anchor = bool(
                (state.metadata or {}).get("query_frame_multi_anchor_mode")
                or (state.metadata or {}).get("query_frame_global_discovery")
                or (state.metadata or {}).get("retrieval_plan_mode") in {
                    "comparison",
                    "multi_candidate",
                    "tour_plan",
                    "lodging_near_anchor",
                }
            )
            if not is_multi_anchor:
                target_norm = normalize_text(self.app._primary_specific_entity_name(state), strip_punct=True)
        for seed in seeds or []:
            meta = getattr(seed, "metadata", {}) or {}
            name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
            if not name:
                continue
            # Skip if this entity was already processed
            name_norm = normalize_text(name, strip_punct=True)
            if name_norm in seen_names:
                continue
            seen_names.add(name_norm)
            if target_norm:
                name_norm = normalize_text(name, strip_punct=True)
                if not (target_norm in name_norm or name_norm in target_norm):
                    continue
            labels = meta.get("labels") or []
            label = labels[0] if isinstance(labels, list) and labels else meta.get("type")
            if label:
                lines.append(f"{name} thuộc loại {label}")
            node_type = meta.get("type")
            if node_type and str(node_type) != str(label or ""):
                lines.append(f"Loại hình {name}: {node_type}")
            address = meta.get("address")
            if address:
                lines.append(f"Địa chỉ {name}: {address}")
            phone = meta.get("phone")
            if phone:
                lines.append(f"SĐT {name}: {phone}")
            lat = meta.get("lat")
            lng = meta.get("lng")
            if lat is not None and lng is not None:
                lines.append(f"Tọa độ {name}: WGS84Point({lng}, {lat})")
        return lines
