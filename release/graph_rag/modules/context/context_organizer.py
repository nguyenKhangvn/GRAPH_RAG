from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, Iterable, List, Sequence, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # QueryState removed — use QueryPlan or Any

from graph_rag.config import (
    CROSS_ENCODER_RERANK_TIMEOUT_SEC,
    CROSS_ENCODER_RERANK_TOP_N,
    CROSS_ENCODER_RERANKER_MODEL,
    ENABLE_CROSS_ENCODER_RERANKER,
    ENABLE_HARD_KEEP_1HOP,
    ENABLE_STRUCTURAL_CONTEXT,
    ENABLE_TEXTUAL_MMR,
    RELATIONSHIP_MAP,
)
from graph_rag.config import cfg as _cfg
from graph_rag.config.constants import NON_GROUNDABLE_ENTITY_TYPES
from graph_rag.core.intents import IntentType
from graph_rag.core.state import QuestionShape
from graph_rag.utils.node_utils import get_node_labels, seed_name
from graph_rag.utils.relation_utils import detect_relation_type
from graph_rag.utils.text import token_overlap
from graph_rag.modules.generation.pruner import ContextPruner

from .context_models import ContextItem, ContextOrganizationResult, MainEntitySelection
from .reranker import CrossEncoderTextualReranker

# Load scoring weights from JSON config
_sw = _cfg.scoring_weights()


class ContextOrganizerV2:
    """Organize graph context before prompt rendering.

    The organizer is intentionally conservative: it does not change retrieval
    or traversal. It only protects high-confidence structural facts and lets
    the existing MMR pruner operate on textual evidence.
    """

    # Loaded from scoring_weights.json
    RELATION_PRIORITY_BY_INTENT = _sw.get("relation_priority_by_intent", {})
    STRUCTURAL_BUDGET_BY_INTENT = _sw.get("structural_budget_by_intent", {})
    CONFIDENCE_WEIGHTS = _sw.get("context_organizer", {})

    GLOBAL_QUERY_MARKERS = (
        "co nhung",
        "nhung",
        "nao",
        "danh sach",
        "goi y",
        "de xuat",
        "cac",
        "top",
        "gan bien",
        "o dau",
    )

    ATTRIBUTE_PREFIXES = (
        "dia chi ",
        "sdt ",
        "toa do ",
        "loai hinh ",
        "thong tin ",
    )

    def __init__(self, normalize_text: Callable[[str], str]):
        self.normalize_text = normalize_text
        self._relation_phrase_to_type = {
            self.normalize_text(value): key
            for key, value in (RELATIONSHIP_MAP or {}).items()
            if value
        }

    def organize(
        self,
        raw_context: Sequence[str],
        seeds: Sequence[Any],
        entities: Sequence[Dict[str, Any]],
        primary_intent: str,
        query_text: str,
        max_items: int,
        query_embedding: List[float] | None = None,
        embedding_service: Any = None,
        is_multi_anchor: bool = False,
        is_global_discovery: bool = False,
        retrieval_policy: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        query_state: Optional[Any] = None,
    ) -> ContextOrganizationResult:
        started = time.time()
        unique_context = self._deduplicate(raw_context)
        main_entity = self.select_main_entity(
            seeds,
            entities,
            primary_intent,
            query_text=query_text,
            is_multi_anchor=is_multi_anchor,
            retrieval_policy=retrieval_policy,
            metadata=metadata,
            query_state=query_state,
        )

        structural_items, textual_items = self._split_context(unique_context, main_entity)
        kept_structural = self._select_structural_items(
            structural_items=structural_items,
            main_entity=main_entity,
            primary_intent=primary_intent,
            seeds=seeds,
            is_global_discovery=is_global_discovery,
            query_state=query_state,
        )

        textual_budget = max(0, int(max_items or 0) - len(kept_structural))
        textual_lines = [item.text for item in textual_items]
        reranker_debug = {
            "reranker_enabled": bool(ENABLE_CROSS_ENCODER_RERANKER),
            "reranker_applied": False,
            "reranker_error": "not_run",
        }
        if textual_budget <= 0:
            selected_textual_context = ""
            selected_textual_count = 0
        elif ENABLE_TEXTUAL_MMR:
            selected_textual_context = ContextPruner.prune(
                textual_lines,
                max_items=textual_budget,
                query_embedding=query_embedding,
                embedding_service=embedding_service,
                query_text=query_text,
            )
            selected_textual_lines = self._parse_bullets(selected_textual_context)
            rerank_result = self._rerank_textual_context(query_text, selected_textual_lines)
            reranker_debug = rerank_result.debug
            selected_textual_context = self._render_bullets(rerank_result.texts)
            selected_textual_count = self._count_bullets(selected_textual_context)
        else:
            selected_textual = textual_lines[:textual_budget]
            rerank_result = self._rerank_textual_context(query_text, selected_textual)
            reranker_debug = rerank_result.debug
            selected_textual_context = self._render_bullets(rerank_result.texts)
            selected_textual_count = len(rerank_result.texts)

        final_context = self._render_context(kept_structural, selected_textual_context)

        # 1. Map seed names to labels for category logging
        name_to_labels = {}
        for seed in (seeds or []):
            name = seed_name(seed)
            labels = get_node_labels(seed)
            if name:
                name_to_labels[self.normalize_text(name)] = labels

        selected_category_distribution = {}
        for item in kept_structural:
            item_text_norm = self.normalize_text(item.text)
            matched_labels = set()
            for seed_name_norm, labels in name_to_labels.items():
                if seed_name_norm in item_text_norm:
                    matched_labels.update(labels)
            
            cats = set()
            for lbl in matched_labels:
                for cat in ["Restaurant", "Accommodation", "TouristAttraction", "Event", "Location"]:
                    if lbl.lower() == cat.lower():
                        cats.add(cat)
            if not cats:
                if any(kw in item_text_norm for kw in ["nha hang", "quan an", "mon an", "dac san", "dish"]):
                    cats.add("Restaurant")
                elif any(kw in item_text_norm for kw in ["khach san", "nha nghi", "homestay", "resort", "lodging"]):
                    cats.add("Accommodation")
                elif any(kw in item_text_norm for kw in ["diem choi", "tham quan", "check in", "di dau", "attraction"]):
                    cats.add("TouristAttraction")
                elif any(kw in item_text_norm for kw in ["le hoi", "su kien", "festival", "event"]):
                    cats.add("Event")
                elif "dia chi" in item_text_norm or "location" in item_text_norm:
                    cats.add("Location")
                    
            for cat in (cats or ["Other"]):
                selected_category_distribution[cat] = selected_category_distribution.get(cat, 0) + 1

        dropped_structural = [
            item.text for item in structural_items 
            if item.id not in {k.id for k in kept_structural}
        ]

        shape = query_state.question_shape if query_state else None

        debug = {
            "context_builder_version": "v2",
            "context_mode": main_entity.query_mode,
            "shape": shape.value if shape else "unknown",
            "selected_category_distribution": selected_category_distribution,
            "dropped_facts_count": len(dropped_structural),
            "dropped_facts_summary": dropped_structural[:10],
            "main_entity": main_entity.to_debug_dict(),
            "structural_candidate_count": len(structural_items),
            "structural_kept_count": len(kept_structural),
            "textual_candidate_count": len(textual_items),
            "textual_kept_count": selected_textual_count,
            "final_context_items": len(kept_structural) + selected_textual_count,
            "context_budget": max_items,
            "context_budget_violation": (len(kept_structural) + selected_textual_count) > max_items,
            "kept_structural_facts": [item.to_debug_dict() for item in kept_structural],
            "reranker": reranker_debug,
            "elapsed_ms": round((time.time() - started) * 1000, 2),
        }
        return ContextOrganizationResult(
            final_context=final_context,
            structural_items=structural_items,
            textual_items=textual_items,
            kept_structural_items=kept_structural,
            selected_textual_context=selected_textual_context,
            main_entity=main_entity,
            debug=debug,
        )

    def select_main_entity(
        self,
        seeds: Sequence[Any],
        entities: Sequence[Dict[str, Any]],
        primary_intent: str,
        query_text: str = "",
        is_multi_anchor: bool = False,
        retrieval_policy: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None,
        query_state: Optional[Any] = None,
    ) -> MainEntitySelection:
        candidate_entities = [
            str(entity.get("name") or "").strip()
            for entity in (entities or [])
            if self._is_groundable_entity(entity)
        ]
        seed_list = list(seeds or [])
        if not seed_list:
            return MainEntitySelection(reason="unresolved", query_mode="unresolved")

        best = None
        best_score = -1.0
        best_reason = "fallback_top_seed"
        best_components: Dict[str, float] = {}
        best_lexical_score = 0.0
        query_norm = self.normalize_text(query_text)
        global_query_hint = self._is_global_query(query_norm, candidate_entities, primary_intent)

        if query_state is not None:
            if query_state.question_shape in {
                QuestionShape.LIST,
                QuestionShape.LIST_RANKING,
                QuestionShape.RECOMMENDATION_LIST,
                QuestionShape.DISCOVERY,
                QuestionShape.COMPARISON,
                QuestionShape.ITINERARY
            }:
                is_multi_anchor = True
                global_query_hint = True

        # Multi-anchor queries (comparison, tour_plan with multiple anchors) should never be single_entity
        if is_multi_anchor:
            global_query_hint = True

        for seed_index, seed in enumerate(seed_list):
            sname = seed_name(seed)
            seed_norm = self.normalize_text(sname)
            if not seed_norm:
                continue
            lexical_score = 0.0
            entity_extraction_match = 0.0
            reason = "fallback_top_seed" if seed_index == 0 else "top_seed"

            for entity_name in candidate_entities:
                entity_norm = self.normalize_text(entity_name)
                if not entity_norm:
                    continue
                if seed_norm == entity_norm:
                    lexical_score = 1.0
                    entity_extraction_match = 1.0
                    reason = "exact_alias_match"
                    break
                if entity_norm in seed_norm or seed_norm in entity_norm:
                    lexical_score = max(lexical_score, 0.88)
                    entity_extraction_match = max(entity_extraction_match, 0.90)
                    reason = "near_alias_match"
                    continue
                overlap = token_overlap(entity_norm, seed_norm)
                if overlap >= 0.8:
                    lexical_score = max(lexical_score, 0.78)
                    entity_extraction_match = max(entity_extraction_match, 0.75)
                    reason = "entity_name_overlap"
                elif overlap >= 0.55:
                    lexical_score = max(lexical_score, 0.62)
                    entity_extraction_match = max(entity_extraction_match, 0.55)
                    reason = "entity_name_overlap"

            labels = get_node_labels(seed)

            # Event query intent guard: prevent Restaurant/Dish seeds from being selected as main entity
            if query_state is not None and (query_state.target_class == "Event" or primary_intent == IntentType.EVENT):
                is_food_node = any(lbl.lower() in {"restaurant", "dish"} for lbl in labels)
                if is_food_node:
                    food_keywords = ["ăn", "uống", "nhà hàng", "quán", "ẩm thực", "đặc sản", "món"]
                    has_food_signal = any(kw in query_norm for kw in food_keywords)
                    if not has_food_signal:
                        continue

            # Itinerary guard: prevent Accommodation from becoming main_entity unless
            # the query explicitly mentions lodging. Tour/TouristAttraction is backbone for itineraries.
            if query_state is not None and query_state.question_shape == QuestionShape.ITINERARY:
                is_accommodation = any(lbl.lower() == "accommodation" for lbl in labels)
                if is_accommodation:
                    lodging_keywords = ["khách sạn", "nhà nghỉ", "homestay", "resort", "lưu trú", "ngủ", "khach san", "nha nghi"]
                    has_lodging_signal = any(kw in query_norm for kw in lodging_keywords)
                    if not has_lodging_signal:
                        continue

            if not candidate_entities and query_norm:
                query_overlap = token_overlap(seed_norm, query_norm)
                if query_overlap >= 0.8:
                    lexical_score = max(lexical_score, 0.70)
                    reason = "query_seed_overlap"
                elif query_overlap >= 0.55:
                    lexical_score = max(lexical_score, 0.55)
                    reason = "query_seed_overlap"

            retrieval_rank_score = self._rank_score(seed_index)
            
            # Policy-driven scoring adjustments
            is_blocked = False
            if retrieval_policy:
                primary_labels = retrieval_policy.get("primary_labels") or []
                allowed_labels = retrieval_policy.get("allowed_labels") or []
                blocked_labels = retrieval_policy.get("blocked_labels") or []

                is_allowed = not allowed_labels or any(
                    any(lbl.lower() == alw.lower() for alw in allowed_labels)
                    for lbl in labels
                )
                is_blocked = (not is_allowed) or any(
                    any(lbl.lower() == blk.lower() for blk in blocked_labels)
                    for lbl in labels
                )
                if is_blocked:
                    intent_label_match = 0.0
                    score_factor = 0.1
                else:
                    is_primary = any(
                        any(lbl.lower() == pri.lower() for pri in primary_labels)
                        for lbl in labels
                    )
                    intent_label_match = 1.0 if is_primary else 0.0

                    context_budget = retrieval_policy.get("context_budget") or {}
                    budget_weight = 0.0
                    for lbl in labels:
                        for policy_lbl, val in context_budget.items():
                            if policy_lbl.lower() == lbl.lower():
                                budget_weight = max(budget_weight, val)
                    score_factor = 0.5 + budget_weight
            else:
                intent_label_match = 1.0 if self._labels_match_intent(labels, primary_intent) else 0.0
                score_factor = 1.0

            # Calculate target class match boost
            target_class_match = 0.0
            if (
                query_state is not None
                and query_state.target_class
                and getattr(query_state, "target_class_confidence", 0.0) >= 0.8
            ):
                if any(lbl.lower() == query_state.target_class.lower() for lbl in labels):
                    target_class_match = 1.0

            # Calculate additional features
            query_term_match = 0.0
            if query_norm:
                query_term_match = token_overlap(seed_norm, query_norm)

            semantic_score = float(getattr(seed, "score", 0.0) or 0.0)

            source_type = str(getattr(seed, "source_type", "") or "").lower()
            source_boost = 0.0
            if "semantic" in source_type:
                source_boost = 0.3
            elif "exact" in source_type:
                source_boost = 0.2
            elif "fuzzy" in source_type:
                source_boost = 0.1

            region_match = 0.0
            current_location = (metadata or {}).get("current_location")
            if current_location:
                loc_norm = self.normalize_text(current_location)
                seed_meta = getattr(seed, "metadata", {}) or {}
                seed_text = self.normalize_text(
                    sname + " " + str(seed_meta.get("address") or "") + " " + str(seed_meta.get("description") or "")
                )
                if loc_norm in seed_text:
                    region_match = 1.0

            components = {
                "lexical_score": lexical_score,
                "retrieval_rank_score": retrieval_rank_score,
                "intent_label_match": intent_label_match,
                "entity_extraction_match": entity_extraction_match,
                "query_term_match": query_term_match,
                "semantic_score": semantic_score,
                "source_boost": source_boost,
                "region_match": region_match,
            }
            base_score = sum(
                self.CONFIDENCE_WEIGHTS.get(key, 0.25) * value
                for key, value in {k: v for k, v in components.items() if k in self.CONFIDENCE_WEIGHTS}.items()
            )
            score = (
                base_score
                + 0.2 * query_term_match
                + 0.1 * semantic_score
                + source_boost
                + 0.15 * region_match
                + 0.2 * target_class_match
            )
            score *= score_factor

            # Itinerary backbone preference: boost Tour/TouristAttraction, demote support labels
            if query_state is not None and query_state.question_shape == QuestionShape.ITINERARY:
                backbone_labels = {"tour", "touristattraction"}
                support_labels = {"accommodation", "restaurant", "dish", "specialty"}
                label_set_lower = {lbl.lower() for lbl in labels}
                if label_set_lower & backbone_labels:
                    score += 0.12  # backbone boost
                elif label_set_lower & support_labels:
                    score -= 0.08  # support demotion

            if is_blocked:
                score = min(score, 0.20)
            elif retrieval_policy and intent_label_match == 0.0:
                score = min(score, 0.45)

            if candidate_entities and entity_extraction_match <= 0.0:
                score = min(score, 0.55)
            if not candidate_entities:
                score = min(score, 0.60)
            if lexical_score < 0.50:
                score = min(score, 0.58)

            if score > best_score:
                best = seed
                best_score = min(score, 0.99)
                best_reason = reason
                best_components = components
                best_lexical_score = lexical_score

        if best is None:
            return MainEntitySelection(reason="unresolved", query_mode="unresolved")

        confidence = max(0.0, min(float(best_score), 0.99))
        query_mode = self._query_mode(
            confidence=confidence,
            lexical_score=best_lexical_score,
            has_candidate_entities=bool(candidate_entities),
            global_query_hint=global_query_hint,
            query_state=query_state,
        )
        hard_keep_enabled = bool(
            ENABLE_HARD_KEEP_1HOP
            and query_mode == "single_entity"
            and confidence >= 0.60
        )
        return MainEntitySelection(
            name=seed_name(best),
            node_id=str(getattr(best, "id", "") or ""),
            labels=get_node_labels(best),
            confidence=confidence,
            reason=best_reason,
            hard_keep_enabled=hard_keep_enabled,
            query_mode=query_mode,
            confidence_components=best_components,
        )

    def _split_context(
        self,
        raw_context: Sequence[str],
        main_entity: MainEntitySelection,
    ) -> tuple[List[ContextItem], List[ContextItem]]:
        structural: List[ContextItem] = []
        textual: List[ContextItem] = []
        for idx, line in enumerate(raw_context or []):
            text = str(line or "").strip()
            if not text:
                continue
            relation_type = detect_relation_type(text)
            is_structural = bool(ENABLE_STRUCTURAL_CONTEXT and relation_type)
            item = ContextItem(
                id=f"ctx:{idx}",
                kind="structural" if is_structural else "textual",
                text=text,
                relation_type=relation_type,
                must_keep=False,
                confidence=main_entity.confidence if is_structural else None,
            )
            if is_structural:
                structural.append(item)
            else:
                textual.append(item)
        return structural, textual

    def _select_structural_items(
        self,
        structural_items: Sequence[ContextItem],
        main_entity: MainEntitySelection,
        primary_intent: str,
        seeds: Sequence[Any] = None,
        is_global_discovery: bool = False,
        query_state: Optional[Any] = None,
    ) -> List[ContextItem]:
        if not structural_items:
            return []
        if not ENABLE_STRUCTURAL_CONTEXT:
            return []

        budget = self._structural_budget(primary_intent, main_entity.confidence)
        if is_global_discovery:
            budget = max(budget, 25)
        if budget <= 0:
            return []

        shape = query_state.question_shape if query_state else None
        query_mode = main_entity.query_mode or "unresolved"

        # 1. Map seed names to labels
        name_to_labels = {}
        for seed in (seeds or []):
            name = seed_name(seed)
            labels = get_node_labels(seed)
            if name:
                name_to_labels[self.normalize_text(name)] = labels

        # Metadata-first category helper
        def get_item_categories(item_text: str) -> set[str]:
            item_text_norm = self.normalize_text(item_text)
            matched_labels = set()
            for seed_name_norm, labels in name_to_labels.items():
                if seed_name_norm in item_text_norm:
                    matched_labels.update(labels)
            
            item_cats = set()
            for lbl in matched_labels:
                for cat in ["Restaurant", "Accommodation", "TouristAttraction", "Event", "Location"]:
                    if lbl.lower() == cat.lower():
                        item_cats.add(cat)
            
            if not item_cats:
                # Text fallback checking
                if any(kw in item_text_norm for kw in ["nha hang", "quan an", "mon an", "dac san", "dish"]):
                    item_cats.add("Restaurant")
                if any(kw in item_text_norm for kw in ["khach san", "nha nghi", "homestay", "resort", "lodging"]):
                    item_cats.add("Accommodation")
                if any(kw in item_text_norm for kw in ["diem choi", "tham quan", "check in", "di dau", "attraction"]):
                    item_cats.add("TouristAttraction")
                if any(kw in item_text_norm for kw in ["le hoi", "su kien", "festival", "event"]):
                    item_cats.add("Event")
                if "dia chi" in item_text_norm or "location" in item_text_norm:
                    item_cats.add("Location")
            
            return item_cats

        # 2. Pinned Facts Invariant
        pinned_items: List[ContextItem] = []
        non_pinned_items: List[ContextItem] = []
        
        target_entity = query_state.metadata.get("target_entity") if query_state else None
        target_norm = self.normalize_text(target_entity) if target_entity else ""
        
        requested_attrs = query_state.requested_attributes if query_state else []
        attribute_keywords = {
            "address": ["dia chi", "ở đâu", "vi tri", "toa do", "nam o", "tỉnh", "thành phố"],
            "phone": ["sdt", "dien thoai", "hotline", "lien he"],
            "price": ["gia", "chi phi", "dong/dem", "vnd"],
            "rating": ["rating", "sao", "danh gia", "stars"],
            "reputation": ["noi tieng", "noi bat", "thu hut"],
            # Event attributes
            "name": ["ten", "festival", "le hoi", "su kien"],
            "month": ["thang", "to chuc"],
            "year": ["nam"],
            "activities": ["hoat dong", "trinh dien", "giao luu"],
            "description": ["mo ta", "gioi thieu", "dip", "khong gian"],
        }
        boost_keywords = []
        for attr in requested_attrs:
            boost_keywords.extend(attribute_keywords.get(attr, [attr]))
            
        for item in structural_items:
            item_text_norm = self.normalize_text(item.text)
            is_pinned = False
            
            # Exact match invariant
            if target_norm and target_norm in item_text_norm:
                is_pinned = True
                
            # Requested attribute invariant for single fact
            if shape in {QuestionShape.SINGLE_FACT, QuestionShape.YES_NO} and boost_keywords:
                for kw in boost_keywords:
                    kw_norm = self.normalize_text(kw)
                    if kw_norm and kw_norm in item_text_norm:
                        is_pinned = True
                        break
            
            if is_pinned:
                pinned_item = ContextItem(**item.__dict__)
                pinned_item.must_keep = True
                pinned_item.selection_reason = "pinned_fact_invariant"
                pinned_items.append(pinned_item)
            else:
                non_pinned_items.append(item)

        kept = list(pinned_items)
        if len(kept) >= budget:
            return kept[:budget]

        # 3. Shape Routing on non-pinned items
        if shape == QuestionShape.ITINERARY:
            categories = ["Restaurant", "Accommodation", "TouristAttraction", "Event", "Location"]
            category_groups = {cat: [] for cat in categories}
            category_groups["Other"] = []
            
            for item in non_pinned_items:
                cats = get_item_categories(item.text)
                if cats:
                    for cat in cats:
                        category_groups[cat].append(item)
                else:
                    category_groups["Other"].append(item)
                    
            relation_priority = self.RELATION_PRIORITY_BY_INTENT.get(primary_intent, [])
            item_index = {id(it): idx for idx, it in enumerate(structural_items)}
            for cat in category_groups:
                category_groups[cat].sort(
                    key=lambda it: (self._relation_rank(it.relation_type, relation_priority), item_index.get(id(it), 999))
                )
                
            has_more = True
            active_categories = categories + ["Other"]
            while len(kept) < budget and has_more:
                has_more = False
                for cat in active_categories:
                    if len(kept) >= budget:
                        break
                    if category_groups[cat]:
                        item = category_groups[cat].pop(0)
                        selected = ContextItem(**item.__dict__)
                        selected.selection_reason = "itinerary_category_diversity_round_robin"
                        kept.append(selected)
                        has_more = True
            return kept

        elif shape == QuestionShape.COMPARISON:
            comparison_subjects = []
            if query_state:
                meta = query_state.metadata or {}
                comparison_subjects = meta.get("comparison_subjects_expected") or meta.get("query_frame_anchor_names") or []
            if not comparison_subjects and seeds:
                comparison_subjects = [seed_name(s) for s in seeds if seed_name(s)]
                
            if len(comparison_subjects) >= 2:
                subject_groups = {sub: [] for sub in comparison_subjects}
                unmatched_items = []
                
                for item in non_pinned_items:
                    item_text_norm = self.normalize_text(item.text)
                    matched_any = False
                    for sub in comparison_subjects:
                        sub_norm = self.normalize_text(sub)
                        if sub_norm and sub_norm in item_text_norm:
                            subject_groups[sub].append(item)
                            matched_any = True
                    if not matched_any:
                        unmatched_items.append(item)
                        
                relation_priority = self.RELATION_PRIORITY_BY_INTENT.get(primary_intent, [])
                item_index = {id(it): idx for idx, it in enumerate(structural_items)}
                for sub in comparison_subjects:
                    subject_groups[sub].sort(
                        key=lambda it: (self._relation_rank(it.relation_type, relation_priority), item_index.get(id(it), 999))
                    )
                unmatched_items.sort(
                    key=lambda it: (self._relation_rank(it.relation_type, relation_priority), item_index.get(id(it), 999))
                )
                
                has_more = True
                while len(kept) < budget and has_more:
                    has_more = False
                    for sub in comparison_subjects:
                        if len(kept) >= budget:
                            break
                        if subject_groups[sub]:
                            item = subject_groups[sub].pop(0)
                            selected = ContextItem(**item.__dict__)
                            selected.selection_reason = "comparison_subject_quota_round_robin"
                            kept.append(selected)
                            has_more = True
                            
                leftovers = []
                for sub in comparison_subjects:
                    leftovers.extend(subject_groups[sub])
                leftovers.sort(
                    key=lambda it: (self._relation_rank(it.relation_type, relation_priority), item_index.get(id(it), 999))
                )
                for item in leftovers + unmatched_items:
                    if len(kept) >= budget:
                        break
                    selected = ContextItem(**item.__dict__)
                    selected.selection_reason = "comparison_subject_quota_fill"
                    kept.append(selected)
                return kept

        elif shape in {QuestionShape.SINGLE_FACT, QuestionShape.YES_NO}:
            scored = []
            relation_priority = self.RELATION_PRIORITY_BY_INTENT.get(primary_intent, [])
            target_class = query_state.target_class if query_state else None
            
            for idx, item in enumerate(non_pinned_items):
                text_norm = self.normalize_text(item.text)
                relation_rank = self._relation_rank(item.relation_type, relation_priority)
                
                score = 0
                if item.relation_type in relation_priority:
                    score += 30
                score += max(0, 20 - relation_rank)
                score -= idx * 0.01
                
                # Attribute keyword boost
                attribute_matched = False
                for kw in boost_keywords:
                    kw_norm = self.normalize_text(kw)
                    if kw_norm and kw_norm in text_norm:
                        attribute_matched = True
                        break
                if attribute_matched:
                    # Dynamically calculate boost
                    attribute_match_boost = max(0.15 * score, 0.5)
                    score += attribute_match_boost
                    
                if target_class:
                    target_norm = self.normalize_text(target_class)
                    if target_norm in text_norm:
                        score += 5.0
                        
                scored.append((score, idx, item))
                
            scored.sort(key=lambda row: row[0], reverse=True)
            for score, _idx, item in scored:
                if len(kept) >= budget:
                    break
                selected = ContextItem(**item.__dict__)
                selected.selection_reason = "single_fact_priority_boost"
                kept.append(selected)
            return kept

        elif shape in {QuestionShape.LIST, QuestionShape.LIST_RANKING, QuestionShape.RECOMMENDATION_LIST, QuestionShape.DISCOVERY}:
            scored = []
            relation_priority = self.RELATION_PRIORITY_BY_INTENT.get(primary_intent, [])
            target_class = query_state.target_class if query_state else None
            
            rating_attrs = {"rating", "stars", "reputation"}
            has_rating_request = any(attr in rating_attrs for attr in requested_attrs)
            rating_keywords = ["rating", "sao", "danh gia", "stars", "review", "noi tieng", "yeu thich"]

            for idx, item in enumerate(non_pinned_items):
                text_norm = self.normalize_text(item.text)
                relation_rank = self._relation_rank(item.relation_type, relation_priority)
                
                score = 0
                if item.relation_type in relation_priority:
                    score += 30
                score += max(0, 20 - relation_rank)
                score -= idx * 0.01
                
                # Match target class
                if target_class:
                    target_norm = self.normalize_text(target_class)
                    if target_norm in text_norm:
                        score += 15.0
                        
                # Prioritize rating evidence
                if has_rating_request:
                    rating_matched = any(self.normalize_text(kw) in text_norm for kw in rating_keywords)
                    if rating_matched:
                        score += 10.0
                        
                scored.append((score, idx, item))
                
            scored.sort(key=lambda row: row[0], reverse=True)
            
            # List Diversification: limit facts count per distinct entity name
            entity_counts = {}
            seed_names = []
            for seed in (seeds or []):
                name = None
                if hasattr(seed, "metadata") and isinstance(seed.metadata, dict):
                    name = seed.metadata.get("name")
                if not name and hasattr(seed, "content"):
                    name = seed.content
                if name:
                    seed_names.append(self.normalize_text(str(name)))

            def get_entity_key(text: str) -> str:
                text_n = self.normalize_text(text)
                for s_name in seed_names:
                    if s_name in text_n:
                        return s_name
                # Fallback: extract markdown bold text
                match = re.search(r"\*\*(.*?)\*\*", text)
                if match:
                    return self.normalize_text(match.group(1))
                return text_n[:30]

            # Pass 1: Select items prioritizing diversity (max 2 facts per entity)
            for score, _idx, item in scored:
                if len(kept) >= budget:
                    break
                ent_key = get_entity_key(item.text)
                if entity_counts.get(ent_key, 0) < 2:
                    selected = ContextItem(**item.__dict__)
                    selected.selection_reason = "list_diversified_target"
                    kept.append(selected)
                    entity_counts[ent_key] = entity_counts.get(ent_key, 0) + 1

            # Pass 2: Fill remaining budget with other facts
            for score, _idx, item in scored:
                if len(kept) >= budget:
                    break
                if any(k.text == item.text for k in kept):
                    continue
                selected = ContextItem(**item.__dict__)
                selected.selection_reason = "list_target_class_priority_fallback"
                kept.append(selected)

            return kept

        # Fallback legacy sorting
        scored = []
        main_norm = self.normalize_text(main_entity.name) if main_entity.name else ""
        relation_priority = self.RELATION_PRIORITY_BY_INTENT.get(primary_intent, [])
        
        for idx, item in enumerate(non_pinned_items):
            text_norm = self.normalize_text(item.text)
            mentions_main = bool(main_norm and main_norm in text_norm)
            relation_rank = self._relation_rank(item.relation_type, relation_priority)
            
            score = 0
            if query_mode == "single_entity" and main_entity.hard_keep_enabled and mentions_main:
                score += 100
            elif query_mode == "global_or_multi_seed":
                score += 10
            elif main_entity.confidence < 0.60:
                score -= 50
                
            if item.relation_type in relation_priority:
                score += 30
            score += max(0, 20 - relation_rank)
            score -= idx * 0.01
            scored.append((score, idx, mentions_main, item))
            
        scored.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        for score, _idx, mentions_main, item in scored:
            if len(kept) >= budget:
                break
            if query_mode == "unresolved" and score < 20:
                continue
            if query_mode != "global_or_multi_seed" and main_entity.confidence < 0.60 and score < 20:
                continue
                
            selected = ContextItem(**item.__dict__)
            selected.must_keep = bool(
                query_mode == "single_entity"
                and main_entity.hard_keep_enabled
                and mentions_main
            )
            selected.selection_reason = (
                "hard_keep_main_entity_1hop"
                if selected.must_keep
                else (
                    "global_intent_relevant_relation"
                    if query_mode == "global_or_multi_seed"
                    else "structural_intent_priority"
                )
            )
            kept.append(selected)
        return kept

    def _render_context(self, structural_items: Sequence[ContextItem], textual_context: str) -> str:
        parts: List[str] = []
        if structural_items:
            parts.append("[STRUCTURAL FACTS - MUST KEEP]")
            parts.extend(f"- {self._render_structural_fact(item)}" for item in structural_items)
        if textual_context and textual_context.strip():
            parts.append("[TEXTUAL EVIDENCE]")
            parts.append(textual_context.strip())
        if not parts:
            return "Khong co thong tin ngu canh."
        return "\n".join(parts)

    def _structural_budget(self, primary_intent: str, confidence: float) -> int:
        base = self.STRUCTURAL_BUDGET_BY_INTENT.get(primary_intent, 10)
        if confidence >= 0.85:
            return base
        if confidence >= 0.60:
            return max(3, min(base, 6))
        return min(base, 3)

    def structural_budget(self, primary_intent: str, confidence: float) -> int:
        return self._structural_budget(primary_intent, confidence)

    def _rank_score(self, seed_index: int) -> float:
        if seed_index <= 0:
            return 1.0
        if seed_index == 1:
            return 0.70
        if seed_index == 2:
            return 0.50
        return max(0.10, 0.50 - (seed_index - 2) * 0.08)

    def _is_global_query(
        self,
        query_norm: str,
        candidate_entities: Sequence[str],
        primary_intent: str,
    ) -> bool:
        if not query_norm:
            return False
        # DISCOVERY/TOUR_PLAN intent is always global — entities are anchors, not single main subject
        if primary_intent in {IntentType.DISCOVERY, IntentType.TOUR_PLAN}:
            return True
        if candidate_entities:
            return False
        has_marker = any(marker in query_norm for marker in self.GLOBAL_QUERY_MARKERS)
        return bool(has_marker or primary_intent in {IntentType.DISCOVERY, IntentType.TOUR_PLAN})

    def _query_mode(
        self,
        confidence: float,
        lexical_score: float,
        has_candidate_entities: bool,
        global_query_hint: bool,
        query_state: Optional[Any] = None,
    ) -> str:
        if query_state is not None:
            if query_state.question_shape in {
                QuestionShape.LIST,
                QuestionShape.LIST_RANKING,
                QuestionShape.RECOMMENDATION_LIST,
                QuestionShape.DISCOVERY,
                QuestionShape.COMPARISON,
                QuestionShape.ITINERARY
            }:
                return "global_or_multi_seed"
            if query_state.question_shape in {QuestionShape.SINGLE_FACT, QuestionShape.YES_NO}:
                if confidence >= 0.60 and lexical_score >= 0.50 and not global_query_hint:
                    return "single_entity"
                return "global_or_multi_seed"

        if confidence >= 0.60 and lexical_score >= 0.50 and not global_query_hint:
            return "single_entity"
        if global_query_hint or not has_candidate_entities:
            return "global_or_multi_seed"
        return "unresolved"

    def _render_structural_fact(self, item: ContextItem) -> str:
        text = str(item.text or "").strip()
        relation_type = str(item.relation_type or "").strip()
        if not text or not relation_type:
            return text
        if f"({relation_type})" in text:
            return text
        relation_phrase = RELATIONSHIP_MAP.get(relation_type, "")
        if not relation_phrase:
            return f"{text} ({relation_type})"
        pattern = re.compile(re.escape(str(relation_phrase)), flags=re.IGNORECASE)
        if pattern.search(text):
            return pattern.sub(f"{relation_phrase} ({relation_type})", text, count=1)
        return f"{text} ({relation_type})"

    def _relation_rank(self, relation_type: str | None, priority: Sequence[str]) -> int:
        if not relation_type:
            return 999
        try:
            return list(priority).index(relation_type)
        except ValueError:
            return 100

    def _is_groundable_entity(self, entity: Dict[str, Any]) -> bool:
        if not isinstance(entity, dict):
            return False
        name = str(entity.get("name") or "").strip()
        if not name or name.isdigit():
            return False
        entity_type = str(entity.get("type") or "").strip().lower()
        return entity_type not in NON_GROUNDABLE_ENTITY_TYPES

    def _labels_match_intent(self, labels: Iterable[str], primary_intent: str) -> bool:
        label_set = {str(label or "").lower() for label in labels or []}
        intent = str(primary_intent or "").upper()
        if intent == IntentType.FOOD:
            return bool(label_set & {"restaurant", "dish"})
        if intent == IntentType.ACCOMMODATION:
            return "accommodation" in label_set
        if intent == IntentType.TOURISM:
            return "touristattraction" in label_set
        if intent == IntentType.EVENT:
            return "event" in label_set
        if intent == IntentType.TOUR_PLAN:
            return bool(label_set & {"tour", "touristattraction", "restaurant", "accommodation"})
        return True

    def _deduplicate(self, values: Sequence[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for value in values or []:
            text = str(value or "").strip()
            if not text:
                continue
            # Normalize for near-duplicate detection
            norm = self.normalize_text(text) if self.normalize_text else text.lower()
            if norm in seen:
                continue
            seen.add(norm)
            result.append(text)
        return result

    def _count_bullets(self, text: str) -> int:
        return sum(1 for line in str(text or "").splitlines() if line.strip().startswith("- "))

    def _parse_bullets(self, text: str) -> List[str]:
        lines: List[str] = []
        for raw_line in str(text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("- "):
                line = line[2:].strip()
            if line:
                lines.append(line)
        return lines

    def _render_bullets(self, lines: Sequence[str]) -> str:
        return "\n".join(f"- {str(line).strip()}" for line in lines if str(line).strip())

    def _rerank_textual_context(self, query_text: str, lines: Sequence[str]):
        reranker = CrossEncoderTextualReranker(
            enabled=ENABLE_CROSS_ENCODER_RERANKER,
            model_name=CROSS_ENCODER_RERANKER_MODEL,
            top_n=CROSS_ENCODER_RERANK_TOP_N,
            timeout_sec=CROSS_ENCODER_RERANK_TIMEOUT_SEC,
        )
        return reranker.rerank(query_text=query_text, texts=lines)
