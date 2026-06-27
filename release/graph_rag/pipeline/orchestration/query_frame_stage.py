from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


import re


from typing import Any, Dict, List



from graph_rag.core.intents import IntentType


from graph_rag.modules.query_frame import QueryFrameExtractor, QueryFrameValidator


from graph_rag.pipeline.orchestration.query_frame_contract import QueryFrame as QueryFrameContract


class QueryFrameStage:
    """Guarded QueryFrame integration for Phase 5.

    This stage patches metadata only when a validated frame is available. If the
    frame is invalid, it records debug data and leaves the old pipeline intact.
    """

    def __init__(self, normalizer=None, min_confidence: float = 0.6):
        self.extractor = QueryFrameExtractor(normalizer=normalizer)
        self.validator = QueryFrameValidator(min_confidence=min_confidence)
        self._normalize = normalizer or (lambda value: str(value or "").strip().lower())

    def build_and_apply(
        self,
        *,
        query: str,
        metadata: Dict[str, Any],
        entities: List[Dict[str, Any]],
        primary_intent: str,
        role_aware_grounding: bool = False,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        frame = self.validator.validate(
            self.extractor.extract(query=query, metadata=metadata, primary_intent=primary_intent)
        )
        frame_dict = frame.to_dict()
        metadata["query_frame"] = frame_dict
        metadata["query_frame_valid"] = frame.valid

        if not frame.valid:
            metadata["query_frame_fallback_reason"] = frame.fallback_reason
            return metadata, entities, frame_dict

        if not role_aware_grounding:
            metadata["query_frame_applied"] = False
            metadata["query_frame_apply_reason"] = "role_aware_grounding_disabled"
            return metadata, entities, frame_dict

        patched_entities = self._patch_entities(entities, frame)
        target_before = str(metadata.get("target_entity") or "").strip()
        target_after = self._select_role_target(frame, target_before)

        if target_after:
            metadata["target_entity"] = target_after
            metadata["query_frame_applied"] = True
            metadata["query_frame_target_before"] = target_before
            metadata["query_frame_target_after"] = target_after
        else:
            metadata["query_frame_applied"] = bool(patched_entities)
            metadata["query_frame_apply_reason"] = "role_entities_without_single_target" if patched_entities else "no_safe_role_target"

        plan = frame.retrieval_plan
        metadata["retrieval_plan_mode"] = plan.mode

        # ALWAYS set anchor names from contract groundable_mentions
        # This ensures seed retriever uses contract entities first, before falling back to fulltext search
        if not metadata.get("query_frame_anchor_names"):
            all_mentions = list(frame.groundable_mentions or []) + list(plan.anchors or [])
            anchor_names = self._anchor_names(all_mentions)
            if anchor_names:
                metadata["query_frame_anchor_names"] = anchor_names
                logger.info("   -> [QueryFrame] Set anchor names from contract: %s", anchor_names)

        if plan.mode in {"comparison", "multi_candidate"}:
            metadata["query_frame_anchor_names"] = self._anchor_names(
                plan.candidate_entities if plan.mode == "multi_candidate" else plan.anchors
            )
            metadata["query_frame_multi_anchor_mode"] = True
            metadata["target_entity"] = ""
            metadata["query_frame_target_after"] = ""
            metadata["query_frame_target_policy"] = "multi_anchor_no_single_target"
            metadata["query_frame_applied"] = True
            if plan.mode == "comparison":
                comparison_rels = list(dict.fromkeys(plan.required_relations or []))
                if not comparison_rels:
                    comparison_rels = ["NEAR", "LOCATED_IN", "BELONGS_TO", "HAS", "OFFERS", "INCLUDES"]
                metadata["query_frame_traversal_relations"] = comparison_rels
                metadata["query_frame_target_policy"] = "comparison_multi_anchor"
            if plan.mode == "multi_candidate" and "HAS" in plan.required_relations:
                metadata["query_frame_traversal_intent"] = IntentType.FOOD
                metadata["query_frame_traversal_relations"] = list(dict.fromkeys(plan.required_relations))
                metadata["query_frame_target_policy"] = "multi_candidate_with_dish_constraint"
        elif plan.mode == "global_discovery":
            metadata["query_frame_global_discovery"] = True
            metadata["target_entity"] = ""
            metadata["query_frame_target_after"] = ""
            metadata["query_frame_target_policy"] = "global_no_single_target"
            q_norm = self._normalize(query)
            prior_intent = str(metadata.get("intent") or primary_intent or "").upper()
            # Event signals: take priority over food to prevent "lễ hội" queries
            # from being misrouted to FOOD_RECOMMENDATION
            event_signals = {
                "le hoi", "su kien", "festival", "dien ra", "van hoa",
            }
            has_event_signal = any(signal in q_norm for signal in event_signals)
            food_signals = {
                "am thuc", "dac san", "mon ngon", "hai san",
                "nha hang", "quan an", "nen an", "an gi",
                "an gi ngon", "mon an", "do an",
            }
            # Shopping/purchase signals: "mua hải sản", "mua đặc sản", "chợ hải sản"
            # Must check BEFORE food signals to prevent misrouting to FOOD_RECOMMENDATION
            shopping_signals = {
                "mua", "mua o dau", "mua tai dau", "mua duoc",
                "ban", "cho hai san", "cho dau", "vua",
            }
            food_context_signals = {
                "hai san", "tom", "cua", "ca", "muc", "oc",
                "dac san", "mon ngon", "do an", "thuc pham",
            }
            has_shopping_signal = (
                any(signal in q_norm for signal in shopping_signals)
                and any(signal in q_norm for signal in food_context_signals)
            )
            if has_event_signal:
                metadata["intent"] = IntentType.EVENT
                metadata["query_frame_traversal_intent"] = IntentType.EVENT
                metadata["retrieval_allowed_labels"] = ["Event", "TravelInfo"]
                metadata["forbidden_labels"] = ["Dish", "Restaurant", "Accommodation"]
            elif has_shopping_signal:
                # Shopping query: "mua hải sản ở đâu", "mua đặc sản Quy Nhơn"
                # Route to TRAVEL_ADVICE to retrieve TravelInfo (topic=shopping)
                metadata["intent"] = IntentType.TRAVEL_ADVICE
                metadata["query_frame_traversal_intent"] = IntentType.TRAVEL_ADVICE
                metadata["retrieval_allowed_labels"] = ["TravelInfo", "Restaurant", "Location", "TouristAttraction"]
                metadata["fallback_policy"] = "seafood_shopping_guided_fallback"
            elif ("FOOD" in prior_intent or any(signal in q_norm for signal in food_signals)) and "ACCOMMODATION" not in prior_intent:
                metadata["intent"] = IntentType.FOOD
                metadata["query_frame_traversal_intent"] = IntentType.FOOD
                metadata["retrieval_allowed_labels"] = ["Restaurant", "Dish", "Location"]
            else:
                metadata["intent"] = IntentType.DISCOVERY
                metadata["retrieval_allowed_labels"] = ["TouristAttraction"]
            metadata["query_frame_applied"] = True
        elif plan.mode == "dish_to_restaurant":
            metadata["query_frame_anchor_names"] = self._anchor_names(plan.anchors or [])
            metadata["intent"] = IntentType.FOOD
            metadata["query_frame_traversal_intent"] = IntentType.FOOD
            metadata["query_frame_traversal_relations"] = list(dict.fromkeys(plan.required_relations or ["HAS"]))
            metadata["query_frame_target_policy"] = "dish_anchor_to_restaurant"
            metadata["retrieval_allowed_labels"] = ["Dish", "Restaurant"]
            metadata["query_frame_applied"] = True
        elif plan.mode == "lodging_near_anchor":
            metadata["query_frame_anchor_names"] = self._anchor_names(plan.anchors or [])
            metadata["query_frame_multi_anchor_mode"] = True
            metadata["intent"] = IntentType.ACCOMMODATION
            metadata["query_frame_traversal_intent"] = IntentType.ACCOMMODATION
            metadata["query_frame_traversal_relations"] = list(dict.fromkeys(plan.required_relations or ["NEAR", "LOCATED_IN"]))
            metadata["query_frame_target_policy"] = "lodging_near_proximity_anchor"
            metadata["retrieval_allowed_labels"] = ["TouristAttraction", "Accommodation"]
            metadata["query_frame_applied"] = True
        elif plan.mode == "class_search" and (plan.context_policy or {}).get("target_class") == "Tour":
            metadata["target_entity"] = ""
            metadata["target_class"] = "Tour"
            metadata["answer_mode"] = "tour_list"
            metadata["retrieval_allowed_labels"] = ["Tour"]
            metadata["query_frame_target_after"] = ""
            metadata["query_frame_target_policy"] = "tour_availability_class_search"
            metadata["query_frame_applied"] = True
        elif frame_dict.get("query_operator") == "ticket_price":
            # Ticket price query: "Giá vé vào Eo Gió và đồi cát Phương Mai là bao nhiêu?"
            # Force correct contract for price queries
            metadata["intent"] = IntentType.ENTITY_FACT
            metadata["query_frame_traversal_intent"] = IntentType.ENTITY_FACT
            metadata["retrieval_allowed_labels"] = ["TouristAttraction", "TravelInfo"]
            metadata["requested_attributes"] = ["ticket_price", "price_note", "description"]
            metadata["forbidden_labels"] = ["Dish", "Restaurant", "Accommodation", "TravelAgency"]
            metadata["query_frame_target_policy"] = "ticket_price_lookup"
            metadata["query_frame_applied"] = True
            # Clean anchors: remove "Giá vé vào" prefix
            cleaned_anchors = []
            for anchor in (metadata.get("query_frame_anchor_names") or []):
                cleaned = self._clean_price_anchor(anchor)
                if cleaned:
                    cleaned_anchors.append(cleaned)
            if cleaned_anchors:
                metadata["query_frame_anchor_names"] = cleaned_anchors
            logger.info("   -> Ticket price contract: labels=%s, anchors=%s", metadata['retrieval_allowed_labels'], cleaned_anchors)
        elif plan.mode == "constrained_nearby_search":
            # Multi-hop chain reasoning — short-circuit to GraphReasoningExecutor
            metadata["target_entity"] = ""
            metadata["target_class"] = (plan.context_policy or {}).get("answer_set_label", "Accommodation")
            metadata["retrieval_plan_mode"] = "constrained_nearby_search"
            metadata["query_frame_target_policy"] = "chain_reasoning"
            metadata["query_frame_chain"] = (plan.context_policy or {}).get("chain", [])
            metadata["query_frame_chain_location_scope"] = (plan.context_policy or {}).get("location_scope", "")
            metadata["query_frame_applied"] = True
        elif plan.mode == "tour_plan":
            metadata["query_frame_anchor_names"] = self._anchor_names(
                plan.anchors or [],
                preferred_roles=["origin_accommodation", "accommodation", "tour", "anchor_entity"],
            )
            metadata["query_frame_multi_anchor_mode"] = True
            anchor_roles = [
                str(item.role)
                for item in frame.groundable_mentions
                if getattr(item, "role", "")
            ]
            if any(role == "origin_accommodation" for role in anchor_roles):
                metadata["query_frame_traversal_intent"] = IntentType.ACCOMMODATION
                metadata["query_frame_traversal_relations"] = ["NEAR", "LOCATED_IN"]
                metadata["query_frame_target_policy"] = "tour_plan_origin_accommodation"
        else:
            metadata["query_frame_anchor_names"] = self._anchor_names(
                ([target_after] if target_after else []) + list(plan.anchors or [])
            )
        if frame.location_scope and not metadata.get("geo_anchor_location"):
            metadata["geo_anchor_location"] = frame.location_scope
        return metadata, patched_entities, frame_dict

    def build_query_frame_contract(
        self,
        metadata: Dict[str, Any],
        entities: List[Dict[str, Any]],
        primary_intent: str,
    ) -> QueryFrameContract:
        """Build QueryFrame contract from metadata (semantic contract).

        This should be called AFTER build_and_apply() to sync the contract.
        """
        contract = QueryFrameContract()

        # Intent
        contract.intent = str(metadata.get("intent") or primary_intent or "UNKNOWN").upper()
        contract.original_intent = str(metadata.get("original_intent") or contract.intent).upper()
        contract.operator = str(
            metadata.get("query_frame", {}).get("query_operator") or "default"
        )
        contract.answer_mode = str(metadata.get("answer_mode") or "fact_answer")

        # Geographic Scope
        contract.geo_scope = str(metadata.get("region_focus") or "all")
        contract.province = str(metadata.get("current_province") or "")
        contract.legacy_province = metadata.get("legacy_province") or None
        contract.region = metadata.get("display_region") or None
        contract.region_focus = str(metadata.get("region_focus") or "all")
        contract.region_group = metadata.get("region_group") or None

        # Retrieval Targets
        contract.target_labels = list(metadata.get("retrieval_allowed_labels") or [])
        contract.required_evidence = list(metadata.get("requested_attributes") or [])
        contract.forbidden_fallbacks = list(metadata.get("forbidden_fallbacks") or [])

        # Grounding - extract from entities
        for ent in entities or []:
            if isinstance(ent, dict):
                name = str(ent.get("name") or "").strip()
                if name:
                    contract.anchors.append(name)
                    contract.anchor_types[name] = str(ent.get("type") or "Unknown")

        # Constraints
        contract.constraints = dict(metadata.get("constraints") or {})
        contract.max_results = int(metadata.get("max_results") or 10)

        # Confidence
        contract.confidence = float(metadata.get("confidence") or 0.0)
        contract.source = str(metadata.get("source") or "")

        # Trace
        contract.add_trace(f"Created from metadata by QueryFrameStage")
        contract.add_trace(f"Intent: {contract.intent}, Geo: {contract.geo_scope}")

        return contract

    def _patch_entities(self, entities: List[Dict[str, Any]], frame) -> List[Dict[str, Any]]:
        plan_mode = getattr(getattr(frame, "retrieval_plan", None), "mode", "")
        by_norm: Dict[str, Dict[str, Any]] = {}
        if plan_mode not in {"comparison", "multi_candidate", "global_discovery"}:
            for entity in entities or []:
                if not isinstance(entity, dict):
                    continue
                name = str(entity.get("name") or "").strip()
                if not name or self._is_non_groundable_target(name, frame.non_groundable_phrases):
                    continue
                by_norm[self._normalize(name)] = dict(entity)

        if plan_mode == "comparison":
            role_mentions = list(frame.comparison_subjects)
        elif plan_mode == "multi_candidate":
            role_mentions = list(frame.candidate_entities)
        elif plan_mode == "dish_to_restaurant":
            role_mentions = list(getattr(frame.retrieval_plan, "anchors", []) or [])
        elif plan_mode == "lodging_near_anchor":
            role_mentions = list(getattr(frame.retrieval_plan, "anchors", []) or [])
        elif plan_mode == "global_discovery":
            role_mentions = []
        else:
            role_mentions = list(frame.groundable_mentions)
        for mention in role_mentions:
            if mention.groundability != "groundable":
                continue
            name = str(mention.text or "").strip()
            if not name:
                continue
            if self._target_looks_like_query_phrase(name):
                continue
            key = self._normalize(name)
            by_norm[key] = {
                "name": name,
                "type": mention.type_hint or "Place",
                "role": mention.role,
                "query_frame_confidence": mention.confidence,
            }
        return list(by_norm.values())

    def _select_role_target(self, frame, current_target: str) -> str:
        plan_mode = getattr(getattr(frame, "retrieval_plan", None), "mode", "")
        if plan_mode in {"comparison", "multi_candidate", "global_discovery"}:
            return ""
        if plan_mode == "dish_to_restaurant":
            return self._first_mention_text(list(getattr(frame.retrieval_plan, "anchors", []) or []), ["dish"])
        if plan_mode == "lodging_near_anchor":
            return self._first_mention_text(list(getattr(frame.retrieval_plan, "anchors", []) or []), ["proximity_anchor", "anchor_entity"])
        mentions = (
            list(frame.groundable_mentions)
            + list(frame.comparison_subjects)
            + list(frame.candidate_entities)
        )
        requested_rels = set(getattr(frame, "requested_relations", []) or [])
        if plan_mode != "tour_plan" and "INCLUDES" in requested_rels:
            target = self._first_mention_text(mentions, ["tour"])
            if target:
                return target
        if plan_mode == "tour_plan":
            target = self._first_mention_text(mentions, ["origin_accommodation", "accommodation"])
            if target:
                return target
            specific_mentions = [
                str(getattr(mention, "text", "") or "").strip()
                for mention in mentions
                if str(getattr(mention, "text", "") or "").strip()
                and not self._target_looks_like_query_phrase(str(getattr(mention, "text", "") or ""))
            ]
            if len(specific_mentions) >= 2:
                return ""
            target = self._first_specific_mention_text(mentions)
            if target:
                return target

        best_mention = self._first_specific_mention_text(mentions)
        if (
            best_mention
            and (
                not current_target
                or self._is_non_groundable_target(current_target, frame.non_groundable_phrases)
                or self._target_looks_like_query_phrase(current_target)
                or self._normalize(best_mention) in self._normalize(current_target)
            )
        ):
            return best_mention

        if current_target and not self._is_non_groundable_target(current_target, frame.non_groundable_phrases):
            return current_target

        preferred_roles = [
            "origin_accommodation",
            "comparison_subject",
            "choice_candidate",
            "anchor_entity",
            "restaurant",
            "accommodation",
        ]
        for role in preferred_roles:
            for mention in mentions:
                if mention.role == role and mention.text:
                    return mention.text
        return ""

    def _anchor_names(self, mentions: List[Any], preferred_roles: List[str] | None = None) -> List[str]:
        if preferred_roles:
            role_rank = {role: idx for idx, role in enumerate(preferred_roles)}
            mentions = sorted(
                list(mentions or []),
                key=lambda item: role_rank.get(str(getattr(item, "role", "") or ""), len(role_rank)),
            )
        names: List[str] = []
        for mention in mentions or []:
            text = str(getattr(mention, "text", mention) or "").strip()
            text = self._clean_anchor_name(text)
            if not text:
                continue
            if self._target_looks_like_query_phrase(text):
                continue
            type_hint = self._normalize(getattr(mention, "type_hint", "") or "")
            if type_hint in {"duration", "time", "number", "count"}:
                continue
            norm = self._normalize(text)
            if any(existing == norm for existing in [self._normalize(name) for name in names]):
                continue
            names.append(text)
        return names

    def _clean_anchor_name(self, text: str) -> str:
        value = str(text or "").strip(" ,.;:!?")
        if not value:
            return ""
        value = value.split(":", 1)[0].strip(" ,.;:!?")
        stop_patterns = [
            r"\s+(?:chúng|chung)\s+(?:cùng|cung)\b.*$",
            r"\s+(?:dựa trên|dua tren|theo|nếu|neu|về|ve)\b.*$",
            r"\s+(?:cơ sở|co so|địa điểm|dia diem)\s+nào\b.*$",
        ]
        for pattern in stop_patterns:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip(" ,.;:!?")
        return value

    def _clean_price_anchor(self, text: str) -> str:
        """Clean anchor name for price queries.

        Remove price-related prefixes like "Giá vé vào", "Vé vào cổng", etc.
        "Giá vé vào Eo Gió" -> "Eo Gió"
        "đồi cát Phương Mai hiện nay" -> "Đồi cát Phương Mai"
        """
        value = str(text or "").strip()
        if not value:
            return ""

        # Remove price-related prefixes
        price_prefixes = [
            r"^giá\s+vé\s+(?:vào|của)\s+",
            r"^vé\s+vào\s+cổng\s+",
            r"^phí\s+tham quan\s+",
            r"^giá\s+tham quan\s+",
            r"^vé\s+vào\s+",
            r"^giá\s+vé\s+",
        ]
        for pattern in price_prefixes:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

        # Remove time-related suffixes
        time_suffixes = [
            r"\s+(?:hiện\s+nay|hiện\s+tại|bây\s+giờ|năm\s+\d{4})\s*$",
            r"\s+(?:là\s+bao\s+nhiêu|mất\s+bao\s+nhiêu)\s*$",
        ]
        for pattern in time_suffixes:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip()

        return value.strip(" ,.;:!?")

    def _first_mention_text(self, mentions: List[Any], roles: List[str]) -> str:
        for role in roles:
            for mention in mentions:
                if getattr(mention, "role", "") == role and getattr(mention, "text", ""):
                    text = str(mention.text).strip()
                    if text and not self._target_looks_like_query_phrase(text):
                        return text
        return ""

    def _first_specific_mention_text(self, mentions: List[Any]) -> str:
        for mention in mentions:
            text = str(getattr(mention, "text", "") or "").strip()
            if not text:
                continue
            if self._target_looks_like_query_phrase(text):
                continue
            norm = self._normalize(text)
            type_hint = self._normalize(getattr(mention, "type_hint", "") or "")
            role = self._normalize(getattr(mention, "role", "") or "")
            is_named = any(
                prefix in norm
                for prefix in [
                    "khach san",
                    "nha nghi",
                    "homestay",
                    "resort",
                    "nha hang",
                    "quan ",
                    "bien ",
                    "thac ",
                    "dap ",
                    "chua ",
                    "bao tang",
                    "lang nghe",
                    "khu du lich",
                    "tour ",
                ]
            )
            if is_named or type_hint in {"accommodation", "restaurant", "touristattraction", "tour", "dish"} or role in {"origin_accommodation", "restaurant", "accommodation", "dish"}:
                return text
        return ""

    def _is_non_groundable_target(self, value: str, phrases: List[str]) -> bool:
        target_norm = self._normalize(value)
        if not target_norm:
            return True
        for phrase in phrases or []:
            phrase_norm = self._normalize(phrase)
            if phrase_norm and (target_norm == phrase_norm or target_norm.startswith(phrase_norm)):
                return True
        variable_terms = [
            "goi y mot lich trinh",
            "goi y lich trinh",
            "mot du khach luu tru",
            "mot du khach muon luu tru",
            "dia diem nao sau day",
            "trong hai dia diem sau",
            "trong hai thuc the sau",
            "nha hang nao trong",
            "quan an nao trong",
            "so sanh cac",
            "neu du khach luu tru",
            "mot du khach dang luu tru",
            "mot du khach luu tru",
            "hay goi y mot diem",
            "goi y mot diem",
            "dua tren du lieu",
            "vi tri dia ly",
            "mat dia ly",
            "quan trong ngay cho du",
            "so dien thoai cua",
            "loai hinh luu tru",
            "resort & spa la",
        ]
        return any(target_norm.startswith(term) for term in variable_terms)

    def _target_looks_like_query_phrase(self, value: str) -> bool:
        target_norm = self._normalize(value)
        if not target_norm:
            return True
        phrase_markers = [
            "so dien thoai cua",
            "thuoc loai hinh",
            "la loai hinh",
            "loai hinh luu tru",
            "co mon",
            "dac trung nao",
            "mon dac trung",
            "dac san gi",
            "thuc don",
            "menu",
            "quán nào",
            "quan nao",
            "dia diem nao",
            "du khach",
            "hay goi y",
            "goi y",
            "dua tren du lieu",
            "vi tri dia ly",
            "mat dia ly",
            "so sanh ",
        ]
        return any(marker in target_norm for marker in phrase_markers)
