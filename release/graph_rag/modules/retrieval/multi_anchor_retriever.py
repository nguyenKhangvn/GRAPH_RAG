from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


import re
from typing import Any, Dict, List, Optional

from graph_rag.config import RELATIONSHIP_MAP
from graph_rag.core.intents import IntentType
from graph_rag.utils.text import normalize_text


# Category phrases that map to labels for label-based retrieval.
# When an anchor is a category phrase, we search by label instead of by name.
CATEGORY_PHRASE_TO_LABELS: Dict[str, List[str]] = {
    "nhà nghỉ": ["Accommodation"],
    "các nhà nghỉ": ["Accommodation"],
    "nhà nghi": ["Accommodation"],
    "khách sạn": ["Accommodation"],
    "khach san": ["Accommodation"],
    "homestay": ["Accommodation"],
    "resort": ["Accommodation"],
    "lưu trú": ["Accommodation"],
    "luu tru": ["Accommodation"],
    "di tích": ["TouristAttraction"],
    "di tich": ["TouristAttraction"],
    "di tích lịch sử": ["TouristAttraction"],
    "di tich lich su": ["TouristAttraction"],
    "di tích lịch sử - khảo cổ": ["TouristAttraction"],
    "di tich lich su khao co": ["TouristAttraction"],
    "di tích khảo cổ": ["TouristAttraction"],
    "di tich khao co": ["TouristAttraction"],
    "danh lam thắng cảnh": ["TouristAttraction"],
    "danh lam thang canh": ["TouristAttraction"],
    "làng nghề truyền thống": ["TouristAttraction"],
    "lang nghe truyen thong": ["TouristAttraction"],
    "nhà hàng": ["Restaurant"],
    "nha hang": ["Restaurant"],
    "quán ăn": ["Restaurant"],
    "quan an": ["Restaurant"],
    "ẩm thực": ["Restaurant", "Dish"],
    "am thuc": ["Restaurant", "Dish"],
    "món ăn": ["Dish"],
    "mon an": ["Dish"],
}


class MultiAnchorRetriever:
    RELATIONS_BY_INTENT = {
        "comparison": ["NEAR", "LOCATED_IN", "BELONGS_TO"],
        "multi_entity_nearby": ["NEAR", "LOCATED_IN"],
        "constraint_matching": ["NEAR", "LOCATED_IN", "BELONGS_TO", "HAS"],
        "dish_to_restaurant": ["HAS", "LOCATED_IN", "NEAR"],
        "tour_plan": ["INCLUDES", "OFFERS", "NEAR", "LOCATED_IN", "BELONGS_TO"],
        "single_anchor": ["LOCATED_IN", "BELONGS_TO", "NEAR", "HAS", "HELD_AT"],
        "negative": ["LOCATED_IN", "BELONGS_TO", "NEAR", "HAS"],
    }

    INTENT_TO_TRAVERSAL = {
        "comparison": IntentType.DISCOVERY,
        "multi_entity_nearby": IntentType.DISCOVERY,
        "constraint_matching": IntentType.DISCOVERY,
        "dish_to_restaurant": IntentType.FOOD,
        "tour_plan": IntentType.TOUR_PLAN,
        "single_anchor": IntentType.ENTITY_FACT,
        "negative": IntentType.DISCOVERY,
    }

    LABELS_BY_INTENT = {
        "dish_to_restaurant": ["Restaurant", "Dish"],
        "tour_plan": ["TouristAttraction", "Restaurant", "Dish", "Accommodation", "Tour"],
        "single_anchor": None,
        "comparison": None,
        "multi_entity_nearby": None,
        "constraint_matching": None,
        "negative": None,
    }

    def __init__(
        self,
        retriever: Any,
        traverser: Any,
        max_facts_per_anchor: int = 18,
    ):
        self.retriever = retriever
        self.traverser = traverser
        self.max_facts_per_anchor = max(1, int(max_facts_per_anchor or 18))
        self._relation_phrase_map = {
            normalize_text(value): key
            for key, value in (RELATIONSHIP_MAP or {}).items()
            if value
        }

    def retrieve(self, intent_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata = metadata or {}
        anchors = intent_data.get("anchors") or []
        intent_mode = str(intent_data.get("intent_mode") or "single_anchor")
        relations = self.RELATIONS_BY_INTENT.get(intent_mode, [])
        traversal_intent = self.INTENT_TO_TRAVERSAL.get(intent_mode, IntentType.DISCOVERY)

        grouped: Dict[str, Any] = {}
        for anchor in anchors:
            anchor_name = str(anchor or "").strip()
            if not anchor_name:
                continue
            result = self._retrieve_anchor(
                anchor_name,
                intent_mode=intent_mode,
                traversal_intent=traversal_intent,
                relations=relations,
                metadata=metadata,
            )
            grouped[anchor_name] = result
        return grouped

    def _retrieve_anchor(
        self,
        anchor: str,
        intent_mode: str,
        traversal_intent: str,
        relations: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        local_metadata = dict(metadata or {})
        local_metadata["intent"] = traversal_intent
        local_metadata["target_entity"] = anchor
        local_metadata["entities"] = [
            {"name": anchor, "type": self._infer_entity_type(anchor)}
        ]
        label_override = self.LABELS_BY_INTENT.get(intent_mode)
        if label_override is not None:
            local_metadata["retrieval_allowed_labels"] = list(label_override)

        # Clear query-level grounding anchors and query frame anchor names to prevent polluting specific anchor retrieval
        local_metadata.pop("grounded_anchor_nodes", None)
        local_metadata.pop("query_frame_anchor_names", None)

        # HARD RULE: For category phrases, use label-based retrieval instead of name-based.
        # "nhà nghỉ", "di tích lịch sử" are NOT entity names — they are category constraints.
        anchor_norm = normalize_text(anchor)
        category_labels = CATEGORY_PHRASE_TO_LABELS.get(anchor_norm)
        if category_labels and self.retriever._is_category_phrase(anchor):
            logger.info("       Category anchor detected: '%s' → labels=%s", anchor, category_labels)
            # Use hybrid search with label filter instead of name-based search
            seeds = self.retriever._hybrid_search(
                anchor,
                local_metadata,
                top_k=5,
                allowed_labels=category_labels,
                location_filter=None,
            )
        else:
            seeds = self.retriever.find_seeds(anchor, metadata=local_metadata)
            if not seeds:
                seeds = self.retriever.ground_entities(local_metadata["entities"])

        facts: List[str] = []
        if seeds:
            facts = self.traverser.traverse(
                [seeds[0]],
                intent=traversal_intent,
                location_filter="",
                requested_relations=relations,
            )
        
        limit = self.max_facts_per_anchor
        if intent_mode == "tour_plan":
            limit = 60
        elif intent_mode in ("comparison", "constraint_matching", "multi_entity_nearby"):
            limit = 30
        elif intent_mode == "single_anchor":
            limit = 18
        elif intent_mode == "negative":
            limit = 12
            
        facts = list(facts or [])[:limit]

        grouped_relations, attributes = self._group_facts(anchor, facts)
        entity = self._anchor_entity(seed=seeds[0]) if seeds else {"name": anchor}

        return {
            "entity": entity,
            "relations": grouped_relations,
            "attributes": attributes,
            "raw_facts": facts,
            "seed_count": len(seeds),
        }

    def _group_facts(self, anchor: str, facts: List[str]) -> tuple[Dict[str, List[str]], Dict[str, str]]:
        relations: Dict[str, List[str]] = {key: [] for key in RELATIONSHIP_MAP.keys()}
        attributes: Dict[str, str] = {}
        for fact in facts or []:
            text = str(fact or "").strip()
            if not text:
                continue
            attr_match = re.match(r"(?i)^(?:dia chi|\u0111\u1ecba ch\u1ec9|sdt|s\u0111t|thong tin)\s+(.+?):\s*(.+)$", text)
            if attr_match:
                key = normalize_text(attr_match.group(1)).replace(" ", "_")
                attributes[key] = attr_match.group(2).strip()
                continue
            rel_type, obj = self._extract_relation(text)
            if rel_type and obj:
                relations.setdefault(rel_type, []).append(obj)
        # Remove empty relation entries
        relations = {k: v for k, v in relations.items() if v}
        return relations, attributes

    def _extract_relation(self, text: str) -> tuple[str, str]:
        for phrase_norm, rel_type in self._relation_phrase_map.items():
            phrase = None
            for candidate in (RELATIONSHIP_MAP.get(rel_type, ""),):
                if candidate and candidate in text:
                    phrase = candidate
                    break
            if not phrase:
                if phrase_norm in normalize_text(text):
                    phrase = phrase_norm
            if not phrase:
                continue
            parts = re.split(re.escape(phrase), text, maxsplit=1)
            if len(parts) != 2:
                continue
            obj = parts[1].strip()
            if obj:
                obj = obj.split("(", 1)[0].strip()
                return rel_type, obj
        return "", ""

    def _anchor_entity(self, seed: Any) -> Dict[str, Any]:
        metadata = getattr(seed, "metadata", {}) or {}
        return {
            "id": str(getattr(seed, "id", "") or ""),
            "name": str(metadata.get("name") or getattr(seed, "content", "") or "").strip(),
            "labels": metadata.get("labels") or [],
            "address": str(metadata.get("address") or "").strip(),
            "description": str(metadata.get("description") or "").strip(),
            "lat": metadata.get("lat"),
            "lng": metadata.get("lng"),
        }

    def _infer_entity_type(self, anchor: str) -> str:
        norm = normalize_text(anchor)
        if norm.startswith(("nha hang", "quan ")):
            return "Restaurant"
        if norm.startswith(("khach san", "nha nghi", "homestay", "resort")):
            return "Accommodation"
        if "tour" in norm:
            return "Tour"
        if any(token in norm for token in ["mon", "dac san", "am thuc"]):
            return "Dish"
        return "Place"
