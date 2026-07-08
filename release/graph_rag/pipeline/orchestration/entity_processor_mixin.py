from __future__ import annotations
"""Entity classification, grounding selection, and canonicalization."""
import logging

logger = logging.getLogger(__name__)


import re


from types import SimpleNamespace


from typing import Any, Dict, List



from graph_rag.core import keywords


from graph_rag.core.intents import IntentType


from graph_rag.config.constants import NON_GROUNDABLE_ENTITY_TYPES
from graph_rag.utils.text import normalize_text


from graph_rag.config.deictic_patterns import (
    is_deictic_query,
)


from .dto import PipelineRunState
from graph_rag.modules.pipeline_support.distance_intent_service import DistanceQueryParser


class EntityProcessorMixin:
    """Mixin for entity processing, classification, and grounding selection."""

    # Re-export shared constant for subclass access via self.NON_GROUNDABLE_ENTITY_TYPES
    NON_GROUNDABLE_ENTITY_TYPES = NON_GROUNDABLE_ENTITY_TYPES

    OUT_OF_REGION_TERMS = list(keywords.OUT_OF_REGION_TERMS)

    IN_SCOPE_REGION_TERMS = list(keywords.IN_SCOPE_REGION_TERMS)

    _ACCOMMODATION_HINT_TOKENS = keywords.ACCOMMODATION_HINT_TOKENS
    _HERITAGE_HINT_TOKENS = keywords.HERITAGE_HINT_TOKENS
    _TOURISM_HINT_TOKENS = keywords.TOURISM_HINT_TOKENS
    _CATEGORY_LABEL_MAP = {
        "Accommodation": None,  # filled from _ACCOMMODATION_HINT_TOKENS
        "TouristAttraction": None,  # filled from _HERITAGE_HINT_TOKENS + _TOURISM_HINT_TOKENS
    }

    def _canonicalize_entity_name(self, name: str) -> str:
        text = str(name or "").strip()
        if not text:
            return ""

        cleaned = re.sub(r"_{2,}", "", text).strip(" ,.;:!?")
        cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned).strip(" ,.;:!?")
        prefix_patterns = [
            r"^(?:nhà\s+hàng|nha\s+hang)\s+",
            r"^(?:quán|quan)\s+",
            r"^(?:khách\s+sạn|khach\s+san)\s+",
            r"^(?:nhà\s+nghỉ|nha\s+nghi)\s+",
            r"^(?:khu\s+du\s+lịch|khu\s+du\s+lich)\s+",
        ]
        for pattern in prefix_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" ,.;:!?")

        suffix_patterns = [
            r"\s+(?:được\s+đặt|duoc\s+dat|đặt|dat)\s+(?:tại|tai|ở|o)\b.*$",
            r"\s+(?:tọa\s+lạc|toa\s+lac)\s+(?:tại|tai|ở|o)\b.*$",
            r"\s+(?:nằm|nam)\s+(?:tại|tai|ở|o)\b.*$",
            r"\s+(?:thuộc|thuoc)\b.*$",
            r"\s+(?:theo)\b.*$",
            r"\s+(?:ở|o)\b.*$",
            r"\s+(?:đối\s+với|doi\s+voi).*$",
            r"\s+(?:và|va)\s+(?:du\s+lich|thanh\s+pho).*$",
            r"\s+(?:hãy|har)\s+.*$",
            r"\s+(?:tại|tai)\b.*$",
        ]
        for _ in range(2):
            previous = cleaned
            for pattern in suffix_patterns:
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip(" ,.;:!?")
            if cleaned == previous:
                break
        cleaned = self._strip_entity_tail_noise(cleaned).strip(" ,.;:!?")
        # Phase 3: Truncate overly long entity names (>15 words likely malformed)
        words = cleaned.split()
        if len(words) > 15:
            # Try truncating at first comma
            comma_idx = cleaned.find(',')
            if comma_idx > 10:
                cleaned = cleaned[:comma_idx].strip()
            else:
                # Keep first 12 words
                cleaned = ' '.join(words[:12]).strip()
        return cleaned or text

    def _canonicalize_entities_for_grounding(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        canonicalized: List[Dict[str, Any]] = []
        for entity in entities or []:
            if not isinstance(entity, dict):
                canonicalized.append(entity)
                continue
            raw_name = str(entity.get("name") or "").strip()
            raw_norm = normalize_text(raw_name, strip_punct=True)
            e_type = str(entity.get("type") or "").strip().lower()
            is_lodging = e_type in {"accommodation", "hotel", "lodging"} or raw_norm.startswith(
                ("nha nghi ", "khach san ", "homestay ", "resort ")
            )
            is_short_named_quan = (
                e_type in {"restaurant", "food", "eatery"}
                and raw_norm.startswith("quan ")
                and len([part for part in raw_norm.split() if part]) <= 3
            )
            is_named_establishment = raw_norm.startswith(
                ("quan ", "nha hang ", "nha nghi ", "khach san ", "homestay ", "resort ", "tour ")
            )
            preserved_name = re.sub(r"\s*\([^)]*\)\s*$", "", raw_name).strip(" ,.;:!?")
            canonicalized.append({
                **entity,
                "name": preserved_name if (is_lodging or is_short_named_quan or is_named_establishment) else self._canonicalize_entity_name(raw_name),
            })
        return canonicalized

    def _prune_generic_recovered_anchors(self, anchors: List[str], query: str = "") -> List[str]:
        """Drop broad recovered anchors when specific anchors already exist.

        Example: keep "Bien Ho T'Nung" and "Bien Ho Che", drop the generic
        recovered anchor "Bien Ho". This prevents alias search from pulling
        every lake that contains "Ho".
        """
        cleaned = [str(anchor or "").strip() for anchor in anchors or [] if str(anchor or "").strip()]
        if len(cleaned) < 2:
            return cleaned

        generic_terms = keywords.GENERIC_ANCHOR_TERMS
        norms = {anchor: normalize_text(anchor, strip_punct=True) for anchor in cleaned}
        kept: List[str] = []
        for anchor in cleaned:
            norm = norms.get(anchor, "")
            tokens = norm.split()
            is_generic = norm in generic_terms or (len(tokens) <= 2 and any(t in generic_terms for t in [norm, tokens[-1] if tokens else ""]))
            if is_generic:
                has_specific_sibling = any(
                    other != anchor
                    and norm
                    and norm in other_norm
                    and len(other_norm.split()) > len(tokens)
                    for other, other_norm in norms.items()
                )
                if has_specific_sibling:
                    continue
            kept.append(anchor)
        return kept

    def _correct_entity_types_from_query_context(
        self,
        entities: List[Dict[str, Any]],
        query: str,
        metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Correct high-impact entity type drift before grounding.

        Router/analyzer can confuse place names containing food words (e.g.
        "Che" in "Bien Ho Che") with Dish. Query-level context such as
        entrance-fee or sightseeing terms is a stronger signal that anchors are
        TouristAttraction nodes.
        """
        q_norm = normalize_text(query or "", strip_punct=True)
        if not q_norm:
            return entities or []

        tourism_signals = keywords.QUERY_CONTEXT_TOURISM_SIGNALS
        if not any(signal in q_norm for signal in tourism_signals):
            return entities or []

        corrected: List[Dict[str, Any]] = []
        changed = False
        for entity in entities or []:
            if not isinstance(entity, dict):
                corrected.append(entity)
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            name_norm = normalize_text(str(entity.get("name") or ""), strip_punct=True)
            should_correct = (
                e_type in {"dish", "food", "specialty"}
                and bool(name_norm)
                and not self._is_generic_category_phrase(name_norm)
            )
            if should_correct:
                updated = dict(entity)
                updated["type"] = "TouristAttraction"
                updated["source"] = f"{entity.get('source') or 'unknown'}+query_context_type_correction"
                updated["type_corrected_by"] = "query_context"
                corrected.append(updated)
                changed = True
            else:
                corrected.append(entity)

        if changed and metadata is not None:
            metadata["entity_type_correction_applied"] = True
        return corrected

    def _prune_generic_entities_with_specific_siblings(
        self,
        entities: List[Dict[str, Any]],
        metadata: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        """Drop generic recovered entities when specific entities already exist."""
        if len(entities or []) < 2:
            return entities or []

        names = [
            str(entity.get("name") or "").strip()
            for entity in entities or []
            if isinstance(entity, dict) and str(entity.get("name") or "").strip()
        ]
        pruned_names = set(self._prune_generic_recovered_anchors(names))
        pruned_entities: List[Dict[str, Any]] = []
        changed = False
        for entity in entities or []:
            if not isinstance(entity, dict):
                pruned_entities.append(entity)
                continue
            name = str(entity.get("name") or "").strip()
            if name and name not in pruned_names:
                changed = True
                continue
            pruned_entities.append(entity)

        if changed and metadata is not None:
            metadata["generic_entity_pruning_applied"] = True
        return pruned_entities

    def _infer_entity_type_from_hint(self, hint: str) -> str:
        norm = normalize_text(hint, strip_punct=True)
        if norm.startswith(("quan ", "nha hang ")):
            return "Restaurant"
        if norm.startswith(("nha nghi ", "khach san ", "homestay ", "resort ")):
            return "Accommodation"
        return "Place"

    def _is_deictic_reference_query(self, query: str) -> bool:
        normalized = normalize_text(query, strip_punct=True)
        if not normalized:
            return False
        return is_deictic_query(normalized)

    def _build_grounded_anchor(self, node: Any, location_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if not node:
            return {}

        metadata = getattr(node, "metadata", {}) or {}
        labels = metadata.get("labels") or []
        anchor_location = ""
        if isinstance(location_context, dict):
            anchor_location = str(location_context.get("name") or "").strip()

        return {
            "id": str(getattr(node, "id", "") or ""),
            "name": str(metadata.get("name") or getattr(node, "content", "") or "").strip(),
            "labels": [str(label) for label in labels if str(label).strip()],
            "address": str(metadata.get("address") or "").strip(),
            "lat": metadata.get("lat"),
            "lng": metadata.get("lng"),
            "location": anchor_location or str(metadata.get("address") or "").strip(),
        }

    def _build_anchor_node(self, anchor: Dict[str, Any]) -> Any | None:
        if not isinstance(anchor, dict):
            return None
        name = str(anchor.get("name") or "").strip()
        if not name:
            return None
        metadata = {
            "name": name,
            "labels": [str(label) for label in (anchor.get("labels") or []) if str(label).strip()],
            "address": str(anchor.get("address") or anchor.get("location") or "").strip(),
            "lat": anchor.get("lat"),
            "lng": anchor.get("lng"),
        }
        return SimpleNamespace(
            id=str(anchor.get("id") or name),
            content=name,
            metadata=metadata,
        )

    def _is_groundable_entity(self, entity: Dict[str, Any] | None) -> bool:
        if not isinstance(entity, dict):
            return False
        e_type = str(entity.get("type") or "").strip().lower()
        if e_type in self.NON_GROUNDABLE_ENTITY_TYPES:
            return False
        e_name = str(entity.get("name") or "").strip()
        if not e_name:
            return False
        if self._is_generic_category_phrase(e_name):
            return False
        # Ignore pure numeric placeholders (e.g., group size "2").
        if e_name.isdigit():
            return False
        # Reject relation/proximity markers masquerading as entities
        e_norm = normalize_text(e_name, strip_punct=True)
        if e_norm in keywords.RELATION_MARKER_NAMES:
            return False
        # Reject very short tokens that are likely function words (e.g., "Xung")
        if len(e_norm) <= 3 and e_norm not in keywords.GROUNDABLE_SHORT_NAMES:
            return False
        return True

    def _is_generic_category_phrase(self, text: str) -> bool:
        norm = normalize_text(text, strip_punct=True)
        if not norm:
            return False
        # Regex patterns for accommodation-specific generic phrases
        generic_patterns = [
            r"\bkhach\s+san\s+(?:nao|khac\s+khong|trung\s+tam|phu\s+hop|gan\s+day)\b",
            r"\bnha\s+nghi\s+(?:nao|khac\s+khong|trung\s+tam|phu\s+hop|gan\s+day)\b",
            r"\bhomestay\s+(?:nao|khac\s+khong|trung\s+tam|phu\s+hop|gan\s+day)\b",
            r"\b(?:khach\s+san|nha\s+nghi|homestay)\s+.+\bbao\s+gom\b",
            r"\b(?:khach\s+san|nha\s+nghi|homestay)\s+.+\bco\s+nhung\b",
            r"\b(?:khach\s+san|nha\s+nghi|homestay)\s+.+\bnhung\b.+\bnao\b",
            r"\bkhach\s+san\s+khac\s+khong\b",
        ]
        if any(re.search(pattern, norm) for pattern in generic_patterns):
            return True
        # Exact-match generic category phrases (synced with frame_extractor.NON_GROUNDABLE_GENERIC_PHRASES)
        return norm in keywords.NON_GROUNDABLE_GENERIC_PHRASES

    def _repair_distance_entities(self, user_query: str, entities: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
        original = list(entities or [])
        src, dst = DistanceQueryParser.parse(user_query)
        repaired = []
        if src:
            repaired.append({
                "name": src,
                "type": "Location",
                "role": "origin",
                "source": "distance_parser",
                "confidence": 1.0,
                "trusted": True
            })
        if dst:
            dst_type = "Location"
            if len(original) >= 2 and isinstance(original[1], dict):
                hinted = str(original[1].get("type") or "").strip()
                if hinted:
                    dst_type = hinted
            elif len(original) == 1 and isinstance(original[0], dict):
                hinted = str(original[0].get("type") or "").strip()
                if hinted:
                    dst_type = hinted
            repaired.append({
                "name": dst,
                "type": dst_type,
                "role": "destination",
                "source": "distance_parser",
                "confidence": 1.0,
                "trusted": True
            })
        
        malformed = False
        if len(original) >= 1 and isinstance(original[0], dict):
            first = normalize_text(str(original[0].get("name") or ""), strip_punct=True)
            malformed = any(f" {conn} " in first for conn in keywords.DISTANCE_CONNECTORS)
            
        if (malformed or len(original) < 2) and repaired:
            return repaired
            
        return original

    def _select_entities_for_grounding(self, state: PipelineRunState) -> List[Dict[str, Any]]:
        # Exclude category hints — they expand retrieval labels, not grounding.
        candidates = [
            e for e in (state.entities or [])
            if self._is_groundable_entity(e) and not e.get("is_category_hint")
        ]
        if not candidates:
            return []

        # Broad admin location filter removed — downstream RRF + MMR + BAAI reranker
        # handle ranking quality; pre-filtering broke ENTITY_FACT queries about locations.

        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent

        if not candidates:
            return []

        if (state.metadata or {}).get("ticket_price_contract_active") and len(candidates) > 1:
            return candidates

        target_entity = str((state.metadata or {}).get("target_entity") or "").strip()
        if target_entity and intent not in {IntentType.DISTANCE, IntentType.TOUR_PLAN}:
            # Skip target_entity filter when multiple candidates exist for
            # FOOD_RECOMMENDATION / DISCOVERY intents — all food entities should
            # be grounded, not just the first one that became target_entity.
            _multi_entity_intents = {IntentType.FOOD, IntentType.DISCOVERY, IntentType.TOURISM}
            if len(candidates) > 1 and intent in _multi_entity_intents:
                pass  # keep all candidates
            else:
                target_norm = normalize_text(target_entity, strip_punct=True)
                target_matches = [
                    e for e in candidates
                    if normalize_text(str(e.get("name") or ""), strip_punct=True) == target_norm
                ]
                if target_matches:
                    return target_matches

        # For fact verification, favor target entities and avoid administrative
        # location fragments that often trigger noisy fuzzy matches.
        if intent == IntentType.ENTITY_FACT:
            non_location = [
                e for e in candidates
                if str(e.get("type") or "").strip().lower() != "location"
            ]
            if non_location:
                return non_location

        return candidates

    def _classify_entity(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        """Classify an entity as specific or category hint."""
        name = str(entity.get("name") or "").strip()
        name_norm = normalize_text(name, strip_punct=True)
        entity_type = str(entity.get("type") or "").strip().lower()

        # Already typed as Category by LLM
        if entity_type == "category":
            label = self._category_label_from_text(name_norm)
            if label:
                return {**entity, "is_category_hint": True, "label_hint": label}

        # Determine if name has a proper-name component
        all_category_tokens = set(
            self._ACCOMMODATION_HINT_TOKENS + self._HERITAGE_HINT_TOKENS + self._TOURISM_HINT_TOKENS
            + keywords.ADDITIONAL_CATEGORY_TOKENS
        )
        tokens = name_norm.split()
        has_proper_name = any(tok not in all_category_tokens for tok in tokens)

        # Check against known category patterns
        if any(tok in name_norm for tok in self._ACCOMMODATION_HINT_TOKENS):
            if not has_proper_name and (not re.search(r"\d+", name) or len(name_norm.split()) <= 3):
                return {**entity, "is_category_hint": True, "label_hint": "Accommodation"}
        if any(tok in name_norm for tok in self._HERITAGE_HINT_TOKENS):
            if not has_proper_name:
                label = self._category_label_from_text(name_norm)
                if label:
                    return {**entity, "is_category_hint": True, "label_hint": label}
        if any(tok in name_norm for tok in self._TOURISM_HINT_TOKENS):
            if not has_proper_name and len(name_norm.split()) <= 4:
                return {**entity, "is_category_hint": True, "label_hint": "TouristAttraction"}

        return entity

    def _category_label_from_text(self, name_norm: str) -> str:
        if any(tok in name_norm for tok in self._ACCOMMODATION_HINT_TOKENS):
            return "Accommodation"
        if any(tok in name_norm for tok in self._HERITAGE_HINT_TOKENS):
            return "TouristAttraction"
        if any(tok in name_norm for tok in self._TOURISM_HINT_TOKENS):
            return "TouristAttraction"
        return ""

    def _expand_labels_from_category_hints(
        self,
        current_labels: List[str],
        v3_intent_data: Dict[str, Any],
        query_norm: str,
    ) -> List[str]:
        """Expand retrieval_allowed_labels when query signals multiple entity types."""
        label_hints = list(v3_intent_data.get("label_hints") or [])
        if not label_hints:
            has_accommodation = any(tok in query_norm for tok in self._ACCOMMODATION_HINT_TOKENS)
            has_heritage = any(tok in query_norm for tok in self._HERITAGE_HINT_TOKENS)
            has_tourism = any(tok in query_norm for tok in self._TOURISM_HINT_TOKENS)
            if has_accommodation:
                label_hints.append("Accommodation")
            if has_heritage or has_tourism:
                if "TouristAttraction" not in label_hints:
                    label_hints.append("TouristAttraction")
        if not label_hints:
            return current_labels
        current_set = set(current_labels)
        hint_set = set(label_hints)
        if not hint_set - current_set:
            return current_labels
        expanded = list(dict.fromkeys(current_labels + label_hints))
        return expanded
