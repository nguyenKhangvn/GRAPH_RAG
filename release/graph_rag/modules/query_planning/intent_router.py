import json
import re
from typing import Any, Dict, List, Optional
from graph_rag.core import keywords
from graph_rag.core.intents import IntentMode
from graph_rag.utils.text import normalize_text


class IntentRouter:
    INTENT_MODES = {
        IntentMode.SINGLE_ANCHOR,
        IntentMode.COMPARISON,
        IntentMode.CONSTRAINT_MATCHING,
        IntentMode.MULTI_ENTITY_NEARBY,
        IntentMode.DISH_TO_RESTAURANT,
        IntentMode.TOUR_PLAN,
        IntentMode.NEGATIVE,
    }

    RELATIONS_BY_INTENT = {
        IntentMode.COMPARISON: ["NEAR", "LOCATED_IN", "BELONGS_TO", "HAS"],
        IntentMode.MULTI_ENTITY_NEARBY: ["NEAR", "LOCATED_IN"],
        IntentMode.CONSTRAINT_MATCHING: ["NEAR", "LOCATED_IN", "BELONGS_TO", "HAS"],
        IntentMode.DISH_TO_RESTAURANT: ["HAS", "LOCATED_IN", "NEAR"],
        IntentMode.TOUR_PLAN: ["INCLUDES", "OFFERS", "NEAR", "LOCATED_IN", "BELONGS_TO"],
        IntentMode.SINGLE_ANCHOR: ["LOCATED_IN", "BELONGS_TO", "NEAR", "HAS", "HELD_AT"],
        IntentMode.NEGATIVE: ["LOCATED_IN", "BELONGS_TO", "NEAR", "HAS"],
    }

    def __init__(
        self,
        llm_service: Any | None = None,
        enable_llm: bool = False,
    ):
        self.llm_service = llm_service
        self.enable_llm = enable_llm

    def parse(self, question: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        metadata = metadata or {}
        rule_based = self._rule_based_parse(question, metadata)
        if not self.enable_llm or not self.llm_service:
            return rule_based

        try:
            llm_result = self._llm_parse(question, rule_based)
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError):
            llm_result = None

        if self._is_valid(llm_result):
            return llm_result
        return rule_based

    def _rule_based_parse(self, question: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_text(question, strip_punct=True)
        intent_mode = self._infer_intent_mode(normalized, metadata)
        label_hints = self._extract_category_label_hints(normalized)
        anchors = self._extract_anchors(question, metadata, normalized, keep_category_terms=bool(label_hints))
        if not anchors:
            target = str((metadata or {}).get("target_entity") or "").strip()
            if target:
                anchors = [target]
        if not anchors:
            if metadata.get("is_follow_up"):
                # Follow-up without new entities: use inherited location as anchor
                # instead of the full query text (which would fail exact/fuzzy match).
                inherited = str(metadata.get("detected_location") or "").strip()
                anchors = [inherited] if inherited else []
            else:
                # Use detected location as anchor instead of full query text.
                # Full query text fails exact/fuzzy match and produces garbage results.
                loc = str((metadata or {}).get("detected_location") or "").strip()
                anchors = [loc] if loc else [question.strip()]

        primary_anchor = anchors[0] if anchors else ""
        required_conditions = self._extract_required_conditions(question, metadata, normalized)
        target_entities = self._extract_target_entities(metadata, anchors)
        constraints = {
            "relations": list(self.RELATIONS_BY_INTENT.get(intent_mode, [])),
            "required_conditions": required_conditions,
            "target_entities": target_entities,
            "target_category": self._extract_target_category(metadata, normalized),
        }
        answer_contract = {
            "must_compare_all_anchors": intent_mode == IntentMode.COMPARISON,
            "must_validate_constraints": intent_mode in {IntentMode.COMPARISON, IntentMode.CONSTRAINT_MATCHING},
            "allow_no_candidate": intent_mode in {IntentMode.CONSTRAINT_MATCHING, IntentMode.NEGATIVE},
            "forbidden": ["invent_distance", "invent_price", "invent_travel_time"],
        }

        result = {
            "intent_mode": intent_mode,
            "primary_anchor": primary_anchor,
            "anchors": anchors,
            "constraints": constraints,
            "answer_contract": answer_contract,
        }
        if label_hints:
            result["label_hints"] = label_hints
        return result

    _ACCOMMODATION_SIGNALS = keywords.ACCOMMODATION_SIGNALS
    _HERITAGE_SIGNALS = keywords.HERITAGE_SIGNALS
    _TOURISM_SIGNALS = keywords.TOURISM_SIGNALS

    def _extract_category_label_hints(self, normalized: str) -> List[str]:
        """Extract entity-type label hints from query — these are category signals, not specific entities."""
        hints: List[str] = []
        if any(tok in normalized for tok in self._ACCOMMODATION_SIGNALS):
            hints.append("Accommodation")
        if any(tok in normalized for tok in self._HERITAGE_SIGNALS):
            hints.append("TouristAttraction")
        if any(tok in normalized for tok in self._TOURISM_SIGNALS):
            if "TouristAttraction" not in hints:
                hints.append("TouristAttraction")
        # Dedupe preserving order
        return list(dict.fromkeys(hints))

    def _infer_intent_mode(self, normalized: str, metadata: Dict[str, Any]) -> str:
        if str((metadata or {}).get("question_type") or "").strip().lower() == "tour-plan":
            return IntentMode.TOUR_PLAN
        retrieval_plan = str((metadata or {}).get("retrieval_plan_mode") or "").strip().lower()
        if retrieval_plan == "comparison":
            return IntentMode.COMPARISON
        if retrieval_plan == "multi_candidate":
            return IntentMode.CONSTRAINT_MATCHING
        if retrieval_plan == "dish_to_restaurant":
            return IntentMode.DISH_TO_RESTAURANT
        if retrieval_plan == "tour_plan":
            return IntentMode.TOUR_PLAN

        has_constraint_signal = any(
            token in normalized
            for token in keywords.CONSTRAINT_SIGNALS
        )
        has_filter_signal = any(
            token in normalized
            for token in keywords.FILTER_SIGNALS
        )
        if has_constraint_signal and has_filter_signal:
            return IntentMode.CONSTRAINT_MATCHING
        if any(token in normalized for token in keywords.COMPARISON_SIGNALS):
            return IntentMode.COMPARISON

        # Multi-category combination: query mentions 2+ entity types with combination signal
        has_combine_signal = any(
            token in normalized
            for token in keywords.COMBINE_SIGNALS
        )
        if has_combine_signal:
            category_count = 0
            if any(tok in normalized for tok in self._ACCOMMODATION_SIGNALS):
                category_count += 1
            if any(tok in normalized for tok in self._HERITAGE_SIGNALS):
                category_count += 1
            if any(tok in normalized for tok in self._TOURISM_SIGNALS):
                category_count += 1
            if any(tok in normalized for tok in keywords.FOOD_SIGNALS):
                category_count += 1
            if category_count >= 2:
                return IntentMode.MULTI_ENTITY_NEARBY

        if any(token in normalized for token in keywords.PROXIMITY_SIGNALS):
            return IntentMode.MULTI_ENTITY_NEARBY
        if any(token in normalized for token in keywords.FILTER_SIGNALS):
            return IntentMode.DISH_TO_RESTAURANT
        if any(token in normalized for token in keywords.CONSTRAINT_SIGNALS):
            return IntentMode.CONSTRAINT_MATCHING
        if any(token in normalized for token in keywords.NEGATIVE_SIGNALS):
            return IntentMode.NEGATIVE
        if any(token in normalized for token in keywords.TOUR_PLAN_SIGNALS):
            return IntentMode.TOUR_PLAN
        return IntentMode.SINGLE_ANCHOR

    def _extract_anchors(self, question: str, metadata: Dict[str, Any], normalized: str, keep_category_terms: bool = False) -> List[str]:
        anchors: List[str] = []

        for anchor in (metadata or {}).get("query_frame_anchor_names") or []:
            anchor = str(anchor or "").strip()
            if anchor:
                anchors.extend(self._split_anchor_list(anchor))

        anchors = self._dedupe(anchors, keep_category_terms=keep_category_terms)
        if len(anchors) >= 2:
            return anchors

        original_anchors = self._extract_comparison_anchors(question)
        if original_anchors:
            return original_anchors

        # Pronouns/deictics that should never be treated as anchors
        _PRONOUN_ANCHORS = {
            "day", "do", "nay", "kia", "ay", "no", "chung", "toi", "ban",
            "đây", "đó", "này", "kia", "ấy", "nó", "chúng", "tôi", "bạn",
        }
        for entity in (metadata or {}).get("entities") or []:
            if not isinstance(entity, dict):
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            if e_type in {"province", "city", "district", "ward", "commune", "location", "category"}:
                continue
            name = str(entity.get("name") or "").strip()
            if name and normalize_text(name, strip_punct=True) not in _PRONOUN_ANCHORS:
                anchors.extend(self._split_anchor_list(name))

        anchors = self._dedupe(anchors, keep_category_terms=keep_category_terms)
        if len(anchors) >= 2:
            return anchors

        for name in (metadata or {}).get("evidence_names") or []:
            name = str(name or "").strip()
            if name:
                anchors.append(name)
        anchors = self._dedupe(anchors, keep_category_terms=keep_category_terms)
        if len(anchors) >= 2:
            return anchors

        original_anchors = self._extract_comparison_anchors(question)
        if original_anchors:
            return original_anchors

        return anchors

    def _extract_comparison_anchors(self, question: str) -> List[str]:
        original_comparison = re.search(
            r"(?i)\b(?:giữa|giua)\s+(.+?)\s+(?:và|va|với|voi|vs)\s+(.+?)(?:\s+(?:thông qua|thong qua|dựa trên|dua tren|đối với|doi voi|nếu|neu)\b|[?.!]|$)",
            question,
        )
        if original_comparison:
            extracted = []
            for value in [original_comparison.group(1).strip(), original_comparison.group(2).strip()]:
                extracted.extend(self._split_anchor_list(value))
            return self._dedupe(extracted)

        original_compare_prefix = re.search(
            r"(?i)\bso\s+sánh\s+(?:vị trí của|lợi thế vị trí của|các|cac)?\s*(.+?)\s+(?:và|va|với|voi|vs)\s+(.+?)(?:\s+(?:dựa trên|dua tren|đối với|doi voi|về|ve|chúng|chung|nếu|neu)\b|[?.!]|$)",
            question,
        )
        if original_compare_prefix:
            extracted = []
            for value in [original_compare_prefix.group(1).strip(), original_compare_prefix.group(2).strip()]:
                extracted.extend(self._split_anchor_list(value))
            return self._dedupe(extracted)

        # Pattern: "A khác biệt so với B", "A khác B", "A khác gì B"
        diff_match = re.search(
            r"(?i)(.+?)\s+(?:khác\s+(?:biệt\s+(?:so\s+với|với)|nhau\s+với|gì\s+với|với))\s+(.+?)(?:\s+(?:thông qua|thong qua|dựa trên|dua tren|về|ve)\b|[?.!]|$)",
            question,
        )
        if diff_match:
            extracted = []
            # Clean subject 1: remove trailing question words like "có gì", "thế nào"
            raw_subject1 = re.sub(r"(?i)\s+(?:có\s+gì|thế\s+nào|như\s+thế\s+nào|ra\s+sao)\s*$", "", diff_match.group(1).strip())
            for value in [raw_subject1, diff_match.group(2).strip()]:
                extracted.extend(self._split_anchor_list(value))
            return self._dedupe(extracted)

        return []

    def _split_anchor_list(self, value: str) -> List[str]:
        text = str(value or "").strip(" ,.;:!?")
        if not text:
            return []
        parts = re.split(r"\s*(?:,|\s+và\s+|\s+va\s+)\s*", text, flags=re.IGNORECASE)
        cleaned = [part.strip(" ,.;:!?") for part in parts if part and part.strip(" ,.;:!?")]
        return cleaned or [text]

    def _extract_required_conditions(
        self,
        question: str,
        metadata: Dict[str, Any],
        normalized: str,
    ) -> List[str]:
        conditions: List[str] = []
        frame = (metadata or {}).get("query_frame") or {}
        plan = frame.get("retrieval_plan") or {}
        for rel in (
            (metadata or {}).get("query_frame_traversal_relations")
            or plan.get("required_relations")
            or []
        ):
            rel = str(rel or "").strip().upper()
            if rel:
                conditions.append(rel)
        if any(token in normalized for token in ["gan", "xung quanh", "lan can", "near"]):
            conditions.append("NEAR")
        if any(token in normalized for token in ["co mon", "phuc vu mon", "mon "]):
            conditions.append("HAS")
        if any(token in normalized for token in ["thuoc loai", "loai hinh", "phan loai", "thuoc nhom", "the loai", "loai hinh du lich"]):
            conditions.append("BELONGS_TO")
        if any(token in normalized for token in ["o dau", "dia chi", "nam tai", "khu vuc"]):
            conditions.append("LOCATED_IN")
        return self._dedupe(conditions)

    def _extract_target_entities(self, metadata: Dict[str, Any], anchors: List[str]) -> List[str]:
        targets: List[str] = []
        anchor_norms = {normalize_text(anchor, strip_punct=True) for anchor in anchors}
        for key in ("proximity_anchor", "target_entity"):
            _raw = (metadata or {}).get(key)
            value = str((_raw.get("text") if isinstance(_raw, dict) else _raw) or "").strip()
            if value and normalize_text(value, strip_punct=True) not in anchor_norms:
                targets.append(value)
        for name in (metadata or {}).get("evidence_names") or []:
            name = str(name or "").strip()
            if name and normalize_text(name, strip_punct=True) not in anchor_norms:
                targets.append(name)
        return self._dedupe(targets)

    def _extract_target_category(self, metadata: Dict[str, Any], normalized: str) -> Optional[str]:
        for entity in (metadata or {}).get("entities") or []:
            if not isinstance(entity, dict):
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            if e_type == "category":
                cat_name = str(entity.get("name") or "").strip()
                cat_norm = normalize_text(cat_name, strip_punct=True)
                if "di tich" in cat_norm:
                    return "Di tích lịch sử - Văn hóa"
                if "danh lam" in cat_norm:
                    return "Danh lam thắng cảnh"
                if "lang nghe" in cat_norm:
                    return "Làng nghề truyền thống"

        category = str((metadata or {}).get("multi_choice_target_category") or "").strip()
        if category:
            return category
        if "di tich lich su" in normalized:
            return "Di tích lịch sử - Văn hóa"
        if "danh lam thang canh" in normalized:
            return "Danh lam thắng cảnh"
        if "lang nghe truyen thong" in normalized:
            return "Làng nghề truyền thống"
        return None

    def _llm_parse(self, question: str, rule_based: Dict[str, Any]) -> Dict[str, Any]:
        system_prompt = (
            "You are an intent router for a travel knowledge graph. "
            "Return a strict JSON object with keys: intent_mode, primary_anchor, anchors, "
            "constraints, answer_contract."
        )
        user_prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Rule-based guess:\n"
            f"{json.dumps(rule_based, ensure_ascii=True)}\n\n"
            "Return JSON only."
        )
        raw = self.llm_service.generate_text(system_prompt, user_prompt)
        data = json.loads(raw)
        return data if isinstance(data, dict) else rule_based

    def _is_valid(self, data: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(data, dict):
            return False
        intent_mode = str(data.get("intent_mode") or "").strip()
        anchors = data.get("anchors")
        if intent_mode not in self.INTENT_MODES:
            return False
        if not isinstance(anchors, list) or not anchors:
            return False
        return True

    def _dedupe(self, items: List[str], keep_category_terms: bool = False) -> List[str]:
        categories_to_exclude = {
            "di tich lich su van hoa", "di tich lich su", "danh lam thang canh", "lang nghe truyen thong",
            "nha nghi", "khach san", "homestay", "resort", "am thuc", "mon an", "nha hang", "quan an",
            "diem du lich", "diem tham quan"
        }
        seen = set()
        result = []
        for item in items:
            item = self._clean_anchor(item)
            norm = normalize_text(item, strip_punct=True)
            if not norm or norm in seen:
                continue
            if not keep_category_terms and norm in categories_to_exclude:
                continue
            seen.add(norm)
            result.append(item)
        return result

    def _clean_anchor(self, value: str) -> str:
        text = str(value or "").strip(" ,.;:!?")
        if not text:
            return ""
        text = re.sub(r"(?i)^(?:hãy|hay)\s+", "", text).strip(" ,.;:!?")
        text = re.sub(r"(?i)^(?:so\s+sánh|so\s+sanh)\s+", "", text).strip(" ,.;:!?")
        text = re.sub(r"(?i)^(?:vị\s+trí\s+của|vi\s+tri\s+cua|lợi\s+thế\s+vị\s+trí\s+của|loi\s+the\s+vi\s+tri\s+cua)\s+", "", text).strip(" ,.;:!?")
        stop_patterns = [
            r"\s*:\s*(?:chúng|chung|nếu|neu)\b.*$",
            r"\s+(?:chúng|chung)\s+.*$",
            r"\s+(?:dựa trên|dua tren|thông qua|thong qua|đối với|doi voi|về|ve|nếu|neu)\b.*$",
            r"\s+(?:cùng|cung)\s+.*$",
        ]
        for pattern in stop_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip(" ,.;:!?")
        return text

