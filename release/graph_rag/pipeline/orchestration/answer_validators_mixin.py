from __future__ import annotations
"""Validation, safety guards, abstain logic, and hallucination checks."""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List

from graph_rag.config import RELATIONSHIP_MAP
from graph_rag.core.intents import IntentType
from graph_rag.utils.text import normalize_text
from .dto import PipelineRunState


class AnswerValidatorsMixin:
    """Mixin for answer validation, safety guards, and abstain logic."""

    _RELATION_RE = re.compile(r"^-\s*(.+?)\s+\[(\w+)\]\s*->\s*(.+?)\s*$")
    _ATTRIBUTE_RE = re.compile(r"^-\s*(\w+):\s*(.+)$")

    _ENTITY_STOPWORDS = frozenset({
        "la", "cua", "va", "o", "tai", "gan", "voi", "co", "duoc", "nam",
        "mot", "cac", "nhung", "nay", "do", "ay", "nao", "gi", "ai",
    })

    _LOCATION_PATTERN = re.compile(
        r"(?:^|\s)(?:ở|tại|tai|(?<!\w)o(?!\w))\s+([^\s,.;:!?]+(?:\s+(?!(?:năm|nam|tháng|thang|ngày|ngay|mùa|mua|như|the|nao|nào|bao|giá|gia|thế|như thế|có|không|khong|ở|tai|tại)\b)[^\s,.;:!?]+){0,4})",
        re.UNICODE,
    )

    _APOLOGY_PATTERNS = [
        r"xin\s+loi",
        r"khong\s+tim\s+thay",
        r"khong\s+co\s+thong\s+tin",
        r"khong\s+du\s+thong\s+tin",
        r"chua\s+(?:co|tim\s+thay)",
        r"khong\s+the\s+tra\s+loi",
        r"hien\s+chua\s+co",
        r"khong\s+suy\s+doan",
        r"ban\s+co\s+the\s+cung\s+cap\s+them",
    ]
    _APOLOGY_RE = re.compile(
        "|".join(f"(?:{p})" for p in _APOLOGY_PATTERNS), re.IGNORECASE
    )

    def _is_apology_answer(self, answer: str) -> bool:
        """Detect apology answers by checking only the last 2 lines.

        Avoids false positives from normalize_text collapsing newlines,
        which previously caused answers with facts + apology tail to be
        flagged incorrectly.
        """
        if not answer:
            return False
        lines = answer.strip().splitlines()
        non_empty = [l.strip() for l in lines if l.strip()]
        if not non_empty:
            return False
        # Only check last 2 lines — apologies are concluding statements
        tail = non_empty[-2:] if len(non_empty) > 2 else non_empty
        for line in tail:
            normalized = normalize_text(line, strip_punct=True)
            if normalized and self._APOLOGY_RE.search(normalized):
                return True
        return False

    def _enforce_closed_form_answer_format(
        self, answer: str, question_type: str, question: str = ""
    ) -> str:
        """Enforce deterministic answer format for closed-form question types."""
        if not answer or question_type not in {
            "True-or-False", "Multi-Choice", "Multi-Select"
        }:
            return answer

        normalized = normalize_text(answer, strip_punct=True)

        if question_type == "True-or-False":
            tf_match = re.match(
                r"^(đúng|sai|true|false|đúng\.|sai\.)",
                normalized,
                re.IGNORECASE,
            )
            if tf_match:
                verdict = "Đúng" if "dung" in normalize_text(tf_match.group(1), strip_punct=True) else "Sai"
                rest = answer[len(tf_match.group(0)):].strip().lstrip(". ")
                first_sentence = rest.split(".")[0].strip() if rest else ""
                if first_sentence:
                    return f"{verdict}. {first_sentence}."
                return f"{verdict}."
            if any(token in normalized for token in [
                "khong du thong tin", "khong the xac minh",
                "khong co du thong tin", "chua du thong tin",
            ]):
                return "Không đủ thông tin trong dữ liệu để xác minh."
            return answer

        if question_type in {"Multi-Choice", "Multi-Select"}:
            mc_match = re.match(r"^([A-D](?:\s*,\s*[A-D])*)\s*[.:)\-]", answer.strip())
            if mc_match:
                letters = mc_match.group(1).strip()
                rest = answer[mc_match.end():].strip().lstrip(". ")
                first_sentence = rest.split(".")[0].strip() if rest else ""
                if first_sentence:
                    return f"{letters}. {first_sentence}."
                return f"{letters}."
            single_match = re.match(r"^([A-D])\b", answer.strip())
            if single_match:
                return f"{single_match.group(1)}."
            return answer

        return answer

    def _context_has_facts(self, raw_context: List[str], entity_name: str = "") -> bool:
        if not raw_context:
            return False
        entity_norm = normalize_text(entity_name, strip_punct=True) if entity_name else ""
        for line in raw_context:
            line_str = str(line or "").strip()
            if not line_str:
                continue
            rel_match = self._RELATION_RE.match(line_str)
            if rel_match:
                if not entity_norm:
                    return True
                left = normalize_text(rel_match.group(1), strip_punct=True)
                right = normalize_text(rel_match.group(3), strip_punct=True)
                if entity_norm in left or entity_norm in right or left in entity_norm or right in entity_norm:
                    return True
                continue
            attr_match = self._ATTRIBUTE_RE.match(line_str)
            if attr_match:
                key = attr_match.group(1).strip().lower()
                if key in ("address", "phone", "description", "name", "type", "email", "opening_hours", "price", "duration", "activities", "image", "category"):
                    if not entity_norm:
                        return True
                    return True
            if "**THỰC THỂ CHÍNH:**" in line_str or "Thuc the chinh" in normalize_text(line_str, strip_punct=True):
                if not entity_norm:
                    return True
                header_norm = normalize_text(line_str, strip_punct=True)
                if entity_norm in header_norm:
                    return True
            if len(line_str) > 50 and not line_str.startswith("-"):
                if not entity_norm:
                    return True
        return False

    # Category terms that should NOT be treated as specific entity targets
    _CATEGORY_TERMS = frozenset({
        "quan cafe", "quan ca phe", "cafe", "coffee",
        "khach san", "nha nghi", "homestay", "resort",
        "nha hang", "quan an", "quán ăn",
        "diem du lich", "diem tham quan", "danh lam thang canh",
        "su kien", "le hoi", "mon an", "mon ngon",
        "tour", "lich trinh", "bien", "nui", "cho", "chua",
    })

    @classmethod
    def _is_category_phrase(cls, name_norm: str) -> bool:
        """Check if a normalized entity name is a category phrase (not a specific entity).

        Matches exact terms AND compound phrases that START with a category term.
        e.g. "khach san tot" starts with "khach san" → category phrase.
        """
        if not name_norm:
            return False
        if name_norm in cls._CATEGORY_TERMS:
            return True
        for term in cls._CATEGORY_TERMS:
            if name_norm.startswith(term + " ") or name_norm.startswith(term):
                return True
        return False

    def _primary_specific_entity_name(self, state: PipelineRunState) -> str:
        target = str((state.metadata or {}).get("target_entity") or "").strip()
        if target:
            target_norm = normalize_text(target, strip_punct=True)
            if not self._is_category_phrase(target_norm):
                return target
        for entity in state.entities or []:
            if not self._is_groundable_entity(entity):
                continue
            if entity.get("is_category_hint"):
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            if e_type in {"province", "city", "district", "ward", "commune", "location"}:
                continue
            e_name = str(entity.get("name") or "").strip()
            if e_name:
                e_norm = normalize_text(e_name, strip_punct=True)
                if self._is_category_phrase(e_norm):
                    continue
                return e_name
        return ""

    def _requires_entity_validation(self, state: PipelineRunState) -> bool:
        if self._is_category_listing_query(state.user_query):
            return False
        if self._is_service_availability_query(state.user_query):
            return bool(self._primary_specific_entity_name(state))
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        if intent in {IntentType.DISCOVERY, IntentType.TOUR_PLAN, IntentType.DISTANCE}:
            return False
        q = normalize_text(state.user_query, strip_punct=True)
        is_spatial_strategy_analysis = (
            any(token in q for token in ["phan tich", "chien luoc", "loi the", "tiem nang", "dinh vi", "phat trien"])
            and any(token in q for token in ["khong gian", "vi tri", "gan", "lan can", "xung quanh", "moi quan he", "dia ly", "di san"])
        )
        if is_spatial_strategy_analysis:
            return False
        return bool(self._primary_specific_entity_name(state))

    def _is_service_availability_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        return any(
            token in q
            for token in [
                "co wifi", "wifi mien phi", "cho dau xe", "dau xe",
                "phong vip", "dua don", "san bay", "mon chay",
                "khu vuc rieng", "tre em", "co cung cap", "co phuc vu", "co cho phep",
            ]
        )

    def _build_missing_data_answer(self, state: PipelineRunState, target_entity: str = "") -> str:
        entity = (target_entity or self._primary_specific_entity_name(state) or "địa điểm này").strip()
        q = normalize_text(state.user_query, strip_punct=True)
        service_terms = []
        term_map = [
            ("phong vip", "phòng VIP"), ("wifi", "wifi"), ("cho dau xe", "chỗ đậu xe"),
            ("dau xe", "chỗ đậu xe"), ("dua don", "dịch vụ đưa đón"), ("san bay", "dịch vụ sân bay"),
            ("mon chay", "món chay"), ("khu vuc rieng", "khu vực riêng"),
            ("tre em", "dịch vụ/khu vực cho trẻ em"), ("dieu hoa", "điều hòa"), ("nuoc nong", "nước nóng"),
        ]
        for marker, label in term_map:
            if marker in q and label not in service_terms:
                service_terms.append(label)
        detail = f" về {', '.join(service_terms)}" if service_terms else ""
        return (
            f"Xin lỗi, hiện hệ thống dữ liệu du lịch chưa có đủ thông tin{detail} "
            f"của {entity} để trả lời chính xác. Mình không suy đoán khi dữ liệu chưa xác minh."
        )

    def _retrieval_evidence_contains_entity(
        self, entity_name: str, seeds: List[Any], raw_context: List[str],
    ) -> bool:
        if not entity_name:
            return True
        entity_norm = normalize_text(entity_name, strip_punct=True)
        if not entity_norm:
            return True

        evidence_parts: List[str] = []
        for seed in seeds or []:
            metadata = getattr(seed, "metadata", {}) or {}
            evidence_parts.extend([
                str(getattr(seed, "content", "") or ""),
                str(metadata.get("name") or ""),
                str(metadata.get("address") or ""),
            ])
        evidence_parts.extend([str(item or "") for item in raw_context or []])
        evidence_norm = normalize_text(" ".join(part for part in evidence_parts if part), strip_punct=True)

        if entity_norm in evidence_norm:
            return True

        entity_keywords = set(entity_norm.split()) - self._ENTITY_STOPWORDS
        if entity_keywords:
            evidence_words = set(evidence_norm.split())
            overlap = len(entity_keywords & evidence_words)
            if overlap / len(entity_keywords) >= 0.8:
                return True

        longest = SequenceMatcher(
            None, entity_norm, evidence_norm
        ).find_longest_match(0, len(entity_norm), 0, len(evidence_norm))
        return longest.size >= len(entity_norm) * 0.8

    def _extract_location_from_query(self, query: str) -> str:
        """Extract location mentioned after 'ở/tại' in the query."""
        match = self._LOCATION_PATTERN.search(query or "")
        if match:
            return match.group(1).strip()
        return ""

    def _entity_validation_abstain(self, state: PipelineRunState, entity_name: str) -> Dict[str, Any]:
        state.runtime.metadata["entity_validation_failed"] = True
        state.runtime.metadata["entity_validation_target"] = entity_name
        return {
            "early_result": {
                "answer": (
                    f"Xin lỗi, tôi chưa tìm thấy dữ liệu đủ chắc chắn về {entity_name} "
                    "trong hệ thống dữ liệu du lịch hiện có để trả lời câu hỏi này."
                ),
                "metadata": state.runtime.metadata,
            }
        }

    def _validate_requested_context(self, state: PipelineRunState, raw_context: List[str]) -> Dict[str, Any]:
        query_norm = normalize_text(state.user_query, strip_punct=True)
        requested_attributes = [
            str(attr or "").strip()
            for attr in (state.metadata or {}).get("requested_attributes", [])
            if str(attr or "").strip()
        ]
        requested_attributes = [
            attr for attr in requested_attributes
            if any(hint in query_norm for hint in self.REQUESTED_ATTRIBUTE_QUERY_HINTS.get(attr, [attr]))
        ]
        requested_relations = [
            str(rel or "").strip().upper()
            for rel in (state.metadata or {}).get("requested_relations", [])
            if str(rel or "").strip()
        ]
        is_event_descriptor_query = self._is_event_descriptor_query(state.user_query)
        if is_event_descriptor_query:
            requested_relations = [rel for rel in requested_relations if rel == "HELD_AT"]
            if "HELD_AT" not in requested_relations:
                requested_relations.append("HELD_AT")
        if not requested_attributes and not requested_relations:
            return {"ok": True, "missing_attributes": [], "missing_relations": []}

        context_norm = normalize_text("\n".join(str(item or "") for item in raw_context or []), strip_punct=True)
        attribute_present: Dict[str, bool] = {}
        missing_attributes = []
        for attr in requested_attributes:
            markers = self.REQUESTED_ATTRIBUTE_LABELS.get(attr, [attr])
            present = any(normalize_text(marker, strip_punct=True) in context_norm for marker in markers)
            attribute_present[attr] = present
            if not present:
                missing_attributes.append(attr)

        missing_relations = []
        for rel in requested_relations:
            if rel == "LOCATED_IN" and attribute_present.get("address"):
                continue
            rel_marker = RELATIONSHIP_MAP.get(rel, rel)
            rel_markers = [rel, rel_marker]
            if not any(normalize_text(marker, strip_punct=True) in context_norm for marker in rel_markers):
                missing_relations.append(rel)

        return {
            "ok": not missing_attributes and not missing_relations,
            "missing_attributes": missing_attributes,
            "missing_relations": missing_relations,
        }

    def _should_hard_fail_context_validation(self, state: PipelineRunState, validation: Dict[str, Any]) -> bool:
        if validation.get("ok", True):
            return False
        q = normalize_text(state.user_query, strip_punct=True)
        if any(hint in q for hint in self.ANALYTICAL_LOCATION_HINTS):
            return False
        if (getattr(state, "metadata", None) or {}).get("query_frame_multi_anchor_mode") and getattr(state, "raw_context", None):
            return False
        missing_attrs = set(validation.get("missing_attributes") or [])
        missing_rels = set(validation.get("missing_relations") or [])
        if (
            self._is_event_descriptor_query(state.user_query)
            and missing_rels.issubset({"HAS", "INCLUDES", "OFFERS"})
        ):
            return False

        context_has_facts = self._context_has_facts(getattr(state, "raw_context", None))

        fact_attr_missing = bool(
            missing_attrs
            & {"address", "phone", "price", "ticket_price", "price_range", "opening_hours", "service_features"}
        )
        if fact_attr_missing and context_has_facts:
            return False

        tour_relation_missing = bool(
            missing_rels
            and set(missing_rels).issubset({"INCLUDES", "OFFERS"})
            and "tour" in q
        )
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        fact_relation_missing = bool(
            missing_rels
            and intent == IntentType.ENTITY_FACT
            and not tour_relation_missing
        )
        if fact_relation_missing and context_has_facts:
            return False

        negative_or_service_query = any(
            token in q
            for token in [
                "co cung cap", "co cho phep", "dua don", "thu cung", "an chay", "hai san tuoi song",
            ]
        )
        return fact_attr_missing or fact_relation_missing or negative_or_service_query

    def _requested_context_abstain(self, state: PipelineRunState, validation: Dict[str, Any]) -> Dict[str, Any]:
        state.runtime.metadata["context_validation_failed"] = True
        state.runtime.metadata["context_validation"] = validation
        missing = []
        if validation.get("missing_attributes"):
            missing.append(self._format_missing_attributes(validation["missing_attributes"]))
        if validation.get("missing_relations"):
            missing.append("thông tin liên kết: " + self._format_missing_relations(validation["missing_relations"]))
        missing_text = "; ".join(missing) or "thông tin được hỏi"
        return {
            "early_result": {
                "answer": (
                    "Xin lỗi, hệ thống dữ liệu du lịch hiện chưa có đủ "
                    f"{missing_text} để trả lời chính xác câu hỏi này."
                ),
                "metadata": state.runtime.metadata,
            }
        }

    @staticmethod
    def _format_missing_attributes(attributes: List[str]) -> str:
        labels = {
            "address": "địa chỉ", "phone": "số điện thoại", "price": "giá",
            "ticket_price": "giá vé", "price_range": "giá phòng",
            "opening_hours": "giờ mở cửa", "service_features": "thông tin dịch vụ",
            "room_count": "số lượng phòng", "room_type": "loại phòng", "amenities": "tiện nghi",
        }
        readable = [labels.get(attr, attr.replace("_", " ")) for attr in attributes]
        return "thông tin " + ", ".join(dict.fromkeys(readable))

    @staticmethod
    def _format_missing_relations(relations: List[str]) -> str:
        labels = {
            "INCLUDES": "điểm đến/hoạt động trong tour", "OFFERS": "đơn vị tổ chức tour",
            "NEAR": "địa điểm gần đó", "LOCATED_IN": "khu vực/địa chỉ",
            "BELONGS_TO": "loại hình", "HAS": "món ăn/dịch vụ", "HELD_AT": "địa điểm tổ chức",
        }
        readable = [labels.get(str(rel).upper(), str(rel).replace("_", " ").lower()) for rel in relations]
        return ", ".join(dict.fromkeys(readable))

    @staticmethod
    def _extract_entity_from_line(line: str) -> str:
        """Extract entity name from various line formats."""
        stripped = line.strip()
        if not stripped:
            return ""

        time_pattern = re.compile(r"\b\d{1,2}:\d{2}\b")
        bolds = re.findall(r"\*\*(.+?)\*\*", stripped)
        if bolds:
            for val in bolds:
                val = val.strip()
                if time_pattern.search(val):
                    continue
                if val.lower() in ["nghỉ trưa", "an trua", "ăn trưa", "nghi trua"]:
                    continue
                return val

        # Format 3: "Thông tin Name:" or "Thông tin của Name:"
        m = re.match(r"^Thông tin\s+(?:của\s+)?(.+?):", stripped)
        if m:
            return m.group(1).strip()
        # Format 4: "Name — ..." (standalone with dash separator)
        m = re.match(r"^([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*)\s+—\s", stripped)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _deduplicate_entities_in_answer(answer: str) -> str:
        """Remove duplicate entity blocks from answer text.

        Detects entities from various formats (**Name**, "Thông tin Name:", "Name — ...")
        and removes duplicate entries, keeping the first (usually richest) occurrence.
        """
        if not answer:
            return answer

        lines = answer.split("\n")
        result_lines = []
        seen_entities: set[str] = set()
        skip_block = False

        for line in lines:
            stripped = line.strip()

            entity_name = AnswerValidatorsMixin._extract_entity_from_line(line)
            if entity_name:
                entity_norm = normalize_text(entity_name, strip_punct=True)
                if entity_norm and entity_norm in seen_entities:
                    # Duplicate entity — skip this line and its continuation block
                    skip_block = True
                    continue
                elif entity_norm:
                    seen_entities.add(entity_norm)
                    skip_block = False
            elif skip_block:
                # Skip continuation lines (indented, empty, or sub-bullets)
                if stripped and not stripped.startswith(("- ", "  ", "\t")):
                    skip_block = False
                else:
                    continue

            result_lines.append(line)

        return "\n".join(result_lines)

    @staticmethod
    def _sanitize_answer_text(answer: str) -> str:
        # Delegate to AnswerGenerator's canonical implementation to avoid duplication
        from graph_rag.modules.generation.llm_client import AnswerGenerator
        cleaned = AnswerGenerator._sanitize_answer_text(answer)
        if not cleaned:
            return ""
        # Additional pipeline-specific: fix broken markdown tables
        cleaned = AnswerValidatorsMixin._normalize_markdown_tables(cleaned)
        # Deduplicate entities in the answer
        cleaned = AnswerValidatorsMixin._deduplicate_entities_in_answer(cleaned)
        return cleaned.strip()

    @staticmethod
    def _sanitize_answer_text_legacy(answer: str) -> str:
        """Legacy implementation kept for reference. Use _sanitize_answer_text instead."""
        if not answer:
            return ""
        replacements = {
            "确实": "thực sự", "远离": "tránh xa", "如": " như ",
            "nearby": "gần đó", "events": "sự kiện", "refresh": "thư giãn",
        }
        cleaned = str(answer)
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)

        natural_language_replacements = [
            (r"(?i)dựa trên thông tin trong\s+context\b", "Dựa trên thông tin hiện có"),
            (r"(?i)dựa trên\s+context\b", "Dựa trên thông tin hiện có"),
            (r"(?i)\bcontext\b", "thông tin hiện có"),
        ]
        for pattern, target in natural_language_replacements:
            cleaned = re.sub(pattern, target, cleaned)

        cleaned = re.sub(r"(?i)(?:có\s+)?tọa\s+độ\s+WGS84Point\([^)]*\)", "", cleaned)
        cleaned = re.sub(r"(?i)WGS84Point\([^)]*\)", "", cleaned)
        cleaned = re.sub(r"(?i)Point\([^)]*\)", "", cleaned)

        cleaned = re.sub(r"(?i)\s*[\(\[](?:NEAR|LOCATED_IN|BELONGS_TO|HAS|HELD_AT|INCLUDES|OFFERS)[\)\]]\s*", " ", cleaned)
        cleaned = re.sub(r"\s*(?:->|-->|<-|<--)\s*", " ", cleaned)

        cleaned = re.sub(r"[㐀-䶿一-鿿豈-﫿]", "", cleaned)

        cleaned = re.sub(r",\s*\.", ".", cleaned)
        cleaned = re.sub(r",\s*,", ",", cleaned)
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

        # Fix broken markdown tables: split rows that are on the same line
        cleaned = AnswerValidatorsMixin._normalize_markdown_tables(cleaned)

        return cleaned.strip()

    @staticmethod
    def _normalize_markdown_tables(text: str) -> str:
        """Fix broken markdown tables where multiple rows are on one line.

        Strategy: find separator pattern (|---|---|...) in the line, then
        split everything before it as header, after it as data rows.
        """
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("|"):
                result.append(line)
                continue

            # Find separator pattern: | --- | --- | ... (with possible spaces)
            sep_match = re.search(r"(\|[\s]*[-:]+[\s]*(?:\|[\s]*[-:]+[\s]*)+\|)", stripped)
            if not sep_match:
                # No separator — regular data row or non-table, keep as-is
                result.append(line)
                continue

            sep_str = sep_match.group(1)
            before_sep = stripped[:sep_match.start()].strip()
            after_sep = stripped[sep_match.end():].strip()

            if not before_sep and not after_sep:
                # Separator only line — already correct
                result.append(line)
                continue

            # Count columns from separator
            col_count = sep_str.count("|") - 1  # pipes - 1 = columns

            # Emit header (everything before separator)
            if before_sep:
                # Clean up: ensure it ends with |
                if not before_sep.endswith("|"):
                    before_sep += " |"
                result.append(before_sep)

            # Emit separator
            result.append(sep_str)

            # Emit data rows (everything after separator, split into rows)
            if after_sep:
                # Split after_sep into individual rows by finding "| " pattern
                # Each row starts with "|" after the separator
                # Use regex to find row boundaries
                row_parts = re.split(r"\|\s*(?=\S)", after_sep)
                data_cells = []
                for part in row_parts:
                    cells = [c.strip() for c in part.split("|") if c.strip()]
                    data_cells.extend(cells)

                # Emit rows of col_count cells each
                for i in range(0, len(data_cells), col_count):
                    row = data_cells[i:i + col_count]
                    if row:
                        result.append("| " + " | ".join(row) + " |")

        return "\n".join(result)

    def _question_has_negative_claim(self, question: str) -> bool:
        q = normalize_text(question, strip_punct=True)
        return any(marker in q for marker in [
            "khong phai", "khong thuoc", "khong nam", "khong gan", "chua duoc", "khong co",
        ])

    def _is_negative_option_question(self, question: str) -> bool:
        q = normalize_text(question, strip_punct=True)
        return any(marker in q for marker in [
            "khong phai", "khong thuoc", "khong nam trong", "dau khong", "lua chon nao khong", "phuong an nao khong",
        ])

    def _negative_abstain_answer(self, state: PipelineRunState, target_entity: str = "") -> str:
        entity = (target_entity or self._primary_specific_entity_name(state) or "đối tượng được hỏi").strip()
        return (
            f"Xin lỗi, hệ thống dữ liệu du lịch hiện chưa có đủ thông tin được xác minh về {entity} "
            "để trả lời chắc chắn câu hỏi này. Mình không suy đoán ngoài dữ liệu hiện có."
        )
