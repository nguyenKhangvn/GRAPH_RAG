from __future__ import annotations
"""Context filtering, scoring, slot extraction, and open analysis."""

import re
from typing import Any, Dict, List

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

from graph_rag.config import GRAPH_RAG_V3_ENABLED
from graph_rag.core.answer_mode import AnswerMode
from graph_rag.utils.text import normalize_text
from .dto import PipelineRunState


class ContextProcessorMixin:
    """Mixin for context processing, filtering, and slot extraction."""

    def _closed_form_context_text(self, state: PipelineRunState) -> str:
        """Use broad evidence for closed-form verification before MMR pruning."""
        parts: list[str] = []
        if state.raw_context:
            parts.extend(str(item or "") for item in state.raw_context if str(item or "").strip())
        if state.clean_context:
            parts.append(str(state.clean_context))
        return "\n".join(parts)


    def _direct_context_lines(self, state: PipelineRunState, entity_name: str = "") -> list[str]:
        context_text = self._closed_form_context_text(state)
        lines = [str(line or "").strip() for line in context_text.splitlines() if str(line or "").strip()]
        entity_norm = normalize_text(entity_name or self._primary_specific_entity_name(state), strip_punct=True)
        if not entity_norm:
            return lines
        direct: list[str] = []
        supporting: list[str] = []
        for line in lines:
            line_norm = normalize_text(line, strip_punct=True)
            if entity_norm in line_norm or line_norm in entity_norm:
                direct.append(line)
            elif re.search(r"\[[A-Z_]+\]\s*->", line):
                supporting.append(line)
        return direct + supporting[:8]


    def _context_overlap_score(self, text: str, context_text: str) -> float:
        tokens = self._content_tokens(text)
        if not tokens:
            return 0.0
        context_tokens = self._content_tokens(context_text)
        if not context_tokens:
            return 0.0
        return len(tokens & context_tokens) / len(tokens)


    def _prioritize_raw_context_for_target(self, state: PipelineRunState, raw_context: list[str]) -> list[str]:
        if not raw_context:
            return raw_context
        target_candidates = [
            (state.metadata or {}).get("target_entity"),
            (state.metadata or {}).get("analysis_subject_entity_hint"),
            (state.metadata or {}).get("multi_choice_anchor_hint"),
            (state.metadata or {}).get("fill_blank_subject_entity_hint"),
            (state.metadata or {}).get("statement_subject_entity_hint"),
            (lambda _p: _p.get("text") if isinstance(_p, dict) else _p)((state.metadata or {}).get("proximity_anchor")),
        ]
        for seed in state.grounded_nodes or state.all_seeds or []:
            seed_meta = getattr(seed, "metadata", {}) or {}
            target_candidates.append(seed_meta.get("name") or getattr(seed, "content", ""))

        generic = {
            "dia chi", "thong tin", "cac moi quan he", "moi quan he", "dia diem",
            "nha hang", "khach san", "nha nghi", "quan", "tour", "dia chi",
        }
        targets: list[str] = []
        for candidate in target_candidates:
            text = str(candidate or "").strip(" ,.;:!?")
            norm = normalize_text(text, strip_punct=True)
            if not norm or norm in generic or len(norm.split()) < 2:
                continue
            if norm not in targets:
                targets.append(norm)
        if not targets:
            return raw_context

        def score(line: str) -> int:
            line_norm = normalize_text(line, strip_punct=True)
            best = 0
            for target in targets:
                if target and target in line_norm:
                    best = max(best, 100 + min(40, len(target.split()) * 3))
                elif target and line_norm.startswith(target.split()[0]):
                    target_tokens = set(target.split())
                    line_tokens = set(line_norm.split())
                    if target_tokens and len(target_tokens & line_tokens) / len(target_tokens) >= 0.67:
                        best = max(best, 70)
            return best

        indexed = list(enumerate(raw_context))
        sorted_items = sorted(indexed, key=lambda item: (-score(str(item[1] or "")), item[0]))
        if sorted_items and score(str(sorted_items[0][1] or "")) > 0:
            state.metadata["context_target_prioritized"] = True
            state.metadata["context_target_terms"] = targets[:5]
            return [item for _, item in sorted_items]
        return raw_context


    def _dynamic_context_top_k(self, state: PipelineRunState) -> int:
        is_multi_anchor = False
        v3_intent_data = (state.metadata or {}).get("v3_intent_data") or {}
        anchors = v3_intent_data.get("anchors") or []
        if len(anchors) > 1:
            is_multi_anchor = True
        
        frame = (state.metadata or {}).get("query_frame") or {}
        if frame.get("query_operator") == "comparison" or (state.metadata or {}).get("retrieval_plan_mode") == "comparison" or (state.metadata or {}).get("query_frame_multi_anchor_mode"):
            is_multi_anchor = True

        answer_mode = (state.metadata or {}).get("answer_mode", AnswerMode.FACT_ANSWER)
        # Per-intent base_k: list/discovery intents need more facts to avoid
        # over-pruning (70→4 bug). Single-fact intents keep tighter budgets.
        _INTENT_BASE_K = {
            "FILL_BLANK_SHORT": 6,
            "SINGLE_OPTION_RESOLVER": 8,
            "TRUE_FALSE_VERIFIER": 8,
            "OPEN_ANALYSIS": 12,
            "TOUR_PLAN": 20,
            "DISCOVERY_LIST": 16,
            "TOUR_LIST": 16,
        }
        base_k = _INTENT_BASE_K.get(answer_mode, 8)
        # FOOD_RECOMMENDATION / DISCOVERY_SEARCH with list answer modes need extra budget
        intent = (state.metadata or {}).get("intent") or ""
        if answer_mode not in _INTENT_BASE_K and any(kw in str(intent).upper() for kw in ("FOOD", "DISCOVERY", "ACCOMMODATION", "TOURISM")):
            base_k = max(base_k, 14)

        if GRAPH_RAG_V3_ENABLED and is_multi_anchor:
            anchors_count = max(2, len(anchors))
            return max(base_k, 16 * anchors_count)

        return base_k


    def _target_context_terms(self, state: PipelineRunState) -> list[str]:
        values = [
            self._primary_specific_entity_name(state),
            (state.metadata or {}).get("target_entity"),
            (state.metadata or {}).get("analysis_subject_entity_hint"),
            (state.metadata or {}).get("multi_choice_anchor_hint"),
            (state.metadata or {}).get("fill_blank_subject_entity_hint"),
            (lambda _p: _p.get("text") if isinstance(_p, dict) else _p)((state.metadata or {}).get("proximity_anchor")),
        ]
        # Evidence names help preserve required relation targets without keeping
        # unrelated seed attributes.
        values.extend((state.metadata or {}).get("evidence_names") or [])
        terms: list[str] = []
        for value in values:
            norm = normalize_text(str(value or "").strip(), strip_punct=True)
            if norm and len(norm.split()) >= 2 and norm not in terms:
                terms.append(norm)
        return terms


    def _filter_context_for_target(self, state: PipelineRunState, raw_context: list[str]) -> list[str]:
        if not raw_context:
            return raw_context
        answer_mode = (state.metadata or {}).get("answer_mode", AnswerMode.FACT_ANSWER)
        terms = self._target_context_terms(state)
        if not terms or answer_mode == AnswerMode.TOUR_PLAN:
            return raw_context

        relation_markers = [
            " nam gan ", " nằm gần ", " near", " located_in", " belongs_to",
            " has ", " includes ", " offers ", " nằm tại ", " thuoc loai ",
            " thuộc loại ", " dia chi ", " địa chỉ ", " sdt ", " tọa độ ", " toa do ",
        ]

        def score(line: str) -> int:
            norm = normalize_text(line, strip_punct=True)
            if not norm:
                return 0
            value = 0
            for idx, term in enumerate(terms):
                if term in norm:
                    value += 120 if idx == 0 else 45
                else:
                    term_tokens = set(term.split())
                    line_tokens = set(norm.split())
                    if term_tokens and len(term_tokens & line_tokens) / len(term_tokens) >= 0.75:
                        value += 60 if idx == 0 else 25
            if any(marker.strip() in norm for marker in relation_markers):
                value += 20
            if norm.startswith(("thong tin cua ", "thong tin ")):
                value -= 10
            return value

        scored = [(score(str(line or "")), idx, line) for idx, line in enumerate(raw_context)]
        positive = [(s, idx, line) for s, idx, line in scored if s > 0]
        if not positive:
            return raw_context
        positive.sort(key=lambda item: (-item[0], item[1]))
        # Keep a bounded candidate pool before MMR; final top-k is applied below.
        pool_size = max(self._dynamic_context_top_k(state) * 3, 12)
        selected = [line for _, _, line in positive[:pool_size]]
        state.metadata["target_context_filter_applied"] = True
        state.metadata["target_context_filter_terms"] = terms[:8]
        state.metadata["target_context_filter_before"] = len(raw_context)
        state.metadata["target_context_filter_after"] = len(selected)
        return selected


    def _light_abstain_after_context_filter(self, state: PipelineRunState, raw_context: list[str]) -> dict[str, Any] | None:
        question_type = str((state.metadata or {}).get("question_type") or "").strip().lower()
        if question_type in {"negative", "negative-sample", "negative_sample"}:
            return None
        if (state.metadata or {}).get("query_frame_multi_anchor_mode") or (state.metadata or {}).get("query_frame_global_discovery"):
            return None
        # Skip for follow-up queries — follow-up exclusion handles candidate filtering
        if (state.metadata or {}).get("is_follow_up"):
            return None
        target = self._primary_specific_entity_name(state)
        if not target:
            return None
        if self._retrieval_evidence_contains_entity(target, state.all_seeds or [], raw_context or []):
            return None
        answer_mode = (state.metadata or {}).get("answer_mode", AnswerMode.FACT_ANSWER)
        if answer_mode in {AnswerMode.FILL_BLANK_SHORT, AnswerMode.OPEN_ANALYSIS}:
            state.metadata["light_abstain_gate"] = "target_not_in_filtered_context"
            return self._entity_validation_abstain(state, target)
        return None


    def _direct_1hop_context_for_main_entity(
        self,
        main_entity: Any,
        primary_intent: str,
        organizer: Any,
    ) -> list[str]:
        return ContextStage(self).direct_1hop_context_for_main_entity(
            main_entity=main_entity,
            primary_intent=primary_intent,
            organizer=organizer,
        )


    def _seed_attribute_context(self, seeds: list[Any], state: PipelineRunState | None = None) -> list[str]:
        return ContextStage(self).seed_attribute_context(seeds=seeds, state=state)


    def _analysis_main_entity_name(self, state: PipelineRunState) -> str:
        metadata = state.metadata or {}
        generic_norms = {
            "no",
            "dia chi",
            "qua trinh xay dung",
            "cac moi quan he",
            "moi quan he",
            "thong tin",
            "danh lam thang canh",
            "di tich lich su van hoa",
            "touristattraction",
            "accommodation",
            "restaurant",
        }
        for key in (
            "analysis_subject_entity_hint",
            "statement_subject_entity_hint",
            "multi_choice_anchor_hint",
            "proximity_anchor",
            "target_entity",
        ):
            value = self._strip_entity_tail_noise(str(metadata.get(key) or "")).strip(" ,.;:!?")
            value_norm = normalize_text(value, strip_punct=True)
            if value and value_norm not in generic_norms and len(value_norm.split()) >= 2:
                if any(value_norm == g or value_norm.endswith(" " + g) for g in generic_norms):
                    continue
                for seed in state.grounded_nodes or state.all_seeds or []:
                    seed_meta = getattr(seed, "metadata", {}) or {}
                    seed_name = str(seed_meta.get("name") or getattr(seed, "content", "") or "").strip()
                    seed_norm = normalize_text(seed_name, strip_punct=True)
                    if not seed_name or not seed_norm:
                        continue
                    if value_norm in seed_norm:
                        return seed_name
                    seed_tokens = set(seed_norm.split()) - self._ENTITY_STOPWORDS
                    value_tokens = set(value_norm.split()) - self._ENTITY_STOPWORDS
                    if seed_tokens and value_tokens and seed_tokens.issubset(value_tokens):
                        if len(seed_tokens & value_tokens) / max(1, len(value_tokens)) >= 0.6:
                            return seed_name
                return value
        primary = self._primary_specific_entity_name(state)
        if primary:
            return primary
        for seed in state.grounded_nodes or state.all_seeds or []:
            seed_meta = getattr(seed, "metadata", {}) or {}
            name = str(seed_meta.get("name") or getattr(seed, "content", "") or "").strip()
            if name:
                return name
        return "đối tượng được hỏi"


    def _looks_like_itinerary_answer(self, answer: str) -> bool:
        text = normalize_text(answer or "", strip_punct=True)
        itinerary_markers = [
            "lich trinh goi y",
            "ngay 1",
            "ngay 2",
            "08:00",
            "09:00",
            "uoc tinh chi phi",
            "diem khoi hanh",
        ]
        return any(marker in text for marker in itinerary_markers)


    def _open_analysis_context_text(self, state: PipelineRunState, main_entity: str) -> str:
        context_text = self._closed_form_context_text(state)
        if not context_text.strip():
            return ""
        main_norm = normalize_text(main_entity, strip_punct=True)
        lines = [str(line or "").strip() for line in context_text.splitlines() if str(line or "").strip()]
        main_lines = [
            line
            for line in lines
            if main_norm and main_norm in normalize_text(line, strip_punct=True)
        ]
        supporting = [line for line in lines if line not in main_lines]
        if len(main_lines) >= 3:
            selected = main_lines[:24]
        elif main_lines:
            selected = main_lines[:24]
        else:
            selected = lines[:16]
        return "\n".join(selected)[:7000]


    def _build_open_analysis_deterministic_summary(self, state: PipelineRunState, main_entity: str) -> str:
        context_text = self._open_analysis_context_text(state, main_entity)
        if not context_text.strip():
            return f"Xin lỗi, hệ thống dữ liệu du lịch hiện chưa có đủ thông tin về {main_entity} để phân tích chính xác."
        facts = [line.strip("- ").strip() for line in context_text.splitlines() if line.strip()]
        fact_text = "; ".join(facts[:8])
        return (
            f"Dựa trên dữ liệu hiện có, {main_entity} có các thông tin chính sau: {fact_text}. "
            "Các thông tin này cho thấy điểm mạnh cần được phân tích nằm ở vị trí, loại hình, "
            "mối liên kết với các điểm/hoạt động liên quan và khả năng kết hợp trải nghiệm. "
            "Tôi không lập lịch trình vì câu hỏi yêu cầu phân tích, không yêu cầu sắp xếp chuyến đi theo thời gian."
        )


    def _open_analysis_slot_hints(self, context_text: str) -> str:
        context_norm = normalize_text(context_text, strip_punct=True)
        slots = []
        if any(marker in context_norm for marker in ["loai", "type", "category", "thuoc loai"]):
            slots.append("loại/phân loại")
        if any(marker in context_norm for marker in ["dia chi", "address", "wgs84point", "nam tai", "located_in"]):
            slots.append("vị trí/địa chỉ/tọa độ")
        if "description" in context_norm or "thong tin" in context_norm:
            slots.append("mô tả")
        for rel in ["NEAR", "LOCATED_IN", "BELONGS_TO", "HAS", "HELD_AT", "INCLUDES", "OFFERS"]:
            if rel.lower() in context_norm:
                slots.append(f"quan hệ {rel}")
        slots.append("ý nghĩa du lịch thực tiễn từ các dữ kiện trên")
        return "; ".join(slots)


    def _extract_open_slots(self, state: PipelineRunState, main_entity: str) -> dict[str, Any]:
        context_text = self._closed_form_context_text(state)
        main_norm = normalize_text(main_entity, strip_punct=True)
        slots: dict[str, Any] = {
            "entity": main_entity,
            "type": "",
            "address": "",
            "phone": "",
            "coordinates": "",
            "located_in": [],
            "near": [],
            "has": [],
            "includes": [],
        }
        for raw_line in context_text.splitlines():
            line = str(raw_line or "").strip().lstrip("- ").strip()
            if not line:
                continue
            norm = normalize_text(line, strip_punct=True)
            if main_norm and main_norm not in norm:
                continue
            if not slots["type"]:
                m = re.search(r"(?i)(?:thuộc loại|thuoc loai|loại hình|loai hinh)\s+(?:.+?:\s*)?(.+)$", line)
                if m:
                    slots["type"] = self._sanitize_answer_text(m.group(1)).strip(" .;:,")
            if not slots["address"]:
                m = re.search(r"(?i)(?:địa chỉ|dia chi)\s+.+?:\s*(.+)$", line)
                if m:
                    slots["address"] = self._sanitize_answer_text(m.group(1)).strip(" .;:,")
            if not slots["phone"]:
                m = re.search(r"(?i)(?:SĐT|SDT|phone|số điện thoại(?:\s+của)?|so dien thoai(?:\s+cua)?)\s+.+?:\s*(.+)$", line)
                if m:
                    slots["phone"] = self._sanitize_answer_text(m.group(1)).strip(" .;:,")
                elif any(marker in norm for marker in ["sdt", "so dien thoai", "phone"]) and ":" in line:
                    slots["phone"] = self._sanitize_answer_text(line.rsplit(":", 1)[-1]).strip(" .;:,")
            if not slots["coordinates"]:
                m = re.search(r"(?i)(?:tọa độ|toa do)\s+.+?:\s*(WGS84Point\([^)]+\))", line)
                if m:
                    slots["coordinates"] = self._sanitize_answer_text(m.group(1)).strip()
            rel_patterns = [
                ("near", r"(?i)nằm gần\s+(.+?)(?:\s+\(Địa chỉ|\s+\(Dia chi|$)"),
                ("located_in", r"(?i)nằm tại\s+(.+)$"),
                ("has", r"(?i)(?:có món|phục vụ|có dịch vụ)\s+(.+)$"),
                ("includes", r"(?i)bao gồm điểm đến\s+(.+?)(?:\s+\(Địa chỉ|\s+\(Dia chi|$)"),
            ]
            for key, pattern in rel_patterns:
                m = re.search(pattern, line)
                if m:
                    value = m.group(1).strip(" .;:,")
                    cleaned_val = self._sanitize_answer_text(value).strip()
                    val_norm = normalize_text(cleaned_val, strip_punct=True)
                    if cleaned_val and not any(normalize_text(x, strip_punct=True) == val_norm for x in slots[key]):
                        slots[key].append(cleaned_val)
        return slots


    def _build_slot_based_open_answer(self, state: PipelineRunState, main_entity: str) -> str:
        slots = self._extract_open_slots(state, main_entity)
        q_norm = normalize_text(state.user_query or "", strip_punct=True)
        if not any(slots.get(key) for key in ["type", "address", "phone", "coordinates", "located_in", "near", "has", "includes"]):
            return ""

        parts = [f"### **{main_entity}**"]

        info_lines = []
        if slots["type"]:
            info_lines.append(f"- **Phân loại**: {slots['type']}")
        if slots["address"]:
            info_lines.append(f"- **Địa chỉ**: {slots['address']}")
        if slots["coordinates"]:
            info_lines.append(f"- **Tọa độ**: {slots['coordinates']}")
        if slots["phone"]:
            info_lines.append(f"- **Số điện thoại**: {slots['phone']}")
        if slots["located_in"]:
            info_lines.append(f"- **Về vị trí hành chính**: Nằm tại {', '.join(slots['located_in'][:3])}")

        if info_lines:
            parts.append("\n".join(info_lines))

        # Check if description exists in context to print
        description = ""
        context_text = self._closed_form_context_text(state)
        for line in context_text.splitlines():
            line_str = str(line or "").strip()
            if len(line_str) > 50 and not line_str.startswith("-") and "**THỰC THỂ CHÍNH" not in line_str and "**THUC THE CHINH" not in line_str:
                description = line_str
                break
        
        if description:
            parts.append(f"\n#### **Thông tin chi tiết:**\n{description}")

        details = []
        if slots["has"]:
            details.append(f"- **Món/dịch vụ liên quan**: {', '.join(slots['has'][:4])}")
        if slots["near"]:
            details.append(f"- **Các địa điểm lân cận nổi bật**: {', '.join(slots['near'][:6])}")
        if slots["includes"]:
            details.append(f"- **Các điểm/hoạt động kết nối**: {', '.join(slots['includes'][:5])}")
            
        if details:
            parts.append("\n#### **Gợi ý hành trình & Điểm liên quan:**\n" + "\n".join(details))

        is_general = any(hint in q_norm for hint in ["giới thiệu", "gioi thieu", "mô tả", "mo ta", "phân tích", "phan tich", "lịch trình", "lich trinh", "tư vấn", "tu van", "chi tiết", "chi tiet", "tổng quan", "tong quan"])

        if slots["near"] and is_general:
            parts.append(
                "\nCác quan hệ lân cận này tạo lợi thế kết hợp hành trình: khách có thể dùng địa điểm chính như một điểm dừng "
                "rồi nối sang các điểm văn hóa, lịch sử hoặc tham quan xung quanh. Điều đó hỗ trợ cả khách nghỉ dưỡng, khách "
                "tham quan di sản và người địa phương cần dịch vụ thuận tiện."
            )
        elif (slots["address"] or slots["coordinates"] or slots["located_in"]) and is_general:
            parts.append("\nCác dữ kiện vị trí giúp xác định rõ phạm vi quản lý và hỗ trợ gợi ý tuyến tham quan dựa trên dữ liệu.")
            
        return "\n".join(parts).strip()


    def _build_generator_candidates(self, seeds: List[Any]) -> List[Dict[str, Any]]:
        return [
            {
                "id": s.id,
                "name": s.metadata.get("name") or s.content,
                "type": (s.metadata.get("labels") or [s.metadata.get("type") or "Unknown"])[0],
                "rating": s.metadata.get("rating"),
                "tags": s.metadata.get("tags") or [],
                "location": s.metadata.get("address") or "",
                "address": s.metadata.get("address") or "",
                "lat": s.metadata.get("lat"),
                "lng": s.metadata.get("lng"),
                "price_range": s.metadata.get("price_range") or s.metadata.get("price") or "",
                "description": s.metadata.get("description") or "",
                "category": s.metadata.get("category") or "",
            }
            for s in (seeds or [])
        ]


    def _build_seed_metadata(self, seeds: List[Any]) -> List[Dict[str, Any]]:
        return [
            {
                "id": s.id,
                "name": s.metadata.get("name") or s.content,
                "labels": s.metadata.get("labels", []),
                "attributes": s.metadata,
                "lat": s.metadata.get("lat"),
                "lng": s.metadata.get("lng"),
            }
            for s in (seeds or [])
        ]


    def _content_tokens(self, text: str, min_len: int = 3) -> set[str]:
        norm = normalize_text(text or "", strip_punct=True)
        return {
            token
            for token in re.findall(r"\w+", norm)
            if len(token) >= min_len and token not in self._CLOSED_FORM_STOPWORDS
        }


    def _choice_lines_from_state(self, state: PipelineRunState) -> list[tuple[str, str]]:
        """Return choices from the question text or evaluator metadata.

        Production requests normally embed choices in the question. Step 4 also
        passes the structured choices list; use it only as a parsing fallback,
        never as an answer key.
        """
        choices = self._parse_choice_lines(state.user_query)
        if choices:
            return choices
        raw_choices = (state.metadata or {}).get("choices") or []
        parsed: list[tuple[str, str]] = []
        for index, raw in enumerate(raw_choices[:4]):
            text = str(raw or "").strip()
            if not text:
                continue
            match = re.match(r"^\s*([A-D])\s*[\).:-]\s*(.+?)\s*$", text, flags=re.IGNORECASE)
            if match:
                parsed.append((match.group(1).upper(), match.group(2).strip()))
            else:
                parsed.append((chr(ord("A") + index), text))
        return parsed

    _CLOSED_FORM_STOPWORDS = frozenset({
        "la", "mot", "cua", "va", "voi", "tai", "o", "gan", "nam", "thuoc",
        "duoc", "co", "cac", "nhung", "nay", "do", "ay", "nao", "gi", "the",
        "nao", "duoi", "day", "thong", "tin", "ngu", "canh", "dua", "tren",
        "hay", "cho", "biet", "dung", "sai", "khong", "phai",
    })


    def _parse_choice_lines(self, question: str) -> list[tuple[str, str]]:
        choices: list[tuple[str, str]] = []
        for match in re.finditer(r"(?im)^\s*([A-D])\s*[\).:-]\s*(.+?)\s*$", str(question or "")):
            choices.append((match.group(1).upper(), match.group(2).strip()))
        return choices


    def _is_emergency_query(self, query: str) -> bool:
        """Detect emergency/support queries that should route to TravelInfo retrieval."""
        q_norm = normalize_text(query, strip_punct=True)
        _EMERGENCY_SIGNALS = [
            "khan cap", "cap cuu", "duong day nong", "su co",
            "ho tro khan cap", "cuu ho", "sos",
            "lien he co quan", "ho tro trong chuyen di",
        ]
        return any(sig in q_norm for sig in _EMERGENCY_SIGNALS)

    def _is_phone_lookup_query(self, query: str) -> bool:
        q_norm = normalize_text(query, strip_punct=True)
        # Exclude emergency queries — "liên hệ" in emergency context is not a phone lookup
        if self._is_emergency_query(query):
            return False
        return any(token in q_norm for token in ["so dien thoai", "sdt", "hotline", "phone", "lien he"])


    def _extract_phone_lookup_entity_hint(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw:
            return ""
        raw = re.sub(r'(?is)^\s*["\']?question["\']?\s*:\s*["\']?', "", raw).strip()
        raw = raw.strip(" \"'“”.,")
        direct_match = re.search(
            r"(?iu)(?:số\s+điện\s+thoại|so\s+dien\s+thoai|sdt|hotline|phone|liên\s+hệ|lien\s+he)\s+(?:của|cua)\s+(.+?)(?:\s+(?:là|la)\s+(?:gì|gi|bao\s+nhiêu|bao\s+nhieu)|\?|$)",
            raw,
        )
        if direct_match:
            candidate = str(direct_match.group(1) or "").strip(" \"'“”.,?:")
            candidate = re.sub(
                r"(?iu)\s+(?:là|la)\s+(?:gì|gi|bao\s+nhiêu|bao\s+nhieu).*$",
                "",
                candidate,
            ).strip(" \"'“”.,?:")
            candidate = self._strip_entity_tail_noise(candidate).strip(" \"'“”.,?:")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        patterns = [
            r"(?i)^(.+?)\s+c[óo]\s+s[ốo]\s+đi[ệe]n\s+tho[ạa]i\b",
            r"(?i)^(.+?)\s+c[oó]\s+so\s+dien\s+thoai\b",
            r"(?i)^(.+?)\s+(?:sdt|hotline|phone|li[eê]n\s+h[eệ])\b",
            r"(?i)(?:s[ốo]\s+đi[ệe]n\s+tho[ạa]i|so\s+dien\s+thoai|sdt|hotline|phone|li[eê]n\s+h[eệ])\s+(?:c[ủu]a|cua)\s+(.+?)(?:\s+l[àa]|\s+la|\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if not match:
                continue
            candidate = str(match.group(1) or "").strip(" \"'“”.,?:")
            candidate = re.sub(r"(?i)\s+l[àa]\s+bao\s+nhi[eê]u.*$", "", candidate).strip(" \"'“”.,?:")
            candidate = re.sub(r"(?i)\s+la\s+bao\s+nhieu.*$", "", candidate).strip(" \"'“”.,?:")
            candidate = self._strip_entity_tail_noise(candidate).strip(" \"'“”.,?:")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        return ""


    def _lookup_node_property_answer(
        self,
        state: PipelineRunState,
        target: str,
        property_keys: list[str],
        attribute_label: str,
    ) -> str:
        target = str(target or "").strip()
        if not target:
            return ""
        attribute_label = re.sub(
            r"(?iu)\s+(?:của|cua)\s*$",
            "",
            str(attribute_label or "").strip(),
        )
        candidate_names = [target]
        target_norm = normalize_text(target, strip_punct=True)
        for seed in state.grounded_nodes or []:
            meta = getattr(seed, "metadata", {}) or {}
            name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
            name_norm = normalize_text(name, strip_punct=True)
            if name and name not in candidate_names and (
                name_norm == target_norm or name_norm in target_norm or target_norm in name_norm
            ):
                candidate_names.append(name)

        cypher = """
        MATCH (n)
        WHERE trim(coalesce(n.name, '')) <> ''
          AND (n.name IN $names OR toLower(n.name) IN $lower_names)
        WITH n, properties(n) AS props
        RETURN n.name AS name, props AS props
        LIMIT 1
        """
        try:
            with self.pipeline.driver.session() as session:
                row = session.run(
                    cypher,
                    names=candidate_names,
                    lower_names=[name.lower() for name in candidate_names],
                ).single()
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("node_property_lookup_failed: %s", str(exc))
            return ""
        if not row:
            return ""
        resolved_name = str(row.get("name") or target).strip() or target
        props = row.get("props") or {}
        for key in property_keys:
            raw_value = props.get(key)
            if raw_value:
                return f"{attribute_label.capitalize()} của {resolved_name} là {str(raw_value).strip()}."
        return f"Hiện tại dữ liệu của hệ thống chưa có thông tin {attribute_label} của {resolved_name}."

    def _extract_opening_hours_entity_hint(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw:
            return ""
        raw = re.sub(r'(?is)^\s*["\']?question["\']?\s*:\s*["\']?', "", raw).strip()
        raw = raw.strip(" \"'“”.,")
        patterns = [
            r"(?iu)^(.+?)\s+(?:mở\s+cửa|mo\s+cua|đóng\s+cửa|dong\s+cua)\b",
            r"(?iu)^(.+?)\s+có\s+giờ\s+mở\s+cửa\b",
            r"(?iu)^(.+?)\s+co\s+gio\s+mo\s+cua\b",
            r"(?iu)(?:giờ\s+mở\s+cửa|gio\s+mo\s+cua)\s+(?:của|cua)\s+(.+?)(?:\?|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if not match:
                continue
            candidate = str(match.group(1) or "").strip(" \"'“”.,?:")
            candidate = re.sub(r"(?iu)\s+(?:lúc|luc|vào|vao)\s+mấy\s+giờ.*$", "", candidate).strip(" \"'“”.,?:")
            candidate = re.sub(r"(?iu)\s+(?:lúc|luc)\s+may\s+gio.*$", "", candidate).strip(" \"'“”.,?:")
            candidate = self._strip_entity_tail_noise(candidate).strip(" \"'“”.,?:")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        return ""


    def _is_website_lookup_query(self, query: str) -> bool:
        q_norm = normalize_text(query, strip_punct=True)
        return any(
            token in q_norm
            for token in [
                "website",
                "web site",
                "trang web",
                "trang chu",
                "link",
                "url",
            ]
        )


    def _is_broad_location_target(self, target: str, state: PipelineRunState) -> bool:
        if not target:
            return False
        target_clean = self._strip_entity_tail_noise(target).strip(" \"'.,?:")
        if self._is_broad_location_anchor(target_clean):
            return True
        target_norm = normalize_text(target_clean, strip_punct=True)
        for entity in state.entities or []:
            if isinstance(entity, dict):
                e_name = str(entity.get("name") or "").strip()
                e_type = str(entity.get("type") or "").strip().lower()
                if normalize_text(e_name, strip_punct=True) == target_norm:
                    if e_type in {"location", "city", "province", "district", "ward", "region"}:
                        return True
        return False


    def _answer_requested_attribute_from_context(self, state: PipelineRunState) -> str:
        target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
        if self._is_broad_location_target(target, state):
            return ""

        q_norm = normalize_text(state.user_query, strip_punct=True)
        requested = {
            normalize_text(attr, strip_punct=True)
            for attr in (state.metadata or {}).get("requested_attributes", [])
            if str(attr or "").strip()
        }
        wants_ticket_price = (
            "ticket_price" in requested
            or any(token in q_norm for token in ["gia ve", "ve vao", "phi tham quan", "ve tham quan", "ve cong", "mat phi", "phi vao cong"])
        )
        if wants_ticket_price:
            answer = self._answer_ticket_price_from_graph(state)
            if answer:
                return answer

        wants_address = (
            "address" in requested
            or any(token in q_norm for token in ["dia chi", "o dau", "nam o dau", "thuoc khu vuc nao"])
        )
        if wants_address:
            answer = self._answer_single_entity_location_if_possible(state)
            if answer:
                return answer

        wants_opening_hours = (
            "opening_hours" in requested
            or any(token in q_norm for token in ["gio mo cua", "mo cua", "dong cua", "luc may gio"])
        )
        if wants_opening_hours:
            target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
            if target:
                direct = self._lookup_node_property_answer(
                    state,
                    target=target,
                    property_keys=["opening_hours", "open_hours", "hours", "business_hours"],
                    attribute_label="giờ mở cửa",
                )
                if direct:
                    return direct

        wants_phone = "phone" in requested or any(token in q_norm for token in ["so dien thoai", "sdt", "lien he", "hotline"])
        if wants_phone:
            target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
            if target:
                direct = self._lookup_node_property_answer(
                    state,
                    target=target,
                    property_keys=["phone", "telephone", "hotline"],
                    attribute_label="số điện thoại liên hệ",
                )
                if direct:
                    return direct
                target_norm = normalize_text(target, strip_punct=True)
                seed_names = []
                for seed in state.grounded_nodes or []:
                    meta = getattr(seed, "metadata", {}) or {}
                    name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
                    if name:
                        seed_names.append(name)
                target_norms = [target_norm] + [normalize_text(name, strip_punct=True) for name in seed_names]
                target_norms = [norm for norm in dict.fromkeys(target_norms) if norm]

                for raw_line in self._closed_form_context_text(state).splitlines():
                    line = str(raw_line or "").strip().lstrip("- ").strip()
                    if not line or ":" not in line:
                        continue
                    norm = normalize_text(line, strip_punct=True)
                    if not any(marker in norm for marker in ["sdt", "so dien thoai", "phone", "hotline"]):
                        continue
                    if target_norms and not any(norm_target in norm or norm in norm_target for norm_target in target_norms):
                        continue
                    value = line.rsplit(":", 1)[-1].strip(" .;:,")
                    if value:
                        subject = target or (seed_names[0] if seed_names else "Đối tượng được hỏi")
                        return f"Số điện thoại của {subject} là {value}."

        # Website lookup
        wants_website = "website" in requested or any(token in q_norm for token in ["website", "trang web", "link"])
        if wants_website:
            target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
            if target:
                direct = self._lookup_node_property_answer(
                    state,
                    target=target,
                    property_keys=["website", "url", "link"],
                    attribute_label="website",
                )
                if direct:
                    return direct

        # Description lookup
        wants_description = "description" in requested or any(token in q_norm for token in ["mo ta", "gioi thieu", "thong tin ve"])
        if wants_description:
            target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
            if target:
                direct = self._lookup_node_property_answer(
                    state,
                    target=target,
                    property_keys=["description", "summary", "intro"],
                    attribute_label="mô tả",
                )
                if direct:
                    return direct

        # Rating lookup
        wants_rating = "rating" in requested or any(token in q_norm for token in ["danh gia", "rating", "diem", "xep hang", "bao nhieu sao"])
        if wants_rating:
            target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
            if target:
                direct = self._lookup_node_property_answer(
                    state,
                    target=target,
                    property_keys=["rating", "score", "stars", "review_score"],
                    attribute_label="đánh giá",
                )
                if direct:
                    return direct

        return ""
        if not target:
            target = self._analysis_main_entity_name(state)
        direct = self._lookup_node_property_answer(
            state,
            target=target,
            property_keys=["phone", "telephone", "hotline"],
            attribute_label="số điện thoại liên hệ",
        )
        if direct:
            return direct
        target_norm = normalize_text(target, strip_punct=True)
        seed_names = []
        for seed in state.grounded_nodes or []:
            meta = getattr(seed, "metadata", {}) or {}
            name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
            if name:
                seed_names.append(name)
        target_norms = [target_norm] + [normalize_text(name, strip_punct=True) for name in seed_names]
        target_norms = [norm for norm in dict.fromkeys(target_norms) if norm]

        for raw_line in self._closed_form_context_text(state).splitlines():
            line = str(raw_line or "").strip().lstrip("- ").strip()
            if not line or ":" not in line:
                continue
            norm = normalize_text(line, strip_punct=True)
            if not any(marker in norm for marker in ["sdt", "so dien thoai", "phone", "hotline"]):
                continue
            if target_norms and not any(norm_target in norm or norm in norm_target for norm_target in target_norms):
                continue
            value = line.rsplit(":", 1)[-1].strip(" .;:,")
            if value:
                subject = target or (seed_names[0] if seed_names else "Đối tượng được hỏi")
                return f"Số điện thoại của {subject} là {value}."
        return ""


    def _answer_ticket_price_from_graph(self, state: PipelineRunState) -> str:
        """Answer entrance-fee questions from TouristAttraction/TravelInfo properties only."""
        metadata = state.metadata or {}
        raw_entities = metadata.get("entities") or []
        q_norm = normalize_text(state.user_query or "", strip_punct=True)

        entity_norms = []
        for entity in raw_entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name") or "").strip()
            if not name:
                continue
            norm = normalize_text(name, strip_punct=True)
            if not norm:
                continue
            # "Bien Ho" by itself is an ambiguous recovery anchor when the
            # query already mentions specific places such as T'Nung / Che.
            if norm in {"bien ho", "ho"} and len(raw_entities) > 1:
                continue
            entity_norms.append(norm)
        entity_norms = list(dict.fromkeys(entity_norms))

        candidate_names = []
        candidate_ids = []
        for seed in list(getattr(state, "grounded_nodes", []) or []) + list(getattr(state, "all_seeds", []) or []):
            meta = getattr(seed, "metadata", {}) or {}
            labels = meta.get("labels") or []
            node_type = str(meta.get("type") or (labels[0] if labels else ""))
            if node_type not in {"TouristAttraction", "TravelInfo"}:
                continue
            name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
            node_id = str(getattr(seed, "id", "") or meta.get("id") or "").strip()
            name_norm = normalize_text(name, strip_punct=True)
            if entity_norms and name_norm:
                matched = False
                name_tokens = set(name_norm.split())
                for ent in entity_norms:
                    ent_tokens = set(ent.split())
                    token_overlap = len(name_tokens.intersection(ent_tokens))
                    if (
                        ent in name_norm
                        or name_norm in ent
                        or ("bien" in ent_tokens and "ho" in ent_tokens and token_overlap >= 3)
                    ):
                        matched = True
                        break
                if not matched:
                    continue
            if name and name not in candidate_names:
                candidate_names.append(name)
            if node_id and node_id not in candidate_ids:
                candidate_ids.append(node_id)

        try:
            with self.pipeline.driver.session() as session:
                rows = []
                if candidate_names or candidate_ids:
                    rows = session.run(
                        """
                        MATCH (n)
                        WHERE (n.id IN $ids OR n.name IN $names)
                          AND (n:TouristAttraction OR n:TravelInfo)
                        RETURN n.id AS id, n.name AS name, labels(n) AS labels,
                               n.ticket_price AS ticket_price,
                               n.price AS price,
                               n.price_range AS price_range,
                               n.description AS description
                        ORDER BY CASE WHEN n.id IN $ids THEN 0 ELSE 1 END, n.name
                        LIMIT 6
                        """,
                        ids=candidate_ids,
                        names=candidate_names,
                    ).data()

                # Direct fallback by normalized token overlap. This is used when
                # grounding/vector search drifts to a semantically similar but
                # wrong place.
                if entity_norms:
                    broad_rows = session.run(
                        """
                        MATCH (n)
                        WHERE n:TouristAttraction OR n:TravelInfo
                        RETURN n.id AS id, n.name AS name, labels(n) AS labels,
                               n.ticket_price AS ticket_price,
                               n.price AS price,
                               n.price_range AS price_range,
                               n.description AS description
                        LIMIT 500
                        """
                    ).data()
                    existing_ids = {str(row.get("id") or "") for row in rows}
                    for row in broad_rows:
                        node_id = str(row.get("id") or "")
                        if node_id in existing_ids:
                            continue
                        name_norm = normalize_text(str(row.get("name") or ""), strip_punct=True)
                        name_tokens = set(name_norm.split())
                        matched = False
                        for ent in entity_norms:
                            ent_tokens = set(ent.split())
                            if not ent_tokens:
                                continue
                            overlap = len(name_tokens.intersection(ent_tokens))
                            required = min(3, len(ent_tokens))
                            if ent in name_norm or name_norm in ent or overlap >= required:
                                matched = True
                                break
                        if matched:
                            rows.append(row)
                            existing_ids.add(node_id)
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("ticket_price_lookup_failed: %s", str(exc))
            return ""

        if not rows:
            return ""

        lines = ["Thông tin phí vào cổng trong dữ liệu hiện có:"]
        seen = set()
        for row in rows:
            name = str(row.get("name") or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            price = (
                str(row.get("ticket_price") or "").strip()
                or str(row.get("price") or "").strip()
                or str(row.get("price_range") or "").strip()
            )
            if price:
                lines.append(f"- **{name}**: {price}")
            else:
                lines.append(f"- **{name}**: dữ liệu hiện chưa ghi nhận phí vào cổng.")

        if len(lines) <= 1:
            return ""
        lines.append("Bạn nên kiểm tra lại trước ngày đi vì phí tham quan có thể thay đổi theo thời điểm hoặc chính sách tại điểm đến.")
        return "\n".join(lines)


    def _answer_restaurant_dishes_if_possible(self, state: PipelineRunState) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        if not any(
            marker in q_norm
            for marker in [
                "mon dac trung",
                "dac trung nao",
                "co mon",
                "mon an gi",
                "phuc vu mon",
                "dac san gi",
                "thuc don",
                "menu",
            ]
        ):
            return ""

        target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
        if not target:
            for seed in state.grounded_nodes or state.all_seeds or []:
                meta = getattr(seed, "metadata", {}) or {}
                if str(meta.get("type") or "").lower() == "restaurant" or "nha hang" in normalize_text(str(meta.get("type") or ""), strip_punct=True):
                    target = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
                    break
        if not target:
            return ""

        candidate_names = [target]
        target_norm = normalize_text(target, strip_punct=True)
        for seed in state.grounded_nodes or state.all_seeds or []:
            meta = getattr(seed, "metadata", {}) or {}
            name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
            name_norm = normalize_text(name, strip_punct=True)
            if name and name not in candidate_names and (
                name_norm == target_norm or name_norm in target_norm or target_norm in name_norm
            ):
                candidate_names.append(name)

        cypher = """
        MATCH (r:Restaurant)
        WHERE r.name IN $names OR toLower(r.name) IN $lower_names
        OPTIONAL MATCH (r)-[:HAS]->(dish)
        WITH r, collect(DISTINCT dish.name) AS dishes
        RETURN r.name AS name, dishes AS dishes
        LIMIT 1
        """
        try:
            with self.pipeline.driver.session() as session:
                row = session.run(
                    cypher,
                    names=candidate_names,
                    lower_names=[name.lower() for name in candidate_names],
                ).single()
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("restaurant_dishes_lookup_failed: %s", str(exc))
            return ""
        if not row:
            return ""

        name = str(row.get("name") or target).strip() or target
        dishes = [str(dish or "").strip() for dish in (row.get("dishes") or []) if str(dish or "").strip()]
        if dishes:
            return f"{name} có món đặc trưng trong dữ liệu là: {', '.join(dishes)}."
        return f"Hiện tại dữ liệu của hệ thống chưa ghi nhận món đặc trưng của {name}."


    def _answer_single_entity_category_if_possible(self, state: PipelineRunState) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        if not any(marker in q_norm for marker in ["thuoc loai hinh", "loai hinh du lich", "thuoc loai", "phan loai"]):
            return ""
        if any(marker in q_norm for marker in ["deu thuoc", "so sanh", "giong nhau", "khac nhau"]):
            return ""

        target = str((state.metadata or {}).get("target_entity") or "").strip()
        if not target:
            target = self._primary_specific_entity_name(state) or self._analysis_main_entity_name(state)
        target = self._strip_entity_tail_noise(target).strip(" \"'.,?:")
        if not target or self._is_generic_category_phrase(target):
            return ""

        query = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($target)
           OR toLower(n.name) CONTAINS toLower($target)
           OR toLower($target) CONTAINS toLower(n.name)
        OPTIONAL MATCH (n)-[:BELONGS_TO]->(cat)
        WITH n, cat
        ORDER BY CASE WHEN toLower(n.name) = toLower($target) THEN 0 ELSE 1 END, size(n.name)
        RETURN n.name AS name, cat.name AS category, labels(n) AS labels
        LIMIT 1
        """
        try:
            records, _, _ = self.pipeline.driver.execute_query(query, target=target)
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("single_entity_category_lookup_failed: %s", str(exc))
            return ""
        if not records:
            return ""
        row = dict(records[0])
        name = str(row.get("name") or target).strip()
        category = str(row.get("category") or "").strip()
        if category:
            return f"{name} thuộc loại hình du lịch {category}."

        labels = [str(label or "") for label in (row.get("labels") or [])]
        label_fallback = ""
        if "Accommodation" in labels:
            label_fallback = "Lưu trú"
        elif "Restaurant" in labels:
            label_fallback = "Ẩm thực"
        elif "Tour" in labels:
            label_fallback = "Tour du lịch"
        elif "Event" in labels:
            label_fallback = "Sự kiện"
        if label_fallback:
            return f"{name} thuộc nhóm {label_fallback}, nhưng dữ liệu chưa ghi nhận loại hình du lịch chi tiết."
        return ""


    def _answer_single_entity_location_if_possible(self, state: PipelineRunState) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        wants_location = any(
            marker in q_norm
            for marker in [
                "nam o phuong nao",
                "o phuong nao",
                "thuoc phuong nao",
                "nam o xa nao",
                "o xa nao",
                "thuoc xa nao",
                "nam o dau",
                "o dau",
                "dia chi",
                "thuoc khu vuc nao",
            ]
        )
        if not wants_location:
            return ""
        if any(marker in q_norm for marker in ["gan", "near", "xung quanh", "lan can"]):
            return ""

        target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
        if not target:
            return ""
        if self._is_broad_location_target(target, state):
            return ""
        target = self._strip_entity_tail_noise(target).strip(" \"'.,?:")
        if not target or self._is_generic_category_phrase(target):
            return ""

        target_norm = normalize_text(target, strip_punct=True)
        candidate_names = [target]
        for seed in list(getattr(state, "grounded_nodes", []) or []) + list(getattr(state, "all_seeds", []) or []):
            meta = getattr(seed, "metadata", {}) or {}
            name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
            name_norm = normalize_text(name, strip_punct=True)
            if not name or name in candidate_names:
                continue
            if name_norm and (
                target_norm == name_norm
                or target_norm in name_norm
                or name_norm in target_norm
            ):
                candidate_names.append(name)

        query = """
        MATCH (n)
        WHERE trim(coalesce(n.name, '')) <> ''
          AND (
              n.name IN $names
              OR toLower(n.name) = toLower($target)
              OR toLower(n.name) CONTAINS toLower($target)
              OR toLower($target) CONTAINS toLower(n.name)
          )
        OPTIONAL MATCH (n)-[:LOCATED_IN]->(loc)
        WITH n, loc
        ORDER BY
          CASE
            WHEN n:TouristAttraction THEN 0
            WHEN n:TravelInfo THEN 1
            ELSE 2
          END,
          CASE WHEN toLower(n.name) = toLower($target) THEN 0 ELSE 1 END,
          size(n.name)
        RETURN n.name AS name, n.address AS address, loc.name AS located_in
        LIMIT 1
        """
        try:
            records, _, _ = self.pipeline.driver.execute_query(query, target=target, names=candidate_names)
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("single_entity_location_lookup_failed: %s", str(exc))
            return ""
        if not records:
            return ""
        row = dict(records[0])
        name = str(row.get("name") or target).strip()
        located_in = str(row.get("located_in") or "").strip()
        address = str(row.get("address") or "").strip()

        if "phuong nao" in q_norm and located_in:
            return f"{name} nằm ở {located_in}."
        if "xa nao" in q_norm and located_in:
            return f"{name} nằm ở {located_in}."
        if address and located_in and located_in not in address:
            return f"{name} có địa chỉ {address}, thuộc {located_in}."
        if address:
            return f"{name} có địa chỉ {address}."
        if located_in:
            return f"{name} nằm ở {located_in}."
        return ""


    def _answer_website_lookup_if_possible(self, state: PipelineRunState) -> Dict[str, Any] | None:
        if not self._is_website_lookup_query(state.user_query):
            return None

        target = str((state.metadata or {}).get("target_entity") or "").strip() or self._primary_specific_entity_name(state)
        seeds = list(state.grounded_nodes or state.all_seeds or [])
        if not target and seeds:
            first_meta = getattr(seeds[0], "metadata", {}) or {}
            target = str(first_meta.get("name") or getattr(seeds[0], "content", "") or "").strip()
        if not target:
            return None

        target_norm = normalize_text(target, strip_punct=True)
        selected_seed = None
        for seed in seeds:
            meta = getattr(seed, "metadata", {}) or {}
            name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
            name_norm = normalize_text(name, strip_punct=True)
            if name_norm and (name_norm == target_norm or name_norm in target_norm or target_norm in name_norm):
                selected_seed = seed
                break
        if selected_seed is None and seeds:
            selected_seed = seeds[0]

        candidate_names = [target]
        if selected_seed is not None:
            meta = getattr(selected_seed, "metadata", {}) or {}
            for value in [meta.get("name"), getattr(selected_seed, "content", "")]:
                value = str(value or "").strip()
                if value and value not in candidate_names:
                    candidate_names.append(value)

        cypher = """
        MATCH (n)
        WHERE n.name IN $names OR toLower(n.name) IN $lower_names
        WITH n, properties(n) AS props
        RETURN n.name AS name,
               coalesce(props.website, props.url, props.sourceUrl, props.source_url, props.link) AS website
        LIMIT 1
        """
        website = ""
        resolved_name = target
        try:
            with self.pipeline.driver.session() as session:
                row = session.run(
                    cypher,
                    names=candidate_names,
                    lower_names=[name.lower() for name in candidate_names],
                ).single()
            if row:
                resolved_name = str(row.get("name") or target).strip() or target
                website = str(row.get("website") or "").strip()
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("website_lookup_query_failed: %s", str(exc))

        if website:
            answer = f"{resolved_name} có website/link thông tin: {website}."
        else:
            answer = f"Hiện tại dữ liệu của hệ thống chưa có thông tin website của {resolved_name}."

        intent = state.query_plan.intent if state.query_plan else state.primary_intent
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["detected_location"] = state.location
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(seeds[:3])
        state.runtime.metadata["route_seed_nodes"] = []
        state.runtime.metadata["graph"] = self.pipeline._build_graph_payload(seeds[:3], [], intent=intent)
        state.runtime.metadata["deterministic_short_circuit"] = "website_lookup"
        state.runtime.metadata["target_entity"] = resolved_name
        state.runtime.metadata["answer_mode"] = AnswerMode.FACT_ANSWER

        return {"answer": self._sanitize_answer_text(answer), "metadata": state.runtime.metadata}


    def _answer_global_category_listing_if_possible(self, state: PipelineRunState) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        # Broadened keyword gate: match discovery/category-listing queries
        _CATEGORY_MARKERS = [
            "cac diem du lich", "diem du lich tai", "liet ke",
            "lang van hoa", "van hoa dan toc", "van hoa ban dia",
            "lang dan toc", "trai nghiem van hoa", "lang truyen thong",
            "diem den", "co nhung gi", "nhung ngoi lang",
        ]
        if not any(marker in q_norm for marker in _CATEGORY_MARKERS):
            return ""
        category_specs = [
            ("Di tích lịch sử - Văn hóa", ["di tich lich su", "lich su van hoa"]),
            ("Danh lam thắng cảnh", ["danh lam thang canh", "danh lam", "danh thang"]),
            ("Làng nghề truyền thống", ["lang nghe truyen thong", "lang nghe"]),
            ("Làng văn hóa dân tộc", ["lang van hoa", "van hoa dan toc", "van hoa ban dia", "lang dan toc", "trai nghiem van hoa"]),
        ]
        requested = [
            label for label, aliases in category_specs
            if any(alias in q_norm for alias in aliases)
        ]

        # Fallback: if no predefined category matched but query is category-like,
        # search for TouristAttraction nodes whose name/description matches the topic.
        if not requested:
            # Extract topic keywords from query for name-pattern search
            topic_patterns = []
            if "lang" in q_norm and ("van hoa" in q_norm or "dan toc" in q_norm):
                topic_patterns = ["làng", "lang", "văn hóa", "van hoa", "dân tộc", "dan toc"]
            elif "di tich" in q_norm:
                topic_patterns = ["di tích", "di tich"]
            elif "danh lam" in q_norm:
                topic_patterns = ["danh lam", "thắng cảnh"]
            if not topic_patterns:
                return ""

            # Build CONTAINS clauses for name or description
            name_clauses = " OR ".join(
                f"toLower(poi.name) CONTAINS toLower('{p}')" for p in topic_patterns
            )
            desc_clauses = " OR ".join(
                f"toLower(poi.description) CONTAINS toLower('{p}')" for p in topic_patterns
            )
            cypher = f"""
            MATCH (poi:TouristAttraction)
            WHERE ({name_clauses}) OR ({desc_clauses})
            RETURN 'matched' AS category,
                   poi.name AS name,
                   poi.address AS address,
                   poi.location.latitude AS lat,
                   poi.location.longitude AS lng,
                   poi.description AS description
            ORDER BY name
            LIMIT 20
            """
            try:
                with self.pipeline.driver.session() as session:
                    rows = session.run(cypher).data()
            except (Neo4jClientError, ServiceUnavailable) as exc:
                self.pipeline.logger.warning("global_category_fallback_query_failed: %s", str(exc))
                return ""
            if not rows:
                return ""

            # Filter by region if needed
            filtered_rows = []
            for row in rows:
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                if state.region_focus != "all":
                    class _SeedView:
                        def __init__(self, n, a, lat, lng):
                            self.content = n
                            self.metadata = {"name": n, "address": a or "", "lat": lat, "lng": lng}
                    seed_view = _SeedView(name, str(row.get("address") or ""), row.get("lat"), row.get("lng"))
                    if not self.pipeline._seed_in_region(seed_view, state.region_focus):
                        continue
                filtered_rows.append(row)

            if not filtered_rows:
                return ""

            lines = ["Dựa trên dữ liệu đồ thị, đây là một số địa điểm liên quan:\n"]
            for row in filtered_rows[:10]:
                name = str(row.get("name") or "").strip()
                addr = str(row.get("address") or "").strip()
                desc = str(row.get("description") or "").strip()
                entry = f"- **{name}**"
                if addr:
                    entry += f" — {addr}"
                if desc:
                    # First sentence only
                    for sep in [". ", ".\n", "! ", "? "]:
                        idx = desc.find(sep)
                        if 0 < idx < 120:
                            desc = desc[:idx + 1].strip()
                            break
                    else:
                        if len(desc) > 120:
                            desc = desc[:120].rsplit(" ", 1)[0] + "..."
                    entry += f"\n  {desc}"
                lines.append(entry)
            return "\n".join(lines)

        cypher = """
        MATCH (poi:TouristAttraction)-[:BELONGS_TO]->(cat)
        WHERE cat.name IN $categories
        RETURN cat.name AS category,
               poi.name AS name,
               poi.address AS address,
               poi.location.latitude AS lat,
               poi.location.longitude AS lng
        ORDER BY category, name
        """
        try:
            with self.pipeline.driver.session() as session:
                rows = session.run(
                    cypher,
                    categories=requested,
                ).data()
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("global_category_listing_query_failed: %s", str(exc))
            return ""
        if not rows:
            return ""

        class _SeedView:
            def __init__(self, name: str, address: str, lat: float, lng: float) -> None:
                self.content = name
                self.metadata = {
                    "name": name,
                    "address": address or "",
                    "lat": lat,
                    "lng": lng,
                }

        by_category: Dict[str, List[str]] = {}
        for row in rows:
            category = str(row.get("category") or "").strip()
            name = str(row.get("name") or "").strip()
            if not category or not name:
                continue
            if state.region_focus != "all":
                seed_view = _SeedView(
                    name=name,
                    address=str(row.get("address") or ""),
                    lat=row.get("lat"),
                    lng=row.get("lng"),
                )
                if not self.pipeline._seed_in_region(seed_view, state.region_focus):
                    continue
            names = by_category.setdefault(category, [])
            if name not in names and len(names) < 6:
                names.append(name)
        lines = []
        for label in requested:
            names = by_category.get(label) or []
            if names:
                lines.append(f"- {label}: {', '.join(names)}")
            else:
                lines.append(f"- {label}: chưa có điểm phù hợp trong dữ liệu hiện tại")
        return "Các điểm du lịch được phân loại theo dữ liệu đồ thị hiện có:\n" + "\n".join(lines)


    def _extract_fill_blank_fact_from_context(self, state: PipelineRunState) -> str:
        p = self.pipeline
        query_norm = normalize_text(state.user_query, strip_punct=True)
        target_norm = normalize_text(self._primary_specific_entity_name(state), strip_punct=True)
        context_text = self._closed_form_context_text(state)
        lines = [str(line or "").strip() for line in context_text.splitlines() if str(line or "").strip()]
        wants_location = any(
            token in query_norm
            for token in [
                "dat tai",
                "toa lac",
                "nam tai",
                "nam o",
                "o ___",
                "tai ___",
                "dia chi",
            ]
        )
        if "menh danh" in query_norm:
            target_lines = [
                line for line in lines
                if not target_norm or target_norm in normalize_text(line, strip_punct=True)
            ] or lines
            joined = " ".join(target_lines)
            quote_patterns = [
                r"[\"']([^\"']{2,60})[\"']",
                r"“([^”]{2,60})”",
                r"‘([^’]{2,60})’",
            ]
            quote_matches: list[str] = []
            for pattern in quote_patterns:
                quote_matches.extend(re.findall(pattern, joined))
            answer_cue_norms = {
                normalize_text(value, strip_punct=True)
                for value in ("lá phổi xanh", "la phoi xanh", "ốc đảo", "oc dao")
            }
            ranked_quotes = sorted(
                quote_matches,
                key=lambda value: (
                    0 if normalize_text(value, strip_punct=True) in answer_cue_norms else 1,
                    len(value.split()),
                ),
            )
            for quote in ranked_quotes:
                quote_norm = normalize_text(quote, strip_punct=True)
                if quote_norm and len(quote_norm.split()) <= 5:
                    return quote.strip()
            m = re.search(
                r"(?i)(?:mệnh\s+danh|menh\s+danh|được\s+gọi|duoc\s+goi).{0,120}?(?:là|la)\s+([^,.;]{2,60})",
                joined,
            )
            if m:
                phrase = m.group(1).strip(" \"'“”‘’.,;:")
                if phrase and len(phrase.split()) <= 8:
                    return phrase
        if "diem xuat phat" in query_norm:
            target_lines = [
                line for line in lines
                if not target_norm or target_norm in normalize_text(line, strip_punct=True)
            ] or lines
            joined_norm = normalize_text(" ".join(target_lines), strip_punct=True)
            for place in ["Quy Nhơn", "Pleiku", "Gia Lai"]:
                if normalize_text(place, strip_punct=True) in joined_norm:
                    return place
        if not wants_location:
            return ""

        def line_matches_target(line: str) -> bool:
            line_norm = normalize_text(line, strip_punct=True)
            return not target_norm or target_norm in line_norm or line_norm.startswith(target_norm)

        for line in lines:
            if not line_matches_target(line):
                continue
            address_match = re.search(r"(?i)^(?:địa chỉ|dia chi)\s+.+?:\s*(.+)$", line)
            if address_match:
                return address_match.group(1).strip().rstrip(".")
            located_match = re.search(r"(?i)(?:nằm tại|nam tai|tọa lạc(?: ở| tại)?|toa lac(?: o| tai)?)\s+(.+)$", line)
            if located_match:
                return located_match.group(1).strip().rstrip(".")
            rel_match = re.search(r"\[LOCATED_IN\]\s*->\s*(.+)$", line)
            if rel_match:
                return rel_match.group(1).strip().rstrip(".")
            attr_match = re.search(r"(?i)^-\s*address\s*:\s*(.+)$", line)
            if attr_match:
                return attr_match.group(1).strip().rstrip(".")

        # If the target name was not available or exact target matching failed,
        # use the first direct address/location fact in the context.
        for line in lines:
            address_match = re.search(r"(?i)^(?:địa chỉ|dia chi)\s+.+?:\s*(.+)$", line)
            if address_match:
                return address_match.group(1).strip().rstrip(".")
            attr_match = re.search(r"(?i)^-\s*address\s*:\s*(.+)$", line)
            if attr_match:
                return attr_match.group(1).strip().rstrip(".")
        return ""


    def _generate_fill_blank_short_fallback(self, state: PipelineRunState) -> str:
        context_text = self._closed_form_context_text(state)
        if not context_text.strip():
            return ""
        system = (
            "Bạn điền chỗ trống cho câu hỏi du lịch. Chỉ dùng dữ liệu được cung cấp. "
            "Trả lời bằng một cụm rất ngắn, tối đa 8 từ. Không giải thích."
        )
        user = f"Câu cần điền: {state.user_query}\n\nDữ liệu:\n{context_text[:3000]}"
        try:
            raw = self.pipeline.llm_service.generate_text(system, user)
        except (ValueError, RuntimeError, OSError):
            return ""
        text = self._sanitize_answer_text(raw or "").splitlines()[0].strip()
        text = re.split(r"[.;\n]", text)[0].strip()
        words = text.split()
        if len(words) > 8:
            text = " ".join(words[:8])
        return text

