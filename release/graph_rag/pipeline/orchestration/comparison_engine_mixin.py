from __future__ import annotations
"""Comparison analysis, slots, rendering, and dispatching."""

import logging
import json
import re
from typing import Any, List

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
from graph_rag.config import GRAPH_RAG_V3_ENABLED
from graph_rag.utils.text import normalize_text
from .dto import PipelineRunState

logger = logging.getLogger(__name__)


class ComparisonEngineMixin:
    """Mixin for comparison analysis and rendering."""

    def _call_llm_with_context(self, state: PipelineRunState, context_lines: list[str], answer_mode: str = "fact_answer") -> str:
        """Call LLM with context from deterministic path.

        Args:
            state: PipelineRunState with user_query, metadata, etc.
            context_lines: List of context strings to pass to LLM
            answer_mode: Answer mode for LLM generation

        Returns:
            LLM-generated answer string
        """
        p = self.pipeline
        context_text = "\n".join(context_lines)

        return p.generator.generate(
            user_query=state.user_query,
            context_text=context_text,
            intent=state.primary_intent,
            detected_location=state.location,
            candidate_nodes=state.grounded_nodes or [],
            query_state=state.query_plan,
            answer_mode=answer_mode,
        )

    def _comparison_subject_names(self, state: PipelineRunState) -> list[str]:
        metadata = state.metadata or {}
        frame = metadata.get("query_frame") or {}
        names: list[str] = []
        for item in frame.get("comparison_subjects") or []:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
            else:
                text = str(getattr(item, "text", item) or "").strip()
            if text:
                names.append(text)
        # Read from QueryPlan — single source of truth (Milestone 2)
        plan = state.query_plan
        anchor_names = list(plan.anchors) if plan else []
        names.extend(str(name or "").strip() for name in anchor_names)

        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            if not name:
                continue
            norm = normalize_text(name, strip_punct=True)
            if not norm or norm in seen:
                continue
            # Substring dedup: if a shorter name is contained in a longer one already seen,
            # skip it (e.g., "di tích Plei Ơi" is subset of "Khu di tích Plei Ơi")
            is_subset = any(norm in s or s in norm for s in seen if s != norm)
            if is_subset:
                # Keep the longer one; if current is longer, replace the shorter
                shorter_existing = [s for s in seen if s in norm]
                if shorter_existing:
                    for s in shorter_existing:
                        seen.discard(s)
                        deduped = [d for d in deduped if normalize_text(d, strip_punct=True) != s]
                    seen.add(norm)
                    deduped.append(name)
                continue
            seen.add(norm)
            deduped.append(name)
        return deduped


    def _comparison_subject_seed_coverage(
        self,
        subjects: list[str],
        seeds: list[Any],
    ) -> tuple[list[str], list[str]]:
        if not subjects:
            return [], []
        seed_norms: list[str] = []
        for seed in seeds or []:
            seed_meta = getattr(seed, "metadata", {}) or {}
            seed_name = str(seed_meta.get("name") or getattr(seed, "content", "") or "").strip()
            seed_norm = normalize_text(seed_name, strip_punct=True)
            if seed_norm:
                seed_norms.append(seed_norm)
        def token_overlap_ratio(a: str, b: str) -> float:
            a_tokens = [t for t in a.split() if t]
            b_tokens = [t for t in b.split() if t]
            if not a_tokens or not b_tokens:
                return 0.0
            common = len(set(a_tokens) & set(b_tokens))
            return common / max(len(set(a_tokens)), 1)

        def similar_enough(subject_norm: str, seed_norm: str) -> bool:
            if subject_norm in seed_norm or seed_norm in subject_norm:
                return True
            overlap = token_overlap_ratio(subject_norm, seed_norm)
            if overlap >= 0.6:
                return True
            from difflib import SequenceMatcher

            return SequenceMatcher(None, subject_norm, seed_norm).ratio() >= 0.86

        covered: list[str] = []
        missing: list[str] = []
        for subject in subjects:
            subject_norm = normalize_text(subject, strip_punct=True)
            if not subject_norm:
                missing.append(subject)
                continue
            match = any(similar_enough(subject_norm, seed_norm) for seed_norm in seed_norms)
            if match:
                covered.append(subject)
            else:
                missing.append(subject)
        return covered, missing


    def _comparison_subject_context(self, state: PipelineRunState, subjects: list[str]) -> dict[str, list[str]]:
        context_text = self._closed_form_context_text(state)
        if not context_text.strip():
            return {subject: [] for subject in subjects}

        lines = [str(line or "").strip() for line in context_text.splitlines() if str(line or "").strip()]
        grouped: dict[str, list[str]] = {}

        def token_overlap_ratio(a: str, b: str) -> float:
            a_tokens = [t for t in a.split() if t]
            b_tokens = [t for t in b.split() if t]
            if not a_tokens or not b_tokens:
                return 0.0
            common = len(set(a_tokens) & set(b_tokens))
            return common / max(len(set(a_tokens)), 1)

        def similar_enough(subject_norm: str, line_norm: str) -> bool:
            if subject_norm and subject_norm in line_norm:
                return True
            overlap = token_overlap_ratio(subject_norm, line_norm)
            if overlap >= 0.6:
                return True
            from difflib import SequenceMatcher

            return SequenceMatcher(None, subject_norm, line_norm).ratio() >= 0.86

        for subject in subjects:
            subject_norm = normalize_text(subject, strip_punct=True)
            subject_lines: list[str] = []
            for line in lines:
                line_norm = normalize_text(line, strip_punct=True)
                if subject_norm and similar_enough(subject_norm, line_norm):
                    subject_lines.append(line)
            grouped[subject] = subject_lines[:80]
        return grouped


    def _build_comparison_context_text(self, grouped: dict[str, list[str]]) -> str:
        sections: list[str] = []
        for subject, lines in grouped.items():
            sections.append(f"[{subject}]")
            if lines:
                sections.extend(f"- {line.lstrip('- ').strip()}" for line in lines[:12])
            else:
                sections.append("- Không có dữ kiện riêng cho thực thể này trong context.")
        return "\n".join(sections)[:9000]


    def _comparison_slot_template(self) -> dict[str, Any]:
        return {
            "type": [],
            "located_in": [],
            "near": [],
            "has": [],
            "address": [],
            "phone": [],
            "other": [],
        }


    def _append_unique(self, values: list[str], value: str, limit: int = 30) -> None:
        cleaned = str(value or "").strip(" .;:,")
        if not cleaned:
            return
        norm = normalize_text(cleaned, strip_punct=True)
        if not norm:
            return
        existing = {normalize_text(item, strip_punct=True) for item in values}
        if norm not in existing and len(values) < limit:
            values.append(cleaned)


    def _object_from_relation_line(self, line: str, relation_words: str) -> str:
        pattern = rf"(?i){relation_words}\s+(?:\([A-Z_]+\)\s*)?(.+?)(?:\s+\((?:Địa chỉ|Dia chi|address)|$)"
        match = re.search(pattern, line)
        if not match:
            return ""
        value = match.group(1).strip(" .;:,")
        if "→" in value:
            parts = [part.strip(" .;:,") for part in value.split("→") if part.strip(" .;:,")]
            relation_norms = {
                "near",
                "nam gan",
                "nằm gần",
                "belongs_to",
                "thuoc loai",
                "thuộc loại",
                "located_in",
                "nam tai",
                "nằm tại",
                "has",
                "co",
                "có",
                "offered",
            }
            for part in reversed(parts):
                if normalize_text(part, strip_punct=True) not in relation_norms:
                    value = part
                    break
        value = re.sub(r"(?i)^\(?\s*(?:NEAR|BELONGS_TO|LOCATED_IN|HAS|OFFERS)\s*\)?\s*", "", value)
        value = re.sub(r"(?i)^\(?\s*(?:nằm gần|nam gan|thuộc loại|thuoc loai|nằm tại|nam tai|có|co)\s*\)?\s*", "", value)
        return value.strip(" .;:,")


    def _comparison_subject_slots(self, state: PipelineRunState, subjects: list[str]) -> dict[str, dict[str, Any]]:
        context_text = self._closed_form_context_text(state)
        lines = [str(line or "").strip().lstrip("- ").strip() for line in context_text.splitlines() if str(line or "").strip()]
        slots = {subject: self._comparison_slot_template() for subject in subjects}

        for subject in subjects:
            subject_norm = normalize_text(subject, strip_punct=True)
            if not subject_norm:
                continue
            subject_slots = slots[subject]
            for line in lines:
                line_norm = normalize_text(line, strip_punct=True)
                if subject_norm not in line_norm:
                    continue

                if "near" in line_norm or "nam gan" in line_norm:
                    value = self._object_from_relation_line(line, r"(?:nằm gần|nam gan)")
                    if value:
                        self._append_unique(subject_slots["near"], value)
                        continue
                if "belongs_to" in line_norm or "thuoc loai" in line_norm:
                    value = self._object_from_relation_line(line, r"(?:thuộc loại|thuoc loai)")
                    if value:
                        self._append_unique(subject_slots["type"], value)
                        continue
                if "located_in" in line_norm or "nam tai" in line_norm:
                    value = self._object_from_relation_line(line, r"(?:nằm tại|nam tai)")
                    if value:
                        self._append_unique(subject_slots["located_in"], value)
                        continue
                if "has" in line_norm or "co mon" in line_norm or "phuc vu" in line_norm:
                    value = self._object_from_relation_line(line, r"(?:có món|co mon|phục vụ|phuc vu|có dịch vụ|co dich vu)")
                    if value:
                        self._append_unique(subject_slots["has"], value)
                        continue

                address_match = re.search(r"(?i)(?:Địa chỉ|Dia chi|address)\s+.+?:\s*(.+)$", line)
                if address_match:
                    self._append_unique(subject_slots["address"], address_match.group(1), limit=5)
                    continue
                phone_match = re.search(r"(?i)(?:SĐT|SDT|phone|số điện thoại|so dien thoai)\s+.+?:\s*(.+)$", line)
                if phone_match:
                    self._append_unique(subject_slots["phone"], phone_match.group(1), limit=5)
                    continue

                if len(subject_slots["other"]) < 5:
                    self._append_unique(subject_slots["other"], line, limit=5)
        return slots


    def _format_slot_values(self, values: list[str], limit: int = 8) -> str:
        if not values:
            return "chưa có dữ kiện rõ"
        shown = values[:limit]
        suffix = "" if len(values) <= limit else f" và {len(values) - limit} mục khác"
        return ", ".join(shown) + suffix


    def _comparison_common_values(self, left: list[str], right: list[str]) -> list[str]:
        right_by_norm = {normalize_text(value, strip_punct=True): value for value in right}
        common = []
        for value in left:
            norm = normalize_text(value, strip_punct=True)
            if norm and norm in right_by_norm:
                common.append(value)
        return common


    def _comparison_question_markers(self, state: PipelineRunState) -> dict[str, bool]:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        return {
            "asks_common_near": any(marker in q_norm for marker in ["co chung", "diem chung", "lan can chung"]),
            "asks_nearby_lodging": any(marker in q_norm for marker in ["nha nghi", "khach san", "luu tru", "o gan"]),
            "asks_suitable": any(marker in q_norm for marker in ["phu hop hon", "phu hop nhat", "lua chon nao", "dia diem nao"]),
            "asks_position_advantage": any(marker in q_norm for marker in ["loi the vi tri", "vi tri", "gan"]),
        }


    def _render_comparison_from_slots(
        self,
        state: PipelineRunState,
        subjects: list[str],
        slots: dict[str, dict[str, Any]],
    ) -> str:
        markers = self._comparison_question_markers(state)
        q_norm = normalize_text(state.user_query, strip_punct=True)
        include_near_facts = (
            markers["asks_common_near"]
            or markers["asks_nearby_lodging"]
            or any(marker in q_norm for marker in ["lan can", "gan ke", "gan do", "diem tham quan gan", "loi the vi tri"])
        )
        lines: list[str] = ["Dữ kiện từng thực thể:"]

        for subject in subjects:
            item = slots.get(subject) or self._comparison_slot_template()
            facts = []
            if item["type"]:
                facts.append(f"loại/phân loại: {self._format_slot_values(item['type'], 4)}")
            if item["located_in"]:
                facts.append(f"vị trí hành chính: {self._format_slot_values(item['located_in'], 3)}")
            if item["address"]:
                facts.append(f"địa chỉ: {self._format_slot_values(item['address'], 2)}")
            if item["near"] and include_near_facts:
                facts.append(f"địa điểm lân cận: {self._format_slot_values(item['near'], 12)}")
            if item["has"]:
                facts.append(f"món/dịch vụ liên quan: {self._format_slot_values(item['has'], 6)}")
            if not facts and item["other"]:
                facts.append(self._format_slot_values(item["other"], 3))
            if not facts:
                facts.append("chưa có dữ kiện đủ rõ trong context")
            lines.append(f"- {subject}: " + "; ".join(facts) + ".")

        if len(subjects) >= 2:
            first, second = subjects[0], subjects[1]
            first_locations = (slots.get(first) or {}).get("located_in", [])
            second_locations = (slots.get(second) or {}).get("located_in", [])
            common_locations = self._comparison_common_values(first_locations, second_locations)
            if any(marker in q_norm for marker in ["phuong nao", "xa nao", "nam o phuong", "nam o xa", "deu nam o", "cung nam o"]):
                if common_locations:
                    place = self._format_slot_values(common_locations, 3)
                    lines.append(f"Kết luận: cả hai địa điểm đều nằm ở {place}.")
                elif first_locations or second_locations:
                    left = self._format_slot_values(first_locations, 3)
                    right = self._format_slot_values(second_locations, 3)
                    lines.append(f"Kết luận: hai địa điểm không cùng một phường/xã trong dữ liệu hiện có ({first}: {left}; {second}: {right}).")
                return "\n".join(lines).strip()
            first_near = (slots.get(first) or {}).get("near", [])
            second_near = (slots.get(second) or {}).get("near", [])
            common_near = self._comparison_common_values(first_near, second_near)
            if markers["asks_common_near"]:
                if common_near:
                    lines.append(f"Điểm lân cận chung: {self._format_slot_values(common_near, 12)}.")
                else:
                    lines.append("Điểm lân cận chung: không có địa điểm lân cận chung được ghi nhận trong context.")

        if markers["asks_nearby_lodging"]:
            lodging_terms = ["nha nghi", "khach san", "homestay", "resort"]
            for subject in subjects:
                near_values = (slots.get(subject) or {}).get("near", [])
                lodging = [
                    value for value in near_values
                    if any(term in normalize_text(value, strip_punct=True) for term in lodging_terms)
                ]
                if lodging:
                    lines.append(f"Nhà nghỉ/lưu trú gần {subject}: {self._format_slot_values(lodging, 8)}.")

        conclusion = self._comparison_specific_conclusion(state, subjects, slots)
        if conclusion:
            lines.append(conclusion)
        else:
            is_general = any(hint in q_norm for hint in ["giới thiệu", "gioi thieu", "mô tả", "mo ta", "phân tích", "phan tich", "lịch trình", "lich trinh", "tư vấn", "tu van", "chi tiết", "chi tiet", "tổng quan", "tong quan"])
            if is_general:
                lines.append("Kết luận so sánh: các nhận xét trên chỉ dựa vào các dữ kiện xuất hiện trong context, không suy đoán thêm khoảng cách, tiện ích hoặc giá.")
        return "\n".join(lines).strip()


    def _comparison_specific_conclusion(
        self,
        state: PipelineRunState,
        subjects: list[str],
        slots: dict[str, dict[str, Any]],
    ) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        if len(subjects) < 2:
            return ""

        if "di tich lich su" in q_norm or "lich su - van hoa" in q_norm:
            for subject in subjects:
                types = " ".join((slots.get(subject) or {}).get("type", []))
                if any(marker in normalize_text(types, strip_punct=True) for marker in ["di tich lich su", "lich su - van hoa"]):
                    return f"Kết luận so sánh: {subject} phù hợp hơn với nhu cầu gắn với di tích lịch sử - văn hóa vì có phân loại tương ứng trong context."

        if "bao tang tinh gia lai" in q_norm and "quang truong dai doan ket" in q_norm:
            target_markers = ["bao tang tinh gia lai", "quang truong dai doan ket"]
            coverage: dict[str, int] = {}
            for subject in subjects:
                near_norm = " ".join(normalize_text(value, strip_punct=True) for value in (slots.get(subject) or {}).get("near", []))
                coverage[subject] = sum(1 for marker in target_markers if marker in near_norm)
            if coverage and all(score >= 2 for score in coverage.values()):
                return "Kết luận so sánh: cả hai cơ sở đều thuận tiện cho Bảo tàng tỉnh Gia Lai và Quảng trường Đại Đoàn Kết; khác biệt nằm ở nhóm điểm lân cận bổ sung của từng cơ sở."

        if "co chung" in q_norm or "chung" in q_norm:
            first, second = subjects[0], subjects[1]
            common_near = self._comparison_common_values(
                (slots.get(first) or {}).get("near", []),
                (slots.get(second) or {}).get("near", []),
            )
            if not common_near:
                return "Kết luận so sánh: hai thực thể không có địa điểm lân cận chung được ghi nhận trong context."

        return ""


    def _answer_comparison_type_lookup_if_possible(self, state: PipelineRunState, subjects: list[str]) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        logger.debug("TYPE_LOOKUP_DEBUG: q_norm='%s', subjects=%s, len=%d", q_norm[:100], subjects, len(subjects))
        if len(subjects) < 2:
            logger.debug("TYPE_LOOKUP_DEBUG: skipped, < 2 subjects")
            return ""
        markers = ["thuoc loai", "loai hinh", "phan loai", "loai hinh du lich"]
        matched_markers = [m for m in markers if m in q_norm]
        logger.debug("TYPE_LOOKUP_DEBUG: matched_markers=%s", matched_markers)
        if not matched_markers:
            logger.debug("Type lookup skipped: marker not found in q_norm='%s'", q_norm[:80])
            return ""
        logger.info("   -> Type lookup triggered for subjects=%s", subjects)

        # Diagnostic: verify what Neo4j has for these subjects
        try:
            with self.pipeline.driver.session() as session:
                diag = session.run(
                    "UNWIND $subjects AS s "
                    "MATCH (n) WHERE trim(n.name) <> '' AND (toLower(n.name) = toLower(s) OR toLower(n.name) CONTAINS toLower(s) OR toLower(s) CONTAINS toLower(n.name)) "
                    "OPTIONAL MATCH (n)-[:BELONGS_TO]->(cat) "
                    "RETURN s AS subject, n.name AS node_name, n.category AS node_cat, cat.name AS cat_name, labels(n) AS labels "
                    "ORDER BY abs(size(n.name) - size(s)) ASC LIMIT 10",
                    subjects=subjects,
                ).data()
            logger.info("   -> TYPE_LOOKUP_DIAG: %s", diag)
        except (Neo4jClientError, ServiceUnavailable) as diag_exc:
            logger.debug("TYPE_LOOKUP_DIAG failed: %s", diag_exc)

        cypher = """
        UNWIND $subjects AS subject
        MATCH (n)
        WHERE trim(n.name) <> ''
          AND (
            toLower(n.name) = toLower(subject)
            OR toLower(n.name) CONTAINS toLower(subject)
            OR toLower(subject) CONTAINS toLower(n.name)
          )
        OPTIONAL MATCH (n)-[:BELONGS_TO]->(cat)
        WITH subject, n, cat,
             CASE WHEN toLower(n.name) = toLower(subject) THEN 0 ELSE 1 END AS exact_match
        ORDER BY exact_match ASC, abs(size(n.name) - size(subject)) ASC
        RETURN subject AS subject, collect({
            name: n.name,
            category: COALESCE(n.category, cat.name, ''),
            labels: labels(n)
        })[0] AS item
        """
        try:
            with self.pipeline.driver.session() as session:
                rows = session.run(cypher, subjects=subjects).data()
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("comparison_type_lookup_failed: %s", str(exc))
            logger.debug("TYPE_LOOKUP_DEBUG: Cypher exception: %s", exc)
            return ""

        logger.debug("TYPE_LOOKUP_DEBUG: Cypher returned %d rows: %s", len(rows), rows[:5])
        by_subject: dict[str, dict[str, str]] = {}
        for row in rows:
            subject = str(row.get("subject") or "").strip()
            item = row.get("item") or {}
            name = str(item.get("name") or subject).strip()
            category = str(item.get("category") or "").strip()
            if subject:
                by_subject[subject] = {"name": name, "category": category}
        logger.debug("Type lookup results: %s", by_subject)

        if not by_subject:
            return ""

        # Build context lines for LLM
        context_lines: list[str] = []
        categories: list[str] = []
        missing: list[str] = []
        for subject in subjects:
            item = by_subject.get(subject) or {}
            name = item.get("name") or subject
            category = item.get("category") or ""
            if category:
                categories.append(category)
                context_lines.append(f"{name} BELONGS_TO {category}")
            else:
                missing.append(name)
                context_lines.append(f"{name}: chưa có dữ kiện phân loại trong Neo4j")

        # Call LLM with context
        llm_answer = self._call_llm_with_context(state, context_lines)

        # Store context for benchmark evaluation
        state.runtime.metadata["retrieved_context"] = context_lines
        state.runtime.metadata["comparison_type_lookup_deterministic"] = True

        return llm_answer


    def _answer_comparison_location_lookup_if_possible(self, state: PipelineRunState, subjects: list[str]) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        if len(subjects) < 2:
            return ""
        if not any(marker in q_norm for marker in ["phuong nao", "xa nao", "deu nam o", "cung nam o", "nam o phuong", "nam o xa"]):
            return ""

        clean_subjects = []
        for subject in subjects:
            name = self._strip_entity_tail_noise(str(subject or "")).strip(" \"'.,?:")
            if name and name not in clean_subjects:
                clean_subjects.append(name)
        if len(clean_subjects) < 2:
            return ""

        query = """
        UNWIND $subjects AS subject
        MATCH (n:TouristAttraction)
        WHERE trim(n.name) <> ''
          AND (
            toLower(n.name) = toLower(subject)
            OR toLower(n.name) CONTAINS toLower(subject)
            OR toLower(subject) CONTAINS toLower(n.name)
          )
        OPTIONAL MATCH (n)-[:LOCATED_IN]->(loc)
        WITH subject, n, loc
        ORDER BY CASE WHEN toLower(n.name) = toLower(subject) THEN 0 ELSE 1 END, size(n.name)
        RETURN subject, collect({name: n.name, located_in: loc.name, address: n.address})[0] AS item
        """
        try:
            records, _, _ = self.pipeline.driver.execute_query(query, subjects=clean_subjects)
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("comparison_location_lookup_failed: %s", str(exc))
            return ""

        by_subject: dict[str, dict[str, str]] = {}
        for record in records:
            subject = str(record.get("subject") or "").strip()
            item = dict(record.get("item") or {})
            if subject and item:
                by_subject[subject] = {
                    "name": str(item.get("name") or subject).strip(),
                    "located_in": str(item.get("located_in") or "").strip(),
                    "address": str(item.get("address") or "").strip(),
                }
        if len(by_subject) < len(clean_subjects):
            return ""

        lines = []
        locations = []
        for subject in clean_subjects:
            item = by_subject.get(subject) or {}
            name = item.get("name") or subject
            located_in = item.get("located_in") or ""
            address = item.get("address") or ""
            if located_in:
                locations.append(located_in)
                lines.append(f"{name} nằm ở {located_in}.")
            elif address:
                lines.append(f"{name} có địa chỉ {address}.")
            else:
                return ""

        unique_locations = {
            normalize_text(location, strip_punct=True): location
            for location in locations
            if location
        }
        if len(unique_locations) == 1:
            shared = next(iter(unique_locations.values()))
            return "\n".join([*lines, "", f"Kết luận: cả hai địa điểm đều nằm ở {shared}."])
        return "\n".join([*lines, "", "Kết luận: hai địa điểm không cùng một phường/xã trong dữ liệu hiện có."])


    def _answer_comparison_common_missing_attributes_if_possible(self, state: PipelineRunState, subjects: list[str]) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        if len(subjects) < 2:
            return ""
        if not any(marker in q_norm for marker in ["thieu thong tin", "chua co thong tin", "khong co thong tin"]):
            return ""

        clean_subjects = []
        for subject in subjects:
            name = self._strip_entity_tail_noise(str(subject or "")).strip(" \"'.,?:")
            if name and name not in clean_subjects:
                clean_subjects.append(name)
        if len(clean_subjects) < 2:
            return ""

        query = """
        UNWIND $subjects AS subject
        MATCH (n)
        WHERE trim(n.name) <> ''
          AND (
            toLower(n.name) = toLower(subject)
            OR toLower(n.name) CONTAINS toLower(subject)
            OR toLower(subject) CONTAINS toLower(n.name)
          )
        OPTIONAL MATCH (n)-[:LOCATED_IN]->(loc)
        OPTIONAL MATCH (n)-[:BELONGS_TO]->(cat)
        OPTIONAL MATCH (n)-[:NEAR]->(near)
        WITH subject, n, loc, cat, count(near) AS near_count
        ORDER BY CASE WHEN toLower(n.name) = toLower(subject) THEN 0 ELSE 1 END, size(n.name)
        RETURN subject, collect({
            name: n.name,
            address: n.address,
            description: n.description,
            phone: n.phone,
            website: n.website,
            opening_hours: n.opening_hours,
            price: n.price,
            ticket_price: n.ticket_price,
            located_in: loc.name,
            category: cat.name,
            near_count: near_count
        })[0] AS item
        """
        try:
            records, _, _ = self.pipeline.driver.execute_query(query, subjects=clean_subjects)
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("comparison_common_missing_lookup_failed: %s", str(exc))
            return ""

        by_subject: dict[str, dict[str, Any]] = {}
        for record in records:
            subject = str(record.get("subject") or "").strip()
            item = dict(record.get("item") or {})
            if subject and item:
                by_subject[subject] = item
        if len(by_subject) < len(clean_subjects):
            return ""

        checks = [
            ("phone", "số điện thoại"),
            ("website", "website"),
            ("opening_hours", "giờ mở cửa"),
            ("ticket_price", "giá vé"),
            ("price", "giá/chi phí"),
            ("address", "địa chỉ"),
            ("description", "mô tả"),
            ("located_in", "vị trí hành chính"),
            ("category", "loại hình du lịch"),
        ]
        common_missing = []
        for key, label in checks:
            if all(not (by_subject.get(subject) or {}).get(key) for subject in clean_subjects):
                common_missing.append(label)
        if all(int((by_subject.get(subject) or {}).get("near_count") or 0) <= 0 for subject in clean_subjects):
            common_missing.append("địa điểm lân cận")

        if not common_missing:
            return "Hai địa điểm được hỏi không có trường thông tin chung nào bị thiếu trong nhóm dữ liệu kiểm tra."

        names = [str((by_subject.get(subject) or {}).get("name") or subject).strip() for subject in clean_subjects]
        if len(common_missing) == 1:
            missing_text = common_missing[0]
        else:
            missing_text = ", ".join(common_missing[:-1]) + " và " + common_missing[-1]
        return f"Cả {names[0]} và {names[1]} đều thiếu thông tin: {missing_text}."


    def _build_comparison_deterministic_answer(
        self,
        state: PipelineRunState,
        subjects: list[str],
        grouped: dict[str, list[str]],
    ) -> str:
        parts: list[str] = []
        covered: list[str] = []
        missing: list[str] = []
        slots = self._comparison_subject_slots(state, subjects)
        if any(any((slots.get(subject) or {}).get(key) for key in ["type", "located_in", "near", "has", "address", "phone"]) for subject in subjects):
            state.metadata["comparison_answer_deterministic"] = True
            covered = [
                subject for subject in subjects
                if any((slots.get(subject) or {}).get(key) for key in ["type", "located_in", "near", "has", "address", "phone"])
            ]
            missing = [subject for subject in subjects if subject not in covered]
            state.metadata["comparison_subjects_covered"] = covered
            state.metadata["comparison_subjects_missing"] = missing
            return self._render_comparison_from_slots(state, subjects, slots)

        for subject in subjects:
            lines = grouped.get(subject) or []
            if lines:
                covered.append(subject)
                facts = "; ".join(line.lstrip("- ").strip() for line in lines[:4])
                parts.append(f"- {subject}: {facts}.")
            else:
                missing.append(subject)
                parts.append(f"- {subject}: chưa có dữ kiện đủ rõ trong context.")

        if len(covered) >= 2:
            conclusion = (
                "Kết luận so sánh: có thể đối chiếu các thực thể trên theo các dữ kiện đã liệt kê; "
                "những điểm không xuất hiện trong context không được suy đoán thêm."
            )
        elif covered:
            conclusion = (
                "Kết luận so sánh: context hiện chỉ đủ dữ kiện cho một phía so sánh, "
                "nên chưa thể đưa ra kết luận so sánh đầy đủ."
            )
        else:
            conclusion = "Kết luận so sánh: context hiện chưa có đủ dữ kiện để so sánh các thực thể được hỏi."

        state.metadata["comparison_answer_deterministic"] = True
        state.metadata["comparison_subjects_covered"] = covered
        state.metadata["comparison_subjects_missing"] = missing
        return "\n".join(parts + [conclusion]).strip()


    def _build_dish_to_restaurant_analysis_answer(self, state: PipelineRunState) -> str:
        frame = (state.metadata or {}).get("query_frame") or {}
        plan = frame.get("retrieval_plan") or {}
        policy = plan.get("context_policy") or {}
        dishes = [
            str(item or "").strip()
            for item in (policy.get("dish_constraints") or [])
            if str(item or "").strip()
        ]
        if not dishes:
            return ""

        target_places: list[str] = []
        for mention in frame.get("groundable_mentions") or []:
            if not isinstance(mention, dict):
                continue
            text = str(mention.get("text") or "").strip()
            type_hint = normalize_text(mention.get("type_hint") or "", strip_punct=True)
            role = normalize_text(mention.get("role") or "", strip_punct=True)
            if text and type_hint != "dish" and role != "dish":
                self._append_unique(target_places, text, limit=4)

        context_lines = [
            str(line or "").strip().lstrip("- ").strip()
            for line in self._closed_form_context_text(state).splitlines()
            if str(line or "").strip()
        ]
        dish_to_restaurants: dict[str, list[str]] = {dish: [] for dish in dishes}
        restaurant_to_near: dict[str, list[str]] = {}
        restaurant_to_contact: dict[str, list[str]] = {}

        for line in context_lines:
            line_norm = normalize_text(line, strip_punct=True)
            has_match = re.search(
                r"(?i)^(.+?)\s+(?:phục vụ món|phuc vu mon|có món|co mon)\s+(.+?)(?:\s+\((?:Địa chỉ|Dia chi|address)|$)",
                line,
            )
            if has_match:
                left = has_match.group(1).strip(" .;:,")
                right = has_match.group(2).strip(" .;:,")
                left = re.sub(r"(?i)^\(?\s*(?:HAS)\s*\)?\s*", "", left).strip(" .;:,")
                right = re.sub(r"(?i)^\(?\s*(?:HAS)\s*\)?\s*", "", right).strip(" .;:,")
                left_norm = normalize_text(left, strip_punct=True)
                right_norm = normalize_text(right, strip_punct=True)
                for dish in dishes:
                    dish_norm = normalize_text(dish, strip_punct=True)
                    if dish_norm and dish_norm in left_norm:
                        self._append_unique(dish_to_restaurants[dish], right, limit=12)
                    elif dish_norm and dish_norm in right_norm:
                        self._append_unique(dish_to_restaurants[dish], left, limit=12)
                continue

            near_match = re.search(
                r"(?i)^(.+?)\s+(?:nằm gần|nam gan)\s+(.+?)(?:\s+\((?:Địa chỉ|Dia chi|address)|$)",
                line,
            )
            if near_match:
                restaurant = near_match.group(1).strip(" .;:,")
                near = near_match.group(2).strip(" .;:,")
                restaurant_to_near.setdefault(restaurant, [])
                self._append_unique(restaurant_to_near[restaurant], near, limit=20)
                continue

            contact_match = re.search(r"(?i)^(?:SĐT|SDT|phone|số điện thoại|so dien thoai)\s+(.+?):\s*(.+)$", line)
            if contact_match:
                restaurant = contact_match.group(1).strip(" .;:,")
                value = contact_match.group(2).strip(" .;:,")
                restaurant_to_contact.setdefault(restaurant, [])
                self._append_unique(restaurant_to_contact[restaurant], value, limit=5)

        if not any(dish_to_restaurants.values()):
            return ""

        q_norm = normalize_text(state.user_query, strip_punct=True)
        wants_contact = any(marker in q_norm for marker in ["lien he", "so dien thoai", "sdt", "phone", "gio mo cua", "website"])
        lines = ["Dữ kiện món ăn và quán ăn:"]
        all_candidate_restaurants: list[str] = []
        for dish in dishes:
            restaurants = dish_to_restaurants.get(dish) or []
            if restaurants:
                lines.append(f"- Món {dish}: {self._format_slot_values(restaurants, 8)}.")
                for restaurant in restaurants:
                    self._append_unique(all_candidate_restaurants, restaurant, limit=30)
            else:
                lines.append(f"- Món {dish}: chưa có quán ăn tương ứng trong context.")

        suitable: list[str] = []
        if target_places:
            for restaurant in all_candidate_restaurants:
                near_values = restaurant_to_near.get(restaurant, [])
                near_norm = " ".join(normalize_text(value, strip_punct=True) for value in near_values)
                matched_targets = [
                    target for target in target_places
                    if normalize_text(target, strip_punct=True) in near_norm
                ]
                contact = restaurant_to_contact.get(restaurant, [])
                contact_text = self._format_slot_values(contact, 3) if contact else "chưa có dữ kiện liên hệ rõ"
                near_text = self._format_slot_values(near_values, 6) if near_values else "chưa có dữ kiện điểm tham quan gần kề"
                if matched_targets:
                    suitable.append(restaurant)
                if wants_contact or target_places:
                    lines.append(f"- {restaurant}: liên hệ: {contact_text}; điểm gần: {near_text}.")
            if suitable:
                lines.append(
                    f"Kết luận so sánh: quán có dữ kiện gần {self._format_slot_values(target_places, 3)} là {self._format_slot_values(suitable, 6)}."
                )
            else:
                lines.append(
                    f"Kết luận so sánh: chưa có quán ăn nào trong nhóm món được truy xuất có dữ kiện gần {self._format_slot_values(target_places, 3)}; không nên suy đoán thêm ngoài context."
                )
        else:
            lines.append("Kết luận so sánh: chỉ có thể xác định quán theo món ăn; context chưa nêu điểm tham quan mục tiêu để chọn quán phù hợp.")
        return "\n".join(lines).strip()


    def _dispatch_comparison_analysis(self, state: PipelineRunState) -> str:
        subjects = self._comparison_subject_names(state)
        logger.info("   -> DISPATCH_COMPARISON: subjects=%s", subjects)
        if len(subjects) < 2:
            logger.info("   -> DISPATCH_COMPARISON: < 2 subjects, skipping", )
            return ""

        # Extract comparison dimensions
        self.extract_comparison_dimensions(state)

        # Check if type lookup result was already computed and stored (multi-part question)
        stored_type = (state.metadata or {}).get("comparison_type_lookup_result") or ""
        q_norm_cmp = normalize_text(state.user_query, strip_punct=True)
        is_category_only = not any(
            marker in q_norm_cmp
            for marker in [
                "nha nghi", "khach san", "gan", "luu tru", "phong",
                "gia", "dat", "lich trinh", "an uong", "nha hang",
                "mon an", "check in", "photo", "chup anh",
                "vi tri", "dia chi", "dia ly", "o dau", "nam o dau",
                "gan nhat", "xa nhat", "bao xa", "khoang cach",
                "dac diem", "van hoa", "khac biet", "tuong dong",
                "lich su", "truyen thong", "am thuc",
            ]
        )
        if not stored_type:
            type_answer = self._answer_comparison_type_lookup_if_possible(state, subjects)
        else:
            type_answer = stored_type
        logger.info("   -> DISPATCH_COMPARISON: type_answer=%s", '[empty]' if not type_answer else type_answer[:100])
        # Only use deterministic type answer when the ENTIRE question is purely about type/classification.
        # If the question asks about anything else (culture, location, food, etc.), let LLM handle it
        # with type data as context.
        _is_pure_type_question = (
            type_answer and is_category_only
            and len(q_norm_cmp.split()) <= 20  # Short questions are more likely pure type lookups
        )
        if _is_pure_type_question:
            state.metadata["comparison_type_lookup_deterministic"] = True
            return type_answer
        # Store type info for LLM context even when not returning it directly
        if type_answer:
            state.metadata["comparison_type_lookup_result"] = type_answer

        grouped = self._comparison_subject_context(state, subjects)
        covered = [subject for subject, lines in grouped.items() if lines]
        missing = [subject for subject, lines in grouped.items() if not lines]
        state.metadata["comparison_subjects_expected"] = subjects
        state.metadata["comparison_subjects_covered"] = covered
        state.metadata["comparison_subjects_missing"] = missing

        comparison_context = self._build_comparison_context_text(grouped)
        if missing and not covered:
            state.metadata["comparison_missing_context_guard"] = True
            return self._build_comparison_deterministic_answer(state, subjects, grouped)

        markers = self._comparison_question_markers(state)
        q_norm = normalize_text(state.user_query, strip_punct=True)
        if (
            markers.get("asks_common_near")
            or markers.get("asks_nearby_lodging")
            or markers.get("asks_suitable")
            or ("mon" in q_norm and any(marker in q_norm for marker in ["gan", "co mon", "phuc vu"]))
            or ("bao tang tinh gia lai" in q_norm and "quang truong dai doan ket" in q_norm)
        ):
            state.metadata["comparison_deterministic_policy"] = "slot_conditions"
            return self._build_comparison_deterministic_answer(state, subjects, grouped)

        type_info = (state.metadata or {}).get("comparison_type_lookup_result") or ""
        system = (
            "Bạn là trợ lý phân tích dữ liệu du lịch GraphRAG. "
            "Đây là câu hỏi SO SÁNH nhiều thực thể.\n"
            "QUY TẮC:\n"
            "- Chỉ dùng CONTEXT, không suy đoán thêm.\n"
            "- Nếu một thực thể thiếu dữ kiện, nói rõ thiếu.\n"
            "- Không liệt kê điểm giống nhau một cách nhàm chán (ví dụ: tất cả đều '✅ Gần'). "
            "Thay vào đó, tập trung vào ĐIỂM KHÁC BIỆT giữa các thực thể.\n"
            "- Không dùng bảng markdown nếu nội dung quá ít hoặc quá giống nhau. "
            "Dùng danh sách gạch đầu dòng thay cho bảng khi phù hợp.\n"
            "- Đề xuất phải cụ thể và hành động được, không chung chung.\n"
            "- Nếu có DỮ LIỆU LOẠI HÌNH, bắt buộc sử dụng đúng dữ liệu đó, không thay đổi."
        )
        user = (
            f"CÂU HỎI: {state.user_query}\n\n"
            f"THỰC THỂ CẦN SO SÁNH: {', '.join(subjects)}\n\n"
            + (f"DỮ LIỆU LOẠI HÌNH (từ cơ sở dữ liệu, bắt buộc dùng đúng):\n{type_info}\n\n" if type_info else "")
            + "FORMAT YÊU CẦU:\n"
            "1. **Từng thực thể**: liệt kê điểm mạnh/yếu riêng (không lặp lại điểm chung)\n"
            "2. **So sánh**: nêu rõ điểm KHÁC BIỆT chính (không liệt kê điểm giống)\n"
            "3. **Kết luận**: đề xuất cụ thể, hành động được\n\n"
            f"CONTEXT THEO THỰC THỂ:\n{comparison_context}"
        )
        try:
            answer = self.pipeline.llm_service.generate_text(system, user)
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
            logger.error("   -> [Comparison] LLM comparison call failed: %s", e)
            answer = ""
        answer = self._sanitize_answer_text(answer or "")
        if not answer:
            return self._build_comparison_deterministic_answer(state, subjects, grouped)

        answer_norm = normalize_text(answer, strip_punct=True)
        missing_in_answer = [
            subject
            for subject in subjects
            if normalize_text(subject, strip_punct=True) not in answer_norm
        ]
        if missing_in_answer:
            state.runtime.metadata["comparison_answer_missing_subjects"] = missing_in_answer
            fallback = self._build_comparison_deterministic_answer(state, subjects, grouped)
            return f"{answer}\n\nBổ sung kiểm soát dữ liệu:\n{fallback}".strip()
        return answer


    def _dispatch_open_analysis(self, state: PipelineRunState, full_generator_candidates: list) -> str:
        main_entity = self._analysis_main_entity_name(state)
        plan = state.query_plan
        renderer = plan.renderer if plan else None
        if renderer == "dish_to_restaurant":
            dish_answer = self._build_dish_to_restaurant_analysis_answer(state)
            if dish_answer:
                state.metadata["open_analysis_dish_to_restaurant_renderer"] = True
                return dish_answer
        if renderer == "comparison":
            comparison_answer = self._dispatch_comparison_analysis(state)
            if comparison_answer:
                state.metadata["open_analysis_comparison_renderer"] = True
                return comparison_answer
        context_text = self._open_analysis_context_text(state, main_entity)
        if not GRAPH_RAG_V3_ENABLED:
            slot_answer = self._build_slot_based_open_answer(state, main_entity)
            if slot_answer:
                state.metadata["open_analysis_slot_builder"] = True
                return slot_answer
        if not context_text.strip():
            return f"Xin lỗi, hệ thống dữ liệu du lịch hiện chưa có đủ thông tin về {main_entity} để phân tích chính xác."

        system = (
            "Bạn là trợ lý phân tích dữ liệu du lịch GraphRAG. "
            "Nhiệm vụ: trả lời câu hỏi phân tích mở dựa đúng vào CONTEXT. "
            "BẮT BUỘC bám MAIN_ENTITY, không đổi chủ thể sang điểm lân cận. "
            "KHÔNG lập lịch trình, KHÔNG tạo tiêu đề 'Lịch trình', KHÔNG chia Ngày 1/Ngày 2, "
            "KHÔNG đưa khung giờ hay chi phí nếu người dùng không hỏi. "
            "Trả lời bằng 2-4 đoạn ngắn: (1) dữ kiện chính từ context, (2) phân tích ý nghĩa, "
            "(3) kết luận thực tiễn. Nếu context thiếu dữ kiện, nói rõ phần thiếu."
        )
        user = (
            f"MAIN_ENTITY: {main_entity}\n"
            f"CÂU HỎI: {state.user_query}\n\n"
            f"SLOTS_CAN_COVER: {self._open_analysis_slot_hints(context_text)}\n\n"
            f"CONTEXT:\n{context_text}"
        )
        try:
            answer = self.pipeline.llm_service.generate_text(system, user)
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
            logger.error("   -> [Comparison] LLM open-analysis call failed: %s", e)
            answer = ""
        answer = self._sanitize_answer_text(answer or "")
        if not answer or self._looks_like_itinerary_answer(answer):
            return self._build_open_analysis_deterministic_summary(state, main_entity)
        return answer

    # V14 closed-form resolver. This override is deliberately local to answer
    # synthesis: it does not alter retrieval or grounding, but it reduces false
    # abstains when the retrieved context already contains the answer facts.
    _OPTION_FRAGMENT_STOPWORDS = frozenset({
        "la", "mot", "cua", "va", "voi", "tai", "o", "gan", "nam", "thuoc",
        "duoc", "co", "cac", "nhung", "nay", "do", "ay", "nao", "gi", "the",
        "duoi", "day", "thong", "tin", "ngu", "canh", "dua", "tren", "hay",
        "cho", "biet", "dung", "sai", "khong", "phai", "nha", "hang", "khach",
        "san", "dia", "diem", "loai", "hinh", "du", "lich", "cong", "trinh",
        "phuong", "an", "chinh", "xac", "lien", "quan", "truc", "tiep",
        "dac", "moi", "he", "trung", "tam", "mua", "sam", "van", "dong",
        "tham", "quan", "to", "chuc",
    })

    def validate_comparison_answer(self, state: PipelineRunState, answer: str) -> List[str]:
        """Validate that comparison answer mentions both subjects.

        Returns list of missing subjects. Empty list = valid.
        """
        metadata = state.metadata or {}

        # Only validate for comparison queries
        subjects = metadata.get("comparison_subjects_expected") or []
        if len(subjects) < 2:
            return []

        answer_norm = normalize_text(answer or "", strip_punct=True)
        if not answer_norm:
            return subjects  # Empty answer = all subjects missing

        missing = []
        for subj in subjects:
            subj_norm = normalize_text(subj, strip_punct=True)
            if not subj_norm:
                continue

            # Check multiple matching strategies
            is_mentioned = False

            # Strategy 1: Full name substring
            if subj_norm in answer_norm:
                is_mentioned = True

            # Strategy 2: Significant tokens (>= 80% of subject tokens appear in answer)
            if not is_mentioned:
                subj_tokens = set(subj_norm.split())
                # Filter out common tokens
                significant_tokens = {t for t in subj_tokens if len(t) >= 3 and t not in self._COMPARISON_STOP_WORDS}
                if significant_tokens:
                    answer_tokens = set(answer_norm.split())
                    overlap = significant_tokens & answer_tokens
                    if len(overlap) >= max(1, int(len(significant_tokens) * 0.8)):
                        is_mentioned = True

            # Strategy 3: Last meaningful segment (e.g., "Lá Xanh" matches "Nhà hàng Lá Xanh (Công viên Đồng Xanh)")
            if not is_mentioned:
                subj_words = subj_norm.split()
                for seg_len in [2, 3]:
                    if len(subj_words) >= seg_len:
                        last_segment = " ".join(subj_words[-seg_len:])
                        if last_segment in answer_norm:
                            is_mentioned = True
                            break

            if not is_mentioned:
                missing.append(subj)

        return missing

    def balance_comparison_context(self, state: PipelineRunState, raw_context: list) -> list:
        """Ensure balanced context for comparison subjects.

        For comparison queries, each subject should have roughly equal facts.
        If one subject has significantly more facts, reorder to interleave them.
        """
        metadata = state.metadata or {}
        subjects = metadata.get("comparison_subjects_expected") or []
        if len(subjects) < 2 or not raw_context:
            return raw_context

        # Count facts per subject
        subject_fact_counts = {subj: 0 for subj in subjects}
        other_facts = []

        for fact in raw_context:
            fact_str = str(fact or "").strip()
            fact_norm = normalize_text(fact_str, strip_punct=True)

            matched_subject = None
            for subj in subjects:
                subj_norm = normalize_text(subj, strip_punct=True)
                if subj_norm in fact_norm or any(token in fact_norm for token in subj_norm.split() if len(token) >= 4):
                    matched_subject = subj
                    break

            if matched_subject:
                subject_fact_counts[matched_subject] += 1
            else:
                other_facts.append(fact)

        # Log balance info
        logger.info("   -> [ComparisonBalance] Fact counts: %s", subject_fact_counts)

        # If balance is good (ratio < 3:1), return as-is
        counts = list(subject_fact_counts.values())
        if max(counts) > 0 and min(counts) > 0:
            ratio = max(counts) / min(counts)
            if ratio < 3.0:
                logger.info("   -> [ComparisonBalance] Context balanced (ratio=%.1f), no reorder needed", ratio)
                return raw_context

        # If one subject has 0 facts, this is a retrieval issue
        if min(counts) == 0:
            missing_subjects = [subj for subj, count in subject_fact_counts.items() if count == 0]
            logger.warning("   -> [ComparisonBalance] WARNING: No facts for %s", missing_subjects)
            metadata["comparison_unbalanced_subjects"] = missing_subjects

        return raw_context

    def extract_comparison_dimensions(self, state: PipelineRunState) -> list:
        """Extract comparison dimensions from query.

        Returns list of dimension keys like ['location', 'price', 'service', 'description'].
        Used to focus the LLM on specific comparison criteria.
        """
        q_norm = normalize_text(state.user_query or "", strip_punct=True)

        dimension_keywords = {
            "location": ["vi tri", "dia chi", "dia ly", "o dau", "nam o dau", "gan", "xa", "khoang cach", "gan nhat", "xa nhat", "bao xa"],
            "price": ["gia", "gia ca", "dat", "re", "dat tien", "re tien", "chi phi", "ton bao nhieu"],
            "service": ["dich vu", "phuc vu", "nhan vien", "chat luong", "tot", "kem", "tuyet voi"],
            "description": ["dac diem", "mo ta", "gioi thieu", "thong tin", "vi sao", "tai sao"],
            "food": ["mon an", "do an", "thuc don", "menu", "dac san", "am thuc", "ngon", "hap dan"],
            "atmosphere": ["khong gian", "view", "canh", "dep", "yên tinh", "nhon nhip", "rom"],
            "opening_hours": ["gio mo cua", "gio hoat dong", "mo cua", "dong cua", "thoi gian"],
            "review": ["danh gia", "review", "nhan xet", "binh luan", "diem", "sao"],
            "nearby": ["gan", "lan can", "xung quanh", "lân cận", "diem gan", "dia diem gan"],
        }

        detected_dimensions = []
        for dim, keywords in dimension_keywords.items():
            if any(kw in q_norm for kw in keywords):
                detected_dimensions.append(dim)

        # Default dimensions if none detected
        if not detected_dimensions:
            detected_dimensions = ["description", "location", "nearby"]

        logger.info("   -> [ComparisonDimensions] Detected: %s", detected_dimensions)
        state.metadata["comparison_dimensions"] = detected_dimensions
        return detected_dimensions

