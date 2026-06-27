"""Context-based answer builder — structured parsing + LLM fallback answers."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from graph_rag.core import keywords

logger = logging.getLogger(__name__)

from .dto import PipelineRunState


class DeterministicContextMixin:
    """Mixin for context-based deterministic answers (structured parsing, LLM calls)."""

    _RELATION_RE = re.compile(r"^-\s*(.+?)\s+\[(\w+)\]\s*->\s*(.+?)\s*$")
    _ATTRIBUTE_RE = re.compile(r"^-\s*(\w+):\s*(.+)$")
    _HEADER_RE = re.compile(r"\*\*THỰC THỂ CHÍNH:\*\*\s*(.+?)\s*\(Loại:\s*(.+?)\)")

    _CATEGORY_KEYWORDS = keywords.CATEGORY_KEYWORDS

    def _call_llm_with_context(
        self, state: PipelineRunState, context_lines: List[str], answer_mode: str = "fact_answer"
    ) -> str:
        """Call LLM with context from deterministic path."""
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

    def _category_matches_line(self, category_label: str, context_line: str) -> bool:
        if not category_label or not context_line:
            return True
        kw_list = self._CATEGORY_KEYWORDS.get(category_label, [])
        if not kw_list:
            return True
        from graph_rag.utils.text import normalize_text
        line_norm = normalize_text(context_line, strip_punct=True)
        return any(kw in line_norm for kw in kw_list)

    def _parse_raw_context(self, raw_context: List[str]) -> Dict[str, Any]:
        """Parse raw_context lines into structured data."""
        relations: Dict[str, List[Tuple[str, str]]] = {}
        attributes: Dict[str, str] = {}
        entity_name = ""
        entity_type = ""
        description = ""

        for line in raw_context:
            line_str = str(line or "").strip()
            if not line_str:
                continue

            header_match = self._HEADER_RE.search(line_str)
            if header_match:
                entity_name = header_match.group(1).strip()
                entity_type = header_match.group(2).strip()
                continue

            rel_match = self._RELATION_RE.match(line_str)
            if rel_match:
                left = rel_match.group(1).strip()
                rel_type = rel_match.group(2).strip().upper()
                right = rel_match.group(3).strip()
                relations.setdefault(rel_type, []).append((left, right))
                continue

            attr_match = self._ATTRIBUTE_RE.match(line_str)
            if attr_match:
                key = attr_match.group(1).strip().lower()
                value = attr_match.group(2).strip()
                attributes[key] = value
                continue

            if len(line_str) > 50 and not line_str.startswith("-"):
                if not description:
                    description = line_str

        return {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "relations": relations,
            "attributes": attributes,
            "description": description,
        }

    def _build_context_based_answer(
        self,
        state: PipelineRunState,
        target_category: str = "",
    ) -> str | None:
        """Build a factual answer from raw_context structured data.

        Used as fallback when LLM returns an apology but context has valid facts.
        Optionally filters NEAR results by target_category.
        """
        raw_context = state.raw_context or []
        if not raw_context:
            return None

        parsed = self._parse_raw_context(raw_context)
        entity_name = parsed["entity_name"]
        relations = parsed["relations"]
        attributes = parsed["attributes"]
        description = parsed["description"]

        if not entity_name and not relations and not attributes and not description:
            return None

        parts: List[str] = []

        if description:
            desc_short = description[:300] + "..." if len(description) > 300 else description
            parts.append(f"**{entity_name or 'Địa điểm này'}**: {desc_short}")

        for left, right in relations.get("LOCATED_IN", []):
            parts.append(f"{left} nằm tại {right}.")

        for left, right in relations.get("BELONGS_TO", []):
            parts.append(f"{left} thuộc loại {right}.")

        near_targets = []
        for left, right in relations.get("NEAR", []):
            if target_category:
                if self._category_matches_line(target_category, right):
                    near_targets.append(right)
            else:
                near_targets.append(right)
        if near_targets:
            anchor = entity_name or "Địa điểm này"
            parts.append(f"{anchor} nằm gần: {', '.join(dict.fromkeys(near_targets))}.")

        has_items = [right for _, right in relations.get("HAS", [])]
        if has_items:
            parts.append(f"{entity_name or 'Địa điểm này'} phục vụ: {', '.join(dict.fromkeys(has_items))}.")

        for left, right in relations.get("HELD_AT", []):
            parts.append(f"{left} tổ chức tại {right}.")

        for left, right in relations.get("OFFERS", []):
            parts.append(f"{left} cung cấp tour: {right}.")

        includes = [right for _, right in relations.get("INCLUDES", [])]
        if includes:
            parts.append(f"Bao gồm các điểm đến: {', '.join(dict.fromkeys(includes))}.")

        if attributes.get("address") and not any("nằm tại" in p for p in parts):
            parts.append(f"Địa chỉ: {attributes['address']}.")
        if attributes.get("phone"):
            parts.append(f"Điện thoại: {attributes['phone']}.")
        if attributes.get("type") and entity_name:
            parts.append(f"Loại hình: {attributes['type']}.")

        if not parts:
            return None

        return "\n".join(parts)

    def _answer_constrained_nearby_search_if_possible(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        """Handle constrained nearby search via GraphReasoningExecutor."""
        if (state.metadata or {}).get("retrieval_plan_mode") != "constrained_nearby_search":
            return None

        chain = (state.metadata or {}).get("query_frame_chain") or []
        location_scope = (
            (state.metadata or {}).get("query_frame_chain_location_scope")
            or state.location
            or ""
        )
        answer_set_label = (state.metadata or {}).get("target_class") or "Accommodation"

        if not chain:
            return None

        from graph_rag.core.graph_reasoning import GraphReasoningExecutor, render_chain_answer

        executor = GraphReasoningExecutor(self.pipeline.driver)
        res = executor.execute_chain(
            answer_set_label=answer_set_label,
            chain=chain,
            location_scope=location_scope,
        )

        if res.fallback or not res.answer_nodes:
            state.runtime.metadata["constrained_nearby_search_failed"] = True
            return None

        answer_text = render_chain_answer(
            query=state.user_query,
            result=res,
            answer_set_label=answer_set_label,
            location_scope=location_scope,
        )

        intent = state.query_plan.intent if state.query_plan else state.primary_intent
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["constrained_nearby_search_short_circuit"] = True
        state.runtime.metadata["detected_location"] = location_scope

        from types import SimpleNamespace
        seeds_objects = [
            SimpleNamespace(
                id=n.get("id"),
                content=n.get("name", ""),
                metadata={
                    "name": n.get("name"),
                    "labels": n.get("labels", []),
                    "lat": n.get("lat"),
                    "lng": n.get("lng"),
                    "address": n.get("address"),
                },
            )
            for n in res.answer_nodes
        ]
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(seeds_objects)
        state.runtime.metadata["route_seed_nodes"] = []
        p = self.pipeline
        state.runtime.metadata["graph"] = p._build_graph_payload(
            seeds_objects, [], intent=intent
        )
        return {"answer": answer_text, "metadata": state.runtime.metadata}
