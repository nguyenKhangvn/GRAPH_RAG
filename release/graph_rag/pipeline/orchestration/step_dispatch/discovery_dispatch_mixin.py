from __future__ import annotations
"""Discovery list, event schedule, and food discovery renderers."""
import logging

logger = logging.getLogger(__name__)


import re


from typing import Any, Dict



from graph_rag.config.region_patterns import TYPE_HEADER_MAP


from graph_rag.utils.text import normalize_text


from ..dto import PipelineRunState


class DiscoveryDispatchMixin:
    """Mixin providing deterministic discovery, event, and food list dispatch."""

    def _dispatch_discovery_list(self, state: PipelineRunState, generator_candidates: list) -> str:
        """Deterministic discovery list: render topic-based entity info without LLM.

        Short-circuit chain:
        1. _answer_global_category_listing_if_possible (Cypher-based category listing)
        2. Context-enriched rendering (use retrieved facts for rich answers)
        3. Candidate-only fallback (name + address when no context)
        Returns empty string ONLY if zero candidates AND zero context → caller falls through to LLM.
        """
        from graph_rag.core.intents import IntentType

        plan = state.query_plan
        intent_type = IntentType.from_value(plan.intent) if plan else IntentType.DISCOVERY
        exclusion_ctx = state.runtime.metadata.get("exclusion_context")
        force_det = exclusion_ctx.should_force_deterministic if exclusion_ctx else state.runtime.metadata.get("force_deterministic", False)
        
        candidates = generator_candidates or []
        
        if not force_det and candidates:
            if intent_type == IntentType.TOURISM:
                curated_ctx = self._prepare_curated_tourism_context(state, candidates)
                if curated_ctx:
                    state.runtime.metadata["curated_tourism_context"] = curated_ctx
                    state.runtime.metadata["curated_tourism_entities"] = [
                        (c.get("name") if isinstance(c, dict) else getattr(c, "content", ""))
                        for c in candidates[:20]
                    ]
                    logger.info("   -> [Curated] Prepared %d tourism places for LLM curation", len(candidates))
                    return ""
            elif intent_type == IntentType.ACCOMMODATION:
                curated_ctx = self._prepare_curated_accommodation_context(state, candidates)
                if curated_ctx:
                    state.runtime.metadata["curated_accommodation_context"] = curated_ctx
                    state.runtime.metadata["curated_accommodation_entities"] = [
                        (c.get("name") if isinstance(c, dict) else getattr(c, "content", ""))
                        for c in candidates[:10]
                    ]
                    logger.info("   -> [Curated] Prepared %d accommodations for LLM curation", len(candidates))
                else:
                    # Even without curated context, populate entity names so LLM
                    # has candidate list for synthesis instead of falling through
                    # to deterministic renderer (which produces thin answers).
                    state.runtime.metadata["curated_accommodation_entities"] = [
                        (c.get("name") if isinstance(c, dict) else getattr(c, "content", ""))
                        for c in candidates[:10]
                    ]
                    logger.info("   -> [Curated] No rich context for %d accommodations, forcing LLM synthesis with pruned facts", len(candidates))
                # Always return "" for ACCOMMODATION to force LLM synthesis path.
                # Deterministic renderer produces thin name-only lists; LLM can
                # leverage pruned facts (address, description, amenities) for richer answers.
                return ""

        community_answer = self._answer_community_advice_if_possible(state, generator_candidates)
        if community_answer:
            return community_answer

        event_answer = self._answer_event_schedule_if_possible(state, generator_candidates)
        if event_answer:
            return event_answer
        if state.runtime.metadata.get("curated_event_context"):
            return ""

        food_answer = self._answer_food_discovery_if_possible(state)
        if food_answer:
            return food_answer

        # 1. Try Cypher-based category listing first
        category_answer = self._answer_global_category_listing_if_possible(state)
        if category_answer:
            # Store as curated context and fall through to LLM instead of returning directly
            state.runtime.metadata["curated_food_context"] = category_answer
            logger.info("   -> [Curated] Stored category listing for LLM synthesis (%d chars)", len(category_answer))
            return ""

        candidates = generator_candidates or []
        clean_context = str(state.clean_context or "").strip()
        has_rich_context = len(clean_context) > 80

        # Post-filter: Cham Pa specific queries should only show Cham Pa sites
        q_norm = normalize_text(state.user_query or "", strip_punct=True)
        if any(sig in q_norm for sig in ["cham pa", "champa", "thap cham", "van hoa cham"]):
            cham_pa_candidates = []
            for c in candidates:
                c_name = str(c.get("name") if isinstance(c, dict) else getattr(c, "content", "")).strip().lower()
                c_desc = str(c.get("description") if isinstance(c, dict) else getattr(c, "metadata", {}).get("description", "")).strip().lower()
                c_category = str(c.get("category") if isinstance(c, dict) else getattr(c, "metadata", {}).get("category", "")).strip().lower()
                # Keep nodes with "Tháp" in name or "Chăm" in description/category
                if "tháp" in c_name or "thap" in c_name or "chăm" in c_desc or "cham" in c_desc or "chăm" in c_category or "cham" in c_category:
                    cham_pa_candidates.append(c)
            if cham_pa_candidates:
                logger.info("   -> [DiscoveryList] Cham Pa filter: %s -> %s candidates", len(candidates), len(cham_pa_candidates))
                candidates = cham_pa_candidates

        # 2. Context-enriched rendering: store as curated context and fall through to LLM
        if has_rich_context and candidates:
            rendered = self._render_discovery_from_context(state, candidates, clean_context)
            if rendered:
                state.runtime.metadata["curated_food_context"] = rendered
                logger.info("   -> [Curated] Stored discovery context for LLM synthesis (%d chars)", len(rendered))
                return ""

        # 3. Candidate-only fallback (thin context)
        if not candidates:
            return ""

        return self._render_discovery_from_candidates(state, candidates)

    def _answer_community_advice_if_possible(self, state: PipelineRunState, candidates: list) -> str:
        """Render community/forum travel advice from TravelInfo only."""
        metadata = state.metadata or {}
        q_norm = normalize_text(state.user_query or "", strip_punct=True)
        is_community = (
            metadata.get("topic") == "community"
            or metadata.get("fallback_policy") == "community_guided_fallback"
            or (
                any(sig in q_norm for sig in ["cong dong", "dien dan", "forum", "nhom du lich", "chia se trai nghiem"])
                and "du lich" in q_norm
            )
        )
        if not is_community:
            return ""

        nodes = []
        for node in list(candidates or []) + list(state.grounded_nodes or []):
            node_meta = getattr(node, "metadata", None) if not isinstance(node, dict) else node
            if not isinstance(node_meta, dict):
                continue
            labels = node_meta.get("labels") or []
            node_type = str(node_meta.get("type") or (labels[0] if labels else ""))
            topic = str(node_meta.get("topic") or "").lower()
            if node_type == "TravelInfo" and topic == "community":
                nodes.append(node)

        if not nodes:
            try:
                from graph_rag.core.state import NodeItem
                with self.pipeline.driver.session() as session:
                    rows = session.run(
                        """
                        MATCH (t:TravelInfo)
                        WHERE t.topic = 'community'
                        RETURN t.id AS id, t.name AS name, t.description AS description,
                               t.topic AS topic, labels(t) AS labels
                        LIMIT 3
                        """
                    ).data()
                for row in rows:
                    nodes.append(NodeItem(
                        id=row.get("id") or row.get("name") or "travelinfo-community",
                        content=row.get("name") or "",
                        score=1.0,
                        source_type="direct_community_fallback",
                        metadata={
                            "name": row.get("name") or "",
                            "description": row.get("description") or "",
                            "topic": row.get("topic") or "community",
                            "type": "TravelInfo",
                            "labels": row.get("labels") or ["TravelInfo"],
                        },
                    ))
            except (ValueError, RuntimeError, OSError) as exc:
                logger.error("   -> [CommunityAdvice] fallback lookup failed: %s", exc)

        if not nodes:
            return "Dữ liệu hiện có chưa có mục TravelInfo về cộng đồng hoặc diễn đàn du lịch uy tín."

        seen = set()
        lines = ["Bạn có thể tham khảo các cộng đồng/diễn đàn du lịch sau:\n"]
        for node in nodes:
            meta = getattr(node, "metadata", None) if not isinstance(node, dict) else node
            if not isinstance(meta, dict):
                continue
            name = str(meta.get("name") or getattr(node, "content", "") or "Cộng đồng du lịch").strip()
            desc = str(meta.get("description") or "").strip()
            key = (name.lower(), desc[:80].lower())
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- **{name}**")
            if desc:
                lines.append(desc)

        lines.append("\nNên ưu tiên cộng đồng có kiểm duyệt, bài review có ảnh thật/ngày đăng rõ ràng, và kiểm chứng lại giá hoặc lịch hoạt động trước khi chốt kế hoạch.")
        return "\n".join(lines).strip()

    def _answer_event_schedule_if_possible(self, state: PipelineRunState, candidates: list) -> str:
        """Render event schedule answers from Event/TravelInfo evidence only."""
        q_norm = normalize_text(state.user_query or "", strip_punct=True)
        metadata = state.metadata or {}
        plan = state.query_plan
        intent = plan.intent.upper()
        target_class = plan.target_class or ""
        is_event_query = (
            "EVENT" in intent
            or target_class == "Event"
            or any(sig in q_norm for sig in ["le hoi", "su kien", "festival", "dien ra"])
        )
        if not is_event_query:
            return ""

        def _candidate_dict(c: Any) -> dict:
            if isinstance(c, dict):
                labels = c.get("labels") or ([c.get("type")] if c.get("type") else [])
                return {
                    "name": str(c.get("name") or c.get("content") or "").strip(),
                    "address": str(c.get("address") or "").strip(),
                    "type": str(c.get("type") or (labels[0] if labels else "") or "").strip(),
                    "labels": labels,
                    "description": str(c.get("description") or "").strip(),
                    "month": c.get("month"),
                    "date": c.get("date") or c.get("event_date") or c.get("date_range"),
                    "topic": str(c.get("topic") or "").strip(),
                }

            meta = getattr(c, "metadata", {}) or {}
            labels = meta.get("labels") or ([meta.get("type")] if meta.get("type") else [])
            return {
                "name": str(getattr(c, "content", "") or meta.get("name") or "").strip(),
                "address": str(meta.get("address") or "").strip(),
                "type": str((labels[0] if labels else meta.get("type")) or "").strip(),
                "labels": labels,
                "description": str(meta.get("description") or "").strip(),
                "month": meta.get("month"),
                "date": meta.get("date") or meta.get("event_date") or meta.get("date_range"),
                "topic": str(meta.get("topic") or "").strip(),
            }

        event_items: list[dict] = []
        seen: set[str] = set()
        for c in candidates or []:
            item = _candidate_dict(c)
            labels = {str(lbl) for lbl in (item.get("labels") or [])}
            item_type = item.get("type") or ""
            is_event = item_type == "Event" or "Event" in labels
            is_event_info = item_type == "TravelInfo" and (item.get("topic") or "").lower() == "event"
            if not (is_event or is_event_info):
                continue
            name = item.get("name") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            event_items.append(item)

        clean_context = self._remove_internal_context_lines(str(state.clean_context or ""))
        lines = [line.strip() for line in clean_context.splitlines() if line.strip()]
        for item in event_items:
            name_norm = normalize_text(item["name"], strip_punct=True)
            snippets = []
            for line in lines:
                line_norm = normalize_text(line, strip_punct=True)
                if name_norm and name_norm in line_norm and len(line) > 20:
                    snippets.append(re.sub(r"^\[[A-Z_]+\]\s*", "", line))
            item["snippets"] = snippets[:3]

        months = metadata.get("time_constraint_months") or []
        if not event_items:
            if months:
                time_note = f" trong giai doan thang {min(months)}-{max(months)}"
            else:
                time_note = ""
            return (
                f"Dữ chưa ghi nhận lịch cụ thể các lễ hội/sự kiện văn hóa{time_note} "
                "cho khu vực bạn hỏi. Hệ thống sẽ không gợi ý địa điểm tham quan thay thế khi thiếu bằng chứng sự kiện."
            )

        exclusion_ctx = state.runtime.metadata.get("exclusion_context")
        force_det = exclusion_ctx.should_force_deterministic if exclusion_ctx else state.runtime.metadata.get("force_deterministic", False)
        
        if not force_det:
            entries = []
            for item in event_items[:10]:
                name = item.get("name") or ""
                if not name:
                    continue
                # Only send name + time (factual, LLM may not know exact dates).
                # Description excluded — LLM generates richer descriptions itself.
                entry = name
                date = item.get("date") or ""
                if date:
                    entry += f" ({date})"
                elif item.get("month"):
                    entry += f" (tháng {item['month']})"
                addr = item.get("address") or ""
                if addr:
                    entry += f" - {addr}"
                entries.append(entry)

            if entries:
                location = state.location or state.runtime.metadata.get("detected_location") or ""
                location_str = f" tại {location}" if location else ""
                header = f"DANH SÁCH LỄ HỘI / SỰ KIỆN{location_str}:\n"
                body = "\n".join(f"- {e}" for e in entries)
                state.runtime.metadata["curated_event_context"] = header + body
                state.runtime.metadata["curated_event_entities"] = [item["name"] for item in event_items[:10]]
                logger.info("   -> [Curated] Prepared %d events for LLM curation", len(event_items))
                return ""

        if months:
            parts = [f"Một số lễ hội/sự kiện phù hợp giai đoạn tháng {min(months)}-{max(months)}:"]
        else:
            parts = ["Một số lễ hội/sự kiện phù hợp:"]

        uncertain = False
        for item in event_items[:6]:
            entry = f"\n**{item['name']}**"
            if item.get("address"):
                entry += f" - {item['address']}"
            if item.get("date"):
                entry += f"\n- Thoi gian: {item['date']}"
            elif item.get("month"):
                entry += f"\n- Thoi gian: thang {item['month']}"
            else:
                uncertain = True

            desc = item.get("description") or ""
            snippets = item.get("snippets") or []
            if desc:
                entry += f"\n- {desc}"
            elif snippets:
                entry += f"\n- {snippets[0]}"
            parts.append(entry)

        if uncertain and months:
            parts.append(
                "\nLưu ý: một số sự kiện trong du liệu chưa có ngày/tháng cụ thể, "
                "nên cần xác nhận lịch tổ chức trước khi đi."
            )
        return "\n".join(parts)

    def _answer_event_time_fact_if_possible(self, state: PipelineRunState) -> str:
        """Render a single event date/time fact from grounded Event evidence."""
        q_norm = normalize_text(state.user_query or "", strip_punct=True)
        metadata = state.metadata or {}
        plan = state.query_plan
        intent = plan.intent.upper()
        target_class = plan.target_class or ""
        asks_time = any(
            token in q_norm
            for token in [
                "thoi gian nao",
                "luc nao",
                "khi nao",
                "ngay nao",
                "to chuc vao",
                "dien ra vao",
                "to chuc luc",
                "dien ra luc",
            ]
        )
        is_event_query = "EVENT" in intent or target_class == "Event"
        if not (is_event_query and asks_time):
            return ""

        def _meta_from_seed(seed: Any) -> dict:
            meta = getattr(seed, "metadata", {}) or {}
            labels = meta.get("labels") or ([meta.get("type")] if meta.get("type") else [])
            return {
                "name": str(meta.get("name") or getattr(seed, "content", "") or "").strip(),
                "labels": labels,
                "type": str((labels[0] if labels else meta.get("type")) or "").strip(),
                "address": str(meta.get("address") or meta.get("location") or "").strip(),
                "date": str(
                    meta.get("date")
                    or meta.get("event_date")
                    or meta.get("date_range")
                    or meta.get("time")
                    or meta.get("event_time")
                    or ""
                ).strip(),
                "month": str(meta.get("month") or "").strip(),
                "description": str(meta.get("description") or "").strip(),
                "score": float(getattr(seed, "score", 0.0) or 0.0),
            }

        event_items: list[dict] = []
        seen: set[str] = set()
        for seed in state.all_seeds or []:
            item = _meta_from_seed(seed)
            labels = {str(label) for label in item.get("labels") or []}
            is_event = item.get("type") == "Event" or "Event" in labels
            if not is_event:
                continue
            name = item.get("name") or ""
            if not name or name in seen:
                continue
            seen.add(name)
            event_items.append(item)

        if not event_items:
            return ""

        # Prefer an event whose name/description matches the activity words in the query.
        activity_tokens = [
            token for token in re.findall(r"\w+", q_norm)
            if len(token) >= 4 and token not in {"thoi", "gian", "gia", "lai", "nam", "nao", "chuc", "dien"}
        ]

        def _rank(item: dict) -> tuple[int, int, float]:
            haystack = normalize_text(
                f"{item.get('name', '')} {item.get('description', '')}",
                strip_punct=True,
            )
            overlap = sum(1 for token in activity_tokens if token in haystack)
            has_date = 1 if (item.get("date") or item.get("month")) else 0
            return (overlap, has_date, float(item.get("score") or 0.0))

        event_items.sort(key=_rank, reverse=True)
        item = event_items[0]
        name = item.get("name") or "sự kiện này"
        date_text = (
            item.get("date")
            or self._extract_event_time_from_text(item.get("description") or "")
            or (f"tháng {item['month']}" if item.get("month") else "")
        )
        if not date_text:
            return f"Dữ liệu hiện có ghi nhận {name}, nhưng chưa có thời gian tổ chức cụ thể."

        lines = [
            f"Theo dữ liệu trong hệ thống, {name} tổ chức vào {date_text}."
        ]
        if item.get("address"):
            lines.append(f"Địa điểm: {item['address']}.")
        desc = item.get("description") or ""
        if desc:
            lines.append(f"Mô tả: {desc}")
        return "\n\n".join(lines).strip()

    def _extract_event_time_from_text(self, text: str) -> str:
        """Best-effort extraction of event schedule phrases from description text."""
        raw = str(text or "").strip()
        if not raw:
            return ""
        patterns = [
            r"(?iu)(ngày\s+\d{1,2}\s*(?:[-–]\s*\d{1,2})?\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
            r"(?iu)(từ\s+ngày\s+\d{1,2}\s*(?:[-–]\s*\d{1,2})?\s+tháng\s+\d{1,2}\s+năm\s+\d{4})",
            r"(?iu)(\d{1,2}\s*[-–]\s*\d{1,2}/\d{1,2}/\d{4})",
            r"(?iu)(\d{1,2}/\d{1,2}/\d{4})",
            r"(?iu)(tháng\s+\d{1,2}/\d{4})",
            r"(?iu)(tháng\s+\d{1,2}\s+năm\s+\d{4})",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                return match.group(1).strip(" .,:;")
        return ""

    def _remove_internal_context_lines(self, context: str) -> str:
        """Drop internal evidence/debug markers before deterministic rendering."""
        if not context:
            return ""
        cleaned: list[str] = []
        skip_structural = False
        for raw in context.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith("[STRUCTURAL FACTS"):
                skip_structural = True
                continue
            if line.startswith("[TEXTUAL EVIDENCE"):
                skip_structural = False
                continue
            if skip_structural:
                continue
            if line.startswith("Missing:") or line.startswith("must_compare_all_anchors:"):
                continue
            cleaned.append(raw)
        return "\n".join(cleaned)

    def _render_discovery_from_context(
        self, state: PipelineRunState, candidates: list, clean_context: str
    ) -> str:
        """Render discovery answer from retrieved context facts + candidate metadata.

        For discovery queries (asking for a LIST), render ALL candidates with
        equal weight instead of focusing on a single main_entity.
        """
        import re as _re

        loc_suffix = f" ở {state.location}" if getattr(state, "location", None) else ""

        # Collect candidate info (including description from metadata)
        candidate_info: list[dict] = []
        for c in candidates:
            if isinstance(c, dict):
                name = str(c.get("name") or "").strip()
                addr = str(c.get("address") or "").strip()
                c_type = str(c.get("type") or "").strip()
                price = str(c.get("price_range") or c.get("price") or "").strip()
                desc = str(c.get("description") or "").strip()
            else:
                name = str(getattr(c, "content", "") or "").strip()
                meta = getattr(c, "metadata", {}) or {}
                addr = str(meta.get("address") or "").strip()
                labels = meta.get("labels") or []
                c_type = labels[0] if labels else str(meta.get("type") or "").strip()
                price = str(meta.get("price_range") or meta.get("price") or "").strip()
                desc = str(meta.get("description") or "").strip()
            if name:
                candidate_info.append({"name": name, "address": addr, "type": c_type, "price_range": price, "description": desc})

        if not candidate_info:
            return ""

        # Filter by target class if query explicitly asks for a specific type
        target_class = (state.metadata or {}).get("target_class") or ""
        has_transport_hint = bool((state.metadata or {}).get("transport_hint"))
        if target_class and target_class not in {"Unknown", ""}:
            filtered = [ci for ci in candidate_info if ci.get("type") == target_class]
            # Keep TravelInfo nodes when transport_hint is set (user asks about di chuyen)
            if has_transport_hint:
                travel_info = [ci for ci in candidate_info if ci.get("type") == "TravelInfo"]
                if travel_info:
                    filtered = list(dict.fromkeys(filtered + travel_info))
            if filtered:
                candidate_info = filtered

        # Filter by location: remove candidates whose address doesn't match target location
        # Location context from QueryPlan (single contract)
        plan = state.query_plan
        if plan and plan.legacy_province:
            target_location = plan.legacy_province
        else:
            target_location = (getattr(state, "location", None) or "").strip()

        if target_location and len(target_location) > 2:
            loc_norm = target_location.lower()
            # Keep candidates that match location OR have no address (can't filter)
            location_filtered = [
                ci for ci in candidate_info
                if not ci.get("address") or loc_norm in ci["address"].lower()
            ]
            if location_filtered:
                candidate_info = location_filtered

        # Parse context lines into structured facts
        clean_context = self._remove_internal_context_lines(clean_context)
        lines = [line.strip() for line in clean_context.splitlines() if line.strip()]

        # Group facts by relation type
        facts_by_type: Dict[str, list] = {}
        entity_facts: Dict[str, list] = {}  # facts mentioning a specific entity

        for line in lines:
            # Match both formats:
            #   "- Subject [RELATION] -> Object"  (graph edge format)
            #   "[RELATION] value"                 (legacy format)
            rel_match = _re.match(r"-\s*(.+?)\s+\[([A-Z_]+)\]\s*->\s*(.+)", line)
            if not rel_match:
                rel_match = _re.match(r"\[([A-Z_]+)\]\s*(.+)", line)
                if rel_match:
                    rel_type = rel_match.group(1)
                    rel_value = rel_match.group(2).strip()
                    subject = ""
                else:
                    rel_match = None
            else:
                subject = rel_match.group(1).strip()
                rel_type = rel_match.group(2)
                rel_value = f"{subject} -> {rel_match.group(3).strip()}"

            if rel_match:
                facts_by_type.setdefault(rel_type, []).append(rel_value)
                # Try to associate fact with a candidate
                for ci in candidate_info:
                    ci_name_lower = ci["name"].lower()
                    # Match against subject (graph edge) or full value (legacy)
                    if (subject and ci_name_lower in subject.lower()) or ci_name_lower in rel_value.lower():
                        entity_facts.setdefault(ci["name"], []).append(f"[{rel_type}] {rel_value}")
                        break
            else:
                facts_by_type.setdefault("DESCRIPTION", []).append(line)
                for ci in candidate_info:
                    if ci["name"].lower() in line.lower():
                        entity_facts.setdefault(ci["name"], []).append(line)
                        break

        # Dedup facts_by_type and entity_facts (exact + near-duplicate)
        def _normalize_for_dedup(s: str) -> str:
            """Strip common prefixes and normalize for near-duplicate detection."""
            import re as _re2
            import unicodedata as _ud
            s = s.strip().lower()
            # Remove [RELATION] prefix from graph edges
            s = _re2.sub(r"^\[[a-z_]+\]\s*", "", s)
            # Remove common Vietnamese info prefixes (legacy)
            s = _re2.sub(r"^thông tin\s+(của\s+)?", "", s)
            s = _re2.sub(r"^mô tả\s+(của\s+)?", "", s)
            # Remove new format: "{name}: {desc}" → extract name only
            s = _re2.sub(r"^(.+?):\s+.*$", r"\1", s)
            # Remove attribute format: "{name} - Địa chỉ/SĐT: ..."
            s = _re2.sub(r"\s*-\s*(địa chỉ|sđt|loại hình)\s*:.*$", "", s)
            # Remove multi-hop format: "X (liên kết N bước: ...) → Y"
            s = _re2.sub(r"\s*\(liên kết\s+\d+\s+bước:.*?\)\s*→\s*", " ", s)
            # Remove confidence scores
            s = _re2.sub(r"\[độ tin cậy path:\s*[\d.]+\]", "", s)
            # Strip accents for comparison
            norm = _ud.normalize("NFKD", s)
            norm = "".join(ch for ch in norm if not _ud.combining(ch))
            norm = norm.replace("đ", "d")
            # Remove leading punctuation/whitespace
            norm = norm.strip(" :.-")
            return norm

        def _dedup_facts(facts: list) -> list:
            seen_exact = set()
            seen_core = set()
            result = []
            for f in facts:
                norm = f.strip().lower()
                if norm in seen_exact:
                    continue
                seen_exact.add(norm)
                # Near-duplicate: check normalized core content
                core = _normalize_for_dedup(f)
                if core and core in seen_core:
                    continue
                if core:
                    seen_core.add(core)
                result.append(f)
            return result

        for k in facts_by_type:
            facts_by_type[k] = _dedup_facts(facts_by_type[k])
        for k in entity_facts:
            entity_facts[k] = _dedup_facts(entity_facts[k])

        # Filter by semantic_category: skip irrelevant relation types
        # Actual graph relations: NEAR, LOCATED_IN, INCLUDES, BELONGS_TO,
        # HAS, OFFERS, HELD_AT, SUPERSEDED_BY
        semantic_category = getattr(getattr(state, "query_plan", None), "semantic_category", None) or ""
        _CATEGORY_IRRELEVANT_RELS = {
            "cultural_village": {"HAS", "OFFERS"},      # skip food & tour agency
            "heritage": {"HAS", "OFFERS"},              # skip food & tour agency
            "natural_landmark": {"HAS", "OFFERS"},      # skip food & tour agency
            "spiritual": {"HAS", "OFFERS"},             # skip food & tour agency
        }
        if semantic_category in _CATEGORY_IRRELEVANT_RELS:
            _skip_rels = _CATEGORY_IRRELEVANT_RELS[semantic_category]
            facts_by_type = {k: v for k, v in facts_by_type.items() if k not in _skip_rels}
            for k in list(entity_facts):
                entity_facts[k] = [f for f in entity_facts[k]
                                   if not any(f"[{r}]" in f for r in _skip_rels)]

        # Build answer: render ALL candidates equally
        parts = []
        # Determine header from dominant entity type
        type_counts: Dict[str, int] = {}
        for ci in candidate_info:
            t = ci.get("type") or ""
            type_counts[t] = type_counts.get(t, 0) + 1
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else ""
        header_label = TYPE_HEADER_MAP.get(dominant_type, "địa điểm")
        parts.append(f"Một số {header_label}{loc_suffix}:")

        for ci in candidate_info[:8]:
            name = ci["name"]
            addr = ci["address"]
            price = ci.get("price_range") or ""
            entry = f"\n**{name}**"
            if addr:
                entry += f" — {addr}"
            if price:
                entry += f" (Giá: {price})"

            # Attach entity-specific facts if available
            e_facts = entity_facts.get(name, [])
            if e_facts:
                for f in e_facts[:3]:
                    # Clean up fact format for display
                    f = _re.sub(r"^\[[A-Z_]+\]\s*", "", f)
                    if len(f) > 10:
                        entry += f"\n  - {f}"
            elif ci.get("description"):
                # Use candidate metadata description when no context facts
                snippet = ci["description"]
                for sep in [". ", ".\n", "! ", "? "]:
                    idx = snippet.find(sep)
                    if 0 < idx < 150:
                        snippet = snippet[:idx + 1].strip()
                        break
                else:
                    if len(snippet) > 150:
                        snippet = snippet[:150].rsplit(" ", 1)[0] + "..."
                if snippet and len(snippet) > 10:
                    entry += f"\n  {snippet}"

            parts.append(entry)

        # Add general facts (not tied to any specific candidate) as context
        general_descs = facts_by_type.get("DESCRIPTION", [])
        general_descs = [d for d in general_descs if len(d) > 10 and not any(
            ci["name"].lower() in d.lower() for ci in candidate_info
        )]
        if general_descs:
            parts.append(f"\n**Thông tin thêm:**")
            for desc in general_descs[:3]:
                parts.append(f"- {desc}")

        answer = "\n".join(parts)
        # If answer is too thin (< 100 chars), return empty to fall through to LLM
        return answer if len(answer) > 100 else ""

    def _render_discovery_from_candidates(
        self, state: PipelineRunState, candidates: list
    ) -> str:
        """Fallback: render candidate names + addresses + descriptions when available."""
        loc_suffix = f" ở {state.location}" if getattr(state, "location", None) else ""

        by_type: Dict[str, list] = {}
        seen_names = set()
        for c in candidates:
            name = ""
            addr = ""
            c_type = ""
            price = ""
            desc = ""
            if isinstance(c, dict):
                name = str(c.get("name") or "").strip()
                addr = str(c.get("address") or "").strip()
                c_type = str(c.get("type") or "").strip()
                price = str(c.get("price_range") or c.get("price") or "").strip()
                desc = str(c.get("description") or "").strip()
            else:
                name = str(getattr(c, "content", "") or "").strip()
                meta = getattr(c, "metadata", {}) or {}
                addr = str(meta.get("address") or "").strip()
                labels = meta.get("labels") or []
                c_type = labels[0] if labels else str(meta.get("type") or "").strip()
                price = str(meta.get("price_range") or meta.get("price") or "").strip()
                desc = str(meta.get("description") or "").strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            by_type.setdefault(c_type, []).append({
                "name": name,
                "address": addr,
                "price_range": price,
                "description": desc,
            })

        if not by_type:
            return ""

        lines = [f"Dựa trên dữ liệu đã truy xuất, đây là một số gợi ý{loc_suffix}:"]
        for c_type, items in by_type.items():
            type_label = TYPE_HEADER_MAP.get(c_type, "địa điểm")
            lines.append(f"\n**{type_label.title()}:**")
            for item in items[:6]:
                entry = f"- **{item['name']}**"
                if item["address"]:
                    entry += f" — {item['address']}"
                if item.get("price_range"):
                    entry += f" (Giá: {item['price_range']})"
                # Add description snippet if available
                if item.get("description"):
                    snippet = item["description"]
                    # Get first sentence
                    for sep in [". ", ".\n", "! ", "? "]:
                        idx = snippet.find(sep)
                        if 0 < idx < 150:
                            snippet = snippet[:idx + 1].strip()
                            break
                    else:
                        if len(snippet) > 150:
                            snippet = snippet[:150].rsplit(" ", 1)[0] + "..."
                    if snippet and len(snippet) > 10:
                        entry += f"\n  {snippet}"
                lines.append(entry)

        return "\n".join(lines)

    def _answer_food_discovery_if_possible(self, state: PipelineRunState) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        plan = state.query_plan
        intent = plan.intent.upper()
        target_dish = normalize_text(plan.target_dish or "", strip_punct=True)
        is_food = "FOOD" in intent or any(
            marker in q_norm
            for marker in ["an hai san", "hai san", "nha hang", "quan an", "an gi", "mon ngon"]
        )
        if not is_food or ("hai san" not in q_norm and "hai san" not in target_dish):
            return ""

        driver = getattr(self.pipeline, "driver", None)
        if not driver:
            return ""

        try:
            with driver.session() as session:
                rows = session.run(
                    """
                    MATCH (r:Restaurant)
                    OPTIONAL MATCH (r)-[:HAS]->(d:Dish)
                    RETURN r.name AS name,
                           r.address AS address,
                           r.phone AS phone,
                           r.type AS type,
                           r.tags AS tags,
                           collect(d.name) AS dishes
                    LIMIT 300
                    """
                ).data()
        except (ValueError, TypeError) as exc:
            self.pipeline.logger.warning("food_discovery_query_failed: %s", str(exc))
            return ""

        # Location context from QueryPlan (single contract)
        if plan and plan.legacy_province:
            loc_norm = normalize_text(plan.legacy_province, strip_punct=True)
        else:
            loc_norm = normalize_text(str(state.location or ""), strip_punct=True)

        location_terms = []
        if any(term in loc_norm or term in q_norm for term in ["binh dinh", "quy nhon", "nhon ly"]):
            location_terms = ["binh dinh", "quy nhon", "nhon ly", "xuan dieu", "ghenh rang"]
        elif "pleiku" in loc_norm or "gia lai" in loc_norm:
            location_terms = ["pleiku", "gia lai"]

        # Detect food category from query
        is_seafood_query = any(term in q_norm for term in ["hai san", "hải sản", "tom", "muc", "cua", "ốc", "ghe"])
        is_coffee_query = any(term in q_norm for term in ["ca phe", "cà phê", "coffee", "cafe"])

        selected = []
        for row in rows:
            name = str(row.get("name") or "").strip()
            address = str(row.get("address") or "").strip()
            tags = row.get("tags") or []
            dishes = ", ".join(str(x or "") for x in (row.get("dishes") or []))
            text_norm = normalize_text(" ".join([name, address, dishes]), strip_punct=True)

            # Location filter
            if location_terms and not any(term in text_norm for term in location_terms):
                continue

            # Food category filter - skip non-matching restaurants
            if is_seafood_query:
                # Skip coffee/bakery shops for seafood queries
                if "cà phê" in tags or "bánh ngọt" in tags:
                    continue
                # Require seafood evidence (tag or name/dish mention)
                has_seafood = "hải sản" in tags or "hai san" in text_norm
                if not has_seafood:
                    continue
            elif is_coffee_query:
                # For coffee queries, skip non-coffee shops
                if "cà phê" not in tags and "ca phe" not in text_norm:
                    continue

            score = 0
            if "hai san" in text_norm or "hải sản" in tags:
                score += 5
            if "nhon ly" in text_norm or "xuan dieu" in text_norm:
                score += 2
            if "quy nhon" in text_norm or "binh dinh" in text_norm:
                score += 1
            if score <= 0:
                continue
            selected.append((score, name, address, str(row.get("phone") or "").strip()))

        selected.sort(key=lambda item: (-item[0], item[1]))
        if not selected:
            return ""

        location_label = state.metadata.get("matched_admin_alias") or state.location or "Bình Định/Quy Nhơn"
        lines = [f"Một số gợi ý ăn hải sản tươi ngon, giá hợp lý ở {location_label}:"]
        for _, name, address, phone in selected[:6]:
            line = f"- **{name}**"
            if address:
                line += f" - {address}"
            if phone:
                line += f" (ĐT: {phone})"
            lines.append(line)
        lines.append("\nNên gọi trước để hỏi giá trong ngày và hải sản còn sẵn, vì giá thường thay đổi theo mùa và nguồn hàng.")
        return "\n".join(lines)

    def _prepare_curated_tourism_context(self, state: PipelineRunState, candidates: list) -> str:
        """Lean context: names only (+ address). LLM generates descriptions itself."""
        entries = []
        seen = set()
        for c in candidates:
            if isinstance(c, dict):
                name = str(c.get("name") or "").strip()
                addr = str(c.get("address") or "").strip()
            else:
                name = str(getattr(c, "content", "") or "").strip()
                meta = getattr(c, "metadata", {}) or {}
                addr = str(meta.get("address") or "").strip()

            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())

            entry = name
            if addr:
                entry += f" ({addr})"
            entries.append(entry)
            if len(entries) >= 10:
                break

        if not entries:
            return ""

        location = state.metadata.get("matched_admin_alias") or state.location or state.runtime.metadata.get("detected_location") or ""
        location_str = f" tại {location}" if location else ""
        header = f"DANH SÁCH ĐỊA ĐIỂM DU LỊCH/THAM QUAN{location_str}:\n"
        body = "\n".join(f"- {e}" for e in entries)
        return header + body

    def _prepare_curated_accommodation_context(self, state: PipelineRunState, candidates: list) -> str:
        """Lean context: names + address + price (factual). LLM generates descriptions itself."""
        entries = []
        seen = set()
        for c in candidates:
            if isinstance(c, dict):
                name = str(c.get("name") or "").strip()
                addr = str(c.get("address") or "").strip()
                price = str(c.get("price_range") or c.get("price") or "").strip()
            else:
                name = str(getattr(c, "content", "") or "").strip()
                meta = getattr(c, "metadata", {}) or {}
                addr = str(meta.get("address") or "").strip()
                price = str(meta.get("price_range") or meta.get("price") or "").strip()

            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())

            entry = name
            if addr:
                entry += f" ({addr})"
            if price:
                entry += f" - {price}"
            entries.append(entry)
            if len(entries) >= 10:
                break

        if not entries:
            return ""

        location = state.metadata.get("matched_admin_alias") or state.location or state.runtime.metadata.get("detected_location") or ""
        location_str = f" tại {location}" if location else ""
        header = f"DANH SÁCH ĐỊA ĐIỂM LƯU TRÚ/KHÁCH SẠN/HOMESTAY{location_str}:\n"
        body = "\n".join(f"- {e}" for e in entries)
        return header + body
