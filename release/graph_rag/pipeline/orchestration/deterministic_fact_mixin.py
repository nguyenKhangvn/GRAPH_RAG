"""Entity fact lookups — address, menu, emergency, description, travel info, dish-to-restaurant."""

import logging
import re
from typing import Any, Dict, List

from graph_rag.core import keywords
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)

from .dto import PipelineRunState

# Hard label sets for display.
_TECHNICAL_LABELS = {
    "TouristAttraction", "TravelInfo", "Restaurant", "Dish",
    "Accommodation", "Event", "Tour", "Location", "Specialty",
}

_LABEL_DISPLAY = {
    "TouristAttraction": "điểm du lịch",
    "TravelInfo": "thông tin du lịch",
    "Restaurant": "nhà hàng",
    "Dish": "món ăn",
    "Accommodation": "cơ sở lưu trú",
    "Event": "sự kiện",
    "Tour": "tour",
    "Location": "địa điểm",
    "Specialty": "đặc sản",
}


class DeterministicFactMixin:
    """Mixin for fact-lookup deterministic answers (address, menu, emergency, etc.)."""

    _HEADER_RE = re.compile(r"\*\*THỰC THỂ CHÍNH:\*\*\s*(.+?)\s*\(Loại:\s*(.+?)\)")

    def _answer_emergency_info_if_possible(self, state: PipelineRunState) -> str | None:
        q_mode = (state.metadata or {}).get("answer_mode")
        if q_mode != "emergency_info_deterministic":
            return None

        nodes = []
        seen_ids = set()

        def add_node(n):
            if not n:
                return
            n_id = None
            if hasattr(n, "id"):
                n_id = str(n.id)
            elif isinstance(n, dict):
                n_id = str(n.get("id") or "")
            if not n_id or n_id in seen_ids:
                return
            node_type = ""
            topic = ""
            if hasattr(n, "metadata") and n.metadata:
                node_type = str(n.metadata.get("type") or "")
                topic = str(n.metadata.get("topic") or "")
            elif isinstance(n, dict):
                node_type = str(
                    n.get("type") or n.get("labels", [""])[0] if n.get("labels") else ""
                )
                topic = str(n.get("topic") or "")
            if "TravelInfo" in node_type or topic == "emergency":
                if topic == "emergency":
                    nodes.append(n)
                    seen_ids.add(n_id)

        for node in state.grounded_nodes or []:
            add_node(node)
        for node in state.all_seeds or []:
            add_node(node)

        if nodes:
            lines = ["## Thông tin hỗ trợ khẩn cấp đường dây nóng\n"]
            for node in nodes:
                if hasattr(node, "metadata") and node.metadata:
                    name = node.metadata.get("name", "")
                    desc = node.metadata.get("description", "")
                    contact = node.metadata.get("contact", "")
                elif isinstance(node, dict):
                    name = node.get("name", "")
                    desc = node.get("description", "")
                    contact = node.get("contact", "")
                else:
                    continue
                lines.append(f"**{name}**")
                if desc:
                    lines.append(desc)
                if contact:
                    lines.append(f"Hotline/Liên hệ: {contact}")
                lines.append("")
            return "\n".join(lines).strip()

        lines = [
            str(item or "").strip()
            for item in (state.raw_context or [])
            if str(item or "").strip()
        ]
        if lines:
            emergency_lines = []
            for line in lines:
                line_lower = line.lower()
                if any(
                    k in line_lower
                    for k in [
                        "khan cap", "cap cuu", "duong day nong", "su co",
                        "cuu ho", "police", "benh vien", "tram y te",
                        "cong an", "cuu thuong", "hotline", "dien thoai",
                    ]
                ):
                    emergency_lines.append(line)
            if emergency_lines:
                return (
                    "## Thông tin hỗ trợ khẩn cấp đường dây nóng\n\n"
                    + "\n".join(emergency_lines)
                )

        return None

    def _answer_tour_offer_includes_if_possible(
        self, state: PipelineRunState
    ) -> str | None:
        q = normalize_text(state.user_query, strip_punct=True)
        if "tour" not in q or not any(
            token in q for token in ["cong ty", "to chuc", "bao gom", "hoat dong"]
        ):
            return None

        lines = [
            str(item or "").strip()
            for item in (state.raw_context or [])
            if str(item or "").strip()
        ]
        if not lines:
            return None

        organizer = ""
        includes: List[str] = []
        tour_name = ""
        for line in lines:
            offer_match = re.search(
                r"^-\s*(.+?)\s+\[OFFERS\]\s*->\s*(.+?)\s*$", line
            )
            if offer_match:
                organizer = offer_match.group(1).strip()
                tour_name = offer_match.group(2).strip()
                continue
            include_match = re.search(
                r"^-\s*(.+?)\s+\[INCLUDES\]\s*->\s*(.+?)\s*$", line
            )
            if include_match:
                if not tour_name:
                    tour_name = include_match.group(1).strip()
                includes.append(include_match.group(2).strip())

        if not organizer and not includes:
            return None

        if not tour_name:
            tour_name = str(
                (state.metadata or {}).get("target_entity") or "Tour này"
            ).strip()
        parts = []
        if organizer:
            parts.append(f"{tour_name} do {organizer} tổ chức")
        else:
            parts.append(
                f"{tour_name} hiện chưa có thông tin đơn vị tổ chức rõ ràng trong dữ liệu"
            )
        if includes:
            parts.append(
                "bao gồm các điểm/hoạt động chính: "
                + ", ".join(dict.fromkeys(includes))
            )
        return ". ".join(parts).rstrip(".") + "."

    def _answer_nearby_reason_if_possible(self, state: PipelineRunState) -> str | None:
        q = normalize_text(state.user_query, strip_punct=True)
        if not any(token in q for token in ["thuan tien", "diem dung chan", "vi ly do"]):
            return None
        if "tham quan" not in q and "ghe tham" not in q:
            return None

        lines = [
            str(item or "").strip()
            for item in (state.raw_context or [])
            if str(item or "").strip()
        ]
        located_in = ""
        near_targets: List[str] = []
        subject = ""
        for line in lines:
            loc_match = re.search(
                r"^-\s*(.+?)\s+\[LOCATED_IN\]\s*->\s*(.+?)\s*$", line
            )
            if loc_match:
                subject = loc_match.group(1).strip()
                located_in = loc_match.group(2).strip()
                continue
            near_match = re.search(
                r"^-\s*(.+?)\s+\[NEAR\]\s*->\s*(.+?)\s*$", line
            )
            if near_match:
                if not subject:
                    subject = near_match.group(1).strip()
                target = near_match.group(2).strip()
                target_norm = normalize_text(target, strip_punct=True)
                if target_norm and target_norm in q:
                    near_targets.append(target)

        near_targets = list(dict.fromkeys(near_targets))
        if not subject or len(near_targets) < 2:
            return None

        location_part = f" tại {located_in}" if located_in else ""
        return (
            f"{subject} có thể là điểm dừng chân thuận tiện vì quán nằm{location_part} "
            f"và ở gần các điểm du khách muốn tham quan: {', '.join(near_targets)}."
        )

    def _answer_description_fill_blank_if_possible(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        query = str(state.user_query or "")
        if "___" not in query:
            return None
        context_text = "\n".join(
            str(item or "") for item in (state.raw_context or [])
        )
        if not context_text:
            return None

        q_norm = normalize_text(query, strip_punct=True)
        answer = ""
        if "duoc menh danh la" in q_norm:
            for text in state.raw_context or []:
                match = re.search(
                    r"(?i)(?:được\s+mệnh\s+danh\s+là|duoc\s+menh\s+danh\s+la)\s+([^,.;\n]+)",
                    str(text or ""),
                )
                if match:
                    answer = match.group(1).strip(" ,.;:!?")
                    break
        elif "thuoc ___" in q_norm or ("thuoc" in q_norm and "___" in query):
            for text in state.raw_context or []:
                match = re.search(
                    r"(?i)(?:thuộc|thuoc)\s+([^,.;\n]+)", str(text or "")
                )
                if match:
                    answer = match.group(1).strip(" ,.;:!?")
                    break
        if answer:
            intent = (
                state.query_plan.intent if state.query_plan else state.primary_intent
            )
            state.runtime.metadata["intent"] = intent
            state.runtime.metadata["description_fill_blank_short_circuit"] = True
            state.runtime.metadata["detected_location"] = state.location
            seeds = (state.grounded_nodes or state.all_seeds or [])[:3]
            state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(seeds)
            state.runtime.metadata["route_seed_nodes"] = []
            state.runtime.metadata["graph"] = self.pipeline._build_graph_payload(
                seeds, [], intent=intent
            )
            return {"answer": answer, "metadata": state.runtime.metadata}

        if "tai ___ do" in q_norm:
            for text in state.raw_context or []:
                match = re.search(
                    r"(?i)(?:tại|tai)\s+([^,.;\n]+?)\s+(?:do|bởi|boi)\b",
                    str(text or ""),
                )
                if match:
                    answer = match.group(1).strip(" ,.;:!?")
                    if answer:
                        intent = (
                            state.query_plan.intent
                            if state.query_plan
                            else state.primary_intent
                        )
                        state.runtime.metadata["intent"] = intent
                        state.runtime.metadata[
                            "description_fill_blank_short_circuit"
                        ] = True
                        state.runtime.metadata["detected_location"] = state.location
                        seeds = (state.grounded_nodes or state.all_seeds or [])[:3]
                        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(
                            seeds
                        )
                        state.runtime.metadata["route_seed_nodes"] = []
                        state.runtime.metadata["graph"] = (
                            self.pipeline._build_graph_payload(seeds, [], intent=intent)
                        )
                        return {"answer": answer, "metadata": state.runtime.metadata}
        return None

    def _answer_shared_location_fill_blank_if_possible(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        if not self._is_shared_location_fill_blank_query(state.user_query):
            return None

        p = self.pipeline
        grounded_nodes = list(state.grounded_nodes or [])
        if not grounded_nodes:
            return None

        hint_norms = [
            normalize_text(hint, strip_punct=True)
            for hint in self._extract_shared_location_entity_hints(state.user_query)
            if hint
        ]
        focus_nodes = []
        for node in grounded_nodes:
            node_text_norm = normalize_text(
                " ".join(
                    [
                        str(node.metadata.get("name") or ""),
                        str(node.content or ""),
                        str(node.metadata.get("address") or ""),
                    ]
                )
            )
            if not hint_norms or any(
                h in node_text_norm or node_text_norm in h for h in hint_norms
            ):
                focus_nodes.append(node)

        if len(focus_nodes) < 2:
            focus_nodes = grounded_nodes[:2]

        location_counts: Dict[str, int] = {}
        for node in focus_nodes:
            location = self._location_from_node_text(node)
            if location:
                location_counts[location] = location_counts.get(location, 0) + 1
        if not location_counts:
            return None

        shared_location = sorted(
            location_counts.items(), key=lambda item: item[1], reverse=True
        )[0][0]
        names = [
            str(node.metadata.get("name") or node.content or "").strip()
            for node in focus_nodes[:2]
            if str(node.metadata.get("name") or node.content or "").strip()
        ]
        subject_text = " và ".join(names) if names else "Hai địa điểm này"
        answer = f"{subject_text} đều nằm ở {shared_location}."

        intent = (
            state.query_plan.intent if state.query_plan else state.primary_intent
        )
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["detected_location"] = state.location
        seeds = focus_nodes[:3]
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(seeds)
        state.runtime.metadata["route_seed_nodes"] = []
        state.runtime.metadata["graph"] = p._build_graph_payload(
            seeds, [], intent=intent
        )
        state.runtime.metadata["shared_location_fill_blank_short_circuit"] = True
        return {"answer": answer, "metadata": state.runtime.metadata}

    def _answer_address_lookup_if_possible(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        from graph_rag.core.intents import IntentType

        p = self.pipeline
        if not self._is_address_lookup_query(state.user_query):
            return None
        if state.primary_intent not in {
            IntentType.FOOD,
            IntentType.ENTITY_FACT,
            IntentType.DISCOVERY,
        }:
            return None
        if self._is_mixed_address_and_description_query(state.user_query):
            return None

        grounded_nodes = list(state.grounded_nodes or [])
        if not grounded_nodes:
            return None

        target_entities = []
        for entity in state.entities or []:
            if not self._is_groundable_entity(entity):
                continue
            e_name = str(entity.get("name") or "").strip()
            if e_name:
                target_entities.append(normalize_text(e_name, strip_punct=True))

        def node_name(node: Any) -> str:
            return str(node.metadata.get("name") or node.content or "").strip()

        def node_addr(node: Any) -> str:
            return str(node.metadata.get("address") or "").strip()

        focus_nodes = []
        for node in grounded_nodes:
            n_name_norm = normalize_text(node_name(node), strip_punct=True)
            if target_entities and not any(
                t in n_name_norm or n_name_norm in t for t in target_entities
            ):
                continue
            focus_nodes.append(node)

        if not focus_nodes:
            focus_nodes = grounded_nodes

        context_lines = []
        with_address = [n for n in focus_nodes if node_addr(n)]
        if with_address:
            top = with_address[0]
            top_name = node_name(top)
            top_addr = node_addr(top)
            context_lines.append(f"{top_name} address: {top_addr}")
            focused_for_map = with_address[:3]
        else:
            dish_node = None
            for node in focus_nodes:
                labels = set(
                    str(x)
                    for x in (
                        (getattr(node, "metadata", {}) or {}).get("labels") or []
                    )
                )
                if labels & {"Dish", "Specialty"}:
                    dish_node = node
                    break

            if dish_node is not None:
                restaurants = self._fetch_restaurants_serving_dish(dish_node)
                if restaurants:
                    dish_name = node_name(dish_node)
                    context_lines.append(f"Món {dish_name} có thể tìm thấy tại:")
                    for idx, r in enumerate(restaurants[:5], 1):
                        r_name = str(r.get("name") or "").strip()
                        r_addr = str(r.get("address") or "").strip()
                        line = f"{idx}. {r_name}"
                        if r_addr:
                            line += f" — {r_addr}"
                        context_lines.append(line)
                    focused_for_map = [dish_node]
                    intent = (
                        state.query_plan.intent
                        if state.query_plan
                        else state.primary_intent
                    )
                    state.runtime.metadata["intent"] = intent
                    state.runtime.metadata["detected_location"] = state.location
                    state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(
                        [dish_node]
                    )
                    state.runtime.metadata["route_seed_nodes"] = []
                    state.runtime.metadata["graph"] = p._build_graph_payload(
                        [dish_node], [], intent=intent
                    )
                    state.runtime.metadata["address_lookup_short_circuit"] = True
                    state.runtime.metadata["dish_to_restaurants"] = restaurants
                    llm_answer = self._call_llm_with_context(state, context_lines)
                    state.runtime.metadata["retrieved_context"] = context_lines
                    return {"answer": llm_answer, "metadata": state.runtime.metadata}

            top = focus_nodes[0]
            top_name = node_name(top)
            context_lines.append(f"{top_name} là một địa điểm du lịch")
            focused_for_map = focus_nodes[:1]

        intent = (
            state.query_plan.intent if state.query_plan else state.primary_intent
        )
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["detected_location"] = state.location
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(focused_for_map)
        state.runtime.metadata["route_seed_nodes"] = []
        state.runtime.metadata["graph"] = p._build_graph_payload(
            focused_for_map, [], intent=intent
        )
        state.runtime.metadata["address_lookup_short_circuit"] = True

        llm_answer = self._call_llm_with_context(state, context_lines)
        state.runtime.metadata["retrieved_context"] = context_lines

        return {"answer": llm_answer, "metadata": state.runtime.metadata}

    def _fetch_restaurants_serving_dish(
        self, dish_node: Any, limit: int = 5
    ) -> List[Dict[str, Any]]:
        p = self.pipeline
        dish_name = str(
            dish_node.metadata.get("name") or dish_node.content or ""
        ).strip()
        dish_id = str(getattr(dish_node, "id", "") or "").strip()
        if not dish_id and not dish_name:
            return []

        cypher_by_id = """
        MATCH (r:Restaurant)-[:HAS]->(d)
        WHERE d.id = $dish_id
        RETURN r.id AS id, r.name AS name, r.address AS address
        ORDER BY r.name ASC
        LIMIT $limit
        """
        cypher_by_name = """
        MATCH (r:Restaurant)-[:HAS]->(d)
        WHERE toLower(d.name) = toLower($dish_name)
           OR toLower(d.name) CONTAINS toLower($dish_name)
           OR toLower($dish_name) CONTAINS toLower(d.name)
        RETURN r.id AS id, r.name AS name, r.address AS address
        ORDER BY r.name ASC
        LIMIT $limit
        """
        try:
            with p.driver.session() as session:
                rows = session.run(
                    cypher_by_id, dish_id=dish_id, limit=int(limit)
                ).data()
                if not rows:
                    rows = session.run(
                        cypher_by_name, dish_name=dish_name, limit=int(limit)
                    ).data()
            return [
                {
                    "id": str(r.get("id") or ""),
                    "name": str(r.get("name") or "").strip(),
                    "address": str(r.get("address") or "").strip(),
                }
                for r in rows
                if str(r.get("name") or "").strip()
            ]
        except (ValueError, TypeError) as exc:
            p.logger.warning("dish_to_restaurant_query_failed: %s", str(exc))
            return []

    def _proximity_anchor_grounding_guard(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        if not state.metadata.get("proximity_anchor_required"):
            return None
        if not self._is_proximity_query(state.user_query):
            return None

        _pa = state.metadata.get("proximity_anchor") or {}
        anchor = str(
            _pa.get("text") if isinstance(_pa, dict) else _pa or ""
        ).strip()
        if not anchor or self._is_broad_location_anchor(anchor):
            return None

        anchor_norm = normalize_text(anchor, strip_punct=True)
        grounded_nodes = list(state.grounded_nodes or [])

        for node in grounded_nodes:
            metadata = getattr(node, "metadata", {}) or {}
            node_name = str(
                metadata.get("name") or getattr(node, "content", "") or ""
            ).strip()
            node_norm = normalize_text(node_name, strip_punct=True)
            if anchor_norm and node_norm and (
                anchor_norm in node_norm or node_norm in anchor_norm
            ):
                state.runtime.metadata["proximity_anchor_grounded"] = True
                return None
            if anchor_norm and node_norm:
                anchor_words = set(anchor_norm.split())
                node_words = set(node_norm.split())
                if (
                    anchor_words
                    and len(anchor_words & node_words) / len(anchor_words) >= 0.5
                ):
                    state.runtime.metadata["proximity_anchor_grounded"] = True
                    return None

        if grounded_nodes:
            state.runtime.metadata["proximity_anchor_grounded"] = False
            state.runtime.metadata["proximity_anchor_soft_skip"] = True
            return None

        anchor_meta = state.metadata.get("proximity_anchor") or {}
        if isinstance(anchor_meta, dict) and not anchor_meta.get(
            "grounding_required", True
        ):
            logger.info(
                "proximity_anchor_guard: anchor='%s' type='%s' grounding_required=False — skipping guard.",
                anchor,
                anchor_meta.get("type", "?"),
            )
            state.runtime.metadata["proximity_anchor_grounded"] = False
            state.runtime.metadata["proximity_anchor_is_generic"] = True
            return None

        state.runtime.metadata["anchor_grounding_failed"] = True
        state.runtime.metadata["proximity_anchor_grounded"] = False
        state.runtime.metadata["detected_location"] = state.location
        state.runtime.metadata["intent"] = state.primary_intent
        answer = (
            f"Xin lỗi, tôi chưa tìm thấy thông tin về {anchor} trong hệ thống "
            f"dữ liệu du lịch, nên không thể xác minh các địa điểm nằm gần nơi này. "
            "Tôi không suy đoán từ các địa điểm văn hóa/du lịch chung chung."
        )
        return {"answer": answer, "metadata": state.runtime.metadata}

    def _build_entity_fact_fallback_answer(
        self,
        state: PipelineRunState,
        candidate_nodes: List[Dict[str, Any]],
    ) -> str | None:
        from graph_rag.core.intents import IntentType

        p = self.pipeline
        if state.primary_intent != IntentType.ENTITY_FACT:
            return None
        if self._count_bulleted_lines(state.clean_context) > 0:
            return None
        if not candidate_nodes:
            return None
        if not state.raw_context:
            return None

        target_entities: List[str] = []
        for entity in state.entities or []:
            if not self._is_groundable_entity(entity):
                continue
            e_name = str(entity.get("name") or "").strip()
            if e_name:
                target_entities.append(normalize_text(e_name, strip_punct=True))

        selected = candidate_nodes[0]
        for node in candidate_nodes:
            node_name_norm = normalize_text(
                str(node.get("name") or ""), strip_punct=True
            )
            if not node_name_norm:
                continue
            if target_entities and any(
                t in node_name_norm or node_name_norm in t for t in target_entities
            ):
                selected = node
                break

        name = str(selected.get("name") or "Địa điểm này").strip()
        address = str(
            selected.get("address") or selected.get("location") or ""
        ).strip()
        detected_location = str(state.location or "").strip()

        if address:
            if self._is_address_lookup_query(state.user_query):
                return f"{name} nằm tại: {address}."
            return f"{name} thuộc khu vực {address}."

        lat = selected.get("lat")
        lng = selected.get("lng")
        if lat is not None and lng is not None:
            if detected_location:
                return (
                    f"{name} thuộc khu vực {detected_location}. "
                    f"Tọa độ tham chiếu: {float(lat):.6f}, {float(lng):.6f}."
                )
            return (
                f"{name} có tọa độ tham chiếu: {float(lat):.6f}, {float(lng):.6f}."
            )

        if detected_location:
            return f"{name} thuộc khu vực {detected_location}."
        return (
            f"Mình đã xác định được địa điểm {name}, nhưng hiện chưa có "
            f"địa chỉ chi tiết trong hệ thống dữ liệu du lịch."
        )

    def _answer_location_type_if_possible(
        self, state: PipelineRunState
    ) -> str | None:
        """Answer 'X nằm ở đâu?', 'X thuộc loại gì?' from raw_context."""
        q = normalize_text(state.user_query, strip_punct=True)
        if any(
            m in q
            for m in [
                "ngu o dau",
                "an o dau",
                "tim noi ngu",
                "tim khach san",
                "tim nha nghi",
                "o dau tot",
                "o dau dep",
            ]
        ):
            return None
        is_location_query = any(
            token in q
            for token in [
                "nam o dau",
                "o dau",
                "vi tri",
                "dia chi",
                "thuoc khu vuc",
                "nam tai",
                "nam trong",
                "thuoc tinh",
                "dia diem",
            ]
        )
        is_type_query = any(
            token in q
            for token in [
                "thuoc loai",
                "loai hinh",
                "la gi",
                "la loai",
                "phan loai",
                "thuoc nhom",
                "danh muc",
            ]
        )
        if not is_location_query and not is_type_query:
            return None

        context_lines: List[str] = []
        for item in state.raw_context or []:
            context_lines.extend(
                line.strip() for line in str(item or "").splitlines() if line.strip()
            )
        if not context_lines:
            return None

        subject = ""
        subject_type = ""
        located_in = ""
        belongs_to = ""
        address = ""
        description = ""
        category_from_context = ""

        for line in context_lines:
            header_match = self._HEADER_RE.search(line)
            if header_match:
                subject = header_match.group(1).strip()
                subject_type = header_match.group(2).strip()
                continue
            loc_match = re.search(
                r"^-\s*(.+?)\s+\[LOCATED_IN\]\s*->\s*(.+?)\s*$", line
            )
            if loc_match:
                subject = subject or loc_match.group(1).strip()
                located_in = loc_match.group(2).strip()
                continue
            cat_match = re.search(
                r"^-\s*(.+?)\s+\[BELONGS_TO\]\s*->\s*(.+?)\s*$", line
            )
            if cat_match:
                subject = subject or cat_match.group(1).strip()
                belongs_to = cat_match.group(2).strip()
                continue
            if not category_from_context:
                cat_line_match = re.search(
                    r"(?:category|loai hinh|phan loai|type)\s*[:=]\s*(.+?)$",
                    line,
                    re.IGNORECASE,
                )
                if cat_line_match:
                    category_from_context = cat_line_match.group(1).strip(" .;:,")
            addr_match = re.search(r"^-\s*address:\s*(.+?)\s*$", line)
            if addr_match:
                address = addr_match.group(1).strip()
                continue
            if len(line) > 50 and not line.startswith("-") and not description:
                description = line

        if not subject:
            for entity in state.entities or []:
                if isinstance(entity, dict) and str(
                    entity.get("name") or ""
                ).strip():
                    subject = str(entity.get("name")).strip()
                    break

        if not subject:
            return None

        # Fallback: get subject_type and category from grounded nodes
        if not subject_type or not belongs_to:
            for node in state.grounded_nodes or state.all_seeds or []:
                node_meta = getattr(node, "metadata", {}) or {}
                node_name = str(
                    node_meta.get("name") or getattr(node, "content", "") or ""
                ).strip()
                node_norm = normalize_text(node_name, strip_punct=True)
                subject_norm = normalize_text(subject, strip_punct=True)
                if subject_norm and (
                    subject_norm in node_norm or node_norm in subject_norm
                ):
                    if not subject_type:
                        labels = node_meta.get("labels") or []
                        if labels:
                            subject_type = str(labels[0])
                    if not belongs_to:
                        cat = (
                            node_meta.get("category")
                            or node_meta.get("type")
                            or node_meta.get("loai_hinh")
                            or node_meta.get("loai_hinh_du_lich")
                            or ""
                        )
                        if (
                            cat
                            and str(cat).strip()
                            and str(cat).strip() not in _TECHNICAL_LABELS
                        ):
                            belongs_to = str(cat).strip()
                    break

        # Fallback: query BELONGS_TO from Neo4j if still no category
        if is_type_query and not belongs_to and subject:
            try:
                p = self.pipeline
                if hasattr(p, "graph_store") and p.graph_store:
                    cypher = (
                        "MATCH (n)-[:BELONGS_TO]->(cat) "
                        "WHERE n.name = $name OR n.name CONTAINS $name "
                        "RETURN cat.name AS category LIMIT 1"
                    )
                    result = p.graph_store.query(cypher, {"name": subject})
                    if result and result[0].get("category"):
                        belongs_to = str(result[0]["category"]).strip()
            except (ValueError, TypeError):
                pass

        parts: List[str] = []
        if is_location_query:
            if located_in:
                parts.append(f"{subject} nằm tại {located_in}.")
            if address:
                parts.append(f"Địa chỉ: {address}.")
        if is_type_query:
            if belongs_to:
                parts.append(f"{subject} thuộc loại {belongs_to}.")
            elif category_from_context:
                parts.append(f"{subject} thuộc loại {category_from_context}.")
            elif subject_type and subject_type not in _TECHNICAL_LABELS:
                parts.append(f"{subject} thuộc loại {subject_type}.")
            elif subject_type:
                display = _LABEL_DISPLAY.get(subject_type, subject_type.lower())
                parts.append(f"{subject} thuộc loại {display}.")

        if not parts:
            return None
        return " ".join(parts)

    def _answer_menu_items_if_possible(self, state: PipelineRunState) -> str | None:
        q = normalize_text(state.user_query, strip_punct=True)
        if not any(
            token in q
            for token in [
                "mon",
                "thuc don",
                "menu",
                "phuc vu",
                "dac san",
                "mon an",
                "mon ngon",
                "an gi",
                "goi y mon",
                "co mon",
            ]
        ):
            return None

        context_lines: List[str] = []
        for item in state.raw_context or []:
            context_lines.extend(
                line.strip() for line in str(item or "").splitlines() if line.strip()
            )
        if not context_lines:
            return None

        subject = ""
        dishes: List[str] = []
        for line in context_lines:
            header_match = self._HEADER_RE.search(line)
            if header_match:
                subject = header_match.group(1).strip()
                continue
            has_match = re.search(
                r"^-\s*(.+?)\s+\[HAS\]\s*->\s*(.+?)\s*$", line
            )
            if has_match:
                subject = subject or has_match.group(1).strip()
                dishes.append(has_match.group(2).strip())

        dishes = list(dict.fromkeys(dishes))
        if not dishes:
            return None

        subject = subject or "Nhà hàng này"
        return f"{subject} phục vụ các món: {', '.join(dishes)}."

    def _answer_dish_to_restaurant_if_possible(
        self, state: PipelineRunState
    ) -> str | None:
        if (state.metadata or {}).get("retrieval_plan_mode") != "dish_to_restaurant":
            return None

        frame = (state.metadata or {}).get("query_frame") or {}
        plan = frame.get("retrieval_plan") or {}
        policy = plan.get("context_policy") or {}
        dishes = [
            str(item or "").strip()
            for item in (policy.get("dish_constraints") or [])
            if str(item or "").strip()
        ]
        if not dishes:
            target = str(
                (state.metadata or {}).get("target_entity") or ""
            ).strip()
            if target:
                dishes = [target]
        if not dishes:
            return None

        def norm(value: str) -> str:
            return normalize_text(value, strip_punct=True)

        dish_norms = {norm(dish): dish for dish in dishes}
        matches: Dict[str, List[str]] = {dish: [] for dish in dishes}
        context_items = list(state.raw_context or [])
        clean = state.clean_context or ""
        if isinstance(clean, str):
            context_items.extend(clean.splitlines())
        elif isinstance(clean, list):
            context_items.extend(str(item or "") for item in clean)

        for item in context_items:
            line = str(item or "").strip()
            if not line:
                continue
            candidates = [
                re.search(
                    r"^-?\s*(.+?)\s+\[HAS\]\s*->\s*(.+?)\s*$", line
                ),
                re.search(
                    r"^-?\s*(.+?)\s+phục vụ món\s+(?:\(HAS\)\s*)?(.+?)\s*(?:\(|$)",
                    line,
                    re.IGNORECASE,
                ),
                re.search(
                    r"^-?\s*(.+?)\s+phuc vu mon\s+(?:\(HAS\)\s*)?(.+?)\s*(?:\(|$)",
                    line,
                    re.IGNORECASE,
                ),
            ]
            for match in candidates:
                if not match:
                    continue
                left = match.group(1).strip(" -")
                right = match.group(2).strip(" .")
                left_norm = norm(left)
                right_norm = norm(right)
                for dish_norm, dish in dish_norms.items():
                    if dish_norm and dish_norm in left_norm:
                        matches.setdefault(dish, []).append(right)
                    elif dish_norm and dish_norm in right_norm:
                        matches.setdefault(dish, []).append(left)

        dish_descriptions: Dict[str, str] = {}
        current_entity_norm = ""
        for item in context_items:
            line = str(item or "").strip()
            header_match = self._HEADER_RE.search(line)
            if header_match:
                current_entity_norm = norm(header_match.group(1).strip())
                continue
            desc_match = re.search(
                r"^-\s*(?:description|mo_ta):\s*(.+)$", line, re.IGNORECASE
            )
            if desc_match:
                desc_text = desc_match.group(1).strip()
                for dish_norm, dish in dish_norms.items():
                    if dish_norm and (
                        dish_norm in current_entity_norm
                        or current_entity_norm in dish_norm
                    ):
                        dish_descriptions[dish] = desc_text
                        break

        lines: List[str] = []
        for dish in dishes:
            restaurants = []
            seen_restaurant_norms = set()
            for restaurant in matches.get(dish) or []:
                restaurant_text = str(restaurant or "").strip()
                restaurant_norm = norm(restaurant_text)
                if not restaurant_text or restaurant_norm in seen_restaurant_norms:
                    continue
                seen_restaurant_norms.add(restaurant_norm)
                restaurants.append(restaurant_text)

            dish_entry = f"**{dish}**"
            if dish in dish_descriptions:
                dish_entry += f": {dish_descriptions[dish]}"
            if restaurants:
                dish_entry += f" (có tại: {', '.join(restaurants)})"
            lines.append(dish_entry)

        if not lines:
            return None

        header = "Đây là một số đặc sản bạn có thể mua về làm quà:\n"
        return header + "\n".join(f"- {line}" for line in lines)

    def _answer_travel_info_topic_deterministic(
        self, state: PipelineRunState
    ) -> str | None:
        metadata = state.metadata or {}
        policy = metadata.get("fallback_policy") or ""

        policy_to_topics = {
            "emergency_guided_fallback": ["emergency"],
            "payment_guided_fallback": ["payment"],
            "booking_guided_fallback": ["accommodation_tips", "booking"],
            "weather_guided_fallback": ["weather"],
            "transport_local_guided_fallback": ["transport", "transport_local"],
            "airport_guided_fallback": ["airport"],
            "seafood_shopping_guided_fallback": ["shopping", "seafood"],
            "budget_guided_fallback": ["budget"],
            "health_guided_fallback": ["health"],
            "community_guided_fallback": ["community"],
            "event_schedule_guided_fallback": ["event"],
            "general_practical_guided_fallback": ["general"],
        }

        target_topics = policy_to_topics.get(policy)
        if not target_topics:
            if policy.endswith("_guided_fallback"):
                target_topics = [policy.split("_guided_fallback")[0]]
            else:
                return None

        nodes = []
        seen_ids = set()

        def add_node(n):
            if not n:
                return
            n_id = None
            if hasattr(n, "id"):
                n_id = str(n.id)
            elif isinstance(n, dict):
                n_id = str(n.get("id") or "")
            if n_id and n_id in seen_ids:
                return
            labels = []
            topic = ""
            desc = ""
            name = ""
            if hasattr(n, "metadata") and n.metadata:
                labels = n.metadata.get("labels") or [n.metadata.get("type", "")]
                topic = n.metadata.get("topic") or ""
                desc = n.metadata.get("description") or getattr(n, "content", "") or ""
                name = n.metadata.get("name") or ""
            elif isinstance(n, dict):
                labels = n.get("labels") or [n.get("type", "")]
                topic = n.get("topic") or ""
                desc = n.get("description") or n.get("content") or ""
                name = n.get("name") or ""
            else:
                return
            labels = [str(l).strip() for l in labels]
            is_travel_info = "TravelInfo" in labels or topic in [
                "emergency", "payment", "weather", "transport", "airport",
                "shopping", "budget", "health", "community", "event", "general",
            ]
            topic_matched = False
            if target_topics:
                topic_matched = any(
                    topic.lower() == t for t in target_topics
                ) or any(t in topic.lower() for t in target_topics)
            else:
                topic_matched = is_travel_info
            if is_travel_info and topic_matched and desc:
                nodes.append((name, desc, topic))
                if n_id:
                    seen_ids.add(n_id)

        for n in state.grounded_nodes or []:
            add_node(n)
        for n in state.all_seeds or []:
            add_node(n)

        if nodes:
            lines = []
            for name, desc, topic in nodes:
                lines.append(desc.strip())
            return "\n\n".join(lines).strip()

        return None
