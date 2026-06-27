"""Proximity / nearby answer mixin — accommodation, cultural categories, and itinerary."""

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
import logging
import re
from typing import Any, Dict, List, Optional

from graph_rag.core.intents import IntentType
from graph_rag.core import keywords
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)

from .dto import PipelineRunState


class DeterministicProximityMixin:
    """Mixin for proximity-based deterministic answers (nearby, itinerary, cultural)."""

    # Inherited helpers (provided by PipelineApplicationService):
    #   _resolve_nearby_anchor_node, _fetch_nearby_accommodations_ranked,
    #   _is_nearby_accommodation_itinerary_query, _fetch_nearby_non_lodging_nodes,
    #   _build_nearby_accommodation_itinerary_answer, _build_accommodation_based_itinerary,
    #   _resolve_proximity_category_anchor_node, _fetch_nearby_cultural_categories,
    #   _is_nearby_cultural_category_query, _is_nearby_accommodation_query

    def _answer_nearby_accommodation_if_possible(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        p = self.pipeline
        logger.debug(
            "_accom_debug: intent=%s, is_nearby=%s",
            state.primary_intent,
            self._is_nearby_accommodation_query(state.user_query),
        )
        if state.primary_intent != IntentType.ACCOMMODATION:
            logger.debug("_accom_debug: skipped, intent is not ACCOMMODATION")
            return None
        if not self._is_nearby_accommodation_query(state.user_query):
            logger.debug("_accom_debug: skipped, not nearby accommodation query")
            return None

        # Skip itinerary generation for comparison questions
        q_norm = normalize_text(state.user_query, strip_punct=True)
        if "so sanh" in q_norm:
            return None
        frame = (state.metadata or {}).get("query_frame") or {}
        if frame.get("query_operator") == "comparison":
            return None
        # Skip for multiple-choice / multi-select questions
        if re.search(r"(?im)^\s*[A-D]\s*[\).:-]\s+\S+", state.user_query or ""):
            logger.debug("_accom_debug: skipped, question has inline choices")
            return None
        if (state.metadata or {}).get("choices"):
            logger.debug("_accom_debug: skipped, question has choices in metadata")
            return None

        norm = lambda t: normalize_text(t, strip_punct=True)
        anchor_names = []
        for entity in state.entities or []:
            if not isinstance(entity, dict):
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            if e_type in {"location", "accommodation"}:
                continue
            e_name = str(entity.get("name") or "").strip()
            if e_name:
                anchor_names.append(norm(e_name))

        anchor_node = self._resolve_nearby_anchor_node(state)
        logger.debug(
            "_accom_debug: anchor_node=%s, location='%s'",
            anchor_node is not None,
            state.location,
        )
        if anchor_node is None:
            q_norm = normalize_text(state.user_query, strip_punct=True)
            is_followup = any(
                m in q_norm
                for m in ["ngu o dau", "tim noi ngu", "tim khach san", "tim nha nghi"]
            )
            logger.debug("_accom_debug: no anchor, is_followup=%s", is_followup)
            if not is_followup:
                return None

            location = (
                (state.metadata or {}).get("user_provided_location") or ""
            )
            if not location:
                location = state.location or ""
            if not location:
                return None
            try:
                with p.driver.session() as session:
                    rows = session.run(
                        "MATCH (a:Accommodation)-[:LOCATED_IN]->(l:Location) "
                        "WHERE toLower(l.name) CONTAINS toLower($loc) "
                        "RETURN a.name AS name, a.address AS address "
                        "LIMIT 5",
                        loc=location,
                    ).data()
                    if not rows:
                        rows = session.run(
                            "MATCH (a:Accommodation) "
                            "WHERE toLower(coalesce(a.address, '')) CONTAINS toLower($loc) "
                            "RETURN a.name AS name, a.address AS address "
                            "LIMIT 5",
                            loc=location,
                        ).data()
            except (ValueError, TypeError) as exc:
                p.logger.warning("accom_followup_query_failed: %s", str(exc))
                return None
            if not rows:
                return None
            lines = [f"Mình tìm thấy một số chỗ nghỉ khu vực {location}:"]
            for i, row in enumerate(rows, 1):
                name = str(row.get("name") or "").strip()
                addr = str(row.get("address") or "").strip()
                line = f"{i}. {name}"
                if addr:
                    line += f" - {addr}"
                lines.append(line)
            intent = state.query_plan.intent if state.query_plan else state.primary_intent
            state.runtime.metadata["intent"] = intent
            return {"answer": "\n".join(lines), "metadata": state.runtime.metadata}

        if not anchor_names:
            anchor_names.append(
                norm(str(anchor_node.metadata.get("name") or anchor_node.content or ""))
            )

        anchor_name = str(
            anchor_node.metadata.get("name") or anchor_node.content or "Địa điểm"
        )
        anchor_labels = set((anchor_node.metadata.get("labels") or []))
        is_anchor_accommodation = "Accommodation" in anchor_labels

        ranked_hotels = self._fetch_nearby_accommodations_ranked(
            anchor_node, limit=5, max_distance_m=15000
        )

        if self._is_nearby_accommodation_itinerary_query(state.user_query):
            if is_anchor_accommodation:
                itinerary = self._build_accommodation_based_itinerary(
                    state, anchor_node, anchor_name
                )
                if itinerary is not None:
                    return itinerary
            else:
                itinerary = self._build_nearby_accommodation_itinerary_answer(
                    state=state,
                    anchor_node=anchor_node,
                    anchor_name=anchor_name,
                    ranked_hotels=ranked_hotels,
                )
                if itinerary is not None:
                    return itinerary

        lines = [f"Các khách sạn gần {anchor_name} (xếp theo khoảng cách):"]
        seed_nodes_for_meta: List[Any] = []
        if ranked_hotels:
            for idx, item in enumerate(ranked_hotels, 1):
                name = str(item.get("name") or "").strip()
                address = str(item.get("address") or "").strip()
                distance_km = item.get("distance_km")
                line = f"{idx}. {name}"
                if distance_km is not None:
                    line += f" (~{distance_km:.2f} km)"
                if address:
                    line += f" - {address}"
                lines.append(line)
            metadata_nearby_hotels = ranked_hotels
        else:
            nearby_nodes = p.retriever._proximity_anchor_search(
                [anchor_node],
                target_labels=["Accommodation"],
                top_k=5,
                max_distance_m=5000,
            )
            if not nearby_nodes:
                return None
            for idx, node in enumerate(nearby_nodes[:5], 1):
                name = str(node.metadata.get("name") or node.content or "")
                address = str(node.metadata.get("address") or "").strip()
                line = f"{idx}. {name}"
                if address:
                    line += f" - {address}"
                lines.append(line)
            lines.append(
                "(Chưa đủ tọa độ để xếp hạng chính xác theo km, "
                "danh sách trên theo mức độ gần trong đồ thị.)"
            )
            seed_nodes_for_meta = nearby_nodes
            metadata_nearby_hotels = [
                {
                    "id": str(node.id),
                    "name": str(node.metadata.get("name") or node.content or ""),
                    "address": str(node.metadata.get("address") or "").strip(),
                    "distance_km": None,
                }
                for node in nearby_nodes
            ]

        intent = state.query_plan.intent if state.query_plan else state.primary_intent
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["nearby_short_circuit"] = True
        state.runtime.metadata["nearby_anchor"] = {
            "id": str(anchor_node.id),
            "name": anchor_name,
            "labels": anchor_node.metadata.get("labels", []),
        }
        state.runtime.metadata["nearby_hotels"] = metadata_nearby_hotels

        from types import SimpleNamespace

        seed_nodes_for_meta = [
            SimpleNamespace(
                id=item.get("id"),
                content=item.get("name", ""),
                metadata={
                    "name": item.get("name"),
                    "labels": ["Accommodation"],
                    "lat": item.get("lat"),
                    "lng": item.get("lng"),
                    "address": item.get("address"),
                },
            )
            for item in ranked_hotels
            if item.get("lat") is not None and item.get("lng") is not None
        ]

        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(seed_nodes_for_meta)
        state.runtime.metadata["route_seed_nodes"] = []
        state.runtime.metadata["graph"] = p._build_graph_payload(
            seed_nodes_for_meta, [], intent=intent
        )
        state.runtime.metadata["detected_location"] = state.location
        return {"answer": "\n".join(lines), "metadata": state.runtime.metadata}

    def _resolve_proximity_category_anchor_node(self, state: PipelineRunState) -> Any | None:
        candidates = list(state.grounded_nodes or [])
        if not candidates:
            return None

        entity_names = []
        for entity in state.entities or []:
            if not isinstance(entity, dict):
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            if e_type in {"location", "province", "city", "district", "ward", "commune"}:
                continue
            e_name = str(entity.get("name") or "").strip()
            if e_name:
                entity_names.append(normalize_text(e_name, strip_punct=True))

        preferred_labels = {"Accommodation", "Restaurant", "TouristAttraction"}

        def labels(node: Any) -> set:
            metadata = getattr(node, "metadata", {}) or {}
            return {
                str(x)
                for x in (metadata.get("labels") or [metadata.get("type", "")])
                if str(x).strip()
            }

        def name_norm(node: Any) -> str:
            metadata = getattr(node, "metadata", {}) or {}
            return normalize_text(
                str(metadata.get("name") or getattr(node, "content", "") or ""),
                strip_punct=True,
            )

        for node in candidates:
            node_labels = labels(node)
            if node_labels.isdisjoint(preferred_labels):
                continue
            n_name = name_norm(node)
            if entity_names and any(
                e and (e in n_name or n_name in e) for e in entity_names
            ):
                return node

        for node in candidates:
            if not labels(node).isdisjoint(preferred_labels):
                return node

        return None

    def _fetch_nearby_cultural_categories(
        self, anchor_node: Any, limit: int = 8
    ) -> List[Dict[str, Any]]:
        p = self.pipeline
        anchor_id = str(getattr(anchor_node, "id", "") or "").strip()
        if not anchor_id:
            return []

        cypher_near = """
        MATCH (anchor)
        WHERE anchor.id = $anchor_id
        MATCH (anchor)-[:NEAR]-(poi:TouristAttraction)
        OPTIONAL MATCH (poi)-[:BELONGS_TO]->(category)
        RETURN poi.id AS id,
               poi.name AS name,
               poi.address AS address,
               category.name AS category,
               'NEAR' AS evidence
        ORDER BY poi.name ASC
        LIMIT $limit
        """
        cypher_same_location = """
        MATCH (anchor)
        WHERE anchor.id = $anchor_id
        OPTIONAL MATCH (anchor)-[:LOCATED_IN]->(loc)
        WITH anchor, loc
        MATCH (poi:TouristAttraction)
        WHERE (
            loc IS NOT NULL AND EXISTS { MATCH (poi)-[:LOCATED_IN]->(loc) }
        ) OR (
            loc IS NOT NULL
            AND toLower(coalesce(poi.address, '')) CONTAINS toLower(coalesce(loc.name, ''))
        ) OR (
            loc IS NULL
            AND coalesce(anchor.address, '') <> ''
            AND coalesce(poi.address, '') <> ''
            AND any(part IN split(toLower(anchor.address), ',')
                    WHERE size(trim(part)) >= 6 AND toLower(poi.address) CONTAINS trim(part))
        )
        OPTIONAL MATCH (poi)-[:BELONGS_TO]->(category)
        RETURN poi.id AS id,
               poi.name AS name,
               poi.address AS address,
               category.name AS category,
               CASE WHEN loc IS NULL THEN 'ADDRESS_AREA' ELSE 'LOCATED_IN' END AS evidence
        ORDER BY poi.name ASC
        LIMIT $limit
        """

        def rows_to_items(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            items = []
            seen = set()
            for row in rows or []:
                poi_id = str(row.get("id") or "").strip()
                name = str(row.get("name") or "").strip()
                if not poi_id or not name or poi_id in seen:
                    continue
                seen.add(poi_id)
                items.append(
                    {
                        "id": poi_id,
                        "name": name,
                        "address": str(row.get("address") or "").strip(),
                        "category": str(row.get("category") or "").strip() or "Chưa rõ loại hình",
                        "evidence": str(row.get("evidence") or "").strip(),
                    }
                )
            return items

        try:
            with p.driver.session() as session:
                rows = session.run(cypher_near, anchor_id=anchor_id, limit=int(limit)).data()
                items = rows_to_items(rows)
                if items:
                    return items
                rows = session.run(
                    cypher_same_location, anchor_id=anchor_id, limit=int(limit)
                ).data()
                return rows_to_items(rows)
        except (Neo4jClientError, ServiceUnavailable) as exc:
            p.logger.warning("nearby_cultural_category_query_failed: %s", str(exc))
            return []

    def _answer_nearby_cultural_categories_if_possible(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        if not self._is_nearby_cultural_category_query(state.user_query):
            return None

        p = self.pipeline
        anchor_node = self._resolve_proximity_category_anchor_node(state)
        if anchor_node is None:
            return None

        anchor_name = str(
            anchor_node.metadata.get("name") or anchor_node.content or "địa điểm này"
        )
        items = self._fetch_nearby_cultural_categories(anchor_node, limit=8)
        if not items:
            state.runtime.metadata["nearby_cultural_fallback_failed"] = True
            state.runtime.metadata["nearby_cultural_anchor"] = {
                "id": str(anchor_node.id),
                "name": anchor_name,
                "labels": anchor_node.metadata.get("labels", []),
            }
            return None

        category_to_places: Dict[str, List[str]] = {}
        for item in items:
            category_to_places.setdefault(item["category"], []).append(item["name"])

        evidence = items[0].get("evidence") or ""
        if evidence == "NEAR":
            source_text = "dựa trên dữ liệu liên kết lân cận hiện có"
        elif evidence == "LOCATED_IN":
            source_text = "dựa trên các điểm cùng đơn vị hành chính trong dữ liệu hiện có"
        else:
            source_text = "dựa trên vùng địa chỉ trùng khớp trong dữ liệu hiện có"

        lines = [
            f"Gần {anchor_name}, khách du lịch có thể kết hợp tham quan "
            f"các loại hình văn hóa sau ({source_text}):"
        ]
        for category, places in category_to_places.items():
            place_text = ", ".join(places[:3])
            lines.append(f"- {category}: {place_text}")

        intent = state.query_plan.intent if state.query_plan else state.primary_intent
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["nearby_cultural_short_circuit"] = True
        state.runtime.metadata["nearby_cultural_anchor"] = {
            "id": str(anchor_node.id),
            "name": anchor_name,
            "labels": anchor_node.metadata.get("labels", []),
        }
        state.runtime.metadata["nearby_cultural_items"] = items
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata([anchor_node])
        state.runtime.metadata["route_seed_nodes"] = []
        state.runtime.metadata["graph"] = p._build_graph_payload(
            [anchor_node], [], intent=intent
        )
        state.runtime.metadata["detected_location"] = state.location
        return {"answer": "\n".join(lines), "metadata": state.runtime.metadata}

    def _resolve_nearby_anchor_node(self, state: PipelineRunState) -> Any | None:
        norm = lambda t: normalize_text(t, strip_punct=True)
        anchor_names = []
        for entity in state.entities or []:
            if not isinstance(entity, dict):
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            if e_type in {"location", "accommodation"}:
                continue
            e_name = str(entity.get("name") or "").strip()
            if e_name:
                anchor_names.append(norm(e_name))

        candidates = list(state.grounded_nodes or [])
        if not candidates and self._is_deictic_reference_query(state.user_query):
            fallback_anchor = (
                state.metadata.get("active_grounded_anchor")
                or self.pipeline.conversation_state.get("last_grounded_anchor")
                or {}
            )
            fallback_node = self._build_anchor_node(fallback_anchor)
            if fallback_node is not None:
                candidates = [fallback_node]
                if not anchor_names:
                    anchor_names.append(
                        norm(str(fallback_node.metadata.get("name") or fallback_node.content or ""))
                    )

        if not candidates or not anchor_names:
            for entity in state.entities or []:
                if not isinstance(entity, dict):
                    continue
                e_type = str(entity.get("type") or "").strip().lower()
                e_name = str(entity.get("name") or "").strip()
                if e_name and e_type in {"accommodation", "hotel", "lodging"}:
                    for node in candidates:
                        n_name = norm(str(node.metadata.get("name") or node.content or ""))
                        if norm(e_name) in n_name or n_name in norm(e_name):
                            return node
                    try:
                        with self.pipeline.driver.session() as session:
                            row = session.run(
                                "MATCH (n:Accommodation) WHERE toLower(n.name) = toLower($name) "
                                "OR toLower(n.name) CONTAINS toLower($name) RETURN n LIMIT 1",
                                name=e_name,
                            ).single()
                        if row:
                            node_data = row["n"]
                            return self._build_anchor_node(
                                {
                                    "id": node_data.get("id", ""),
                                    "name": node_data.get("name", ""),
                                    "content": node_data.get("name", ""),
                                    "labels": list(node_data.labels),
                                    "metadata": dict(node_data),
                                }
                            )
                    except (Neo4jClientError, ServiceUnavailable) as e:
                        logger.error("   -> [Deterministic] Neo4j node lookup failed: %s", e)
            return None

        def _labels(node: Any) -> List[str]:
            labels = (getattr(node, "metadata", {}) or {}).get("labels") or []
            return [str(x) for x in labels]

        for node in candidates:
            labels = set(_labels(node))
            if "Accommodation" in labels:
                continue
            n_name = norm(str(node.metadata.get("name") or node.content or ""))
            if any(a and (a in n_name or n_name in a) for a in anchor_names):
                return node

        for entity in state.entities or []:
            if not isinstance(entity, dict):
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            e_name = str(entity.get("name") or "").strip()
            if e_name and e_type in {"accommodation", "hotel", "lodging"}:
                try:
                    with self.pipeline.driver.session() as session:
                        row = session.run(
                            "MATCH (n:Accommodation) WHERE toLower(n.name) = toLower($name) "
                            "OR toLower(n.name) CONTAINS toLower($name) RETURN n LIMIT 1",
                            name=e_name,
                        ).single()
                    if row:
                        node_data = row["n"]
                        return self._build_anchor_node(
                            {
                                "id": node_data.get("id", ""),
                                "name": node_data.get("name", ""),
                                "content": node_data.get("name", ""),
                                "labels": list(node_data.labels),
                                "metadata": dict(node_data),
                            }
                        )
                except (Neo4jClientError, ServiceUnavailable) as e:
                    logger.error("   -> [Deterministic] Neo4j node lookup failed: %s", e)

        return None

    def _fetch_nearby_accommodations_by_near_relation(
        self,
        anchor_node: Any,
        *,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        p = self.pipeline
        anchor_id = str(getattr(anchor_node, "id", "") or "").strip()
        if not anchor_id:
            return []

        cypher = """
        MATCH (anchor)
        WHERE anchor.id = $anchor_id
        MATCH (a:Accommodation)-[:NEAR]-(anchor)
        RETURN a.id AS id,
            a.name AS name,
            a.address AS address
        ORDER BY a.name ASC
        LIMIT $limit
        """

        try:
            with p.driver.session() as session:
                rows = session.run(cypher, anchor_id=anchor_id, limit=int(limit)).data()
            return [
                {
                    "id": str(r.get("id") or ""),
                    "name": str(r.get("name") or "").strip(),
                    "address": str(r.get("address") or "").strip(),
                    "distance_km": None,
                    "evidence": "NEAR",
                }
                for r in rows
                if str(r.get("name") or "").strip()
            ]
        except (ValueError, TypeError) as exc:
            p.logger.warning("nearby_accommodation_near_relation_query_failed: %s", str(exc))
            return []

    def _is_nearby_accommodation_itinerary_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        return any(
            marker in q
            for marker in [
                "lich trinh",
                "nua ngay",
                "tham quan",
                "bat dau tu",
                "goi y mot lich",
                "goi y lich",
            ]
        )

    def _fetch_nearby_non_lodging_nodes(
        self, anchor_node: Any, limit: int = 8
    ) -> List[Dict[str, Any]]:
        p = self.pipeline
        cypher = """
        MATCH (anchor {id: $anchor_id})
        OPTIONAL MATCH (anchor)-[:LOCATED_IN]->(anchor_admin)
        OPTIONAL MATCH (anchor)-[:NEAR]-(n)
        WHERE NOT n:Accommodation
        OPTIONAL MATCH (n)-[:LOCATED_IN]->(n_admin)
        WITH anchor, n, anchor_admin, n_admin
        WHERE n IS NULL
           OR anchor_admin IS NULL
           OR n_admin IS NULL
           OR anchor_admin.name = n_admin.name
           OR toLower(coalesce(n.address, '')) CONTAINS toLower(coalesce(anchor.address, ''))
        RETURN n.id AS id,
               n.name AS name,
               labels(n) AS labels,
               n.address AS address,
               n.type AS type
        LIMIT $limit
        """
        try:
            with p.driver.session() as session:
                rows = session.run(cypher, anchor_id=str(anchor_node.id), limit=int(limit)).data()
            return [
                {
                    "id": str(row.get("id") or ""),
                    "name": str(row.get("name") or "").strip(),
                    "labels": list(row.get("labels") or []),
                    "address": str(row.get("address") or "").strip(),
                    "type": str(row.get("type") or "").strip(),
                }
                for row in rows
                if str(row.get("name") or "").strip()
            ]
        except (ValueError, TypeError) as exc:
            p.logger.warning("nearby_non_lodging_query_failed: %s", str(exc))
            return []

    def _build_nearby_accommodation_itinerary_answer(
        self,
        state: PipelineRunState,
        anchor_node: Any,
        anchor_name: str,
        ranked_hotels: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        p = self.pipeline
        hotel = ranked_hotels[0] if ranked_hotels else None
        if hotel is None:
            fallback_hotels = p.retriever._proximity_anchor_search(
                [anchor_node],
                target_labels=["Accommodation"],
                top_k=1,
                max_distance_m=5000,
            )
            if fallback_hotels:
                node = fallback_hotels[0]
                hotel = {
                    "name": str(node.metadata.get("name") or node.content or "").strip(),
                    "address": str(node.metadata.get("address") or "").strip(),
                    "distance_km": None,
                }
        if not hotel or not str(hotel.get("name") or "").strip():
            return None

        nearby = self._fetch_nearby_non_lodging_nodes(anchor_node, limit=8)
        attractions = [
            item
            for item in nearby
            if "TouristAttraction" in set(item.get("labels") or [])
        ]
        restaurants = [
            item
            for item in nearby
            if "Restaurant" in set(item.get("labels") or [])
        ]
        extra = attractions[0] if attractions else (restaurants[0] if restaurants else None)

        hotel_name = str(hotel.get("name") or "").strip()
        distance_km = hotel.get("distance_km")
        hotel_suffix = (
            f" (~{distance_km:.2f} km từ {anchor_name})" if distance_km is not None else ""
        )

        lines = [
            f"Mình gợi ý bắt đầu từ {hotel_name}{hotel_suffix}, một cơ sở lưu trú gần {anchor_name}.",
            "",
            "Lịch trình nửa ngày:",
            f"- 08:00: Xuất phát từ {hotel_name}.",
            f"- 08:20 - 10:30: Tham quan {anchor_name}.",
        ]
        if extra:
            label_text = (
                "điểm tham quan gần đó" if attractions else "điểm dừng ăn uống gần đó"
            )
            lines.append(f"- 10:45 - 11:45: Ghé {extra['name']}, {label_text} trong dữ liệu.")
        else:
            lines.append(
                "- 10:45 - 11:45: Dữ liệu hiện chưa ghi nhận thêm điểm tham quan "
                "gần đó để đưa vào lịch trình."
            )
        lines.append("- 12:00: Quay lại nơi lưu trú hoặc nghỉ trưa.")

        if not attractions:
            lines.extend(
                [
                    "",
                    f"Lưu ý: trong dữ liệu hiện có, các điểm gần {anchor_name} "
                    f"chủ yếu là cơ sở lưu trú và nhà hàng; chưa có thêm một điểm tham quan "
                    f"khác được ghi nhận rõ ràng gần đó.",
                ]
            )

        from types import SimpleNamespace

        itinerary_nodes = [anchor_node]
        for item in nearby or []:
            item_node = SimpleNamespace(
                id=item.get("id"),
                content=item.get("name", ""),
                metadata={
                    "name": item.get("name"),
                    "labels": item.get("labels", []),
                    "lat": item.get("lat"),
                    "lng": item.get("lng"),
                    "address": item.get("address"),
                },
            )
            itinerary_nodes.append(item_node)

        state.runtime.metadata["intent"] = state.primary_intent
        state.runtime.metadata["nearby_itinerary_short_circuit"] = True
        state.runtime.metadata["nearby_anchor"] = {
            "id": str(anchor_node.id),
            "name": anchor_name,
            "labels": anchor_node.metadata.get("labels", []),
        }
        state.runtime.metadata["selected_lodging"] = hotel
        state.runtime.metadata["nearby_non_lodging"] = nearby
        state.runtime.metadata["detected_location"] = state.location
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(itinerary_nodes)
        state.runtime.metadata["route_seed_nodes"] = []
        state.runtime.metadata["graph"] = p._build_graph_payload(
            itinerary_nodes, [], intent=state.primary_intent
        )

        return {"answer": "\n".join(lines), "metadata": state.runtime.metadata}

    def _build_accommodation_based_itinerary(
        self,
        state: PipelineRunState,
        anchor_node: Any,
        anchor_name: str,
    ) -> Dict[str, Any] | None:
        """Build itinerary when the anchor IS the accommodation."""
        p = self.pipeline
        nearby = self._fetch_nearby_non_lodging_nodes(anchor_node, limit=8)
        attractions = [
            item
            for item in nearby
            if "TouristAttraction" in set(item.get("labels") or [])
        ]
        restaurants = [
            item
            for item in nearby
            if "Restaurant" in set(item.get("labels") or [])
        ]

        if not attractions and not restaurants:
            return None

        first = attractions[0] if attractions else restaurants[0]
        second = (
            attractions[1]
            if len(attractions) > 1
            else (restaurants[0] if attractions and restaurants else None)
        )

        lines = [
            f"Lịch trình nửa ngày bắt đầu từ {anchor_name}:",
            "",
            f"- 08:00: Xuất phát từ {anchor_name}.",
            f"- 08:15 - 10:00: Tham quan {first['name']} "
            f"({first.get('address') or 'gần nơi lưu trú'}).",
        ]
        if second:
            lines.append(
                f"- 10:15 - 11:30: Ghé {second['name']} "
                f"({second.get('address') or 'gần đó'})."
            )
        elif len(attractions) == 1:
            lines.append(f"- 10:15 - 11:30: Dành thêm thời gian tại {first['name']}.")
        if restaurants and not second:
            lines.append(f"- 11:30: Ăn trưa tại {restaurants[0]['name']}.")
        lines.append(f"- 12:00: Quay lại {anchor_name}.")

        intent = state.query_plan.intent if state.query_plan else state.primary_intent
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["nearby_itinerary_short_circuit"] = True
        state.runtime.metadata["nearby_anchor"] = {
            "id": str(anchor_node.id),
            "name": anchor_name,
            "labels": anchor_node.metadata.get("labels", []),
        }
        state.runtime.metadata["nearby_non_lodging"] = nearby
        state.runtime.metadata["detected_location"] = state.location

        from types import SimpleNamespace

        itinerary_nodes = [anchor_node]
        for item in nearby or []:
            item_node = SimpleNamespace(
                id=item.get("id"),
                content=item.get("name", ""),
                metadata={
                    "name": item.get("name"),
                    "labels": item.get("labels", []),
                    "lat": item.get("lat"),
                    "lng": item.get("lng"),
                    "address": item.get("address"),
                },
            )
            itinerary_nodes.append(item_node)
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(itinerary_nodes)
        state.runtime.metadata["route_seed_nodes"] = []
        state.runtime.metadata["graph"] = p._build_graph_payload(
            itinerary_nodes, [], intent=intent
        )
        return {"answer": "\n".join(lines), "metadata": state.runtime.metadata}

    def _answer_nearby_accommodation_from_context_if_possible(
        self, state: PipelineRunState
    ) -> str | None:
        if state.primary_intent != IntentType.ACCOMMODATION:
            return None
        if not self._is_nearby_accommodation_query(state.user_query):
            return None

        anchor_names = []
        for entity in state.entities or []:
            if not isinstance(entity, dict):
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            if e_type in {"location", "accommodation", "hotel", "lodging"}:
                continue
            e_name = str(entity.get("name") or "").strip()
            if e_name:
                anchor_names.append(normalize_text(e_name, strip_punct=True))

        for node in state.grounded_nodes or state.all_seeds or []:
            labels = set(
                str(x) for x in ((getattr(node, "metadata", {}) or {}).get("labels") or [])
            )
            if "Accommodation" in labels:
                continue
            name = str(
                (getattr(node, "metadata", {}) or {}).get("name")
                or getattr(node, "content", "")
                or ""
            )
            if name:
                anchor_names.append(normalize_text(name, strip_punct=True))

        relation_lines = [
            str(item or "").strip()
            for item in (state.raw_context or [])
            if "[NEAR]" in str(item or "")
        ]
        if not relation_lines:
            relation_lines = [
                line.strip()
                for line in str(state.clean_context or "").splitlines()
                if "[NEAR]" in line
            ]

        hotels: List[str] = []
        anchor_display = ""
        for line in relation_lines:
            match = re.search(r"^-\s*(.+?)\s+\[NEAR\]\s*->\s*(.+?)\s*$", line)
            if not match:
                continue
            left = match.group(1).strip()
            right = match.group(2).strip()
            left_norm = normalize_text(left, strip_punct=True)
            right_norm = normalize_text(right, strip_punct=True)
            left_is_lodging = (
                left_norm.startswith(("nha nghi ", "khach san ", "homestay ", "resort "))
                or "hotel" in left_norm
            )
            right_is_lodging = (
                right_norm.startswith(("nha nghi ", "khach san ", "homestay ", "resort "))
                or "hotel" in right_norm
            )

            if left_is_lodging and (
                not anchor_names
                or any(a and (a in right_norm or right_norm in a) for a in anchor_names)
            ):
                hotels.append(left)
                anchor_display = anchor_display or right
            elif right_is_lodging and (
                not anchor_names
                or any(a and (a in left_norm or left_norm in a) for a in anchor_names)
            ):
                hotels.append(right)
                anchor_display = anchor_display or left

        hotels = list(dict.fromkeys(hotels))
        if not hotels:
            return None
        anchor_display = anchor_display or "địa điểm này"
        return f"Có. Gần {anchor_display}, dữ liệu hệ thống ghi nhận chỗ nghỉ: {', '.join(hotels)}."

    def _answer_nearby_cultural_from_context_if_possible(
        self, state: PipelineRunState
    ) -> str | None:
        if not self._is_nearby_cultural_category_query(state.user_query):
            return None

        context_lines: List[str] = []
        for item in state.raw_context or []:
            context_lines.extend(
                line.strip() for line in str(item or "").splitlines() if line.strip()
            )
        if state.clean_context:
            context_lines.extend(
                line.strip()
                for line in str(state.clean_context or "").splitlines()
                if line.strip()
            )
        context_lines = list(dict.fromkeys(context_lines))
        if not context_lines:
            return None

        subject = ""
        near_places: List[str] = []
        for line in context_lines:
            main_match = re.search(
                r"\*\*(?:THỰC THỂ CHÍNH|THUC THE CHINH|TH.+?C TH.+? CH.+?NH):\*\*\s*(.+?)\s*\(",
                line,
            )
            if main_match:
                subject = main_match.group(1).strip()
                continue

            near_match = re.search(r"^-\s*(.+?)\s+\[NEAR\]\s*->\s*(.+?)\s*$", line)
            if not near_match:
                continue
            left = near_match.group(1).strip()
            right = near_match.group(2).strip()
            if not subject:
                subject = left
            subject_norm = normalize_text(subject, strip_punct=True)
            left_norm = normalize_text(left, strip_punct=True)
            right_norm = normalize_text(right, strip_punct=True)
            if subject_norm == left_norm:
                near_places.append(right)
            elif subject_norm == right_norm:
                near_places.append(left)
            else:
                near_places.append(right)

        if not subject:
            for entity in state.entities or []:
                if isinstance(entity, dict) and str(entity.get("name") or "").strip():
                    subject = str(entity.get("name")).strip()
                    break

        near_places = list(dict.fromkeys([p for p in near_places if p]))
        if not subject or not near_places:
            return None

        def classify(place: str) -> str:
            n = normalize_text(place, strip_punct=True)
            if any(token in n for token in keywords.CLASSIFY_HERITAGE_KEYWORDS):
                return "văn hóa - lịch sử"
            if any(token in n for token in keywords.CLASSIFY_CRAFT_KEYWORDS):
                return "văn hóa bản địa - làng nghề"
            if any(token in n for token in keywords.CLASSIFY_PUBLIC_SPACE_KEYWORDS):
                return "không gian công cộng - sinh hoạt văn hóa"
            return "tham quan văn hóa tổng hợp"

        groups: Dict[str, List[str]] = {}
        for place in near_places:
            groups.setdefault(classify(place), []).append(place)

        lines = [
            f"Có. Gần {subject}, dữ liệu hệ thống ghi nhận "
            f"các địa điểm văn hóa/vùng lân cận sau:"
        ]
        for group, places in groups.items():
            lines.append(f"- {group}: {', '.join(places)}")
        lines.append(
            "Các địa điểm này phù hợp để khách lưu trú kết hợp tham quan ngắn "
            "quanh khu vực; dữ liệu hiện chưa có khoảng cách chính xác theo km "
            "nên cần kiểm tra thêm khi lập lịch trình chi tiết."
        )
        return "\n".join(lines)

    def _fetch_nearby_accommodations_ranked(
        self,
        anchor_node: Any,
        *,
        limit: int = 5,
        max_distance_m: int = 15000,
    ) -> List[Dict[str, Any]]:
        p = self.pipeline
        anchor_lat = anchor_node.metadata.get("lat")
        anchor_lng = anchor_node.metadata.get("lng")
        if anchor_lat is None or anchor_lng is None:
            return []

        cypher = """
        MATCH (a:Accommodation)
        WITH a,
             CASE
               WHEN a.location IS NOT NULL THEN point.distance(
                    point({latitude: toFloat($anchor_lat), longitude: toFloat($anchor_lng)}),
                    a.location
               )
               WHEN a.lat IS NOT NULL AND a.lng IS NOT NULL THEN point.distance(
                    point({latitude: toFloat($anchor_lat), longitude: toFloat($anchor_lng)}),
                    point({latitude: toFloat(a.lat), longitude: toFloat(a.lng)})
               )
               ELSE NULL
             END AS distance_m
        WHERE distance_m IS NOT NULL AND distance_m <= toFloat($max_distance_m)
        RETURN a.id AS id,
               a.name AS name,
               a.address AS address,
               round(distance_m / 10.0) / 100.0 AS distance_km,
               CASE
                 WHEN a.location IS NOT NULL AND toLower(toString(a.location)) STARTS WITH 'point'
                 THEN a.location.latitude
                 ELSE a.lat
               END AS lat,
               CASE
                 WHEN a.location IS NOT NULL AND toLower(toString(a.location)) STARTS WITH 'point'
                 THEN a.location.longitude
                 ELSE a.lng
               END AS lng
        ORDER BY distance_m ASC
        LIMIT $limit
        """

        try:
            with p.driver.session() as session:
                rows = session.run(
                    cypher,
                    anchor_lat=float(anchor_lat),
                    anchor_lng=float(anchor_lng),
                    max_distance_m=float(max_distance_m),
                    limit=int(limit),
                ).data()
            return [
                {
                    "id": str(r.get("id") or ""),
                    "name": str(r.get("name") or "").strip(),
                    "address": str(r.get("address") or "").strip(),
                    "distance_km": float(r.get("distance_km"))
                    if r.get("distance_km") is not None
                    else None,
                    "lat": float(r.get("lat")) if r.get("lat") is not None else None,
                    "lng": float(r.get("lng")) if r.get("lng") is not None else None,
                }
                for r in rows
                if str(r.get("name") or "").strip()
            ]
        except (ValueError, TypeError) as exc:
            p.logger.warning("nearby_accommodation_ranked_query_failed: %s", str(exc))
            return []
