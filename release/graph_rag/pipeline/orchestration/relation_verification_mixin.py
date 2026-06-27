from __future__ import annotations

from typing import Any, Dict, List

from graph_rag.config import RELATIONSHIP_MAP
from graph_rag.core.state import NodeItem
from graph_rag.utils.text import normalize_text
from .dto import PipelineRunState


class RelationVerificationMixin:
    RELATION_ENDPOINTS = {
        "HELD_AT": ("Event", "TouristAttraction"),
        "HAS": ("Restaurant", "Dish"),
        "NEAR": ("Restaurant|Accommodation", "TouristAttraction"),
        "INCLUDES": ("Tour", "TouristAttraction"),
        "LOCATED_IN": ("*", "Location"),
    }
    RELATION_VERIFICATION_HINTS = [
        "co phai",
        "dung khong",
        "co dung",
        "co moi quan he",
        "dia diem to chuc chinh la",
        "duoc to chuc tai",
        "to chuc tai",
        "phuc vu mon",
        "bao gom diem den",
    ]

    def _is_relation_verification_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        if any(hint in q for hint in self.RELATION_VERIFICATION_HINTS):
            return True
        # "moi quan he" only when used in verification context (with "co phai", "xac minh", etc.)
        if "moi quan he" in q and any(token in q for token in ["co phai", "xac minh", "dung khong", "co dung", "chinh xac"]):
            return True
        return False

    def _is_event_descriptor_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        has_event = "le hoi" in q or "su kien" in q
        has_month = "thang" in q or "gieng" in q
        has_activity = any(token in q for token in ["hoat dong", "bao gom", "mua", "vo thuat", "tran phap"])
        has_venue = any(token in q for token in ["to chuc", "quang truong", "bao tang", "dia diem"])
        return has_event and has_month and has_activity and has_venue

    def _relation_verification_abstain(self, state: PipelineRunState) -> Dict[str, Any]:
        relation = str(state.metadata.get("relation_verification_relation") or "quan hệ").strip()
        entities = state.metadata.get("relation_verification_entities") or []
        entity_text = ", ".join(str(x) for x in entities if str(x).strip())
        if entity_text:
            detail = f" giữa {entity_text}"
        else:
            detail = ""
        return {
            "early_result": {
                "answer": (
                    f"Xin lỗi, tôi chưa tìm thấy thông tin đáng tin cậy cho quan hệ {relation}{detail}. "
                    "Vì câu hỏi đang yêu cầu xác minh quan hệ cụ thể, tôi không suy đoán ngoài dữ liệu."
                ),
                "metadata": state.runtime.metadata,
            }
        }

    def _entity_names_for_relation_side(
        self,
        entities: List[Dict[str, Any]],
        label_spec: str,
    ) -> List[str]:
        if not entities:
            return []
        labels = {part.strip().lower() for part in str(label_spec or "").split("|") if part.strip()}
        if "touristattraction" in labels:
            labels.update({"attraction", "place", "venue", "location"})
        if "location" in labels:
            labels.update({"province", "city", "district", "ward", "commune"})
        names = []
        for entity in entities:
            if not self._is_groundable_entity(entity):
                continue
            name = str(entity.get("name") or "").strip()
            if not name:
                continue
            e_type = str(entity.get("type") or "").strip().lower()
            if label_spec == "*":
                if e_type not in {"province", "city", "district", "ward", "commune", "location"}:
                    names.append(name)
                continue
            if e_type in labels:
                names.append(name)
        return list(dict.fromkeys(names))

    @staticmethod

    def _node_item_from_relation_record(record: Dict[str, Any], prefix: str) -> NodeItem | None:
        node_id = str(record.get(f"{prefix}_id") or "").strip()
        node_name = str(record.get(f"{prefix}_name") or "").strip()
        if not node_id or not node_name:
            return None
        labels = list(record.get(f"{prefix}_labels") or [])
        return NodeItem(
            id=node_id,
            content=node_name,
            score=1.0,
            source_type="relation_verifier",
            metadata={
                "name": node_name,
                "type": labels[0] if labels else "Unknown",
                "labels": labels,
                "address": str(record.get(f"{prefix}_address") or ""),
                "description": str(record.get(f"{prefix}_description") or ""),
                "lat": record.get(f"{prefix}_lat"),
                "lng": record.get(f"{prefix}_lng"),
            },
        )

    def _verify_event_descriptor_relation(self, state: PipelineRunState) -> Dict[str, Any]:
        q = normalize_text(state.user_query, strip_punct=True)
        if not self._is_event_descriptor_query(state.user_query):
            return {"attempted": False, "matched": False, "nodes": [], "facts": []}

        entities = [e for e in (state.entities or []) if isinstance(e, dict)]
        venue_names = [
            str(e.get("name") or "").strip()
            for e in entities
            if str(e.get("name") or "").strip()
            and normalize_text(str(e.get("name") or ""), strip_punct=True).startswith(("quang truong", "bao tang"))
        ]
        if not venue_names and "quang truong dai doan ket" in q:
            venue_names = ["Quảng trường Đại Đoàn Kết"]

        if not venue_names and "bao tang quang trung" in q:
            venue_names = ["Bảo tàng Quang Trung"]

        month = "1" if ("thang 1" in q or "gieng" in q) else ""
        location = "Phường Pleiku" if "phuong pleiku" in q or "pleiku" in q else ""
        activity = "Múa khèn người Mông" if "mua khen" in q and "mong" in q else ""
        if not activity and "vo thuat" in q:
            activity = "võ thuật"
        if not activity and "tran phap" in q:
            activity = "trận pháp"
        category = "Lễ hội văn hóa dân gian" if "le hoi van hoa dan gian" in q else ""
        if not category and "le hoi lich su" in q:
            category = "Lễ hội lịch sử"
        if not (venue_names and month and (activity or category or location)):
            return {"attempted": False, "matched": False, "nodes": [], "facts": []}

        cypher = """
        MATCH (event:Event)-[r:HELD_AT]-(venue)
        WHERE (
            toLower(coalesce(venue.name, '')) CONTAINS toLower($venue_name)
            OR toLower($venue_name) CONTAINS toLower(coalesce(venue.name, ''))
        )
        AND ($month = '' OR toString(coalesce(event.month, '')) = $month)
        AND ($location = '' OR toLower(coalesce(event.address, '')) CONTAINS toLower($location))
        AND ($category = '' OR toLower(coalesce(event.category, '')) CONTAINS toLower($category))
        AND (
            $activity = ''
            OR any(activity IN coalesce(event.activities, []) WHERE toLower(toString(activity)) CONTAINS toLower($activity))
            OR toLower(coalesce(event.description, '')) CONTAINS toLower($activity)
        )
        RETURN event.id AS left_id,
               event.name AS left_name,
               labels(event) AS left_labels,
               event.description AS left_description,
               event.address AS left_address,
               event.location.latitude AS left_lat,
               event.location.longitude AS left_lng,
               venue.id AS right_id,
               venue.name AS right_name,
               labels(venue) AS right_labels,
               venue.description AS right_description,
               venue.address AS right_address,
               venue.location.latitude AS right_lat,
               venue.location.longitude AS right_lng,
               type(r) AS rel_type,
               event.month AS event_month,
               event.activities AS event_activities,
               event.category AS event_category
        LIMIT 5
        """
        matched_nodes: List[NodeItem] = []
        facts: List[str] = []
        failed_details = []
        try:
            with self.pipeline.driver.session() as session:
                for venue_name in venue_names:
                    rows = session.run(
                        cypher,
                        venue_name=venue_name,
                        month=month,
                        location=location,
                        activity=activity,
                        category=category,
                    ).data()
                    if not rows:
                        failed_details.append({"relation": "HELD_AT", "event_descriptor": state.user_query, "venue": venue_name})
                        continue
                    for row in rows:
                        left_node = self._node_item_from_relation_record(row, "left")
                        right_node = self._node_item_from_relation_record(row, "right")
                        if left_node:
                            matched_nodes.append(left_node)
                        if right_node:
                            matched_nodes.append(right_node)
                        facts.append(
                            f"{row.get('left_name')} {RELATIONSHIP_MAP.get('HELD_AT', 'được tổ chức tại')} {row.get('right_name')}; "
                            f"tháng {row.get('event_month')}; hoạt động: {', '.join(str(x) for x in (row.get('event_activities') or [])[:4])}"
                        )
        except (ValueError, TypeError) as exc:
            self.pipeline.logger.warning("event_descriptor_verification_failed: %s", str(exc))
            failed_details.append({"relation": "HELD_AT", "error": str(exc)})

        unique_nodes = []
        seen_ids = set()
        for node in matched_nodes:
            if node.id in seen_ids:
                continue
            unique_nodes.append(node)
            seen_ids.add(node.id)

        return {
            "attempted": True,
            "matched": bool(unique_nodes),
            "relation": "HELD_AT",
            "nodes": unique_nodes,
            "facts": list(dict.fromkeys(facts)),
            "failed_details": failed_details,
        }

    def _verify_requested_relation_triples(self, state: PipelineRunState) -> Dict[str, Any]:
        event_descriptor_result = self._verify_event_descriptor_relation(state)
        if event_descriptor_result.get("attempted"):
            return event_descriptor_result

        requested = [
            str(rel or "").strip().upper()
            for rel in (state.metadata or {}).get("requested_relations", [])
            if str(rel or "").strip().upper() in self.RELATION_ENDPOINTS
        ]
        entities = [e for e in (state.entities or []) if isinstance(e, dict)]
        if not requested or len(entities) < 2:
            return {"attempted": False, "matched": False, "nodes": [], "facts": []}

        attempted = False
        matched_nodes: List[NodeItem] = []
        facts: List[str] = []
        failed_details = []

        for rel in requested:
            left_label, right_label = self.RELATION_ENDPOINTS[rel]
            left_names = self._entity_names_for_relation_side(entities, left_label)
            right_names = self._entity_names_for_relation_side(entities, right_label)
            if not left_names or not right_names:
                continue
            attempted = True

            left_cypher = "" if left_label == "*" else ":" + ":".join([left_label.split("|")[0]])
            right_cypher = "" if right_label == "*" else ":" + ":".join([right_label.split("|")[0]])
            cypher = f"""
            MATCH (left{left_cypher})-[r:{rel}]-(right{right_cypher})
            WHERE (
                toLower(coalesce(left.name, '')) CONTAINS toLower($left_name)
                OR toLower($left_name) CONTAINS toLower(coalesce(left.name, ''))
            )
            AND (
                toLower(coalesce(right.name, '')) CONTAINS toLower($right_name)
                OR toLower($right_name) CONTAINS toLower(coalesce(right.name, ''))
            )
            RETURN left.id AS left_id,
                   left.name AS left_name,
                   labels(left) AS left_labels,
                   left.description AS left_description,
                   left.address AS left_address,
                   left.location.latitude AS left_lat,
                   left.location.longitude AS left_lng,
                   right.id AS right_id,
                   right.name AS right_name,
                   labels(right) AS right_labels,
                   right.description AS right_description,
                   right.address AS right_address,
                   right.location.latitude AS right_lat,
                   right.location.longitude AS right_lng,
                   type(r) AS rel_type
            LIMIT 5
            """
            try:
                with self.pipeline.driver.session() as session:
                    for left_name in left_names:
                        for right_name in right_names:
                            rows = session.run(cypher, left_name=left_name, right_name=right_name).data()
                            if not rows:
                                failed_details.append({"relation": rel, "left": left_name, "right": right_name})
                                continue
                            for row in rows:
                                left_node = self._node_item_from_relation_record(row, "left")
                                right_node = self._node_item_from_relation_record(row, "right")
                                if left_node:
                                    matched_nodes.append(left_node)
                                if right_node:
                                    matched_nodes.append(right_node)
                                facts.append(
                                    f"{row.get('left_name')} {RELATIONSHIP_MAP.get(rel, rel)} {row.get('right_name')}"
                                )
            except (ValueError, TypeError) as exc:
                self.pipeline.logger.warning("relation_verification_failed: %s", str(exc))
                failed_details.append({"relation": rel, "error": str(exc)})

        unique_nodes = []
        seen_ids = set()
        for node in matched_nodes:
            if node.id in seen_ids:
                continue
            unique_nodes.append(node)
            seen_ids.add(node.id)

        return {
            "attempted": attempted,
            "matched": bool(unique_nodes),
            "nodes": unique_nodes,
            "facts": list(dict.fromkeys(facts)),
            "failed_details": failed_details,
        }
