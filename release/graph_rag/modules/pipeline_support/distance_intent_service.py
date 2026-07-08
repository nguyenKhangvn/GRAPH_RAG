from __future__ import annotations

import re
from typing import Any, Dict, List, TypedDict, Literal
from dataclasses import dataclass

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

from graph_rag.core.intents import IntentType
from graph_rag.utils.text import normalize_text
from graph_rag.config.distance_patterns import (
    DISTANCE_TAIL_PATTERNS,
    EXPLICIT_DISTANCE_PATTERNS,
    DESTINATION_STOP_PATTERN,
    INTENT_PHRASES_BLACKLIST
)

class PipelineEntity(TypedDict, total=False):
    name: str
    surface_name: str
    normalized_name: str
    type: str
    role: str
    source: str
    confidence: float
    trusted: bool


LocationRole = Literal["origin", "destination"]


@dataclass(frozen=True)
class LocationCandidate:
    surface_text: str
    role: LocationRole
    source: str
    confidence: float | None = None


@dataclass(frozen=True)
class DistanceQuery:
    origin: LocationCandidate | None
    destination: LocationCandidate | None

    @property
    def is_complete(self) -> bool:
        return self.origin is not None and self.destination is not None


class DistanceEntityAdapter:
    def from_pipeline_entities(self, entities: list[dict[str, Any]]) -> DistanceQuery:
        origin = None
        destination = None
        for raw in (entities or []):
            candidate = self._to_location_candidate(raw)
            if candidate is None:
                continue
            if candidate.role == "origin" and origin is None:
                origin = candidate
            elif candidate.role == "destination" and destination is None:
                destination = candidate
        return DistanceQuery(origin=origin, destination=destination)

    def _to_location_candidate(self, raw: dict[str, Any]) -> LocationCandidate | None:
        name = str(raw.get("surface_name") or raw.get("name") or "").strip()
        role = str(raw.get("role") or "").lower()
        source = str(raw.get("source") or "legacy_pipeline")
        if not name or role not in {"origin", "destination"}:
            return None

        confidence_raw = raw.get("confidence")
        try:
            confidence = float(confidence_raw) if confidence_raw is not None else None
        except (ValueError, TypeError):
            confidence = None

        return LocationCandidate(
            surface_text=name,
            role=role,
            source=source,
            confidence=confidence
        )


class DistanceQueryParser:
    @staticmethod
    def parse(user_query: str) -> tuple[str, str]:
        """Parse user query to extract origin and destination location candidates (surface text)."""
        text = str(user_query or "").strip()
        if not text:
            return "", ""

        # Normalize space characters
        text = re.sub(r"\s+", " ", text)
        
        # Remove common tail/suffix question patterns (case-insensitive)
        clean_query = text.strip(" ?.!;:")
        for pat in DISTANCE_TAIL_PATTERNS:
            clean_query = re.sub(pat, "", clean_query, flags=re.IGNORECASE).strip()

        # Match explicit distance query patterns from configuration
        for pattern in EXPLICIT_DISTANCE_PATTERNS:
            m = re.search(pattern, clean_query, flags=re.IGNORECASE)
            if m:
                try:
                    src = m.group("origin").strip()
                    dst = m.group("destination").strip().strip(" ?.!;:")
                except IndexError:
                    src = m.group(1).strip()
                    dst = m.group(2).strip().strip(" ?.!;:")
                
                # Remove leading intent fragments
                src = re.sub(r"^(khoang\s+cach|khoảng\s+cách)\s+", "", src, flags=re.IGNORECASE).strip()
                
                # Verify and reject intent-only words
                if src.lower() in INTENT_PHRASES_BLACKLIST or len(src) < 3:
                    src = ""
                if dst.lower() in INTENT_PHRASES_BLACKLIST or len(dst) < 3:
                    dst = ""
                    
                return src, dst
                
        return "", ""
            
        return "", ""


class DistanceIntentService:
    def __init__(
        self,
        *,
        logger,
        retriever,
        directions_service,
        haversine_fn,
    ):
        self.logger = logger
        self.retriever = retriever
        self.directions_service = directions_service
        self._haversine_km = haversine_fn

    def preflight_followup_direction(
        self,
        *,
        user_query: str,
        metadata: Dict[str, Any],
        entities: List[Dict[str, Any]],
        detected_location: str,
        conversation_state: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """Hard-stop unresolved deictic direction queries before retrieval."""
        q_norm = normalize_text(user_query or "", strip_punct=True)
        has_direction_signal = any(
            sig in q_norm
            for sig in [
                "di nhu the nao",
                "duong di",
                "chi duong",
                "di den",
                "di toi",
                "den cho do",
                "den do",
                "toi do",
            ]
        )
        has_user_origin = any(
            sig in q_norm
            for sig in ["vi tri cua toi", "cho toi", "tu day", "o day", "toi dang o"]
        )
        has_deictic_destination = any(
            sig in q_norm
            for sig in ["cho do", "dia diem do", "noi do", "den do", "toi do"]
        )

        if not (has_direction_signal and (has_user_origin or has_deictic_destination)):
            return None

        last_active = (conversation_state or {}).get("last_active_entity") or {}
        last_anchor = (conversation_state or {}).get("last_grounded_anchor") or {}
        destination = last_active if last_active.get("name") else last_anchor
        dest_name = str(destination.get("name") or destination.get("content") or "").strip()

        if has_deictic_destination and not dest_name:
            return {
                "answer": "Bạn muốn đi đến địa điểm nào? Mình chưa xác định được \"chỗ đó\" trong ngữ cảnh hiện tại.",
                "metadata": {
                    **(metadata or {}),
                    "clarification_needed": True,
                    "clarification_reason": "missing_followup_destination",
                    "detected_location": detected_location,
                },
            }

        origin = str((metadata or {}).get("user_gps") or (metadata or {}).get("user_provided_location") or detected_location or "").strip()
        if has_user_origin and not origin:
            return {
                "answer": "Bạn có thể cho mình vị trí hiện tại hoặc điểm xuất phát cụ thể không?",
                "metadata": {
                    **(metadata or {}),
                    "clarification_needed": True,
                    "clarification_reason": "missing_followup_origin",
                    "detected_location": detected_location,
                },
            }

        if dest_name and not entities:
            metadata["distance_followup_destination"] = dest_name
            metadata["distance_followup_destination_id"] = destination.get("id") or ""
            entities.append({"name": dest_name, "type": "Location", "source": "last_active_entity"})
            self.logger.info("distance_preflight resolved deictic destination -> %s", dest_name)

        return None

    def _repair_distance_entities(self, user_query: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        current = list(entities or [])
        if len(current) < 2:
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
                if len(current) == 1 and isinstance(current[0], dict):
                    hinted = str(current[0].get("type") or "").strip()
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
            if len(repaired) >= 2 or (len(repaired) == 1 and len(current) == 1):
                self.logger.info("distance_entity_local_repair applied (parser-fallback): %s", repaired)
                return repaired
            return current

        def trim_tail(text: str) -> str:
            out = str(text or "").strip()
            for pat in [
                r"\s+la\s+bao\s+nhieu\s+km\s*$",
                r"\s+la\s+bao\s+nhieu\s*$",
                r"\s+bao\s+nhieu\s+km\s*$",
                r"\s+bao\s+nhieu\s*$",
                r"\s+là\s+bao\s+nhiêu\s+km\s*$",
                r"\s+là\s+bao\s+nhiêu\s*$",
                r"\s+bao\s+nhiêu\s+km\s*$",
                r"\s+bao\s+nhiêu\s*$",
                r"\s+bao\s+xa\s*$",
            ]:
                out = re.sub(pat, "", out, flags=re.IGNORECASE).strip()
            return out

        first_name_raw = str((current[0] or {}).get("name") or "")
        second_name_raw = str((current[1] or {}).get("name") or "")
        first_norm = normalize_text(first_name_raw)
        second_norm = normalize_text(second_name_raw)

        if second_norm and second_norm in first_norm and any(c in first_norm for c in [" den ", " toi "]):
            m = re.search(r"^(.+?)\s+(?:den|toi)\s+(.+)$", first_norm)
            if m:
                src = trim_tail(m.group(1))
                dst = trim_tail(second_name_raw) or trim_tail(m.group(2))
                src = re.sub(r"^(khoang\s+cach|khoảng\s+cách)\s+", "", src, flags=re.IGNORECASE).strip()
                if len(src) >= 3 and len(dst) >= 3:
                    repaired = [
                        {"name": src, "type": "Location", "role": "origin", "trusted": True},
                        {"name": dst, "type": (current[1] or {}).get("type") or "Location", "role": "destination", "trusted": True},
                    ]
                    self.logger.info("distance_entity_local_repair applied (entity-first): %s -> %s", current[:2], repaired)
                    return repaired

        # Fallback to DistanceQueryParser
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
            repaired.append({
                "name": dst,
                "type": (current[1] or {}).get("type") or "Location",
                "role": "destination",
                "source": "distance_parser",
                "confidence": 1.0,
                "trusted": True
            })
        if len(repaired) >= 2:
            self.logger.info("distance_entity_local_repair applied (parser-fallback): %s", repaired)
            return repaired

        return current

    def _entity_type_matches(self, labels: List[str], entity_type: str) -> bool:
        if not entity_type:
            return True
        normalized_labels = {str(x) for x in (labels or [])}
        mapping = {
            "Accommodation": {"Accommodation"},
            "Restaurant": {"Restaurant"},
            "Dish": {"Dish"},
            "TouristAttraction": {"TouristAttraction"},
            "Event": {"Event"},
            # In distance queries, "Location" from LLM often points to a concrete POI.
            "Location": {
                "Location",
                "TouristAttraction",
                "Restaurant",
                "Accommodation",
                "Event",
            },
            "Tour": {"Tour"},
        }
        expected = mapping.get(entity_type, {entity_type})
        return bool(normalized_labels.intersection(expected))

    def _score_candidate(self, entity_name: str, node: Any) -> int:
        norm_entity = normalize_text(entity_name)
        name = node.metadata.get("name") or node.content or ""
        norm_name = normalize_text(name)

        score = 0
        if norm_name == norm_entity:
            score += 100
        if norm_entity and norm_entity in norm_name:
            score += 50
        if norm_name and norm_name in norm_entity:
            score += 20
        score += int(node.score * 10) if getattr(node, "score", None) else 0
        if node.metadata.get("lat") is not None and node.metadata.get("lng") is not None:
            score += 3
        return score

    def _select_best_grounded_for_entity(self, entity: Dict[str, Any], grounded_nodes: List[Any]) -> Any:
        if not entity:
            return None
        entity_name = entity.get("name") or ""
        entity_type = entity.get("type") or ""

        candidates = []
        candidate_debug = []
        fallback_pool = []
        for node in grounded_nodes or []:
            name = node.metadata.get("name") or node.content or ""
            labels = node.metadata.get("labels") or []
            has_coords = node.metadata.get("lat") is not None and node.metadata.get("lng") is not None
            type_match = self._entity_type_matches(labels, entity_type)
            base_score = self._score_candidate(entity_name, node)
            if not type_match:
                candidate_debug.append(
                    {
                        "name": name,
                        "labels": labels,
                        "score": 0,
                        "name_score": base_score,
                        "type_match": False,
                        "has_coords": has_coords,
                    }
                )
                # Distance fallback: keep strong name matches that have coordinates,
                # even when LLM type is broader/narrower than graph labels.
                if has_coords and base_score >= 100:
                    fallback_pool.append((base_score, node))
                continue

            score = base_score
            candidates.append((score, node))
            candidate_debug.append(
                {
                    "name": name,
                    "labels": labels,
                    "score": score,
                    "type_match": True,
                    "has_coords": has_coords,
                }
            )

        if not candidates:
            if fallback_pool:
                fallback_pool.sort(key=lambda x: x[0], reverse=True)
                selected = fallback_pool[0][1]
                selected_name = selected.metadata.get("name") or selected.content or ""
                selected_labels = selected.metadata.get("labels") or []
                self.logger.info(
                    "distance_entity_resolution entity='%s' type='%s' -> candidates=%s -> selected={'name': '%s', 'labels': %s} (fallback=name+coords)",
                    entity_name,
                    entity_type,
                    candidate_debug,
                    selected_name,
                    selected_labels,
                )
                return selected
            self.logger.info(
                "distance_entity_resolution entity='%s' type='%s' -> candidates=%s -> selected=None",
                entity_name,
                entity_type,
                candidate_debug,
            )
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[0][1]
        selected_name = selected.metadata.get("name") or selected.content or ""
        selected_labels = selected.metadata.get("labels") or []
        self.logger.info(
            "distance_entity_resolution entity='%s' type='%s' -> candidates=%s -> selected={'name': '%s', 'labels': %s}",
            entity_name,
            entity_type,
            candidate_debug,
            selected_name,
            selected_labels,
        )
        return selected

    def _extract_travel_mode(self, query: str) -> str:
        q = normalize_text(query)
        walking_hints = {"di bo", "walking", "walk", "đi bộ", "đi bo"}
        if any(h in q for h in walking_hints):
            return "walking"
        return "driving"

    def _get_directions(self, source: Dict[str, Any], target: Dict[str, Any], mode: str) -> Dict[str, Any]:
        try:
            data = self.directions_service.get_directions(source, target, mode)
            data["map_url"] = self.directions_service.build_external_map_url(source, target)
            return data
        except (ValueError, RuntimeError, OSError) as exc:
            self.logger.warning("directions_api_fallback_straight_line: %s", str(exc))
            return {
                "provider": "none",
                "travel_mode": mode,
                "road_distance_km": None,
                "duration_min": None,
                "route_polyline": [],
                "map_url": self.directions_service.build_external_map_url(source, target),
            }

    def _lookup_travel_info_for_distance(self, source_name: str, dest_name: str) -> str:
        """Lookup TravelInfo for distance/transport information.

        Searches for TravelInfo entries that mention both source and destination.
        Returns description if found, else empty string.
        """
        try:
            with self.retriever.driver.session() as session:
                # Search for TravelInfo entries mentioning both locations
                source_norm = normalize_text(source_name, strip_punct=True)
                dest_norm = normalize_text(dest_name, strip_punct=True)

                # Query TravelInfo with topic 'transport' or containing location names
                result = session.run(
                    """
                    MATCH (t:TravelInfo)
                    WHERE t.topic IN ['transport', 'travel_info']
                    AND (
                        toLower(t.name) CONTAINS toLower($source) OR
                        toLower(t.name) CONTAINS toLower($dest) OR
                        toLower(t.description) CONTAINS toLower($source) OR
                        toLower(t.description) CONTAINS toLower($dest)
                    )
                    RETURN t.name AS name, t.description AS description, t.topic AS topic
                    LIMIT 5
                    """,
                    source=source_name,
                    dest=dest_name,
                ).data()

                if not result:
                    # Fallback: search with normalized names
                    result = session.run(
                        """
                        MATCH (t:TravelInfo)
                        WHERE t.topic IN ['transport', 'travel_info']
                        RETURN t.name AS name, t.description AS description, t.topic AS topic
                        LIMIT 20
                        """
                    ).data()

                    # Filter in Python for better matching
                    filtered = []
                    for row in result:
                        name_norm = normalize_text(row.get("name", ""), strip_punct=True)
                        desc_norm = normalize_text(row.get("description", ""), strip_punct=True)
                        combined = f"{name_norm} {desc_norm}"

                        if (source_norm in combined and dest_norm in combined):
                            filtered.append(row)

                    result = filtered[:3]

                if result:
                    # Return the most relevant entry
                    best = result[0]
                    return best.get("description", "")

        except (Neo4jClientError, ServiceUnavailable) as e:
            self.logger.warning("TravelInfo lookup failed: %s", e)

        return ""

    def run_distance_intent(
        self,
        user_query: str,
        metadata: Dict[str, Any],
        grounded_nodes: List[Any],
        entities: List[Dict[str, Any]],
        detected_location: str,
    ) -> Dict[str, Any]:
        entities = self._repair_distance_entities(user_query, entities or [])
        metadata["intent"] = IntentType.DISTANCE

        adapter = DistanceEntityAdapter()
        distance_query = adapter.from_pipeline_entities(entities)

        gps_source = None
        if not distance_query.is_complete:
            loc = (detected_location or "").strip()
            if loc and distance_query.destination is not None and distance_query.origin is None:
                if "," in loc:
                    try:
                        parts = loc.split(",")
                        lat, lng = float(parts[0].strip()), float(parts[1].strip())
                        gps_source = {"lat": lat, "lng": lng, "name": f"Vị trí hiện tại ({lat:.4f}, {lng:.4f})"}
                        self.logger.info("distance_intent: using GPS coords as source: %s", gps_source)
                    except (ValueError, IndexError):
                        pass
                if not gps_source:
                    origin_candidate = LocationCandidate(
                        surface_text=loc,
                        role="origin",
                        source="detected_location"
                    )
                    distance_query = DistanceQuery(origin=origin_candidate, destination=distance_query.destination)
                    self.logger.info("distance_intent: using detected_location='%s' as origin", loc)
            if not gps_source and not distance_query.is_complete:
                return {
                    "answer": "Mình chưa xác định đủ 2 địa điểm để tính khoảng cách. Bạn hãy nêu rõ điểm đi và điểm đến.",
                    "metadata": metadata,
                }

        resolved_nodes = []
        used_node_ids = set()
        
        candidates_to_resolve = []
        if gps_source is None and distance_query.origin is not None:
            candidates_to_resolve.append(distance_query.origin)
        if distance_query.destination is not None:
            candidates_to_resolve.append(distance_query.destination)

        for candidate in candidates_to_resolve:
            entity_dict = {
                "name": candidate.surface_text,
                "type": "Location",
                "role": candidate.role,
                "source": candidate.source
            }
            available_grounded = [
                n for n in (grounded_nodes or [])
                if str(getattr(n, "id", "")) not in used_node_ids
            ]
            best = self._select_best_grounded_for_entity(entity_dict, available_grounded)
            if not best:
                more = self.retriever.ground_entities([entity_dict])
                available_more = [
                    n for n in (more or [])
                    if str(getattr(n, "id", "")) not in used_node_ids
                ]
                best = self._select_best_grounded_for_entity(entity_dict, available_more)
            if best:
                resolved_nodes.append(best)
                used_node_ids.add(str(getattr(best, "id", "")))

        if gps_source:
            if not resolved_nodes:
                return {
                    "answer": "Mình chưa xác định được địa điểm đến trong dữ liệu.",
                    "metadata": metadata,
                }
            dst_node = resolved_nodes[0]
            dst_name = dst_node.metadata.get("name") or dst_node.content
            dst_lat = dst_node.metadata.get("lat")
            dst_lng = dst_node.metadata.get("lng")
            if dst_lat is None or dst_lng is None:
                return {
                    "answer": f"Mình đã xác định '{dst_name}' nhưng thiếu tọa độ để tính khoảng cách.",
                    "metadata": metadata,
                }
            src_name = gps_source["name"]
            src_lat, src_lng = gps_source["lat"], gps_source["lng"]
            straight_km = round(self._haversine_km(src_lat, src_lng, float(dst_lat), float(dst_lng)), 2)
            travel_mode = self._extract_travel_mode(user_query)
            directions_data = self._get_directions(
                {"lat": src_lat, "lng": src_lng},
                {"lat": float(dst_lat), "lng": float(dst_lng)},
                travel_mode,
            )
            road_km = directions_data.get("road_distance_km")
            duration_min = directions_data.get("duration_min")
            if road_km is not None and duration_min is not None:
                answer = (
                    f"Từ {src_name} đến {dst_name}: khoảng cách đường chim bay khoảng {straight_km} km. "
                    f"Quãng đường {travel_mode} ước tính {road_km} km, thời gian khoảng {duration_min} phút."
                )
            else:
                answer = (
                    f"Từ {src_name} đến {dst_name}: khoảng cách đường chim bay khoảng {straight_km} km. "
                    "Hiện chưa lấy được lộ trình đường đi thời gian thực."
                )
            metadata["seed_nodes"] = [
                {"id": "", "name": src_name, "labels": [], "lat": src_lat, "lng": src_lng},
                {
                    "id": dst_node.id,
                    "name": dst_name,
                    "labels": dst_node.metadata.get("labels", []),
                    "lat": dst_lat,
                    "lng": dst_lng,
                },
            ]
            metadata["distance"] = {
                "source_name": src_name,
                "target_name": dst_name,
                "straight_distance_km": straight_km,
                "road_distance_km": road_km,
                "duration_min": duration_min,
                "route_polyline": directions_data.get("route_polyline", []),
            }
            return {"answer": answer, "metadata": metadata}

        if len(resolved_nodes) < 2:
            source_name = distance_query.origin.surface_text if distance_query.origin else ""
            dest_name = distance_query.destination.surface_text if distance_query.destination else ""

            if source_name and dest_name:
                travel_info = self._lookup_travel_info_for_distance(source_name, dest_name)
                if travel_info:
                    self.logger.info("distance_travel_info_fallback: found TravelInfo for '%s' -> '%s'", source_name, dest_name)
                    return {
                        "answer": travel_info,
                        "metadata": metadata,
                    }

            return {
                "answer": "Dữ liệu hiện chưa có thông tin khoảng cách hoặc tuyến xe cụ thể giữa hai địa điểm này. Bạn vui lòng tham khảo bản đồ trực tiếp hoặc các gợi ý di chuyển công cộng khác.",
                "metadata": metadata,
            }

        src = resolved_nodes[0]
        dst = resolved_nodes[1]
        if str(getattr(src, "id", "")) == str(getattr(dst, "id", "")):
            self.logger.warning(
                "distance_same_node_selected entity_source='%s' entity_target='%s' node_id='%s'",
                distance_query.origin.surface_text if distance_query.origin else "",
                distance_query.destination.surface_text if distance_query.destination else "",
                str(getattr(src, "id", "")),
            )
            return {
                "answer": "Mình đang map trùng 2 địa điểm về cùng một nơi, nên chưa thể tính khoảng cách chính xác. Bạn giúp mình nêu rõ tên đầy đủ của điểm đi/điểm đến nhé.",
                "metadata": metadata,
            }
        src_name = src.metadata.get("name") or src.content
        dst_name = dst.metadata.get("name") or dst.content

        src_lat = src.metadata.get("lat")
        src_lng = src.metadata.get("lng")
        dst_lat = dst.metadata.get("lat")
        dst_lng = dst.metadata.get("lng")

        if None in (src_lat, src_lng, dst_lat, dst_lng):
            answer = (
                f"Mình đã xác định được '{src_name}' và '{dst_name}', "
                "nhưng thiếu tọa độ để tính khoảng cách chính xác."
            )
            metadata["seed_nodes"] = [
                {
                    "id": s.id,
                    "name": s.metadata.get("name") or s.content,
                    "labels": s.metadata.get("labels", []),
                    "attributes": s.metadata,
                    "lat": s.metadata.get("lat"),
                    "lng": s.metadata.get("lng"),
                }
                for s in resolved_nodes
            ]
            metadata["distance"] = {
                "source_name": src_name,
                "target_name": dst_name,
                "straight_distance_km": None,
                "road_distance_km": None,
                "duration_min": None,
                "route_polyline": [],
            }
            return {"answer": answer, "metadata": metadata}

        straight_km = round(self._haversine_km(float(src_lat), float(src_lng), float(dst_lat), float(dst_lng)), 2)
        travel_mode = self._extract_travel_mode(user_query)
        directions_data = self._get_directions(
            {"lat": float(src_lat), "lng": float(src_lng)},
            {"lat": float(dst_lat), "lng": float(dst_lng)},
            travel_mode,
        )
        road_km = directions_data.get("road_distance_km")
        duration_min = directions_data.get("duration_min")
        map_url = directions_data.get("map_url") or ""

        if road_km is not None and duration_min is not None:
            answer = (
                f"Từ {src_name} đến {dst_name}: khoảng cách đường chim bay khoảng {straight_km} km. "
                f"Quãng đường {travel_mode} ước tính {road_km} km, thời gian khoảng {duration_min} phút."
            )
        else:
            answer = (
                f"Từ {src_name} đến {dst_name}: khoảng cách đường chim bay khoảng {straight_km} km. "
                "Hiện chưa lấy được lộ trình đường đi thời gian thực, nên mình tạm trả về khoảng cách ước tính theo tọa độ."
            )
        if map_url:
            answer = f"{answer} Xem đường đi: {map_url}"

        metadata["detected_location"] = detected_location
        metadata["intent"] = IntentType.DISTANCE
        metadata["seed_nodes"] = [
            {
                "id": s.id,
                "name": s.metadata.get("name") or s.content,
                "labels": s.metadata.get("labels", []),
                "attributes": s.metadata,
                "lat": s.metadata.get("lat"),
                "lng": s.metadata.get("lng"),
            }
            for s in resolved_nodes
        ]
        metadata["distance"] = {
            "source_name": src_name,
            "target_name": dst_name,
            "straight_distance_km": straight_km,
            "road_distance_km": road_km,
            "duration_min": duration_min,
            "travel_mode": directions_data.get("travel_mode") or travel_mode,
            "provider": directions_data.get("provider") or "none",
            "route_polyline": directions_data.get("route_polyline", []),
            "map_url": map_url,
        }
        metadata["graph"] = {
            "nodes": [
                {
                    "id": str(s.id),
                    "name": s.metadata.get("name") or s.content,
                    "labels": s.metadata.get("labels", []),
                    "lat": s.metadata.get("lat"),
                    "lng": s.metadata.get("lng"),
                }
                for s in resolved_nodes
            ],
            "links": [
                {
                    "source": str(src.id),
                    "target": str(dst.id),
                    "relation": "DISTANCE_TO",
                }
            ],
        }

        return {"answer": answer, "metadata": metadata}
