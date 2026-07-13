from __future__ import annotations

import re
from typing import Any, Dict, List

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

from graph_rag.core.intents import IntentType
from graph_rag.utils.text import normalize_text


from graph_rag.config.distance_patterns import (
    DISTANCE_TAIL_PATTERNS,
    EXPLICIT_DISTANCE_PATTERNS,
    INTENT_PHRASES_BLACKLIST,
)

# Sentinel value: origin slot matched a deictic self-reference ("ở đây", etc.)
_USER_LOCATION_SELF = "__USER_LOCATION_SELF__"

# Normalised (no-accent) aliases that mean "my current location".
# Checked against the normalised origin string extracted by DistanceQueryParser.
_USER_LOCATION_ALIASES: frozenset[str] = frozenset([
    # Vietnamese with accents
    "ở đây", "đây", "chỗ này", "nơi này", "vị trí của tôi",
    "vị trí hiện tại", "vị trí tôi", "chỗ tôi đang", "tôi đang ở đây",
    # Common normalised (no-accent) equivalents
    "o day", "day", "cho nay", "noi nay", "vi tri cua toi",
    "vi tri hien tai", "vi tri toi", "cho toi dang", "toi dang o day",
    # Short shorthands
    "here", "my location", "current location",
])


class DistanceQueryParser:
    @staticmethod
    def parse(user_query: str) -> tuple[str, str]:
        """Parse user query to extract origin and destination location candidates (surface text)."""
        text = str(user_query or "").strip()
        if not text:
            return "", ""

        # Normalize space characters
        text = re.sub(r"\s+", " ", text)

        # Clean leading typos/ellipses at the start of distance queries:
        # "ừ đây đến..." -> "từ đây đến..."
        # "u day den..." -> "tu day den..."
        # "đây đến..." -> "từ đây đến..."
        # "day den..." -> "tu day den..."
        # "o day den..." -> "tu o day den..."
        # "ở đây đến..." -> "từ ở đây đến..."
        text = re.sub(r"^[ừu]\s+", "từ ", text, flags=re.IGNORECASE)
        # If it starts with "đây đến" / "day den" / "ở đây đến" / "o day den" etc., prefix with "từ"
        if re.match(r"^(?:đây|day|ở đây|o day)\s+(?:đến|toi|tới|den)\b", text, flags=re.IGNORECASE):
            text = "từ " + text
        
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

                # Detect deictic self-references for the origin slot
                # ("ở đây", "đây", "vị trí của tôi", etc.) and replace with
                # the sentinel so callers can substitute GPS coords.
                if src.lower() in _USER_LOCATION_ALIASES:
                    src = _USER_LOCATION_SELF

                # Verify and reject intent-only words (only when not the sentinel)
                if src != _USER_LOCATION_SELF and (
                    src.lower() in INTENT_PHRASES_BLACKLIST or len(src) < 3
                ):
                    src = ""
                if dst.lower() in INTENT_PHRASES_BLACKLIST or len(dst) < 3:
                    dst = ""
                    
                return src, dst
                
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
            return current

        # Don't repair when the first entity is the USER_LOCATION_SELF sentinel —
        # it must survive intact so run_distance_intent() can swap it for GPS coords.
        if (current[0] or {}).get("name") in (_USER_LOCATION_SELF, "USER_LOCATION_SELF"):
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
                        {"name": src, "type": "Location"},
                        {"name": dst, "type": (current[1] or {}).get("type") or "Location"},
                    ]
                    self.logger.info("distance_entity_local_repair applied (entity-first): %s -> %s", current[:2], repaired)
                    return repaired

        query_norm = normalize_text(user_query or "")
        query_norm_no_punct = normalize_text(user_query or "", strip_punct=True)
        if any(
            phrase in query_norm_no_punct
            for phrase in [
                "vi tri cua toi",
                "cho do",
                "den cho do",
                "dia diem do",
                "noi do",
                "di nhu the nao",
            ]
        ):
            self.logger.info("distance_entity_local_repair skipped deictic/constraint query: %s", query_norm_no_punct)
            return current
        # Match "từ X đến Y", "từ X đi Y", "X đến Y", "X đi Y"
        # Stop destination at question keywords like "mất bao lâu", "phương tiện", etc.
        _DEST_STOP = r"(?:\s+(?:mat|bao lau|phuong tien|thuan tien|gia|chi phi|nhu the nao|the nao|khong|ha)\b|$)"
        m2 = re.search(r"(?:^|\s)(?:tu|từ)\s+(.+?)\s+(?:den|đến|toi|tới|di|đi)\s+(.+?)" + _DEST_STOP, query_norm, flags=re.IGNORECASE)
        if not m2:
            m2 = re.search(r"^(.+?)\s+(?:den|đến|toi|tới|di|đi)\s+(.+?)" + _DEST_STOP, query_norm, flags=re.IGNORECASE)
        if m2:
            src = trim_tail(m2.group(1))
            dst = trim_tail(m2.group(2))
            src = re.sub(r"^(khoang\s+cach|khoảng\s+cách)\s+", "", src, flags=re.IGNORECASE).strip()
            # Reject source that is just a travel-intent verb phrase, not a real location.
            _INTENT_PHRASES = {
                "duong di", "duong dan", "di toi", "di den", "dan den",
                "dan toi", "chi duong", "chi dan", "tim duong",
            }
            if src in _INTENT_PHRASES:
                return current
            if len(src) >= 3 and len(dst) >= 3:
                repaired = [
                    {"name": src, "type": "Location"},
                    {"name": dst, "type": (current[1] or {}).get("type") or "Location"},
                ]
                self.logger.info("distance_entity_local_repair applied (query-fallback): %s -> %s", current[:2], repaired)
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
            # "Place" is the generic type emitted by DistanceQueryParser for named destinations.
            # It should resolve against any concrete POI label in the graph.
            "Place": {
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

        # Minimum score threshold: reject candidates with very low name similarity.
        # score < 20 means no meaningful overlap between entity name and candidate name,
        # preventing false positives like "Quảng trường Nguyễn Tất Thành" → "Eo Gió" (score=11).
        _MIN_SCORE = 20

        if not candidates:
            if fallback_pool:
                fallback_pool.sort(key=lambda x: x[0], reverse=True)
                best_score, best_node = fallback_pool[0]
                if best_score < _MIN_SCORE:
                    self.logger.info(
                        "distance_entity_resolution entity='%s' type='%s' -> fallback candidate score=%d below threshold=%d -> selected=None",
                        entity_name, entity_type, best_score, _MIN_SCORE,
                    )
                    return None
                selected = best_node
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
        best_score_main = candidates[0][0]
        if best_score_main < _MIN_SCORE:
            self.logger.info(
                "distance_entity_resolution entity='%s' type='%s' -> best candidate score=%d below threshold=%d -> selected=None",
                entity_name, entity_type, best_score_main, _MIN_SCORE,
            )
            return None
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

        Returns description ONLY when both source AND destination are found in
        the same TravelInfo node's content — prevents false positives where a
        generic term like '\u1edf \u0111\u00e2y' matches unrelated transport entries.
        """
        # Reject degenerate inputs that will produce false positives
        _DEICTIC_REJECTS = {"o day", "day", "o do", "cho nay", "noi nay", "vi tri cua toi",
                            "vi tri hien tai", "\u1edf \u0111\u00e2y", "\u0111\u00e2y", "here", "current location"}
        source_norm = normalize_text(source_name, strip_punct=True)
        dest_norm = normalize_text(dest_name, strip_punct=True)

        # If source is a deictic or very short generic term, skip lookup entirely
        if source_norm in _DEICTIC_REJECTS or len(source_norm) < 4:
            self.logger.info("travel_info_lookup skipped: source is deictic/generic '%s'", source_name)
            return ""
        if dest_norm in _DEICTIC_REJECTS or len(dest_norm) < 4:
            self.logger.info("travel_info_lookup skipped: dest is deictic/generic '%s'", dest_name)
            return ""

        try:
            with self.retriever.driver.session() as session:
                # Require BOTH source AND destination to appear in the same node.
                # Using AND prevents returning unrelated transport info when only
                # one name (or a generic word) happens to match.
                result = session.run(
                    """
                    MATCH (t:TravelInfo)
                    WHERE t.topic IN ['transport', 'travel_info']
                    AND (
                        toLower(t.name) CONTAINS toLower($source)
                        OR toLower(t.description) CONTAINS toLower($source)
                    )
                    AND (
                        toLower(t.name) CONTAINS toLower($dest)
                        OR toLower(t.description) CONTAINS toLower($dest)
                    )
                    RETURN t.name AS name, t.description AS description, t.topic AS topic
                    LIMIT 3
                    """,
                    source=source_name,
                    dest=dest_name,
                ).data()

                # NOTE: The broad fallback scan (LIMIT 20 with OR filter) is intentionally
                # removed — it caused false positives with generic/deictic terms.

                if result:
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

        # ── Resolve USER_LOCATION_SELF sentinel ────────────────────────────────
        # DistanceQueryParser marks origin as _USER_LOCATION_SELF when the user
        # says "ở đây", "đây", "vị trí của tôi", etc.  Convert that sentinel to
        # a GPS-based source dict so we never pass it to the graph retriever.
        gps_source = None

        def _try_parse_gps(loc_str: str) -> dict | None:
            """Return {lat, lng, name} if loc_str is a 'lat,lng' pair, else None."""
            if loc_str and "," in loc_str:
                try:
                    parts = loc_str.split(",")
                    lat, lng = float(parts[0].strip()), float(parts[1].strip())
                    # Sanity-check plausible lat/lng ranges
                    if -90 <= lat <= 90 and -180 <= lng <= 180:
                        return {
                            "lat": lat,
                            "lng": lng,
                            "name": f"Vị trí hiện tại ({lat:.5f}, {lng:.5f})",
                        }
                except (ValueError, IndexError):
                    pass
            return None

        # Check if USER_LOCATION_SELF sentinel is in entities, pop it and
        # try to resolve to GPS coords from metadata / detected_location.
        self_entity_idx = -1
        for i, ent in enumerate(entities or []):
            if (ent or {}).get("name") in (_USER_LOCATION_SELF, "USER_LOCATION_SELF"):
                self_entity_idx = i
                break

        if self_entity_idx != -1:
            entities = list(entities)
            entities.pop(self_entity_idx)  # drop sentinel entity
            # Priority: user_gps in metadata → detected_location string
            raw_gps = (
                str(metadata.get("user_gps") or "").strip()
                or str(detected_location or "").strip()
            )
            gps_source = _try_parse_gps(raw_gps)
            if gps_source:
                self.logger.info(
                    "distance_intent: origin='ở đây' resolved to GPS %s", gps_source
                )
            else:
                # GPS unavailable — ask the user to share location
                dest_hint = str((entities[0] or {}).get("name") or "") if entities else ""
                dest_label = f" đến **{dest_hint}**" if dest_hint else ""
                return {
                    "answer": (
                        f"Để chỉ đường từ vị trí của bạn{dest_label}, "
                        "mình cần biết tọa độ hiện tại của bạn. "
                        "Bạn vui lòng chia sẻ vị trí (GPS) qua ứng dụng nhé!"
                    ),
                    "metadata": {
                        **metadata,
                        "clarification_needed": True,
                        "clarification_reason": "missing_user_gps",
                    },
                }

        # When only 1 entity (destination) and no sentinel, try detected_location as source.
        if gps_source is None and len(entities or []) < 2:
            loc = (detected_location or "").strip()
            if loc and len(entities) == 1:
                # Check if loc is GPS coords (e.g. "13.9,108.0")
                gps_source = _try_parse_gps(loc)
                if gps_source:
                    self.logger.info("distance_intent: using GPS coords as source: %s", gps_source)
                else:
                    entities = [{"name": loc, "type": "Location"}] + list(entities)
                    self.logger.info("distance_intent: using detected_location='%s' as source", loc)
            if gps_source is None and len(entities or []) < 2:
                return {
                    "answer": "Mình chưa xác định đủ 2 địa điểm để tính khoảng cách. Bạn hãy nêu rõ điểm đi và điểm đến.",
                    "metadata": metadata,
                }

        resolved_nodes = []
        used_node_ids = set()
        for entity in entities[:2]:
            available_grounded = [
                n for n in (grounded_nodes or [])
                if str(getattr(n, "id", "")) not in used_node_ids
            ]
            best = self._select_best_grounded_for_entity(entity, available_grounded)
            if not best:
                more = self.retriever.ground_entities([entity])
                available_more = [
                    n for n in (more or [])
                    if str(getattr(n, "id", "")) not in used_node_ids
                ]
                best = self._select_best_grounded_for_entity(entity, available_more)
            if best:
                resolved_nodes.append(best)
                used_node_ids.add(str(getattr(best, "id", "")))

        # GPS source: resolve only destination, use GPS coords for source.
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
                # No coords for destination — try a name-based Google Maps link
                name_map_url = self.directions_service.build_external_map_url_flexible(
                    origin_coords=gps_source,
                    destination_name=dst_name,
                )
                answer = f"Mình đã xác định '{dst_name}' nhưng thiếu tọa độ chính xác."
                if name_map_url:
                    answer = f"{answer} Bạn có thể mở Google Maps để xem đường đi:\n\n📍 {name_map_url}"
                return {"answer": answer, "metadata": {**metadata, "map_url": name_map_url}}
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
            map_url = directions_data.get("map_url") or self.directions_service.build_external_map_url(
                {"lat": src_lat, "lng": src_lng},
                {"lat": float(dst_lat), "lng": float(dst_lng)},
            )
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
            if map_url:
                answer = f"{answer}\n\n📍 Xem đường đi trên Google Maps: {map_url}"
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
                "map_url": map_url,
            }
            return {"answer": answer, "metadata": metadata}

        if len(resolved_nodes) < 2:
            # When GPS source is available but destination not resolved in graph:
            # Skip TravelInfo entirely (no graph node to search for) and go straight
            # to Google Maps link — TravelInfo would only return generic/wrong content.
            if gps_source:
                dest_entity_name = str((entities[0] or {}).get("name") or "") if entities else ""
                fallback_map_url = self.directions_service.build_external_map_url_flexible(
                    origin_coords=gps_source,
                    destination_name=dest_entity_name,
                )
                fallback_answer = (
                    f"M\u00ecnh ch\u01b0a t\u00ecm th\u1ea5y '\u200b{dest_entity_name}' trong c\u01a1 s\u1edf d\u1eef li\u1ec7u \u0111\u1ecba \u0111i\u1ec3m n\u00eay."
                    if dest_entity_name
                    else "M\u00ecnh ch\u01b0a x\u00e1c \u0111\u1ecbnh \u0111\u01b0\u1ee3c \u0111\u1ecba \u0111i\u1ec3m \u0111\u1ebfn."
                )
                if fallback_map_url:
                    fallback_answer = (
                        f"{fallback_answer} Tuy nhi\u00ean b\u1ea1n c\u00f3 th\u1ec3 m\u1edf Google Maps \u0111\u1ec3 xem \u0111\u01b0\u1eddng \u0111i tr\u1ef1c ti\u1ebfp:\n\n"
                        f"📍 {fallback_map_url}"
                    )
                else:
                    fallback_answer = f"{fallback_answer} B\u1ea1n vui l\u00f2ng t\u00ecm ki\u1ebfm tr\u00ean Google Maps nh\u00e9."
                return {
                    "answer": fallback_answer,
                    "metadata": {**metadata, "map_url": fallback_map_url},
                }

            # No GPS source — both endpoints are named places, try TravelInfo
            source_name = str((entities or [{}])[0].get("name") or "")
            dest_name = str((entities or [{}, {}])[1].get("name") or "") if len(entities) > 1 else ""

            # Build a Google Maps link for the fallback even without full coord resolution.
            fallback_map_url = self.directions_service.build_external_map_url_flexible(
                origin_coords=None,
                origin_name=source_name,
                destination_node=resolved_nodes[0] if resolved_nodes else None,
                destination_name=dest_name,
            )

            if source_name and dest_name:
                travel_info = self._lookup_travel_info_for_distance(source_name, dest_name)
                if travel_info:
                    self.logger.info("distance_travel_info_fallback: found TravelInfo for '%s' -> '%s'", source_name, dest_name)
                    answer = travel_info
                    if fallback_map_url:
                        answer = f"{answer}\n\n📍 Xem \u0111\u01b0\u1eddng \u0111i tr\u00ean Google Maps: {fallback_map_url}"
                    return {
                        "answer": answer,
                        "metadata": {**metadata, "map_url": fallback_map_url},
                    }

            fallback_answer = "D\u1eef li\u1ec7u hi\u1ec7n ch\u01b0a c\u00f3 th\u00f4ng tin kho\u1ea3ng c\u00e1ch ho\u1eb7c tuy\u1ebfn xe c\u1ee5 th\u1ec3 gi\u1eefa hai \u0111\u1ecba \u0111i\u1ec3m n\u00e0y."
            if fallback_map_url:
                fallback_answer = f"{fallback_answer}\n\n📍 B\u1ea1n c\u00f3 th\u1ec3 xem \u0111\u01b0\u1eddng \u0111i tr\u00ean Google Maps: {fallback_map_url}"
            else:
                fallback_answer = f"{fallback_answer} B\u1ea1n vui l\u00f2ng tham kh\u1ea3o b\u1ea3n \u0111\u1ed3 tr\u1ef1c ti\u1ebfp ho\u1eb7c c\u00e1c g\u1ee3i \u00fd di chuy\u1ec3n c\u00f4ng c\u1ed9ng kh\u00e1c."
            return {
                "answer": fallback_answer,
                "metadata": {**metadata, "map_url": fallback_map_url},
            }

        src = resolved_nodes[0]
        dst = resolved_nodes[1]
        if str(getattr(src, "id", "")) == str(getattr(dst, "id", "")):
            self.logger.warning(
                "distance_same_node_selected entity_source='%s' entity_target='%s' node_id='%s'",
                str((entities or [{}])[0].get("name") or ""),
                str((entities or [{}, {}])[1].get("name") or ""),
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
            answer = f"{answer}\n\n📍 Xem đường đi trên Google Maps: {map_url}"

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
