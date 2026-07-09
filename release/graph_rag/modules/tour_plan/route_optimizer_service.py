from __future__ import annotations

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

import json
import logging
import re
from typing import Any, Dict, List, Optional

from graph_rag.config import cfg as _cfg
from graph_rag.core.intents import IntentType
from graph_rag.core.state import ConstraintSpec
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)

_kw = _cfg.keywords()

class TourRouteOptimizerService:
    # Zone definitions loaded from config/route_zones.py
    from graph_rag.config.route_zones import KHUVUC_ADJACENCY, ZONE_KEYWORDS, ZONE_CENTERS, PROFILE_ZONE_MAP

    # Tour là gói du lịch (multi-point itinerary), không phải 1 waypoint.
    # Route optimizer chỉ dùng điểm đơn (TouristAttraction, Restaurant, Accommodation).
    TOUR_ROUTE_ALLOWED_TYPES = {"TouristAttraction", "Restaurant", "Accommodation"}

    DISTANCE_OPTIMIZE_HINTS = set(_kw.get("distance_optimize_hints", []))
    WALKING_HINTS = set(_kw.get("walking_hints", []))
    LOW_MOBILITY_HINTS = set(_kw.get("low_mobility_hints", []))

    def __init__(
        self,
        driver,
        *,
        logger,
        haversine_fn,
        tour_plan_max_hop_km: float,
        walking_max_hop_km: float,
        senior_family_max_hop_km: float,
    ):
        self.driver = driver
        self.logger = logger
        self._haversine_km = haversine_fn
        self.tour_plan_max_hop_km = float(tour_plan_max_hop_km)
        self.walking_max_hop_km = float(walking_max_hop_km)
        self.senior_family_max_hop_km = float(senior_family_max_hop_km)

    def extract_trip_days(self, user_query: str) -> int:
        norm_query = normalize_text(user_query)
        if any(token in norm_query for token in [
            "nua ngay", "nửa ngày", "trong ngay", "1 buoi", "1 buổi",
            "trong mot ngay", "trong 1 ngay", "1 ngay",
        ]):
            return 1
        compact = re.search(r"\b(\d{1,2})\s*n\s*(\d{1,2})\s*d\b", norm_query)
        if compact:
            return max(1, min(int(compact.group(1)), 14))
        day_match = re.search(r"\b(\d{1,2})\s*(?:ngay|nay|ngy)\b", norm_query)
        if day_match:
            return max(1, min(int(day_match.group(1)), 14))
        return 2

    def compute_adaptive_max_hop_km(self, intent: str, trip_days: int, mobility_mode: str = "default") -> float:
        if mobility_mode == "walking":
            return self.walking_max_hop_km
        if mobility_mode == "low_mobility":
            return self.senior_family_max_hop_km

        if intent == IntentType.DISCOVERY:
            return 8.0
        if intent == IntentType.ACCOMMODATION:
            return 15.0
        if intent == IntentType.FOOD:
            return 10.0
        if intent == IntentType.TOUR_PLAN:
            if trip_days >= 5:
                return 100.0
            if trip_days >= 3:
                return 60.0
            return 25.0
        return 12.0

    def extract_route_constraints(
        self,
        primary_intent: str,
        user_query: str,
        analyzer_constraints: Optional[Dict[str, Any]],
        trip_days: int = 2,
    ) -> Dict[str, Any]:
        if primary_intent != IntentType.TOUR_PLAN:
            return {"optimize_distance": False, "mobility_mode": "default", "max_hop_km_override": None}

        norm_query = normalize_text(user_query)
        is_walking = any(hint in norm_query for hint in self.WALKING_HINTS)
        is_low_mobility = any(hint in norm_query for hint in self.LOW_MOBILITY_HINTS)
        mobility_mode = "walking" if is_walking else ("low_mobility" if is_low_mobility else "default")

        hop_override = self.compute_adaptive_max_hop_km(
            primary_intent,
            trip_days=trip_days,
            mobility_mode=mobility_mode,
        )

        constraints = analyzer_constraints or {}
        if isinstance(constraints.get("optimize_distance"), bool):
            return {
                "optimize_distance": bool(constraints.get("optimize_distance")),
                "mobility_mode": mobility_mode,
                "max_hop_km_override": hop_override,
            }

        optimize_distance = any(hint in norm_query for hint in self.DISTANCE_OPTIMIZE_HINTS)
        return {
            "optimize_distance": optimize_distance,
            "mobility_mode": mobility_mode,
            "max_hop_km_override": hop_override,
        }

    @staticmethod
    def extract_mobility_mode_from_constraints(constraints: List[ConstraintSpec]) -> str:
        """Determine mobility mode from a list of ConstraintSpec objects.

        Returns 'walking' if any constraint has feature=='walking',
        'low_mobility' if any has feature=='low_mobility',
        'default' otherwise.
        """
        for constraint in constraints:
            if constraint.feature == "walking":
                return "walking"
            if constraint.feature == "low_mobility":
                return "low_mobility"
        return "default"

    def build_tour_route_candidates(
        self,
        seeds: List[Any],
        intent: str,
        *,
        query_state: Any = None,
    ) -> List[Dict[str, Any]]:
        if intent != IntentType.TOUR_PLAN or not seeds:
            return []

        # Extract constraint signals from query_state
        coastal_required = getattr(query_state, "coastal_required", False)
        sunset_required = getattr(query_state, "sunset_required", False)
        island_required = getattr(query_state, "island_required", False)

        candidates: List[Dict[str, Any]] = []
        seen = set()
        skipped_type = 0
        tour_expanded = 0
        for seed in seeds:
            lat = seed.metadata.get("lat")
            lng = seed.metadata.get("lng")

            labels = seed.metadata.get("labels") or []
            node_type = (labels[0] if labels else (seed.metadata.get("type") or "")).strip()

            # Tour nodes: expand into included TouristAttraction points
            if node_type == "Tour":
                included = self._expand_tour_included_points(seed)
                for point in included:
                    key = str(point["id"])
                    if key not in seen:
                        seen.add(key)
                        candidates.append(point)
                        tour_expanded += 1
                continue

            # Location nodes: expand into child TouristAttraction/Restaurant/Accommodation
            # via LOCATED_IN relationship.  When PolicyRanker keeps only a Location seed
            # (e.g. "Gia Lai") and drops TouristAttraction seeds, this ensures the route
            # optimizer still has actual map points to work with.
            if node_type == "Location":
                included = self._expand_location_points(seed)
                for point in included:
                    key = str(point["id"])
                    if key not in seen:
                        seen.add(key)
                        candidates.append(point)
                        tour_expanded += 1
                continue

            if node_type not in self.TOUR_ROUTE_ALLOWED_TYPES:
                skipped_type += 1
                continue

            key = str(seed.id)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "id": seed.id,
                    "name": seed.metadata.get("name") or seed.content,
                    "labels": labels,
                    "attributes": seed.metadata,
                    "lat": lat,
                    "lng": lng,
                }
            )
        if skipped_type or tour_expanded:
            logger.info("   -> [RouteOptimizer] build_tour_route_candidates: %s seeds → %s candidates (skipped %s by type, expanded %s from Tour/Location)", len(seeds), len(candidates), skipped_type, tour_expanded)

        # Constraint-aware prioritization: ensure coastal/sunset/island candidates
        # are in the backbone when required
        if coastal_required or sunset_required or island_required:
            # Check if constraint is hard (mandatory) vs soft (preferable)
            _constraints = getattr(query_state, "constraints", []) or []
            _is_hard_coastal = any(
                getattr(c, "feature", "") == "coastal" and getattr(c, "is_hard", False)
                for c in _constraints
            )
            candidates = self._prioritize_constraint_candidates(
                candidates,
                coastal_required=coastal_required,
                sunset_required=sunset_required,
                island_required=island_required,
                hard_filter=_is_hard_coastal,
            )

        days = 3
        if query_state:
            days = getattr(query_state, "duration_days", 0) or 3
            if days <= 0 and getattr(query_state, "query", ""):
                days = self.extract_trip_days(query_state.query)
        return self._apply_tour_candidate_semantic_quotas(candidates, max_items=12, days=days)

    # Constraint keyword sets for route candidate matching
    # Extended to include general coastal terms so queries like "điểm có biển" or
    # "ven biển Bình Định" can match even without naming a specific beach.
    _COASTAL_ROUTE_TERMS = {
        # Specific coastal landmarks in Bình Định / Quy Nhơn
        "ky co", "eo gio", "cu lao xanh", "hon kho", "nhon ly",
        "trung luong", "cat tien", "bai xep",
        # General coastal terms (also match "bãi biển", "ven biển", "biển Quy Nhơn", …)
        "bai bien", "bien quy nhon", "ven bien", "bien dong",
        "bien binh dinh", "ho tay", "bai cat", "vinh quy nhon",
        # Extra: address-level cues in Bình Định coastal nodes
        "nhon hai", "nhon ly", "nhon hoi", "ghenh rang",
        # Catch-all single token — lowest priority, needed when description says "biển"
        "bien",
    }
    _SUNSET_ROUTE_TERMS = set(_kw.get("sunset_route_terms", []))
    _ISLAND_ROUTE_TERMS = set(_kw.get("island_route_terms", []))

    def _is_constraint_match(self, candidate: Dict[str, Any], terms: set) -> bool:
        attrs = candidate.get("attributes") or {}
        fields = [
            candidate.get("name", ""),
            attrs.get("address", ""),
            attrs.get("description", ""),
            attrs.get("source_tour", ""),
            candidate.get("content", ""),
        ]
        text = normalize_text(" ".join(str(f) for f in fields), strip_punct=True)
        norm_terms = [normalize_text(t, strip_punct=True) for t in terms]
        return any(t in text for t in norm_terms)

    def _prioritize_constraint_candidates(
        self,
        candidates: List[Dict[str, Any]],
        *,
        coastal_required: bool,
        sunset_required: bool,
        island_required: bool,
        hard_filter: bool = False,
    ) -> List[Dict[str, Any]]:
        """Reorder candidates so constraint-matching nodes come first (backbone priority).

        When hard_filter=True, non-matching candidates are REMOVED (not just deprioritized).
        This enforces hard constraints like "yêu cầu có biển".

        Returns the reordered list AND sets ``_constraint_match_count`` attribute on the
        returned list object so callers can detect zero-match situations without re-scanning.
        """
        constraint_matches = []
        rest = []

        all_terms = set()
        if coastal_required:
            all_terms |= self._COASTAL_ROUTE_TERMS
        if sunset_required:
            all_terms |= self._SUNSET_ROUTE_TERMS
        if island_required:
            all_terms |= self._ISLAND_ROUTE_TERMS

        for c in candidates:
            if self._is_constraint_match(c, all_terms):
                constraint_matches.append(c)
            else:
                rest.append(c)

        # Hard filter: remove non-matching candidates when constraint is mandatory
        if hard_filter and constraint_matches and rest:
            _removed = len(rest)
            logger.info("   -> [RouteOptimizer] Hard filter: removed %s candidates not matching constraints", _removed)
            rest = []

        if constraint_matches:
            logger.info(
                "   -> [RouteOptimizer] Constraint-aware: %s"
                " constraint-matching candidates prioritized (coastal=%s,"
                " sunset=%s, island=%s)",
                len(constraint_matches), coastal_required, sunset_required, island_required,
            )
        else:
            logger.info(
                "   -> [RouteOptimizer] WARNING: No candidates match"
                " coastal=%s/sunset=%s/island=%s constraints."
                " Total candidates=%s. Returning unsorted list.",
                coastal_required, sunset_required, island_required, len(candidates),
            )

        result = constraint_matches + rest
        # Track count so optimize_tour_route_nodes can build constraint_warning
        self._last_constraint_match_count = len(constraint_matches)
        self._last_constraint_required = {
            "coastal": coastal_required,
            "sunset": sunset_required,
            "island": island_required,
        }
        return result

    def _expand_tour_included_points(self, tour_seed: Any) -> List[Dict[str, Any]]:
        """Expand a Tour node into its included TouristAttraction points via INCLUDES relationship."""
        tour_id = getattr(tour_seed, "id", None)
        if not tour_id or not self.driver:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (t:Tour {id: $tour_id})-[:INCLUDES]->(a:TouristAttraction)
                    WHERE a.location IS NOT NULL
                    RETURN a.id AS id, a.name AS name, a.description AS description,
                           a.location.latitude AS lat, a.location.longitude AS lng,
                           a.address AS address, labels(a) AS labels
                    """,
                    tour_id=str(tour_id),
                )
                points = []
                for record in result:
                    points.append({
                        "id": record["id"],
                        "name": record["name"],
                        "labels": record["labels"] or ["TouristAttraction"],
                        "attributes": {
                            "name": record["name"],
                            "description": record.get("description"),
                            "address": record.get("address"),
                            "source_tour": getattr(tour_seed, "metadata", {}).get("name", ""),
                        },
                        "lat": record["lat"],
                        "lng": record["lng"],
                    })
                return points
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("   -> [RouteOptimizer] _expand_tour_included_points error: %s", e)
            return []

    def _expand_location_points(self, location_seed: Any) -> List[Dict[str, Any]]:
        """Expand a Location node into its child TouristAttraction / Restaurant / Accommodation
        points via the LOCATED_IN relationship.  This ensures the route optimizer has actual
        map points when the only seed is a Location (e.g. after PolicyRanker drops other types)."""
        location_id = getattr(location_seed, "id", None)
        if not location_id or not self.driver:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (a)-[:LOCATED_IN]->(l:Location {id: $location_id})
                    WHERE (a:TouristAttraction OR a:Restaurant OR a:Accommodation)
                      AND a.location IS NOT NULL
                    RETURN a.id AS id, a.name AS name, a.description AS description,
                           a.location.latitude AS lat, a.location.longitude AS lng,
                           a.address AS address, labels(a) AS labels
                    LIMIT 20
                    """,
                    location_id=str(location_id),
                )
                points = []
                for record in result:
                    points.append({
                        "id": record["id"],
                        "name": record["name"],
                        "labels": record["labels"] or ["TouristAttraction"],
                        "attributes": {
                            "name": record["name"],
                            "description": record.get("description"),
                            "address": record.get("address"),
                            "source_location": getattr(location_seed, "metadata", {}).get("name", ""),
                        },
                        "lat": record["lat"],
                        "lng": record["lng"],
                    })
                if points:
                    logger.info("   -> [RouteOptimizer] Expanded Location '%s' into %d route points",
                                getattr(location_seed, "metadata", {}).get("name", location_id), len(points))
                return points
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("   -> [RouteOptimizer] _expand_location_points error: %s", e)
            return []

    def _apply_tour_candidate_semantic_quotas(
        self,
        candidates: List[Dict[str, Any]],
        *,
        max_items: int,
        days: int = 3,
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        def primary_label(item: Dict[str, Any]) -> str:
            labels = item.get("labels") or []
            if labels:
                return str(labels[0])
            return str((item.get("attributes") or {}).get("type") or "")

        food_labels = {"Restaurant", "Dish", "Specialty"}
        attraction_labels = {"TouristAttraction", "Tour"}
        accommodation_labels = {"Accommodation"}

        attractions = [x for x in candidates if primary_label(x) in attraction_labels]
        foods = [x for x in candidates if primary_label(x) in food_labels]
        accommodations = [x for x in candidates if primary_label(x) in accommodation_labels]
        others = [x for x in candidates if primary_label(x) not in (food_labels | attraction_labels | accommodation_labels)]

        selected: List[Dict[str, Any]] = []
        selected_ids = set()
        accommodation_count = 0

        def push(item: Dict[str, Any]) -> bool:
            nonlocal accommodation_count
            key = str(item.get("id"))
            if key in selected_ids:
                return False
            label = primary_label(item)
            if label in accommodation_labels and accommodation_count >= 1:
                return False
            selected.append(item)
            selected_ids.add(key)
            if label in accommodation_labels:
                accommodation_count += 1
            return True

        # Limit restaurant quota = min(days if days > 0 else 2, 3) — 1 per meal slot per day, capped
        restaurant_quota = min(days if days > 0 else 2, 3)
        # Need 2 attractions per day (morning + afternoon)
        attraction_quota = min(len(attractions), max(4, days * 2))

        for item in attractions[:attraction_quota]:
            push(item)
        for item in foods[:restaurant_quota]:
            push(item)
        for item in accommodations[:1]:
            push(item)

        for pool in (attractions, foods, others, accommodations):
            for item in pool:
                if len(selected) >= max_items:
                    break
                push(item)
            if len(selected) >= max_items:
                break

        return selected[:max_items]

    def _node_lat_lng(self, node: Dict[str, Any]):
        lat = node.get("lat")
        lng = node.get("lng")
        if lat is None or lng is None:
            return None
        try:
            return (float(lat), float(lng))
        except (ValueError, TypeError):
            return None

    def _path_total_km(self, route_nodes: List[Dict[str, Any]]) -> float:
        if len(route_nodes) < 2:
            return 0.0
        total = 0.0
        for i in range(len(route_nodes) - 1):
            a = self._node_lat_lng(route_nodes[i])
            b = self._node_lat_lng(route_nodes[i + 1])
            if not a or not b:
                continue
            total += self._haversine_km(a[0], a[1], b[0], b[1])
        return total

    def _compute_hop_distances_km(self, route_nodes: List[Dict[str, Any]]) -> List[float]:
        if len(route_nodes) < 2:
            return []
        hops: List[float] = []
        for i in range(len(route_nodes) - 1):
            a = self._node_lat_lng(route_nodes[i])
            b = self._node_lat_lng(route_nodes[i + 1])
            if not a or not b:
                continue
            hops.append(round(self._haversine_km(a[0], a[1], b[0], b[1]), 2))
        return hops

    def _build_route_optimizer_metrics(
        self,
        *,
        optimization_requested: bool,
        optimization_applied: bool,
        input_points: int,
        points_before_guardrail: int,
        points_after_guardrail: int,
        max_hop_km: float,
        hop_distances_km: List[float],
        dropped_route_points: List[str],
        total_distance_km: float,
        route_engine: str,
    ) -> Dict[str, Any]:
        return {
            "optimization_requested": bool(optimization_requested),
            "optimization_applied": bool(optimization_applied),
            "input_points": int(input_points),
            "points_before_guardrail": int(points_before_guardrail),
            "points_after_guardrail": int(points_after_guardrail),
            "dropped_points": int(len(dropped_route_points or [])),
            "max_hop_km_config": float(max_hop_km),
            "max_hop_km_actual": round(max(hop_distances_km), 2) if hop_distances_km else 0.0,
            "total_distance_km": round(float(total_distance_km), 2),
            "route_engine": route_engine,
        }

    def fetch_lodging_suggestions(
        self,
        detected_location: str,
        route_nodes: List[Dict[str, Any]],
        *,
        max_lodging: int = 3,
        region_focus: str = "",
    ) -> List[Dict[str, Any]]:
        """Fetch Accommodation nodes near the tour area for multi-day lodging suggestions.

        Returns a list of dicts with keys: id, name, address, type, lat, lng.
        These are NOT added to the touring route — they are rendered in a separate
        "Gợi ý nghỉ đêm" section.
        """
        if not self.driver:
            return []

        location_norm = normalize_text(detected_location or "")
        location_filter = ""
        if "quy nhon" in location_norm or "binh dinh" in location_norm:
            location_filter = "Quy Nhơn"
        elif "gia lai" in location_norm or "pleiku" in location_norm:
            location_filter = "Gia Lai"

        try:
            with self.driver.session() as session:
                search_centers = self._lodging_search_centers(
                    detected_location,
                    route_nodes,
                    max_lodging=max_lodging,
                    region_focus=region_focus,
                )
                lodging = []
                seen_names = set()
                for center_lat, center_lng in search_centers:
                    cypher = """
                        MATCH (a:Accommodation)
                        WHERE a.location IS NOT null
                        WITH a, point.distance(
                            a.location,
                            point({latitude: $lat, longitude: $lng})
                        ) AS dist
                        ORDER BY dist ASC
                        LIMIT $limit
                        RETURN a.name AS name,
                               a.address AS address,
                               a.type AS type,
                               a.phone AS phone,
                               a.location.latitude AS lat,
                               a.location.longitude AS lng,
                               elementId(a) AS id,
                               dist
                    """
                    result = session.run(
                        cypher,
                        lat=center_lat,
                        lng=center_lng,
                        limit=max_lodging,
                    )
                    for record in result:
                        name = record.get("name")
                        if not name or name in seen_names:
                            continue
                        seen_names.add(name)
                        lodging.append({
                            "id": record.get("id"),
                            "name": name,
                            "address": record.get("address") or "",
                            "type": record.get("type") or "Accommodation",
                            "phone": record.get("phone") or "",
                            "lat": record.get("lat"),
                            "lng": record.get("lng"),
                        })
                        if len(lodging) >= max_lodging:
                            return lodging
                if lodging:
                    return lodging

                if location_filter:
                    cypher = """
                        MATCH (a:Accommodation)-[:LOCATED_IN]->(l:Location)
                        WHERE l.name CONTAINS $loc
                        RETURN a.name AS name,
                               a.address AS address,
                               a.type AS type,
                               a.phone AS phone,
                               a.location.latitude AS lat,
                               a.location.longitude AS lng,
                               elementId(a) AS id
                        LIMIT $limit
                    """
                    result = session.run(cypher, loc=location_filter, limit=max_lodging)
                else:
                    cypher = """
                        MATCH (a:Accommodation)
                        WHERE a.location IS NOT null
                        RETURN a.name AS name,
                               a.address AS address,
                               a.type AS type,
                               a.phone AS phone,
                               a.location.latitude AS lat,
                               a.location.longitude AS lng,
                               elementId(a) AS id
                        LIMIT $limit
                    """
                    result = session.run(cypher, limit=max_lodging)

                for record in result:
                    name = record.get("name")
                    if not name:
                        continue
                    lodging.append({
                        "id": record.get("id"),
                        "name": name,
                        "address": record.get("address") or "",
                        "type": record.get("type") or "Accommodation",
                        "phone": record.get("phone") or "",
                        "lat": record.get("lat"),
                        "lng": record.get("lng"),
                    })
                return lodging
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
            self.logger.warning("fetch_lodging_suggestions error: %s", e)
            return []

    def _lodging_search_centers(
        self,
        detected_location: str,
        route_nodes: List[Dict[str, Any]],
        *,
        max_lodging: int,
        region_focus: str = "",
    ) -> List[tuple[float, float]]:
        grouped: Dict[str, List[tuple[float, float]]] = {}
        for node in route_nodes or []:
            ll = self._node_lat_lng(node)
            if not ll:
                continue
            kv = self._assign_khu_vuc(node, detected_location, region_focus)
            region = self._ZONE_TO_REGION.get(kv, "unknown")
            grouped.setdefault(region, []).append(ll)

        region_order = {"inland": 0, "coastal": 1, "unknown": 2}
        centers: List[tuple[float, float]] = []
        for region in sorted(grouped.keys(), key=lambda r: region_order.get(r, 99)):
            points = grouped[region]
            if not points:
                continue
            centers.append((
                sum(p[0] for p in points) / len(points),
                sum(p[1] for p in points) / len(points),
            ))

        if len(centers) >= max_lodging:
            return centers[:max_lodging]

        all_points = [p for points in grouped.values() for p in points]
        if all_points:
            centers.append((
                sum(p[0] for p in all_points) / len(all_points),
                sum(p[1] for p in all_points) / len(all_points),
            ))
        return centers[:max_lodging]

    def _build_neo4j_distance_matrix(self, route_nodes: List[Dict[str, Any]]) -> tuple[Dict[tuple, float], str]:
        if len(route_nodes) < 2:
            return {}, "local_fallback"

        node_by_id = {str(n.get("id")): n for n in route_nodes if n.get("id") is not None}
        ids = list(node_by_id.keys())
        if len(ids) < 2:
            return {}, "local_fallback"

        matrix: Dict[tuple, float] = {}
        try:
            with self.driver.session() as session:
                records = session.run(
                    """
                    MATCH (a)
                    WHERE toString(a.id) IN $ids AND a.location IS NOT NULL
                    MATCH (b)
                    WHERE toString(b.id) IN $ids AND b.location IS NOT NULL
                    AND toString(a.id) <> toString(b.id)
                    RETURN toString(a.id) AS src_id,
                        toString(b.id) AS dst_id,
                        point.distance(a.location, b.location) / 1000.0 AS km
                    """,
                    ids=ids,
                )
                for record in records:
                    src_id = record.get("src_id")
                    dst_id = record.get("dst_id")
                    km = record.get("km")
                    if src_id and dst_id and isinstance(km, (int, float)):
                        matrix[(src_id, dst_id)] = float(km)
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.logger.warning("neo4j_route_matrix_fallback: %s", str(exc))

        for src in route_nodes:
            src_id = str(src.get("id"))
            src_ll = self._node_lat_lng(src)
            if not src_ll:
                continue
            for dst in route_nodes:
                dst_id = str(dst.get("id"))
                if src_id == dst_id:
                    continue
                if (src_id, dst_id) in matrix:
                    continue
                dst_ll = self._node_lat_lng(dst)
                if not dst_ll:
                    continue
                matrix[(src_id, dst_id)] = self._haversine_km(src_ll[0], src_ll[1], dst_ll[0], dst_ll[1])

        engine = "neo4j_point_distance" if any(k in matrix for k in matrix.keys()) else "local_fallback"
        return matrix, engine

    def _graph_distance(self, a: Dict[str, Any], b: Dict[str, Any], matrix: Dict[tuple, float]) -> float:
        a_id = str(a.get("id"))
        b_id = str(b.get("id"))
        dist = matrix.get((a_id, b_id))
        if dist is not None:
            return dist
        a_ll = self._node_lat_lng(a)
        b_ll = self._node_lat_lng(b)
        if not a_ll or not b_ll:
            return 1e9
        return self._haversine_km(a_ll[0], a_ll[1], b_ll[0], b_ll[1])

    def _graph_nearest_neighbor_route(self, nodes: List[Dict[str, Any]], matrix: Dict[tuple, float]) -> List[Dict[str, Any]]:
        valid = [n for n in nodes if self._node_lat_lng(n)]
        if len(valid) <= 2:
            return valid

        route = [valid[0]]
        remaining = valid[1:]
        while remaining:
            last = route[-1]
            next_idx = min(
                range(len(remaining)),
                key=lambda idx: self._graph_distance(last, remaining[idx], matrix),
            )
            route.append(remaining.pop(next_idx))
        return route

    def _graph_path_total_km(self, route_nodes: List[Dict[str, Any]], matrix: Dict[tuple, float]) -> float:
        if len(route_nodes) < 2:
            return 0.0
        total = 0.0
        for i in range(len(route_nodes) - 1):
            total += self._graph_distance(route_nodes[i], route_nodes[i + 1], matrix)
        return total

    def _khu_vuc_profile(self, detected_location: str, region_focus: str = "") -> str:
        """Determine zone profile based on location and region focus.

        After administrative merger, generic "Gia Lai" uses the merged profile
        that includes both Pleiku inland and Quy Nhơn coastal zones.
        """
        norm = normalize_text(detected_location or "")
        # Merged province trips span Pleiku inland and Quy Nhon/Binh Dinh
        # coastal areas. Honor the explicit route scope even when the detected
        # location resolves to only one end of the route.
        if region_focus in ("all", "gia_lai_new"):
            return "gia_lai_new"
        if "gia lai" in norm or "pleiku" in norm:
            return "gia_lai"
        return "quy_nhon"

    def _assign_khu_vuc(self, node: Dict[str, Any], detected_location: str, region_focus: str = "") -> str:
        profile = self._khu_vuc_profile(detected_location, region_focus)
        text = normalize_text(
            " ".join(
                [
                    str(node.get("name") or ""),
                    str((node.get("attributes") or {}).get("address") or ""),
                ]
            )
        )
        ll = self._node_lat_lng(node)

        # Try keyword-based assignment from config
        zone_sets = self.PROFILE_ZONE_MAP.get(profile, [])
        for zone_set_name in zone_sets:
            for zone_name, keywords in (self.ZONE_KEYWORDS.get(zone_set_name) or {}).items():
                if any(token in text for token in keywords):
                    return zone_name

        # Try legacy_province or province based assignment
        lp = node.get("legacy_province") or (node.get("attributes") or {}).get("legacy_province") or node.get("province") or (node.get("attributes") or {}).get("province")
        if not lp:
            lp = node.get("entity_legacy_province") or (node.get("metadata") or {}).get("legacy_province") or (node.get("metadata") or {}).get("province")
        if lp:
            lp_norm = normalize_text(str(lp), strip_punct=True)
            if "gia lai" in lp_norm:
                return "Khu Ngoai O Pleiku"
            elif "binh dinh" in lp_norm:
                return "Khu Ven Bien Nam"

        # Distance-based fallback from config
        for zone_name, center in self.ZONE_CENTERS.items():
            if ll and self._haversine_km(center[0], center[1], ll[0], ll[1]) <= 9.0:
                return zone_name

        # Default based on profile
        if "gia_lai" in profile:
            return "Khu Ngoai O Pleiku"
        return "Khu Ven Bien Nam"

    def _distance_from_quy_nhon_center(self, lat: float, lng: float) -> float:
        return self._haversine_km(13.7820, 109.2197, float(lat), float(lng))

    def _distance_from_pleiku_center(self, lat: float, lng: float) -> float:
        return self._haversine_km(13.9833, 108.0000, float(lat), float(lng))

    def _khu_vuc_adjacent(self, a: str, b: str, profile: str) -> bool:
        if a == b:
            return True
        adjacency = self.KHUVUC_ADJACENCY.get(profile, {})
        return b in adjacency.get(a, set())

    # ── Region mapping: zone → high-level region ──────────────────────

    _ZONE_TO_REGION = {
        "Khu Trung Tam Pleiku": "inland",
        "Khu Cao Nguyen": "inland",
        "Khu Ngoai O Pleiku": "inland",
        "Khu Trung Tam": "coastal",
        "Khu Ban Dao": "coastal",
        "Khu Ven Bien Bac": "coastal",
        "Khu Ven Bien Nam": "coastal",
    }

    def _node_region(self, node: Dict[str, Any]) -> str:
        """Get high-level region (inland/coastal) from node's zone."""
        lp = node.get("legacy_province") or (node.get("attributes") or {}).get("legacy_province") or node.get("province") or (node.get("attributes") or {}).get("province")
        if not lp:
            lp = node.get("entity_legacy_province") or (node.get("metadata") or {}).get("legacy_province") or (node.get("metadata") or {}).get("province")
        if lp:
            lp_norm = normalize_text(str(lp), strip_punct=True)
            if "gia lai" in lp_norm:
                return "inland"
            elif "binh dinh" in lp_norm:
                return "coastal"

        khu_vuc = node.get("khu_vuc", "")
        return self._ZONE_TO_REGION.get(khu_vuc, "unknown")

    def _build_daily_cluster_plan(
        self,
        route_nodes: List[Dict[str, Any]],
        days: int,
        detected_location: str,
        region_focus: str = "",
    ) -> List[Dict[str, Any]]:
        if not route_nodes:
            return []

        profile = self._khu_vuc_profile(detected_location, region_focus)
        enriched = []
        for node in route_nodes:
            kv = self._assign_khu_vuc(node, detected_location, region_focus)
            clone = dict(node)
            clone["khu_vuc"] = kv
            enriched.append(clone)

        # Detect multi-region: count nodes per high-level region
        region_counts: Dict[str, int] = {}
        for node in enriched:
            r = self._node_region(node)
            region_counts[r] = region_counts.get(r, 0) + 1

        active_regions = {r: c for r, c in region_counts.items() if r != "unknown" and c > 0}

        # If multiple regions with enough nodes, use region-aware allocation
        if len(active_regions) >= 2 and days >= 2:
            return self._build_multi_region_daily_plan(enriched, days, profile, active_regions)

        # Single-region fallback: existing zone-based clustering
        return self._build_single_region_daily_plan(enriched, days, profile)

    def _build_multi_region_daily_plan(
        self,
        enriched: List[Dict[str, Any]],
        days: int,
        profile: str,
        region_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """Allocate days across regions proportionally, then cluster within each region.

        Example: 3 days, Gia Lai (4 nodes) + Bình Định (6 nodes)
        → Day 1: Gia Lai (1 day)
        → Day 2-3: Bình Định (2 days)

        CRITICAL: Route nodes must be reordered to match this plan.
        Use reorder_route_by_daily_plan() after calling this method.
        """
        total_nodes = sum(region_counts.values())
        # Geographic order: inland (Gia Lai/Pleiku) first, then coastal (Bình Định/Quy Nhơn)
        # NOT alphabetical — "coastal" < "inland" alphabetically but inland should come first
        _REGION_ORDER = {"inland": 0, "coastal": 1, "unknown": 2}
        regions_sorted = sorted(region_counts.keys(), key=lambda r: _REGION_ORDER.get(r, 99))

        # Allocate days proportionally based on node count
        day_allocation: Dict[str, int] = {}
        remaining_days = days
        for i, region in enumerate(regions_sorted):
            if i == len(regions_sorted) - 1:
                day_allocation[region] = remaining_days
            else:
                alloc = max(1, round(days * region_counts[region] / total_nodes))
                alloc = min(alloc, remaining_days - (len(regions_sorted) - i - 1))
                day_allocation[region] = alloc
                remaining_days -= alloc

        # Split nodes by region
        region_nodes: Dict[str, List[Dict[str, Any]]] = {}
        for node in enriched:
            r = self._node_region(node)
            region_nodes.setdefault(r, []).append(node)

        # Build plan: each region gets its allocated days
        plan: List[Dict[str, Any]] = []
        day_counter = 1
        region_name_map = {
            "inland": "Gia Lai",
            "coastal": "Bình Định/Quy Nhơn",
        }

        for region in regions_sorted:
            nodes = region_nodes.get(region, [])
            region_days = day_allocation.get(region, 1)
            if not nodes or region_days <= 0:
                continue

            # Distribute nodes across region's days using zone-based clustering
            sub_plan = self._build_single_region_daily_plan(nodes, region_days, profile)

            # Re-number days to be global
            for entry in sub_plan:
                entry["day"] = day_counter
                entry["region"] = region
                entry["region_label"] = region_name_map.get(region, "")
                plan.append(entry)
                day_counter += 1

        return plan

    def _build_single_region_daily_plan(
        self,
        enriched: List[Dict[str, Any]],
        days: int,
        profile: str,
    ) -> List[Dict[str, Any]]:
        """Original zone-based clustering for a single region, with node deduplication and region labels."""
        # Dedup nodes by normalized name, keeping first occurrence (highest priority)
        seen_names = set()
        deduped = []
        for node in enriched:
            name = node.get("name")
            if name:
                norm = normalize_text(str(name))
                if norm not in seen_names:
                    seen_names.add(norm)
                    deduped.append(node)
            else:
                deduped.append(node)
        enriched = deduped

        day_count = max(1, min(days, len(enriched)))
        day_buckets: List[List[Dict[str, Any]]] = [[] for _ in range(day_count)]
        idx = 0
        for node in enriched:
            placed = False
            for _ in range(day_count):
                bucket = day_buckets[idx]
                areas = {x.get("khu_vuc") for x in bucket if x.get("khu_vuc")}
                candidate_area = node.get("khu_vuc")
                if not areas:
                    bucket.append(node)
                    placed = True
                    break
                if len(areas) == 1:
                    existing = next(iter(areas))
                    if self._khu_vuc_adjacent(existing, candidate_area, profile):
                        bucket.append(node)
                        placed = True
                        break
                if candidate_area in areas:
                    bucket.append(node)
                    placed = True
                    break
                idx = (idx + 1) % day_count

            if not placed:
                day_buckets[idx].append(node)
            idx = (idx + 1) % day_count

        plan: List[Dict[str, Any]] = []
        region_name_map = {
            "inland": "Gia Lai",
            "coastal": "Bình Định/Quy Nhơn",
        }
        for d_idx, bucket in enumerate(day_buckets, start=1):
            area_set = sorted({x.get("khu_vuc") for x in bucket if x.get("khu_vuc")})
            allowed = len(area_set) <= 2
            if len(area_set) == 2:
                allowed = self._khu_vuc_adjacent(area_set[0], area_set[1], profile)
            
            # Infer dominant region label for the single region plan
            regions = [self._node_region(x) for x in bucket]
            non_empty_regions = [r for r in regions if r != "unknown"]
            dominant_region = "unknown"
            if non_empty_regions:
                dominant_region = max(set(non_empty_regions), key=non_empty_regions.count)
            label = region_name_map.get(dominant_region, "")

            plan.append(
                {
                    "day": d_idx,
                    "areas": area_set,
                    "point_names": [x.get("name") for x in bucket if x.get("name")],
                    "rule_ok": bool(allowed),
                    "region_label": label,
                }
            )

        return plan

    def _validate_daily_cluster_plan(self, daily_cluster_plan: List[Dict[str, Any]], min_points_per_day: int = 2) -> tuple[bool, List[int]]:
        if not daily_cluster_plan:
            return False, []

        invalid_days: List[int] = []
        for day in daily_cluster_plan:
            if not isinstance(day, dict):
                continue
            point_count = len(day.get("point_names") or [])
            day_value = day.get("day")
            if point_count < min_points_per_day:
                try:
                    invalid_days.append(int(day_value))
                except (ValueError, TypeError):
                    pass

        return len(invalid_days) == 0, invalid_days

    def _evaluate_route_viability(
        self,
        route_nodes: List[Dict[str, Any]],
        daily_cluster_plan: List[Dict[str, Any]],
        min_points_required: int,
        min_points_per_day: int,
    ) -> tuple[bool, str]:
        plan_ok, invalid_days = self._validate_daily_cluster_plan(
            daily_cluster_plan,
            min_points_per_day=min_points_per_day,
        )
        if len(route_nodes) < min_points_required:
            return False, f"too_few_points({len(route_nodes)}<{min_points_required})"
        if not plan_ok:
            return False, f"invalid_daily_plan(days={invalid_days})"
        return True, ""

    def _build_raw_route_fallback(
        self,
        fallback_nodes: List[Dict[str, Any]],
        days: int,
        detected_location: str,
        route_engine: str,
        engine_suffix: str,
        region_focus: str = "",
    ) -> tuple[List[Dict[str, Any]], List[str], List[float], List[Dict[str, Any]], str, bool]:
        nodes = list(fallback_nodes)
        dropped_points: List[str] = []
        hop_distances = self._compute_hop_distances_km(nodes)
        daily_cluster_plan = self._build_daily_cluster_plan(nodes, days, detected_location, region_focus)
        resolved_engine = f"{route_engine}{engine_suffix}"
        optimization_applied = False
        return (
            nodes,
            dropped_points,
            hop_distances,
            daily_cluster_plan,
            resolved_engine,
            optimization_applied,
        )

    def _apply_max_hop_guardrail(
        self,
        route_nodes: List[Dict[str, Any]],
        max_hop_km: float,
        allow_drops: bool = True,
    ) -> tuple[List[Dict[str, Any]], List[str], List[float]]:
        if len(route_nodes) < 2:
            return route_nodes, [], []

        if not allow_drops:
            return route_nodes, [], self._compute_hop_distances_km(route_nodes)

        filtered = [route_nodes[0]]
        dropped_names: List[str] = []
        hop_km = []

        for i in range(1, len(route_nodes)):
            prev = filtered[-1]
            curr = route_nodes[i]
            prev_ll = self._node_lat_lng(prev)
            curr_ll = self._node_lat_lng(curr)
            if not prev_ll or not curr_ll:
                continue

            step_km = self._haversine_km(prev_ll[0], prev_ll[1], curr_ll[0], curr_ll[1])
            if step_km > max_hop_km:
                dropped_names.append(curr.get("name") or str(curr.get("id") or "unknown"))
                continue

            filtered.append(curr)
            hop_km.append(round(step_km, 2))

        if len(filtered) >= 2:
            return filtered, dropped_names, hop_km

        closest_pair = None
        closest_dist = None
        for i in range(len(route_nodes) - 1):
            a = self._node_lat_lng(route_nodes[i])
            if not a:
                continue
            for j in range(i + 1, len(route_nodes)):
                b = self._node_lat_lng(route_nodes[j])
                if not b:
                    continue
                dist = self._haversine_km(a[0], a[1], b[0], b[1])
                if dist > max_hop_km:
                    continue
                if closest_dist is None or dist < closest_dist:
                    closest_dist = dist
                    closest_pair = (route_nodes[i], route_nodes[j])

        if closest_pair:
            return [closest_pair[0], closest_pair[1]], dropped_names, [round(float(closest_dist), 2)]

        return filtered, dropped_names, hop_km

    def _build_region_clustered_route(
        self,
        route_nodes: List[Dict[str, Any]],
        days: int,
        detected_location: str,
        region_focus: str = "",
    ) -> List[Dict[str, Any]]:
        """Order route nodes by region (inland first, then coastal), then by zone within region."""
        if not route_nodes:
            return []

        profile = self._khu_vuc_profile(detected_location, region_focus)
        enriched = []
        for node in route_nodes:
            kv = self._assign_khu_vuc(node, detected_location, region_focus)
            clone = dict(node)
            clone["khu_vuc"] = kv
            clone["_region"] = self._ZONE_TO_REGION.get(kv, "unknown")
            enriched.append(clone)

        # Multi-region: sort inland first, then coastal (Gia Lai → Bình Định)
        region_order = {"inland": 0, "coastal": 1, "unknown": 2}
        enriched.sort(key=lambda n: (
            region_order.get(n["_region"], 2),
            n.get("khu_vuc", ""),
        ))

        # Then cluster by day within each region
        day_count = max(1, min(days, len(enriched)))
        day_buckets: List[List[Dict[str, Any]]] = [[] for _ in range(day_count)]
        idx = 0

        for node in enriched:
            placed = False
            for _ in range(day_count):
                bucket = day_buckets[idx]
                areas = {x.get("khu_vuc") for x in bucket if x.get("khu_vuc")}
                candidate_area = node.get("khu_vuc")

                if not areas:
                    bucket.append(node)
                    placed = True
                    break
                if len(areas) == 1:
                    existing = next(iter(areas))
                    if self._khu_vuc_adjacent(existing, candidate_area, profile):
                        bucket.append(node)
                        placed = True
                        break
                if candidate_area in areas:
                    bucket.append(node)
                    placed = True
                    break
                idx = (idx + 1) % day_count

            if not placed:
                day_buckets[idx].append(node)
            idx = (idx + 1) % day_count

        clustered_route = []
        for bucket in day_buckets:
            clustered_route.extend(bucket)

        return clustered_route

    def _reorder_route_by_daily_plan(
        self,
        route_nodes: List[Dict[str, Any]],
        daily_cluster_plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reorder route nodes so they match the daily cluster plan order.

        Ensures all Day 1 points (inland) come before Day 2 points (coastal),
        preventing cross-region interleaving within a single day.
        """
        if not daily_cluster_plan or len(route_nodes) <= 1:
            return route_nodes

        # Build name → node map
        name_to_node = {}
        for node in route_nodes:
            name = node.get("name")
            if name:
                name_to_node[name] = node

        # Reorder by daily plan
        reordered = []
        seen_names = set()
        for day_info in daily_cluster_plan:
            for point_name in day_info.get("point_names", []):
                node = name_to_node.get(point_name)
                if node and point_name not in seen_names:
                    reordered.append(node)
                    seen_names.add(point_name)

        # Append any nodes not in the plan
        for node in route_nodes:
            name = node.get("name")
            if name not in seen_names:
                reordered.append(node)

        if len(reordered) == len(route_nodes):
            return reordered
        return route_nodes  # fallback if mismatch

    def _apply_two_opt_within_days(
        self,
        route_nodes: List[Dict[str, Any]],
        daily_cluster_plan: List[Dict[str, Any]],
        matrix: Dict[tuple, float],
    ) -> List[Dict[str, Any]]:
        if not daily_cluster_plan or not route_nodes:
            return route_nodes

        node_id_to_day = {}
        node_idx = 0
        for day_info in daily_cluster_plan:
            day_points = day_info.get("point_names", [])
            for _ in range(len(day_points)):
                if node_idx < len(route_nodes):
                    node_id_to_day[node_idx] = day_info.get("day", 1)
                    node_idx += 1

        best = list(route_nodes)
        improved = True
        iteration = 0

        while improved and iteration < 2:
            improved = False
            iteration += 1
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    day_i = node_id_to_day.get(i)
                    day_j = node_id_to_day.get(j)
                    if day_i != day_j:
                        continue

                    candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                    candidate_dist = self._graph_path_total_km(candidate, matrix)
                    best_dist = self._graph_path_total_km(best, matrix)

                    if candidate_dist + 0.05 < best_dist:
                        best = candidate
                        improved = True

        return best

    def _build_constraint_warning(self) -> Optional[Dict[str, Any]]:
        """Build a constraint warning dict if the last candidate prioritization found zero matches.

        Returns None if no warning is needed, or a dict with keys:
            - ``coastal``, ``sunset``, ``island`` (bool): which constraints were active
            - ``message``: Vietnamese warning message for the frontend to display
        """
        count = getattr(self, "_last_constraint_match_count", None)
        required = getattr(self, "_last_constraint_required", {})
        if count is None or count > 0:
            return None  # no warning needed
        if not any(required.values()):
            return None

        missing_parts = []
        if required.get("coastal"):
            missing_parts.append("biển/đảo")
        if required.get("sunset"):
            missing_parts.append("điểm ngắm hoàng hôn")
        if required.get("island"):
            missing_parts.append("đảo")

        return {
            "coastal": required.get("coastal", False),
            "sunset": required.get("sunset", False),
            "island": required.get("island", False),
            "message": (
                f"⚠️ Lưu ý: Chưa tìm đủ địa điểm {', '.join(missing_parts)} "
                f"trong dữ liệu để đảm bảo yêu cầu của bạn. "
                f"Lịch trình dưới đây là gợi ý tốt nhất hiện có — "
                f"bạn có thể hỏi thêm về các điểm ven biển cụ thể như Kỳ Co, Eo Gió, Cù Lao Xanh."
            ),
        }

    def optimize_tour_route_nodes(
        self,
        route_nodes: List[Dict[str, Any]],
        optimize_distance: bool,
        max_hop_km: float,
        days: int,
        detected_location: str,
        region_focus: str = "",
    ) -> Dict[str, Any]:
        min_points_required = max(5, days * 2)
        min_points_per_day = 2 if days >= 2 else 1

        # Fetch lodging suggestions for multi-day tours (separate from route)
        lodging_suggestions = []
        if days >= 2:
            lodging_suggestions = self.fetch_lodging_suggestions(
                detected_location,
                route_nodes,
                region_focus=region_focus,
            )

        if len(route_nodes) < 2:
            hop_distances = self._compute_hop_distances_km(route_nodes)
            daily_cluster_plan = self._build_daily_cluster_plan(route_nodes, days, detected_location, region_focus)
            metrics = self._build_route_optimizer_metrics(
                optimization_requested=optimize_distance,
                optimization_applied=False,
                input_points=len(route_nodes),
                points_before_guardrail=len(route_nodes),
                points_after_guardrail=len(route_nodes),
                max_hop_km=max_hop_km,
                hop_distances_km=hop_distances,
                dropped_route_points=[],
                total_distance_km=self._path_total_km(route_nodes),
                route_engine="neo4j_point_distance",
            )
            return {
                "nodes": route_nodes,
                "nearby_mode": False,
                "max_hop_km": max_hop_km,
                "dropped_route_points": [],
                "hop_distances_km": hop_distances,
                "optimization_applied": False,
                "graph_ordering_applied": False,
                "route_engine": "neo4j_point_distance",
                "daily_cluster_plan": daily_cluster_plan,
                "route_optimizer_metrics": metrics,
                "lodging_suggestions": lodging_suggestions,
            }

        if days >= 3:
            optimized = self._build_region_clustered_route(route_nodes, days, detected_location, region_focus)
            pre_guardrail_nodes = list(optimized)
            daily_cluster_plan = self._build_daily_cluster_plan(optimized, days, detected_location, region_focus)

            matrix, _ = self._build_neo4j_distance_matrix(route_nodes)
            optimized = self._apply_two_opt_within_days(optimized, daily_cluster_plan, matrix)
            points_before_guardrail = len(optimized)

            optimized, dropped_points, hop_distances = self._apply_max_hop_guardrail(
                optimized,
                max_hop_km,
                allow_drops=bool(optimize_distance),
            )

            daily_cluster_plan = self._build_daily_cluster_plan(optimized, days, detected_location, region_focus)

            nearby_mode = False
            route_engine = "region_clustering"
            optimization_applied = len(dropped_points) > 0

            route_viable, fallback_reason = self._evaluate_route_viability(
                route_nodes=optimized,
                daily_cluster_plan=daily_cluster_plan,
                min_points_required=min_points_required,
                min_points_per_day=min_points_per_day,
            )
            if not route_viable:
                self.logger.warning(
                    "route_optimizer_fallback_raw reason=%s days=%s optimized_points=%s",
                    fallback_reason,
                    days,
                    len(optimized),
                )
                (
                    optimized,
                    dropped_points,
                    hop_distances,
                    daily_cluster_plan,
                    route_engine,
                    optimization_applied,
                ) = self._build_raw_route_fallback(
                    fallback_nodes=pre_guardrail_nodes,
                    days=days,
                    detected_location=detected_location,
                    route_engine=route_engine,
                    engine_suffix="_fallback_raw",
                    region_focus=region_focus,
                )

            metrics = self._build_route_optimizer_metrics(
                optimization_requested=False,
                optimization_applied=optimization_applied,
                input_points=len(route_nodes),
                points_before_guardrail=points_before_guardrail,
                points_after_guardrail=len(optimized),
                max_hop_km=max_hop_km,
                hop_distances_km=hop_distances,
                dropped_route_points=dropped_points,
                total_distance_km=self._path_total_km(optimized),
                route_engine=route_engine,
            )

            self.logger.info(
                "route_optimizer_3plus_days=%s days, engine=region_clustering, nodes=%d",
                days,
                len(optimized),
            )

            return {
                "nodes": optimized,
                "nearby_mode": nearby_mode,
                "max_hop_km": max_hop_km,
                "dropped_route_points": dropped_points,
                "hop_distances_km": hop_distances,
                "optimization_applied": optimization_applied,
                "graph_ordering_applied": False,
                "route_engine": route_engine,
                "daily_cluster_plan": daily_cluster_plan,
                "route_optimizer_metrics": metrics,
                "lodging_suggestions": lodging_suggestions,
            }

        matrix, route_engine = self._build_neo4j_distance_matrix(route_nodes)
        optimized = self._graph_nearest_neighbor_route(route_nodes, matrix)

        daily_cluster_plan = self._build_daily_cluster_plan(optimized, days, detected_location, region_focus)
        # Reorder route so points are grouped by day/region (inland first, then coastal)
        optimized = self._reorder_route_by_daily_plan(optimized, daily_cluster_plan)
        optimized = self._apply_two_opt_within_days(optimized, daily_cluster_plan, matrix)
        pre_guardrail_nodes = list(optimized)

        points_before_guardrail = len(optimized)

        nearby_mode = bool(optimize_distance)
        optimized, dropped_points, hop_distances = self._apply_max_hop_guardrail(
            optimized,
            max_hop_km,
            allow_drops=bool(optimize_distance),
        )

        daily_cluster_plan = self._build_daily_cluster_plan(optimized, days, detected_location, region_focus)
        route_viable, fallback_reason = self._evaluate_route_viability(
            route_nodes=optimized,
            daily_cluster_plan=daily_cluster_plan,
            min_points_required=min_points_required,
            min_points_per_day=min_points_per_day,
        )
        if not route_viable:
            self.logger.warning(
                "route_optimizer_fallback_raw reason=%s days=%s optimized_points=%s",
                fallback_reason,
                days,
                len(optimized),
            )
            (
                optimized,
                dropped_points,
                hop_distances,
                daily_cluster_plan,
                route_engine,
                optimize_distance,
            ) = self._build_raw_route_fallback(
                fallback_nodes=pre_guardrail_nodes,
                days=days,
                detected_location=detected_location,
                route_engine=route_engine,
                engine_suffix="_fallback_raw",
                region_focus=region_focus,
            )

        metrics = self._build_route_optimizer_metrics(
            optimization_requested=optimize_distance,
            optimization_applied=optimize_distance,
            input_points=len(route_nodes),
            points_before_guardrail=points_before_guardrail,
            points_after_guardrail=len(optimized),
            max_hop_km=max_hop_km,
            hop_distances_km=hop_distances,
            dropped_route_points=dropped_points,
            total_distance_km=self._path_total_km(optimized),
            route_engine=route_engine,
        )

        self.logger.info("route_optimizer_metrics=%s", json.dumps(metrics, ensure_ascii=False))

        constraint_warning = self._build_constraint_warning()
        return {
            "nodes": optimized,
            "nearby_mode": nearby_mode,
            "max_hop_km": max_hop_km,
            "dropped_route_points": dropped_points,
            "hop_distances_km": hop_distances,
            "optimization_applied": optimize_distance,
            "graph_ordering_applied": True,
            "route_engine": route_engine,
            "daily_cluster_plan": daily_cluster_plan,
            "route_optimizer_metrics": metrics,
            "lodging_suggestions": lodging_suggestions,
            "constraint_warning": constraint_warning,
        }
