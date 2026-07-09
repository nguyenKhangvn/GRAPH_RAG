"""Tour plan & proximity retrieval — semantic quotas, multi-pass collection, geo proximity search."""

import logging
from typing import List, Dict, Optional
from graph_rag.core.state import NodeItem
from graph_rag.config import TOP_K
from graph_rag.utils.text import normalize_text
from graph_rag.core import keywords, thresholds
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

logger = logging.getLogger(__name__)


class TourPlanRetriever:
    """Strategy class for tour plan and proximity-based retrieval.

    Handles semantic quota allocation for multi-day itineraries, related seed
    search via graph relationships, category member search, and proximity
    anchor search (NEAR edge + geospatial fallback).
    """

    def __init__(self, driver):
        self.driver = driver

    # ── Semantic quotas for TOUR_PLAN ──────────────────────────────────

    def apply_tour_plan_semantic_quotas(
        self,
        seeds: List[NodeItem],
        *,
        metadata: Dict,
        user_query: str,
        top_k: int,
        exact_seeds: List[NodeItem],
        trip_days: int,
    ) -> List[NodeItem]:
        """Apply semantic quotas to ensure balanced multi-day itinerary.

        Ensures minimum attractions, food, and accommodation per trip day.
        """
        if not seeds:
            return []

        target_k = max(6, int(top_k or TOP_K))
        min_food = max(2, 2 * trip_days)
        min_attraction = 2 if trip_days == 1 else min(6, 2 * trip_days)
        target_attraction = 3 if trip_days == 1 else min(8, 3 * trip_days)
        max_accommodation = max(3, trip_days + 1)

        label_hints = (metadata or {}).get("label_hints") or []
        v3_data = (metadata or {}).get("v3_intent_data") or {}
        if not label_hints:
            label_hints = v3_data.get("label_hints") or []
        min_accommodation = 0
        min_attraction_from_hints = 0
        if "Accommodation" in label_hints:
            min_accommodation = max(3, trip_days * 2)
        if min_accommodation == 0:
            intents = (metadata or {}).get("intents") or []
            intent_str = " ".join(str(i).upper() for i in intents)
            question_type = str((metadata or {}).get("question_type") or "").strip().lower()
            is_tour_plan = (
                "TOUR_PLAN" in intent_str
                or "TRAVEL_PLAN" in intent_str
                or question_type == "tour-plan"
            )
            if is_tour_plan:
                min_accommodation = max(1, trip_days)
                logger.info("       [Quota] TOUR_PLAN auto-set min_accommodation=%d", min_accommodation)
        if "TouristAttraction" in label_hints:
            min_attraction_from_hints = max(3, trip_days * 2)
        min_attraction = max(min_attraction, min_attraction_from_hints)

        food_labels = {"Restaurant", "Dish", "Specialty"}
        attraction_labels = {"TouristAttraction", "Tour"}
        accommodation_labels = {"Accommodation"}

        exact_ids = {str(s.id) for s in (exact_seeds or []) if getattr(s, "id", None) is not None}
        unique = self._deduplicate_seeds(seeds)

        attractions: List[NodeItem] = []
        foods: List[NodeItem] = []
        accommodations: List[NodeItem] = []
        others: List[NodeItem] = []

        for seed in unique:
            label = self._seed_primary_label(seed)
            if label in attraction_labels:
                attractions.append(seed)
            elif label in food_labels:
                foods.append(seed)
            elif label in accommodation_labels:
                accommodations.append(seed)
            else:
                others.append(seed)

        selected: List[NodeItem] = []
        selected_ids = set()

        def push(seed: NodeItem) -> bool:
            sid = str(seed.id)
            if sid in selected_ids:
                return False
            selected.append(seed)
            selected_ids.add(sid)
            return True

        for seed in unique:
            if str(seed.id) in exact_ids:
                push(seed)

        food_count = sum(1 for s in selected if self._seed_primary_label(s) in food_labels)
        attraction_count = sum(1 for s in selected if self._seed_primary_label(s) in attraction_labels)
        accommodation_count = sum(1 for s in selected if self._seed_primary_label(s) in accommodation_labels)

        for seed in attractions:
            if attraction_count >= target_attraction:
                break
            if push(seed):
                attraction_count += 1

        for seed in foods:
            if food_count >= min_food:
                break
            if push(seed):
                food_count += 1

        for seed in attractions:
            if attraction_count >= min_attraction:
                break
            if push(seed):
                attraction_count += 1

        if min_accommodation > 0:
            for seed in accommodations:
                if accommodation_count >= max(min_accommodation, max_accommodation):
                    break
                if push(seed):
                    accommodation_count += 1

        for seed in accommodations:
            if accommodation_count >= max_accommodation:
                break
            if push(seed):
                accommodation_count += 1

        for pool in (accommodations, attractions, foods, others):
            for seed in pool:
                if len(selected) >= target_k:
                    break
                if self._seed_primary_label(seed) in accommodation_labels and accommodation_count >= max_accommodation:
                    continue
                if push(seed) and self._seed_primary_label(seed) in accommodation_labels:
                    accommodation_count += 1
            if len(selected) >= target_k:
                break

        logger.info(
            "       Semantic Quotas Applied: "
            f"days={trip_days}, attractions={attraction_count}, food={food_count}, "
            f"accommodation={accommodation_count}, total={len(selected)}"
        )
        return selected[:target_k]

    # ── Related seed search (graph relations) ──────────────────────────

    def query_plan_relation_constrained_seeds(
        self, *,
        user_query: str,
        metadata: Dict,
        anchor_seeds: List[NodeItem],
        top_k: int,
        region_filter_params: tuple,
        region_address_aliases_fn,
    ) -> List[NodeItem]:
        """Recover answer-set seeds through graph relations before vector fallback."""
        plan_mode = str((metadata or {}).get("retrieval_plan_mode") or "").strip()
        q_norm = normalize_text(user_query)
        requested_categories = self._requested_category_names(q_norm)
        results: List[NodeItem] = []

        if plan_mode == "lodging_near_anchor":
            proximity_anchors = [
                seed for seed in (anchor_seeds or [])
                if self._seed_primary_label(seed) in {"TouristAttraction", "Event", "Tour"}
                and not self._looks_like_category(seed.content)
            ]
            if proximity_anchors:
                results.extend(
                    self._related_seed_search(
                        anchor_ids=[seed.id for seed in proximity_anchors],
                        relation_types=["NEAR"],
                        target_labels=["Accommodation"],
                        top_k=top_k,
                        region_filter_params=region_filter_params,
                        region_address_aliases_fn=region_address_aliases_fn,
                    )
                )
            if requested_categories:
                results.extend(
                    self._category_member_seed_search(
                        requested_categories, top_k=max(3, top_k // 2),
                        metadata=metadata,
                        region_filter_params=region_filter_params,
                        region_address_aliases_fn=region_address_aliases_fn,
                    )
                )

        elif plan_mode == "tour_plan":
            anchor_ids = [seed.id for seed in (anchor_seeds or []) if getattr(seed, "id", None)]
            if anchor_ids:
                related = self._related_seed_search(
                    anchor_ids=anchor_ids,
                    relation_types=["NEAR", "LOCATED_IN", "BELONGS_TO"],
                    target_labels=["TouristAttraction", "Accommodation", "Tour", "Restaurant", "Dish"],
                    top_k=max(top_k, 10),
                    region_filter_params=region_filter_params,
                    region_address_aliases_fn=region_address_aliases_fn,
                )
                accommodation_cap = 2
                accommodation_count = 0
                for seed in related:
                    if self._seed_primary_label(seed) == "Accommodation":
                        accommodation_count += 1
                        if accommodation_count > accommodation_cap:
                            continue
                    results.append(seed)
            if requested_categories:
                results.extend(
                    self._category_member_seed_search(
                        requested_categories, top_k=max(3, top_k // 2),
                        metadata=metadata,
                        region_filter_params=region_filter_params,
                        region_address_aliases_fn=region_address_aliases_fn,
                    )
                )

        elif plan_mode == "comparison":
            anchor_ids = [seed.id for seed in (anchor_seeds or []) if getattr(seed, "id", None)]
            if anchor_ids:
                relation_types = ["NEAR", "LOCATED_IN", "BELONGS_TO", "HAS", "OFFERS", "INCLUDES"]
                target_labels = ["TouristAttraction", "Accommodation", "Restaurant", "Dish", "Tour"]
                if requested_categories:
                    relation_types = ["NEAR", "BELONGS_TO"]
                    target_labels = ["TouristAttraction"]
                results.extend(
                    self._related_seed_search(
                        anchor_ids=anchor_ids,
                        relation_types=relation_types,
                        target_labels=target_labels,
                        top_k=max(top_k, 10),
                        categories=requested_categories,
                        region_filter_params=region_filter_params,
                        region_address_aliases_fn=region_address_aliases_fn,
                    )
                )

        elif plan_mode == "dish_to_restaurant":
            dish_anchors = [
                seed for seed in (anchor_seeds or [])
                if self._seed_primary_label(seed) == "Dish"
            ]
            if not dish_anchors:
                dish_anchors = anchor_seeds or []
            anchor_ids = [seed.id for seed in dish_anchors if getattr(seed, "id", None)]
            if anchor_ids:
                results.extend(
                    self._related_seed_search(
                        anchor_ids=anchor_ids,
                        relation_types=["HAS"],
                        target_labels=["Restaurant"],
                        top_k=top_k,
                        region_filter_params=region_filter_params,
                        region_address_aliases_fn=region_address_aliases_fn,
                    )
                )

        elif plan_mode == "global_discovery" or (
            requested_categories and any(token in q_norm for token in ["cac diem du lich", "diem du lich tai", "liet ke"])
        ):
            results.extend(
                self._category_member_seed_search(
                    requested_categories, top_k=80,
                    metadata=metadata,
                    region_filter_params=region_filter_params,
                    region_address_aliases_fn=region_address_aliases_fn,
                )
            )

        return self._deduplicate_seeds(results)

    def _related_seed_search(
        self, *,
        anchor_ids: List[str],
        relation_types: List[str],
        target_labels: List[str],
        top_k: int,
        categories: List[str] = None,
        metadata: Dict = None,
        region_filter_params: tuple = (None, None),
        region_address_aliases_fn=None,
    ) -> List[NodeItem]:
        if not anchor_ids or not relation_types or not target_labels:
            return []
        cypher = """
        MATCH (anchor)-[r]-(target)
        WHERE anchor.id IN $anchor_ids
          AND type(r) IN $relation_types
          AND any(lbl IN labels(target) WHERE lbl IN $target_labels)
        OPTIONAL MATCH (target)-[:LOCATED_IN]->(loc:Location)
        OPTIONAL MATCH (target)-[:BELONGS_TO]->(cat)
        WITH target, loc, cat
        WHERE ($region_group IS NULL
               OR coalesce(loc.region_group, '') = $region_group
               OR any(alias IN $region_aliases WHERE toLower(coalesce(target.address, '')) CONTAINS alias))
          AND ($legacy_province IS NULL
               OR coalesce(loc.legacy_province, loc.current_province, '') = $legacy_province
               OR any(alias IN $region_aliases WHERE toLower(coalesce(target.address, '')) CONTAINS alias))
          AND (loc IS NULL
               OR loc.admin_status IS NULL
               OR loc.admin_status <> 'merged'
               OR $legacy_province IS NOT NULL)
          AND (
                $category_names IS NULL
                OR coalesce(cat.name, '') IN $category_names
                OR any(cat_norm IN $category_norms
                       WHERE toLower(coalesce(cat.name, '')) CONTAINS cat_norm
                          OR cat_norm CONTAINS toLower(coalesce(cat.name, '')))
              )
        RETURN DISTINCT target.id AS id, target.name AS name, labels(target) AS labels,
               target.description AS description, target.address AS address, target.topic AS topic,
               CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.latitude IS NOT NULL THEN target.location.latitude ELSE toFloat(target.lat) END AS lat,
               CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.longitude IS NOT NULL THEN target.location.longitude ELSE toFloat(target.lng) END AS lng,
               loc.name AS commune_name,
               loc.region_group AS region_group,
               coalesce(loc.legacy_province, loc.current_province, '') AS legacy_province,
               coalesce(loc.admin_level, '') AS admin_level,
               coalesce(loc.admin_status, '') AS admin_status,
               1.0 AS score
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                region_group, legacy_province = region_filter_params
                region_group_cypher = region_group if not isinstance(region_group, list) else None
                if region_group_cypher == "tay_nguyen":
                    region_group_cypher = "gia_lai_core"
                elif region_group_cypher == "duyen_hai_nam_trung_bo":
                    region_group_cypher = "binh_dinh_legacy"
                region_aliases = region_address_aliases_fn(legacy_province, region_group) if region_address_aliases_fn else []
                records = session.run(
                    cypher,
                    anchor_ids=anchor_ids,
                    relation_types=relation_types,
                    target_labels=target_labels,
                    category_names=[str(c or "").strip() for c in (categories or []) if str(c or "").strip()] or None,
                    category_norms=[normalize_text(c) for c in (categories or []) if c] or [],
                    region_group=region_group_cypher,
                    legacy_province=legacy_province,
                    region_aliases=region_aliases,
                    limit=int(top_k or TOP_K),
                )
                return [
                    self._record_to_nodeitem(record, "query_plan_relation", record.get("score", 1.0))
                    for record in records
                ]
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.error("       QueryFrame relation seed search failed: %s", exc)
            return []

    def _category_member_seed_search(self, categories: List[str], top_k: int,
                                     metadata: Dict = None,
                                     region_filter_params: tuple = (None, None),
                                     region_address_aliases_fn=None) -> List[NodeItem]:
        if not categories:
            return []
        category_norms = [normalize_text(c) for c in categories if c]
        category_names = [str(c or "").strip().lower() for c in categories if str(c or "").strip()]
        cypher = """
        MATCH (target:TouristAttraction)-[:BELONGS_TO]-(cat)
        OPTIONAL MATCH (target)-[:LOCATED_IN]->(loc:Location)
        WITH target, cat, loc, toLower(coalesce(cat.name, cat.id, '')) AS cat_name
        WHERE ($region_group IS NULL
               OR coalesce(loc.region_group, '') = $region_group
               OR any(alias IN $region_aliases WHERE toLower(coalesce(target.address, '')) CONTAINS alias))
          AND ($legacy_province IS NULL
               OR coalesce(loc.legacy_province, loc.current_province, '') = $legacy_province
               OR any(alias IN $region_aliases WHERE toLower(coalesce(target.address, '')) CONTAINS alias))
          AND (loc IS NULL OR loc.admin_status IS NULL OR loc.admin_status <> 'merged' OR $legacy_province IS NOT NULL)
          AND (any(cat_name_raw IN $category_names
                  WHERE cat_name CONTAINS cat_name_raw OR cat_name_raw CONTAINS cat_name)
           OR any(cat_norm IN $category_norms
                  WHERE cat_name CONTAINS cat_norm OR cat_norm CONTAINS cat_name))
        RETURN DISTINCT target.id AS id, target.name AS name, labels(target) AS labels,
               target.description AS description, target.address AS address, target.topic AS topic,
               CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.latitude IS NOT NULL THEN target.location.latitude ELSE toFloat(target.lat) END AS lat,
               CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.longitude IS NOT NULL THEN target.location.longitude ELSE toFloat(target.lng) END AS lng,
               loc.name AS commune_name, loc.region_group AS region_group, coalesce(loc.legacy_province, loc.current_province, '') AS legacy_province,
               coalesce(loc.admin_level, '') AS admin_level, coalesce(loc.admin_status, '') AS admin_status,
               CASE
                 WHEN coalesce(loc.legacy_province, loc.current_province, '') = $legacy_province THEN 1.4
                 WHEN any(alias IN $region_aliases WHERE toLower(coalesce(target.address, '')) CONTAINS alias) THEN 1.2
                 ELSE 1.0
               END AS score
        ORDER BY score DESC, name ASC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                region_group, legacy_province = region_filter_params
                region_group_cypher = region_group if not isinstance(region_group, list) else None
                if region_group_cypher == "tay_nguyen":
                    region_group_cypher = "gia_lai_core"
                elif region_group_cypher == "duyen_hai_nam_trung_bo":
                    region_group_cypher = "binh_dinh_legacy"
                region_aliases = region_address_aliases_fn(legacy_province, region_group) if region_address_aliases_fn else []
                records = session.run(
                    cypher,
                    category_norms=category_norms,
                    category_names=category_names,
                    region_group=region_group_cypher,
                    legacy_province=legacy_province,
                    region_aliases=region_aliases,
                    limit=int(top_k or TOP_K),
                )
                return [
                    self._record_to_nodeitem(record, "query_plan_category", record.get("score", 1.0))
                    for record in records
                ]
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.error("       QueryFrame category seed search failed: %s", exc)
            return []

    # ── Proximity anchor search ────────────────────────────────────────

    def proximity_anchor_search(
        self, anchor_nodes: List[NodeItem], target_labels: List[str],
        top_k: int = 5, max_distance_m: int = thresholds.PROXIMITY_SEARCH_MAX_M,
    ) -> List[NodeItem]:
        """Find target_label nodes near anchor_nodes via NEAR edge or geospatial fallback."""
        results: List[NodeItem] = []
        seen_ids: set = set()
        anchor_ids = [n.id for n in anchor_nodes]
        geo_anchors = [
            (n.metadata.get("lat"), n.metadata.get("lng"))
            for n in anchor_nodes
            if n.metadata.get("lat") is not None and n.metadata.get("lng") is not None
        ]

        cypher_near = """
        MATCH (target)-[:NEAR]->(anchor)
        WHERE anchor.id IN $anchor_ids
          AND any(lbl IN labels(target) WHERE lbl IN $target_labels)
        RETURN target.id AS id, target.name AS name, labels(target) AS labels,
               target.description AS description, target.address AS address, target.topic AS topic,
               CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.latitude IS NOT NULL THEN target.location.latitude ELSE toFloat(target.lat) END AS lat,
               CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.longitude IS NOT NULL THEN target.location.longitude ELSE toFloat(target.lng) END AS lng,
               1.0 AS score
        UNION
        MATCH (anchor)-[:NEAR]->(target)
        WHERE anchor.id IN $anchor_ids
          AND any(lbl IN labels(target) WHERE lbl IN $target_labels)
        RETURN target.id AS id, target.name AS name, labels(target) AS labels,
               target.description AS description, target.address AS address, target.topic AS topic,
               CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.latitude IS NOT NULL THEN target.location.latitude ELSE toFloat(target.lat) END AS lat,
               CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.longitude IS NOT NULL THEN target.location.longitude ELSE toFloat(target.lng) END AS lng,
               0.9 AS score
        """
        try:
            with self.driver.session() as session:
                records = session.run(cypher_near, anchor_ids=anchor_ids, target_labels=target_labels)
                for record in records:
                    rid = record["id"]
                    if rid in seen_ids:
                        continue
                    seen_ids.add(rid)
                    lbls = list(record["labels"]) if record["labels"] else []
                    results.append(NodeItem(
                        id=rid, content=record["name"], score=record["score"],
                        source_type="proximity_near_edge",
                        metadata={
                            "name": record["name"], "type": lbls[0] if lbls else "Unknown",
                            "labels": lbls, "address": record.get("address") or "",
                            "description": record.get("description") or "",
                            "topic": record.get("topic") or "",
                            "lat": record.get("lat"), "lng": record.get("lng"),
                        },
                    ))
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.error("       Proximity NEAR-edge search failed: %s", exc)

        if len(results) < top_k and geo_anchors:
            lat, lng = geo_anchors[0]
            cypher_geo = """
            MATCH (target)
            WHERE any(lbl IN labels(target) WHERE lbl IN $target_labels)
              AND target.location IS NOT NULL
              AND toLower(toString(target.location)) STARTS WITH 'point'
            WITH target,
                 point.distance(target.location, point({latitude: $lat, longitude: $lng})) AS dist
            WHERE dist <= $max_dist
            ORDER BY dist ASC
            RETURN target.id AS id, target.name AS name, labels(target) AS labels,
                   target.description AS description, target.address AS address, target.topic AS topic,
                   CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.latitude IS NOT NULL THEN target.location.latitude ELSE toFloat(target.lat) END AS lat,
                   CASE WHEN target.location IS NOT NULL AND toLower(toString(target.location)) STARTS WITH 'point' AND target.location.longitude IS NOT NULL THEN target.location.longitude ELSE toFloat(target.lng) END AS lng,
                   dist
            LIMIT $top_k
            """
            try:
                with self.driver.session() as session:
                    records = session.run(
                        cypher_geo, target_labels=target_labels, lat=lat, lng=lng,
                        max_dist=float(max_distance_m), top_k=top_k,
                    )
                    for record in records:
                        rid = record["id"]
                        if rid in seen_ids:
                            continue
                        seen_ids.add(rid)
                        lbls = list(record["labels"]) if record["labels"] else []
                        dist_m = record.get("dist") or max_distance_m
                        results.append(NodeItem(
                            id=rid, content=record["name"],
                            score=max(0.0, 1.0 - dist_m / max_distance_m),
                            source_type="proximity_geo",
                            metadata={
                                "name": record["name"], "type": lbls[0] if lbls else "Unknown",
                                "labels": lbls, "address": record.get("address") or "",
                                "description": record.get("description") or "",
                                "topic": record.get("topic") or "",
                                "lat": record.get("lat"), "lng": record.get("lng"),
                            },
                        ))
            except (Neo4jClientError, ServiceUnavailable) as exc:
                logger.error("       Proximity geo search failed: %s", exc)

        return results[:top_k]

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _seed_primary_label(seed: NodeItem) -> str:
        labels = seed.metadata.get("labels") or []
        if labels:
            return str(labels[0])
        return str(seed.metadata.get("type") or "")

    @staticmethod
    def _requested_category_names(query_norm: str) -> List[str]:
        categories: List[str] = []
        for category, aliases in keywords.CATEGORY_ALIASES:
            if any(alias in query_norm for alias in aliases):
                categories.append(category)
        return categories

    @staticmethod
    def _looks_like_category(value: str) -> bool:
        norm = normalize_text(value)
        return any(marker in norm for marker in keywords.CATEGORY_MARKERS)

    @staticmethod
    def _record_to_nodeitem(record, source_type: str, default_score: float = 1.0) -> NodeItem:
        labels = list(record["labels"]) if record["labels"] else []
        return NodeItem(
            id=record["id"],
            content=record["name"],
            score=record.get("score", default_score),
            source_type=source_type,
            metadata={
                "name": record["name"],
                "type": labels[0] if labels else "Unknown",
                "labels": labels,
                "address": record.get("address", ""),
                "description": record.get("description", ""),
                "topic": record.get("topic", ""),
                "category": record.get("category", ""),
                "star_rating": record.get("star_rating", 0),
                "price_range": record.get("price_range", ""),
                "lat": record.get("lat"),
                "lng": record.get("lng"),
                "commune_name": record.get("commune_name", ""),
                "region_group": record.get("region_group", ""),
                "legacy_province": record.get("legacy_province", ""),
                "legacy_district": record.get("legacy_district", ""),
                "admin_level": record.get("admin_level", ""),
                "admin_status": record.get("admin_status", ""),
            },
        )

    @staticmethod
    def _deduplicate_seeds(seeds: List[NodeItem]) -> List[NodeItem]:
        from graph_rag.utils.text import normalize_text
        seen_ids: set = set()
        seen_names: Dict[str, NodeItem] = {}
        unique_list: List[NodeItem] = []

        for s in seeds:
            if s.id in seen_ids:
                continue
            seen_ids.add(s.id)
            name = str(s.metadata.get("name") or s.content or "").strip()
            labels = s.metadata.get("labels") or []
            node_type = labels[0] if labels else ""
            if name and node_type in {"Dish", "Specialty", "TouristAttraction", "Restaurant"}:
                name_norm = normalize_text(name, strip_punct=True)
                if name_norm in seen_names:
                    existing = seen_names[name_norm]
                    if s.score > existing.score:
                        for i, node in enumerate(unique_list):
                            if node.id == existing.id:
                                unique_list[i] = s
                                break
                        seen_names[name_norm] = s
                    continue
                seen_names[name_norm] = s
            unique_list.append(s)
        return unique_list
