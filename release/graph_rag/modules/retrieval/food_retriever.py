"""Food / dish & restaurant retrieval — category detection, specialty search, food filtering."""

import logging
from typing import List, Dict, Optional
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
from graph_rag.core.state import NodeItem
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)


class FoodRetriever:
    """Strategy class for food/dish and restaurant retrieval.

    Handles food category detection (seafood, BBQ, coffee, specialty),
    Dish node property-based search, and food category post-filtering.
    """

    def __init__(self, driver):
        self.driver = driver

    # ── Food category detection ────────────────────────────────────────

    def detect_food_category(self, query_norm: str) -> str:
        """Detect specific food category from query.

        Returns category name if detected, else empty string.
        """
        seafood_terms = ["hai san", "hải sản", "tom", "tôm", "muc", "mực", "cua",
                         "ốc", "ghe", "ghẹ", "so", "sò", "ca bien", "cá biển"]
        if any(term in query_norm for term in seafood_terms):
            return "hải sản"

        bbq_terms = ["nuong", "nướng", "bbq", "lau", "lẩu", "grill"]
        if any(term in query_norm for term in bbq_terms):
            return "nướng"

        coffee_terms = ["ca phe", "cà phê", "coffee", "cafe"]
        if any(term in query_norm for term in coffee_terms):
            return "cà phê"

        traditional_terms = ["dac san", "đặc sản", "truyen thong", "truyền thống"]
        if any(term in query_norm for term in traditional_terms):
            return "đặc sản"

        return ""

    def filter_by_food_category(self, seeds: list, category: str) -> list:
        """Post-filter seeds by food category.

        - Restaurant seeds: filtered by tags property
        - Dish seeds: filtered by category property
        - Other seeds: pass through unchanged
        """
        if not category:
            return seeds

        restaurant_seeds = [s for s in seeds if 'Restaurant' in (s.metadata.get("labels") or [])]
        dish_seeds = [s for s in seeds if 'Dish' in (s.metadata.get("labels") or [])]
        other_seeds = [s for s in seeds
                       if 'Restaurant' not in (s.metadata.get("labels") or [])
                       and 'Dish' not in (s.metadata.get("labels") or [])]

        # Filter Restaurant seeds by tags
        filtered_restaurants = []
        if restaurant_seeds:
            restaurant_ids = [s.id for s in restaurant_seeds]
            try:
                with self.driver.session() as session:
                    result = session.run(
                        "MATCH (r:Restaurant) WHERE r.id IN $ids "
                        "RETURN r.id AS id, r.tags AS tags, r.name AS name",
                        ids=restaurant_ids
                    )
                    tag_map = {row['id']: row['tags'] or [] for row in result}
                    name_map = {row['id']: row['name'] or '' for row in result}
            except (Neo4jClientError, ServiceUnavailable):
                return seeds

            for seed in restaurant_seeds:
                tags = tag_map.get(seed.id, [])
                name = name_map.get(seed.id, '')
                if category in tags:
                    filtered_restaurants.append(seed)
                elif category in normalize_text(name, strip_punct=True):
                    filtered_restaurants.append(seed)
                elif not tags:
                    filtered_restaurants.append(seed)

        # Filter Dish seeds by category property
        filtered_dishes = []
        if dish_seeds:
            dish_ids = [s.id for s in dish_seeds]
            try:
                with self.driver.session() as session:
                    result = session.run(
                        "MATCH (d:Dish) WHERE d.id IN $ids "
                        "RETURN d.id AS id, d.category AS category, d.name AS name",
                        ids=dish_ids
                    )
                    cat_map = {row['id']: row['category'] or '' for row in result}
                    name_map = {row['id']: row['name'] or '' for row in result}
            except (Neo4jClientError, ServiceUnavailable):
                return seeds

            for seed in dish_seeds:
                dish_cat = cat_map.get(seed.id, '')
                name = name_map.get(seed.id, '')
                if category.lower() in dish_cat.lower():
                    filtered_dishes.append(seed)
                elif category in normalize_text(name, strip_punct=True):
                    filtered_dishes.append(seed)
                elif not dish_cat:
                    filtered_dishes.append(seed)

        return other_seeds + filtered_restaurants + filtered_dishes

    # ── Food specialty search ──────────────────────────────────────────

    def food_specialty_search(self, region_group=None, legacy_province: str = None,
                              top_k: int = 10) -> List[NodeItem]:
        """Search directly Dish nodes by region properties.

        Used for food_specialty queries (đặc sản). Dish uses location/region_group/province
        properties hoặc SPECIALTY_OF -> Location.
        """
        region_group_cypher = region_group if not isinstance(region_group, list) else None
        cypher = """
        MATCH (d:Dish)
        WHERE d.category = 'Đặc sản'
          AND ($region_group IS NULL OR d.region_group = $region_group)
          AND ($legacy_province IS NULL
               OR toLower(coalesce(d.province, '')) CONTAINS toLower($legacy_province)
               OR toLower(coalesce(d.location, '')) CONTAINS toLower($legacy_province))
        RETURN d.id AS id, d.name AS name, d.description AS description,
               coalesce(d.province, '') AS province, coalesce(d.region_group, '') AS region_group,
               1.0 AS score
        ORDER BY d.name
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                records = session.run(
                    cypher,
                    region_group=region_group_cypher,
                    legacy_province=legacy_province,
                    limit=top_k,
                )
                results = []
                for record in records:
                    results.append(NodeItem(
                        id=record["id"],
                        content=record["name"],
                        score=record.get("score", 1.0),
                        source_type="food_specialty_graph",
                        metadata={
                            "name": record["name"],
                            "type": "Dish",
                            "labels": ["Dish"],
                            "description": record.get("description", ""),
                            "region_group": record.get("region_group", ""),
                            "province": record.get("province", ""),
                        },
                    ))
                return results
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.error("       [food_specialty_search] Error: %s", exc)
            return []
