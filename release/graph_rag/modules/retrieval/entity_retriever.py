"""Entity lookup & grounding — exact, fuzzy, alias, normalized, and semantic search."""

import re
import logging
from typing import List, Dict, Optional, Set
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
from graph_rag.core.state import NodeItem
from graph_rag.core import keywords, thresholds
from graph_rag.utils.text import normalize_text
from graph_rag.core.keywords import CATEGORY_PHRASES

logger = logging.getLogger(__name__)


class EntityRetriever:
    """Strategy class for entity name lookup and grounding against Neo4j.

    Handles the cascade: exact → contains/alias → normalized → semantic → fuzzy.
    Used by SeedRetriever for entity-first retrieval and entity grounding.
    """

    _ALLOWED_FETCH_LABELS = frozenset({
        "TouristAttraction", "Location", "Dish", "Accommodation",
        "Event", "Festival", "Restaurant", "Tour", "TravelInfo",
    })

    def __init__(self, driver, fuzzy_matcher=None):
        self.driver = driver
        self._fuzzy_matcher = fuzzy_matcher

    def set_fuzzy_matcher(self, fuzzy_matcher):
        self._fuzzy_matcher = fuzzy_matcher

    # ── Public interface ───────────────────────────────────────────────

    def exact_match_search(self, entity_name: str) -> List[NodeItem]:
        """Level 1: Exact name match via Cypher."""
        cypher = """
        MATCH (n)
        WHERE toLower(n.name) = toLower($name)
        RETURN coalesce(n.id, elementId(n)) AS id, n.name AS name, labels(n) AS labels,
               n.description AS description, n.address AS address, n.topic AS topic,
               CASE WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' AND n.location.latitude IS NOT NULL THEN n.location.latitude ELSE toFloat(n.lat) END AS lat,
               CASE WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' AND n.location.longitude IS NOT NULL THEN n.location.longitude ELSE toFloat(n.lng) END AS lng
        LIMIT 3
        """
        try:
            with self.driver.session() as session:
                records = session.run(cypher, name=entity_name)
                results = []
                for record in records:
                    item = NodeItem(
                        id=record["id"],
                        content=record["name"],
                        score=1.0,
                        source_type="exact_match",
                        metadata={
                            "name": record["name"],
                            "type": record["labels"][0] if record["labels"] else "Unknown",
                            "labels": list(record["labels"]) if record["labels"] else [],
                            "address": record.get("address", ""),
                            "description": record.get("description", ""),
                            "topic": record.get("topic", ""),
                            "lat": record.get("lat"),
                            "lng": record.get("lng"),
                        }
                    )
                    results.append(item)
                return results
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.error("Error in exact match: %s", e)
            return []

    def contains_alias_search(self, entity_name: str, allowed_labels: List[str] = None) -> List[NodeItem]:
        """Level 2: Contains/alias match — name contains alias or vice versa."""
        raw = str(entity_name or "").strip()
        if not raw:
            return []

        aliases = [raw]
        for lookup_name in self.entity_lookup_candidates(raw)[1:]:
            if lookup_name and lookup_name.lower() != raw.lower() and lookup_name not in aliases:
                aliases.append(lookup_name)
        stripped = re.sub(
            r"(?i)^(?:quán|quan|nhà\s+hàng|nha\s+hang|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi)\s+",
            "",
            raw,
        ).strip()
        if stripped and stripped.lower() != raw.lower():
            aliases.append(stripped)

        cypher = f"""
        MATCH (n)
        WITH n, trim(coalesce(n.name, '')) AS node_name
        WHERE node_name <> ''
          AND ($allowed_labels IS NULL OR any(lbl IN labels(n) WHERE lbl IN $allowed_labels))
          AND any(alias IN $aliases
              WHERE toLower(node_name) CONTAINS toLower(alias)
                 OR toLower(alias) CONTAINS toLower(node_name))
        RETURN coalesce(n.id, elementId(n)) AS id, n.name AS name, labels(n) AS labels,
               n.description AS description, n.address AS address, n.topic AS topic,
               CASE WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' AND n.location.latitude IS NOT NULL THEN n.location.latitude ELSE toFloat(n.lat) END AS lat,
               CASE WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' AND n.location.longitude IS NOT NULL THEN n.location.longitude ELSE toFloat(n.lng) END AS lng,
               CASE
                 WHEN any(alias IN $aliases WHERE toLower(node_name) = toLower(alias)) THEN 1.0
                 ELSE {thresholds.CONTAINS_MATCH_SCORE}
               END AS score
        ORDER BY score DESC, size(node_name) ASC
        LIMIT 5
        """
        try:
            with self.driver.session() as session:
                records = session.run(cypher, aliases=aliases, allowed_labels=allowed_labels)
                results = []
                for record in records:
                    labels = list(record["labels"]) if record["labels"] else []
                    results.append(NodeItem(
                        id=record["id"],
                        content=record["name"],
                        score=record["score"],
                        source_type="contains_alias_match",
                        metadata={
                            "name": record["name"],
                            "type": labels[0] if labels else "Unknown",
                            "labels": labels,
                            "address": record.get("address", ""),
                            "description": record.get("description", ""),
                            "topic": record.get("topic", ""),
                            "lat": record.get("lat"),
                            "lng": record.get("lng"),
                        },
                    ))
                return results
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.error("Error in contains alias search: %s", e)
            return []

    def normalized_name_search(self, entity_name: str, allowed_labels: List[str] = None) -> List[NodeItem]:
        """Level 3: Unicode-safe normalized name search with token overlap scoring."""
        raw = str(entity_name or "").strip()
        if not raw:
            return []

        aliases = list(self.entity_lookup_candidates(raw))
        try:
            if self._fuzzy_matcher:
                synonym_variants = self._fuzzy_matcher.expand_entity_name(raw)
                aliases.extend(synonym_variants)
        except (ValueError, TypeError, RuntimeError) as exc:
            logger.debug("Fuzzy matcher expansion failed for '%s': %s", raw, exc)

        alias_norms = [
            normalize_text(alias)
            for alias in aliases
            if normalize_text(alias)
        ]
        if not alias_norms:
            return []

        alias_token_sets = []
        for norm in alias_norms:
            tokens = set(norm.split())
            if tokens:
                alias_token_sets.append(tokens)

        cypher = """
        MATCH (n)
        WITH n, trim(coalesce(n.name, '')) AS node_name
        WHERE node_name <> ''
          AND ($allowed_labels IS NULL OR any(lbl IN labels(n) WHERE lbl IN $allowed_labels))
        RETURN coalesce(n.id, elementId(n)) AS id, n.name AS name, labels(n) AS labels,
               n.description AS description, n.address AS address, n.topic AS topic,
               coalesce(n.star_rating, 0) AS star_rating,
               coalesce(n.price_range, '') AS price_range,
               CASE WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' AND n.location.latitude IS NOT NULL THEN n.location.latitude ELSE toFloat(n.lat) END AS lat,
               CASE WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' AND n.location.longitude IS NOT NULL THEN n.location.longitude ELSE toFloat(n.lng) END AS lng
        LIMIT 500
        """
        try:
            with self.driver.session() as session:
                records = session.run(cypher, allowed_labels=allowed_labels)
                matches = []
                for record in records:
                    node_norm = normalize_text(str(record["name"] or ""))
                    if not node_norm:
                        continue
                    score = 0.0
                    for alias_norm in alias_norms:
                        if node_norm == alias_norm:
                            score = max(score, 1.0)
                            break
                        elif alias_norm in node_norm or node_norm in alias_norm:
                            score = max(score, thresholds.SUBSTRING_MATCH_SCORE)

                    if score == 0.0 and alias_token_sets:
                        node_tokens = set(node_norm.split())
                        if node_tokens:
                            for token_set in alias_token_sets:
                                overlap = len(token_set & node_tokens)
                                min_len = min(len(token_set), len(node_tokens))
                                if min_len > 0 and overlap / min_len >= 0.5:
                                    overlap_score = 0.6 * (overlap / min_len)
                                    score = max(score, overlap_score)

                    if score > 0:
                        matches.append(self._record_to_nodeitem(record, "normalized_name_match", score))
                matches.sort(key=lambda item: (-item.score, len(str(item.content or ""))))
                return matches[:5]
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.error("Error in normalized name search: %s", e)
            return []

    def semantic_entity_search(self, entity_name: str, embedder, allowed_labels: List[str] = None) -> List[NodeItem]:
        """Level 4: Vector similarity search for entity grounding."""
        if not entity_name or not embedder:
            return []

        _VECTOR_INDEX_LABEL_MAP = {
            "tourist_vec_idx": "TouristAttraction",
            "restaurant_vec_idx": "Restaurant",
            "accommodation_vec_idx": "Accommodation",
            "tour_vec_idx": "Tour",
            "event_vec_idx": "Event",
            "dish_vec_idx": "Dish",
        }

        embedding = embedder.embed_query(entity_name)
        if not embedding:
            return []

        target_indexes = {
            idx: label
            for idx, label in _VECTOR_INDEX_LABEL_MAP.items()
            if allowed_labels is None or label in allowed_labels
        }

        results = []
        with self.driver.session() as session:
            for index_name in target_indexes:
                cypher = """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                WHERE score >= $threshold
                RETURN node.id AS id, node.name AS name, labels(node) AS labels,
                       node.description AS description, node.address AS address, node.topic AS topic,
                       coalesce(node.star_rating, 0) AS star_rating,
                       coalesce(node.price_range, '') AS price_range,
                       CASE WHEN node.location IS NOT NULL AND toLower(toString(node.location)) STARTS WITH 'point' AND node.location.latitude IS NOT NULL THEN node.location.latitude ELSE toFloat(node.lat) END AS lat,
                       CASE WHEN node.location IS NOT NULL AND toLower(toString(node.location)) STARTS WITH 'point' AND node.location.longitude IS NOT NULL THEN node.location.longitude ELSE toFloat(node.lng) END AS lng,
                       score
                ORDER BY score DESC
                LIMIT $top_k
                """
                try:
                    records = session.run(
                        cypher,
                        index_name=index_name,
                        top_k=thresholds.SEMANTIC_GROUNDING_TOP_K,
                        embedding=embedding,
                        threshold=thresholds.SEMANTIC_GROUNDING_THRESHOLD,
                    )
                    for record in records:
                        results.append(self._record_to_nodeitem(
                            record, "semantic_match", record["score"]
                        ))
                except (Neo4jClientError, ServiceUnavailable):
                    continue

        results.sort(key=lambda x: -x.score)
        return self._deduplicate_seeds(results)

    def fuzzy_name_search(self, entity_name: str,
                          allowed_labels: List[str] = None,
                          location_filter: str = None,
                          region_group=None, legacy_province=None) -> List[NodeItem]:
        """Level 5: Fulltext fuzzy name search."""
        from graph_rag.modules.retrieval.fulltext_search import search_fulltext_loop
        seeds_dict = search_fulltext_loop(
            self.driver,
            entity_name,
            k=3,
            filter_labels=allowed_labels,
            filter_city=location_filter,
            region_group=region_group,
            legacy_province=legacy_province
        )
        return self._convert_dict_to_nodeitems(seeds_dict, source_override="fuzzy_match")

    def fetch_nodes_by_label(self, label: str, limit: int = 15, region_group=None, legacy_province=None) -> List[NodeItem]:
        """Fetch nodes directly from graph by label (fallback when vector search returns too few)."""
        if label not in self._ALLOWED_FETCH_LABELS:
            logger.error("       [fetch_nodes_by_label] Rejected invalid label: '%s'", label)
            return []
        region_group_cypher = region_group if not isinstance(region_group, list) else None
        if label in ("Dish", "Specialty"):
            region_filter = "AND n.region_group = $region_group" if region_group_cypher else ""
        else:
            region_filter = ("AND EXISTS { MATCH (n)-[:LOCATED_IN]->(l:Location) "
                             "WHERE l.region_group = $region_group AND "
                             "(l.admin_status IS NULL OR l.admin_status <> 'merged') }"
                             ) if region_group_cypher else ""
        cypher = f"""
            MATCH (n:{label})
            WHERE n.name IS NOT NULL
            {region_filter}
            RETURN coalesce(n.id, elementId(n)) AS id, n.name AS name, n.description AS description,
                   n.address AS address, labels(n) AS labels, n.topic AS topic,
                   CASE WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' AND n.location.latitude IS NOT NULL THEN n.location.latitude ELSE toFloat(n.lat) END AS lat,
                   CASE WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' AND n.location.longitude IS NOT NULL THEN n.location.longitude ELSE toFloat(n.lng) END AS lng
            ORDER BY n.name
            LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher, limit=limit,
                                     **({"region_group": region_group_cypher} if region_group_cypher else {}))
                nodes = []
                for record in result:
                    nodes.append(NodeItem(
                        id=record["id"],
                        content=record["name"] or "",
                        score=0.5,
                        source_type="graph_scan",
                        metadata={
                            "name": record["name"],
                            "description": record.get("description"),
                            "address": record.get("address"),
                            "topic": record.get("topic", ""),
                            "labels": record["labels"] or [label],
                            "type": label,
                            "lat": record.get("lat"),
                            "lng": record.get("lng"),
                        },
                    ))
                return nodes
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.error("       [fetch_nodes_by_label] Error: %s", e)
            return []

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def entity_lookup_candidates(entity_name: str) -> List[str]:
        """Conservative name variants for entity-first lookup."""
        raw = str(entity_name or "").strip(" ,.;:!?")
        if not raw:
            return []
        candidates = [raw]
        for pattern in [
            r"(?i)^(?:nhà\s+hàng|nha\s+hang)\s+",
            r"(?i)^(?:quán|quan)\s+",
            r"(?i)^(?:khách\s+sạn|khach\s+san)\s+",
            r"(?i)^(?:nhà\s+nghỉ|nha\s+nghi)\s+",
            r"(?i)^(?:khu\s+du\s+lịch|khu\s+du\s+lich)\s+",
            r"(?i)^(?:di\s+tích(?:\s+khảo\s+cổ)?|di\s+tich(?:\s+khao\s+co)?)\s+",
            r"(?i)^(?:bảo\s+tàng|bao\s+tang)\s+",
            r"(?i)^(?:suối|suoi|thác|thac|biển|bien|làng\s+nghề|lang\s+nghe)\s+",
            r"(?i)^(?:chùa|chua|đền|den|đình|dinh)\s+",
            r"(?i)^(?:hồ|ho|núi|nui|đảo|dao)\s+",
            r"(?i)^(?:công viên|cong vien|quảng trường|quang truong)\s+",
            r"(?i)^(?:làng|lang)\s+(?:văn hóa|van hoa|nghệ|nghe|du lịch|du lich)\s+",
        ]:
            stripped = re.sub(pattern, "", raw).strip(" ,.;:!?")
            if stripped and stripped.lower() != raw.lower() and stripped not in candidates:
                candidates.append(stripped)

        if " và " in raw or " va " in raw.lower():
            parts = re.split(r"\s+và\s+|\s+va\s+", raw, flags=re.IGNORECASE)
            for part in parts:
                cleaned = part.strip(" ,.;:!?")
                cleaned = re.sub(r"\s+(?:đều|deu|cả|ca)\s*$", "", cleaned, flags=re.IGNORECASE).strip()
                if cleaned and len(cleaned) >= 3 and cleaned not in candidates:
                    candidates.append(cleaned)

        return candidates

    @staticmethod
    def is_category_phrase(text: str) -> bool:
        """Check if text is a category/type phrase rather than a specific entity name."""
        norm = normalize_text(text)
        if not norm:
            return False
        if norm in CATEGORY_PHRASES:
            return True
        for cat in CATEGORY_PHRASES:
            if norm == cat or cat.startswith(norm) or norm.startswith(cat):
                return True
        return False

    @staticmethod
    def rank_by_type_preference(nodes: List[NodeItem], preferred_type: str) -> List[NodeItem]:
        """Rank nodes so that those matching preferred_type come first."""
        preferred = []
        fallback = []
        preferred_norm = preferred_type.lower().strip()
        for node in nodes:
            labels = [str(l).lower().strip() for l in (node.metadata.get("labels") or [])]
            if preferred_norm in labels:
                preferred.append(node)
            else:
                fallback.append(node)
        return preferred + fallback

    @staticmethod
    def filter_nodes_by_type(nodes: List[NodeItem], expected_labels: List[str]) -> List[NodeItem]:
        """Filter nodes to only those whose labels intersect with expected_labels. Fail-open."""
        if not expected_labels:
            return nodes
        expected_set = {lbl.lower().strip() for lbl in expected_labels}
        filtered = []
        for node in nodes:
            node_labels = {str(l).lower().strip() for l in (node.metadata.get("labels") or [])}
            if node_labels & expected_set:
                filtered.append(node)
        if not filtered:
            return nodes
        return filtered

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
    def _convert_dict_to_nodeitems(seeds_dict: List[Dict], source_override: str = None) -> List[NodeItem]:
        node_items = []
        for seed in seeds_dict:
            if source_override:
                source_str = source_override
            else:
                sources = seed.get('found_by', [])
                source_str = ",".join(sources) if isinstance(sources, list) else str(sources)
            node_type = seed.get('type') or 'Unknown'
            item = NodeItem(
                id=seed.get('id'),
                content=seed.get('name'),
                score=seed.get('final_score', seed.get('score', 0.0)),
                source_type=source_str,
                metadata={
                    "name": seed.get('name'),
                    "type": node_type,
                    "labels": [node_type] if node_type and node_type != 'Unknown' else [],
                    "address": seed.get('address', ''),
                    "description": seed.get('description', ''),
                    "topic": seed.get('topic', ''),
                    "lat": seed.get('lat'),
                    "lng": seed.get('lng'),
                    "commune_name": seed.get('commune_name', ''),
                    "region_group": seed.get('region_group', ''),
                    "legacy_province": seed.get('legacy_province', ''),
                    "legacy_district": seed.get('legacy_district', ''),
                }
            )
            node_items.append(item)
        return node_items

    @staticmethod
    def _deduplicate_seeds(seeds: List[NodeItem]) -> List[NodeItem]:
        """Deduplicate seeds by ID and normalized name, keeping higher-score nodes."""
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
