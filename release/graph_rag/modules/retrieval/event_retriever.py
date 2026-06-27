"""Event-specific retrieval — category detection, location search, month/year filtering."""

import re
import logging
from typing import List, Dict, Optional, Set
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
from graph_rag.core.state import NodeItem
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)


class EventRetriever:
    """Strategy class for Event-specific retrieval.

    Handles event category detection (festival vs sports), LOCATED_IN-based
    location fallback, month/year constraints, and event query shortening.
    """

    def __init__(self, driver):
        self.driver = driver

    # ── Event category detection ───────────────────────────────────────

    def detect_event_category(self, query_norm: str) -> str:
        """Detect event sub-category from query keywords.

        Returns:
            'lễ hội' for cultural/folk/historical festivals
            'giải chạy' for marathon/trail running events
            '' for general event queries (no filter)
        """
        sport_terms = ["marathon", "chạy bộ", "giải chạy", "chay bo", "trail", "ultra"]
        if any(term in query_norm for term in sport_terms):
            return "giải chạy"

        cultural_terms = [
            "lễ hội", "le hoi", "festival", "văn hóa", "van hoa",
            "dân gian", "dan gian", "tín ngưỡng", "tin nguong",
            "lịch sử", "lich su", "cồng chiêng", "cong chieng",
            "đâm trâu", "dam trau", "lễ hội văn hóa", "sự kiện văn hóa",
        ]
        if any(term in query_norm for term in cultural_terms):
            return "lễ hội"

        return ""

    def filter_by_event_category(self, seeds: list, category_filter: str) -> list:
        """Post-filter Event seeds by category property.

        Falls back to original seeds if all events are filtered out.
        """
        if not category_filter:
            return seeds

        event_seeds = [s for s in seeds if 'Event' in (s.metadata.get("labels") or [])]
        other_seeds = [s for s in seeds if 'Event' not in (s.metadata.get("labels") or [])]

        if not event_seeds:
            return seeds

        event_ids = [s.id for s in event_seeds]
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (e:Event) WHERE e.id IN $ids "
                    "RETURN e.id AS id, e.category AS category, e.name AS name",
                    ids=event_ids
                )
                cat_map = {row['id']: row['category'] or '' for row in result}
                name_map = {row['id']: row['name'] or '' for row in result}
        except (Neo4jClientError, ServiceUnavailable):
            return seeds

        filtered_events = []
        for seed in event_seeds:
            cat = cat_map.get(seed.id, '')
            if category_filter == "lễ hội":
                if "lễ hội" in cat.lower() or "văn hóa" in cat.lower():
                    filtered_events.append(seed)
                elif not cat:
                    filtered_events.append(seed)
            elif category_filter == "giải chạy":
                if "giải chạy" in cat.lower() or "marathon" in cat.lower():
                    filtered_events.append(seed)
                elif not cat:
                    filtered_events.append(seed)
            else:
                filtered_events.append(seed)

        if not filtered_events:
            logger.warning("         ⚙️ Event category filter '%s' removed all events; reverting to unfiltered", category_filter)
            return seeds

        logger.info("         ⚙️ Event category filter '%s': %d -> %d events",
                     category_filter, len(event_seeds), len(filtered_events))
        return other_seeds + filtered_events

    # ── Event location search ──────────────────────────────────────────

    def search_events_by_location(self, location: str, limit: int = 10,
                                  month_range: set = None, year: int = None,
                                  category_filter: str = None) -> list:
        """Fallback: search Event nodes by HELD_AT -> TouristAttraction -> LOCATED_IN -> Location."""
        fetch_limit = int(limit) * 3 if category_filter else int(limit)
        params: dict = {"loc": location, "limit": fetch_limit}
        month_filter = ""
        if month_range:
            month_list = sorted(month_range)
            params["month_list"] = month_list
            month_filter = "AND e.month IN $month_list"
        year_filter = ""
        if year:
            params["year"] = int(year)
            year_filter = "AND (e.year = $year OR e.year IS NULL)"
        cypher = f"""
        MATCH (e:Event)
        OPTIONAL MATCH (e)-[:HELD_AT]->(t:TouristAttraction)-[:LOCATED_IN]->(l:Location)
        WHERE (toLower(coalesce(l.name, '')) CONTAINS toLower($loc)
           OR toLower(coalesce(e.address, '')) CONTAINS toLower($loc))
           {month_filter}
           {year_filter}
        RETURN DISTINCT e.id AS id, e.name AS name, e.description AS description,
               e.address AS address, e.year AS year, e.month AS month, e.category AS category,
               labels(e) AS labels
        ORDER BY e.name ASC
        LIMIT $limit
        """
        try:
            with self.driver.session() as session:
                rows = session.run(cypher, **params).data()
            results = []
            for row in rows:
                node_id = str(row.get("id") or "").strip()
                name = str(row.get("name") or "").strip()
                if not node_id or not name:
                    continue
                results.append(NodeItem(
                    id=node_id,
                    content=name,
                    metadata={
                        "name": name,
                        "description": str(row.get("description") or "").strip(),
                        "address": str(row.get("address") or "").strip(),
                        "category": str(row.get("category") or "").strip(),
                        "labels": list(row.get("labels") or []),
                        "label": "Event",
                    },
                    score=1.0,
                    source_type="event_location_fallback",
                ))

            if category_filter:
                results = self.filter_by_event_category(results, category_filter)
                results = results[:int(limit)]

            return results
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.error("         ⚙️ EVENT location fallback failed: %s", exc)
            return []

    # ── Month / Year extraction ────────────────────────────────────────

    def extract_month_range(self, query: str) -> set:
        """Extract month numbers from Vietnamese query text.

        Returns a set of month integers (1-12) if time references found, else empty set.
        """
        q = normalize_text(str(query or ""), strip_punct=True).lower()
        months = set()

        for m in re.findall(r'thang\s*(\d{1,2})', q):
            mo = int(m)
            if 1 <= mo <= 12:
                months.add(mo)

        range_match = re.search(
            r'tu\s*thang\s*(\d{1,2})\s*(?:den|[-–]|toi)\s*thang\s*(\d{1,2})', q
        )
        if range_match:
            start, end = int(range_match.group(1)), int(range_match.group(2))
            if 1 <= start <= 12 and 1 <= end <= 12:
                if start <= end:
                    months.update(range(start, end + 1))
                else:
                    months.update(range(start, 13))
                    months.update(range(1, end + 1))

        _seasons = [
            (['mua he'], {6, 7, 8}),
            (['mua xuan'], {1, 2, 3}),
            (['mua thu'], {9, 10, 11}),
            (['mua dong'], {12, 1, 2}),
            (['tet', 'nguyen dan', 'tet am lich'], {1, 2}),
        ]
        for keywords_set, season_months in _seasons:
            if any(re.search(rf'\b{re.escape(kw)}\b', q) for kw in keywords_set):
                months.update(season_months)

        return months

    def extract_year(self, query: str) -> Optional[int]:
        """Extract year from Vietnamese query text."""
        q = normalize_text(str(query or ""), strip_punct=True).lower()
        match = re.search(r'(?:nam\s+)?(\d{4})', q)
        if match:
            year = int(match.group(1))
            if 2020 <= year <= 2030:
                return year
        return None

    def shorten_event_query(self, query: str) -> str:
        """Shorten long EVENT queries by extracting the event name."""
        q = str(query or "").strip()
        if len(q) <= 60:
            return q

        quoted = re.findall(r"['‘’“”]([^'‘’“”]{5,80})['‘’“”]", q)
        if quoted:
            event_name = quoted[0].strip()
            prefix_match = re.match(r"^(lễ hội|festival|sự kiện|event)\s+", q, re.IGNORECASE)
            prefix = prefix_match.group(0) if prefix_match else ""
            return f"{prefix}{event_name}".strip()

        match = re.match(r"(lễ hội|festival|sự kiện)\s+(.{5,60}?)(?:\s+năm|\s+\d{4}|\s+có gì|\s+tôi|\s+cho)", q, re.IGNORECASE)
        if match:
            return f"{match.group(1)} {match.group(2)}".strip()

        for sep in ['?', 'có gì', 'tôi có thể', 'cho tôi']:
            idx = q.lower().find(sep)
            if idx > 30:
                return q[:idx].strip()

        return q[:80].strip()

    def filter_events_by_month(self, seeds: list, month_range: set) -> list:
        """Post-filter Event seeds by their month property."""
        if not month_range:
            return seeds

        event_seeds = [s for s in seeds if 'Event' in (s.metadata.get("labels") or [])]
        non_event_seeds = [s for s in seeds if 'Event' not in (s.metadata.get("labels") or [])]

        if not event_seeds:
            return seeds

        event_ids = [s.id for s in event_seeds]
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (e:Event) WHERE e.id IN $ids "
                    "RETURN e.id AS id, e.month AS month",
                    ids=event_ids
                )
                month_map = {row['id']: row['month'] for row in result}
        except (Neo4jClientError, ServiceUnavailable):
            return seeds

        filtered = []
        for seed in event_seeds:
            mo = month_map.get(seed.id)
            if mo is None:
                filtered.append(seed)
            else:
                try:
                    if int(mo) in month_range:
                        filtered.append(seed)
                except (ValueError, TypeError):
                    filtered.append(seed)

        return non_event_seeds + filtered
