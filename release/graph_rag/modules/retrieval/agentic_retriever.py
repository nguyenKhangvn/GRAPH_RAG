import logging

logger = logging.getLogger(__name__)

import json
import re
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
import time
import hashlib
from collections import Counter
from typing import Any, Dict, List, Set

from graph_rag.core.intents import IntentType
from graph_rag.core.state import NodeItem


# ── Schema context (static, no LLM needed) ──────────────────────────────

_SCHEMA_CONTEXT = """
ĐỒI THỊ TRI THỨC DU LỊCH (Neo4j):

Node Types:
- TouristAttraction (201): điểm tham quan, danh lam thắng cảnh, di tích
- Restaurant (132): nhà hàng, quán ăn, cafe
- Dish (151): món ăn đặc sản
- Accommodation (360): khách sạn, homestay, resort, nhà nghỉ
- Tour (36): tour trọn gói (bao gồm nhiều TouristAttraction)
- Event (24): lễ hội, sự kiện

Relationships:
- Restaurant -[NEAR]-> TouristAttraction (849 rels): nhà hàng GẦN điểm tham quan
- Accommodation -[NEAR]-> TouristAttraction (2856 rels): lưu trú GẦN điểm tham quan
- Restaurant -[HAS]-> Dish (169 rels): nhà hàng PHỤC VỤ món ăn
- Tour -[INCLUDES]-> TouristAttraction (206 rels): tour BAO GỒM các điểm
- TouristAttraction -[BELONGS_TO]-> Category (192 rels): phân loại điểm tham quan
- Event -[HELD_AT]-> TouristAttraction (27 rels): sự kiện TẠI điểm tham quan

Chiến lược retrieval:
1. TouristAttraction là TRUNG TÂM — mọi node khác liên quan qua nó
2. Tìm Restaurant/Accommodation qua NEAR (gần điểm tham quan đã có)
3. Tìm Dish qua HAS (món ăn của nhà hàng đã có)
4. Mở rộng Tour qua INCLUDES (lấy thêm TouristAttraction)
"""

# ── Quota rules per trip duration ────────────────────────────────────────

_QUOTA_RULES = {
    "tourist_attractions": {"min_per_day": 2, "absolute_min": 3},
    "restaurant":          {"min_per_day": 1, "absolute_min": 2},
    "accommodation":       {"min_per_day": 0.5, "absolute_min": 1},  # per night
    "dish":                {"min_per_day": 1, "absolute_min": 2},
}


class AgenticRetriever:
    """
    Schema-aware iterative retrieval.

    Workflow:
      1) Build schema + current state context
      2) Single LLM call → retrieval strategies (gap + decompose combined)
      3) Execute strategies: vector_search + graph_traverse
      4) Stop when sufficient or no meaningful gains
    """

    def __init__(
        self,
        base_retriever,
        llm_service,
        max_iterations: int = 2,
        max_sub_queries: int = 3,
    ):
        self.base_retriever = base_retriever
        self.llm_service = llm_service
        self.max_iterations = max(1, int(max_iterations))
        self.max_sub_queries = max(1, int(max_sub_queries))
        self._last_llm_ts = 0.0
        self._llm_min_interval_sec = 1.2
        self._llm_cache: Dict[str, Dict[str, Any]] = {}
        self._llm_cache_ttl_sec = 240

    # ── Cache helpers ────────────────────────────────────────────────────

    def _cache_get(self, key: str):
        payload = self._llm_cache.get(key)
        if not payload:
            return None
        if (time.time() - payload["ts"]) > self._llm_cache_ttl_sec:
            self._llm_cache.pop(key, None)
            return None
        return payload["value"]

    def _cache_set(self, key: str, value):
        self._llm_cache[key] = {"value": value, "ts": time.time()}

    def _cache_key(self, prefix: str, content: str) -> str:
        norm = re.sub(r"\s+", " ", (content or "").strip().lower())
        return hashlib.sha1(f"{prefix}:{norm[:1200]}".encode("utf-8")).hexdigest()

    def _allow_llm(self) -> bool:
        now = time.time()
        if now - self._last_llm_ts < self._llm_min_interval_sec:
            return False
        self._last_llm_ts = now
        return True

    # ── Main loop ────────────────────────────────────────────────────────

    def retrieve_iterative(
        self,
        query: str,
        metadata: Dict[str, Any],
        initial_results: List[NodeItem] = None,
    ) -> List[NodeItem]:
        all_results: List[NodeItem] = self.base_retriever._deduplicate_seeds(initial_results or [])

        if not query:
            return all_results

        trip_days = self._infer_trip_days(metadata, query)
        initial_dist = self._type_distribution(all_results)
        logger.info("       [Agentic] START: max_iter=%s, seeds=%d, dist=%s, trip_days=%s",
                     self.max_iterations, len(all_results), dict(initial_dist), trip_days)

        for iteration in range(self.max_iterations):
            # ── Quick rule check: skip if all quotas met ─────────────────
            gaps = self._compute_gaps(all_results, trip_days, metadata)
            if not gaps:
                logger.info("       [Agentic] All quotas met at iteration %s → stop.", iteration + 1)
                break

            # ── Skip iteration 2+ if no improvement possible ─────────────
            if iteration > 0 and not self._has_graph_traversal_potential(all_results, gaps):
                logger.info("       [Agentic] No graph traversal potential for remaining gaps → stop.")
                break

            # ── Schema-aware strategy (1 LLM call) ───────────────────────
            strategies = self._schema_aware_strategy(query, metadata, all_results, gaps, trip_days)
            if not strategies:
                logger.info("       [Agentic] No strategies generated → stop.")
                break

            # ── Execute strategies ────────────────────────────────────────
            new_results = self._execute_strategies(strategies, metadata, all_results)
            if not new_results:
                logger.info("       [Agentic] No new seeds from strategies → stop.")
                break

            before = len(all_results)
            all_results = self.base_retriever._deduplicate_seeds(all_results + new_results)
            gained = len(all_results) - before
            logger.info("       [Agentic] Iteration %s: +%d unique seeds (strategies=%d)",
                         iteration + 1, gained, len(strategies))

            if gained <= 0:
                break

        final_dist = self._type_distribution(all_results)
        logger.info("       [Agentic] DONE: seeds=%d, dist=%s", len(all_results), dict(final_dist))
        return all_results

    # ── Gap computation (rule-based, no LLM) ─────────────────────────────

    def _compute_gaps(
        self, results: List[NodeItem], trip_days: int, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Compute gaps with quantity awareness. Returns list of {type, current, needed}."""
        dist = self._type_distribution(results)
        intent = str(metadata.get("intent") or "")
        semantic_cat = (metadata.get("query_state") or {}).get("semantic_category") or ""
        is_natural_only = semantic_cat == "natural_landmark"
        gaps = []

        if intent in (IntentType.TOUR_PLAN, IntentType.DISCOVERY):
            # TouristAttractions: need 2 per day, min 3
            quota = _QUOTA_RULES["tourist_attractions"]
            needed = max(quota["absolute_min"], int(quota["min_per_day"] * trip_days))
            current = dist.get("TouristAttraction", 0)
            if current < needed:
                gaps.append({"type": "tourist_attractions", "current": current, "needed": needed})

            # Restaurants/Dish: need 1 per day, min 2
            if not is_natural_only:
                quota = _QUOTA_RULES["restaurant"]
                needed = max(quota["absolute_min"], int(quota["min_per_day"] * trip_days))
                current_food = dist.get("Restaurant", 0) + dist.get("Dish", 0)
                if current_food < needed:
                    gaps.append({"type": "food_places", "current": current_food, "needed": needed})

                # Accommodation: need 1 per night
                quota = _QUOTA_RULES["accommodation"]
                needed = max(quota["absolute_min"], int(quota["min_per_day"] * max(1, trip_days - 1)))
                current_acc = dist.get("Accommodation", 0)
                if current_acc < needed:
                    gaps.append({"type": "accommodations", "current": current_acc, "needed": needed})

        return gaps

    def _has_graph_traversal_potential(self, results: List[NodeItem], gaps: List[Dict]) -> bool:
        """Check if graph traversal can fill remaining gaps."""
        has_attractions = any(n.metadata.get("labels", [None])[0] == "TouristAttraction" for n in results)
        has_restaurants = any(n.metadata.get("labels", [None])[0] == "Restaurant" for n in results)

        for gap in gaps:
            if gap["type"] == "food_places" and has_attractions:
                return True  # Can traverse NEAR from attractions
            if gap["type"] == "accommodations" and has_attractions:
                return True  # Can traverse NEAR from attractions
            if gap["type"] == "tourist_attractions":
                return True  # Can always do vector search
        return False

    # ── Schema-aware strategy (single LLM call) ──────────────────────────

    def _schema_aware_strategy(
        self,
        query: str,
        metadata: Dict[str, Any],
        current_results: List[NodeItem],
        gaps: List[Dict],
        trip_days: int,
    ) -> List[Dict[str, Any]]:
        """Single LLM call: gap analysis + decomposition + graph strategy."""
        location = metadata.get("current_location") or metadata.get("detected_location") or ""
        dist = self._type_distribution(current_results)

        # Build current state summary
        state_lines = []
        for gap in gaps:
            state_lines.append(f"- {gap['type']}: có {gap['current']}, cần ≥{gap['needed']}")
        state_summary = "\n".join(state_lines) if state_lines else "- Đã đủ tất cả"

        # Existing node snapshot (for graph traversal context)
        existing_snapshot = []
        for item in current_results[:15]:
            label = (item.metadata.get("labels") or [item.metadata.get("type") or "?"])[0]
            existing_snapshot.append(f"- {item.content} ({label})")

        system_prompt = (
            "Bạn là chuyên gia retrieval cho đồ thị tri thức du lịch. "
            "Phân tích gaps và đề xuất chiến lược retrieval.\n\n"
            "Trả về JSON: {\"strategies\": [{\n"
            "  \"gap\": \"tên gap\",\n"
            "  \"action\": \"vector_search\" hoặc \"graph_traverse\",\n"
            "  \"target_labels\": [\"NodeLabel\"],\n"
            "  \"search_text\": \"truy vấn tìm kiếm\" (null nếu graph_traverse),\n"
            "  \"graph_hint\": \"NEAR|HAS|INCLUDES\" (null nếu vector_search),\n"
            "  \"priority\": 1-3\n"
            "}]}\n\n"
            "QUY TẮC:\n"
            "1. Ưu tiên TouristAttraction TRƯỚC (destination chính)\n"
            "2. Restaurant/Dish: Ưu tiên graph_traverse NEAR từ TouristAttraction đã có\n"
            "3. Accommodation: Ưu tiên graph_traverse NEAR từ TouristAttraction đã có\n"
            "4. search_text PHẢI bằng tiếng Việt, ngắn gọn\n"
            "5. Tối đa 3 strategies"
        )

        user_prompt = (
            f"{_SCHEMA_CONTEXT}\n\n"
            f"TRẠNG THÁI HIỆN TẠI ({len(current_results)} nodes, {trip_days} ngày):\n{state_summary}\n\n"
            f"DANH SÁCH NODE HIỆN CÓ:\n{chr(10).join(existing_snapshot) if existing_snapshot else '- (trống)'}\n\n"
            f"CÂU HỎI: {query}\n"
            f"INTENT: {metadata.get('intent')}\n"
            f"LOCATION: {location}\n"
        )

        # Cache
        cache_key = self._cache_key("strategy", user_prompt)
        cached = self._cache_get(cache_key)
        if cached is not None:
            logger.info("       [Agentic.Strategy] CACHE HIT → %s", cached)
            return cached

        if not self._allow_llm():
            logger.info("       [Agentic.Strategy] RATE LIMITED → fallback to rule-based")
            return self._rule_based_strategies(gaps, location)

        logger.info("       [Agentic.Strategy] CALLING LLM (single call for gap+strategy)...")
        try:
            data = self.llm_service.generate_json(system_prompt, user_prompt)
            logger.info("       [Agentic.Strategy] LLM RESPONSE: %s", data)

            strategies = []
            if isinstance(data, dict) and isinstance(data.get("strategies"), list):
                strategies = data["strategies"]
            elif isinstance(data, list):
                strategies = data

            # Validate and filter
            valid = []
            for s in strategies:
                if not isinstance(s, dict):
                    continue
                action = s.get("action", "vector_search")
                target_labels = s.get("target_labels", [])
                search_text = s.get("search_text")
                graph_hint = s.get("graph_hint")

                if action == "graph_traverse" and graph_hint:
                    valid.append({
                        "gap": s.get("gap", ""),
                        "action": "graph_traverse",
                        "target_labels": target_labels,
                        "graph_hint": graph_hint,
                        "priority": s.get("priority", 2),
                    })
                elif action == "vector_search" and search_text:
                    valid.append({
                        "gap": s.get("gap", ""),
                        "action": "vector_search",
                        "target_labels": target_labels,
                        "search_text": search_text,
                        "priority": s.get("priority", 1),
                    })

            # Sort by priority, limit to max_sub_queries
            valid.sort(key=lambda x: x.get("priority", 99))
            valid = valid[:self.max_sub_queries]

            if valid:
                self._cache_set(cache_key, valid)
                logger.info("       [Agentic.Strategy] PARSED %d strategies: %s",
                            len(valid), [s.get("gap") for s in valid])
                return valid

            logger.warning("       [Agentic.Strategy] No valid strategies from LLM, using rule-based")
            return self._rule_based_strategies(gaps, location)

        except (ValueError, RuntimeError, OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("       [Agentic.Strategy] LLM FAILED: %s → fallback to rule-based", e)
            return self._rule_based_strategies(gaps, location)

    def _rule_based_strategies(self, gaps: List[Dict], location: str) -> List[Dict]:
        """Deterministic fallback when LLM is unavailable."""
        strategies = []
        for gap in gaps:
            gtype = gap["type"]
            if gtype == "tourist_attractions":
                strategies.append({
                    "gap": gtype, "action": "vector_search",
                    "target_labels": ["TouristAttraction"],
                    "search_text": f"điểm tham quan nổi tiếng {location}",
                    "priority": 1,
                })
            elif gtype == "food_places":
                strategies.append({
                    "gap": gtype, "action": "graph_traverse",
                    "target_labels": ["Restaurant", "Dish"],
                    "graph_hint": "NEAR",
                    "priority": 2,
                })
            elif gtype == "accommodations":
                strategies.append({
                    "gap": gtype, "action": "graph_traverse",
                    "target_labels": ["Accommodation"],
                    "graph_hint": "NEAR",
                    "priority": 3,
                })
        return strategies[:self.max_sub_queries]

    # ── Strategy execution ───────────────────────────────────────────────

    def _execute_strategies(
        self,
        strategies: List[Dict],
        metadata: Dict[str, Any],
        current_results: List[NodeItem],
    ) -> List[NodeItem]:
        """Execute retrieval strategies and return new nodes."""
        existing_ids: Set[str] = {str(x.id) for x in current_results}
        new_items: List[NodeItem] = []

        for i, strategy in enumerate(strategies):
            action = strategy.get("action", "vector_search")
            target_labels = strategy.get("target_labels", [])
            gap = strategy.get("gap", "")

            if action == "graph_traverse":
                hint = strategy.get("graph_hint", "NEAR")
                seeds = self._execute_graph_traverse(
                    current_results, hint, target_labels, metadata, existing_ids
                )
            else:
                search_text = strategy.get("search_text", "")
                seeds = self._execute_vector_search(
                    search_text, target_labels, metadata, existing_ids
                )

            new_count = sum(1 for s in seeds if str(s.id) not in existing_ids)
            logger.info("       [Agentic.Execute] strategy[%d] gap=%s action=%s → %d seeds, %d NEW",
                        i, gap, action, len(seeds), new_count)

            for seed in seeds:
                sid = str(seed.id)
                if sid not in existing_ids:
                    existing_ids.add(sid)
                    new_items.append(seed)

        logger.info("       [Agentic.Execute] Total new items: %d", len(new_items))
        return new_items

    def _execute_vector_search(
        self,
        search_text: str,
        target_labels: List[str],
        metadata: Dict[str, Any],
        existing_ids: Set[str],
    ) -> List[NodeItem]:
        """Vector/text search with targeted labels."""
        sub_meta = dict(metadata or {})
        sub_meta["search_keywords"] = [search_text]
        sub_meta["agentic_subquery"] = True
        sub_meta.pop("target_entity", None)
        sub_meta.pop("entities", None)
        sub_meta.pop("resolved_entities", None)

        seeds = self.base_retriever.find_seeds(search_text, metadata=sub_meta)
        return seeds

    # ── Graph traversal ──────────────────────────────────────────────────

    def _execute_graph_traverse(
        self,
        current_results: List[NodeItem],
        hint: str,
        target_labels: List[str],
        metadata: Dict[str, Any],
        existing_ids: Set[str],
    ) -> List[NodeItem]:
        """Graph traversal from existing nodes using relationship hints."""
        hint_upper = hint.upper() if hint else ""

        if "HAS" in hint_upper:
            return self._traverse_has(current_results, target_labels, existing_ids)
        elif "INCLUDES" in hint_upper:
            return self._traverse_includes(current_results, target_labels, existing_ids)
        else:  # NEAR (default)
            return self._traverse_near(current_results, target_labels, existing_ids)

    def _traverse_near(
        self,
        anchors: List[NodeItem],
        target_labels: List[str],
        existing_ids: Set[str],
    ) -> List[NodeItem]:
        """Find nodes NEAR existing TouristAttraction anchors via graph."""
        anchor_ids = [
            str(n.id) for n in anchors
            if "TouristAttraction" in (n.metadata.get("labels") or [])
        ]
        if not anchor_ids:
            logger.info("       [Agentic.Traverse] NEAR: no TouristAttraction anchors")
            return []

        # Build label filter for Cypher
        label_filter = " OR ".join(f"t:{lbl}" for lbl in target_labels) if target_labels else "true"

        cypher = f"""
            MATCH (a:TouristAttraction)-[:NEAR]-(t)
            WHERE a.id IN $anchor_ids AND ({label_filter})
              AND NOT t.id IN $existing_ids
            RETURN DISTINCT t.id AS id, t.name AS name, labels(t)[0] AS label,
                   t.address AS address, t.description AS description
            LIMIT 10
        """
        return self._run_traversal_cypher(cypher, {"anchor_ids": anchor_ids, "existing_ids": list(existing_ids)},
                                          "NEAR")

    def _traverse_has(
        self,
        restaurants: List[NodeItem],
        target_labels: List[str],
        existing_ids: Set[str],
    ) -> List[NodeItem]:
        """Find Dish via Restaurant-[HAS]->Dish."""
        rest_ids = [
            str(n.id) for n in restaurants
            if "Restaurant" in (n.metadata.get("labels") or [])
        ]
        if not rest_ids:
            logger.info("       [Agentic.Traverse] HAS: no Restaurant anchors")
            return []

        cypher = """
            MATCH (r:Restaurant)-[:HAS]->(d)
            WHERE r.id IN $rest_ids AND (d:Dish OR d:Specialty)
              AND NOT d.id IN $existing_ids
            RETURN DISTINCT d.id AS id, d.name AS name, labels(d)[0] AS label,
                   d.description AS description, d.category AS category
            LIMIT 10
        """
        return self._run_traversal_cypher(cypher, {"rest_ids": rest_ids, "existing_ids": list(existing_ids)},
                                          "HAS")

    def _traverse_includes(
        self,
        tours: List[NodeItem],
        target_labels: List[str],
        existing_ids: Set[str],
    ) -> List[NodeItem]:
        """Find TouristAttraction via Tour-[INCLUDES]->TouristAttraction."""
        tour_ids = [
            str(n.id) for n in tours
            if "Tour" in (n.metadata.get("labels") or [])
        ]
        if not tour_ids:
            logger.info("       [Agentic.Traverse] INCLUDES: no Tour anchors")
            return []

        cypher = """
            MATCH (t:Tour)-[:INCLUDES]->(a:TouristAttraction)
            WHERE t.id IN $tour_ids
              AND NOT a.id IN $existing_ids
            RETURN DISTINCT a.id AS id, a.name AS name, labels(a)[0] AS label,
                   a.address AS address, a.description AS description
            LIMIT 10
        """
        return self._run_traversal_cypher(cypher, {"tour_ids": tour_ids, "existing_ids": list(existing_ids)},
                                          "INCLUDES")

    def _run_traversal_cypher(self, cypher: str, params: Dict, label: str) -> List[NodeItem]:
        """Execute a traversal Cypher query and return NodeItems."""
        driver = self.base_retriever.driver
        items: List[NodeItem] = []
        try:
            with driver.session() as session:
                result = session.run(cypher, **params)
                for record in result:
                    node_id = record["id"]
                    name = record.get("name", "")
                    node_label = record.get("label", "Unknown")
                    description = record.get("description", "") or ""
                    address = record.get("address", "") or ""
                    category = record.get("category", "") or ""

                    item = NodeItem(
                        id=str(node_id),
                        content=name,
                        score=1.0,
                        source_type=f"graph_traverse_{label.lower()}",
                        metadata={
                            "labels": [node_label],
                            "type": node_label,
                            "address": address,
                            "description": description,
                            "category": category,
                        },
                    )
                    items.append(item)
            logger.info("       [Agentic.Traverse] %s → %d nodes", label, len(items))
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.warning("       [Agentic.Traverse] %s FAILED: %s", label, e)
        return items

    # ── Helpers ──────────────────────────────────────────────────────────

    def _type_distribution(self, results: List[NodeItem]) -> Counter:
        """Count nodes by type label."""
        dist: Counter = Counter()
        for item in results or []:
            labels = item.metadata.get("labels") or []
            if labels:
                dist[str(labels[0])] += 1
            elif item.metadata.get("type"):
                dist[str(item.metadata["type"])] += 1
        return dist

    def _infer_trip_days(self, metadata: Dict[str, Any], query: str) -> int:
        """Infer trip duration from metadata or query text."""
        # From metadata constraints
        constraints = (metadata or {}).get("constraints") or {}
        trip_days = constraints.get("trip_days")
        if trip_days is not None:
            try:
                return max(1, int(trip_days))
            except (ValueError, TypeError):
                pass

        # From QueryPlan/QueryState
        qs = (metadata or {}).get("query_state")
        if qs and hasattr(qs, "duration_days") and qs.duration_days > 0:
            return qs.duration_days

        # From query text regex
        q_lower = query.lower()
        m = re.search(r"(\d+)\s*(ngày|ngay|day)", q_lower)
        if m:
            return max(1, int(m.group(1)))
        m = re.search(r"(\d+)\s*(đêm|dem|night)", q_lower)
        if m:
            return max(1, int(m.group(1)) + 1)

        return 2  # default 2 days
