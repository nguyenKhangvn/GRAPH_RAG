import logging
import re as _re

logger = logging.getLogger(__name__)

from typing import List, Tuple, Set, Optional
from neo4j import Driver
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
from graph_rag.core.state import NodeItem
from graph_rag.core.intents import IntentType
from graph_rag.config import cfg as _cfg
from graph_rag.utils.text import normalize_text

# Import các config (Giả định file config đã có)
from graph_rag.config import (
    RELATIONSHIP_MAP,
    TRAVERSAL_WHITELIST,
    INTENT_TRAVERSAL_POLICY,
)

# Load scoring weights from JSON config

ATTRIBUTE_LABELS = {
    "address": "Địa chỉ",
    "phone": "Số điện thoại",
    "price": "Giá vé/Chi phí",
    "ticket_price": "Giá vé",
    "price_range": "Mức giá",
    "opening_hours": "Giờ mở cửa",
    "description": "Thông tin",
    "service_features": "Dịch vụ/tiện ích",
}

# Intents cần suy luận sâu (2-3 hops) để kết nối các thực thể gián tiếp
MULTI_HOP_INTENTS = {
    IntentType.TOUR_PLAN,
    IntentType.DISCOVERY,
    IntentType.FOOD,
    IntentType.EVENT,  # Event -> HELD_AT -> TouristAttraction -> LOCATED_IN -> Location
}

# Trọng số độ tin cậy — loaded from scoring_weights.json
RELATIONSHIP_CONFIDENCE = _cfg.relationship_confidence()


class GraphTraverser:
    def __init__(self, driver: Driver):
        self.driver = driver
        self._existing_property_keys = self._load_property_keys()

    def _load_property_keys(self) -> Set[str]:
        """Load property keys once at startup to avoid querying non-existent fields."""
        try:
            with self.driver.session() as session:
                result = session.run(
                    "CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS keys"
                ).single()
                keys = result.get("keys", []) if result else []
                if not keys:
                    logger.info("       [Traverser] Schema introspection returned 0 keys — filtering disabled.")
                return set(keys)
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.error("       [Traverser] Schema introspection failed (non-fatal, filtering disabled): %s", exc)
            return set()

    def _filter_attributes_by_schema(self, attributes: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Keep only attributes known in DB schema to prevent Neo4j warnings."""
        if not attributes or not self._existing_property_keys:
            return attributes

        filtered = [attr for attr in attributes if attr[0] in self._existing_property_keys]
        dropped = [attr[0] for attr in attributes if attr[0] not in self._existing_property_keys]
        if dropped:
            logger.info("       Schema-aware filter dropped unknown attrs: %s", dropped)
        return filtered

    def _requested_attribute_policy(self, requested_attributes: Optional[List[str]]) -> List[Tuple[str, str]]:
        if not requested_attributes:
            return []
        attrs = []
        for attr in requested_attributes:
            key = str(attr or "").strip()
            if key:
                attrs.append((key, ATTRIBUTE_LABELS.get(key, key)))
        return self._filter_attributes_by_schema(attrs)

    def _requested_relation_policy(self, requested_relations: Optional[List[str]]) -> Optional[List[str]]:
        if not requested_relations:
            return None
        valid = set(RELATIONSHIP_MAP.keys())
        rels = [str(rel or "").strip().upper() for rel in requested_relations]
        rels = [rel for rel in rels if rel in valid]
        return list(dict.fromkeys(rels)) or None

    def _get_location_seed_ids(self, session, seed_ids: List[str]) -> Tuple[List[str], List[str]]:
        """Tách seed_ids thành (location_seed_ids, entity_seed_ids).

        Location nodes là các node có label Location — chúng chỉ có incoming
        LOCATED_IN edges nên traverser mặc định không mở rộng được.
        """
        if not seed_ids:
            return [], []
        try:
            result = session.run(
                "MATCH (n) WHERE n.id IN $ids "
                "RETURN n.id AS id, labels(n) AS labels",
                ids=seed_ids,
            )
            location_ids = []
            entity_ids = []
            for record in result:
                nid = record["id"]
                labels = record["labels"] or []
                if "Location" in labels:
                    location_ids.append(nid)
                else:
                    entity_ids.append(nid)
            return location_ids, entity_ids
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.warning("       [Traverser] Location seed detection failed: %s", exc)
            return [], seed_ids

    # ==================================================================
    # PUBLIC ENTRY POINT
    # ==================================================================

    def traverse(
        self,
        seeds: List[NodeItem],
        intent: IntentType = IntentType.DISCOVERY,
        location_filter: Optional[str] = None,
        requested_attributes: Optional[List[str]] = None,
        requested_relations: Optional[List[str]] = None,
        allowed_labels: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Duyệt đồ thị thông minh (Intent-Aware Traversal & Context Builder).

        Pipeline:
          1. 1-hop traversal (nhanh, lấy context trực tiếp)
          2. Multi-hop traversal nếu intent thuộc MULTI_HOP_INTENTS (2-3 hops)
          3. Spatial clustering cho TOUR_PLAN
                """

        if not seeds:
            return []

        # ------------------------------------------------------------------
        # 1. LOAD POLICY
        # ------------------------------------------------------------------
        requested_rel_policy = self._requested_relation_policy(requested_relations)
        allowed_rels = requested_rel_policy or INTENT_TRAVERSAL_POLICY.get(intent, TRAVERSAL_WHITELIST)
        requested_attr_policy = self._requested_attribute_policy(requested_attributes)
        target_attributes = requested_attr_policy or []
        requested_attr_keys = {key for key, _ in target_attributes} if requested_attr_policy else set()

        dynamic_return_clause = ""
        for attr_key, _ in target_attributes:
            dynamic_return_clause += f", start.{attr_key} AS {attr_key}"

        use_multi_hop = intent in MULTI_HOP_INTENTS
        logger.info("       Intent: '%s' | Rels: %s | Attrs: %s | MultiHop: %s", intent, len(allowed_rels), [x[0] for x in target_attributes], use_multi_hop)

        seed_ids = [item.id for item in seeds]
        context_lines: List[str] = []
        seen_facts: Set[str] = set()

        try:
            with self.driver.session() as session:
                # ----------------------------------------------------------
                # 1b. DETECT LOCATION SEEDS — tách thành 2 nhóm
                # ----------------------------------------------------------
                location_seed_ids, entity_seed_ids = self._get_location_seed_ids(session, seed_ids)
                if location_seed_ids:
                    logger.info("       Location seeds detected: %s", location_seed_ids)

                # ----------------------------------------------------------
                # 2. 1-HOP TRAVERSAL (cho entity seeds)
                # ----------------------------------------------------------
                if entity_seed_ids:
                    cypher_1hop = f"""
                    MATCH (start)
                    WHERE start.id IN $seed_ids

                    OPTIONAL MATCH (start)-[r]-(end)
                    WHERE type(r) IN $allowed_rels

                    OPTIONAL MATCH (end)-[:LOCATED_IN]->(end_loc:Location)

                    WITH start, r, end, end_loc
                    WHERE (
                        $location_filter IS NULL OR
                        $location_filter = "" OR
                        end IS NULL OR
                        toLower(coalesce(start.address, '')) CONTAINS toLower($location_filter) OR
                        toLower(coalesce(start.name, '')) CONTAINS toLower($location_filter) OR
                        toLower(coalesce(end.name, '')) CONTAINS toLower($location_filter) OR
                        toLower(coalesce(end.address, '')) CONTAINS toLower($location_filter) OR
                        toLower(coalesce(end_loc.name, '')) CONTAINS toLower($location_filter)
                    )

                    RETURN start.name        AS subject,
                           start.description AS subject_desc,
                           start.address     AS subject_addr,
                           start.phone       AS subject_phone,
                           labels(start)     AS subject_labels
                           {dynamic_return_clause},
                           type(r)           AS rel_type,
                           end.name          AS object,
                          labels(end)       AS object_labels,
                           end.address       AS object_addr
                    LIMIT 100
                    """
                    records_1hop = session.run(
                        cypher_1hop,
                        seed_ids=entity_seed_ids,
                        allowed_rels=allowed_rels,
                        location_filter=location_filter,
                    )
                    self._process_records(
                        records_1hop,
                        context_lines,
                        seen_facts,
                        target_attributes,
                        intent=intent,
                        requested_attr_keys=requested_attr_keys,
                        allowed_labels=allowed_labels,
                    )

                # ----------------------------------------------------------
                # 2b. LOCATION SEED EXPANSION (cho Location seeds)
                # ----------------------------------------------------------
                if location_seed_ids:
                    self._expand_location_seeds(
                        session,
                        location_seed_ids,
                        allowed_rels,
                        context_lines,
                        seen_facts,
                        target_attributes,
                        intent,
                        requested_attr_keys,
                        allowed_labels,
                        location_filter,
                    )

                # FALLBACK: Nếu 1-hop trả về 0 facts, thử lại với full whitelist
                if not context_lines:
                    fallback_rels = TRAVERSAL_WHITELIST
                    if set(fallback_rels) != set(allowed_rels):
                        logger.info("       1-hop returned 0 facts. Retrying with full whitelist...")
                        # Fallback cho entity seeds
                        if entity_seed_ids:
                            records_fallback = session.run(
                                cypher_1hop,
                                seed_ids=entity_seed_ids,
                                allowed_rels=fallback_rels,
                                location_filter=location_filter,
                            )
                            self._process_records(
                                records_fallback,
                                context_lines,
                                seen_facts,
                                target_attributes,
                                intent=intent,
                                requested_attr_keys=requested_attr_keys,
                                allowed_labels=allowed_labels,
                            )
                        # Fallback cho location seeds (bỏ strict attr filter)
                        if location_seed_ids and not context_lines:
                            self._expand_location_seeds(
                                session,
                                location_seed_ids,
                                fallback_rels,
                                context_lines,
                                seen_facts,
                                target_attributes,
                                intent,
                                set(),  # Bỏ strict attr filter cho fallback
                                allowed_labels,
                                location_filter,
                            )
                        if context_lines:
                            logger.warning("       Fallback recovered %s facts.", len(context_lines))
                    else:
                        logger.info("       1-hop returned 0 facts. No broader fallback available.")

                # ----------------------------------------------------------
                # 3. MULTI-HOP TRAVERSAL (chỉ khi intent cần suy luận sâu)
                # ----------------------------------------------------------
                if use_multi_hop:
                    logger.info("       Activating MULTI-HOP REASONING (2-3 hops)...")
                    all_reachable_ids = list(entity_seed_ids)
                    if location_seed_ids:
                        expand_result = session.run(
                            "MATCH (e)-[:LOCATED_IN]->(l:Location) "
                            "WHERE l.id IN $ids "
                            "RETURN collect(DISTINCT e.id) AS entity_ids",
                            ids=location_seed_ids,
                        )
                        expanded = expand_result.single()
                        if expanded:
                            all_reachable_ids = list(set(entity_seed_ids + expanded["entity_ids"]))

                    if all_reachable_ids:
                        multi_hop_facts = self._traverse_multi_hop(
                            session,
                            all_reachable_ids,
                            allowed_rels,
                            max_hops=2,
                            location_filter=location_filter,
                        )
                        for fact in multi_hop_facts:
                            self._add_fact(context_lines, seen_facts, fact)

                # ----------------------------------------------------------
                # 4. SPATIAL CLUSTERING (chỉ cho TOUR_PLAN)
                # ----------------------------------------------------------
                if intent == IntentType.TOUR_PLAN:
                    logger.info("       Activating SPATIAL CLUSTERING for Itinerary...")
                    cluster_facts = self._find_spatial_clusters(seed_ids, location_filter=location_filter)
                    context_lines = cluster_facts + context_lines

        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.error(" Graph traversal error: %s", e)
            return []

        return context_lines

    # ==================================================================
    # LOCATION SEED EXPANSION
    # ==================================================================

    def _expand_location_seeds(
        self,
        session,
        location_seed_ids: List[str],
        allowed_rels: List[str],
        context_lines: List[str],
        seen_facts: Set[str],
        target_attributes: List[Tuple],
        intent: IntentType,
        requested_attr_keys: Set[str],
        allowed_labels: Optional[List[str]],
        location_filter: Optional[str],
    ) -> None:
        """Mở rộng từ Location seed nodes: Location ← LOCATED_IN — Entity — [r] — Neighbor.

        Location nodes chỉ có incoming LOCATED_IN edges. Method này:
        1. Tìm entities có LOCATED_IN → Location, filter theo allowed_labels
        2. Từ mỗi entity, mở rộng 1-hop theo allowed_rels
        3. Trả về facts giống 1-hop traversal thường
        """
        if not location_seed_ids:
            return

        dynamic_return_clause = ""
        for attr_key, _ in target_attributes:
            dynamic_return_clause += f", entity.{attr_key} AS {attr_key}"

        # Xây label filter cho entity (chỉ lấy đúng loại node cho intent)
        label_filter = ""
        if allowed_labels:
            conditions = [f"entity:{lbl}" for lbl in allowed_labels]
            label_filter = "AND (" + " OR ".join(conditions) + ")"

        cypher = f"""
        MATCH (entity)-[:LOCATED_IN]->(loc:Location)
        WHERE loc.id IN $location_seed_ids
          {label_filter}

        OPTIONAL MATCH (entity)-[r]-(neighbor)
        WHERE type(r) IN $allowed_rels
          AND neighbor.id <> loc.id
          AND NOT neighbor:Location

        OPTIONAL MATCH (neighbor)-[:LOCATED_IN]->(neighbor_loc:Location)

        WITH entity, r, neighbor, neighbor_loc, loc
        WHERE (
            $location_filter IS NULL OR
            $location_filter = "" OR
            neighbor IS NULL OR
            toLower(coalesce(entity.address, '')) CONTAINS toLower($location_filter) OR
            toLower(coalesce(entity.name, '')) CONTAINS toLower($location_filter) OR
            toLower(coalesce(neighbor.name, '')) CONTAINS toLower($location_filter) OR
            toLower(coalesce(neighbor.address, '')) CONTAINS toLower($location_filter) OR
            toLower(coalesce(neighbor_loc.name, '')) CONTAINS toLower($location_filter)
        )

        RETURN entity.name        AS subject,
               entity.description AS subject_desc,
               entity.address     AS subject_addr,
               entity.phone       AS subject_phone,
               labels(entity)     AS subject_labels
               {dynamic_return_clause},
               type(r)            AS rel_type,
               neighbor.name      AS object,
               labels(neighbor)   AS object_labels,
               neighbor.address   AS object_addr
        LIMIT 200
        """

        try:
            records = session.run(
                cypher,
                location_seed_ids=location_seed_ids,
                allowed_rels=allowed_rels,
                location_filter=location_filter,
            )
            self._process_records(
                records,
                context_lines,
                seen_facts,
                target_attributes,
                intent=intent,
                requested_attr_keys=requested_attr_keys,
                allowed_labels=allowed_labels,
            )
            logger.info("       Location expansion: found %d facts from %d location seeds",
                        len(context_lines), len(location_seed_ids))
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.warning("       Location expansion warning (non-fatal): %s", exc)

    # ==================================================================
    # MULTI-HOP REASONING
    # ==================================================================

    def _traverse_multi_hop(
        self,
        session,
        seed_ids: List[str],
        allowed_rels: List[str],
        max_hops: int = 2,
        location_filter: Optional[str] = None,
    ) -> List[str]:
        """
        Multi-hop path traversal (2-3 hops) với path scoring.

        Thuật toán:
          - Tìm tất cả path độ dài 2..max_hops từ seed nodes
          - Chỉ đi qua các quan hệ trong allowed_rels
          - Tính path_score = trung bình confidence của từng cạnh trong path
          - Chỉ giữ path có path_score >= 0.65
          - Trả về danh sách fact dạng chuỗi A -> rel1 -> B -> rel2 -> C

        Ví dụ kết quả:
          "Hồ Biển Hồ gần Nhà hàng Pleiku Garden, nơi phục vụ Bò một nắng Tây Nguyên
           (chuỗi suy luận: NEAR -> HAS)"
        """
        facts: List[str] = []
        hop_upper_bound = max(2, min(int(max_hops or 2), 3))

        # Cypher dùng variable-length pattern [*2..max_hops]:
        # - Mỗi relationship trong path phải thuộc allowed_rels
        # - Các node trung gian không được là chính seed node đó
        cypher = f"""
        MATCH path = (seed)-[rels*2..{hop_upper_bound}]-(target)
        WHERE seed.id IN $seed_ids
          AND target.id <> seed.id
          AND ALL(r IN rels WHERE type(r) IN $allowed_rels)
          AND ALL(n IN nodes(path)[1..-1] WHERE n.id <> seed.id)

        OPTIONAL MATCH (seed)-[:LOCATED_IN]->(seed_loc:Location)
        OPTIONAL MATCH (target)-[:LOCATED_IN]->(target_loc:Location)

        WITH seed, target, rels, seed_loc, target_loc
        WHERE (
                $location_filter IS NULL OR
                $location_filter = "" OR
                toLower(coalesce(target.address, '')) CONTAINS toLower($location_filter) OR
                toLower(coalesce(target_loc.name, '')) CONTAINS toLower($location_filter) OR
                toLower(coalesce(seed.address, '')) CONTAINS toLower($location_filter) OR
                toLower(coalesce(seed.name, '')) CONTAINS toLower($location_filter) OR
                toLower(coalesce(seed_loc.name, '')) CONTAINS toLower($location_filter)
            )

        WITH seed, target, rels,
             [r IN rels | type(r)] AS rel_chain,
             size(rels)            AS hop_count

        WITH seed, target, rel_chain, hop_count,
             reduce(
               s = 0.0,
               rel_type IN [r IN rels | type(r)] |
                 s + coalesce(
                   CASE rel_type
                     WHEN 'LOCATED_IN'  THEN 1.0
                     WHEN 'Guide_for'   THEN 1.0
                     WHEN 'BELONGS_TO'  THEN 0.95
                     WHEN 'HAS'         THEN 0.9
                     WHEN 'HELD_AT'     THEN 0.9
                     WHEN 'INCLUDES'    THEN 0.85
                     WHEN 'OFFERS'      THEN 0.8
                     WHEN 'NEAR'        THEN 0.75
                     ELSE 0.6
                   END, 0.6
                 )
             ) / hop_count AS path_score

        WHERE path_score >= 0.65

        RETURN seed.name    AS seed_name,
               target.name  AS target_name,
               target.description AS target_desc,
               target.address     AS target_addr,
               labels(target)     AS target_labels,
               rel_chain,
               hop_count,
               path_score
        ORDER BY path_score DESC
        LIMIT 40
        """

        try:
            result = session.run(
                cypher,
                seed_ids=seed_ids,
                allowed_rels=allowed_rels,
                location_filter=location_filter,
            )
            for record in result:
                fact = self._format_multi_hop_fact(record)
                if fact:
                    facts.append(fact)

        except (Neo4jClientError, ServiceUnavailable) as e:
            # Variable-length path không crash toàn bộ pipeline
            logger.warning("       Multi-hop warning (non-fatal): %s", e)

        logger.info("       Multi-hop facts found: %s", len(facts))
        return facts

    def _format_multi_hop_fact(self, record) -> str:
        """
        Chuyển một multi-hop record thành chuỗi fact có thể đọc được.

        Ví dụ output:
          "Hồ Biển Hồ → (NEAR) → Nhà hàng A → (HAS) → Bò một nắng
           [suy luận 2 bước, độ tin cậy: 0.82]"
        """
        seed   = record.get("seed_name", "")
        target = record.get("target_name", "")
        chain  = record.get("rel_chain", [])
        score  = record.get("path_score", 0.0)
        hops   = record.get("hop_count", 0)

        if not seed or not target or not chain:
            return ""

        # Map tên quan hệ sang tiếng Việt
        chain_vn = [RELATIONSHIP_MAP.get(r, r) for r in chain]
        chain_str = " → ".join(chain_vn)

        fact = f"{seed} (liên kết {hops} bước: {chain_str}) → {target}"

        target_desc = record.get("target_desc")
        target_labels = set(record.get("target_labels") or [])
        if target_desc and "Tour" not in target_labels:
            fact += f": {str(target_desc)[:120]}"

        target_addr = record.get("target_addr")
        if target_addr:
            fact += f" (Địa chỉ: {target_addr})"

        fact += f" [độ tin cậy path: {score:.2f}]"
        return fact

    def _process_records(
        self,
        records,
        context_lines: List[str],
        seen_facts: Set[str],
        target_attributes: List[Tuple],
        intent: IntentType = None,
        requested_attr_keys: Optional[Set[str]] = None,
        allowed_labels: Optional[List[str]] = None,
    ):
        """Hàm phụ trợ: Parse kết quả Neo4j thành văn bản ngữ cảnh"""
        requested_attr_keys = requested_attr_keys or set()
        strict_attrs = bool(requested_attr_keys)
        for record in records:
            subj = record["subject"]
            subj_labels = set(record.get("subject_labels") or [])

            # --- A. Context Cơ bản ---
            # Skip description for Tour nodes — their full_content is too long.
            # LLM generates tour descriptions from structured INCLUDES data instead.
            if record["subject_desc"] and "Tour" not in subj_labels and (not strict_attrs or "description" in requested_attr_keys):
                self._add_fact(context_lines, seen_facts, f"{subj}: {record['subject_desc']}")
            if record["subject_addr"] and (not strict_attrs or "address" in requested_attr_keys):
                self._add_fact(context_lines, seen_facts, f"{subj} - Địa chỉ: {record['subject_addr']}")
            if record.get("subject_phone") and (not strict_attrs or "phone" in requested_attr_keys):
                self._add_fact(context_lines, seen_facts, f"{subj} - SĐT: {record['subject_phone']}")

            # --- B. Context Chuyên sâu (Dynamic Attributes từ Config) ---
            for attr_key, attr_label in target_attributes:
                val = record.get(attr_key)
                if val and str(val).strip():
                    if isinstance(val, list):
                        val = ", ".join(val)
                    self._add_fact(context_lines, seen_facts, f"{attr_label} của {subj}: {val}")

            # --- C. Context Quan hệ (Hàng xóm trực tiếp) ---
            if record["rel_type"] and record["object"]:
                object_labels = record.get("object_labels") or []
                if allowed_labels is not None and object_labels:
                    if not any(lbl in allowed_labels for lbl in object_labels):
                        continue

                rel_raw = record["rel_type"]
                obj = record["object"]
                # Map tên quan hệ sang tiếng Việt cho tự nhiên
                rel_vn = RELATIONSHIP_MAP.get(rel_raw, rel_raw)
                
                fact = f"{subj} {rel_vn} {obj}"
                if record["object_addr"]:
                     fact += f" (Địa chỉ: {record['object_addr']})"
                
                self._add_fact(context_lines, seen_facts, fact)

    @staticmethod
    def _normalize_fact_key(fact: str) -> str:
        """Normalize fact for dedup: strip accents, lowercase, collapse whitespace."""
        norm = normalize_text(fact, strip_punct=True)
        # Strip common prefixes (legacy + new format)
        norm = _re.sub(r"^(thong tin|mo ta|dia chi|sdt)\s+(cua\s+)?", "", norm)
        # Strip new format: "entity name - dia chi: ..." → "entity name"
        norm = _re.sub(r"\s*-\s*(dia chi|sdt|loai hinh)\s*:.*$", "", norm)
        return norm.strip()

    def _add_fact(self, context_list: List[str], seen_set: Set[str], fact: str):
        """Helper để thêm fact và tránh trùng lặp (normalized comparison)."""
        if not fact:
            return
        key = self._normalize_fact_key(fact)
        if not key or key in seen_set:
            return
        context_list.append(fact)
        seen_set.add(key)

    def _find_spatial_clusters(self, seed_ids: List[str], location_filter: Optional[str] = None) -> List[str]:
        """
        Tìm các địa điểm lân cận (cùng đơn vị hành chính) để gợi ý tiện đường đi.
        Logic: Tìm các node khác cùng có quan hệ LOCATED_IN tới cùng một đơn vị hành chính.
        """
        spatial_context = []
        
        # Query này tìm: Node Gốc -> (thuộc) -> Đơn vị hành chính <- (thuộc) -> Node Khác
        # Chỉ lấy Top 5 địa điểm lân cận để tránh nhiễu
        cypher = """
        MATCH (seed)
        WHERE seed.id IN $seed_ids
        
        MATCH (seed)-[:LOCATED_IN]->(admin_unit)
        WHERE (
            $location_filter IS NULL OR
            $location_filter = "" OR
            toLower(coalesce(admin_unit.name, '')) CONTAINS toLower($location_filter) OR
            toLower(coalesce(seed.address, '')) CONTAINS toLower($location_filter) OR
            toLower(coalesce(seed.name, '')) CONTAINS toLower($location_filter)
        )
        
        MATCH (neighbor)-[:LOCATED_IN]->(admin_unit)
        WHERE neighbor.id <> seed.id  // Không lấy chính nó
        
        WITH admin_unit, collect(DISTINCT neighbor.name)[..5] AS nearby_places
        WHERE size(nearby_places) > 0
        
        RETURN admin_unit.name AS location, nearby_places
        """
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher, seed_ids=seed_ids, location_filter=location_filter)
                for record in result:
                    loc = record["location"]
                    places = ", ".join(record["nearby_places"])
                    fact = f"Tại khu vực {loc} còn có các địa điểm lân cận có thể ghé thăm: {places}."
                    spatial_context.append(fact)
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.warning(" Spatial clustering warning: %s", e)
            
        return spatial_context
