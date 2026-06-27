"""SeedRetriever — Coordinator for hybrid retrieval.

Delegates to specialized strategy classes:
- EntityRetriever  — entity lookup (exact, fuzzy, alias, normalised, semantic)
- EventRetriever   — event-specific search (category, location, month/year)
- FoodRetriever     — food/dish & restaurant retrieval
- TourPlanRetriever — tour plan quotas, proximity search, relation-constrained search
"""

import re
import logging
from typing import List, Dict, Optional, Set
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
from graph_rag.core.state import NodeItem
from graph_rag.config import TOP_K
from graph_rag.modules.retrieval.vector_search import search_vector_loop
from graph_rag.modules.retrieval.fulltext_search import search_fulltext_loop
from graph_rag.modules.retrieval.hybrid_fusion import reciprocal_rank_fusion
from graph_rag.modules.retrieval.entity_retriever import EntityRetriever
from graph_rag.modules.retrieval.event_retriever import EventRetriever
from graph_rag.modules.retrieval.food_retriever import FoodRetriever
from graph_rag.modules.retrieval.tour_plan_retriever import TourPlanRetriever
from graph_rag.core.intents import IntentType
from graph_rag.core import keywords, thresholds
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)

# Module-level lazy singleton for AdminRegionMappingService
_admin_region_mapping_svc = None
def _get_admin_region_mapping_service():
    global _admin_region_mapping_svc
    if _admin_region_mapping_svc is None:
        from graph_rag.modules.pipeline_support.admin_region_mapping_service import AdminRegionMappingService
        _admin_region_mapping_svc = AdminRegionMappingService()
    return _admin_region_mapping_svc

CATEGORY_PHRASES: Set[str] = keywords.CATEGORY_PHRASES


class SeedRetriever:
    """Coordinator for all retrieval strategies.

    Delegates to EntityRetriever, EventRetriever, FoodRetriever,
    and TourPlanRetriever.  Owns the hybrid-search fusion logic and
    high-level orchestration in _find_seeds_internal.
    """

    INTENT_TO_LABELS = {
        IntentType.ACCOMMODATION: ["Accommodation"],
        IntentType.FOOD:          ["Restaurant", "Dish"],
        IntentType.TOURISM:       ["TouristAttraction"],
        IntentType.EVENT:         ["Event"],
        IntentType.TOUR_PLAN:     ["TouristAttraction", "Tour", "Restaurant", "Dish", "Accommodation"],
        IntentType.DISTANCE:      ["TouristAttraction", "Restaurant", "Accommodation", "Event"],
        IntentType.DISCOVERY:     ["TouristAttraction", "Restaurant", "Accommodation", "Event", "TravelInfo"],
        IntentType.ENTITY_FACT:   ["TouristAttraction", "Restaurant", "Accommodation", "Event", "Tour", "Dish", "TravelInfo"],
        IntentType.TRANSPORT_INFO: ["TravelInfo", "Location"],
        IntentType.TRAVEL_ADVICE:  ["TravelInfo", "TouristAttraction", "Restaurant", "Specialty", "Location"],
    }

    _VECTOR_INDEX_LABEL_MAP = {
        "tourist_vec_idx": "TouristAttraction",
        "restaurant_vec_idx": "Restaurant",
        "accommodation_vec_idx": "Accommodation",
        "tour_vec_idx": "Tour",
        "event_vec_idx": "Event",
        "dish_vec_idx": "Dish",
    }

    def __init__(self, driver, embedder):
        self.driver = driver
        self.embedder = embedder
        self._fuzzy_matcher = None
        # Strategy classes
        self.entity = EntityRetriever(driver)
        self.event = EventRetriever(driver)
        self.food = FoodRetriever(driver)
        self.tour_plan = TourPlanRetriever(driver)

    def _get_fuzzy_matcher(self):
        """Lazy-init the graph-based fuzzy matcher."""
        if self._fuzzy_matcher is None:
            from graph_rag.utils.fuzzy_matcher import GraphVocabulary, VietnameseFuzzyMatcher
            vocab = GraphVocabulary(self.driver)
            self._fuzzy_matcher = VietnameseFuzzyMatcher(vocab)
            self.entity.set_fuzzy_matcher(self._fuzzy_matcher)
        return self._fuzzy_matcher

    # ── Entity helpers (kept for backward compat / lightweight delegation) ──

    def _entity_name(self, entity) -> str:
        if isinstance(entity, dict):
            return str(entity.get("name") or "").strip()
        return str(entity or "").strip()

    def _entity_type(self, entity) -> str:
        if isinstance(entity, dict):
            return str(entity.get("type") or entity.get("label") or "").strip()
        return ""

    def _is_category_phrase(self, text: str) -> bool:
        return EntityRetriever.is_category_phrase(text)

    def _metadata_entities(self, metadata: Dict) -> List[Dict]:
        entities = (metadata or {}).get("entities") or []
        return entities if isinstance(entities, list) else []

    def _primary_entity_name(self, metadata: Dict) -> str:
        target = str((metadata or {}).get("target_entity") or "").strip()
        if target:
            return target
        for entity in self._metadata_entities(metadata):
            name = self._entity_name(entity)
            if name:
                return name
        return ""

    def _has_specific_entity_signal(self, metadata: Dict) -> bool:
        from graph_rag.modules.pipeline_support.admin_region_mapping_service import AdminRegionMappingService
        _BROAD_LOCATIONS = AdminRegionMappingService._get_broad_location_norms()
        for entity in self._metadata_entities(metadata):
            name = self._entity_name(entity)
            etype = self._entity_type(entity).lower()
            if not name:
                continue
            if etype in {"province", "city", "district", "ward", "commune", "location"}:
                continue
            name_norm = normalize_text(name, strip_punct=True)
            if name_norm in _BROAD_LOCATIONS:
                continue
            return True
        return bool(str((metadata or {}).get("target_entity") or "").strip())

    def _entity_lookup_candidates(self, entity_name: str) -> List[str]:
        return EntityRetriever.entity_lookup_candidates(entity_name)

    def _seed_primary_label(self, seed: NodeItem) -> str:
        labels = seed.metadata.get("labels") or []
        if labels:
            return str(labels[0])
        return str(seed.metadata.get("type") or "")

    def _boost_and_filter_by_labels(self, seeds: List[NodeItem], allowed_labels: List[str] = None) -> List[NodeItem]:
        if not seeds or not allowed_labels:
            return seeds or []
        allowed = set(allowed_labels)
        preferred = []
        fallback = []
        for seed in seeds:
            label = self._seed_primary_label(seed)
            if label in allowed:
                preferred.append(seed)
            else:
                fallback.append(seed)
        return preferred + fallback

    def _anchor_compatible_seeds(self, anchors: List[NodeItem], target_labels: List[str]) -> List[NodeItem]:
        if not anchors or not target_labels:
            return anchors or []
        target = set(target_labels)
        compatible = [seed for seed in anchors if self._seed_primary_label(seed) in target]
        return compatible or anchors

    def _should_hard_filter_location(self, metadata: Dict) -> bool:
        if (metadata or {}).get("disable_current_location_filter"):
            return False
        context = (metadata or {}).get("location_context") or {}
        try:
            confidence = float(context.get("confidence", 0.0))
        except (Neo4jClientError, ServiceUnavailable, ValueError):
            confidence = 0.0
        source = str(context.get("source") or "").lower()
        reason = str(context.get("reason") or "").lower()
        return (
            confidence >= thresholds.GROUNDING_CONFIDENCE_THRESHOLD
            or source in {"user", "ground_truth", "gps"}
            or "explicit" in reason
            or "exact" in reason
        )

    # ── Region / location helpers ──────────────────────────────────────

    def _region_filter_params(self, metadata: Dict, query_plan=None) -> tuple:
        if query_plan is not None and getattr(query_plan, "geo_scope", None) == "multi_region":
            return None, None
        if (metadata or {}).get("geo_scope") == "multi_region":
            return None, None
        if query_plan is not None:
            _region_lock = getattr(query_plan, "region_lock_mode", None)
            if _region_lock is None and isinstance(getattr(query_plan, "constraints", None), dict):
                _region_lock = query_plan.constraints.get("region_lock_mode")
            if _region_lock == "disabled_multi_anchor_comparison":
                return None, None
            region_focus = query_plan.region_focus.lower()
            if region_focus in ("all", ""):
                if query_plan.legacy_province:
                    return None, query_plan.legacy_province
                return None, None
            return query_plan.region_group or None, query_plan.legacy_province or None
        if (metadata or {}).get("region_lock_mode") == "disabled_multi_anchor_comparison":
            return None, None
        region_focus = str((metadata or {}).get("region_focus") or "").strip().lower()
        if region_focus in ("all", ""):
            legacy_province = (metadata or {}).get("legacy_province") or None
            if legacy_province:
                return None, legacy_province
            return None, None
        region_group = (metadata or {}).get("region_group") or None
        legacy_province = (metadata or {}).get("legacy_province") or None
        if legacy_province:
            svc = _get_admin_region_mapping_service()
            merged = svc.get_merged_region_groups_for_province(legacy_province)
            if merged:
                return merged, legacy_province
        return region_group, legacy_province

    @staticmethod
    def _region_group_for_cypher(region_group):
        if isinstance(region_group, list):
            return None
        return region_group

    def _resolve_province_to_region(self, province_norm: str) -> tuple:
        svc = _get_admin_region_mapping_service()
        resolved = svc.resolve(province_norm)
        if resolved and resolved.get("region_group"):
            return resolved["region_group"], resolved.get("legacy_province") or resolved.get("old_province")
        return None, None

    def _region_address_aliases(self, legacy_province: str | None, region_group) -> List[str]:
        from graph_rag.config.region_registry import region_registry
        rg_str = " ".join(region_group) if isinstance(region_group, list) else str(region_group or "")
        text = normalize_text(" ".join([str(legacy_province or ""), rg_str]))
        pid = region_registry.get_province_by_alias(text.strip())
        if not pid:
            matches = region_registry.get_province_by_keyword(text)
            pid = matches[0] if matches else None
        if pid:
            result = list(region_registry.get_keywords(pid))
            config_aliases = keywords.REGION_ADDRESS_ALIASES.get(pid.replace("_", " "), [])
            result.extend(config_aliases)
            return result
        return []

    def _filter_seeds_by_ward_location(self, seeds: list, metadata: Dict) -> list:
        if not seeds or not (metadata or {}).get("has_sub_province_location"):
            return seeds
        ward_name = str(metadata.get("sub_province_location_name") or "").strip()
        if not ward_name:
            return seeds
        try:
            seed_ids = [str(getattr(s, "id", "") or getattr(s, "content", "")) for s in seeds
                        if getattr(s, "id", None) or getattr(s, "content", None)]
            seed_names = [getattr(s, "content", "") for s in seeds]
            if not seed_ids:
                return seeds
            with self.driver.session() as sess:
                loc_rows = sess.run(
                    "MATCH (l:Location) WHERE l.name CONTAINS $loc "
                    "OR ANY(a IN l.aliases WHERE a CONTAINS $loc) "
                    "OR l.legacy_district CONTAINS $loc "
                    "RETURN elementId(l) AS id, l.name AS name",
                    loc=ward_name,
                ).data()
                loc_ids = {str(r["id"]) for r in loc_rows if r.get("id")}
                if not loc_ids:
                    return seeds
                filter_rows = sess.run(
                    "MATCH (n)-[:LOCATED_IN]->(l:Location) "
                    "WHERE elementId(l) IN $loc_ids "
                    "AND (elementId(n) IN $nids OR n.name IN $names) "
                    "RETURN DISTINCT elementId(n) AS nid",
                    loc_ids=list(loc_ids), nids=seed_ids, names=seed_names,
                ).data()
                connected_ids = {str(r["nid"]) for r in filter_rows if r.get("nid")}
                if connected_ids:
                    filtered = [s for s in seeds if str(getattr(s, "id", "")) in connected_ids]
                    if filtered:
                        logger.info("       Ward filter by LOCATED_IN→'%s': %d → %d",
                                     ward_name, len(seeds), len(filtered))
                        return filtered
        except (ValueError, TypeError) as e:
            logger.debug("       Ward filter failed, keeping all: %s", e)
        return seeds

    def _infer_trip_days(self, metadata: Dict, user_query: str) -> int:
        constraints = (metadata or {}).get("constraints") or {}
        trip_days = constraints.get("trip_days")
        try:
            if trip_days is not None:
                return max(1, int(trip_days))
        except (ValueError, Neo4jClientError):
            pass
        qs = (metadata or {}).get("query_state")
        if qs and hasattr(qs, "duration_days") and qs.duration_days > 0:
            return qs.duration_days
        q = normalize_text(str(user_query or ""), strip_punct=True)
        match = re.search(r"(\d+)\s*ngay", q)
        if match:
            try:
                return max(1, int(match.group(1)))
            except (ValueError, IndexError):
                return 1
        return 1

    def _extract_entity_from_query_text(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw:
            return ""
        prefix_match = re.search(
            r"(?i)^((?:nhà\s+hàng|nha\s+hang|quán|quan|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi|"
            r"khu\s+du\s+lịch|khu\s+du\s+lich|bảo\s+tàng|bao\s+tang|chùa|chua)\s+[^,?.!]+?)"
            r"(?:\s*[,?.!]|\s+(?:nằm|nam|tọa|toa|có|co|được|duoc|phục\s+vụ|phuc\s+vu|là|la|gần|gan))\b",
            raw,
        )
        if prefix_match:
            return prefix_match.group(1).strip()
        capitalized_match = re.search(
            r"^([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){1,5})\s+(?:là|la|nằm|nam|có|co|được|duoc)\b",
            raw,
        )
        if capitalized_match:
            return capitalized_match.group(1).strip()
        return ""

    # ── Provinces ──────────────────────────────────────────────────────

    _PROVINCE_ALIASES = None

    @classmethod
    def _get_province_aliases(cls) -> dict:
        if cls._PROVINCE_ALIASES is None:
            from graph_rag.config.region_registry import region_registry
            aliases = {}
            for pid in region_registry.get_all_province_ids():
                for kw in region_registry.get_keywords(pid):
                    aliases[kw] = pid
                for alias in region_registry.get_aliases(pid):
                    aliases[alias.lower()] = pid
                aliases[pid.replace("_", " ")] = pid
            cls._PROVINCE_ALIASES = aliases
        return cls._PROVINCE_ALIASES

    def _extract_province(self, node: NodeItem) -> str:
        parts = [
            str((node.metadata or {}).get("address") or ""),
            str((node.metadata or {}).get("province") or ""),
            str((node.metadata or {}).get("location") or ""),
            str(node.content or ""),
        ]
        text = normalize_text(" ".join(parts), strip_punct=True)
        return self._extract_province_from_text(text)

    def _extract_province_from_text(self, normalized_text: str) -> str:
        for alias, province in self._get_province_aliases().items():
            if alias in normalized_text:
                return province
        return ""

    def _seed_matches_province(self, seed: NodeItem, anchor_province: str) -> bool:
        seed_province = self._extract_province(seed)
        if not seed_province:
            return True
        return seed_province == anchor_province

    # ── Data helpers ───────────────────────────────────────────────────

    def _record_to_nodeitem(self, record, source_type: str, default_score: float = 1.0) -> NodeItem:
        from graph_rag.modules.retrieval.entity_retriever import EntityRetriever
        return EntityRetriever._record_to_nodeitem(record, source_type, default_score)

    def _deduplicate_seeds(self, seeds: List[NodeItem]) -> List[NodeItem]:
        from graph_rag.modules.retrieval.entity_retriever import EntityRetriever
        return EntityRetriever._deduplicate_seeds(seeds)

    def _convert_dict_to_nodeitems(self, seeds_dict: List[Dict], source_override: str = None) -> List[NodeItem]:
        from graph_rag.modules.retrieval.entity_retriever import EntityRetriever
        return EntityRetriever._convert_dict_to_nodeitems(seeds_dict, source_override)

    # ── Ground entities ───────────────────────────────────────────────

    # ── Entity type → allowed labels for type-scoped grounding ─────

    ENTITY_TYPE_TO_LABELS: Dict[str, List[str]] = {
        "Dish":              ["Dish"],
        "Restaurant":        ["Restaurant"],
        "Accommodation":     ["Accommodation"],
        "TouristAttraction": ["TouristAttraction"],
        "Event":             ["Event"],
        "Tour":              ["Tour"],
        "TravelAgency":      ["TravelAgency"],
        "TravelInfo":        ["TravelInfo"],
        "Location":          ["Location"],
        "Specialty":         ["Specialty"],
        "Category":          ["Category"],
    }

    def _entity_labels(self, entity_type: str) -> Optional[List[str]]:
        """Derive allowed_labels from entity type.

        Returns None for unknown/generic types so all indexes are searched.
        """
        return self.ENTITY_TYPE_TO_LABELS.get(entity_type)

    def ground_entities(self, entities) -> List[NodeItem]:
        """Batch-ground entity names to NodeItems.

        Cascade: exact → contains → normalized → semantic → fuzzy.

        Multi-hop awareness: when entity type is known (e.g. Dish),
        all search methods are scoped to that type.  This prevents
        flat semantic search from returning mixed-type results
        (e.g. 30 nodes of Dish + Restaurant + Tour for a Dish query).
        """
        results = []
        for entity in entities:
            entity_name = entity.get("name") if isinstance(entity, dict) else entity
            entity_type = str(entity.get("type") or "").strip() if isinstance(entity, dict) else ""
            if not entity_name or not isinstance(entity_name, str):
                continue

            allowed_labels = self._entity_labels(entity_type)

            is_non_groundable = (
                self.is_constraint_phrase(entity_name)
                or self.is_advice_phrase(entity_name)
                or self.is_time_phrase(entity_name)
                or self.is_price_modifier(entity_name)
            )

            nodes = self.entity.exact_match_search(entity_name)
            if nodes and len(nodes) > 1 and entity_type:
                nodes = self.entity.rank_by_type_preference(nodes, entity_type)
            if not nodes:
                nodes = self.entity.contains_alias_search(entity_name, allowed_labels=allowed_labels)
            if not nodes:
                nodes = self.entity.normalized_name_search(entity_name, allowed_labels=allowed_labels)
            if not nodes and not is_non_groundable:
                nodes = self.entity.semantic_entity_search(entity_name, self.embedder, allowed_labels=allowed_labels)
                if nodes:
                    logger.info("Semantic grounding: '%s' → '%s' (score=%.3f)",
                                 entity_name, nodes[0].content, nodes[0].score)
            if not nodes and not is_non_groundable:
                nodes = self.entity.fuzzy_name_search(entity_name, allowed_labels=allowed_labels)
            results.extend(nodes)
        return self._deduplicate_seeds(results)

    # ── Phrase classification ──────────────────────────────────────────

    def is_constraint_phrase(self, text: str) -> bool:
        norm = normalize_text(text, strip_punct=True)
        phrases = [
            "tham quan ngoai troi", "ngoai troi", "co phu hop khong", "phu hop khong",
            "nen di", "co nen", "phai mang theo", "mang gi", "mac gi", "mac trang phuc"
        ]
        return any(p in norm for p in phrases)

    def is_advice_phrase(self, text: str) -> bool:
        norm = normalize_text(text, strip_punct=True)
        phrases = [
            "kinh nghiem", "meo", "luu y", "nen chuan bi", "can biet", "kinh nghiem dat phong",
            "kinh nghiem di", "meo vat", "luu y khi di", "chia se kinh nghiem"
        ]
        return any(p in norm for p in phrases)

    def is_time_phrase(self, text: str) -> bool:
        norm = normalize_text(text, strip_punct=True)
        phrases = [
            "cuoi nam", "dip cuoi nam", "cuoi thang", "dau nam", "dip le", "tet",
            "mua dong", "mua he", "mua xuan", "mua thu", "mua mua", "mua kho",
            "thang 1", "thang 2", "thang 3", "thang 4", "thang 5", "thang 6",
            "thang 7", "thang 8", "thang 9", "thang 10", "thang 11", "thang 12"
        ]
        return any(p in norm for p in phrases)

    def is_price_modifier(self, text: str) -> bool:
        norm = normalize_text(text, strip_punct=True)
        phrases = [
            "gia re", "tiet kiem", "gia binh dan", "re nhat", "gia tot", "chi phi thap",
            "mien phi", "khong ton tien", "gia ca phai chang"
        ]
        return any(p in norm for p in phrases)

    # ── Static disambiguation helpers ──────────────────────────────────

    @staticmethod
    def filter_nodes_by_type(nodes: List[NodeItem], expected_labels: List[str]) -> List[NodeItem]:
        return EntityRetriever.filter_nodes_by_type(nodes, expected_labels)

    @staticmethod
    def get_intent_labels(intent) -> List[str]:
        if hasattr(intent, 'value'):
            intent = intent.value
        for k, v in SeedRetriever.INTENT_TO_LABELS.items():
            k_val = k.value if hasattr(k, 'value') else str(k)
            if k_val == str(intent):
                return v
        return []

    # ── Hybrid search (core coordination) ──────────────────────────────

    def _rewrite_search_query_for_phase(self, user_query: str, label: str, metadata: Dict) -> str:
        q_norm = normalize_text(user_query, strip_punct=True)
        from graph_rag.config.region_registry import region_registry
        locations = []
        for pid in region_registry.get_all_province_ids():
            for kw in region_registry.get_keywords(pid):
                if kw in q_norm:
                    locations.append(region_registry.get_province_display_name(pid))
                    break
        loc_text = " ".join(locations) if locations else ""
        _PHASE_TEMPLATES = {
            "TouristAttraction": f"điểm tham quan du lịch nổi tiếng {loc_text}",
            "Restaurant": f"nhà hàng quán ăn đặc sản {loc_text}",
            "Accommodation": f"khách sạn homestay {loc_text}",
            "Tour": f"tour du lịch {loc_text}",
        }
        rewritten = _PHASE_TEMPLATES.get(label, user_query)
        return rewritten.strip()

    def _hybrid_search(self, user_query: str, metadata: Dict, top_k: int,
                       allowed_labels: List[str] = None,
                       location_filter: str = None,
                       query_plan=None) -> List[NodeItem]:
        """Hybrid search: vector + fulltext fusion with region/location filters."""
        if metadata and metadata.get("search_keywords"):
            search_text = " ".join([str(w) for w in metadata["search_keywords"] if str(w).strip()])
        else:
            search_text = user_query
        search_terms = str(search_text or "").split()
        if len(search_terms) > 40:
            search_text = " ".join(search_terms[:40])
            logger.info("       Fulltext query shortened to 40 tokens to avoid Lucene clause explosion.")

        query_vector = self.embedder.embed_query(user_query)

        if location_filter:
            _loc_norm = normalize_text(location_filter, strip_punct=True)
            if _loc_norm in _get_admin_region_mapping_service().BROAD_LOCATION_NORMS:
                region_group, legacy_province = self._resolve_province_to_region(_loc_norm)
            else:
                region_group, legacy_province = None, None
        else:
            region_group, legacy_province = self._region_filter_params(metadata or {}, query_plan)

        logger.info("       [DEBUG-HYBRID] search_text='%s' | region_group=%s | legacy_province=%s | "
                     "location_filter=%s | allowed_labels=%s",
                     str(search_text or "")[:80], region_group, legacy_province,
                     location_filter, allowed_labels)

        # Food specialty shortcut
        semantic_category = (metadata or {}).get("semantic_category") or ""
        if query_plan:
            semantic_category = getattr(query_plan, "semantic_category", "") or semantic_category
        if "food_specialty" in semantic_category and "Dish" in (allowed_labels or []):
            is_follow_up = False
            if query_plan:
                is_follow_up = getattr(query_plan, "is_follow_up", False)
            elif metadata:
                is_follow_up = metadata.get("is_follow_up", False)
            specialty_seeds = self.food.food_specialty_search(
                region_group=region_group, legacy_province=legacy_province,
                top_k=top_k * 3 if is_follow_up else top_k,
            )
            if specialty_seeds:
                logger.info("         ⚙️ Food specialty search: found %s Dish nodes", len(specialty_seeds))
                return specialty_seeds

        vector_seeds_dict = search_vector_loop(
            self.driver, query_vector, k=top_k * 2,
            filter_labels=allowed_labels, filter_city=location_filter,
            region_group=region_group, legacy_province=legacy_province,
        )
        keyword_seeds_dict = search_fulltext_loop(
            self.driver, search_text, k=top_k * 2,
            filter_labels=allowed_labels, filter_city=location_filter,
            region_group=region_group, legacy_province=legacy_province,
        )

        # Fallback: retry with legacy_province only if region_group returned 0
        if not vector_seeds_dict and not keyword_seeds_dict and region_group and legacy_province:
            logger.info("         ⚙️ Region-group filter returned 0 results; retrying with legacy_province only")
            vector_seeds_dict = search_vector_loop(
                self.driver, query_vector, k=top_k * 2,
                filter_labels=allowed_labels, filter_city=location_filter,
                region_group=None, legacy_province=legacy_province,
            )
            keyword_seeds_dict = search_fulltext_loop(
                self.driver, search_text, k=top_k * 2,
                filter_labels=allowed_labels, filter_city=location_filter,
                region_group=None, legacy_province=legacy_province,
            )

        combined_seeds_dict = reciprocal_rank_fusion(vector_seeds_dict, keyword_seeds_dict)
        if allowed_labels:
            allowed = set(allowed_labels)
            combined_seeds_dict.sort(
                key=lambda item: (
                    0 if str(item.get("type") or "") in allowed else 1,
                    -float(item.get("final_score") or item.get("score") or 0.0),
                )
            )
        top_seeds_dict = combined_seeds_dict[:top_k]
        return self._convert_dict_to_nodeitems(top_seeds_dict)

    # ── Main entry points ──────────────────────────────────────────────

    def find_seeds(self, user_query: str, metadata: Dict = None, top_k=TOP_K,
                   rank: bool = True, query_plan=None) -> List[NodeItem]:
        seeds = self._find_seeds_internal(user_query, metadata, top_k, rank, query_plan)

        intent = query_plan.intent if query_plan else (
            metadata.get("intent") if metadata else IntentType.DISCOVERY)
        allowed_labels = (query_plan.target_labels if query_plan else
                          ((metadata or {}).get("retrieval_allowed_labels") or None))
        fallback_policy = (metadata or {}).get("fallback_policy") or ""
        _TRAVEL_INFO_INTENTS = {
            IntentType.CASHLESS_PAYMENT, IntentType.WEATHER_ADVICE,
            IntentType.TRANSPORT_INFO, IntentType.TRAVEL_ADVICE, IntentType.EMERGENCY_SUPPORT,
        }
        is_travel_info_contract = (
            intent in _TRAVEL_INFO_INTENTS
            or (metadata or {}).get("intent") in _TRAVEL_INFO_INTENTS
            or fallback_policy.endswith("_guided_fallback")
            or (allowed_labels and len(allowed_labels) == 1 and allowed_labels[0] == "TravelInfo")
        )
        if is_travel_info_contract:
            policy_to_allowed_topics = {
                "emergency_guided_fallback": ["emergency", ""],
                "payment_guided_fallback": ["payment"],
                "booking_guided_fallback": ["accommodation_tips", "booking"],
                "weather_guided_fallback": ["weather"],
                "transport_local_guided_fallback": ["transport", "transport_local", "airport"],
                "airport_guided_fallback": ["airport"],
                "seafood_shopping_guided_fallback": ["shopping", "seafood"],
                "budget_guided_fallback": ["budget"],
                "health_guided_fallback": ["health"],
                "community_guided_fallback": ["community"],
                "event_schedule_guided_fallback": ["event"],
                "general_practical_guided_fallback": ["general"],
            }
            allowed_topics = policy_to_allowed_topics.get(fallback_policy)
            filtered = []
            for seed in seeds:
                labels = (seed.metadata.get("labels", []) if hasattr(seed, "metadata") and seed.metadata else [])
                topic = (str(seed.metadata.get("topic") or "").strip().lower()
                         if hasattr(seed, "metadata") and seed.metadata else "")
                s_name = normalize_text(str(getattr(seed, "content", "") or ""), strip_punct=True)
                if s_name in _get_admin_region_mapping_service().BROAD_LOCATION_NORMS or "Location" in labels:
                    logger.info("       [Topic Filter] Excluding Location/broad location seed: '%s'", seed.content)
                    continue
                if "TravelInfo" in labels:
                    if allowed_topics and not any(topic == t for t in allowed_topics):
                        logger.info("       [Topic Filter] Excluding TravelInfo seed with off-topic '%s': '%s'",
                                     topic, seed.content)
                        continue
                filtered.append(seed)
            seeds = filtered

        forbidden_labels_from_contract = set()
        if query_plan is not None:
            forbidden_labels_from_contract.update(getattr(query_plan, "forbidden_labels", []) or [])
        if metadata:
            if "forbidden_labels" in metadata:
                forbidden_labels_from_contract.update(metadata["forbidden_labels"] or [])
            if "query_plan" in metadata and isinstance(metadata["query_plan"], dict):
                forbidden_labels_from_contract.update(metadata["query_plan"].get("forbidden_labels") or [])

        if forbidden_labels_from_contract:
            filtered_seeds = []
            for seed in seeds:
                labels = set(seed.metadata.get("labels") or [])
                if seed.metadata.get("type"):
                    labels.add(seed.metadata.get("type"))
                if not {l.lower() for l in labels}.intersection({l.lower() for l in forbidden_labels_from_contract}):
                    filtered_seeds.append(seed)
            if len(filtered_seeds) < len(seeds):
                logger.info("       [SeedRetriever] Hard filtered %s seeds matching forbidden labels: %s",
                             len(seeds) - len(filtered_seeds), forbidden_labels_from_contract)
            return filtered_seeds
        return seeds

    def _find_seeds_internal(self, user_query: str, metadata: Dict = None, top_k=TOP_K,
                              rank: bool = True, query_plan=None) -> List[NodeItem]:
        logger.info("   [DEBUG-SEEDS] >>> _find_seeds_internal ENTER: query='%s'", str(user_query or "")[:80])

        # === Extract intent & plan ===
        _meta_intent = (metadata or {}).get("intent")
        if query_plan is not None:
            intent = _meta_intent or query_plan.intent or IntentType.DISCOVERY
            allowed_labels = query_plan.target_labels or None
            legacy_province = query_plan.legacy_province
            region_focus = query_plan.region_focus
        else:
            intent = _meta_intent or IntentType.DISCOVERY
            allowed_labels = (metadata or {}).get("retrieval_allowed_labels") or None
            legacy_province = (metadata or {}).get("legacy_province")
            region_focus = (metadata or {}).get("region_focus")

        fallback_policy = (metadata or {}).get("fallback_policy") or ""
        _TRAVEL_INFO_INTENTS = {
            IntentType.CASHLESS_PAYMENT, IntentType.WEATHER_ADVICE,
            IntentType.TRANSPORT_INFO, IntentType.TRAVEL_ADVICE, IntentType.EMERGENCY_SUPPORT,
        }
        is_travel_info_contract = (
            intent in _TRAVEL_INFO_INTENTS
            or (metadata or {}).get("intent") in _TRAVEL_INFO_INTENTS
            or fallback_policy.endswith("_guided_fallback")
            or (allowed_labels and len(allowed_labels) == 1 and allowed_labels[0] == "TravelInfo")
        )
        broad_location_norms = _get_admin_region_mapping_service().BROAD_LOCATION_NORMS

        def clean_seeds_of_broad_locations(seeds_list):
            if not is_travel_info_contract or not seeds_list:
                return seeds_list
            cleaned = []
            for s in seeds_list:
                s_labels = (s.metadata.get("labels", []) if hasattr(s, "metadata") and s.metadata else [])
                s_name = normalize_text(str(getattr(s, "content", "") or ""), strip_punct=True)
                if s_name in broad_location_norms or "Location" in s_labels:
                    logger.info("       [TravelInfo Contract] Excluding broad location/Location seed: '%s'",
                                 getattr(s, 'content', ''))
                    continue
                cleaned.append(s)
            return cleaned

        # === Direct Weather Retrieval ===
        is_weather_query = (
            intent == IntentType.WEATHER_ADVICE
            or (metadata or {}).get("intent") == IntentType.WEATHER_ADVICE
            or (query_plan is not None and getattr(query_plan, "intent", None) == IntentType.WEATHER_ADVICE)
        )
        if is_weather_query:
            logger.info("       Executing DIRECT WEATHER RETRIEVAL...")
            locations = (metadata or {}).get("locations") or []
            if query_plan is not None and hasattr(query_plan, "month_constraint"):
                months = getattr(query_plan, "month_constraint", []) or []
            else:
                months = (metadata or {}).get("time_constraint_months") or []
            time_terms = []
            for m in months:
                time_terms.append(f"tháng {m}")
                time_terms.append(f"thang {m}")
            for term in ["mùa mưa", "mua mua", "mùa khô", "mua kho", "cuối năm", "cuoi nam",
                         "mùa xuân", "mua xuan", "mùa hè", "mua he", "mùa thu", "mua thu",
                         "mùa đông", "mua dong", "đầu năm", "dau nam"]:
                if term in normalize_text(user_query, strip_punct=True):
                    time_terms.append(term)
            results = []
            try:
                with self.driver.session() as session:
                    cypher = """
                    MATCH (t:TravelInfo)
                    WHERE t.topic = 'weather'
                    RETURN t.id AS id, t.name AS name, labels(t) AS labels, t.description AS description,
                           t.address AS address, t.topic AS topic,
                           CASE WHEN t.location IS NOT NULL AND toLower(toString(t.location)) STARTS WITH 'point'
                                AND t.location.latitude IS NOT NULL THEN t.location.latitude ELSE toFloat(t.lat) END AS lat,
                           CASE WHEN t.location IS NOT NULL AND toLower(toString(t.location)) STARTS WITH 'point'
                                AND t.location.longitude IS NOT NULL THEN t.location.longitude ELSE toFloat(t.lng) END AS lng
                    """
                    records = session.run(cypher).data()
                    for r in records:
                        name = r.get("name") or ""
                        desc = r.get("description") or ""
                        addr = r.get("address") or ""
                        content_to_check = f"{name} {desc} {addr}".lower()
                        matches_location = False
                        if not locations:
                            matches_location = True
                        else:
                            from graph_rag.config.region_registry import region_registry
                            for loc in locations:
                                loc_norm = normalize_text(loc, strip_punct=True)
                                aliases = [loc_norm]
                                pid = region_registry.get_province_by_alias(loc_norm)
                                if not pid:
                                    pmatches = region_registry.get_province_by_keyword(loc_norm)
                                    pid = pmatches[0] if pmatches else None
                                if pid:
                                    aliases.extend([a.lower() for a in region_registry.get_aliases(pid)])
                                    aliases.extend(region_registry.get_keywords(pid))
                                else:
                                    if loc_norm == "gia lai":
                                        aliases.extend(["gialai", "pleiku"])
                                    elif loc_norm == "binh dinh":
                                        aliases.extend(["binhdinh", "quy nhon", "quynhon"])
                                if any(alias in normalize_text(content_to_check, strip_punct=True) for alias in aliases):
                                    matches_location = True
                                    break
                        if matches_location:
                            base_score = 1.0
                            bonus = 0.0
                            if time_terms:
                                desc_norm = normalize_text(desc, strip_punct=True)
                                name_norm = normalize_text(name, strip_punct=True)
                                for term in time_terms:
                                    term_norm = normalize_text(term, strip_punct=True)
                                    if term_norm in desc_norm or term_norm in name_norm:
                                        bonus += 0.5
                            loc_bonus = 0.0
                            for loc in locations:
                                loc_norm = normalize_text(loc, strip_punct=True)
                                if loc_norm in normalize_text(name, strip_punct=True):
                                    loc_bonus += 0.3
                            score = base_score + bonus + loc_bonus
                            r["score"] = score
                            results.append(self._record_to_nodeitem(r, "direct_weather_lookup", score))
                    results.sort(key=lambda x: -x.score)
                    if results:
                        logger.info("       Direct weather search found %s seeds.", len(results))
                        return results
                    logger.info("       Direct weather search found 0 seeds, falling through to hybrid search.")
            except (Neo4jClientError, ServiceUnavailable) as exc:
                logger.error("       Direct weather search failed: %s", exc)

        # === Direct Airport/Transport Retrieval ===
        is_airport_query = (
            intent == IntentType.TRANSPORT_INFO
            or (metadata or {}).get("intent") == IntentType.TRANSPORT_INFO
            or (query_plan is not None and getattr(query_plan, "intent", None) == IntentType.TRANSPORT_INFO)
            or "airport_guided_fallback" in str((metadata or {}).get("fallback_policy", ""))
        )
        if is_airport_query:
            logger.info("       Executing DIRECT AIRPORT/TRANSPORT RETRIEVAL...")
            locations = (metadata or {}).get("locations") or []
            if not locations:
                from graph_rag.config.region_registry import region_registry
                q_norm = normalize_text(user_query, strip_punct=True)
                for pid in region_registry.get_all_province_ids():
                    for kw in region_registry.get_keywords(pid):
                        if kw in q_norm:
                            locations.append(kw)
                            break
            results = []
            try:
                with self.driver.session() as session:
                    cypher = """
                    MATCH (t:TravelInfo)
                    WHERE t.topic = 'airport'
                    RETURN t.id AS id, t.name AS name, labels(t) AS labels, t.description AS description,
                           t.address AS address, t.topic AS topic,
                           CASE WHEN t.location IS NOT NULL AND toLower(toString(t.location)) STARTS WITH 'point'
                                AND t.location.latitude IS NOT NULL THEN t.location.latitude ELSE toFloat(t.lat) END AS lat,
                           CASE WHEN t.location IS NOT NULL AND toLower(toString(t.location)) STARTS WITH 'point'
                                AND t.location.longitude IS NOT NULL THEN t.location.longitude ELSE toFloat(t.lng) END AS lng
                    """
                    records = session.run(cypher).data()
                    for r in records:
                        name = r.get("name") or ""
                        desc = r.get("description") or ""
                        addr = r.get("address") or ""
                        content_to_check = f"{name} {desc} {addr}".lower()
                        matches_location = False
                        if not locations:
                            matches_location = True
                        else:
                            from graph_rag.config.region_registry import region_registry
                            for loc in locations:
                                loc_norm = normalize_text(loc, strip_punct=True)
                                aliases = [loc_norm]
                                pid = region_registry.get_province_by_alias(loc_norm)
                                if not pid:
                                    pmatches = region_registry.get_province_by_keyword(loc_norm)
                                    pid = pmatches[0] if pmatches else None
                                if pid:
                                    aliases.extend([a.lower() for a in region_registry.get_aliases(pid)])
                                    aliases.extend(region_registry.get_keywords(pid))
                                else:
                                    if loc_norm == "gia lai":
                                        aliases.extend(["gialai", "pleiku"])
                                    elif loc_norm == "binh dinh":
                                        aliases.extend(["binhdinh", "quy nhon", "quynhon"])
                                if any(alias in normalize_text(content_to_check, strip_punct=True) for alias in aliases):
                                    matches_location = True
                                    break
                        if matches_location:
                            base_score = 1.0
                            bonus = 0.0
                            q_norm = normalize_text(user_query, strip_punct=True)
                            _airport_kws = ["bay thang", "chuyen bay", "san bay", "pxu"]
                            for pid in region_registry.get_all_province_ids():
                                _airport_kws.extend(region_registry.get_keywords(pid))
                            for kw in _airport_kws:
                                if kw in q_norm:
                                    bonus += 0.3
                            score = base_score + bonus
                            r["score"] = score
                            results.append(self._record_to_nodeitem(r, "direct_airport_lookup", score))
                    results.sort(key=lambda x: -x.score)
                    if results:
                        logger.info("       Direct airport/transport search found %s seeds.", len(results))
                        return results
                    logger.info("       Direct airport/transport search found 0 seeds, falling through to hybrid search.")
            except (Neo4jClientError, ServiceUnavailable) as exc:
                logger.error("       Direct airport/transport search failed: %s", exc)

        # === Direct Community Retrieval ===
        q_norm_direct = normalize_text(user_query, strip_punct=True)
        is_community_query = (
            (metadata or {}).get("topic") == "community"
            or (metadata or {}).get("fallback_policy") == "community_guided_fallback"
            or (query_plan is not None and getattr(query_plan, "operator", None) == "community_advice_lookup")
            or (
                any(sig in q_norm_direct for sig in ["cong dong", "dien dan", "forum", "nhom du lich", "chia se trai nghiem"])
                and "du lich" in q_norm_direct
            )
        )
        if is_community_query and (metadata or {}).get("classification_contract_active"):
            logger.info("       Community retrieval skipped: classification contract active")
            is_community_query = False
        if is_community_query:
            logger.info("       Executing DIRECT COMMUNITY TRAVELINFO RETRIEVAL...")
            results = []
            try:
                with self.driver.session() as session:
                    cypher = """
                    MATCH (t:TravelInfo)
                    WHERE t.topic = 'community'
                       OR toLower(coalesce(t.name, '')) CONTAINS 'community'
                       OR toLower(coalesce(t.description, '')) CONTAINS 'forum'
                       OR toLower(coalesce(t.description, '')) CONTAINS 'diễn đàn'
                       OR toLower(coalesce(t.description, '')) CONTAINS 'cộng đồng'
                       OR toLower(coalesce(t.description, '')) CONTAINS 'nhóm du lịch'
                    RETURN t.id AS id, t.name AS name, labels(t) AS labels, t.description AS description,
                           t.address AS address, t.topic AS topic,
                           CASE WHEN t.location IS NOT NULL AND toLower(toString(t.location)) STARTS WITH 'point'
                                AND t.location.latitude IS NOT NULL THEN t.location.latitude ELSE toFloat(t.lat) END AS lat,
                           CASE WHEN t.location IS NOT NULL AND toLower(toString(t.location)) STARTS WITH 'point'
                                AND t.location.longitude IS NOT NULL THEN t.location.longitude ELSE toFloat(t.lng) END AS lng
                    LIMIT 5
                    """
                    records = session.run(cypher).data()
                    for r in records:
                        topic = str(r.get("topic") or "").lower()
                        desc_norm = normalize_text(r.get("description") or "", strip_punct=True)
                        name_norm = normalize_text(r.get("name") or "", strip_punct=True)
                        score = 1.2 if topic == "community" else 0.8
                        for kw in ["cong dong", "dien dan", "forum", "nhom du lich", "chia se"]:
                            if kw in desc_norm or kw in name_norm:
                                score += 0.2
                        r["score"] = score
                        results.append(self._record_to_nodeitem(r, "direct_community_lookup", score))
                    results.sort(key=lambda x: -x.score)
                    if results:
                        logger.info("       Direct community TravelInfo search found %s seeds.", len(results))
                        return results
                    logger.info("       Direct community TravelInfo search found 0 seeds, falling through to hybrid search.")
            except (Neo4jClientError, ServiceUnavailable) as exc:
                logger.error("       Direct community TravelInfo search failed: %s", exc)

        # === Entity lookup phase ===
        target_entity = self._primary_entity_name(metadata or {})
        if not target_entity:
            target_entity = self._extract_entity_from_query_text(user_query)
            if target_entity:
                logger.warning("       Fallback entity extraction from query text: '%s'", target_entity)
        current_location = metadata.get("current_location") if metadata else None
        retrieval_allowed_labels = (
            (query_plan.target_labels if query_plan and query_plan.target_labels else None)
            or (metadata or {}).get("retrieval_allowed_labels")
        )
        if not isinstance(retrieval_allowed_labels, list):
            retrieval_allowed_labels = None
        retrieval_policy = (metadata or {}).get("retrieval_policy") or {}
        if not retrieval_policy:
            from graph_rag.core.retrieval_policy import RetrievalPolicy
            intents = [intent]
            if metadata and "intents" in metadata:
                intents = metadata["intents"]
            retrieval_policy = RetrievalPolicy.resolve_policy(
                primary_intent=intent, intents=intents, user_query=user_query,
            ).to_dict()
        blocked_labels = retrieval_policy.get("blocked_labels") or []
        allowed_labels = retrieval_allowed_labels or retrieval_policy.get("allowed_labels")
        if allowed_labels is not None:
            allowed_labels = list(allowed_labels)

        broad_location_norms = _get_admin_region_mapping_service().BROAD_LOCATION_NORMS
        service_intents = {IntentType.FOOD, IntentType.ACCOMMODATION}
        if intent in service_intents:
            target_norm_for_service = normalize_text(str(target_entity or ""), strip_punct=True)
            if target_norm_for_service in broad_location_norms:
                target_entity = ""
                if metadata is not None:
                    metadata["target_entity_policy"] = "broad_location_used_as_filter_only"
                logger.info("       Broad location target ignored for service retrieval; using it only as location filter.")

        force_proximity_anchor = bool((metadata or {}).get("force_proximity_anchor", False))
        grounded_anchor_nodes = (metadata or {}).get("grounded_anchor_nodes", [])
        if not isinstance(grounded_anchor_nodes, list):
            grounded_anchor_nodes = []
        if is_travel_info_contract:
            grounded_anchor_nodes = clean_seeds_of_broad_locations(grounded_anchor_nodes)
        has_specific_entity = self._has_specific_entity_signal(metadata or {})
        hard_location_filter = current_location if self._should_hard_filter_location(metadata or {}) else None
        if (metadata or {}).get("region_lock_mode") == "disabled_multi_anchor_comparison":
            hard_location_filter = None

        logger.info("    Strategy Check: Intent='%s' | Loc='%s' | Entity='%s'", intent, current_location, target_entity)
        _dbg_region_group, _dbg_legacy_province = self._region_filter_params(metadata or {}, query_plan)
        logger.info("       [DEBUG-SEEDS] region_focus=%s | region_group=%s | legacy_province=%s",
                     (query_plan.region_focus if query_plan else (metadata or {}).get("region_focus", "")),
                     _dbg_region_group, _dbg_legacy_province)
        logger.info("       [DEBUG-SEEDS] hard_location_filter=%s | allowed_labels=%s | grounded_anchor_nodes=%d",
                     hard_location_filter, allowed_labels, len(grounded_anchor_nodes))

        # Build exact seeds
        exact_seeds = self._deduplicate_seeds(grounded_anchor_nodes)
        if exact_seeds:
            logger.info("       Using grounded entity anchors: %s", [s.content for s in exact_seeds])

        qf_anchor_names = [
            str(name or "").strip()
            for name in ((metadata or {}).get("query_frame_anchor_names") or [])
            if str(name or "").strip()
        ]
        if qf_anchor_names:
            recovered_anchor_seeds = []
            existing_seed_norms = {
                normalize_text(str(seed.metadata.get("name") or seed.content or ""))
                for seed in exact_seeds
            }
            for anchor_name in qf_anchor_names[:6]:
                if intent in service_intents and normalize_text(anchor_name, strip_punct=True) in broad_location_norms:
                    logger.info("       Skipping QueryFrame anchor recovery for broad location: '%s'", anchor_name)
                    continue
                anchor_norms = {
                    normalize_text(candidate)
                    for candidate in self._entity_lookup_candidates(anchor_name)
                }
                if any(
                    anchor_norm
                    and any(anchor_norm in seed_norm or seed_norm in anchor_norm for seed_norm in existing_seed_norms)
                    for anchor_norm in anchor_norms
                ):
                    continue
                if self._is_category_phrase(anchor_name):
                    logger.info("       Skipping anchor recovery for category phrase: '%s'", anchor_name)
                    continue
                for lookup_name in self._entity_lookup_candidates(anchor_name):
                    nodes = self.entity.exact_match_search(lookup_name)
                    if nodes:
                        if allowed_labels:
                            nodes = [n for n in nodes
                                     if any(lbl in allowed_labels for lbl in n.metadata.get("labels", []))]
                        if blocked_labels:
                            nodes = [n for n in nodes
                                     if not any(lbl in blocked_labels for lbl in n.metadata.get("labels", []))]
                    if not nodes:
                        nodes = self.entity.contains_alias_search(lookup_name, allowed_labels=allowed_labels)
                        if nodes and blocked_labels:
                            nodes = [n for n in nodes
                                     if not any(lbl in blocked_labels for lbl in n.metadata.get("labels", []))]
                    if not nodes:
                        nodes = self.entity.normalized_name_search(lookup_name, allowed_labels=allowed_labels)
                        if nodes and blocked_labels:
                            nodes = [n for n in nodes
                                     if not any(lbl in blocked_labels for lbl in n.metadata.get("labels", []))]
                    if not nodes:
                        region_params = self._region_filter_params(metadata or {}, query_plan)
                        nodes = self.entity.fuzzy_name_search(
                            lookup_name, allowed_labels=allowed_labels,
                            location_filter=None,
                            region_group=region_params[0], legacy_province=region_params[1],
                        )
                        if nodes and blocked_labels:
                            nodes = [n for n in nodes
                                     if not any(lbl in blocked_labels for lbl in n.metadata.get("labels", []))]
                    if nodes:
                        recovered_anchor_seeds.extend(nodes)
                        existing_seed_norms.update(
                            normalize_text(str(node.metadata.get("name") or node.content or ""))
                            for node in nodes
                        )
                        break
            exact_seeds = self._deduplicate_seeds(recovered_anchor_seeds + exact_seeds)

        target_norm = str(target_entity or "").strip().lower()
        exact_has_target = False
        if target_norm and exact_seeds:
            lookup_norms = {name.lower() for name in self._entity_lookup_candidates(target_entity)}
            for seed in exact_seeds:
                seed_name = str(seed.metadata.get("name") or seed.content or "").strip().lower()
                if any(candidate in seed_name for candidate in lookup_norms):
                    exact_has_target = True
                    break

        if target_entity and (not exact_seeds or not exact_has_target):
            if self._is_category_phrase(target_entity):
                logger.info("       Skipping fuzzy search for category phrase: '%s'", target_entity)
                recovered_exact_seeds = []
            else:
                lookup_candidates = self._entity_lookup_candidates(target_entity)
                logger.info("       Executing EXACT MATCH for: '%s'", target_entity)
                recovered_exact_seeds = []
                for lookup_name in lookup_candidates:
                    recovered_exact_seeds = self.entity.exact_match_search(lookup_name)
                    if recovered_exact_seeds:
                        if lookup_name != target_entity:
                            logger.info("       Exact match recovered with alias: '%s'", lookup_name)
                        break
                if not recovered_exact_seeds:
                    logger.info("       Exact match failed. Trying Fuzzy Name Search...")
                    label_filter = retrieval_allowed_labels or None
                    region_params = self._region_filter_params(metadata or {}, query_plan)
                    for lookup_name in lookup_candidates:
                        if self._is_category_phrase(lookup_name):
                            logger.info("       Skipping fuzzy for category phrase: '%s'", lookup_name)
                            continue
                        recovered_exact_seeds = self.entity.fuzzy_name_search(
                            lookup_name, allowed_labels=label_filter,
                            location_filter=hard_location_filter,
                            region_group=region_params[0], legacy_province=region_params[1],
                        )
                        if recovered_exact_seeds:
                            break
                if not recovered_exact_seeds:
                    logger.info("       Fuzzy search failed. Trying normalized contains/alias search...")
                    for lookup_name in lookup_candidates:
                        if self._is_category_phrase(lookup_name):
                            logger.info("       Skipping contains for category phrase: '%s'", lookup_name)
                            continue
                        recovered_exact_seeds = self.entity.contains_alias_search(
                            lookup_name, allowed_labels=label_filter)
                        if recovered_exact_seeds:
                            break
            exact_seeds = self._deduplicate_seeds(recovered_exact_seeds + exact_seeds)
            exact_seeds = clean_seeds_of_broad_locations(exact_seeds)
            if exact_seeds:
                logger.info("       Found Anchor Entity: %s", [s.content for s in exact_seeds])
                if intent == IntentType.ENTITY_FACT:
                    fact_labels = list(retrieval_allowed_labels or [])
                    if "Location" not in fact_labels:
                        fact_labels.append("Location")
                    return self._anchor_compatible_seeds(
                        self._deduplicate_seeds(exact_seeds), fact_labels)

        # Relation-constrained seeds
        relation_seeds = self.tour_plan.query_plan_relation_constrained_seeds(
            user_query=user_query, metadata=metadata or {},
            anchor_seeds=exact_seeds, top_k=max(top_k, TOP_K),
            region_filter_params=self._region_filter_params(metadata or {}, query_plan),
            region_address_aliases_fn=self._region_address_aliases,
        )
        if relation_seeds:
            exact_seeds = self._deduplicate_seeds(exact_seeds + relation_seeds)
            if metadata is not None:
                metadata["query_plan_relation_seed_count"] = len(relation_seeds)
            logger.info("       QueryFrame relation-constrained seeds: +%d %s",
                         len(relation_seeds), [s.content for s in relation_seeds[:6]])

        if has_specific_entity and not exact_seeds and intent not in {IntentType.DISCOVERY, IntentType.TOUR_PLAN}:
            logger.info("       Entity-first guard: specific entity was extracted but no graph anchor was found.")
            logger.info("       Falling through to hybrid search to attempt recovery.")

        # Tour availability
        answer_mode = str((metadata or {}).get("answer_mode") or "").strip()
        question_shape = str((metadata or {}).get("question_shape") or "").strip()
        if answer_mode == "tour_list" or question_shape == "tour_availability":
            logger.info("       Executing TOUR AVAILABILITY RETRIEVAL (Tour-only search)...")
            tour_seeds = self._hybrid_search(
                user_query, metadata, top_k=max(top_k, 20),
                allowed_labels=["Tour"], location_filter=current_location, query_plan=query_plan,
            )
            unique_seeds = self._deduplicate_seeds(exact_seeds + tour_seeds)
            logger.info("       Final Tour Availability Seeds: %s nodes.", len(unique_seeds))
            return unique_seeds

        # TOUR_PLAN multi-pass
        if intent.upper() == IntentType.TOUR_PLAN:
            if bool((metadata or {}).get("agentic_subquery")):
                logger.info("       Executing LIGHTWEIGHT RETRIEVAL for agentic sub-query...")
                lite_seeds = self._hybrid_search(
                    user_query, metadata, top_k=min(top_k, 8),
                    allowed_labels=["TouristAttraction", "Restaurant", "Dish", "Accommodation"],
                    location_filter=current_location, query_plan=query_plan,
                )
                unique_seeds = self._deduplicate_seeds(exact_seeds + lite_seeds)
                logger.info("       Final Agentic Subquery Seeds: %s nodes.", len(unique_seeds))
                return unique_seeds

            logger.info("       Executing MULTI-PASS RETRIEVAL for TravelPlan...")
            search_location_filter = current_location if self._should_hard_filter_location(metadata or {}) else None

            phase1_q = self._rewrite_search_query_for_phase(user_query, "TouristAttraction", metadata)
            logger.info("       Phase 1: Fetching Tourist Attractions (Limit 10)...")
            attraction_seeds = self._hybrid_search(
                phase1_q, metadata, top_k=10,
                allowed_labels=["TouristAttraction"],
                location_filter=search_location_filter, query_plan=query_plan,
            )
            if len(attraction_seeds) < 5:
                region_params = self._region_filter_params(metadata or {}, query_plan)
                graph_attractions = self.entity.fetch_nodes_by_label(
                    "TouristAttraction", limit=15,
                    region_group=region_params[0], legacy_province=region_params[1],
                )
                existing_ids = {s.id for s in attraction_seeds}
                for node in graph_attractions:
                    if node.id not in existing_ids:
                        attraction_seeds.append(node)

            phase2_q = self._rewrite_search_query_for_phase(user_query, "Restaurant", metadata)
            logger.info("       Phase 2: Fetching Restaurants/Food (Limit 5)...")
            food_seeds = self._hybrid_search(
                phase2_q, metadata, top_k=5,
                allowed_labels=["Restaurant", "Dish", "Specialty"],
                location_filter=search_location_filter, query_plan=query_plan,
            )

            combined_seeds = exact_seeds + attraction_seeds + food_seeds

            logger.info("       Phase 1b: Fetching Tours (Limit 5)...")
            tour_seeds = self._hybrid_search(
                user_query, metadata, top_k=5,
                allowed_labels=["Tour"],
                location_filter=search_location_filter, query_plan=query_plan,
            )
            combined_seeds.extend(tour_seeds)

            label_hints = (metadata or {}).get("label_hints") or []
            v3_data = (metadata or {}).get("v3_intent_data") or {}
            if not label_hints:
                label_hints = v3_data.get("label_hints") or []
            accommodation_limit = 5 if "Accommodation" in label_hints else 3
            logger.info("       Phase 3: Fetching Accommodation (Limit %s)...", accommodation_limit)
            accommodation_seeds = self._hybrid_search(
                user_query, metadata, top_k=accommodation_limit,
                allowed_labels=["Accommodation"],
                location_filter=search_location_filter, query_plan=query_plan,
            )
            if len(accommodation_seeds) < 2:
                region_params = self._region_filter_params(metadata or {}, query_plan)
                graph_accommodations = self.entity.fetch_nodes_by_label(
                    "Accommodation", limit=10,
                    region_group=region_params[0], legacy_province=region_params[1],
                )
                existing_ids = {s.id for s in accommodation_seeds}
                for node in graph_accommodations:
                    if node.id not in existing_ids:
                        accommodation_seeds.append(node)

            combined_seeds.extend(accommodation_seeds)
            unique_seeds = self._deduplicate_seeds(combined_seeds)
            trip_days = self._infer_trip_days(metadata or {}, user_query)
            unique_seeds = self.tour_plan.apply_tour_plan_semantic_quotas(
                unique_seeds, metadata=metadata or {}, user_query=user_query,
                top_k=top_k, exact_seeds=exact_seeds, trip_days=trip_days,
            )
            logger.info("       Final TravelPlan Seeds: %s nodes.", len(unique_seeds))
            return unique_seeds

        # === DISCOVERY balanced search ===
        frame = (metadata or {}).get("query_plan") or {}
        is_global_discovery = (
            intent.upper() == IntentType.DISCOVERY
            or frame.get("query_operator") == "global_discovery"
        )
        if is_global_discovery:
            logger.info("       Executing BALANCED DISCOVERY SEARCH for: '%s'", user_query)
            search_location_filter = hard_location_filter
            discovery_search_query = ""
            semantic_category = (metadata or {}).get("semantic_category") or ""
            category_search_terms = {
                "natural_landmark": "thắng cảnh thiên nhiên danh lam thắng cảnh",
                "heritage": "di tích lịch sử di tích văn hóa bảo tàng",
                "cultural_village": "làng văn hóa dân tộc làng dân tộc thiểu số trải nghiệm văn hóa bản địa",
                "spiritual": "chùa tịnh xá nhà thờ",
                "craft": "làng nghề dệt thổ cẩm làng nghề truyền thống thủ công mỹ nghệ",
                "public_space": "quảng trường công viên",
                "cultural_activity": "hoạt động văn hóa lễ hội dân tộc thiểu số văn hóa cộng đồng trải nghiệm văn hóa",
            }
            anchor_names = []
            for anchor in exact_seeds:
                anchor_name = str(getattr(anchor, "content", "") or "").strip()
                if anchor_name:
                    anchor_names.append(anchor_name)
            if not anchor_names:
                for anchor in (metadata or {}).get("grounded_anchor_nodes") or []:
                    if isinstance(anchor, dict):
                        anchor_name = str(anchor.get("name") or "").strip()
                    else:
                        anchor_name = str(getattr(anchor, "content", "") or "").strip()
                    if anchor_name:
                        anchor_names.append(anchor_name)
            if semantic_category in category_search_terms:
                base_terms = category_search_terms[semantic_category]
                if anchor_names:
                    discovery_search_query = f"{' '.join(anchor_names[:2])} {base_terms}"
                else:
                    discovery_search_query = base_terms
            search_keywords = (metadata or {}).get("search_keywords") or []
            if search_keywords and not discovery_search_query:
                discovery_search_query = " ".join(search_keywords[:5])
            if discovery_search_query and ("FOOD" in intent or "hai san" in normalize_text(user_query, strip_punct=True)):
                food_keywords = ["hải sản", "nhà hàng", "quán ăn"]
                q_norm_food = normalize_text(user_query, strip_punct=True)
                for kw in food_keywords:
                    kw_norm = normalize_text(kw, strip_punct=True)
                    if kw_norm not in normalize_text(discovery_search_query, strip_punct=True):
                        discovery_search_query = f"{discovery_search_query} {kw}"
            if not discovery_search_query:
                stop_words = {
                    "tôi", "muốn", "tìm", "hiểu", "về", "các", "có", "những",
                    "nào", "và", "không", "thể", "bỏ", "qua", "nên", "đi",
                    "thì", "là", "ở", "đâu", "như", "thế", "cho", "nổi",
                    "tiếng", "khám", "phá", "nhất",
                }
                words = [w for w in user_query.split() if w.lower() not in stop_words]
                discovery_search_query = " ".join(words[:6])
            if discovery_search_query and not (metadata or {}).get("location_from_gps"):
                location_context = ""
                legacy_province = (metadata or {}).get("legacy_province") or ""
                if legacy_province and legacy_province.lower() not in discovery_search_query.lower():
                    location_context = legacy_province
                elif anchor_names and anchor_names[0].lower() not in discovery_search_query.lower():
                    location_context = anchor_names[0]
                if location_context:
                    discovery_search_query = f"{location_context} {discovery_search_query}"

            per_label_k = max(5, top_k)
            primary_intent = str(
                (metadata or {}).get("original_intent")
                or (metadata or {}).get("intent")
                or ""
            ).upper()
            intent_label_map = {
                IntentType.ACCOMMODATION: ["Accommodation"],
                IntentType.FOOD: ["Restaurant", "Dish"],
                IntentType.EVENT: ["Event"],
                IntentType.TOURISM: ["TouristAttraction"],
            }
            primary_labels = intent_label_map.get(primary_intent)
            if primary_labels:
                primary_set = set(primary_labels)
                secondary_groups = [
                    g for g in [["TouristAttraction"], ["Restaurant"], ["Accommodation"], ["Event"], ["TravelInfo"]]
                    if not primary_set.intersection(g)
                ]
                discovery_label_groups = [primary_labels] + secondary_groups
                primary_k = per_label_k
                secondary_k = max(3, per_label_k // 2)
            else:
                allowed = set(
                    (query_plan.target_labels if query_plan and query_plan.target_labels else None)
                    or (metadata or {}).get("retrieval_allowed_labels") or []
                )
                _ALL_GROUPS = [
                    ("TouristAttraction", ["TouristAttraction"]),
                    ("Restaurant", ["Restaurant"]),
                    ("Accommodation", ["Accommodation"]),
                    ("Event", ["Event"]),
                    ("TravelInfo", ["TravelInfo"]),
                ]
                discovery_label_groups = [group for label, group in _ALL_GROUPS if label in allowed]
                if not discovery_label_groups:
                    discovery_label_groups = [["TouristAttraction"], ["Location"]]
                primary_k = per_label_k
                secondary_k = per_label_k
            balanced_seeds = []
            for idx, label_group in enumerate(discovery_label_groups):
                group_k = primary_k if idx == 0 else secondary_k
                group_seeds = self._hybrid_search(
                    discovery_search_query, metadata, top_k=group_k,
                    allowed_labels=label_group, location_filter=search_location_filter,
                    query_plan=query_plan,
                )
                balanced_seeds.extend(group_seeds)

            if balanced_seeds:
                detected_loc = str((metadata or {}).get("detected_location") or "").strip()
                filter_province = ""
                if detected_loc:
                    filter_province = self._extract_province_from_text(
                        normalize_text(detected_loc, strip_punct=True))
                if not filter_province and exact_seeds:
                    filter_province = self._extract_province(exact_seeds[0])
                if filter_province:
                    before_count = len(balanced_seeds)
                    balanced_seeds = [seed for seed in balanced_seeds
                                      if self._seed_matches_province(seed, filter_province)]
                    removed = before_count - len(balanced_seeds)
                    if removed:
                        logger.info("       Location filter: removed %s seeds not in '%s'", removed, filter_province)
            if balanced_seeds:
                month_range = self.event.extract_month_range(user_query)
                if month_range:
                    before_count = len(balanced_seeds)
                    balanced_seeds = self.event.filter_events_by_month(balanced_seeds, month_range)
                    removed = before_count - len(balanced_seeds)
                    if removed:
                        logger.info("       Month filter: removed %s events not in months %s",
                                     removed, sorted(month_range))
            if balanced_seeds and "FOOD" in intent:
                q_norm = normalize_text(user_query, strip_punct=True)
                food_category = self.food.detect_food_category(q_norm)
                if food_category:
                    before_count = len(balanced_seeds)
                    balanced_seeds = self.food.filter_by_food_category(balanced_seeds, food_category)
                    removed = before_count - len(balanced_seeds)
                    if removed:
                        logger.info("       Food category filter: removed %s non-%s restaurants",
                                     removed, food_category)
            unique_seeds = self._deduplicate_seeds(exact_seeds + balanced_seeds)
            logger.info("       Final Discovery Seeds: %s nodes.", len(unique_seeds))
            return unique_seeds

        # === Proximity anchor search ===
        anchor_nodes = (metadata or {}).get("grounded_anchor_nodes", [])
        target_labels_for_intent = retrieval_allowed_labels or self.INTENT_TO_LABELS.get(intent, [])
        non_target_anchors = [
            n for n in anchor_nodes
            if not any(lbl in target_labels_for_intent
                       for lbl in (n.metadata.get("labels") or [n.metadata.get("type", "")]))
        ]
        proximity_intents = {IntentType.ACCOMMODATION, IntentType.FOOD, IntentType.TOURISM}
        if non_target_anchors and (intent in proximity_intents or force_proximity_anchor):
            logger.info("       Executing PROXIMITY ANCHOR SEARCH from: %s",
                         [n.content for n in non_target_anchors])
            proximity_seeds = self.tour_plan.proximity_anchor_search(
                non_target_anchors, target_labels_for_intent, top_k=top_k,
            )
            if proximity_seeds:
                logger.info("       Found %s proximity seeds near anchor(s).", len(proximity_seeds))
                topup_seeds = self._hybrid_search(
                    user_query, metadata, max(2, top_k // 2),
                    allowed_labels=target_labels_for_intent,
                    location_filter=hard_location_filter, query_plan=query_plan,
                )
                _combined = self._deduplicate_seeds(exact_seeds + proximity_seeds + topup_seeds)
                return self._filter_seeds_by_ward_location(_combined, metadata)
            else:
                logger.info("       No proximity seeds found; continuing to standard search.")

        # === Standard hybrid search ===
        allowed_labels = retrieval_allowed_labels or self.INTENT_TO_LABELS.get(intent, None)
        _search_query = user_query
        if intent == IntentType.EVENT and len(user_query) > 60:
            _search_query = self.event.shorten_event_query(user_query)
            if _search_query != user_query:
                logger.info("         ⚙️ Shortened query: '%s'", _search_query)
        logger.info("       Executing STANDARD HYBRID SEARCH for: '%s'", _search_query)
        discovery_seeds = self._hybrid_search(
            _search_query, metadata, top_k,
            allowed_labels=allowed_labels,
            location_filter=hard_location_filter, query_plan=query_plan,
        )
        logger.info("       [DEBUG-SEEDS] STANDARD HYBRID returned %d seeds", len(discovery_seeds or []))

        if not discovery_seeds and hard_location_filter:
            logger.info("         ⚙️ Retrying without location filter (initial search returned 0 results)")
            discovery_seeds = self._hybrid_search(
                user_query, metadata, top_k,
                allowed_labels=allowed_labels, location_filter=None, query_plan=query_plan,
            )

        # Text-to-Cypher
        _has_target_label = any(
            (s.metadata or {}).get("label") in (allowed_labels or [])
            for s in (discovery_seeds or [])
        )
        from graph_rag.config import ENABLE_TEXT_TO_CYPHER
        _t2c_enabled = ENABLE_TEXT_TO_CYPHER and (metadata or {}).get("enable_text_to_cypher", True)
        if _t2c_enabled and (not discovery_seeds or not _has_target_label):
            try:
                from graph_rag.modules.retrieval.text_to_cypher import TextToCypherRetriever
                from graph_rag.services.ai_model import LLMService
                _t2c_llm = LLMService(model_name="deepseek-chat")
                _t2c = TextToCypherRetriever(self.driver, _t2c_llm)
                _t2c_location = legacy_province or hard_location_filter or ""
                _t2c_location_aliases = []
                _has_specific_city = any(
                    isinstance(e, dict) and str(e.get("admin_level") or "").lower() in ("city", "district")
                    for e in (metadata or {}).get("entities") or []
                )
                if _t2c_location and not _has_specific_city:
                    try:
                        _admin_svc = _get_admin_region_mapping_service()
                        _region_focus = str((metadata or {}).get("region_focus") or "").strip()
                        if _region_focus:
                            _t2c_location_aliases = _admin_svc.get_merged_province_names(_region_focus)
                            if _t2c_location_aliases and len(_t2c_location_aliases) > 1:
                                logger.info("         ⚙️ Merged province search: %s → %s",
                                             _t2c_location, _t2c_location_aliases)
                    except (Neo4jClientError, ServiceUnavailable) as e:
                        logger.debug("         ⚙️ Admin region mapping unavailable: %s", e)
                _t2c_is_follow_up = bool((metadata or {}).get("is_follow_up", False))
                _t2c_exclude = (metadata or {}).get("answered_entities") or []
                _t2c_search_query = _search_query
                _t2c_max_results = top_k
                if _t2c_location_aliases and len(_t2c_location_aliases) > 1:
                    _extra_provinces = [a for a in _t2c_location_aliases if a != _t2c_location]
                    if _extra_provinces:
                        _t2c_search_query = f"{_search_query} (bao gồm cả {', '.join(_extra_provinces)})"
                        _t2c_max_results = top_k * len(_t2c_location_aliases)
                _t2c_results = _t2c.retrieve(
                    _t2c_search_query, intent=str(intent), max_results=_t2c_max_results,
                    allowed_labels=allowed_labels, location=_t2c_location,
                    is_follow_up=_t2c_is_follow_up, exclude_entities=_t2c_exclude,
                    location_aliases=_t2c_location_aliases or None,
                )
                if _t2c_results:
                    logger.info("         ⚙️ Text-to-Cypher found %s results", len(_t2c_results))
                    _t2c_results = self._filter_seeds_by_ward_location(_t2c_results, metadata)
                    discovery_seeds = (discovery_seeds or []) + _t2c_results
                    discovery_seeds = self._deduplicate_seeds(discovery_seeds)
                else:
                    logger.info("         ⚙️ Text-to-Cypher returned 0 results")
            except (Neo4jClientError, ServiceUnavailable) as e:
                logger.error("         ⚙️ Text-to-Cypher failed: %s", e)

        # Event fallback
        has_event_seeds = any(
            (s.metadata or {}).get("label") == "Event"
            or "Event" in str((s.metadata or {}).get("labels") or "")
            for s in (discovery_seeds or [])
        )
        _admin_match = (metadata or {}).get("admin_region_match") or {}
        _event_location = (
            current_location
            or (metadata or {}).get("detected_location")
            or (_admin_match.get("matched_alias") if isinstance(_admin_match, dict) else "")
            or ""
        )
        if intent == IntentType.EVENT and (not discovery_seeds or not has_event_seeds) and _event_location:
            month_range = self.event.extract_month_range(user_query)
            year = self.event.extract_year(user_query)
            query_norm = normalize_text(user_query, strip_punct=True)
            event_category_filter = self.event.detect_event_category(query_norm)
            logger.warning("         ⚙️ EVENT fallback: searching Event nodes by HELD_AT→LOCATED_IN('%s')", _event_location)
            discovery_seeds = self.event.search_events_by_location(
                _event_location, top_k, month_range=month_range or None,
                year=year, category_filter=event_category_filter or None,
            )
            if discovery_seeds:
                logger.warning("         ⚙️ EVENT fallback found %s Event nodes", len(discovery_seeds))

        # Label-relaxation fallback
        if not discovery_seeds and allowed_labels:
            relaxed_labels = retrieval_policy.get("relax_labels")
            if not isinstance(relaxed_labels, list):
                all_db_labels = ["Event", "TouristAttraction", "Restaurant", "Accommodation", "Dish", "Tour"]
                relaxed_labels = [lbl for lbl in all_db_labels if lbl not in allowed_labels]
            else:
                relaxed_labels = [lbl for lbl in relaxed_labels if lbl not in allowed_labels]
            if relaxed_labels:
                logger.warning("         ⚙️ Retrying with relaxed fallback labels: %s", relaxed_labels)
                discovery_seeds = self._hybrid_search(
                    user_query, metadata, top_k,
                    allowed_labels=relaxed_labels, location_filter=hard_location_filter,
                    query_plan=query_plan,
                )
                if not discovery_seeds and hard_location_filter:
                    logger.warning("         ⚙️ Retrying relaxed fallback labels without location filter")
                    discovery_seeds = self._hybrid_search(
                        user_query, metadata, top_k,
                        allowed_labels=relaxed_labels, location_filter=None,
                        query_plan=query_plan,
                    )

        # Post-retrieval month filter
        if discovery_seeds:
            month_range = self.event.extract_month_range(user_query)
            if month_range:
                before_count = len(discovery_seeds)
                discovery_seeds = self.event.filter_events_by_month(discovery_seeds, month_range)
                removed = before_count - len(discovery_seeds)
                if removed:
                    logger.info("         ⚙️ Month filter: removed %s events not in months %s",
                                 removed, sorted(month_range))

        discovery_seeds = self._filter_seeds_by_ward_location(discovery_seeds, metadata)
        _final = self._deduplicate_seeds(exact_seeds + discovery_seeds)
        logger.info("       [DEBUG-SEEDS] FINAL return: exact=%d + discovery=%d = %d unique seeds",
                     len(exact_seeds), len(discovery_seeds), len(_final))
        return _final
