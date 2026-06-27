from typing import Dict, Any, List, Tuple
import copy
import hashlib
import os
import re
import logging
import threading
import time

from graph_rag.services.database import Neo4jService
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
from graph_rag.core.intents import IntentType, RegionFocus
from graph_rag.config import (
    AGENTIC_MAX_ITERATIONS,
    AGENTIC_MAX_SUB_QUERIES,
    PIPELINE_LLM_MODEL_NAME,
    QUERY_ANALYZER_LLM_MODEL_NAME,
    RELATIONSHIP_MAP,
    GRAPH_RAG_V3_MAX_FACTS_PER_ANCHOR,
)
from graph_rag.config.constants import NON_GROUNDABLE_ENTITY_TYPES
from graph_rag.utils.geo import haversine_km
from graph_rag.utils.node_utils import get_node_labels
from graph_rag.services.ai_model import LLMService
from graph_rag.services.directions_service import DirectionsService
from graph_rag.modules.retrieval import SeedRetriever, AgenticRetriever
from graph_rag.utils.text import normalize_text
from graph_rag.core.keywords import TIME_RANGE_KEYWORDS, TRANSPORT_NEGATIVE_SIGNALS
from graph_rag.modules.graph import GraphTraverser
from graph_rag.modules.generation.llm_client import AnswerGenerator
from graph_rag.modules.query_analysis.analyzer import QueryAnalyzer
from graph_rag.modules.tour_plan import TourRouteOptimizerService
from graph_rag.modules.pipeline_support import (
    LocationGroundingService,
    DistanceIntentService,
    AdminRegionMappingService,
)
from graph_rag.modules.context.structured_context_builder import StructuredContextBuilder
from graph_rag.modules.generation.structured_generator import StructuredAnswerGenerator
from graph_rag.modules.query_planning.intent_router import IntentRouter
from graph_rag.modules.retrieval.multi_anchor_retriever import MultiAnchorRetriever
from graph_rag.modules.validation.completeness_gate import CompletenessGate
from graph_rag.core import thresholds, keywords
from graph_rag.config import cfg


PIPELINE_LOGGER = logging.getLogger("graph_rag.pipeline.timing")


class RAGPipeline:
    TOUR_PLAN_MAX_HOP_KM = thresholds.TOUR_PLAN_MAX_HOP_KM
    WALKING_MAX_HOP_KM = thresholds.WALKING_MAX_HOP_KM
    SENIOR_FAMILY_MAX_HOP_KM = thresholds.SENIOR_FAMILY_MAX_HOP_KM

    INLAND_GIA_LAI_BOUNDS = {
        "lat_min": thresholds.INLAND_LAT_MIN,
        "lat_max": thresholds.INLAND_LAT_MAX,
        "lng_min": thresholds.INLAND_LNG_MIN,
        "lng_max": thresholds.INLAND_LNG_MAX,
    }

    COASTAL_QUY_NHON_BOUNDS = {
        "lat_min": thresholds.COASTAL_LAT_MIN,
        "lat_max": thresholds.COASTAL_LAT_MAX,
        "lng_min": thresholds.COASTAL_LNG_MIN,
        "lng_max": thresholds.COASTAL_LNG_MAX,
    }

    COASTAL_KEYWORDS = keywords.COASTAL_KEYWORDS
    INLAND_KEYWORDS = keywords.INLAND_KEYWORDS

    logger = logging.getLogger("graph_rag.route_optimizer")
    LOCATION_SOURCE_CONFIDENCE = thresholds.LOCATION_SOURCE_CONFIDENCE

    def __init__(
        self,
        embedding_service,
        llm_api_key=None,
        llm_model_name=None,
        query_analyzer_llm_api_key=None,
        query_analyzer_llm_model_name=None,
    ):
        self.driver = Neo4jService.get_driver()
        self.embedding_service = embedding_service
        self._config = cfg

        # Thread safety for shared state
        self._state_lock = threading.Lock()

        # State (must be initialized before services that depend on it)
        self.conversation_state = {
            "current_location": "",
            "history": [],
            "entity_memory": [],
            "last_grounded_anchor": {},
            "user_geo_location": (os.getenv("USER_GEO_LOCATION", "") or "").strip(),
        }
        
        # Init Services
        self.llm_service = LLMService(api_key=llm_api_key, model_name=llm_model_name or PIPELINE_LLM_MODEL_NAME)
        self.query_llm_service = LLMService(
            api_key=query_analyzer_llm_api_key,
            model_name=query_analyzer_llm_model_name or QUERY_ANALYZER_LLM_MODEL_NAME,
        )

        # Init Modules
        self.query_analyzer = QueryAnalyzer(self.query_llm_service) 
        self.retriever = SeedRetriever(self.driver, self.embedding_service)
        self.directions_service = DirectionsService()
        # Hỗ trợ truy xuất có hướng (agentic retrieval) cho các truy vấn phức tạp cần nhiều bước suy luận và truy xuất phụ
        self.agentic_retriever = AgenticRetriever(
            base_retriever=self.retriever,
            llm_service=self.query_llm_service,
            max_iterations=AGENTIC_MAX_ITERATIONS,
            max_sub_queries=AGENTIC_MAX_SUB_QUERIES,
        )
        self.traverser = GraphTraverser(self.driver)
        self.generator = AnswerGenerator(self.llm_service)
        self.intent_router = IntentRouter(
            llm_service=self.query_llm_service,
            enable_llm=False,
        )
        self.multi_anchor_retriever = MultiAnchorRetriever(
            retriever=self.retriever,
            traverser=self.traverser,
            max_facts_per_anchor=GRAPH_RAG_V3_MAX_FACTS_PER_ANCHOR,
        )
        self.completeness_gate = CompletenessGate()
        self.structured_context_builder = StructuredContextBuilder()
        self.structured_answer_generator = StructuredAnswerGenerator(self.llm_service)
        # Dịch vụ tối ưu hóa lộ trình du lịch cho các truy vấn có intent TOUR_PLAN
        self.tour_route_optimizer = TourRouteOptimizerService(
            self.driver,
            logger=self.logger,
            haversine_fn=haversine_km,
            tour_plan_max_hop_km=self.TOUR_PLAN_MAX_HOP_KM,
            walking_max_hop_km=self.WALKING_MAX_HOP_KM,
            senior_family_max_hop_km=self.SENIOR_FAMILY_MAX_HOP_KM,
        )
        # Hỗ trợ duy trì ngữ cảnh
        self.location_grounding_service = LocationGroundingService(
            driver=self.driver,
            logger=self.logger,
            retriever=self.retriever,
            conversation_state=self.conversation_state,
            coastal_keywords=self.COASTAL_KEYWORDS,
            inland_keywords=self.INLAND_KEYWORDS,
            coastal_bounds=self.COASTAL_QUY_NHON_BOUNDS,
            inland_bounds=self.INLAND_GIA_LAI_BOUNDS,
            location_source_confidence=self.LOCATION_SOURCE_CONFIDENCE,
        )
        # Hỗ trợ xử lý intent khoảng cách và chỉ đường
        self.distance_intent_service = DistanceIntentService(
            logger=self.logger,
            retriever=self.retriever,
            directions_service=self.directions_service,
            haversine_fn=haversine_km,
        )
        self.admin_region_mapping_service = AdminRegionMappingService()

    def close(self):
        Neo4jService.close_driver()

    def _intent_equals(self, intent: str, target: str) -> bool:
        return str(intent or "").upper() == str(target or "").upper()

    def _infer_location_from_grounded_nodes(self, grounded_nodes: List[Any]) -> str:
        return self.location_grounding_service.infer_location_from_grounded_nodes(grounded_nodes)

    def _query_region_signal(self, user_query: str, entities: List[Dict[str, Any]]) -> str:
        return self.location_grounding_service.query_region_signal(user_query, entities)

    def _location_to_region_focus(self, location_text: str) -> str:
        return self.location_grounding_service.location_to_region_focus(location_text)

    # Admin level classification for location scope guard — loaded from config

    def _is_broad_admin_location(self, location_norm: str) -> bool:
        """Check if location is province or district level (broad scope).

        Uses graph admin_level property (primary) with keyword fallback.
        """
        if not location_norm:
            return False
        # Keyword fallback (fast, no DB query)
        for kw in self._config.province_keywords():
            if kw in location_norm:
                return True
        for kw in self._config.district_keywords():
            if kw in location_norm:
                return True
        # Graph-based: query admin_level from Neo4j
        try:
            driver = Neo4jService.get_driver()
            with driver.session() as session:
                result = session.run(
                    "MATCH (l:Location) WHERE toLower(l.name) = $name "
                    "RETURN l.admin_level AS admin_level LIMIT 1",
                    name=location_norm,
                )
                record = result.single()
                if record and record.get("admin_level") in ("province", "area"):
                    return True
        except (Neo4jClientError, ServiceUnavailable):
            pass  # Fallback to keyword-based
        return False

    def _is_narrow_admin_location(self, location_norm: str) -> bool:
        """Check if location is ward/commune level (narrow scope).

        Uses graph admin_level property (primary) with keyword fallback.
        """
        if not location_norm:
            return False
        # Keyword fallback
        for kw in self._config.ward_keywords():
            if kw in location_norm:
                return True
        # Graph-based: query admin_level from Neo4j
        try:
            driver = Neo4jService.get_driver()
            with driver.session() as session:
                result = session.run(
                    "MATCH (l:Location) WHERE toLower(l.name) = $name "
                    "RETURN l.admin_level AS admin_level LIMIT 1",
                    name=location_norm,
                )
                record = result.single()
                if record and record.get("admin_level") == "ward":
                    return True
        except (Neo4jClientError, ServiceUnavailable):
            pass
        # Heuristic fallback
        if not self._is_broad_admin_location(location_norm):
            return len(location_norm.split()) <= 3
        return False

    def _build_location_context(
        self,
        name: str,
        source: str,
        reason: str = "",
        confidence: float = None,
    ) -> Dict[str, Any]:
        return self.location_grounding_service.build_location_context(
            name=name,
            source=source,
            reason=reason,
            confidence=confidence,
        )

    def _choose_location_context(
        self,
        old_ctx: Dict[str, Any],
        new_ctx: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        return self.location_grounding_service.choose_location_context(old_ctx, new_ctx)

    def _build_initial_location_context(
        self,
        current_location: str,
        history: List[Dict],
        analyzer_output: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self.location_grounding_service.build_initial_location_context(current_location, history, analyzer_output=analyzer_output)

    def _has_explicit_location(self, _query: str, entities: List[Dict[str, Any]]) -> bool:
        return self.location_grounding_service.has_explicit_location(entities)

    def _clear_conversation_context(self, new_location: str = "") -> None:
        self.location_grounding_service.clear_conversation_context(new_location=new_location)

    def _infer_grounded_location_context(self, grounded_nodes: List[Any]) -> Dict[str, Any]:
        return self.location_grounding_service.infer_grounded_location_context(grounded_nodes)

    def _extract_anchor_location_from_history(self, history: List[Dict]) -> str:
        return self.location_grounding_service.extract_anchor_location_from_history(history)

    # Sửa chính tả
    def _normalize_known_location_typos(self, query: str) -> str:
        from graph_rag.utils.text import clean_query_format
        text = clean_query_format(str(query or ""))
        if not text:
            return ""
        normalized = text
        for typo, fixed in keywords.TYPO_NORMALIZATION.items():
            normalized = re.sub(rf"\b{re.escape(typo)}\b", fixed, normalized, flags=re.IGNORECASE)
        return normalized

    def _entity_memory_lookup(self, entity_type: str) -> str:
        return self.location_grounding_service._entity_memory_lookup(entity_type)

    def _remember_entities(self, entities: List[Dict[str, Any]]) -> None:
        self.location_grounding_service.remember_entities(entities)

    def _update_conversation_state(
        self,
        history: List[Dict],
        user_query: str,
        answer: str,
        location: str,
        entities: List[Dict[str, Any]],
        last_grounded_anchor: Dict[str, Any] = None,
        is_follow_up: bool = False,
        intent: str = "",
        target_class: str = "",
        answer_mode: str = "",
        region_focus: str = "",
        semantic_category: str = "",
    ) -> None:
        with self._state_lock:
            # Auto-detect from conversation_state if not explicitly passed
            if not is_follow_up:
                is_follow_up = bool(self.location_grounding_service.conversation_state.get("current_is_follow_up", False))
            self.location_grounding_service.update_conversation_state(
                history=history,
                user_query=user_query,
                answer=answer,
                location=location,
                entities=entities,
                last_grounded_anchor=last_grounded_anchor,
                is_follow_up=is_follow_up,
                intent=intent,
                target_class=target_class,
                answer_mode=answer_mode,
                region_focus=region_focus,
                semantic_category=semantic_category,
            )

    def reset_conversation_state(self, preserve_user_geo: bool = True) -> None:
        """Reset all per-question conversation state for isolated batch evaluation."""
        with self._state_lock:
            user_geo = (self.conversation_state.get("user_geo_location") or "").strip() if preserve_user_geo else ""
            self.conversation_state.clear()
            self.conversation_state.update({
                "current_location": "",
                "history": [],
                "entity_memory": [],
                "last_grounded_anchor": {},
                "user_geo_location": user_geo,
            })

    def _node_region(self, node: Any) -> str:
        return self.location_grounding_service._node_region(node)

    def _enforce_grounding_region_consistency(
        self,
        grounded_nodes: List[Any],
        user_query: str,
        entities: List[Dict[str, Any]],
    ) -> tuple[List[Any], str]:
        return self.location_grounding_service.enforce_grounding_region_consistency(
            grounded_nodes,
            user_query=user_query,
            entities=entities,
        )

    def _query_has_duration_signal(self, user_query: str) -> bool:
        q = normalize_text(user_query, strip_punct=True)
        if not q:
            return False
        if re.search(r"\b\d+\s*n\s*\d+\s*d\b", q):
            return True
        if re.search(r"\b\d+\s*(?:ngay|nay|ngy)\b", q) or re.search(r"\b\d+\s*dem\b", q):
            return True
        # "lịch trình" alone is not enough — must co-occur with planning context
        if any(token in q for token in keywords.TOUR_PLAN_SIGNALS):
            return True
        # "lich trinh" needs duration or planning co-signal
        if "lich trinh" in q:
            # Reject if it's about "lịch sử" (history) or analysis context
            if any(tok in q for tok in ["lich su", "phan tich", "giai thich", "tai sao", "vi tri chien"]):
                return False
            # Accept if there's a duration co-signal
            if re.search(r"\d+\s*(?:ngay|nay|ngy)", q) or "goi y" in q or "len" in q or "lap" in q:
                return True
            return False
        return False

    def _is_fact_verification_query(self, user_query: str) -> bool:
        q = normalize_text(user_query, strip_punct=True)
        if not q:
            return False
        if re.search(r"\b(?:nhung|cac)\b.+\bnao\b", q):
            return False
        if any(signal in q for signal in ["bao gom co nhung", "co nhung", "khac khong"]):
            return False

        has_fact_signal = any(s in q for s in keywords.FACT_VERIFICATION_SIGNALS)
        if not has_fact_signal:
            has_fact_signal = bool(re.search(r"\bdo\s+.+\s+to\s+chuc\b", q))

        planning_signals = [
            "goi y",
            "lap lich",
            "lich trinh",
            "ke hoach",
            "nen di",
            "di dau",
        ]
        has_planning_signal = any(s in q for s in planning_signals)

        return has_fact_signal and not has_planning_signal

    # Bắt ý định tạo tour

    def _query_has_distance_signal(self, user_query: str) -> bool:
        q = normalize_text(user_query, strip_punct=True)
        if not q:
            return False
        # Negative signals: analysis/strategy questions are NOT distance queries
        if any(sig in q for sig in keywords.ANALYSIS_SIGNALS):
            return False
        # Negative signals: transport/airport questions are NOT distance queries
        if any(sig in q for sig in TRANSPORT_NEGATIVE_SIGNALS):
            return False
        if any(token in q for token in keywords.DISTANCE_SIGNALS):
            return True
        m = re.search(r"\btu\s+.+\s+(toi|den)\s+.+", q)
        if m:
            # Reject if the matched segment contains time-range keywords
            # (e.g. "tu thang 6 den thang 9" → time range, not distance)
            segment = m.group(0)
            if any(kw in segment for kw in TIME_RANGE_KEYWORDS):
                return False
            return True
        return False

    def _is_multi_intent_travel_query(self, intents: List[str], user_query: str) -> bool:
        normalized_intents = {str(x).upper() for x in (intents or [])}
        has_stay = IntentType.ACCOMMODATION in normalized_intents
        has_play = IntentType.TOURISM in normalized_intents
        has_food = IntentType.FOOD in normalized_intents
        if (has_stay and has_play) or (has_stay and has_food) or (has_play and has_food):
            return True

        q = normalize_text(user_query, strip_punct=True)
        stay_kw = any(k in q for k in ["khach san", "homestay", "noi o", "luu tru", "resort"])
        play_kw = any(k in q for k in ["diem choi", "choi", "tham quan", "check in", "di dau"])
        food_kw = any(k in q for k in ["an gi", "nha hang", "quan an", "dac san"])
        return (stay_kw and play_kw) or (stay_kw and food_kw) or (play_kw and food_kw)

    def _derive_retrieval_allowed_labels(
        self,
        primary_intent: str,
        intents: List[str],
        user_query: str,
    ) -> List[str]:
        from graph_rag.core.retrieval_policy import RetrievalPolicy
        policy = RetrievalPolicy.resolve_policy(primary_intent, intents, user_query)
        return policy.allowed_labels

    #Chốt lại ý định chính của người dùng để ưu tiên truy xuất và sinh câu trả lời
    def _select_primary_intent(self, intents: List[str], user_query: str = "") -> str:
        """Select primary intent with explicit priority for routing-critical intents."""
        intent_list = intents or [IntentType.DISCOVERY]

        if self._is_fact_verification_query(user_query):
            return IntentType.ENTITY_FACT

        if self._query_has_duration_signal(user_query):
            normalized = {str(x).upper() for x in intent_list}
            travel_family = {
                IntentType.TOUR_PLAN,
                IntentType.ACCOMMODATION,
                IntentType.FOOD,
                IntentType.TOURISM,
                IntentType.DISCOVERY,
            }
            q_lower = normalize_text(user_query, strip_punct=True)
            has_planning_kw = any(kw in q_lower for kw in ["du lich", "lich trinh", "chuyen di", "goi y", "tour", "kham pha", "trinh"])
            if normalized.intersection({str(x).upper() for x in travel_family}):
                if "DISCOVERY_SEARCH" in normalized and not has_planning_kw:
                    pass
                else:
                    return IntentType.TOUR_PLAN

        if self._query_has_distance_signal(user_query):
            return IntentType.DISTANCE

        # Routing-critical intents must win even if analyzer places them later.
        priority_order = [
            IntentType.DISTANCE,
            IntentType.TOUR_PLAN,
            IntentType.TRAVEL_ADVICE,
            IntentType.ACCOMMODATION,
            IntentType.FOOD,
            IntentType.EVENT,
            IntentType.TOURISM,
            IntentType.ENTITY_FACT,
            IntentType.DISCOVERY,
        ]

        normalized = {str(x).upper(): x for x in intent_list}
        for candidate in priority_order:
            key = str(candidate).upper()
            if key in normalized:
                return normalized[key]

        # Fallback: return first intent only if it's a valid taxonomy intent
        if intent_list:
            raw = str(intent_list[0]).strip().upper()
            if IntentType.is_valid(raw):
                return raw
        return IntentType.DISCOVERY

    def _grounded_node_priority_score(self, node: Any, allowed_labels: set, primary_labels: set = None) -> Tuple[int, float]:
        labels = set(get_node_labels(node))
        label_hit = 0
        if allowed_labels and labels.intersection(allowed_labels):
            label_hit = 1
        if primary_labels and labels.intersection(primary_labels):
            label_hit = 2  # Higher boost for primary labels
        has_coords = 1 if (
            hasattr(node, "metadata")
            and node.metadata.get("lat") is not None
            and node.metadata.get("lng") is not None
        ) else 0
        node_score = 0.0
        if getattr(node, "score", None) is not None:
            try:
                node_score = float(node.score)
            except (ValueError, TypeError):
                node_score = 0.0
        return (label_hit * 100 + has_coords * 10, node_score)

    def _filter_grounded_nodes_for_intent(
        self,
        grounded_nodes: List[Any],
        primary_intent: str,
        allowed_labels_override: List[str] = None,
        entities: List[Dict[str, Any]] = None,
    ) -> List[Any]:
        if not grounded_nodes:
            return grounded_nodes

        top_k = self._config.grounded_topk_for_intent(str(primary_intent or ""))
        if len(grounded_nodes) <= top_k:
            return grounded_nodes

        if allowed_labels_override:
            allowed_labels = set(allowed_labels_override)
        else:
            allowed_labels = set((self.retriever.INTENT_TO_LABELS or {}).get(primary_intent, []))
        labeled_hits = [
            node for node in grounded_nodes
            if not allowed_labels or set(get_node_labels(node)).intersection(allowed_labels)
        ]

        # If strict label filter is too sparse, keep original pool to avoid over-pruning.
        candidate_pool = labeled_hits if len(labeled_hits) >= max(3, top_k // 3) else grounded_nodes
        from graph_rag.core.retrieval_policy import RetrievalPolicy
        policy = RetrievalPolicy.resolve_policy(primary_intent, [primary_intent], "")
        primary_labels = set(policy.primary_labels)
        ranked = sorted(
            candidate_pool,
            key=lambda n: self._grounded_node_priority_score(n, allowed_labels, primary_labels),
            reverse=True,
        )

        entity_names = []
        for e in (entities or []):
            if not isinstance(e, dict):
                continue
            e_name = str(e.get("name") or "").strip()
            e_type = str(e.get("type") or "").lower()
            if not e_name or e_name.isdigit() or e_type in NON_GROUNDABLE_ENTITY_TYPES:
                continue
            entity_names.append(normalize_text(e_name, strip_punct=True))

        pinned = []
        pinned_ids = set()
        if entity_names:
            for node in ranked:
                node_name = normalize_text(str(node.metadata.get("name") or node.content or ""), strip_punct=True)
                if any(en in node_name or node_name in en for en in entity_names):
                    nid = str(getattr(node, "id", ""))
                    if nid and nid not in pinned_ids:
                        pinned.append(node)
                        pinned_ids.add(nid)

        result = list(pinned)
        for node in ranked:
            if len(result) >= top_k:
                break
            nid = str(getattr(node, "id", ""))
            if nid and nid in pinned_ids:
                continue
            result.append(node)

        return result[:top_k]

    def _extract_recent_mention(self, history: List[Dict], patterns: List[str]) -> str:
        return self.location_grounding_service.extract_recent_mention(history, patterns)

    def _resolve_generic_entities_with_history(
        self,
        entities: List[Dict[str, Any]],
        history: List[Dict],
        intent: str,
        entity_memory: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        return self.location_grounding_service.resolve_generic_entities_with_history(
            entities,
            history,
            intent,
            entity_memory=entity_memory,
        )

    def _extract_travel_mode(self, query: str) -> str:
        return self.distance_intent_service._extract_travel_mode(query)

    def _get_directions(self, source: Dict[str, Any], target: Dict[str, Any], mode: str) -> Dict[str, Any]:
        return self.distance_intent_service._get_directions(source, target, mode)

    def _run_distance_intent(
        self,
        user_query: str,
        metadata: Dict[str, Any],
        grounded_nodes: List[Any],
        entities: List[Dict[str, Any]],
        detected_location: str,
    ) -> Dict[str, Any]:
        return self.distance_intent_service.run_distance_intent(
            user_query=user_query,
            metadata=metadata,
            grounded_nodes=grounded_nodes,
            entities=entities,
            detected_location=detected_location,
        )

    def _build_graph_payload(
        self,
        seeds,
        facts: List[str],
        intent: str = "",
        route_seed_nodes: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build lightweight graph data for FE explainability view."""
        nodes: Dict[str, Dict[str, Any]] = {}
        links = []
        seen_edges = set()

        # Seed nodes are trusted anchors from retrieval.
        for seed in seeds or []:
            seed_name = seed.metadata.get("name") or seed.content
            if not seed_name:
                continue
            seed_id = str(seed.id or seed_name)
            nodes[seed_name] = {
                "id": seed_id,
                "name": seed_name,
                "labels": seed.metadata.get("labels", []),
                "lat": seed.metadata.get("lat"),
                "lng": seed.metadata.get("lng"),
            }

        relation_labels = sorted(set(RELATIONSHIP_MAP.values()), key=len, reverse=True)
        multi_hop_pattern = re.compile(
            r"^(?P<src>.+?)\s+\(liên kết\s+\d+\s+bước:\s+(?P<chain>.+?)\)\s+→\s+(?P<dst>.+?)(?::|\s+\(|\s+\[|$)",
            flags=re.IGNORECASE,
        )

        for fact in facts or []:
            if not isinstance(fact, str):
                continue
            cleaned = fact.strip()
            if not cleaned:
                continue

            # Parse multi-hop fact format from traverser, e.g.
            # A (liên kết 2 bước: nằm gần → phục vụ món) → B ...
            mh = multi_hop_pattern.match(cleaned)
            if mh:
                source_name = mh.group("src").strip()
                target_name = mh.group("dst").strip()
                rel_label = mh.group("chain").strip()

                if source_name and target_name and source_name != target_name:
                    if source_name not in nodes:
                        nodes[source_name] = {
                            "id": f"name:{source_name}",
                            "name": source_name,
                            "labels": [],
                        }
                    if target_name not in nodes:
                        nodes[target_name] = {
                            "id": f"name:{target_name}",
                            "name": target_name,
                            "labels": [],
                        }

                    edge_key = (source_name, rel_label, target_name)
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        links.append(
                            {
                                "source": nodes[source_name]["id"],
                                "target": nodes[target_name]["id"],
                                "relation": rel_label,
                            }
                        )
                continue

            for rel_label in relation_labels:
                marker = f" {rel_label} "
                if marker not in cleaned:
                    continue

                left, right = cleaned.split(marker, 1)
                source_name = left.strip()
                target_name = right.split(" (Địa chỉ:", 1)[0].strip()

                if not source_name or not target_name or source_name == target_name:
                    continue

                if source_name not in nodes:
                    nodes[source_name] = {
                        "id": f"name:{source_name}",
                        "name": source_name,
                        "labels": [],
                    }
                if target_name not in nodes:
                    nodes[target_name] = {
                        "id": f"name:{target_name}",
                        "name": target_name,
                        "labels": [],
                    }

                edge_key = (source_name, rel_label, target_name)
                if edge_key in seen_edges:
                    continue

                seen_edges.add(edge_key)
                links.append(
                    {
                        "source": nodes[source_name]["id"],
                        "target": nodes[target_name]["id"],
                        "relation": rel_label,
                    }
                )
                break

        # TOUR_PLAN fallback: if traversal facts produce no explicit edges,
        # synthesize a minimum route chain so Graph View is never empty.
        if intent == IntentType.TOUR_PLAN and not links:
            ordered_route = route_seed_nodes or []
            if len(ordered_route) < 2:
                ordered_route = []
                for seed in seeds or []:
                    lat = seed.metadata.get("lat")
                    lng = seed.metadata.get("lng")
                    if lat is None or lng is None:
                        continue
                    ordered_route.append(
                        {
                            "id": str(seed.id),
                            "name": seed.metadata.get("name") or seed.content,
                            "labels": seed.metadata.get("labels", []),
                            "lat": lat,
                            "lng": lng,
                        }
                    )
                    if len(ordered_route) >= 6:
                        break

            for idx in range(len(ordered_route) - 1):
                src = ordered_route[idx]
                dst = ordered_route[idx + 1]
                src_name = src.get("name") or ""
                dst_name = dst.get("name") or ""
                if not src_name or not dst_name or src_name == dst_name:
                    continue

                if src_name not in nodes:
                    nodes[src_name] = {
                        "id": str(src.get("id") or f"name:{src_name}"),
                        "name": src_name,
                        "labels": src.get("labels", []),
                        "lat": src.get("lat"),
                        "lng": src.get("lng"),
                    }
                if dst_name not in nodes:
                    nodes[dst_name] = {
                        "id": str(dst.get("id") or f"name:{dst_name}"),
                        "name": dst_name,
                        "labels": dst.get("labels", []),
                        "lat": dst.get("lat"),
                        "lng": dst.get("lng"),
                    }

                links.append(
                    {
                        "source": nodes[src_name]["id"],
                        "target": nodes[dst_name]["id"],
                        "relation": "ROUTE_STEP",
                    }
                )

        # Resolve missing coordinates/labels from Neo4j for parsed fact nodes
        missing_names = [name for name, nd in nodes.items() if nd.get("lat") is None or nd.get("lng") is None]
        if missing_names and self.driver:
            try:
                with self.driver.session() as session:
                    cypher = """
                    MATCH (n)
                    WHERE n.name IN $names
                    RETURN n.name AS name,
                           n.id AS id,
                           labels(n) AS labels,
                           CASE 
                             WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' 
                             THEN n.location.latitude 
                             ELSE n.lat
                           END AS lat,
                           CASE 
                             WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' 
                             THEN n.location.longitude 
                             ELSE n.lng
                           END AS lng
                    """
                    rows = session.run(cypher, names=missing_names).data()
                    id_map = {}
                    for row in rows:
                        name = row.get("name")
                        db_id = row.get("id")
                        lat = row.get("lat")
                        lng = row.get("lng")
                        labels = row.get("labels")
                        if name in nodes:
                            if lat is not None and lng is not None:
                                nodes[name]["lat"] = lat
                                nodes[name]["lng"] = lng
                            if labels and not nodes[name].get("labels"):
                                nodes[name]["labels"] = labels
                            if db_id:
                                old_id = nodes[name]["id"]
                                new_id = str(db_id)
                                if old_id != new_id:
                                    nodes[name]["id"] = new_id
                                    id_map[old_id] = new_id
                    
                    if id_map:
                        for link in links:
                            if link.get("source") in id_map:
                                link["source"] = id_map[link["source"]]
                            if link.get("target") in id_map:
                                link["target"] = id_map[link["target"]]
            except (ValueError, RuntimeError, OSError) as e:
                PIPELINE_LOGGER.warning("graph_coordinates_resolver error: %s", e)

        return {
            "nodes": list(nodes.values()),
            "links": links,
            "edges": links,
        }


    #Phân loại hướng
    def _detect_region_focus(
        self,
        user_query: str,
        detected_location: str,
        entities: List[Dict[str, Any]],
    ) -> str:
        """
        Determine retrieval focus profile to avoid cross-area mixing.
        - inland_gia_lai: prioritize old Gia Lai hinterland (Pleiku, Chư, Ia, Đắk...)
        - coastal_quy_nhon: prioritize coastal former Bình Định area
        - all: do not apply geofence
        """
        # Only use query + detected_location for region focus.
        # Entity names may contain location words that are part of the name
        # (e.g., "Bò né 3 ngon Gia Lai") and should not influence region detection.
        parts = [user_query or "", detected_location or ""]
        full_text = normalize_text(" ".join(parts), strip_punct=True)

        # If the query has a specific named entity (Restaurant, Accommodation,
        # TouristAttraction), skip region filtering — the user is asking about a
        # specific place, not about a region.  Location-only entities (province,
        # city, district) do NOT trigger this exemption.
        for e in entities or []:
            if not isinstance(e, dict):
                continue
            etype = str(e.get("type") or "").lower().replace("_", "")
            if etype in keywords.SPECIFIC_ENTITY_TYPES:
                return RegionFocus.ALL

        # New administrative rule: "Gia Lai + biển" must not be forced inland.
        if "gia lai" in full_text and "bien" in full_text:
            return RegionFocus.ALL

        # Multi-region: "từ Gia Lai đến Bình Định", "Pleiku - Quy Nhơn"
        # REQUIREMENT: At least 2 DISTINCT region names (not connectors).
        # "Đến Bình Định" = single destination, NOT multi-region.
        # "tôi" (I) normalizes to "toi" which falsely matches connector "tới".
        from graph_rag.config.region_registry import region_registry
        _MULTI_REGION_NAMES = []
        for pid in region_registry.get_all_province_ids():
            _MULTI_REGION_NAMES.extend(region_registry.get_keywords(pid))
            _MULTI_REGION_NAMES.extend([a.lower() for a in region_registry.get_aliases(pid)])
        region_name_count = sum(1 for m in _MULTI_REGION_NAMES if m in full_text)
        if region_name_count >= 2:
            return RegionFocus.ALL

        # Check inland BEFORE coastal — inland keywords are more specific
        # (multi-word like "bien ho", "krong pa") and must not be shadowed
        # by bare coastal tokens like "bien" that match as substrings.
        if "gia lai" in full_text:
            _specific_inland = [kw for kw in self.INLAND_KEYWORDS if kw != "gia lai" and kw in full_text]
            if _specific_inland:
                return RegionFocus.INLAND
            return RegionFocus.ALL

        if any(kw in full_text for kw in self.INLAND_KEYWORDS):
            return RegionFocus.INLAND

        if any(kw in full_text for kw in self.COASTAL_KEYWORDS):
            return RegionFocus.COASTAL

        return RegionFocus.ALL

    def _seed_in_region(self, seed: Any, region_focus: str) -> bool:
        if region_focus == RegionFocus.ALL:
            return True

        # Keep nodes with region_focus="all" — they are general resources
        # (e.g., community forums, national travel info) that apply to all regions.
        seed_region_focus = str((getattr(seed, "metadata", {}) or {}).get("region_focus") or "").strip().lower()
        if seed_region_focus in ("all", ""):
            return True

        # Dynamic: resolve region_focus → expected region_groups from RegionRegistry
        from graph_rag.config.region_registry import region_registry
        focus_map = {
            RegionFocus.COASTAL: "coastal",
            RegionFocus.INLAND: "inland",
        }
        target_focus = focus_map.get(region_focus, "")
        expected_region_groups = set()
        if target_focus:
            for pid in region_registry.get_all_province_ids():
                p = region_registry.get_province(pid)
                if p and p.get("region_focus") == target_focus:
                    expected_region_groups.add(p.get("region_group", ""))
                    # Also include merged provinces' groups
                    for merged_pid in region_registry.get_merged_provinces(pid):
                        mp = region_registry.get_province(merged_pid)
                        if mp:
                            expected_region_groups.add(mp.get("region_group", ""))

        seed_region_group = str((getattr(seed, "metadata", {}) or {}).get("region_group") or "").strip()
        if expected_region_groups and seed_region_group:
            return seed_region_group in expected_region_groups

        # First, reject explicit opposite-region textual signals even without coordinates.
        # NOTE: "Biển Hồ", "biển hồ" are INLAND entities despite containing "biên".
        # Use word-boundary-aware matching to avoid false positives.
        text_parts = [
            str(seed.metadata.get("name") or ""),
            str(seed.metadata.get("address") or ""),
            str(seed.content or ""),
        ]
        text_blob = normalize_text(" ".join(text_parts), strip_punct=True)
        # Whitelist: inland entities whose names contain coastal keywords
        _INLAND_NAME_EXCEPTIONS = {"bien ho", "bien hồ", "ho t'nung", "ho t nung"}
        is_inland_exception = any(exc in text_blob for exc in _INLAND_NAME_EXCEPTIONS)
        if region_focus == RegionFocus.COASTAL and any(kw in text_blob for kw in self.INLAND_KEYWORDS):
            if not is_inland_exception:
                return False
        if region_focus == RegionFocus.INLAND and any(kw in text_blob for kw in self.COASTAL_KEYWORDS):
            if not is_inland_exception:
                return False

        lat = seed.metadata.get("lat")
        lng = seed.metadata.get("lng")

        # Non-geocoded nodes are allowed only when text does not contradict region focus.
        if lat is None or lng is None:
            return True

        try:
            lat = float(lat)
            lng = float(lng)
        except (ValueError, TypeError):
            return True

        # Dynamic: use bounding boxes from RegionRegistry
        # For now, allow all geocoded nodes when region_group didn't match
        # (bounding box is a fallback for nodes without region_group)
        return True
    #Lọc dữ liệu khi rõ ràng trong truy vấn
    def _apply_region_focus_filter(self, seeds: List[Any], region_focus: str) -> List[Any]:
        if region_focus == RegionFocus.ALL:
            return seeds

        filtered = [s for s in seeds if self._seed_in_region(s, region_focus)]
        return filtered
    # Chọn các điểm hạt giống có khả năng là các điểm trên tuyến đường dựa trên sự xuất hiện của chúng trong văn bản trả lời TOUR_PLAN.
    def _select_route_seed_nodes(self, answer: str, seeds: List[Any], intent: str) -> List[Dict[str, Any]]:
        """Select map points that are explicitly mentioned in TOUR_PLAN answer text."""
        if intent != IntentType.TOUR_PLAN or not seeds:
            return []

        norm_answer = normalize_text(answer, strip_punct=True)
        if not norm_answer:
            return []

        matched = []
        for seed in seeds:
            name = seed.metadata.get("name") or seed.content
            lat = seed.metadata.get("lat")
            lng = seed.metadata.get("lng")
            if not name or lat is None or lng is None:
                continue

            norm_name = normalize_text(name, strip_punct=True)
            if len(norm_name) < 4:
                continue

            idx = norm_answer.find(norm_name)
            if idx >= 0:
                matched.append(
                    {
                        "id": seed.id,
                        "name": name,
                        "labels": seed.metadata.get("labels", []),
                        "attributes": seed.metadata,
                        "lat": lat,
                        "lng": lng,
                        "order": idx,
                    }
                )

        matched.sort(key=lambda item: item["order"])
        deduped = []
        seen = set()
        for item in matched:
            key = str(item["id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append({k: v for k, v in item.items() if k != "order"})

        
        if len(deduped) < 2:
            for seed in seeds:
                lat = seed.metadata.get("lat")
                lng = seed.metadata.get("lng")
                if lat is None or lng is None:
                    continue
                key = str(seed.id)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(
                    {
                        "id": seed.id,
                        "name": seed.metadata.get("name") or seed.content,
                        "labels": seed.metadata.get("labels", []),
                        "attributes": seed.metadata,
                        "lat": lat,
                        "lng": lng,
                    }
                )
                if len(deduped) >= 6:
                    break

        return deduped

    # Compatibility wrappers: TOUR_PLAN optimization now lives in TourRouteOptimizerService.
    def _extract_trip_days(self, user_query: str) -> int:
        return self.tour_route_optimizer.extract_trip_days(user_query)

    def _extract_route_constraints(
        self,
        primary_intent: str,
        user_query: str,
        analyzer_constraints: Dict[str, Any],
        trip_days: int = 2,
    ) -> Dict[str, Any]:
        return self.tour_route_optimizer.extract_route_constraints(
            primary_intent,
            user_query,
            analyzer_constraints,
            trip_days=trip_days,
        )

    def run(
        self,
        user_query: str,
        chat_history: List[Dict] = None,
        current_location: str = "",
        user_gps: str = "",
        eval_metadata: Dict[str, Any] | None = None,
        on_token=None,
    ):
        from graph_rag.pipeline.orchestration.application_service import PipelineApplicationService

        # BUG-02: Per-request state isolation via deepcopy (under lock to prevent race)
        with self._state_lock:
            request_state = copy.deepcopy(self.conversation_state)
        original_state = self.location_grounding_service.conversation_state
        self.location_grounding_service.conversation_state = request_state

        # OBS-02: Request-level timing
        request_start = time.time()
        request_id = hashlib.md5(
            f"{user_query}:{request_start}".encode()
        ).hexdigest()[:8]
        step_timings: Dict[str, int] = {}

        try:
            service = PipelineApplicationService(self)
            result = service.execute(
                user_query=user_query,
                chat_history=chat_history,
                current_location=current_location,
                user_gps=user_gps,
                eval_metadata=eval_metadata,
                on_token=on_token,
                request_id=request_id,
                step_timings=step_timings,
            )
        finally:
            # BUG-02: Merge back only persistent fields under lock, restore original
            with self._state_lock:
                self.conversation_state["history"] = request_state.get("history", [])
                self.conversation_state["entity_memory"] = request_state.get("entity_memory", [])
                self.conversation_state["last_grounded_anchor"] = request_state.get("last_grounded_anchor", {})
            self.location_grounding_service.conversation_state = original_state

        # OBS-02: Log request completion
        total_ms = int((time.time() - request_start) * 1000)
        metadata = result.get("metadata") or {}
        PIPELINE_LOGGER.info(
            "request_complete request_id=%s total_ms=%d intent=%s retrieval_score=%s llm_latency_ms=%s",
            request_id,
            total_ms,
            metadata.get("intent", ""),
            metadata.get("retrieval_score", "n/a"),
            metadata.get("llm_latency_ms", "n/a"),
        )

        # Store timing in metadata for debug responses
        metadata["request_id"] = request_id
        metadata["step_timings"] = step_timings
        metadata["total_duration_ms"] = total_ms

        return result
