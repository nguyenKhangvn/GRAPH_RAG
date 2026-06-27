from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import re
import time
from typing import Any, Dict, List, Tuple

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

from graph_rag.core.intents import IntentType
from graph_rag.utils.text import normalize_text
from graph_rag.config.deictic_patterns import (
    PROXIMITY_DEICTIC_PATTERNS,
    is_deictic_entity_phrase,
    get_type_hint_for_deictic,
)


class LocationGroundingService:
    # DEICTIC_QUERY_PATTERNS imported from graph_rag.config.deictic_patterns
    # Xem file đó để debug/thêm pattern mới

    def __init__(
        self,
        *,
        driver,
        logger,
        retriever,
        conversation_state: Dict[str, Any],
        coastal_keywords: set[str],
        inland_keywords: set[str],
        coastal_bounds: Dict[str, float],
        inland_bounds: Dict[str, float],
        location_source_confidence: Dict[str, float],
    ):
        self.driver = driver
        self.logger = logger
        self.retriever = retriever
        self.conversation_state = conversation_state
        self.coastal_keywords = coastal_keywords
        self.inland_keywords = inland_keywords
        self.coastal_bounds = coastal_bounds
        self.inland_bounds = inland_bounds
        self.location_source_confidence = location_source_confidence

    # Labels whose nodes NEVER carry location information.
    # Location inference should skip these to avoid type errors
    # (e.g. Dish.location = WGS84Point) and because their location
    # is only reachable via multi-hop traversal (Dish ←[:HAS]← Restaurant).
    _NON_LOCATABLE_LABELS = frozenset({
        "Dish", "Tour", "Specialty", "Category", "TravelAgency",
    })

    def infer_location_from_grounded_nodes(self, grounded_nodes: List[Any]) -> str:
        ids = [n.id for n in (grounded_nodes or []) if getattr(n, "id", None)]
        if not ids:
            return ""

        candidates: List[str] = []
        try:
            with self.driver.session() as session:
                records = session.run(
                    """
                    MATCH (n)
                    WHERE n.id IN $ids
                    OPTIONAL MATCH (n)-[:LOCATED_IN]->(loc:Location)
                    RETURN coalesce(loc.name, '') AS loc_name,
                           coalesce(n.address, '') AS address,
                           coalesce(n.province, '') AS province,
                           coalesce(n.location, '') AS node_location,
                           labels(n) AS node_labels
                    """,
                    ids=ids,
                )
                for record in records:
                    node_labels = record.get("node_labels") or []

                    # Skip nodes that can never have location information.
                    # Their location is reachable only via multi-hop traversal.
                    if any(lbl in self._NON_LOCATABLE_LABELS for lbl in node_labels):
                        continue

                    loc_name = (record.get("loc_name") or "").strip()
                    if loc_name:
                        candidates.append(loc_name)

                    # Fallback: read province/location property directly
                    province = (record.get("province") or "").strip()
                    if province:
                        candidates.append(province)

                    # node_location may be a WGS84Point (spatial) or a string
                    # text (place name).  Only string values are useful here.
                    raw_loc = record.get("node_location")
                    node_loc = ""
                    if isinstance(raw_loc, str):
                        node_loc = raw_loc.strip()
                    if node_loc:
                        candidates.append(node_loc)

                    addr = (record.get("address") or "").strip()
                    if addr:
                        low = normalize_text(addr)
                        if "quy nhon" in low:
                            candidates.append("Quy Nhơn")
                        elif "pleiku" in low:
                            candidates.append("Pleiku")
        except (Neo4jClientError, ServiceUnavailable, ValueError, TypeError) as exc:
            self.logger.warning("grounded_location_inference_warning: %s", str(exc))
            return ""

        if not candidates:
            return ""

        counts: Dict[str, int] = {}
        for c in candidates:
            key = c.strip()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1

        if not counts:
            return ""
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def query_region_signal(self, user_query: str, entities: List[Dict[str, Any]]) -> str:
        parts = [user_query or ""]
        for e in entities or []:
            if isinstance(e, dict):
                parts.append(e.get("name") or "")
        full_text = normalize_text(" ".join(parts))

        # Inland keywords are more specific (multi-word) — check first
        # to prevent bare "bien" from shadowing "bien ho".
        # Use word-boundary matching to prevent substring false-positives (e.g., "ba" matching in "ban")
        has_inland = any(re.search(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', full_text) for kw in self.inland_keywords)
        has_coastal = any(re.search(r'(?<!\w)' + re.escape(kw) + r'(?!\w)', full_text) for kw in self.coastal_keywords)
        if has_inland:
            return "inland_gia_lai"
        if has_coastal:
            return "coastal_quy_nhon"
        return "all"

    def location_to_region_focus(self, location_text: str) -> str:
        text = normalize_text(location_text)
        if not text:
            return "all"
        # Inland keywords are more specific — check first.
        has_inland = any(kw in text for kw in self.inland_keywords)
        has_coastal = any(kw in text for kw in self.coastal_keywords)
        if has_inland:
            return "inland_gia_lai"
        if has_coastal:
            return "coastal_quy_nhon"
        return "all"

    def resolve_location_priority(
        self,
        query: str,
        current_location: str,
        grounded_location: str,
        grounded_reason: str,
        entities: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Centralized location priority resolution.

        Priority order:
        1. Proximity deictic ("gần đây", "từ đây") → current_location wins
        2. Explicit query region ("Tây Nguyên", "Gia Lai") → query wins
        3. No signal → defer to downstream

        Returns dict with:
          - final_location: str
          - source: "explicit_query_region" | "current_location" | "grounded" | "none"
          - disable_current_location_filter: bool
          - region_focus: str
        """
        q_norm = normalize_text(query, strip_punct=True)

        # 1. Proximity deictic → current_location wins
        has_proximity = any(p in q_norm for p in PROXIMITY_DEICTIC_PATTERNS)
        if has_proximity and current_location:
            return {
                "final_location": current_location,
                "source": "current_location",
                "disable_current_location_filter": False,
                "region_focus": self.location_to_region_focus(current_location),
            }

        # 2. Explicit query region signal → query wins
        query_region = self.query_region_signal(query, entities)
        has_explicit_region = query_region != "all"

        if has_explicit_region:
            return {
                "final_location": grounded_location or current_location or "",
                "source": "explicit_query_region",
                "disable_current_location_filter": True,
                "region_focus": query_region,
            }

        # 3. No signal → defer
        return {
            "final_location": current_location or grounded_location or "",
            "source": "current_location" if current_location else ("grounded" if grounded_location else "none"),
            "disable_current_location_filter": False,
            "region_focus": "all",
        }

    def build_location_context(
        self,
        name: str,
        source: str,
        reason: str = "",
        confidence: float = None,
    ) -> Dict[str, Any]:
        src = str(source or "global").strip().lower()
        if confidence is None:
            confidence = float(self.location_source_confidence.get(src, 0.0))
        return {
            "name": (name or "").strip(),
            "source": src,
            "confidence": float(max(0.0, min(1.0, confidence))),
            "reason": reason or "",
        }

    def choose_location_context(
        self,
        old_ctx: Dict[str, Any],
        new_ctx: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        if not old_ctx or not old_ctx.get("name"):
            return new_ctx, bool(new_ctx and new_ctx.get("name"))
        if not new_ctx or not new_ctx.get("name"):
            return old_ctx, False

        old_conf = float(old_ctx.get("confidence", 0.0))
        new_conf = float(new_ctx.get("confidence", 0.0))
        if new_conf > old_conf:
            return new_ctx, True
        return old_ctx, False

    def extract_anchor_location_from_history(self, history: List[Dict]) -> str:
        if not history:
            return ""
        for msg in reversed(history):
            if msg.get("role") not in {"user", "assistant"}:
                continue
            text = normalize_text(msg.get("content") or "")
            if not text:
                continue
            if "quy nhon" in text:
                return "Quy Nhơn"
            if "pleiku" in text:
                return "Pleiku"
            if "gia lai" in text:
                return "Gia Lai"
        return ""

    def build_initial_location_context(
        self,
        current_location: str,
        history: List[Dict],
        analyzer_output: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Domain-agnostic location context initialization.
        Uses dialog state signals (is_follow_up, dialog_act) from LLM analyzer,
        not hardcoded keywords.
        """
        user_gps_location = (current_location or "").strip()

        # 1. Read dialog behavior signals from LLM analyzer output
        is_follow_up = bool((analyzer_output or {}).get("is_follow_up", False))
        detected_from_analyzer = str((analyzer_output or {}).get("detected_location") or "").strip()

        # 2. Get context location saved from previous turn
        prev_active_location = self.conversation_state.get("last_active_location") or ""

        # 3. Dialog Policy: inherit location for follow-up queries
        if is_follow_up and prev_active_location:
            return self.build_location_context(
                name=prev_active_location,
                source="history_state",
                reason="inherited_by_dialog_policy",
            )

        # 4. If analyzer detected a specific location, use it
        if detected_from_analyzer:
            return self.build_location_context(
                name=detected_from_analyzer,
                source="analyzer",
                reason="analyzer_detected_location",
            )

        # 5. Otherwise, use GPS location
        if user_gps_location:
            return self.build_location_context(
                name=user_gps_location,
                source="user",
                reason="request_current_location",
            )

        anchor = (self.conversation_state.get("current_location") or "").strip()
        if anchor:
            return self.build_location_context(
                name=anchor,
                source="history",
                reason="conversation_anchor",
            )

        last_grounded_anchor = self.conversation_state.get("last_grounded_anchor") or {}
        anchor_location = str(
            last_grounded_anchor.get("location")
            or last_grounded_anchor.get("address")
            or ""
        ).strip()
        if anchor_location:
            return self.build_location_context(
                name=anchor_location,
                source="history",
                reason="last_grounded_anchor",
            )

        from_history = self.extract_anchor_location_from_history(history)
        if from_history:
            return self.build_location_context(
                name=from_history,
                source="history",
                reason="history_inference",
            )

        user_geo = (self.conversation_state.get("user_geo_location") or "").strip()
        if user_geo:
            return self.build_location_context(
                name=user_geo,
                source="user",
                reason="user_geo_fallback",
                confidence=0.6,
            )

        return self.build_location_context(name="", source="global", reason="no_anchor")

    def has_explicit_location(self, entities: List[Dict[str, Any]]) -> bool:
        for entity in entities or []:
            if not isinstance(entity, dict):
                continue
            if str(entity.get("type") or "").strip().lower() == "location":
                return True
        return False

    def clear_conversation_context(self, new_location: str = "") -> None:
        self.conversation_state["entity_memory"] = []
        self.conversation_state["history"] = []
        self.conversation_state["last_grounded_anchor"] = {}
        if new_location:
            self.conversation_state["current_location"] = new_location

    def infer_grounded_location_context(self, grounded_nodes: List[Any]) -> Dict[str, Any]:
        if not grounded_nodes:
            return self.build_location_context(name="", source="global", reason="no_grounded")

        exact_nodes = [n for n in grounded_nodes if str(getattr(n, "source_type", "")).lower() == "exact_match"]
        if exact_nodes:
            exact_loc = self.infer_location_from_grounded_nodes(exact_nodes)
            if exact_loc:
                return self.build_location_context(
                    name=exact_loc,
                    source="graph",
                    reason="grounded_entity_exact_match",
                    confidence=1.0,
                )

        inferred = self.infer_location_from_grounded_nodes(grounded_nodes)
        if inferred:
            return self.build_location_context(
                name=inferred,
                source="graph",
                reason="grounded_entity_soft_match",
                confidence=0.85,
            )

        return self.build_location_context(name="", source="global", reason="grounded_no_location")

    def _node_region(self, node: Any) -> str:
        if not node:
            return "unknown"

        # Fast path: nếu node đã có region_focus property từ graph → dùng trực tiếp
        if hasattr(node, "metadata") and isinstance(node.metadata, dict):
            rf = (node.metadata.get("region_focus") or "").strip().lower()
            if rf in ("coastal_quy_nhon", "inland_gia_lai"):
                return rf

        lat = node.metadata.get("lat") if hasattr(node, "metadata") else None
        lng = node.metadata.get("lng") if hasattr(node, "metadata") else None
        try:
            if lat is not None and lng is not None:
                lat = float(lat)
                lng = float(lng)
                if (
                    self.coastal_bounds["lat_min"] <= lat <= self.coastal_bounds["lat_max"]
                    and self.coastal_bounds["lng_min"] <= lng <= self.coastal_bounds["lng_max"]
                ):
                    return "coastal_quy_nhon"
                if (
                    self.inland_bounds["lat_min"] <= lat <= self.inland_bounds["lat_max"]
                    and self.inland_bounds["lng_min"] <= lng <= self.inland_bounds["lng_max"]
                ):
                    return "inland_gia_lai"
        except (ValueError, TypeError):
            pass

        text_parts = []
        if hasattr(node, "content"):
            text_parts.append(node.content or "")
        if hasattr(node, "metadata") and isinstance(node.metadata, dict):
            text_parts.append(node.metadata.get("name") or "")
            text_parts.append(node.metadata.get("address") or "")
        text = normalize_text(" ".join(text_parts))

        if any(kw in text for kw in self.coastal_keywords):
            return "coastal_quy_nhon"
        if any(kw in text for kw in self.inland_keywords):
            return "inland_gia_lai"
        
        # Fallback: If no coordinates and no keyword match, lookup location hierarchy from LOCATED_IN relationships
        if hasattr(node, "id") and node.id:
            try:
                with self.driver.session() as session:
                    # Try to climb up the hierarchy: entity -> location -> parent_location -> ...
                    result = session.run("""
                        MATCH (n)-[:LOCATED_IN*1..3]->(ancestor:Location)
                        WHERE n.id = $node_id
                        RETURN ancestor.name as loc_name, ancestor.lat as loc_lat, ancestor.lng as loc_lng,
                               ancestor.region_focus as region_focus, ancestor.admin_status as admin_status
                        ORDER BY length([(n)-[:LOCATED_IN*1..3]->(ancestor)])
                        LIMIT 10
                    """, node_id=node.id)

                    for loc_record in result:
                        if not loc_record:
                            continue
                        loc_name = (loc_record.get("loc_name") or "").lower()
                        loc_lat = loc_record.get("loc_lat")
                        loc_lng = loc_record.get("loc_lng")

                        # Fast path: use region_focus from ancestor node if available
                        ancestor_rf = (loc_record.get("region_focus") or "").strip().lower()
                        if ancestor_rf in ("coastal_quy_nhon", "inland_gia_lai"):
                            return ancestor_rf

                        # Try location coordinates
                        if loc_lat is not None and loc_lng is not None:
                            try:
                                loc_lat = float(loc_lat)
                                loc_lng = float(loc_lng)
                                if (
                                    self.coastal_bounds["lat_min"] <= loc_lat <= self.coastal_bounds["lat_max"]
                                    and self.coastal_bounds["lng_min"] <= loc_lng <= self.coastal_bounds["lng_max"]
                                ):
                                    return "coastal_quy_nhon"
                                if (
                                    self.inland_bounds["lat_min"] <= loc_lat <= self.inland_bounds["lat_max"]
                                    and self.inland_bounds["lng_min"] <= loc_lng <= self.inland_bounds["lng_max"]
                                ):
                                    return "inland_gia_lai"
                            except (ValueError, TypeError):
                                pass

                        # Try location name keywords
                        if any(kw in loc_name for kw in self.coastal_keywords):
                            return "coastal_quy_nhon"
                        if any(kw in loc_name for kw in self.inland_keywords):
                            return "inland_gia_lai"
            except (Neo4jClientError, ServiceUnavailable):
                pass

        return "unknown"

    def enforce_grounding_region_consistency(
        self,
        grounded_nodes: List[Any],
        user_query: str,
        entities: List[Dict[str, Any]],
    ) -> tuple[List[Any], str]:
        if not grounded_nodes:
            return grounded_nodes, "all"

        explicit_region = self.query_region_signal(user_query, entities)
        counts = {"coastal_quy_nhon": 0, "inland_gia_lai": 0}
        for node in grounded_nodes:
            region = self._node_region(node)
            if region in counts:
                counts[region] += 1

        majority_region = "all"
        if counts["coastal_quy_nhon"] > counts["inland_gia_lai"]:
            majority_region = "coastal_quy_nhon"
        elif counts["inland_gia_lai"] > counts["coastal_quy_nhon"]:
            majority_region = "inland_gia_lai"

        final_region = explicit_region if explicit_region != "all" else majority_region
        if final_region == "all":
            return grounded_nodes, final_region

        filtered = [n for n in grounded_nodes if self._node_region(n) in {final_region, "unknown"}]
        if not filtered:
            return grounded_nodes, final_region

        return filtered, final_region

    def extract_recent_mention(self, history: List[Dict], patterns: List[str]) -> str:
        for msg in reversed(history or []):
            if msg.get("role") not in {"user", "assistant"}:
                continue
            text = msg.get("content") or ""
            for pat in patterns:
                match = re.search(pat, text, flags=re.IGNORECASE)
                if match:
                    value = match.group(1).strip(" ,.;:!?")
                    if len(normalize_text(value)) >= 8:
                        return value
        return ""

    def _entity_memory_lookup(self, entity_type: str) -> str:
        memory = self.conversation_state.get("entity_memory") or []
        if not memory:
            return ""
        normalized_type = str(entity_type or "").strip()
        for item in reversed(memory):
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip() != normalized_type:
                continue
            name = str(item.get("name") or "").strip()
            if len(normalize_text(name)) >= 4:
                return name
        return ""

    def resolve_generic_entities_with_history(
        self,
        entities: List[Dict[str, Any]],
        history: List[Dict],
        intent: str,
        entity_memory: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not entities:
            return []

        supported_intents = {
            IntentType.DISTANCE,
            IntentType.TOUR_PLAN,
            IntentType.ACCOMMODATION,
            IntentType.FOOD,
            IntentType.TOURISM,
        }
        if intent not in supported_intents:
            return entities

        generic_terms = {
            "khach san",
            "nha hang",
            "quan ca phe",
            "quan cafe",
            "cafe",
            "coffee",
            "khach sạn",
            "khách sạn",
            "nhà hàng",
            "quang truong",
            "quang trường",
            "quảng trường",
            "bai bien",
            "bãi biển",
            "diem tham quan",
        }

        history_patterns_by_type = {
            "Accommodation": [r"(khách sạn\s+[^\n,.;:!?]{2,80})"],
            "Restaurant": [
                r"(nhà hàng\s+[^\n,.;:!?]{2,80})",
                r"([A-Za-zÀ-ỹ0-9\-\s]{3,80}(?:Coffee|Cafe|Roasters)[^\n,.;:!?]{0,20})",
            ],
            "Location": [r"(quảng trường\s+[^\n,.;:!?]{2,80})"],
            "TouristAttraction": [r"([A-ZÀ-Ỹ][^\n,.;:!?]{4,80})"],
        }

        resolved = []
        for entity in entities:
            if not isinstance(entity, dict):
                resolved.append(entity)
                continue

            name = (entity.get("name") or "").strip()
            etype = entity.get("type") or ""
            norm = normalize_text(name)

            # ── DEICTIC GUARD: "quán này", "chỗ này", etc. ──
            # Không semantic search → resolve từ last_active_entity
            if is_deictic_entity_phrase(norm):
                type_hint = get_type_hint_for_deictic(norm)  # e.g. "quán" → "Restaurant"
                last_active = self.conversation_state.get("last_active_entity") or {}
                last_name = str(last_active.get("name") or "").strip()
                last_type = str(last_active.get("type") or "").strip()

                # Type checking: "quán này" chỉ resolve nếu last_active là Restaurant
                type_compatible = (
                    not type_hint  # no hint → accept any type (e.g. "chỗ này", "nơi này")
                    or not last_type  # no type info → accept
                    or last_type == type_hint  # exact match
                )

                if last_name and type_compatible:
                    entity = {**entity, "name": last_name, "type": last_type or etype}
                    logger.info("   -> Deictic resolved: '%s' -> '%s' (from last_active_entity)", name, last_name)
                else:
                    # Không có last_active_entity HOẶC type không khớp → thử entity_memory
                    search_type = type_hint or etype
                    if last_name and not type_compatible:
                        logger.info("   -> Deictic type mismatch: '%s' expects %s, but last_active is %s. Searching entity_memory...", name, type_hint, last_type)
                    replacement = self._entity_memory_lookup(search_type)
                    if replacement:
                        entity = {**entity, "name": replacement}
                        logger.info("   -> Deictic resolved: '%s' -> '%s' (from entity_memory)", name, replacement)
                    elif last_name:
                        # Fallback: dùng last_active dù type không khớp
                        entity = {**entity, "name": last_name, "type": last_type or etype}
                        logger.warning("   -> Deictic FALLBACK: '%s' -> '%s' (type mismatch, no memory)", name, last_name)
                    else:
                        logger.info("   -> Deictic UNRESOLVED: '%s' (no last_active_entity, no memory)", name)
                resolved.append(entity)
                continue

            is_generic = norm in {normalize_text(x) for x in generic_terms} or len(norm) <= 8
            if is_generic:
                patterns = history_patterns_by_type.get(etype, [])
                replacement = self.extract_recent_mention(history, patterns)
                if not replacement:
                    temp_memory = entity_memory or []
                    for item in reversed(temp_memory):
                        if str(item.get("type") or "") != str(etype or ""):
                            continue
                        candidate_name = str(item.get("name") or "").strip()
                        if len(normalize_text(candidate_name)) >= 4:
                            replacement = candidate_name
                            break
                if not replacement:
                    replacement = self._entity_memory_lookup(etype)
                if replacement:
                    entity = {**entity, "name": replacement}

            resolved.append(entity)

        return resolved

    def remember_entities(self, entities: List[Dict[str, Any]]) -> None:
        if not entities:
            return
        memory = self.conversation_state.get("entity_memory") or []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name") or "").strip()
            etype = str(ent.get("type") or "").strip()
            if not name or not etype:
                continue
            if len(normalize_text(name)) < 4:
                continue
            memory.append({"type": etype, "name": name, "turn": int(time.time())})

        deduped = []
        seen = set()
        for item in reversed(memory):
            key = (str(item.get("type") or ""), normalize_text(item.get("name") or ""))
            if not key[0] or not key[1] or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        deduped.reverse()
        self.conversation_state["entity_memory"] = deduped[-80:]

    # Region groups for forgetting mechanism — khi user đổi vùng, clear entity_memory
    _INLAND_KEYWORDS = {"pleiku", "gia lai", "chu se", "ia grai", "duc co", "an khe", "ayun pa", "chu puh"}
    _COASTAL_KEYWORDS = {"quy nhon", "binh dinh", "nhon ly", "nhon hai", "phu cat", "hoai nhon", "tam quan", "tuy phuoc"}

    def _location_region(self, loc: str) -> str:
        """Classify location into region group: 'inland', 'coastal', or ''."""
        norm = normalize_text(loc or "", strip_punct=True)
        if any(kw in norm for kw in self._INLAND_KEYWORDS):
            return "inland"
        if any(kw in norm for kw in self._COASTAL_KEYWORDS):
            return "coastal"
        return ""

    def update_conversation_state(
        self,
        history: List[Dict],
        user_query: str,
        answer: str,
        location: str,
        entities: List[Dict[str, Any]],
        last_grounded_anchor: Dict[str, Any] | None = None,
        is_follow_up: bool = False,
        intent: str = "",
        target_class: str = "",
        answer_mode: str = "",
        region_focus: str = "",
        semantic_category: str = "",
    ) -> None:
        base_history = list(history or [])
        base_history.append({"role": "user", "content": user_query})
        if answer:
            base_history.append({"role": "assistant", "content": answer})
        self.conversation_state["history"] = base_history[-24:]

        # Context Expiry: flush inherited context when user starts a new topic
        # CHỈ clear location, KHÔNG clear last_active_entity
        # vì last_active_entity cần tồn tại cho deictic reference ("quán này", "chỗ này")
        # last_active_entity sẽ được ghi đè bởi _finalize_pipeline_response
        if not is_follow_up:
            self.conversation_state["last_active_location"] = ""

        # Forgetting mechanism: clear entity_memory khi user đổi vùng rõ ràng
        # Tránh resolve "quán đó" thành entity cũ từ vùng khác
        if location:
            prev_location = self.conversation_state.get("current_location") or ""
            prev_region = self._location_region(prev_location)
            new_region = self._location_region(location)
            if prev_region and new_region and prev_region != new_region:
                old_count = len(self.conversation_state.get("entity_memory") or [])
                self.conversation_state["entity_memory"] = []
                self.conversation_state["last_active_entity"] = {}
                logger.info("   -> [Forgetting] Region changed: %s -> %s. Cleared %s entities from memory.", prev_region, new_region, old_count)

            self.conversation_state["current_location"] = location
            self.conversation_state["last_active_location"] = location
        self.remember_entities(entities)
        if last_grounded_anchor:
            self.conversation_state["last_grounded_anchor"] = last_grounded_anchor

        # Save turn-level state for ConversationStateResolver (follow-up inheritance)
        if intent:
            self.conversation_state["last_intent"] = intent
        if target_class:
            self.conversation_state["last_target_class"] = target_class
        if answer_mode:
            self.conversation_state["last_answer_mode"] = answer_mode
        if region_focus and region_focus != "all":
            self.conversation_state["last_region_focus"] = region_focus
        self.conversation_state["last_semantic_category"] = semantic_category or ""

        # Save answer and query for follow-up search enhancement
        if answer:
            self.conversation_state["last_answer"] = answer
        if user_query:
            self.conversation_state["last_user_query"] = user_query
