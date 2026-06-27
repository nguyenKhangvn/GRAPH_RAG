from __future__ import annotations
import json
"""V3 structured answer generation and tour route metadata helpers."""
import logging

logger = logging.getLogger(__name__)


import re


from typing import Any, Dict, List



from graph_rag.config import GRAPH_RAG_V3_ENABLED


from graph_rag.core.answer_mode import AnswerMode


from graph_rag.core.intents import RegionFocus




from graph_rag.utils.text import normalize_text


from ..dto import PipelineRunState


class V3StructuredGenerationMixin:
    """Mixin providing V3 structured answer generation and route optimization."""

    def _should_route_v3_structured_generation(self, state: PipelineRunState, answer_mode: str) -> bool:
        if not GRAPH_RAG_V3_ENABLED:
            return False
        metadata = state.metadata or {}
        v3_intent_data = metadata.get("v3_intent_data") or {}
        v3_validation = metadata.get("v3_validation") or {}
        if not v3_intent_data or not v3_validation:
            return False
        if AnswerMode.is_closed_form(answer_mode):
            return False
        if answer_mode == AnswerMode.NEGATIVE_ABSTAIN_GUARD:
            return False

        context_state = v3_validation.get("context_state")
        if context_state == "INSUFFICIENT_EVIDENCE":
            return False

        v3_mode = str(v3_intent_data.get("intent_mode") or "")
        allowed_modes = {
            "comparison",
            "tour_plan",
            "constraint_matching",
            "multi_entity_nearby",
            "dish_to_restaurant",
        }
        if context_state == "NO_CANDIDATE" and v3_mode == "constraint_matching":
            return True
        if context_state == "NO_CANDIDATE":
            return False

        return v3_mode in allowed_modes


    def _generate_v3_structured_answer(self, state: PipelineRunState) -> str:
        metadata = state.metadata or {}
        v3_intent_data = metadata.get("v3_intent_data") or {}
        v3_validation = metadata.get("v3_validation") or {}
        structured_context = self._v3_generation_context(state)

        answer = self.pipeline.structured_answer_generator.generate(
            question=state.user_query,
            structured_context=structured_context,
            intent_data=v3_intent_data,
            validation=v3_validation,
        )
        state.runtime.metadata["v3_structured_generator"] = True

        missing = self._run_step_5_coverage_check(state, answer)
        if missing:
            state.runtime.metadata["v3_coverage_retry_triggered"] = True
            state.runtime.metadata["v3_coverage_missing"] = missing
            retry_instruction = (
                "Lưu ý: Câu trả lời trước của bạn thiếu các yếu tố bắt buộc sau: "
                + "; ".join(missing)
                + ". Hãy sinh lại câu trả lời đầy đủ."
            )
            retry_answer = self.pipeline.structured_answer_generator.generate(
                question=state.user_query,
                structured_context=structured_context,
                intent_data=v3_intent_data,
                validation=v3_validation,
                extra_instruction=retry_instruction,
            )
            if retry_answer:
                answer = retry_answer
        else:
            state.runtime.metadata["v3_coverage_retry_triggered"] = False
        return answer


    def _v3_generation_context(self, state: PipelineRunState) -> str:
        metadata = state.metadata or {}
        parts: List[str] = []
        near_pairs = self._direct_near_evidence_lines(state)
        if near_pairs:
            parts.append(
                "[DIRECT NEAR EVIDENCE - ONLY THESE PAIRS MAY BE DESCRIBED AS NEAR]\n"
                + "\n".join(f"- {line}" for line in near_pairs[:30])
            )
        v3_context = str(metadata.get("v3_structured_context") or "").strip()
        if v3_context:
            parts.append("[V3 STRUCTURED CONTEXT - MUST USE]\n" + v3_context)
        clean_context = str(state.clean_context or metadata.get("clean_context") or "").strip()
        if clean_context:
            parts.append("[CLEAN CONTEXT]\n" + clean_context)
        return "\n\n".join(parts)


    def _direct_near_evidence_lines(self, state: PipelineRunState) -> List[str]:
        evidence: list[str] = []
        seen: set[str] = set()
        sources = []
        sources.extend(state.raw_context or [])
        if state.clean_context:
            sources.extend(str(state.clean_context).splitlines())
        for line in sources:
            text = str(line or "").strip().lstrip("- ").strip()
            if not text or "[NEAR]" not in text:
                continue
            match = re.search(r"(.+?)\s+\[NEAR\]\s*->\s*(.+)", text)
            if not match:
                continue
            left = match.group(1).strip()
            right = match.group(2).strip()
            right = re.sub(r"\s+\([A-Z_]+\)\s*$", "", right).strip()
            pair = f"{left} [NEAR] -> {right}"
            key = normalize_text(pair, strip_punct=True)
            if key and key not in seen:
                seen.add(key)
                evidence.append(pair)
        return evidence


    def _answer_mentions_anchor(self, answer_norm: str, anchor: str) -> bool:
        anchor_norm = normalize_text(anchor, strip_punct=True)
        if not anchor_norm:
            return True
        if anchor_norm in answer_norm:
            return True
        tokens = [token for token in anchor_norm.split() if len(token) >= 3]
        if not tokens:
            return False
        required = 1 if len(tokens) <= 2 else max(2, min(len(tokens), int(len(tokens) * 0.6)))
        return sum(1 for token in tokens if token in answer_norm) >= required


    def _v3_query_requires_near_discussion(self, intent_data: Dict[str, Any], query_norm: str) -> bool:
        constraints = intent_data.get("constraints") or {}
        relations = [str(item or "").upper() for item in (constraints.get("relations") or [])]
        conditions_text = " ".join(str(item or "") for item in (constraints.get("required_conditions") or []))
        conditions_norm = normalize_text(conditions_text, strip_punct=True)
        if "NEAR" in relations:
            return True
        return any(term in query_norm or term in conditions_norm for term in ["gan", "lan can", "xung quanh", "near", "diem chung"])


    def _looks_like_dish_anchor(self, anchor: str) -> bool:
        norm = normalize_text(anchor, strip_punct=True)
        place_prefixes = ("nha hang", "quan", "khach san", "nha nghi", "homestay", "resort", "coffee", "pub")
        if norm.startswith(place_prefixes):
            return False
        return any(token in norm for token in ["mon", "mi", "bun", "pho", "com", "thit", "ca ", "banh", "lau"])

    def _build_tour_route_metadata(self, state: PipelineRunState, generator_candidates: list) -> None:
        """Build tour route optimization metadata for TOUR_PLAN mode."""
        p = self.pipeline
        constraints = state.metadata.get("constraints", {}) if isinstance(state.metadata, dict) else {}
        max_hop_km = float(constraints.get("max_hop_km_override") or p.TOUR_PLAN_MAX_HOP_KM)
        # Merged province: Pleiku ↔ Quy Nhơn ~100km, need larger hop
        if state.region_focus == RegionFocus.ALL:
            max_hop_km = max(max_hop_km, 120.0)
        requested_days = p.tour_route_optimizer.extract_trip_days(state.user_query)
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent

        # ── Coastal constraint early injection ────────────────────────────────
        # When user asks for a tour with beach ("có biển") from an inland region
        # (Gia Lai / Pleiku), the initial seeds won't have coastal nodes because
        # retrieval was scoped to inland_gia_lai. Inject Bình Định coastal seeds
        # BEFORE building route candidates, and switch to RegionFocus.ALL so the
        # hop guardrail doesn't discard Quy Nhơn points (~100 km from Pleiku).
        qs = state.query_plan
        coastal_injection_applied = False
        if (
            qs and qs.coastal_required
            and str(state.region_focus or "").lower() == "inland_gia_lai"
        ):
            coastal_seeds = self._fetch_constraint_seeds(qs, state)
            if coastal_seeds:
                logger.info(
                    f"   -> [CoastalInjection] Injecting {len(coastal_seeds)} coastal seeds "
                    f"from Bình Định into inland_gia_lai tour plan. Switching region to ALL."
                )
                state.all_seeds = list(state.all_seeds or []) + coastal_seeds
                # Switch region focus so optimizer allows cross-region hops
                state.region_focus = RegionFocus.ALL
                # Allow longer hops: Pleiku → Quy Nhơn ≈ 150 km via AH1 highway
                max_hop_km = max(max_hop_km, 150.0)
                coastal_injection_applied = True
            else:
                logger.info(
                    "   -> [CoastalInjection] No coastal seeds found in DB for Bình Định. "
                    "Proceeding with inland seeds only."
                )

        # ─────────────────────────────────────────────────────────────────────
        raw_route_nodes = p.tour_route_optimizer.build_tour_route_candidates(
            state.all_seeds or [],
            intent,
            query_state=state.query_plan,
        )
        # Merged province spans ~100km (Pleiku ↔ Quy Nhơn).
        # Don't allow guardrail to drop distant candidates — they're valid destinations.
        is_merged = state.region_focus == RegionFocus.ALL
        route_opt = p.tour_route_optimizer.optimize_tour_route_nodes(
            raw_route_nodes,
            optimize_distance=bool(constraints.get("optimize_distance", False)) and not is_merged,
            max_hop_km=max_hop_km,
            days=requested_days,
            detected_location=state.location,
            region_focus=state.region_focus,
        )
        state.runtime.metadata["route_seed_nodes"] = route_opt["nodes"]
        state.runtime.metadata["nearby_mode"] = route_opt["nearby_mode"]
        state.runtime.metadata["max_hop_km"] = route_opt["max_hop_km"]
        state.runtime.metadata["dropped_route_points"] = route_opt["dropped_route_points"]
        state.runtime.metadata["hop_distances_km"] = route_opt["hop_distances_km"]
        state.runtime.metadata["optimization_applied"] = route_opt["optimization_applied"]
        state.runtime.metadata["graph_ordering_applied"] = route_opt.get("graph_ordering_applied", False)
        state.runtime.metadata["route_engine"] = route_opt.get("route_engine", "local_fallback")
        state.runtime.metadata["daily_cluster_plan"] = route_opt.get("daily_cluster_plan", [])
        state.runtime.metadata["route_optimizer_metrics"] = route_opt["route_optimizer_metrics"]
        state.runtime.metadata["lodging_suggestions"] = route_opt.get("lodging_suggestions", [])
        # Propagate constraint warning (shown in FE if coastal/sunset/island not satisfied)
        state.runtime.metadata["constraint_warning"] = route_opt.get("constraint_warning")

        # Sanity gate: check if route meets user hard constraints
        qs = state.query_plan
        if qs and (qs.coastal_required or qs.sunset_required or qs.island_required):
            # Normalize route node names for matching (Vietnamese diacritics → non-diacritics)
            route_fields = []
            for n in route_opt["nodes"]:
                route_fields.append(str(n.get("name", "")))
                attrs = n.get("attributes") or {}
                route_fields.append(str(attrs.get("address", "")))
                route_fields.append(str(attrs.get("description", "")))
            route_text = normalize_text(" ".join(route_fields), strip_punct=True)
            # Extended coastal terms: include general terms so "biển" in description matches
            coastal_terms = {
                "ky co", "eo gio", "cu lao xanh", "hon kho", "nhon ly",
                "trung luong", "cat tien", "bai xep",
                "bai bien", "ven bien", "bien quy nhon", "bien binh dinh",
                "nhon hai", "nhon hoi", "ghenh rang",
                # catch-all: any node whose name/address contains "bien" counts
                "bien",
            }
            sunset_terms = {"hoang hon", "eo gio", "ky co"}
            island_terms = {"cu lao xanh", "hon kho", "ky co", "dao"}
            has_coastal = any(t in route_text for t in coastal_terms)
            has_sunset = any(t in route_text for t in sunset_terms)
            has_island = any(t in route_text for t in island_terms)

            needs_retry = False
            if qs.coastal_required and not has_coastal:
                needs_retry = True
            if qs.sunset_required and not has_sunset:
                needs_retry = True
            if qs.island_required and not has_island:
                needs_retry = True

            if needs_retry:
                logger.info("   -> [SanityGate] Route missing constraints. Attempting retry with constraint-matching seeds.", )
                retry_seeds = self._fetch_constraint_seeds(qs, state)
                if retry_seeds:
                    # Merge retry seeds with existing seeds and re-optimize
                    merged_seeds = list(state.all_seeds or []) + retry_seeds
                    retry_candidates = p.tour_route_optimizer.build_tour_route_candidates(
                        merged_seeds,
                        intent,
                        query_state=qs,
                    )
                    if len(retry_candidates) > len(raw_route_nodes):
                        retry_opt = p.tour_route_optimizer.optimize_tour_route_nodes(
                            retry_candidates,
                            optimize_distance=bool(constraints.get("optimize_distance", False)) and not is_merged,
                            max_hop_km=max_hop_km,
                            days=requested_days,
                            detected_location=state.location,
                            region_focus=state.region_focus,
                        )
                        # Verify retry actually fixed the constraint (normalized)
                        retry_fields = []
                        for n in retry_opt["nodes"]:
                            retry_fields.append(str(n.get("name", "")))
                            attrs = n.get("attributes") or {}
                            retry_fields.append(str(attrs.get("address", "")))
                            retry_fields.append(str(attrs.get("description", "")))
                        retry_text = normalize_text(" ".join(retry_fields), strip_punct=True)
                        retry_has_coastal = any(t in retry_text for t in coastal_terms)
                        retry_has_sunset = any(t in retry_text for t in sunset_terms)
                        retry_satisfied = True
                        if qs.coastal_required and not retry_has_coastal:
                            retry_satisfied = False
                        if qs.sunset_required and not retry_has_sunset:
                            retry_satisfied = False

                        if retry_satisfied:
                            logger.info("   -> [SanityGate] Retry SUCCESS: constraint-matching route found.", )
                            route_opt = retry_opt
                            raw_route_nodes = retry_candidates
                        else:
                            logger.error("   -> [SanityGate] Retry FAILED: still missing constraints. Blocking deterministic generation.", )
                            state.runtime.metadata["route_constraint_blocked"] = True
                    else:
                        logger.info("   -> [SanityGate] Retry produced no new candidates. Blocking.", )
                        state.runtime.metadata["route_constraint_blocked"] = True
                else:
                    logger.info("   -> [SanityGate] No constraint-matching seeds found in graph. Blocking.", )
                    state.runtime.metadata["route_constraint_blocked"] = True

    def _fetch_constraint_seeds(self, qs: Any, state: PipelineRunState) -> list:
        """Fetch TouristAttraction nodes matching coastal/sunset/island constraints from Neo4j.

        Uses coordinate-based broad fetch + Python-side normalized text matching
        because Neo4j's toLower() doesn't handle Vietnamese diacritics.
        """
        from graph_rag.core.state import NodeItem

        p = self.pipeline
        if not hasattr(p, 'driver') or not p.driver:
            return []

        # All constraint terms (normalized, no diacritics)
        all_terms = set()
        if getattr(qs, "coastal_required", False):
            all_terms |= {"ky co", "eo gio", "cu lao xanh", "hon kho", "nhon ly", "trung luong", "cat tien", "bai xep"}
        if getattr(qs, "sunset_required", False):
            all_terms |= {"hoang hon", "eo gio", "ky co"}
        if getattr(qs, "island_required", False):
            all_terms |= {"cu lao xanh", "hon kho", "ky co", "dao"}
        if not all_terms:
            return []

        # Broad fetch: all TouristAttraction in coastal region, then filter in Python
        # Coastal Quy Nhơn bounds: lat 13.5-14.0, lng 108.9-109.4
        cypher = """
            MATCH (n:TouristAttraction)
            WHERE n.location IS NOT NULL
              AND n.location.latitude >= 13.4 AND n.location.latitude <= 14.2
              AND n.location.longitude >= 108.7 AND n.location.longitude <= 109.5
            RETURN n.id AS id, n.name AS name, n.description AS description,
                   n.location.latitude AS lat, n.location.longitude AS lng,
                   n.address AS address, labels(n) AS labels
            LIMIT 50
        """

        try:
            with p.driver.session() as session:
                result = session.run(cypher)
                norm_terms = [normalize_text(t, strip_punct=True) for t in all_terms]
                seeds = []
                for record in result:
                    # Python-side normalized matching
                    fields = [
                        record.get("name") or "",
                        record.get("description") or "",
                        record.get("address") or "",
                    ]
                    text = normalize_text(" ".join(fields), strip_punct=True)
                    if not any(t in text for t in norm_terms):
                        continue
                    node = NodeItem(
                        id=record["id"],
                        content=record["name"] or "",
                        score=1.0,
                        source_type="constraint_recovery",
                        metadata={
                            "name": record["name"],
                            "description": record.get("description"),
                            "address": record.get("address"),
                            "lat": record["lat"],
                            "lng": record["lng"],
                            "labels": record["labels"] or ["TouristAttraction"],
                            "type": "TouristAttraction",
                        },
                    )
                    seeds.append(node)
                if seeds:
                    logger.info("   -> [SanityGate] _fetch_constraint_seeds: recovered %s constraint-matching TouristAttraction nodes", len(seeds))
                return seeds
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
            logger.error("   -> [SanityGate] _fetch_constraint_seeds error: %s", e)
            return []
