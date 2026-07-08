# graph_rag/modules/generation/llm_client.py
import os
import time
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from graph_rag.core.intents import IntentType
from graph_rag.core import keywords, thresholds, business_rules
from graph_rag.core.state import QuestionShape
from graph_rag.modules.generation import tour_plan_support
from graph_rag.services.ai_model import LLMService
from graph_rag.utils.text import normalize_text

from .answer_sanitizer import sanitize_answer_text, format_missing_attributes
from .prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Core LLM calling logic — delegates prompt building and sanitization."""

    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self._prompt_builder = PromptBuilder()

    # ------------------------------------------------------------------
    # Sanitization (delegated)
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_answer_text(answer: str) -> str:
        return sanitize_answer_text(answer)

    @staticmethod
    def _format_missing_attributes(attributes: List[str]) -> str:
        return format_missing_attributes(attributes)

    # ------------------------------------------------------------------
    # Prompt helpers (delegated)
    # ------------------------------------------------------------------
    def _critical_system_facts_block(self) -> str:
        return PromptBuilder.critical_system_facts_block()

    def _few_shot_examples_block(self) -> str:
        return PromptBuilder.few_shot_examples_block()

    def _build_system_prompt(
        self,
        answer_mode: Optional[str] = None,
        intent: Optional[str] = None,
        query_state: Optional[Any] = None,
    ) -> str:
        return PromptBuilder.build_system_prompt(
            answer_mode=answer_mode, intent=intent, query_state=query_state,
        )

    def _build_user_prompt(
        self,
        query: str,
        context: str,
        context_validation: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        detected_location: Optional[str] = None,
        query_state: Optional[Any] = None,
        validation_feedback: Optional[str] = None,
        partial_answer_mode: bool = False,
        missing_attrs_text: str = "",
    ) -> str:
        return PromptBuilder.build_user_prompt(
            query=query, context=context,
            context_validation=context_validation,
            intent=intent, detected_location=detected_location,
            query_state=query_state,
            validation_feedback=validation_feedback,
            partial_answer_mode=partial_answer_mode,
            missing_attrs_text=missing_attrs_text,
        )

    def _build_tour_plan_system_prompt(self) -> str:
        return PromptBuilder.build_tour_plan_system_prompt()

    # ------------------------------------------------------------------
    # Main generate entry point
    # ------------------------------------------------------------------
    def generate(
        self,
        user_query: str,
        context_text: str,
        intent: Optional[str] = None,
        detected_location: Optional[str] = None,
        candidate_nodes: Optional[List[Dict[str, Any]]] = None,
        strict_route_nodes: Optional[List[Dict[str, Any]]] = None,
        dropped_route_points: Optional[List[str]] = None,
        daily_cluster_plan: Optional[List[Dict[str, Any]]] = None,
        route_optimizer_metrics: Optional[Dict[str, Any]] = None,
        lodging_suggestions: Optional[List[Dict[str, Any]]] = None,
        context_validation: Optional[Dict[str, Any]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        query_state: Optional[Any] = None,
        validation_feedback: Optional[str] = None,
        answer_mode: Optional[str] = None,
    ) -> str:
        partial_answer_mode = False
        missing_attrs_text = ""
        from graph_rag.core.answer_mode import AnswerMode

        if answer_mode == AnswerMode.PARTIAL_FACT_ANSWER:
            partial_answer_mode = True
            missing_attributes = []
            if context_validation:
                missing_attributes = context_validation.get("missing_attributes") or []
            if not missing_attributes and query_state:
                missing_attributes = query_state.requested_attributes
            if missing_attributes:
                missing_attrs_text = self._format_missing_attributes(missing_attributes)

        if not partial_answer_mode and context_validation and context_validation.get("hard_fail"):
            missing_relations = context_validation.get("missing_relations") or []
            missing_attributes = context_validation.get("missing_attributes") or []
            if not missing_relations and missing_attributes and len(str(context_text or "").strip()) > 50:
                partial_answer_mode = True
                missing_attrs_text = self._format_missing_attributes(missing_attributes)
            else:
                has_relation_facts = bool(re.search(r"\[\w+\]\s*->", str(context_text or "")))
                if not has_relation_facts:
                    missing_parts = []
                    if missing_attributes:
                        missing_parts.append(self._format_missing_attributes(missing_attributes))
                    if missing_relations:
                        missing_parts.append("quan hệ dữ liệu: " + ", ".join(missing_relations))
                    missing_text = "; ".join(missing_parts) or "thông tin được hỏi"
                    return (
                        f"Xin lỗi, hệ thống dữ liệu du lịch hiện chưa có đủ "
                        f"{missing_text} để trả lời chính xác câu hỏi này."
                    )

        if self._should_use_transfer_route_template(user_query):
            return self._generate_transfer_route_answer(
                user_query=user_query,
                context_text=context_text,
                candidate_nodes=candidate_nodes or [],
                detected_location=detected_location,
            )

        if self._should_use_tour_plan_template(user_query, intent, detected_location):
            return self._generate_controlled_tour_plan(
                user_query=user_query,
                context_text=context_text,
                detected_location=detected_location,
                candidate_nodes=candidate_nodes or [],
                strict_route_nodes=strict_route_nodes or [],
                dropped_route_points=dropped_route_points or [],
                daily_cluster_plan=daily_cluster_plan or [],
                route_optimizer_metrics=route_optimizer_metrics or {},
                lodging_suggestions=lodging_suggestions or [],
            )

        system_prompt = self._build_system_prompt(
            answer_mode=answer_mode, intent=intent, query_state=query_state,
        )
        user_prompt = self._build_user_prompt(
            user_query, context_text,
            context_validation=context_validation,
            intent=intent, detected_location=detected_location,
            query_state=query_state,
            validation_feedback=validation_feedback,
            partial_answer_mode=partial_answer_mode,
            missing_attrs_text=missing_attrs_text,
        )

        if on_token:
            raw_answer = self.llm.generate_text_stream(system_prompt, user_prompt, on_token=on_token)
        else:
            raw_answer = self.llm.generate_text(system_prompt, user_prompt)
        return self._sanitize_answer_text(raw_answer)

    # ------------------------------------------------------------------
    # Tour plan detection & generation
    # ------------------------------------------------------------------
    def _should_use_tour_plan_template(
        self, query: str, intent: Optional[str], detected_location: Optional[str],
    ) -> bool:
        normalized_intent = str(intent or "").upper()
        if normalized_intent not in {IntentType.TOUR_PLAN}:
            return False
        normalized_query = normalize_text(query)
        normalized_loc = normalize_text(detected_location or "")
        compact_duration = bool(re.search(r"\b\d+\s*n\s*\d+\s*d\b", normalized_query))
        long_duration = bool(re.search(r"\b\d+\s*(?:ngay|nay|ngy)\b", normalized_query))
        has_duration = compact_duration or long_duration or any(
            token in normalized_query for token in keywords.TOUR_PLAN_SIGNALS
        )
        has_location = bool(normalized_loc) or any(
            token in normalized_query for token in business_rules.LOCATION_DISPLAY_NAMES
        )
        return has_duration and has_location

    def _should_use_transfer_route_template(self, query: str) -> bool:
        q = normalize_text(query)
        has_airport = "san bay" in q
        has_time_constraint = bool(
            re.search(r"\b\d{1,2}[:h]\d{0,2}\b", q)
            or re.search(r"\b(\d{1,2})\s*gio\b", q)
            or "phai co mat" in q
        )
        has_route_intent = any(
            token in q for token in [
                "doc duong", "lo trinh", "tren duong", "tu ", "ra san bay", "ghe nhanh",
            ]
        )
        return has_airport and has_time_constraint and has_route_intent

    def _generate_transfer_route_answer(
        self,
        user_query: str,
        context_text: str,
        candidate_nodes: List[Dict[str, Any]],
        detected_location: Optional[str],
    ) -> str:
        target_location = self._resolve_target_location(user_query, detected_location)
        structured = self._build_structured_context(candidate_nodes, detected_location=target_location)
        context_block = self._format_structured_context(structured, context_text)
        deadline = self._extract_deadline_time(user_query)

        system_prompt = PromptBuilder.build_transfer_route_system_prompt()
        user_prompt = PromptBuilder.build_transfer_route_user_prompt(
            user_query, context_block, deadline,
        )

        answer = self.llm.generate_text(system_prompt, user_prompt)
        verifier = self._verify_transfer_answer(answer)
        if not verifier["ok"]:
            retry_prompt = (
                user_prompt
                + "\n\nVERIFIER FEEDBACK (fix all):\n"
                + verifier["feedback"]
            )
            answer = self.llm.generate_text(system_prompt, retry_prompt)
        return self._sanitize_answer_text(answer)

    def _extract_deadline_time(self, query: str) -> str:
        return tour_plan_support.extract_deadline_time(query, normalize_text)

    def _verify_transfer_answer(self, answer: str) -> Dict[str, Any]:
        low = normalize_text(answer)
        issues: List[str] = []
        required = ["tom tat", "lich trinh theo gio", "diem ghe nhanh", "an trua", "duong di"]
        for r in required:
            if r not in low:
                issues.append(f"Missing section: {r}")
        time_hits = re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", answer)
        if len(time_hits) < 3:
            issues.append("Need at least 3 explicit time points.")
        return {
            "ok": len(issues) == 0,
            "feedback": "\n".join([f"- {x}" for x in issues]) if issues else "",
        }

    def _generate_controlled_tour_plan(
        self,
        user_query: str,
        context_text: str,
        detected_location: Optional[str],
        candidate_nodes: List[Dict[str, Any]],
        strict_route_nodes: List[Dict[str, Any]],
        dropped_route_points: List[str],
        daily_cluster_plan: List[Dict[str, Any]],
        route_optimizer_metrics: Dict[str, Any],
        lodging_suggestions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        target_location = self._resolve_target_location(user_query, detected_location)
        constraints = self._extract_trip_constraints(user_query)
        relaxed_guardrails = bool(constraints.get("relaxed_route_guardrails", False))
        route_guardrails_enabled = len(strict_route_nodes or []) > 0 and not relaxed_guardrails
        strict_candidate_nodes = (
            strict_route_nodes if route_guardrails_enabled
            else (candidate_nodes or strict_route_nodes)
        )
        structured = self._build_structured_context(
            strict_candidate_nodes, target_location, trusted_only=route_guardrails_enabled,
        )
        context_block = self._format_structured_context(structured, context_text)
        trip_days, trip_nights = self._extract_trip_duration(user_query)

        optimizer_has_nodes = len(strict_route_nodes or []) > 0
        optimizer_has_day_plan = len(daily_cluster_plan or []) > 0
        optimizer_has_metrics = bool(route_optimizer_metrics)

        skeleton = ""
        if optimizer_has_day_plan and (optimizer_has_nodes or optimizer_has_metrics):
            skeleton = self._build_compact_skeleton(
                days=trip_days,
                structured_items=structured,
                daily_cluster_plan=daily_cluster_plan,
                lodging_suggestions=lodging_suggestions,
            )
            logger.info(
                "TourPlan compact skeleton built: days=%s nodes=%s clusters=%s chars=%s",
                trip_days, len(strict_route_nodes or []),
                len(daily_cluster_plan or []), len(skeleton),
            )

        system_prompt = self._build_tour_plan_system_prompt()
        answer = ""
        verifier = {"ok": False, "feedback": ""}
        max_retries = max(1, int(os.getenv("TOUR_PLAN_MAX_RETRIES", "2")))
        for attempt in range(max_retries):
            retry_prompt = PromptBuilder.build_tour_plan_user_prompt(
                query=user_query,
                context=context_block,
                target_location=target_location,
                days=trip_days,
                nights=trip_nights,
                constraints=constraints,
                verifier_feedback=verifier["feedback"] if answer else None,
                strict_route_nodes=strict_route_nodes,
                dropped_route_points=dropped_route_points,
                daily_cluster_plan=daily_cluster_plan,
                lodging_suggestions=lodging_suggestions,
                skeleton=skeleton,
            )
            attempt_start = time.time()
            answer = self.llm.generate_text(system_prompt, retry_prompt, max_tokens=4096)
            verifier = self._verify_tour_plan_answer(
                answer=answer, user_query=user_query,
                days=trip_days, nights=trip_nights,
                constraints=constraints,
                strict_route_nodes=strict_route_nodes,
                dropped_route_points=dropped_route_points,
            )
            logger.info(
                "tour_plan_generation_attempt=%s/%s ok=%s latency=%.2fs",
                attempt + 1, max_retries,
                verifier["ok"], time.time() - attempt_start,
            )
            if not verifier["ok"] and verifier.get("feedback"):
                logger.info("tour_plan_verify_feedback attempt=%s: %s", attempt + 1, verifier["feedback"])
            if verifier["ok"]:
                break

        if not verifier["ok"] and verifier.get("feedback"):
            logger.warning("Tour plan verification failed after retries: %s", verifier["feedback"])

        if not verifier["ok"]:
            if len(structured) >= 2 or len(daily_cluster_plan or []) > 0:
                return self._build_deterministic_tour_plan(
                    days=trip_days, nights=trip_nights,
                    target_location=target_location,
                    constraints=constraints,
                    structured_items=structured,
                    daily_cluster_plan=daily_cluster_plan,
                    lodging_suggestions=lodging_suggestions,
                )
            return self._build_tour_plan_safe_fallback(
                user_query=user_query,
                days=trip_days, nights=trip_nights,
                constraints=constraints,
                structured_items=structured,
                target_location=target_location,
                relaxed_mode=relaxed_guardrails,
                lodging_suggestions=lodging_suggestions,
            )

        return self._sanitize_answer_text(answer)

    # ------------------------------------------------------------------
    # Structured context building
    # ------------------------------------------------------------------
    def _build_structured_context(
        self,
        candidate_nodes: List[Dict[str, Any]],
        detected_location: Optional[str],
        trusted_only: bool = False,
    ) -> List[Dict[str, Any]]:
        pool = []
        for node in candidate_nodes:
            labels = node.get("labels") or []
            attrs = node.get("attributes") or {}
            node_type = (node.get("type") or "").strip()
            if not node_type and labels:
                node_type = str(labels[0]).strip()
            if not node_type:
                node_type = str(attrs.get("type") or "").strip()
            if node_type not in {"TouristAttraction", "Accommodation", "Tour", "Restaurant"}:
                continue
            name = (node.get("name") or attrs.get("name") or "").strip()
            if not name:
                continue
            low_name = normalize_text(name)
            rating = self._to_float(
                node.get("rating") if node.get("rating") is not None else attrs.get("rating")
            )
            rating_value = rating if rating is not None else 0.0
            tags = node.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]
            if not tags:
                tags = attrs.get("tags") or []
                if isinstance(tags, str):
                    tags = [tags]
            if node_type == "Restaurant" and not trusted_only:
                tags_norm = [normalize_text(str(t)) for t in tags]
                has_local_specialty = any("local specialty" in t or "dac san" in t for t in tags_norm)
                if rating_value < thresholds.MIN_RESTAURANT_RATING and not has_local_specialty:
                    continue
                if any(x in low_name for x in business_rules.RESTAURANT_EXCLUDE_NAMES):
                    continue
            loc = (node.get("location") or attrs.get("location") or "").strip()
            if not loc:
                loc = (node.get("address") or attrs.get("address") or "").strip()
            lat = self._to_float(
                node.get("lat") if node.get("lat") is not None else attrs.get("lat")
            )
            lng = self._to_float(
                node.get("lng") if node.get("lng") is not None else attrs.get("lng")
            )
            distance_km = self._distance_from_center(lat, lng, detected_location)
            location_cluster = self._infer_location_cluster(loc, distance_km, detected_location)
            best_time = self._infer_best_time(node_type, name)
            pool.append({
                "id": node.get("id"),
                "name": name,
                "type": node_type,
                "rating": rating,
                "location": loc,
                "location_cluster": location_cluster,
                "best_time": best_time,
                "distance_km": distance_km,
            })

        dedup: Dict[str, Dict[str, Any]] = {}
        for item in pool:
            key = normalize_text(item["name"])
            old = dedup.get(key)
            if not old or item["rating"] > old["rating"]:
                dedup[key] = item

        filtered = list(dedup.values())
        filtered = self._filter_by_location(filtered, detected_location)
        if not filtered and dedup:
            filtered = list(dedup.values())
        return self._select_diverse_nodes(filtered, max_items=8)

    def _filter_by_location(
        self,
        items: List[Dict[str, Any]],
        detected_location: Optional[str],
    ) -> List[Dict[str, Any]]:
        normalized_loc = normalize_text(detected_location or "")
        if not normalized_loc:
            return items

        is_gia_lai = any(token in normalized_loc for token in ["gia lai", "pleiku"])
        is_quy_nhon = any(token in normalized_loc for token in ["quy nhon", "binh dinh"])
        merged_province_terms = (
            keywords.IN_SCOPE_REGION_TERMS | keywords.COASTAL_KEYWORDS | keywords.INLAND_KEYWORDS
        )
        outside_terms = keywords.OUT_OF_REGION_TERMS
        disallowed_terms = outside_terms if (is_gia_lai or is_quy_nhon) else set()

        kept = []
        for item in items:
            loc = normalize_text(item.get("location") or "")
            if any(term in loc for term in disallowed_terms):
                continue
            if not loc:
                if item.get("distance_km") is not None and item["distance_km"] <= thresholds.LOCATION_FILTER_MAX_KM:
                    kept.append(item)
                continue
            if normalized_loc in loc:
                kept.append(item)
                continue
            if (is_gia_lai or is_quy_nhon) and any(token in loc for token in merged_province_terms):
                kept.append(item)
                continue
            if item.get("distance_km") is not None and item["distance_km"] <= 35.0:
                kept.append(item)
                continue
            if loc and not any(term in loc for term in outside_terms):
                kept.append(item)
        return kept

    def _select_diverse_nodes(
        self, items: List[Dict[str, Any]], max_items: int,
    ) -> List[Dict[str, Any]]:
        priority = sorted(
            items,
            key=lambda x: (
                x.get("distance_km") if x.get("distance_km") is not None else 999.0,
                -(x.get("rating") or 0.0),
                x.get("name", ""),
            ),
        )
        selected: List[Dict[str, Any]] = []
        for required_type in ["TouristAttraction", "Accommodation", "Tour"]:
            best = next((i for i in priority if i["type"] == required_type and i not in selected), None)
            if best:
                selected.append(best)
        for item in priority:
            if len(selected) >= max_items:
                break
            if item not in selected:
                selected.append(item)
        return selected[:max_items]

    def _format_structured_context(
        self, structured_items: List[Dict[str, Any]], fallback_text: str,
    ) -> str:
        if not structured_items:
            return fallback_text or "No trusted places after location and safety filtering."
        lines = ["["]
        for item in structured_items:
            lines.append(
                '  {'
                f'"name": "{item.get("name", "")}", '
                f'"type": "{item.get("type", "")}", '
                f'"best_time": "{item.get("best_time", "")}", '
                f'"location_cluster": "{item.get("location_cluster", "")}", '
                f'"location": "{item.get("location", "")}"'
                "},"
            )
        lines.append("]")
        lines.append("\nNote: Only trusted filtered places are listed above. Ignore any out-of-region facts.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tour plan verifier
    # ------------------------------------------------------------------
    def _verify_tour_plan_answer(
        self,
        answer: str,
        user_query: str,
        days: int,
        nights: int,
        constraints: Dict[str, bool],
        strict_route_nodes: List[Dict[str, Any]],
        dropped_route_points: List[str],
    ) -> Dict[str, Any]:
        low = normalize_text(answer)
        issues: List[str] = []

        for day in range(1, max(1, days) + 1):
            if f"ngay {day}" not in low and f"day {day}" not in low:
                issues.append(f"Missing day section: day {day}.")
        if str(days) not in low and str(nights) not in low:
            issues.append("Duration summary is unclear (days/nights not reflected).")

        time_hits = re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", answer)
        min_slots = max(4, days * 2)
        if len(time_hits) < min_slots:
            issues.append(f"Not enough realistic time slots (need at least {min_slots}).")

        # Check that each day section has at least 4 time hits (2 slots)
        day_sections = re.split(r"(?:^|\n).{0,10}?(?:Ngày|Day|NGÀY)\s*\d+", answer, flags=re.IGNORECASE)
        if len(day_sections) > 1:
            for d_idx, day_content in enumerate(day_sections[1:max(1, days) + 1], start=1):
                day_time_hits = re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", day_content)
                logger.debug("[Verifier Debug] Day %d hits (%d): %s", d_idx, len(day_time_hits), day_time_hits)
                if len(day_time_hits) < 6:
                    issues.append(f"Ngày {d_idx} không có đủ mốc thời gian cụ thể (cần ít nhất 6 mốc thời gian hh:mm hoặc 3 khung giờ để tạo timeline).")
        logger.debug("[Verifier Debug] day_sections length: %d, days: %d, current issues: %s", len(day_sections), days, issues)

        if days >= 2:
            if "nghi dem" not in low and "nghỉ đêm" not in low and "gợi ý nghỉ đêm" not in low:
                issues.append("Multi-day tour must include a 'Gợi ý nghỉ đêm' section.")
        elif not any(term in low for term in tour_plan_support.ACCOMMODATION_TERMS):
            issues.append("Itinerary should include accommodation mention.")
        if "tour package" not in low and "goi tour" not in low and "chi phi" not in low:
            issues.append("Cost estimation appears incomplete.")
        if len(low.rstrip()) < 200:
            issues.append("Answer may be truncated: output is too short.")

        if constraints.get("no_cano") and any(
            token in low for token in ["cano", "ca no", "tau cao toc", "speedboat", "jet ski", "moto nuoc"]
        ):
            issues.append("Violation: no-cano constraint breached.")
        if constraints.get("no_climb") and any(
            token in low for token in ["leo nui", "trek", "trekking", "chinh phuc dinh", "duong bac thang", "leo doc"]
        ):
            issues.append("Violation: no-climb constraint breached.")
        if constraints.get("low_mobility") and any(
            token in low for token in ["di bo xa", "quang duong dai", "van dong manh", "lich day", "intense", "adventure"]
        ):
            issues.append("Violation: low-mobility pace is not respected.")

        user_low = normalize_text(user_query)
        if "khong" in user_low and "cano" in user_low and "cano" in low:
            issues.append("Violation: explicit user ban for cano is violated.")
        if "khong" in user_low and "leo" in user_low and any(
            token in low for token in ["leo nui", "leo doc", "trek", "trekking"]
        ):
            issues.append("Violation: explicit user ban for climbing is violated.")
        if "da duoc kiem tra va dam bao" in low or "đã được kiểm tra và đảm bảo" in answer.lower():
            issues.append("Do not claim guaranteed compliance; provide practical assumptions instead.")

        for dropped_name in (dropped_route_points or []):
            norm_dropped = normalize_text(dropped_name)
            if norm_dropped and norm_dropped in low:
                issues.append(f"Violation: dropped_route_points contains forbidden place '{dropped_name}'.")

        allowed_names = [normalize_text(n.get("name") or "") for n in (strict_route_nodes or [])]
        allowed_names = [n for n in allowed_names if n]
        if allowed_names:
            covered = sum(1 for n in allowed_names if n in low)
            if covered < min(2, len(allowed_names)):
                issues.append("Itinerary does not use enough Route Optimizer approved places.")

        return {
            "ok": len(issues) == 0,
            "feedback": "\n".join([f"- {x}" for x in issues]) if issues else "",
        }

    def _build_compact_skeleton(
        self,
        days: int,
        structured_items: List[Dict[str, Any]],
        daily_cluster_plan: Optional[List[Dict[str, Any]]] = None,
        lodging_suggestions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        lines: List[str] = []
        type_map: Dict[str, str] = {}
        for item in (structured_items or []):
            name = str(item.get("name") or "").strip()
            itype = str(item.get("type") or "").strip()
            if name:
                type_map[normalize_text(name)] = itype

        for plan in (daily_cluster_plan or []):
            day = plan.get("day")
            areas = plan.get("areas") or []
            points = plan.get("point_names") or []
            region = plan.get("region_label") or ""
            region_str = f" [{region}]" if region else ""
            area_str = ", ".join(areas) if areas else "?"
            lines.append(f"Ngày {day}{region_str} — Khu vực: {area_str}")
            for pt in points:
                pt_type = type_map.get(normalize_text(str(pt)), "")
                tag = f" ({pt_type})" if pt_type else ""
                lines.append(f"  - {pt}{tag}")
            lines.append("")

        if lodging_suggestions and days >= 2:
            lines.append("Lưu trú:")
            for i, lodge in enumerate(lodging_suggestions[:days - 1], start=1):
                name = lodge.get("name") or ""
                addr = lodge.get("address") or ""
                if name:
                    lines.append(f"  Đêm {i}: {name}" + (f" — {addr}" if addr else ""))

        return "\n".join(lines).strip() or "(No skeleton available)"

    def _build_deterministic_tour_plan(
        self,
        days: int,
        nights: int,
        target_location: str,
        constraints: Dict[str, bool],
        structured_items: List[Dict[str, Any]],
        daily_cluster_plan: Optional[List[Dict[str, Any]]] = None,
        lodging_suggestions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not structured_items:
            return self._build_tour_plan_safe_fallback(
                user_query="", days=days, nights=nights,
                constraints=constraints,
                structured_items=structured_items,
                target_location=target_location,
                lodging_suggestions=lodging_suggestions,
            )

        remaining_attractions = []
        remaining_restaurants = []
        seen_rem = set()
        for item in structured_items:
            name = item.get("name")
            if not name:
                continue
            norm = normalize_text(str(name))
            if norm in seen_rem:
                continue
            seen_rem.add(norm)
            if item.get("type") == "Restaurant":
                remaining_restaurants.append(item)
            else:
                remaining_attractions.append(item)

        by_norm_name: Dict[str, Dict[str, Any]] = {}
        for item in structured_items:
            raw_name = str(item.get("name") or "").strip()
            if raw_name:
                by_norm_name[normalize_text(raw_name)] = item

        daily_by_day: Dict[int, Dict[str, Any]] = {}
        for plan in (daily_cluster_plan or []):
            day_value = plan.get("day")
            if isinstance(day_value, int):
                daily_by_day[day_value] = plan

        used_names: set = set()
        used_names_display: Dict[str, str] = {}
        day_lines: List[str] = []
        for day in range(1, max(1, days) + 1):
            day_lines.append(f"### Ngày {day}")
            day_plan = daily_by_day.get(day) or {}
            region_label = day_plan.get("region_label", "")
            if region_label:
                day_lines.append(f"*📍 Khu vực: {region_label}*")
            point_names = day_plan.get("point_names") or []
            day_attractions: List[Tuple[str, str]] = []
            day_restaurants: List[Tuple[str, str]] = []
            for point_name in point_names:
                norm = normalize_text(str(point_name))
                if norm in used_names:
                    continue
                item = by_norm_name.get(norm)
                display_name = str(item.get("name")) if item and item.get("name") else str(point_name).strip()
                itype = str(item.get("type") or "") if item else ""
                if itype == "Restaurant":
                    day_restaurants.append((display_name, norm))
                else:
                    day_attractions.append((display_name, norm))

            # Morning slot
            if day_attractions:
                morning_disp, morning_norm = day_attractions.pop(0)
                used_names.add(morning_norm)
                used_names_display[morning_norm] = morning_disp
            else:
                found = False
                while remaining_attractions:
                    cand = remaining_attractions.pop(0)
                    c_norm = normalize_text(str(cand.get("name")))
                    if c_norm not in used_names:
                        morning_disp = cand.get("name")
                        used_names.add(c_norm)
                        used_names_display[c_norm] = morning_disp
                        found = True
                        break
                if not found:
                    morning_disp = "tham quan khu trung tâm"

            # Noon slot
            if day_restaurants:
                noon_disp, noon_norm = day_restaurants.pop(0)
                used_names.add(noon_norm)
                used_names_display[noon_norm] = noon_disp
            elif day_attractions:
                noon_disp, noon_norm = day_attractions.pop(0)
                used_names.add(noon_norm)
                used_names_display[noon_norm] = noon_disp
            else:
                found = False
                while remaining_restaurants:
                    cand = remaining_restaurants.pop(0)
                    c_norm = normalize_text(str(cand.get("name")))
                    if c_norm not in used_names:
                        noon_disp = cand.get("name")
                        used_names.add(c_norm)
                        used_names_display[c_norm] = noon_disp
                        found = True
                        break
                if not found:
                    while remaining_attractions:
                        cand = remaining_attractions.pop(0)
                        c_norm = normalize_text(str(cand.get("name")))
                        if c_norm not in used_names:
                            noon_disp = cand.get("name")
                            used_names.add(c_norm)
                            used_names_display[c_norm] = noon_disp
                            found = True
                            break
                if not found:
                    noon_disp = "ăn trưa đặc sản địa phương"

            # Afternoon slot
            if day_attractions:
                afternoon_disp, afternoon_norm = day_attractions.pop(0)
                used_names.add(afternoon_norm)
                used_names_display[afternoon_norm] = afternoon_disp
            else:
                found = False
                while remaining_attractions:
                    cand = remaining_attractions.pop(0)
                    c_norm = normalize_text(str(cand.get("name")))
                    if c_norm not in used_names:
                        afternoon_disp = cand.get("name")
                        used_names.add(c_norm)
                        used_names_display[c_norm] = afternoon_disp
                        found = True
                        break
                if not found:
                    afternoon_disp = "thư giãn và dạo quanh khu vực gần"

            day_lines.append(f"- 08:00 - 10:30: {morning_disp}")
            day_lines.append(f"- 11:30 - 13:00: {noon_disp}")
            day_lines.append(f"- 14:30 - 17:00: {afternoon_disp}")

        lodging_lines: List[str] = []
        if days >= 2:
            if lodging_suggestions:
                for i, lodge in enumerate(lodging_suggestions[:nights], start=1):
                    name = lodge.get("name") or ""
                    address = lodge.get("address") or ""
                    phone = lodge.get("phone") or ""
                    if name:
                        line = f"- Đêm {i}: {name}"
                        if address:
                            line += f" - {address}"
                        if phone:
                            line += f" (ĐT: {phone})"
                        lodging_lines.append(line)
            if not lodging_lines:
                for i in range(1, nights + 1):
                    lodging_lines.append(f"- Đêm {i}: Chưa có dữ liệu khách sạn cụ thể")

        hard_rules = []
        if constraints.get("no_cano"):
            hard_rules.append("- Không dùng cano/tàu cao tốc.")
        if constraints.get("no_climb"):
            hard_rules.append("- Không leo dốc/leo núi/bậc thang dài.")
        if constraints.get("low_mobility"):
            hard_rules.append("- Lịch đi chậm, thêm điểm nghỉ ngắn giữa các chặng.")
        if not hard_rules:
            hard_rules.append("- Không có ràng buộc cứng bổ sung.")

        trusted_names = sorted(list(used_names_display.values()))
        trusted_list = (
            "\n".join([f"- {name}" for name in trusted_names])
            if trusted_names else "- Chưa có"
        )

        lodging_section = ""
        if lodging_lines:
            lodging_section = (
                "### 3. Gợi ý nghỉ đêm\n"
                f"{chr(10).join(lodging_lines)}\n\n"
            )

        region_labels = []
        for plan in (daily_cluster_plan or []):
            label = plan.get("region_label", "")
            if label and label not in region_labels:
                region_labels.append(label)

        if len(region_labels) >= 2:
            day_regions = []
            for plan in (daily_cluster_plan or []):
                r_label = plan.get("region_label")
                if r_label:
                    day_regions.append(f"Ngày {plan.get('day')} tại {r_label}")
            overview = (
                f"Hành trình liên tỉnh {' → '.join(region_labels)}, "
                f"ưu tiên di chuyển hợp lý theo ngày: "
                f"{', '.join(day_regions)}.\n\n"
            )
        else:
            overview = (
                "Lịch trình này được dựng từ các địa điểm đã qua lọc vùng và độ tin cậy, "
                "ưu tiên di chuyển hợp lý theo ngày.\n\n"
            )

        return (
            f"## Lịch trình gợi ý {days} ngày {nights} đêm: {target_location}\n\n"
            "### 1. Tổng quan\n"
            f"{overview}"
            "### 2. Lịch trình theo ngày\n"
            f"{chr(10).join(day_lines)}\n\n"
            f"{lodging_section}"
            "### 4. Ước tính chi phí\n"
            f"- {business_rules.TOUR_COST_TEMPLATE} (tùy mùa và hạng dịch vụ).\n"
            "- Tự túc: lưu trú + ăn uống + vé tham quan thường thấp hơn nếu tối giản điểm đi xa.\n\n"
            "### 5. Lưu ý\n"
            "- Kiểm tra thời tiết trước mỗi ngày đi.\n"
            "- Ưu tiên điểm gần nhau để giảm thời gian di chuyển.\n"
            "- Xác nhận giờ mở cửa điểm tham quan trước khi khởi hành.\n\n"
            "### 6. Checklist ràng buộc\n"
            f"{chr(10).join(hard_rules)}\n\n"
            "### 7. Danh sách điểm đã dùng (trusted)\n"
            f"{trusted_list}\n"
        )

    def _build_tour_plan_safe_fallback(
        self,
        user_query: str,
        days: int,
        nights: int,
        constraints: Dict[str, bool],
        structured_items: List[Dict[str, Any]],
        target_location: str,
        relaxed_mode: bool = False,
        lodging_suggestions: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        _ = user_query
        location_label = target_location or "khu vực bạn yêu cầu"
        if not structured_items:
            if relaxed_mode:
                return (
                    "## Chưa đủ dữ liệu để dựng lịch trình chi tiết ngay\n\n"
                    f"Mình đã nới ngưỡng di chuyển theo yêu cầu, nhưng dữ liệu điểm đến "
                    f"tin cậy tại {location_label} vẫn chưa đủ để xếp lịch {days} ngày "
                    f"{nights} đêm theo giờ một cách chắc chắn.\n\n"
                    "Bạn có thể chọn 1 trong 2 cách để mình tiếp tục:\n"
                    "- Giữ 2 ngày nhưng cho phép mình dùng thêm điểm gợi ý (mức tin cậy trung bình).\n"
                    "- Chuyển sang lịch trình 1 ngày để đảm bảo chất lượng."
                )
            return (
                "## Chưa đủ dữ liệu để dựng lịch trình chi tiết\n\n"
                f"Hiện dữ liệu địa điểm đáng tin cậy trong khu vực {location_label} "
                f"còn mỏng để dựng lịch trình {days} ngày {nights} đêm "
                "theo giờ một cách chắc chắn.\n\n"
                "### Ràng buộc đã áp dụng\n"
                f"- Không cano: {'Có' if constraints.get('no_cano') else 'Không'}\n"
                f"- Không leo trèo: {'Có' if constraints.get('no_climb') else 'Không'}\n"
                f"- Nhịp độ chậm/ít di chuyển: {'Có' if constraints.get('low_mobility') else 'Không'}\n\n"
                "Bạn có thể thử lại với yêu cầu ngắn hơn (1 ngày), hoặc cho phép thêm "
                "các điểm gợi ý mức tin cậy trung bình để mình dựng lịch trình đầy đủ hơn."
            )

        names = [x.get("name", "") for x in structured_items[:6] if x.get("name")]
        place_list = (
            "\n".join([f"- {name}" for name in names])
            if names else "- Chưa có địa điểm rõ ràng"
        )

        lodging_section = ""
        if days >= 2 and lodging_suggestions:
            lodging_items = []
            for i, lodge in enumerate(lodging_suggestions[:nights], start=1):
                name = lodge.get("name") or ""
                address = lodge.get("address") or ""
                if name:
                    line = f"- Đêm {i}: {name}"
                    if address:
                        line += f" - {address}"
                    lodging_items.append(line)
            if lodging_items:
                lodging_section = (
                    "\n### Gợi ý nghỉ đêm\n"
                    + "\n".join(lodging_items)
                    + "\n"
                )

        return (
            f"## Gợi ý an toàn tạm thời cho {days} ngày {nights} đêm\n\n"
            f"Mình đã loại bỏ các điểm ngoài vùng {location_label} hoặc không phù hợp ràng buộc. "
            "Đây là danh sách điểm đáng tin cậy để bạn chọn nhanh:\n"
            f"{place_list}\n"
            f"{lodging_section}\n"
            "Nếu bạn muốn, mình sẽ lập lại lịch chi tiết theo giờ chỉ với các điểm trên "
            "để đảm bảo không vi phạm no-cano/no-climb/low-mobility."
        )

    def _resolve_target_location(self, query: str, detected_location: Optional[str]) -> str:
        q = normalize_text(query)
        matched_provinces = []
        for key, display in business_rules.LOCATION_DISPLAY_NAMES.items():
            if key in q:
                idx = q.find(key)
                matched_provinces.append((idx, display))
        matched_provinces.sort(key=lambda x: x[0])
        unique_provinces = []
        for _, display in matched_provinces:
            if display not in unique_provinces:
                unique_provinces.append(display)
        if len(unique_provinces) >= 2:
            return " - ".join(unique_provinces)
        if unique_provinces:
            return unique_provinces[0]

        loc = normalize_text(detected_location or "")
        matched_loc_provinces = []
        for key, display in business_rules.LOCATION_DISPLAY_NAMES.items():
            if key in loc:
                matched_loc_provinces.append((loc.find(key), display))
        matched_loc_provinces.sort(key=lambda x: x[0])
        unique_loc_provinces = []
        for _, display in matched_loc_provinces:
            if display not in unique_loc_provinces:
                unique_loc_provinces.append(display)
        if len(unique_loc_provinces) >= 2:
            return " - ".join(unique_loc_provinces)
        if unique_loc_provinces:
            return unique_loc_provinces[0]
        return detected_location or "khu vực đã hỏi"

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _extract_trip_duration(self, query: str) -> Tuple[int, int]:
        return tour_plan_support.extract_trip_duration(query, normalize_text)

    def _extract_trip_constraints(self, query: str) -> Dict[str, bool]:
        return tour_plan_support.extract_trip_constraints(query, normalize_text)

    def _infer_best_time(self, node_type: str, name: str) -> str:
        n = normalize_text(name)
        if node_type == "Restaurant":
            return "afternoon"
        if any(token in n for token in keywords.COASTAL_KEYWORDS):
            return "morning"
        if node_type == "Accommodation":
            return "evening"
        return "morning"

    def _infer_location_cluster(
        self, location: str, distance_km: Optional[float], detected_location: Optional[str],
    ) -> str:
        return tour_plan_support.infer_location_cluster(
            location, distance_km, detected_location, normalize_text,
        )

    def _distance_from_center(
        self, lat: Optional[float], lng: Optional[float], detected_location: Optional[str],
    ) -> Optional[float]:
        return tour_plan_support.distance_from_center(
            lat, lng, detected_location, normalize_text,
        )

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None
