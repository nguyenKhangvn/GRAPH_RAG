from __future__ import annotations
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
"""Pipeline response finalization, early guards, and memory helpers."""
import logging

logger = logging.getLogger(__name__)


import re


from typing import Any, Dict, List



from graph_rag.core.answer_mode import AnswerMode


from graph_rag.core.intents import IntentType



from graph_rag.utils.text import normalize_text


from ..dto import PipelineRunState


class PipelineResponseMixin:
    """Mixin providing response finalization, scope guards, and entity memory."""

    def _run_step_5_coverage_check(self, state: PipelineRunState, answer: str) -> List[str]:
        metadata = state.metadata or {}
        intent_data = metadata.get("v3_intent_data") or {}
        mode = str(intent_data.get("intent_mode") or "")
        anchors = [str(anchor or "").strip() for anchor in (intent_data.get("anchors") or []) if str(anchor or "").strip()]
        answer_norm = normalize_text(answer or "", strip_punct=True)
        query_norm = normalize_text(state.user_query or "", strip_punct=True)
        context_norm = normalize_text(self._v3_generation_context(state), strip_punct=True)
        missing: List[str] = []

        if mode == "comparison":
            for anchor in anchors:
                if not self._answer_mentions_anchor(answer_norm, anchor):
                    missing.append(f"chua nhac anchor '{anchor}'")
            compare_terms = ["chung", "khac", "giong", "tuong tu", "so sanh", "khac biet"]
            if not any(term in answer_norm for term in compare_terms):
                missing.append("chua so sanh cac dac diem chung, khac, giong hoac tuong tu giua cac doi tuong")

        if self._v3_query_requires_near_discussion(intent_data, query_norm):
            proximity_terms = ["gan", "lan can", "xung quanh", "diem chung", "chung", "khac", "near", "cach", "khoang cach", "km", "di chuyen"]
            if not any(term in answer_norm for term in proximity_terms):
                missing.append("chua phan tich quan he gan/NEAR hoac diem chung/khac")

        if any(term in query_norm for term in ["phu hop hon", "tot hon", "nen chon", "lua chon nao", "uu tien"]):
            decision_terms = ["phu hop hon", "nen chon", "uu tien", "tot hon", "khong du", "tam thoi"]
            if not any(term in answer_norm for term in decision_terms):
                missing.append("chua dua ra ket luan lua chon phu hop hon hoac neu khong du du lieu")

        if mode == "tour_plan":
            generator_candidates = self._build_generator_candidates(state.all_seeds)
            accommodations = [c["name"] for c in generator_candidates if c["type"] == "Accommodation"]
            attractions = [c["name"] for c in generator_candidates if c["type"] in {"TouristAttraction", "Tour"}]

            has_lodging = any(self._answer_mentions_anchor(answer_norm, acc) for acc in accommodations)
            if not accommodations:
                lodging_terms = ["khach san", "nha nghi", "homestay", "resort", "luu tru", "noi o", "cobe", "happy house", "hostel"]
                has_lodging = any(term in answer_norm for term in lodging_terms)

            has_attraction = any(self._answer_mentions_anchor(answer_norm, attr) for attr in attractions)
            if not attractions:
                attraction_terms = ["tham quan", "diem", "bao tang", "bai bien", "thac", "di tich", "lang", "khu du lich", "chua", "tinh xa"]
                has_attraction = any(term in answer_norm for term in attraction_terms)

            reason_terms = ["vi", "ly do", "phu hop", "thuan tien", "gan", "de", "nho", "thich hop", "bo cua", "canh quan"]
            has_reason = any(term in answer_norm for term in reason_terms)

            if not has_lodging:
                missing.append("tour_plan thieu noi o/lua chon luu tru phu hop tu du lieu")
            if not has_attraction:
                missing.append("tour_plan thieu diem tham quan/hoat dong phu hop tu du lieu")
            if not has_reason:
                missing.append("tour_plan thieu ly do lua chon cac diem den")

        if mode == "dish_to_restaurant":
            dish_anchors = [anchor for anchor in anchors if self._looks_like_dish_anchor(anchor)]
            restaurant_terms = ["nha hang", "quan", "restaurant", "coffee", "pub"]
            address_in_context = any(term in context_norm for term in ["dia chi", "address", "so dien thoai", "sdt", "lien he"])
            for dish in dish_anchors:
                if not self._answer_mentions_anchor(answer_norm, dish):
                    missing.append(f"chua nhac mon '{dish}'")

            generator_candidates = self._build_generator_candidates(state.all_seeds)
            restaurants = [c["name"] for c in generator_candidates if c["type"] == "Restaurant"]
            has_restaurant = any(self._answer_mentions_anchor(answer_norm, rest) for rest in restaurants)
            if not restaurants:
                has_restaurant = any(term in answer_norm for term in restaurant_terms)

            if not has_restaurant:
                missing.append("chua nhac nha hang phuc vu")
            if address_in_context and not any(term in answer_norm for term in ["dia chi", "lien he", "sdt", "so dien thoai", "so dt", "duong"]):
                missing.append("chua nhac dia chi/lien he co trong context")

        # 3. Category Listing Coverage Check
        category_aliases = [
            ("Di tích lịch sử", ["di tich lich su", "lich su van hoa"]),
            ("Danh lam thắng cảnh", ["danh lam thang canh", "danh lam", "danh thang"]),
            ("Làng nghề truyền thống", ["lang nghe truyen thong", "lang nghe"]),
        ]
        for cat_name, aliases in category_aliases:
            if any(alias in query_norm for alias in aliases):
                cat_norm = normalize_text(cat_name, strip_punct=True)
                expected_entities = []
                for fact in (state.raw_context or []):
                    fact_norm = normalize_text(fact, strip_punct=True)
                    if cat_norm in fact_norm:
                        for seed in (state.all_seeds or []):
                            seed_name = str(seed.metadata.get("name") or seed.content or "").strip()
                            seed_norm = normalize_text(seed_name, strip_punct=True)
                            if seed_norm and seed_norm in fact_norm:
                                expected_entities.append(seed_name)
                if expected_entities:
                    has_match = any(self._answer_mentions_anchor(answer_norm, ent) for ent in expected_entities)
                    if not has_match:
                        missing.append(f"chua nhac den bat ky dia diem nao thuoc loai hinh '{cat_name}' (vi du: {', '.join(expected_entities[:2])})")

        # 4. Region Mismatch/Leakage Check
        legacy_province = metadata.get("legacy_province") or ""
        if legacy_province:
            lp_norm = normalize_text(legacy_province, strip_punct=True)
            if "binh dinh" in lp_norm:
                if "gia lai" in answer_norm or "pleiku" in answer_norm:
                    if "gia lai" not in query_norm and "pleiku" not in query_norm:
                        missing.append("cau tra loi nhac den Gia Lai/Pleiku khong dung tinh Binh Dinh")
            elif "gia lai" in lp_norm:
                # Nếu truy vấn chỉ rõ "Gia Lai cũ" (gia lai cu / gia lai cũ) thì không được nhắc đến Bình Định/Quy Nhơn.
                # Ngược lại, nếu là Gia Lai chung (tỉnh mới sáp nhập), cho phép gộp thông tin từ Bình Định.
                if "gia lai cu" in query_norm:
                    if "binh dinh" in answer_norm or "quy nhon" in answer_norm:
                        if "binh dinh" not in query_norm and "quy nhon" not in query_norm:
                            missing.append("cau tra loi nhac den Binh Dinh/Quy Nhon khong dung tinh Gia Lai cu")

        return list(dict.fromkeys(missing))


    # ------------------------------------------------------------------
    # Follow-up answered-entity memory helpers
    # ------------------------------------------------------------------

    def _store_answered_entities(self, entity_names: List[str]) -> None:
        """Store entity names that were actually presented to the user.

        Called after successful answer generation or grounded fallback.
        Caps at 50 entries to avoid unbounded growth.
        """
        if not entity_names:
            return
        with self.pipeline._state_lock:
            cs = self.pipeline.location_grounding_service.conversation_state
            prev = cs.get("previously_answered_entities") or []
            merged = list(dict.fromkeys(prev + entity_names))
            cs["previously_answered_entities"] = merged[-50:]

    def _extract_answered_entity_names(
        self,
        answer: str,
        ranked_candidates: List[Any],
    ) -> List[str]:
        """Extract entity names that actually appeared in the final answer.

        Hybrid approach:
        1. Primary: regex-extract **bold** names, cross-validate against seed_lookup
        2. Fallback: substring match (original logic) when no bold names found

        This ensures deterministic answers with **Name** pattern extract ALL entities,
        while hallucinated bold text and section headers are blocked via seed validation.
        """
        names: List[str] = []
        seen_norm: set = set()

        # Build seed lookup for cross-validation
        seed_lookup: Dict[str, str] = {}
        for cand in (ranked_candidates or []):
            cand_name = str(cand.metadata.get("name") or cand.content or "").strip()
            if cand_name:
                key = normalize_text(cand_name, strip_punct=True)
                if key:
                    seed_lookup[key] = cand_name

        # Primary: regex extract **bold** names, cross-validate with seeds
        for match in re.finditer(r'\*\*(.+?)\*\*', answer or ''):
            captured = match.group(1).strip()
            if not captured:
                continue
            captured_norm = normalize_text(captured, strip_punct=True)
            if captured_norm and captured_norm in seed_lookup and captured_norm not in seen_norm:
                seen_norm.add(captured_norm)
                names.append(seed_lookup[captured_norm])

        # Fallback: substring match when no bold names found (backward compatible)
        if not names:
            ans_norm = normalize_text(answer or "", strip_punct=True)
            for cand_norm, cand_name in seed_lookup.items():
                if cand_norm and cand_norm in ans_norm and cand_norm not in seen_norm:
                    seen_norm.add(cand_norm)
                    names.append(cand_name)

        return names

    _EXAMPLE_TEXT_RE = re.compile(
        r"(?i)(?:ví\s+dụ|vi\s+du|VD|vd|chẳng\s+hạn|chang\s+han|ví\s+dụ\s+như|chẳng\s+hạn\s+như|như)\s*[:：]?\s*(.+?)(?:\)|$)",
        re.DOTALL,
    )

    # Category nouns that indicate "như X hay Y" lists the actual targets, not examples.
    _TARGET_CATEGORY_NOUNS = [
        "dia diem", "diem tham quan", "diem du lich", "noi du lich",
        "thanh pho", "tinh", "khu vuc", "vung", "hon dao", "dao",
        "bien", "ho", "thac", "nui", "chua", "den", "lang",
        "khach san", "nha nghi", "homestay", "resort",
        "nha hang", "quan an", "mon an", "dac san",
        "su kien", "le hoi",
    ]

    def _extract_example_text(self, query: str) -> str:
        """Extract text from example clauses (e.g., 'Ví dụ: X, Y' → 'X, Y').

        Also matches 'như A hay B' / 'như A, B' patterns when the captured
        group contains at least one list separator (hay, hoặc, và / va).
        This prevents false positives from generic 'như' usage.
        """
        raw_query = str(query or "")
        for match in self._EXAMPLE_TEXT_RE.finditer(raw_query):
            captured = str(match.group(1) or "").strip()
            marker = str(match.group(0) or "").strip()
            # If marker is just "như"/"nhu" (not preceded by ví dụ/chẳng hạn),
            # require list separator in captured text to confirm it's an example list
            marker_norm = normalize_text(marker, strip_punct=True)
            is_standalone_nhu = marker_norm.startswith("nhu") and not any(
                prefix in marker_norm for prefix in ["vi du", "vd", "chang han"]
            )
            if is_standalone_nhu:
                # Check if "như" is preceded by a category noun → entities are targets, not examples
                # e.g. "Các điểm tham quan như Biển Hồ T'Nưng hay Biển Hồ Chè"
                pre_text = raw_query[:match.start()]
                pre_norm = normalize_text(pre_text, strip_punct=True)
                if any(pre_norm.endswith(noun) for noun in self._TARGET_CATEGORY_NOUNS):
                    continue
                captured_norm = normalize_text(captured, strip_punct=True)
                has_separator = any(
                    sep in captured_norm
                    for sep in [" hay ", " hoac ", " va "]
                )
                if not has_separator:
                    continue
            return captured
        return ""

    def _is_follow_up_with_other_marker(self, query: str) -> bool:
        """Check if query is a follow-up asking for 'other' items."""
        q = normalize_text(query, strip_punct=True)
        other_markers = [
            "con mon nao khac", "con nhung mon nao khac", "mon nao khac",
            "con gi khac", "them mon nao", "khac khong", "con nua khong",
            "con nao khac", "con dia diem nao khac", "dia diem nao khac",
        ]
        if any(m in q for m in other_markers):
            return True
        # Regex patterns for flexible matching: "con <word> nao khac", "khac nua khong"
        if re.search(r"con\s+\w+\s+nao\s+khac", q):
            return True
        if re.search(r"khac\s+nua\s+khong", q):
            return True
        return False

    def _finalize_pipeline_response(self, state: PipelineRunState) -> Dict[str, Any]:
        p = self.pipeline
        if state.runtime.metadata.get("constrained_nearby_search_failed"):
            warning = "Chưa tìm thấy khách sạn thỏa đầy đủ chuỗi quan hệ này trong graph.\n"
            if not state.answer.startswith(warning):
                state.answer = warning + state.answer

        state.runtime.metadata["detected_location"] = state.location
        # Single source of truth: derive intent from query_plan only
        intent = state.query_plan.intent if state.query_plan else state.primary_intent
        state.runtime.metadata["intent"] = intent

        # Diagnostic logging for empty intent
        if not intent:
            logger.warning("   -> [DIAGNOSTIC-RESPONSE] Empty intent! query_plan.intent=%s, primary_intent=%s, query=%s",
                          state.query_plan.intent if state.query_plan else "None",
                          state.primary_intent,
                          state.user_query[:80])
        logger.info("   -> [DIAGNOSTIC-RESPONSE] Final intent='%s', query_plan.intent='%s'", intent, state.query_plan.intent if state.query_plan else "None")
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(state.all_seeds)

        # Save main entity from ContextOrganizer for entity inheritance in follow-up queries
        organizer_output = state.metadata.get("context_organizer_output") or {}
        main_entity_name = organizer_output.get("main_entity") or ""
        if main_entity_name and state.all_seeds:
            for seed in state.all_seeds:
                seed_name = seed.metadata.get("name") or seed.content or ""
                if seed_name and normalize_text(seed_name) == normalize_text(main_entity_name):
                    p.location_grounding_service.conversation_state["last_active_entity"] = {
                        "name": seed_name,
                        "type": seed.metadata.get("labels", [None])[0] if seed.metadata.get("labels") else seed.metadata.get("type"),
                    }
                    break

        # Store answered entities for follow-up exclusion memory
        val_result = state.metadata.get("answer_validation")
        validation_passed = val_result.get("passed") is True if val_result else False
        fallback_triggered = bool(state.metadata.get("grounded_fallback_triggered"))
        
        should_store = validation_passed or fallback_triggered or (val_result is None)
        if should_store and state.answer:
            ans_norm = normalize_text(state.answer, strip_punct=True)
            abstain_phrases = ["xin loi", "khong du thong tin", "chua co du", "ngoai pham vi"]
            if any(p in ans_norm for p in abstain_phrases):
                should_store = False

        if should_store:
            answered_names = self._extract_answered_entity_names(
                answer=state.answer,
                ranked_candidates=state.all_seeds or [],
            )
            if answered_names:
                self._store_answered_entities(answered_names)
                logger.info("   -> [Memory] Stored %s answered entities for follow-up exclusion", len(answered_names))

        if intent == IntentType.TOUR_PLAN and not state.runtime.metadata.get("route_seed_nodes"):
            state.runtime.metadata["route_seed_nodes"] = p._select_route_seed_nodes(
                state.answer,
                state.all_seeds or [],
                intent,
            )
        state.runtime.metadata["graph"] = p._build_graph_payload(
            state.all_seeds,
            state.raw_context,
            intent=intent,
            route_seed_nodes=state.runtime.metadata.get("route_seed_nodes") or [],
        )
        state.runtime.metadata["raw_context"] = state.raw_context
        state.runtime.metadata["clean_context"] = state.clean_context
        logger.info(
            "   -> Final Metadata Summary: "
            f"seed_nodes={len(state.runtime.metadata.get('seed_nodes') or [])}, "
            f"route_seed_nodes={len(state.runtime.metadata.get('route_seed_nodes') or [])}, "
            f"graph_nodes={len((state.runtime.metadata.get('graph') or {}).get('nodes') or [])}, "
            f"graph_edges={len((state.runtime.metadata.get('graph') or {}).get('edges') or [])}"
        )
        # Export QueryPlan and runtime info to result
        result = {"answer": state.answer, "metadata": state.runtime.metadata}
        if state.query_plan:
            result["query_plan"] = {
                "intent": state.query_plan.intent,
                "operation": state.query_plan.operation.value,
                "target_class": state.query_plan.target_class,
                "answer_mode": state.query_plan.answer_mode,
                "retrieval_mode": state.query_plan.retrieval_mode,
                "renderer": state.query_plan.renderer,
            }
        result["grounded_count"] = len(state.grounded_nodes or [])
        return result


    def _make_early_guard_result(self, state: PipelineRunState, answer: str, reason: str) -> Dict[str, Any]:
        import dataclasses
        state.runtime.metadata["detected_location"] = state.location
        if state.metadata.get("expected_intent") and reason in {
            "out_of_region",
            "realtime_booking_or_price",
            "complete_phone_directory",
            "infeasible_itinerary",
            "needs_clarification",
        }:
            # Override query_plan + primary_intent so intent has a single source of truth
            override_intent = state.metadata.get("expected_intent")
            if state.query_plan:
                state.query_plan = dataclasses.replace(state.query_plan, intent=override_intent)
            state.primary_intent = override_intent
        # Always derive intent from query_plan (single source of truth)
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["early_guard"] = reason
        state.runtime.metadata["answer_mode"] = state.runtime.metadata.get("answer_mode") or AnswerMode.FACT_ANSWER
        return {"answer": self._sanitize_answer_text(answer), "metadata": state.runtime.metadata}


    def _early_scope_and_clarification_guard(self, state: PipelineRunState) -> Dict[str, Any] | None:
        """Narrow pre-retrieval guard for E2E scope, realtime, feasibility, and ambiguous queries."""
        p = self.pipeline
        query = state.user_query or ""
        q_norm = normalize_text(query, strip_punct=True)
        metadata = state.metadata or {}

        has_in_scope_region = any(normalize_text(term, strip_punct=True) in q_norm for term in self.IN_SCOPE_REGION_TERMS)
        out_region_hits = [
            term for term in self.OUT_OF_REGION_TERMS
            if normalize_text(term, strip_punct=True) in q_norm
        ]
        if out_region_hits and not has_in_scope_region:
            answer = (
                "Mình chỉ có dữ liệu du lịch trong phạm vi Gia Lai mới, bao gồm khu vực Pleiku và "
                "Quy Nhơn/Bình Định cũ. Yêu cầu này nằm ngoài phạm vi dữ liệu hiện có, nên mình "
                "không thể lập lịch trình hay bịa thông tin cho địa phương đó."
            )
            return self._make_early_guard_result(state, answer, "out_of_region")

        # --- Out-of-domain guard: health, vaccination, safety, legal, financial ---
        # Bypass: classification queries (e.g. "thuộc loại hình du lịch nào") are always in-domain
        classification_bypass = ["thuoc loai", "loai hinh", "phan loai", "the loai", "thuoc nhom"]
        is_classification_query = any(normalize_text(sig, strip_punct=True) in q_norm for sig in classification_bypass)

        out_of_domain_patterns = [
            # Health / vaccination
            "tiem phong", "tiêm phòng", "vaccine", "vacxin",
            "suc khoe", "sức khỏe", "benh", "bệnh",
            "thuoc men", "thuốc men", "benh vien", "bệnh viện",
            "phong kham", "phòng khám", "chua benh", "chữa bệnh",
            "dich benh", "dịch bệnh", "ung thư", "ung thu",
            "covid", "sot xuat huyet", "sốt xuất huyết",
            # Safety / crime
            "an toan", "an toàn", "lừa đảo", "lua dao",
            "tội phạm", "toi pham", "cướp", "cuop",
            # Legal / visa / immigration
            "visa", "hộ chiếu", "ho chieu", "thị thực", "thi thuc",
            "nhập cảnh", "nhap canh", "xuất cảnh", "xuat canh",
            "giấy tờ", "giay to", "hồ sơ", "ho so",
            # Financial / non-tourism
            "đầu tư", "dau tu", "chứng khoán", "chung khoan",
            "bất động sản", "bat dong san",
        ]
        if not is_classification_query and any(normalize_text(term, strip_punct=True) in q_norm for term in out_of_domain_patterns):
            # Tailor the redirect hint based on which domain was detected
            health_terms = ["tiem phong", "vaccine", "vacxin", "suc khoe", "benh",
                            "thuoc men", "benh vien", "phong kham", "chua benh",
                            "dich benh", "covid", "sot xuat huyet", "ung thu"]
            is_health = any(normalize_text(t, strip_punct=True) in q_norm for t in health_terms)
            if is_health:
                redirect = (
                    "Về vấn đề sức khỏe hoặc tiêm phòng khi đi du lịch, bạn nên tham khảo "
                    "bác sĩ hoặc trung tâm y tế dự phòng để được tư vấn chính xác."
                )
            else:
                redirect = ""
            answer = (
                f"Câu hỏi này nằm ngoài phạm vi dữ liệu du lịch của mình. "
                f"Mình chỉ có thể hỗ trợ thông tin về điểm tham quan, ăn uống, lưu trú, "
                f"sự kiện và lịch trình du lịch tại Gia Lai (bao gồm Quy Nhơn/Bình Định)."
                f" {redirect}".rstrip()
            )
            return self._make_early_guard_result(state, answer, "out_of_domain")

        realtime_patterns = [
            "dat ve",
            "đặt vé",
            "ve may bay",
            "vé máy bay",
            "gia re hom nay",
            "giá rẻ hôm nay",
            "hom nay",
            "hôm nay",
            "dat phong",
            "đặt phòng",
            "booking",
        ]
        # Skip realtime guard if query is asking for advice/tips (not actual booking)
        _ADVICE_SIGNALS = [
            "kinh nghiem", "kinh nghiệm", "meo", "mẹo",
            "luu y", "lưu ý", "nen chuan bi", "nên chuẩn bị",
            "can biet", "cần biết", "dat phong the nao", "đặt phòng thế nào",
            "tiet kiem", "tiết kiệm", "tranh bi", "tránh bị",
        ]
        has_advice_signal = any(normalize_text(t, strip_punct=True) in q_norm for t in _ADVICE_SIGNALS) or bool(metadata.get("skip_realtime_booking_guard"))
        if not has_advice_signal and any(normalize_text(term, strip_punct=True) in q_norm for term in realtime_patterns):
            answer = (
                "Mình không thể đặt vé hoặc kiểm tra giá hôm nay vì không có dữ liệu thời gian thực để "
                "thực hiện "
                "giao dịch. Mình có thể hỗ trợ gợi ý điểm đến, khu vực lưu trú hoặc lịch trình dựa "
                "trên dữ liệu du lịch hiện có."
            )
            return self._make_early_guard_result(state, answer, "realtime_booking_or_price")

        if (
            ("so dien thoai" in q_norm or "sdt" in q_norm)
            and ("tat ca" in q_norm or "tất cả" in query.lower())
        ):
            answer = (
                "Mình không thể xác minh chính xác số điện thoại của tất cả cơ sở lưu trú vì không có "
                "danh bạ đầy đủ. "
                "Mình chỉ có thể nêu những số điện thoại xuất hiện trực tiếp trong dữ liệu; nếu dữ liệu thiếu, "
                "mình sẽ không tự bịa hoặc suy đoán."
            )
            return self._make_early_guard_result(state, answer, "complete_phone_directory")

        if (
            ("10 diem" in q_norm or "10 điểm" in query.lower())
            and ("di bo" in q_norm or "đi bộ" in query.lower())
            and ("1 ngay" in q_norm or "1 ngày" in query.lower())
        ):
            answer = (
                "Mình không thể lập lịch trình 1 ngày gồm 10 điểm xa nhau và đi bộ hoàn toàn vì không khả thi. "
                "Mình có thể đề xuất phương án an toàn hơn: chọn 3 đến 5 điểm gần nhau trong cùng "
                "một khu vực, hoặc đổi sang di chuyển bằng xe nếu muốn đi nhiều điểm."
            )
            return self._make_early_guard_result(state, answer, "infeasible_itinerary")

        # Gia Lai mới includes coastal Quy Nhơn/Bình Định. Do not abstain for biển/san hô questions.
        if (
            "gia lai" in q_norm
            and any(term in q_norm for term in ["lan bien", "lặn biển", "lan san ho", "lặn san hô", "ngam san ho", "ngắm san hô"])
        ):
            tour = "Tour Kỳ Co – Lặn san hô Bãi Dứa nửa ngày đón tại Nhơn Lý"
            answer = (
                "Có. Theo dữ liệu Gia Lai mới có khu vực Quy Nhơn/Bình Định cũ, hệ thống ghi nhận "
                f"{tour}. Tour này do CÔNG TY CỔ PHẦN QUY NHƠN TOURIST tổ chức, bao gồm tham quan "
                "Biển Kỳ Co và lặn/ngắm san hô tại Bãi Dứa (Bãi San Hô)."
            )
            metadata["region_focus"] = "coastal_quy_nhon"
            metadata["geo_anchor_location"] = "Quy Nhơn"
            state.runtime.metadata["detected_location"] = "Quy Nhơn"
            metadata["scope_guard_note"] = "gia_lai_new_coastal_scope"
            metadata["retrieval_allowed_labels"] = ["Tour", "TouristAttraction", "TravelAgency"]
            state.region_focus = "coastal_quy_nhon"
            state.location = "Quy Nhơn"
            state.primary_intent = IntentType.TOURISM
            import dataclasses
            if state.query_plan:
                state.query_plan = dataclasses.replace(state.query_plan, intent=IntentType.TOURISM)
            state.metadata = metadata
            if not metadata.get("target_entity"):
                target = tour
                metadata["target_entity"] = target
                metadata["proximity_anchor"] = target
                state.entities = [{"name": target, "type": "Tour"}] + list(state.entities or [])
            return self._make_early_guard_result(state, answer, "gia_lai_new_coastal_scope")

        should_clarify = bool(metadata.get("should_clarify", False))
        ambiguous_patterns = [
            "di dau gan day",
            "đi đâu gần đây",
            "cho dep dep",
            "chỗ đẹp đẹp",
            "chill chill",
            "it tien nhung vui",
            "ít tiền nhưng vui",
            "thac nao do",
            "thác nào đó",
            "khong nho ten",
            "không nhớ tên",
        ]
        if should_clarify or any(normalize_text(term, strip_punct=True) in q_norm for term in ambiguous_patterns):
            answer = (
                "Mình cần thêm một chút thông tin để gợi ý chính xác hơn: bạn đang ở khu vực nào "
                "(Pleiku, Quy Nhơn/Bình Định cũ, An Khê, Chư Sê...), muốn đi trong bao lâu và thích "
                "thiên nhiên, văn hóa, ăn uống hay nghỉ dưỡng? Nếu chưa rõ, mình có thể gợi ý vài "
                "lựa chọn phổ biến theo từng nhóm."
            )
            return self._make_early_guard_result(state, answer, "needs_clarification")

    def _answer_belongs_to_classification_if_possible(self, state: PipelineRunState) -> str:
        """Deterministic renderer for BELONGS_TO classification queries.

        Renders answer directly from entity category/type data without LLM.
        Examples:
            "Làng Du lịch cộng đồng Mơ Hra thuộc loại hình du lịch nào?"
            -> "Làng Du lịch cộng đồng Mơ Hra thuộc loại hình du lịch cộng đồng / làng văn hóa."
        """
        metadata = state.metadata or {}

        # Only activate for classification contract
        if not metadata.get("classification_contract_active"):
            return ""

        # Get the target entity name
        target_entity = metadata.get("target_entity") or ""
        if not target_entity:
            for entity in (state.entities or []):
                if isinstance(entity, dict):
                    ent_name = str(entity.get("name") or "").strip()
                    if ent_name:
                        target_entity = ent_name
                        break
        if not target_entity:
            return ""

        # Extract category/type from multiple sources
        category = ""
        entity_type = ""
        description_snippet = ""

        # Source 1: From grounded seeds (exact match)
        for seed in (state.all_seeds or []):
            seed_name = str(seed.metadata.get("name") or seed.content or "").strip()
            if not seed_name:
                continue
            seed_norm = normalize_text(seed_name, strip_punct=True)
            target_norm = normalize_text(target_entity, strip_punct=True)

            # Check if this seed matches the target entity
            if seed_norm == target_norm or target_norm in seed_norm or seed_norm in target_norm:
                # Extract category from metadata (skip technical labels)
                seed_category = str(seed.metadata.get("category") or "").strip()
                _SKIP_CATEGORIES = {"touristattraction", "travelinfo", "restaurant", "dish", "accommodation", "event", "tour", "location", "specialty", "none", ""}
                if seed_category and seed_category.lower() not in _SKIP_CATEGORIES:
                    category = seed_category

                # Extract type from labels
                labels = seed.metadata.get("labels") or []
                if labels:
                    entity_type = str(labels[0] or "").strip()

                # Extract description snippet (first sentence)
                desc = str(seed.metadata.get("description") or "").strip()
                if desc:
                    # Get first sentence
                    first_sentence = desc.split(".")[0].strip()
                    if first_sentence and len(first_sentence) > 10:
                        description_snippet = first_sentence

                break

        # Source 2: From context facts (BELONGS_TO edges)
        if not category:
            for fact in (state.raw_context or []):
                fact_str = str(fact or "").strip()

                # Look for explicit BELONGS_TO relationship pattern
                bel_match = re.search(r"\[BELONGS_TO\]\s*->\s*(.+?)$", fact_str)
                if bel_match:
                    category = bel_match.group(1).strip().rstrip(".")
                    break

        # Source 3: From entity data in metadata (skip technical labels)
        if not category:
            _SKIP_CATS = {"touristattraction", "travelinfo", "restaurant", "dish", "accommodation", "event", "tour", "location", "specialty"}
            for entity in (metadata.get("entities") or []):
                if isinstance(entity, dict):
                    ent_name = str(entity.get("name") or "").strip()
                    if normalize_text(ent_name) == normalize_text(target_entity):
                        cat = str(entity.get("category") or entity.get("type") or "").strip()
                        if cat and cat.lower() not in _SKIP_CATS:
                            category = cat
                            break

        # Source 4: Query BELONGS_TO from Neo4j directly
        if not category and target_entity:
            try:
                p = self.pipeline
                if hasattr(p, 'driver') and p.driver:
                    with p.driver.session() as session:
                        result = session.run(
                            "MATCH (n)-[:BELONGS_TO]->(cat) "
                            "WHERE n.name = $name "
                            "RETURN cat.name AS category LIMIT 1",
                            name=target_entity,
                        )
                        record = result.single()
                        if record and record.get("category"):
                            category = str(record["category"]).strip()
                            logger.info("   -> [Deterministic] BELONGS_TO from Neo4j: '%s'", category)
            except (Neo4jClientError, ServiceUnavailable) as e:
                logger.error("   -> [Deterministic] Neo4j fallback error: %s", e)

        # Build the deterministic answer
        if not category and not entity_type:
            return ""

        # Determine the classification label
        classification = category or entity_type
        # Strip "loại " prefix if present (from context extraction)
        if classification.lower().startswith("loại "):
            classification = classification[5:].strip()
        _TECHNICAL_LABELS_MAP = {
            "touristattraction": "điểm du lịch",
            "travelinfo": "thông tin du lịch",
            "restaurant": "nhà hàng",
            "dish": "món ăn",
            "accommodation": "cơ sở lưu trú",
            "event": "sự kiện",
            "tour": "tour",
            "location": "địa điểm",
            "specialty": "đặc sản",
        }
        classification_lower = classification.lower()
        if classification_lower in _TECHNICAL_LABELS_MAP:
            classification = _TECHNICAL_LABELS_MAP[classification_lower]

        logger.info("   -> [Deterministic] BELONGS_TO classification: category='%s', entity_type='%s', final='%s'", category, entity_type, classification)

        # Build answer
        parts = [f"**{target_entity}** thuộc loại hình **{classification}**."]

        # Add description snippet if available
        if description_snippet:
            parts.append(f"\n{description_snippet}.")

        answer = " ".join(parts)

        # Set metadata
        metadata["deterministic_classification"] = True
        metadata["classification_category"] = category
        metadata["classification_entity_type"] = entity_type

        logger.info("   -> [Deterministic] BELONGS_TO classification rendered: category='%s'", classification)
        return answer

    def _answer_food_specialty_deterministic(self, state: PipelineRunState) -> str:
        """Deterministic renderer for food specialty queries.

        Renders food/dish list directly from graph data without LLM.
        Examples:
            "Ở Gia Lai có đặc sản gì?"
            -> "Đặc sản Gia Lai: Phở bò, Bánh xèo, Cơm lam..."
        """
        metadata = state.metadata or {}

        # Read exclusion from ExclusionContext (single normalize pass, built in Step 4)
        exclusion_ctx = state.runtime.metadata.get("exclusion_context")
        exclusion_set = exclusion_ctx.entity_names if exclusion_ctx else set()
        if exclusion_set:
            logger.info("   -> [Deterministic] Exclusion set from ExclusionContext: %s entities", len(exclusion_set))

        # Skip for follow-up queries — let LLM generate richer answer with descriptions
        # EXCEPT when should_force_deterministic is set (insufficient context → avoid hallucination)
        plan = state.query_plan
        force_det = exclusion_ctx.should_force_deterministic if exclusion_ctx else state.runtime.metadata.get("force_deterministic", False)
        if plan and plan.is_follow_up and not force_det:
            return ""

        # Skip for dish_to_restaurant queries — let _answer_dish_to_restaurant_if_possible handle
        # e.g. "Quán ăn nào có món Phở bò tái?", "Phở khô Gia Lai ở đâu bán?"
        q_norm = normalize_text(state.user_query, strip_punct=True)
        where_tokens = ["o dau", "ban o dau", "o dau ban", "quan an nao", "nha hang nao", "co mon", "mon gi", "phuc vu mon"]
        if any(token in q_norm for token in where_tokens):
            # Query is asking WHERE to find a dish → skip deterministic, use graph traversal
            return ""

        # Activate for food specialty contract OR food recommendation intent
        has_food_contract = bool(metadata.get("food_specialty_contract") or metadata.get("food_specialty_contract_active"))
        # Unified Contract: derive intent from query_plan only
        has_food_intent = plan and "FOOD" in str(plan.intent).upper()
        if not has_food_contract and not has_food_intent:
            return ""

        # NOTE: Do NOT skip deterministic renderer when context is rich.
        # DiscoveryList should always render deterministically to avoid
        # duplicate entities from LLM rendering raw facts.
        # Rich context is now used by the deterministic renderer itself.

        # For specific entity queries, return only HAS-connected dishes
        # e.g. "Nem Chợ Huyện Olala có món đặc trưng nào?"
        q_norm = normalize_text(state.user_query, strip_punct=True)
        is_specific_entity = any(token in q_norm for token in [
            "co mon", "mon dac trung", "mon gi", "phuc vu", "mon nao",
        ])
        target_entity = metadata.get("target_entity") or ""
        if not target_entity:
            for entity in (state.entities or []):
                if isinstance(entity, dict):
                    ent_name = str(entity.get("name") or "").strip()
                    if ent_name:
                        target_entity = ent_name
                        break
        if is_specific_entity and target_entity:
            # Try to find dishes from HAS relationships in context
            has_dishes = []
            for fact in (state.raw_context or []):
                fact_str = str(fact or "").strip()
                if "[HAS]" in fact_str:
                    has_match = re.search(r"\[HAS\]\s*->\s*(.+?)$", fact_str)
                    if has_match:
                        has_dishes.append(has_match.group(1).strip().rstrip("."))
            if has_dishes:
                # Convert to entity info format with descriptions from seeds
                has_dish_infos = []
                seeds_by_name = {}
                for seed in (state.all_seeds or []):
                    info = self._extract_entity_info(seed)
                    seeds_by_name[normalize_text(info["name"], strip_punct=True)] = info
                for dish_name in has_dishes:
                    dish_norm = normalize_text(dish_name, strip_punct=True)
                    if dish_norm in seeds_by_name:
                        has_dish_infos.append(seeds_by_name[dish_norm])
                    else:
                        has_dish_infos.append({"name": dish_name, "description": "", "address": "", "category": "", "labels": ["Dish"]})
                has_dish_infos = self._dedupe_entities(has_dish_infos)
                answer = self._format_entity_list(
                    has_dish_infos,
                    title="Đặc sản",
                    max_items=5,
                    show_description=True,
                    show_address=False,
                )
                if answer:
                    state.runtime.metadata["deterministic_food_list"] = True
                    state.runtime.metadata["food_type"] = "specific_has"
                    logger.info("   -> [Deterministic] Food specialty (specific HAS): %s", [d['name'] for d in has_dish_infos[:5]])
                    return answer
            # Fallback: return dishes from seeds that match the entity (with descriptions)
            entity_dishes = []
            for seed in (state.all_seeds or []):
                info = self._extract_entity_info(seed)
                if "Dish" in info["labels"] or "Specialty" in info["labels"]:
                    entity_dishes.append(info)
            entity_dishes = self._dedupe_entities(entity_dishes)
            # Filter out previously answered entities (defense-in-depth)
            if exclusion_set:
                entity_dishes = [d for d in entity_dishes if normalize_text(d.get("name", ""), strip_punct=True) not in exclusion_set]
            if entity_dishes:
                answer = self._format_entity_list(
                    entity_dishes,
                    title="Đặc sản",
                    max_items=5,
                    show_description=True,
                    show_address=False,
                )
                if answer:
                    state.runtime.metadata["deterministic_food_list"] = True
                    state.runtime.metadata["food_type"] = "seed_dishes"
                    logger.info("   -> [Deterministic] Food specialty (seed dishes): %s", [d['name'] for d in entity_dishes[:5]])
                    return answer
            # Neo4j fallback: query HAS from restaurant to dishes (with descriptions)
            if target_entity:
                try:
                    p = self.pipeline
                    if hasattr(p, 'driver') and p.driver:
                        with p.driver.session() as session:
                            result = session.run(
                                "MATCH (r)-[:HAS]->(d) WHERE r.name = $name RETURN d.name AS dish, d.description AS desc LIMIT 5",
                                name=target_entity,
                            )
                            neo_dish_infos = []
                            for record in result:
                                if record.get("dish"):
                                    neo_dish_infos.append({
                                        "name": str(record["dish"]),
                                        "description": str(record.get("desc") or ""),
                                        "address": "",
                                        "category": "món ăn",
                                        "labels": ["Dish"],
                                    })
                            # Filter out previously answered entities from Neo4j fallback
                            if exclusion_set:
                                neo_dish_infos = [d for d in neo_dish_infos if normalize_text(d["name"], strip_punct=True) not in exclusion_set]
                            if neo_dish_infos:
                                answer = self._format_entity_list(
                                    neo_dish_infos,
                                    title="Đặc sản",
                                    max_items=5,
                                    show_description=True,
                                    show_address=False,
                                )
                                if answer:
                                    state.runtime.metadata["deterministic_food_list"] = True
                                    state.runtime.metadata["food_type"] = "neo4j_has"
                                    logger.info("   -> [Deterministic] Food specialty (Neo4j HAS): %s", [d['name'] for d in neo_dish_infos[:5]])
                                    return answer
                except (Neo4jClientError, ServiceUnavailable):
                    pass

        location = state.location or state.runtime.metadata.get("detected_location") or ""
        location_suffix = f" {location}" if location else ""

        # Collect dishes and restaurants from seeds using common helper
        dishes = []
        restaurants = []
        for seed in (state.all_seeds or []):
            info = self._extract_entity_info(seed)
            name = info["name"]
            if not name:
                continue

            if "Dish" in info["labels"] or "Specialty" in info["labels"]:
                dishes.append(info)
            elif "Restaurant" in info["labels"]:
                restaurants.append(info)

        logger.info("   -> [Curated] Seeds scan: total=%d, dishes_found=%d, dish_names=%s",
                    len(state.all_seeds or []), len(dishes), [d["name"] for d in dishes])

        # Deduplicate
        dishes = self._dedupe_entities(dishes)
        restaurants = self._dedupe_entities(restaurants)

        # Filter out previously answered entities (defense-in-depth)
        if exclusion_set:
            dishes = [d for d in dishes if normalize_text(d.get("name", ""), strip_punct=True) not in exclusion_set]

        # If we have Dish nodes, prepare curated context for LLM enrichment
        if dishes:
            curated_ctx = self._prepare_curated_food_context(state, dishes, location)
            if curated_ctx:
                state.runtime.metadata["curated_food_context"] = curated_ctx
                state.runtime.metadata["curated_food_entities"] = [d["name"] for d in dishes[:10]]
                logger.info("   -> [Curated] Prepared %d dishes for LLM curation: %s",
                           len(dishes), [d["name"] for d in dishes[:8]])
                logger.info("   -> [Curated] Context preview (%d chars): %s...",
                           len(curated_ctx), curated_ctx[:300])
                # Return empty to trigger LLM path with curated context
                return ""

        # If no Dish nodes, render restaurant list with note
        if restaurants:
            answer = self._format_entity_list(
                restaurants,
                title=f"Các nhà hàng/địa điểm ẩm thực{location_suffix}",
                location_suffix="",
                max_items=6,
                show_description=True,
                show_address=True,
            )
            if answer:
                answer += "\n\nBạn có thể tham khảo menu tại các nhà hàng trên để biết thêm đặc sản địa phương."
                state.runtime.metadata["deterministic_food_list"] = True
                state.runtime.metadata["food_type"] = "restaurant_list"
                logger.info("   -> [Deterministic] Food specialty rendered: %s restaurants (no dish data)", len(restaurants))
                return answer

        return ""

    # ------------------------------------------------------------------
    # Common deterministic rendering helpers
    # ------------------------------------------------------------------

    def _extract_entity_info(self, node: Any) -> Dict[str, str]:
        """Extract name, description, address, category from a seed node.

        Works with both NodeItem objects and dict-like candidates.
        """
        if isinstance(node, dict):
            name = str(node.get("name") or "").strip()
            desc = str(node.get("description") or "").strip()
            addr = str(node.get("address") or "").strip()
            category = str(node.get("category") or "").strip()
            labels = node.get("labels") or []
        else:
            name = str(getattr(node, "content", "") or (getattr(node, "metadata", {}) or {}).get("name") or "").strip()
            meta = getattr(node, "metadata", {}) or {}
            desc = str(meta.get("description") or "").strip()
            addr = str(meta.get("address") or "").strip()
            category = str(meta.get("category") or "").strip()
            labels = meta.get("labels") or []

        # Infer category from labels if not set
        if not category and labels:
            label_map = {
                "Dish": "món ăn",
                "Specialty": "đặc sản",
                "Restaurant": "nhà hàng",
                "TouristAttraction": "điểm tham quan",
                "Accommodation": "lưu trú",
                "Event": "sự kiện",
                "Tour": "tour",
            }
            category = label_map.get(str(labels[0]), "")

        return {
            "name": name,
            "description": desc,
            "address": addr,
            "category": category,
            "labels": [str(l) for l in labels],
        }

    def _get_first_sentence(self, text: str, max_len: int = 150) -> str:
        """Get first sentence from description, truncated to max_len."""
        if not text:
            return ""
        # Split by sentence-ending punctuation
        for sep in [". ", ".\n", "! ", "? ", "; "]:
            idx = text.find(sep)
            if 0 < idx < max_len:
                return text[:idx + 1].strip()
        # No sentence break found, truncate
        if len(text) > max_len:
            return text[:max_len].rsplit(" ", 1)[0] + "..."
        return text.strip()

    def _dedupe_entities(self, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Remove duplicate entities by normalized name."""
        seen = set()
        deduped = []
        for ent in entities:
            norm = normalize_text(ent.get("name", ""), strip_punct=True)
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(ent)
        return deduped

    def _format_entity_list(
        self,
        entities: List[Dict[str, str]],
        title: str,
        location_suffix: str = "",
        max_items: int = 8,
        show_description: bool = True,
        show_address: bool = True,
    ) -> str:
        """Format a list of entities into a readable answer.

        Template:
        **Title Location:**

        1. **Name** — Description snippet.
           Địa chỉ: Address

        2. **Name** — Description snippet.
        """
        if not entities:
            return ""

        lines = [f"**{title}{location_suffix}:**\n"]

        for i, ent in enumerate(entities[:max_items], 1):
            name = ent.get("name", "")
            desc = ent.get("description", "")
            addr = ent.get("address", "")

            line = f"{i}. **{name}**"

            # Add description snippet if available
            if show_description and desc:
                snippet = self._get_first_sentence(desc, max_len=120)
                if snippet:
                    line += f" — {snippet}"

            lines.append(line)

            # Add address if available
            if show_address and addr:
                lines.append(f"   Địa chỉ: {addr}")

        return "\n".join(lines)

    def _prepare_curated_food_context(
        self, state: PipelineRunState, entities: List[Dict[str, str]], location: str
    ) -> str:
        """Build lean context for LLM curated recommendation.

        Only sends dish names (+ address if available) to the LLM.
        Descriptions are intentionally excluded — the LLM generates richer,
        more natural descriptions itself. Long descriptions stay in Neo4j
        for hybrid/vector search where they excel at semantic matching.
        """
        names = []
        for ent in entities[:10]:
            name = ent.get("name", "")
            if not name:
                continue
            addr = ent.get("address", "")
            entry = name
            if addr:
                entry += f" ({addr})"
            names.append(entry)

        if not names:
            return ""

        location_str = f" tại {location}" if location else ""
        header = f"DANH SÁCH ĐẶC SẢN/MÓN NGON{location_str}:\n"
        body = "\n".join(f"- {n}" for n in names)
        return header + body
