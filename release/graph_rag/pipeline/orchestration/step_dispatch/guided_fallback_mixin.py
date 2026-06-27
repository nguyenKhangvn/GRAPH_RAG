from __future__ import annotations
"""Guided fallback lookups and grounded fallback answer builder."""
import logging

logger = logging.getLogger(__name__)




from typing import Any, Dict, List, Optional



from graph_rag.utils.text import normalize_text


from graph_rag.config.region_patterns import TYPE_HEADER_MAP


from ..dto import PipelineRunState


class GuidedFallbackMixin:
    """Mixin providing TravelInfo fallback lookups and grounded fallback answers."""

    def _get_fallback_location_list(self, location: str) -> list[str]:
        if not location:
            return []
        loc_list = [location.lower()]
        from graph_rag.utils.text import normalize_text
        norm = normalize_text(location, strip_punct=True)
        if norm not in loc_list:
            loc_list.append(norm)

        # Add region/province aliases from RegionRegistry
        from graph_rag.config.region_registry import region_registry
        matched_pids = region_registry.get_province_by_keyword(location)
        # Also follow merge targets so merged provinces share each other's terms
        expanded_pids: list[str] = []
        for pid in matched_pids:
            if pid not in expanded_pids:
                expanded_pids.append(pid)
            target = region_registry.get_merge_target(pid)
            if target and target not in expanded_pids:
                expanded_pids.append(target)
            # Also include provinces merged INTO this one
            for merged in region_registry.get_merged_provinces(pid):
                if merged not in expanded_pids:
                    expanded_pids.append(merged)

        for pid in expanded_pids:
            for alias in region_registry.get_aliases(pid):
                a = alias.lower()
                if a not in loc_list:
                    loc_list.append(a)
            for kw in region_registry.get_keywords(pid):
                k = kw.lower()
                if k not in loc_list:
                    loc_list.append(k)
        return loc_list

    def _travel_info_lookup(self, topic: str, location: str = "") -> str:
        """Lookup TravelInfo by topic, filter location in Python (handles Vietnamese diacritics)."""
        try:
            with self.pipeline.driver.session() as session:
                query = """
                MATCH (t:TravelInfo)
                WHERE t.topic = $topic
                RETURN t.name AS name, t.description AS description,
                       t.location AS location, t.region AS region, t.province AS province
                """
                rows = session.run(query, topic=topic).data()
                if not rows:
                    return ""
                # Filter by location in Python (normalize_text handles diacritics)
                if location:
                    loc_norm = normalize_text(location, strip_punct=True)
                    filtered = []
                    for row in rows:
                        haystack = " ".join(normalize_text(str(row.get(f, "") or ""), strip_punct=True)
                                            for f in ["name", "description", "location", "region", "province"])
                        if loc_norm in haystack:
                            filtered.append(row)
                    rows = filtered or rows  # fallback: use all if none match
                return "\n".join(row["description"] for row in rows if row.get("description"))
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error lookup TravelInfo (topic=%s): %s", topic, e)
            return ""

    def _fallback_lookup_weather(self, location: str) -> str:
        return self._travel_info_lookup("weather", location)

    def _fallback_lookup_airport(self, location: str) -> str:
        return self._travel_info_lookup("airport", location)

    def _fallback_lookup_seafood_shopping(self, location: str) -> str:
        return self._travel_info_lookup("shopping", location)

    def _fallback_lookup_emergency(self, location: str) -> str:
        """Emergency fallback with contact info."""
        try:
            with self.pipeline.driver.session() as session:
                rows = session.run(
                    "MATCH (t:TravelInfo) WHERE t.topic = 'emergency' "
                    "RETURN t.name AS name, t.description AS description, t.contact AS contact LIMIT 3"
                ).data()
                if not rows:
                    return ""
                # Filter by location in Python
                if location:
                    loc_norm = normalize_text(location, strip_punct=True)
                    filtered = [r for r in rows if loc_norm in normalize_text(
                        " ".join(str(r.get(f, "") or "") for f in ["name", "description"]), strip_punct=True)]
                    rows = filtered or rows
                parts = []
                for row in rows:
                    part = f"**{row['name']}**\n{row['description']}"
                    if row.get('contact'):
                        part += f"\nHotline/Liên hệ: {row['contact']}"
                    parts.append(part)
                return "\n\n".join(parts)
        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Error lookup emergency: %s", e)
            return ""

    def _fallback_lookup_payment(self, location: str) -> str:
        return self._travel_info_lookup("payment", location)

    def _fallback_lookup_booking(self, location: str) -> str:
        return self._travel_info_lookup("accommodation_tips", location)

    def _fallback_lookup_transport_local(self, location: str) -> str:
        return self._travel_info_lookup("transport", location)

    def _fallback_lookup_budget(self, location: str) -> str:
        return self._travel_info_lookup("budget", location)

    def _fallback_lookup_health(self, location: str) -> str:
        return self._travel_info_lookup("health", location)

    def _fallback_lookup_community(self, location: str) -> str:
        return self._travel_info_lookup("community", location)

    def _fallback_lookup_event_schedule(self, location: str) -> str:
        return self._travel_info_lookup("event", location)

    def _fallback_lookup_general_practical(self, location: str) -> str:
        try:
            with self.pipeline.driver.session() as session:
                rows = session.run(
                    "MATCH (t:TravelInfo) WHERE t.topic IN ['general', 'accommodation_tips', 'budget'] "
                    "RETURN t.name AS name, t.description AS description, "
                    "t.location AS location, t.region AS region, t.province AS province LIMIT 5"
                ).data()
                if not rows:
                    return ""
                if location:
                    loc_norm = normalize_text(location, strip_punct=True)
                    filtered = [r for r in rows if loc_norm in normalize_text(
                        " ".join(str(r.get(f, "") or "") for f in ["name", "description", "location", "region", "province"]),
                        strip_punct=True)]
                    rows = filtered or rows
                return "\n".join(row["description"] for row in rows if row.get("description"))
        except (ValueError, TypeError) as e:
            logger.error("Error lookup general practical: %s", e)
            return ""

    def _check_guided_fallbacks(self, state: PipelineRunState) -> Optional[str]:
        from graph_rag.utils.text import normalize_text
        metadata = state.metadata or {}

        # 1. Weather guided fallback
        if metadata.get("fallback_policy") == "weather_guided_fallback":
            weather_keywords = ["thoi tiet", "nhiet do", "mua", "nang", "weather", "temperature", "khi hau"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_weather_info = any(kw in context_norm for kw in weather_keywords)
            if not has_weather_info:
                weather_data = self._fallback_lookup_weather(state.location)
                if weather_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {weather_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin thời tiết] {weather_data}"
                    logger.info("   -> [Phase5] Injected weather TravelInfo into context.")
                    return weather_data
                else:
                    return "Dữ liệu hiện chưa có thông tin thời tiết cụ thể cho khu vực/thời gian này. Bạn nên theo dõi dự báo thời tiết cục bộ gần ngày khởi hành."

        # 1b. Airport/Transport guided fallback
        if metadata.get("fallback_policy") == "airport_guided_fallback":
            airport_keywords = ["san bay", "bay thang", "chuyen bay", "pxu", "pleiku"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_airport_info = any(kw in context_norm for kw in airport_keywords)
            if not has_airport_info:
                airport_data = self._fallback_lookup_airport(state.location or "Pleiku")
                if airport_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {airport_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin sân bay/đường bay] {airport_data}"
                    logger.info("   -> [Phase5] Injected airport TravelInfo into context.")
                    return airport_data
                else:
                    return "Dữ liệu hiện chưa có thông tin cụ thể về sân bay hoặc các đường bay thẳng cho khu vực này."

        # 1c. Seafood shopping guided fallback
        if metadata.get("fallback_policy") == "seafood_shopping_guided_fallback":
            seafood_keywords = ["hai san", "tom", "cua", "ghe", "oc", "so", "muc", "seafood", "cho", "cang", "vua"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_seafood_info = any(kw in context_norm for kw in seafood_keywords)
            if not has_seafood_info:
                shopping_data = self._fallback_lookup_seafood_shopping(state.location or "Quy Nhơn")
                if shopping_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {shopping_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin mua hải sản] {shopping_data}"
                    logger.info("   -> [Phase5] Injected seafood shopping TravelInfo into context.")
                    return shopping_data
                else:
                    return "Dữ liệu hiện chưa có thông tin cụ thể về địa điểm mua hải sản tươi sống tại khu vực này."

        # 2. Ticket Price guided fallback
        if metadata.get("fallback_policy") == "ticket_price_guided_fallback":
            price_keywords = ["gia ve", "ve vao", "phi tham quan", "ve tham quan", "ve cong", "ticket_price", "price", "vnd", "dong"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_price_info = any(kw in context_norm for kw in price_keywords)
            if not has_price_info:
                return "Dữ liệu hiện chưa ghi nhận giá vé cụ thể cho địa điểm này. Hệ thống chỉ có thông tin mô tả/vị trí, bạn nên liên hệ trực tiếp điểm đến để kiểm tra giá vé mới nhất trước khi đi."

        # 3. Food/Seafood guided fallback
        if metadata.get("require_seafood_match"):
            seafood_keywords = ["hai san", "tom", "cua", "ghe", "oc", "so", "muc", "seafood"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_seafood_info = any(kw in context_norm for kw in seafood_keywords)
            if not has_seafood_info:
                return "Dữ liệu hiện chưa có thông tin chi tiết về các quán ăn phục vụ hải sản tại khu vực này. Bạn có thể tham khảo các đặc sản khác hoặc nhà hàng ẩm thực địa phương khác trong dữ liệu."

        # 4. Emergency guided fallback
        if metadata.get("fallback_policy") == "emergency_guided_fallback":
            emergency_keywords = ["khan cap", "cap cuu", "duong day nong", "su co", "cuu ho", "police", "benh vien", "tram y te", "cong an", "cuu thuong", "hotline", "sdt", "dien thoai"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_emergency_info = any(kw in context_norm for kw in emergency_keywords)
            if not has_emergency_info:
                emergency_data = self._fallback_lookup_emergency(state.location)
                if emergency_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {emergency_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin hỗ trợ khẩn cấp] {emergency_data}"
                    logger.info("   -> [Phase5] Injected emergency TravelInfo into context.")
                    return emergency_data
                else:
                    return "Dữ liệu hiện chưa có thông tin hỗ trợ khẩn cấp hoặc đường dây nóng cụ thể cho khu vực này."

        # 5. Payment guided fallback
        if metadata.get("fallback_policy") == "payment_guided_fallback":
            payment_keywords = ["thanh toan", "khong dung tien mat", "chuyen khoan", "atm", "ngan hang", "vi dien tu", "quet qr", "qr code", "momo", "vnpay"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_payment_info = any(kw in context_norm for kw in payment_keywords)
            if not has_payment_info:
                payment_data = self._fallback_lookup_payment(state.location)
                if payment_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {payment_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin thanh toán] {payment_data}"
                    logger.info("   -> [Phase5] Injected payment TravelInfo into context.")
                    return payment_data
                else:
                    return "Dữ liệu hiện chưa có thông tin cụ thể về các phương thức thanh toán không tiền mặt tại khu vực này."

        # 6. Booking guided fallback
        if metadata.get("fallback_policy") == "booking_guided_fallback":
            booking_keywords = ["dat phong", "booking", "book phong", "khach san", "nha nghi", "resort", "homestay", "tips", "meo", "luu y", "kinh nghiem"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_booking_info = any(kw in context_norm for kw in booking_keywords)
            if not has_booking_info:
                booking_data = self._fallback_lookup_booking(state.location)
                if booking_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {booking_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Kinh nghiệm đặt phòng] {booking_data}"
                    logger.info("   -> [Phase5] Injected booking advice TravelInfo into context.")
                    return booking_data
                else:
                    return "Dữ liệu hiện chưa ghi nhận kinh nghiệm hoặc lưu ý cụ thể về đặt phòng khách sạn tại khu vực này."

        # 7. Local Transport guided fallback
        if metadata.get("fallback_policy") == "transport_local_guided_fallback":
            local_trans_keywords = ["thue xe", "taxi", "xe cong nghe", "di lai", "di chuyen", "phuong tien", "xe khach", "limousine", "grab", "xanh sm", "xe om"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_local_trans_info = any(kw in context_norm for kw in local_trans_keywords)
            if not has_local_trans_info:
                local_trans_data = self._fallback_lookup_transport_local(state.location)
                if local_trans_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {local_trans_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin phương tiện di chuyển] {local_trans_data}"
                    logger.info("   -> [Phase5] Injected local transport TravelInfo into context.")
                    return local_trans_data
                else:
                    return "Dữ liệu hiện chưa có thông tin cụ thể về các phương tiện di chuyển nội tỉnh tại khu vực này."

        # 8. Budget guided fallback
        if metadata.get("fallback_policy") == "budget_guided_fallback":
            budget_keywords = ["chi phi", "kinh phi", "gia ca", "ton bao nhieu", "chi tieu", "bang gia", "gia ve", "phi tham quan"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_budget_info = any(kw in context_norm for kw in budget_keywords)
            if not has_budget_info:
                budget_data = self._fallback_lookup_budget(state.location)
                if budget_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {budget_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin chi phí & giá cả] {budget_data}"
                    logger.info("   -> [Phase5] Injected budget TravelInfo into context.")
                    return budget_data
                else:
                    return "Dữ liệu hiện chưa ghi nhận thông tin chi phí du lịch cụ thể cho khu vực này."

        # 9. Health & Safety guided fallback
        if metadata.get("fallback_policy") == "health_guided_fallback":
            health_keywords = ["tiem phong", "con trung", "sot xuat huyet", "an toan", "duong deo", "y te", "suc khoe", "thuoc men", "benh tat"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_health_info = any(kw in context_norm for kw in health_keywords)
            if not has_health_info:
                health_data = self._fallback_lookup_health(state.location)
                if health_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {health_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Lưu ý y tế & an toàn] {health_data}"
                    logger.info("   -> [Phase5] Injected health TravelInfo into context.")
                    return health_data
                else:
                    return "Dữ liệu hiện chưa ghi nhận thông tin y tế, tiêm phòng hoặc lưu ý an toàn cụ thể cho địa điểm này."

        # 10. Community guided fallback
        if metadata.get("fallback_policy") == "community_guided_fallback":
            community_keywords = ["cong dong", "dien dan", "forum", "nhom du lich", "chia se trai nghiem", "review", "hoi nhom"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_community_info = any(kw in context_norm for kw in community_keywords)
            if not has_community_info:
                community_data = self._fallback_lookup_community(state.location)
                if community_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {community_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin cộng đồng du lịch] {community_data}"
                    logger.info("   -> [Phase5] Injected community TravelInfo into context.")
                    return community_data
                else:
                    return "Dữ liệu hiện chưa có thông tin về các hội nhóm hoặc diễn đàn du lịch cụ thể cho khu vực này."

        # 11. Event Schedule guided fallback
        if metadata.get("fallback_policy") == "event_schedule_guided_fallback":
            event_keywords = ["le hoi", "su kien", "festival", "dien ra", "van hoa", "giai chay"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_event_info = any(kw in context_norm for kw in event_keywords)
            if not has_event_info:
                event_data = self._fallback_lookup_event_schedule(state.location)
                if event_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {event_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Thông tin lễ hội/sự kiện] {event_data}"
                    logger.info("   -> [Phase5] Injected event schedule TravelInfo into context.")
                    return event_data
                else:
                    return "Dữ liệu hiện chưa có thông tin về các sự kiện hoặc lễ hội sắp diễn ra tại khu vực này."

        # 12. General Practical guided fallback
        if metadata.get("fallback_policy") == "general_practical_guided_fallback":
            general_keywords = ["cam nang", "luu y", "kinh nghiem", "meo", "tips", "chuan bi"]
            context_text = " ".join(str(f) for f in (state.raw_context or []))
            context_norm = normalize_text(context_text, strip_punct=True)

            has_general_info = any(kw in context_norm for kw in general_keywords)
            if not has_general_info:
                general_data = self._fallback_lookup_general_practical(state.location)
                if general_data:
                    state.raw_context = list(state.raw_context or []) + [f"[TravelInfo] {general_data}"]
                    state.clean_context = (state.clean_context or "") + f"\n[Cẩm nang lưu ý chung] {general_data}"
                    logger.info("   -> [Phase5] Injected general practical TravelInfo into context.")
                    return general_data
                else:
                    return "Dữ liệu hiện chưa có các cẩm nang hoặc lưu ý du lịch thực tế khác cho khu vực này."

        return None

    def _build_grounded_fallback(self, state: PipelineRunState, candidate_nodes: List[Dict[str, Any]]) -> str:
        from graph_rag.core.state import QuestionShape

        # 1. Resolve allowed labels
        allowed_labels = []
        if getattr(state, "query_plan", None) is not None and getattr(state.query_plan, "intent", "UNKNOWN") != "UNKNOWN":
            try:
                from graph_rag.core.retrieval_policy import RetrievalPolicy
                policy = RetrievalPolicy.resolve_policy_from_query_plan(state.query_plan)
                allowed_labels = policy.allowed_labels
            except (ValueError, RuntimeError, OSError):
                pass
        if not allowed_labels:
            from graph_rag.core.retrieval_policy import RetrievalPolicy
            from graph_rag.core.intents import IntentType
            plan = getattr(state, "query_plan", None)
            primary_intent = plan.intent if plan and plan.intent != "UNKNOWN" else getattr(state, "primary_intent", IntentType.DISCOVERY)
            base = RetrievalPolicy.BASE_POLICIES.get(primary_intent, RetrievalPolicy.BASE_POLICIES[IntentType.DISCOVERY])
            allowed_labels = base.get("allowed_labels", [])

        # 2. Count label distribution
        label_counts = {}
        for c in (candidate_nodes or []):
            if isinstance(c, dict):
                c_type = c.get("type") or ""
            else:
                c_type = (c.metadata.get("labels") or [c.metadata.get("type") or ""])[0]
            if c_type:
                label_counts[c_type] = label_counts.get(c_type, 0) + 1

        # 3. Soft filter candidates (policy labels + candidate label distribution + confidence)
        filtered_candidates = []
        for c in (candidate_nodes or []):
            c_name = c.get("name") if isinstance(c, dict) else getattr(c, "content", "")
            if not c_name:
                continue
            c_type = c.get("type") if isinstance(c, dict) else (c.metadata.get("labels") or [c.metadata.get("type") or ""])[0]
            if c_type in allowed_labels:
                filtered_candidates.append(c)

        if not filtered_candidates:
            filtered_candidates = candidate_nodes or []

        target_class = None
        target_confidence = 0.0
        if getattr(state, "query_plan", None) is not None and state.query_plan.target_class:
            target_class = getattr(state.query_plan, "target_class", None)
            target_confidence = getattr(state.query_plan, "target_class_confidence", 0.0)
        if (not target_class or target_confidence < 0.1) and getattr(state, "query_state", None) is not None:
            target_class = getattr(state.query_state, "target_class", None)
            target_confidence = getattr(state.query_state, "target_class_confidence", 0.0)

        final_candidates = []
        if target_class and target_confidence >= 0.8:
            final_candidates = [
                c for c in filtered_candidates
                if (c.get("type") if isinstance(c, dict) else (c.metadata.get("labels") or [c.metadata.get("type") or ""])[0]) == target_class
            ]

        if not final_candidates:
            if label_counts:
                allowed_label_counts = {lbl: count for lbl, count in label_counts.items() if lbl in allowed_labels}
                if allowed_label_counts:
                    dominant_label = max(allowed_label_counts, key=allowed_label_counts.get)
                    final_candidates = [
                        c for c in filtered_candidates
                        if (c.get("type") if isinstance(c, dict) else (c.metadata.get("labels") or [c.metadata.get("type") or ""])[0]) == dominant_label
                    ]

        if not final_candidates:
            final_candidates = filtered_candidates

        # 4. Format based on shape
        shape = QuestionShape.UNKNOWN
        if getattr(state, "query_plan", None) is not None and state.query_plan.question_shape != QuestionShape.UNKNOWN:
            shape = getattr(state.query_plan, "question_shape", QuestionShape.UNKNOWN)
        if shape == QuestionShape.UNKNOWN and getattr(state, "query_state", None) is not None:
            shape = getattr(state.query_state, "question_shape", QuestionShape.UNKNOWN)

        loc_suffix = f" ở {state.location}" if getattr(state, "location", None) else ""

        dominant_type_vn = "địa điểm / thực thể"
        if final_candidates:
            first_c = final_candidates[0]
            first_type = first_c.get("type") if isinstance(first_c, dict) else (first_c.metadata.get("labels") or [first_c.metadata.get("type") or ""])[0]
            dominant_type_vn = TYPE_HEADER_MAP.get(first_type, "địa điểm / thực thể")

        lines = [line.strip() for line in (state.clean_context or "").splitlines() if line.strip()]

        # If the query asks about a specific attribute (e.g. ticket_price) and
        # the context doesn't contain it, return a "no data" message instead of
        # just listing entity names.
        requested_attrs = (state.metadata or {}).get("requested_attributes") or []
        if requested_attrs:
            context_blob = normalize_text(" ".join(lines), strip_punct=True)
            _ATTR_LABELS = {
                "ticket_price": "giá vé",
                "price": "giá",
                "price_range": "mức giá",
                "phone": "số điện thoại",
                "opening_hours": "giờ mở cửa",
                "address": "địa chỉ",
                "service_features": "dịch vụ",
            }
            missing_labels = []
            for attr in requested_attrs:
                if attr not in _ATTR_LABELS:
                    continue
                attr_norm = normalize_text(attr, strip_punct=True)
                # Check if any keyword from the attribute appears in context
                attr_keywords = {
                    "ticket_price": ["gia ve", "ve vao", "phi vao", "gia vao"],
                    "price": ["gia ", "gia:", "gia la"],
                    "price_range": ["gia ", "muc gia"],
                    "phone": ["sdt", "dien thoai", "lien he"],
                    "opening_hours": ["mo cua", "gio mo", "thoi gian"],
                    "address": ["dia chi", "nam tai", "o "],
                    "service_features": ["dich vu", "tien nghi"],
                }
                keywords = attr_keywords.get(attr, [attr_norm])
                if not any(kw in context_blob for kw in keywords):
                    label = _ATTR_LABELS.get(attr, attr)
                    missing_labels.append(label)
            if missing_labels and final_candidates:
                entity_names = []
                for c in final_candidates[:3]:
                    name = c.get("name") if isinstance(c, dict) else getattr(c, "content", "")
                    if name:
                        entity_names.append(name)
                entities_str = ", ".join(entity_names) if entity_names else "các địa điểm được hỏi"
                attr_str = ", ".join(missing_labels)
                return (
                    f"Dữ liệu hiện có chưa có thông tin {attr_str} cho {entities_str}. "
                    f"Bạn có thể liên hệ trực tiếp hoặc kiểm tra trên các trang du lịch chính thức để có thông tin cập nhật."
                )

        if shape in (QuestionShape.LIST, QuestionShape.LIST_RANKING, QuestionShape.RECOMMENDATION_LIST):
            # LIST / RECOMMENDATION_LIST → liệt kê 4-6 entity grounded
            parts = [f"**Dựa trên dữ liệu đã truy xuất, một số {dominant_type_vn}{loc_suffix} gồm:**\n"]
            seen_names = set()
            count = 0
            for c in final_candidates:
                name = c.get("name") if isinstance(c, dict) else getattr(c, "content", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    # Extract description snippet if available
                    desc = c.get("description") or "" if isinstance(c, dict) else ""
                    if not desc and not isinstance(c, dict):
                        desc = getattr(c, "metadata", {}).get("description") or ""
                    line = f"{count + 1}. **{name}**"
                    if desc:
                        # Lấy câu đầu tiên, tối đa 120 chars
                        first_sentence = desc.split(".")[0].strip()
                        if len(first_sentence) > 120:
                            first_sentence = first_sentence[:117] + "..."
                        line += f" — {first_sentence}"
                    parts.append(line)
                    count += 1
                    if count >= 6:
                        break
            if count > 0:
                return "\n".join(parts)
            else:
                return f"Hiện tại dữ liệu của hệ thống chưa ghi nhận {dominant_type_vn}{loc_suffix}."

        elif shape in (QuestionShape.SINGLE_FACT, QuestionShape.YES_NO):
            # SINGLE_FACT / YES_NO → trả fact ngắn từ context
            target_name = (state.metadata or {}).get("target_entity") or ""
            if not target_name and getattr(state, "grounded_nodes", None):
                target_name = state.grounded_nodes[0].metadata.get("name") or state.grounded_nodes[0].content

            relevant_lines = []
            if target_name:
                target_norm = normalize_text(target_name, strip_punct=True)
                for line in lines:
                    if target_norm in normalize_text(line, strip_punct=True):
                        relevant_lines.append(line)

            if not relevant_lines:
                for line in lines:
                    for c in final_candidates[:3]:
                        c_name = c.get("name") if isinstance(c, dict) else getattr(c, "content", "")
                        if c_name and normalize_text(c_name, strip_punct=True) in normalize_text(line, strip_punct=True):
                            relevant_lines.append(line)
                            break

            if not relevant_lines:
                relevant_lines = lines

            relevant_lines = sorted(relevant_lines, key=len)
            selected_facts = []
            for line in relevant_lines:
                cleaned = line.lstrip("-* ").strip()
                if cleaned and cleaned not in selected_facts:
                    selected_facts.append(cleaned)
                    if len(selected_facts) >= 2:
                        break

            if selected_facts:
                return f"Dựa trên dữ liệu đã truy xuất, thông tin ghi nhận được: {'. '.join(selected_facts)}."
            else:
                target_desc = target_name or "đối tượng"
                return f"Dựa trên dữ liệu đã truy xuất, có thông tin về {target_desc} nhưng chưa đủ dữ kiện để khẳng định chi tiết."

        elif shape == QuestionShape.COMPARISON:
            # COMPARISON → nhóm fact theo subject
            subjects = self._comparison_subject_names(state)
            if not subjects:
                subjects = [c.get("name") if isinstance(c, dict) else getattr(c, "content", "") for c in final_candidates[:2]]
                subjects = [s for s in subjects if s]

            grouped_facts = {}
            for sub in subjects:
                grouped_facts[sub] = []
                sub_norm = normalize_text(sub, strip_punct=True)
                for line in lines:
                    if sub_norm in normalize_text(line, strip_punct=True):
                        cleaned = line.lstrip("-* ").strip()
                        if cleaned and cleaned not in grouped_facts[sub]:
                            grouped_facts[sub].append(cleaned)

            parts = ["Dựa trên dữ liệu đã truy xuất, dưới đây là thông tin so sánh giữa các đối tượng:"]
            has_facts = False
            for sub, facts in grouped_facts.items():
                if facts:
                    has_facts = True
                    parts.append(f"\n**{sub}**:")
                    for fact in facts[:3]:
                        parts.append(f"- {fact}")
                else:
                    parts.append(f"\n**{sub}**: Không có đủ thông tin chi tiết trong dữ liệu.")

            if has_facts:
                return "\n".join(parts)
            else:
                parts = ["**Dựa trên dữ liệu đã truy xuất, danh sách các đối tượng so sánh gồm:**\n"]
                count = 0
                for c in final_candidates[:6]:
                    name = c.get("name") if isinstance(c, dict) else getattr(c, "content", "")
                    if name:
                        parts.append(f"{count + 1}. **{name}**")
                        count += 1
                return "\n".join(parts)

        elif shape == QuestionShape.ITINERARY:
            # ITINERARY → nếu không đủ context thì trả danh sách grounded thay vì bịa lịch trình
            categories = {
                "Điểm tham quan / Trải nghiệm": [],
                "Ẩm thực / Đặc sản": [],
                "Nơi lưu trú": [],
                "Địa điểm khác": []
            }
            for c in final_candidates:
                c_name = c.get("name") if isinstance(c, dict) else getattr(c, "content", "")
                c_type = c.get("type") if isinstance(c, dict) else (c.metadata.get("labels") or [c.metadata.get("type") or ""])[0]
                if c_type in ["TouristAttraction", "Tour"]:
                    categories["Điểm tham quan / Trải nghiệm"].append(c_name)
                elif c_type in ["Restaurant", "Dish"]:
                    categories["Ẩm thực / Đặc sản"].append(c_name)
                elif c_type in ["Accommodation"]:
                    categories["Nơi lưu trú"].append(c_name)
                else:
                    categories["Địa điểm khác"].append(c_name)

            parts = ["Dựa trên dữ liệu đã truy xuất, dưới đây là danh sách các địa điểm được đề xuất cho lịch trình du lịch của bạn:"]
            has_content = False
            for cat_name, names in categories.items():
                if names:
                    has_content = True
                    parts.append(f"\n**{cat_name}**:")
                    seen = set()
                    count = 0
                    for name in names:
                        if name not in seen:
                            seen.add(name)
                            parts.append(f"- {name}")
                            count += 1
                            if count >= 4:
                                break
            if has_content:
                return "\n".join(parts)
            else:
                parts = ["**Một số địa điểm nổi bật:**\n"]
                count = 0
                for c in final_candidates[:6]:
                    name = c.get("name") if isinstance(c, dict) else getattr(c, "content", "")
                    desc = c.get("description") or "" if isinstance(c, dict) else ""
                    line = f"{count + 1}. **{name}**"
                    if desc:
                        first_sentence = desc.split(".")[0].strip()
                        if len(first_sentence) > 120:
                            first_sentence = first_sentence[:117] + "..."
                        line += f" — {first_sentence}"
                    parts.append(line)
                    count += 1
                return "\n".join(parts)

        else:
            parts = [f"**Dựa trên dữ liệu đã truy xuất, một số {dominant_type_vn}{loc_suffix} gồm:**\n"]
            seen_names = set()
            count = 0
            for c in final_candidates:
                name = c.get("name") if isinstance(c, dict) else getattr(c, "content", "")
                if name and name not in seen_names:
                    seen_names.add(name)
                    desc = c.get("description") or "" if isinstance(c, dict) else ""
                    line = f"{count + 1}. **{name}**"
                    if desc:
                        first_sentence = desc.split(".")[0].strip()
                        if len(first_sentence) > 120:
                            first_sentence = first_sentence[:117] + "..."
                        line += f" — {first_sentence}"
                    parts.append(line)
                    count += 1
                    if count >= 6:
                        break
            if count > 0:
                return "\n".join(parts)
            else:
                return f"Hiện tại dữ liệu của hệ thống chưa ghi nhận thông tin {dominant_type_vn}{loc_suffix}."
