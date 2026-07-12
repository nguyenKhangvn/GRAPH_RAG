from __future__ import annotations

import re
from typing import Any, Dict
from graph_rag.core.intents import IntentType
from graph_rag.pipeline.orchestration.contract_signals import ADVICE_SIGNALS, BOOKING_SIGNALS

class ContractValidator:
    """Enforces evidence contracts and resolves conflicts across metadata, QueryPlan, and QueryFrame."""

    # Legacy validate() method has been removed as it is dead code. All validation uses detect() instead.
    @staticmethod
    def _match_any_signal(q_norm: str, signals: list[str] | set[str] | frozenset[str]) -> bool:
        """Helper to match list of signals using strict word boundary patterns."""
        if not q_norm or not signals:
            return False
        pattern = r"\b(" + "|".join(re.escape(sig) for sig in signals) + r")\b"
        return bool(re.search(pattern, q_norm))

    @staticmethod
    def detect(q_norm: str, metadata: Dict[str, Any] | None = None) -> "ContractPatch":
        """Detect which contract applies and return a ContractPatch.

        This is the new Phase 3 API — returns immutable patch instead of mutating.
        QueryPlanBuilder.apply_contract_patch() applies the patch.

        ClosedFormGuard: When answer_mode is closed-form (single_option_resolver,
        multi_option_resolver, true_false, fill_blank), block all open-ended contracts.
        Only allow closed-form-specific contracts (category_check, membership_check, etc.)

        Args:
            q_norm: Normalized query text (no diacritics, no punctuation)
            metadata: Optional metadata dict for context (e.g., existing intent, answer_mode)

        Returns:
            ContractPatch with overrides, or empty ContractPatch if no contract matches.
        """
        from graph_rag.pipeline.orchestration.contract_patch import ContractPatch
        from graph_rag.core.answer_mode import AnswerMode

        meta = metadata or {}

        # Clean q_norm to avoid false positives for short/ambiguous words like "pha" (ferry)
        q_clean = q_norm
        q_clean = q_clean.replace("kham pha", "kham_pha")
        q_clean = q_clean.replace("pha lau", "pha_lau")
        q_clean = q_clean.replace("pha co", "pha_co")
        q_clean = q_clean.replace("pha le", "pha_le")
        q_clean = q_clean.replace("pha phach", "pha_phach")
        q_clean = q_clean.replace("pha hoai", "pha_hoai")

        # ── ClosedFormGuard ──
        # Block open-ended contracts when answer_mode is closed-form.
        # Closed-form queries (multiple-choice, true/false, fill-blank) should
        # NOT be overridden by Event/Food/Weather/TravelInfo contracts.
        answer_mode = str(meta.get("answer_mode") or "")
        is_closed_form = AnswerMode.is_closed_form(answer_mode)

        if is_closed_form:
            # Allow only closed-form-specific contracts:
            # - category_check: "đâu là di tích lịch sử" → BELONGS_TO
            # - membership_check: "đâu không nằm trong tour" → INCLUDES
            # - attribute_check: "đâu có giá vé rẻ nhất" → field comparison
            # Block: Event Schedule, Weather, Food Recommendation, TravelInfo,
            # Community, TourPlan, Airport, Local Transport
            return ContractValidator._detect_closed_form_contract(q_clean, meta, ContractPatch)

        # ── Intent-based Guards ──
        intent_str = str(meta.get("intent") or "")
        # If the query is an itinerary/tour plan query, we bypass all label-restricting contracts
        # because tour planning requires a compound mix of attractions, restaurants, and accommodation.
        if intent_str == "TOUR_PLAN":
            return ContractPatch()

        # Rule 1: Ticket Price
        price_signals = ["gia ve", "ve vao", "phi tham quan", "ve tham quan", "ve cong", "mat phi", "phi vao cong"]
        has_price_signal = ContractValidator._match_any_signal(q_clean, price_signals)
        # Guard: "gia ve may bay" is budget/transport, not ticket price
        if has_price_signal and "may bay" in q_clean:
            has_price_signal = False
        if has_price_signal:
            return ContractPatch(
                contract_name="ticket_price",
                intent=IntentType.TOURISM,
                target_labels=("TouristAttraction", "TravelInfo"),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TravelAgency", "Tour", "Event", "Location"),
                fallback_policy="ticket_price_guided_fallback",
                hard_label_contract=True,
                disable_agentic_retrieval=True,
                disable_generic_discovery=True,
                target_class="TouristAttraction",
                semantic_category="ticket_price",
                requested_attributes=("ticket_price",),
                operation="attribute_lookup",
                operator="ticket_price_lookup",
                answer_mode="fact_answer",
            )

        # Rule 1.5: Emergency Support
        emergency_signals = ["khan cap", "cap cuu", "duong day nong", "su co", "cuu ho", "police", "benh vien", "tram y te", "cong an", "cuu thuong", "hotline", "so dien thoai"]
        is_emergency_patch = ContractValidator._match_any_signal(q_clean, emergency_signals) or meta.get("intent") == IntentType.EMERGENCY_SUPPORT
        # Guard: skip emergency if query is phone lookup for specific entity
        if is_emergency_patch and "so dien thoai" in q_clean:
            has_entity_target = bool(meta.get("target_entity") or meta.get("phone_lookup_entity_hint"))
            has_emergency_keyword = ContractValidator._match_any_signal(q_clean, ["khan cap", "cap cuu", "cuu ho", "cuu thuong", "hotline", "cong an", "benh vien", "police"])
            if has_entity_target and not has_emergency_keyword:
                is_emergency_patch = False
        if is_emergency_patch:
            return ContractPatch(
                contract_name="emergency",
                intent=IntentType.EMERGENCY_SUPPORT,
                target_labels=("TravelInfo",),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TouristAttraction", "Event", "Tour"),
                fallback_policy="emergency_guided_fallback",
                hard_label_contract=True,
                disable_agentic_retrieval=True,
                disable_generic_discovery=True,
                disable_discovery_expansion=True,
                disable_food_keywords=True,
                answer_mode="emergency_info_deterministic",
                requested_attributes=("phone", "description"),
                target_class="TravelInfo",
                semantic_category="emergency",
                operation="attribute_lookup",
                operator="emergency_lookup",
            )

        # Rule 1.6: Cashless Payment
        payment_signals = ["thanh toan", "khong dung tien mat", "chuyen khoan", "atm", "ngan hang", "vi dien tu", "quet qr", "qr code", "momo", "vnpay"]
        if ContractValidator._match_any_signal(q_clean, payment_signals) or meta.get("intent") == IntentType.CASHLESS_PAYMENT:
            return ContractPatch(
                contract_name="cashless_payment",
                intent=IntentType.CASHLESS_PAYMENT,
                target_labels=("TravelInfo",),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TouristAttraction", "Event", "Tour"),
                fallback_policy="payment_guided_fallback",
                hard_label_contract=True,
                disable_agentic_retrieval=True,
                disable_generic_discovery=True,
                requested_attributes=("description",),
                target_class="TravelInfo",
                semantic_category="payment",
                operation="attribute_lookup",
                operator="payment_lookup",
            )

        # Weather
        explicit_weather_signals = ["thoi tiet", "nhiet do", "khi hau", "luong mua"]
        if ContractValidator._match_any_signal(q_clean, explicit_weather_signals):
            return ContractPatch(
                contract_name="weather",
                intent=IntentType.WEATHER_ADVICE,
                target_labels=("TravelInfo",),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "Tour", "TravelAgency", "TouristAttraction", "Event"),
                fallback_policy="weather_guided_fallback",
                disable_entity_grounding=True,
                disable_non_location_grounding=True,
                target_class="TravelInfo",
                semantic_category="weather",
                requested_attributes=("weather", "seasonal_advice", "outdoor_suitability"),
                question_shape="advice",
                operation="attribute_lookup",
                operator="weather_advice_lookup",
                answer_mode="advice_lookup",
            )

        # Advice/Tips
        is_advice = ContractValidator._match_any_signal(q_clean, ADVICE_SIGNALS)

        # Booking Advice (advice + accommodation signals)
        is_booking = ContractValidator._match_any_signal(q_clean, BOOKING_SIGNALS)
        if is_advice and is_booking:
            return ContractPatch(
                contract_name="booking_advice",
                intent=IntentType.TRAVEL_ADVICE,
                target_labels=("TravelInfo", "Accommodation", "Location"),
                forbidden_labels=("Dish", "Restaurant", "TouristAttraction", "Tour", "Event"),
                fallback_policy="booking_guided_fallback",
                hard_label_contract=True,
                requested_attributes=("description", "tips"),
                disable_generic_discovery=True,
                disable_agentic_retrieval=True,
                skip_realtime_booking_guard=True,
                target_class="TravelInfo",
                semantic_category="accommodation_tips",
                operation="attribute_lookup",
                operator="booking_advice_lookup",
            )

        # Budget/Cost (advice + budget signals)
        budget_signals = ["chi phi", "kinh phi", "gia ca", "tiet kiem", "ngan sach"]
        is_budget = ContractValidator._match_any_signal(q_clean, budget_signals)
        if is_advice and is_budget:
            return ContractPatch(
                contract_name="budget_cost",
                intent=IntentType.TRAVEL_ADVICE,
                target_labels=("TravelInfo",),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TouristAttraction", "Tour", "Event"),
                fallback_policy="budget_guided_fallback",
                hard_label_contract=True,
                requested_attributes=("description", "tips"),
                disable_generic_discovery=True,
                disable_agentic_retrieval=True,
                target_class="TravelInfo",
                semantic_category="budget",
                operation="attribute_lookup",
                operator="budget_lookup",
            )

        # Health & Safety (advice + health signals)
        health_signals = ["tiem phong", "con trung", "an toan", "suc khoe", "benh"]
        is_health = ContractValidator._match_any_signal(q_clean, health_signals)
        if is_advice and is_health:
            return ContractPatch(
                contract_name="health_safety",
                intent=IntentType.TRAVEL_ADVICE,
                target_labels=("TravelInfo",),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TouristAttraction", "Tour", "Event"),
                fallback_policy="health_guided_fallback",
                hard_label_contract=True,
                requested_attributes=("description", "tips"),
                disable_generic_discovery=True,
                disable_agentic_retrieval=True,
                target_class="TravelInfo",
                semantic_category="health",
                operation="attribute_lookup",
                operator="health_lookup",
            )

        # General Practical (advice without specific category)
        if is_advice:
            return ContractPatch(
                contract_name="general_practical",
                intent=IntentType.TRAVEL_ADVICE,
                target_labels=("TravelInfo",),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TouristAttraction", "Tour", "Event"),
                fallback_policy="general_practical_guided_fallback",
                hard_label_contract=True,
                requested_attributes=("description", "tips"),
                disable_generic_discovery=True,
                disable_agentic_retrieval=True,
                target_class="TravelInfo",
                semantic_category="general",
                operation="attribute_lookup",
                operator="general_practical_lookup",
            )

        # BELONGS_TO Classification
        classification_signals = ["thuoc loai hinh", "la loai hinh", "loai hinh luu tru", "phan loai"]
        if ContractValidator._match_any_signal(q_clean, classification_signals):
            return ContractPatch(
                contract_name="classification",
                intent=IntentType.ENTITY_FACT,
                original_intent=IntentType.ENTITY_FACT,
                target_labels=("TouristAttraction", "TravelInfo", "Location"),
                requested_relations=("BELONGS_TO",),
                requested_attributes=("description", "category", "type"),
                hard_label_contract=True,
                target_class="TouristAttraction",
                operation="attribute_lookup",
                operator="fact_lookup",
            )

        # Community/Forum
        community_signals = ["cong dong", "dien dan", "forum"]
        if ContractValidator._match_any_signal(q_clean, community_signals):
            return ContractPatch(
                contract_name="community_forum",
                intent=IntentType.TRAVEL_ADVICE,
                target_labels=("TravelInfo",),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TouristAttraction", "Tour", "Event", "Location"),
                fallback_policy="community_guided_fallback",
                disable_entity_grounding=True,
                disable_non_location_grounding=True,
                disable_generic_discovery=True,
                disable_agentic_retrieval=True,
                skip_realtime_booking_guard=True,
                requested_attributes=("description", "tips"),
                target_class="TravelInfo",
                semantic_category="community",
                question_shape="advice",
                operation="attribute_lookup",
                operator="community_advice_lookup",
                answer_mode="advice_lookup",
                topic="community",
            )

        # Event Schedule
        event_signals = ["le hoi", "su kien", "festival", "lich dien ra"]
        if ContractValidator._match_any_signal(q_clean, event_signals):
            return ContractPatch(
                contract_name="event_schedule",
                intent=IntentType.EVENT,
                target_labels=("Event", "TravelInfo"),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "Tour", "TravelAgency"),
                fallback_policy="event_schedule_guided_fallback",
                disable_entity_grounding=True,
                target_class="Event",
                semantic_category="event_schedule",
                requested_attributes=("name", "month", "year", "activities", "description"),
                operation="attribute_lookup",
                operator="event_schedule_lookup",
                topic="event",
            )

        # Accommodation Recommendation
        accommodation_signals = ["khach san", "homestay", "resort", "nha nghi", "hostel", "luu tru", "villa"]
        has_accommodation = ContractValidator._match_any_signal(q_clean, accommodation_signals)
        # Guard: do not override food or tourism recommendation intents
        if intent_str in {"FOOD_RECOMMENDATION", "TOURISM_RECOMMENDATION"}:
            has_accommodation = False
        # Guard: do not override distance/direction queries that merely mention a hotel by name
        # e.g. "từ vị trí của tôi đến Khách sạn Mường Thanh đi như thế nào" is DISTANCE_QUERY
        if intent_str == "DISTANCE_QUERY":
            has_accommodation = False
        # Guard: only activate for recommendation queries, not advice/booking queries
        _advice_booking_signals = ["kinh nghiem", "dat phong", "booking", "meo", "luu y", "gia bao nhieu", "chi phi"]
        is_advice_booking = ContractValidator._match_any_signal(q_clean, _advice_booking_signals)
        if has_accommodation and not is_advice_booking:
            return ContractPatch(
                contract_name="accommodation_recommendation",
                intent=IntentType.ACCOMMODATION,
                target_labels=("Accommodation",),
                forbidden_labels=("Dish", "Restaurant", "Event", "Tour", "TravelAgency"),
                target_class="Accommodation",
                semantic_category="accommodation",
                operation="recommendation",
            )

        # Tourism Recommendation (generic sightseeing, not ticket price)
        tourism_signals = ["dia danh", "diem du lich", "diem tham quan", "phong canh", "bien", "dao", "choi dau", "di dau"]
        has_tourism = ContractValidator._match_any_signal(q_clean, tourism_signals)
        # Guard: do not override food or accommodation recommendation intents
        if intent_str in {"FOOD_RECOMMENDATION", "ACCOMMODATION_RECOMMENDATION"}:
            has_tourism = False
        # Detect transport sub-intent within tourism query
        _transport_sub_signals = ["di chuyen", "lam the nao de di", "di nhu the nao", "di bang gi", "phuong tien"]
        has_transport_sub = ContractValidator._match_any_signal(q_clean, _transport_sub_signals)
        # Guard: skip if already matched by ticket price or event contract
        if has_tourism:
            extra = (("transport_hint", True),) if has_transport_sub else ()
            return ContractPatch(
                contract_name="tourism_recommendation",
                intent=IntentType.TOURISM,
                target_labels=("TouristAttraction",),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "Event", "TravelAgency"),
                target_class="TouristAttraction",
                semantic_category="tourism",
                operation="recommendation",
                extra_metadata=extra,
            )

        # Food Specialty
        food_specialty_signals = ["dac san gi", "mon gi", "an gi", "mon ngon", "dac san"]
        has_food_specialty = ContractValidator._match_any_signal(q_clean, food_specialty_signals)
        # Guard: do not override tourism or accommodation recommendation intents
        if intent_str in {"TOURISM_RECOMMENDATION", "ACCOMMODATION_RECOMMENDATION"}:
            has_food_specialty = False
        if has_food_specialty:
            return ContractPatch(
                contract_name="food_specialty",
                intent=IntentType.FOOD,
                answer_mode="curated_recommendation",
                target_labels=("Dish", "Specialty", "Restaurant", "Location"),
                target_class="Specialty",
                semantic_category="food_specialty",
                operation="recommendation",
                target_class_priority="Specialty",
            )

        # Check if the query is actually a spatial routing query (from A to B)
        # to prevent overriding specific directions queries with generic transport info contracts.
        from graph_rag.modules.pipeline_support.distance_intent_service import DistanceQueryParser
        src, dst = DistanceQueryParser.parse(q_clean)
        is_spatial_routing = bool(src and dst)

        # Airport Transport
        airport_signals = ["san bay", "chuyen bay", "may bay"]
        if ContractValidator._match_any_signal(q_clean, airport_signals) and not is_spatial_routing:
            return ContractPatch(
                contract_name="airport_transport",
                intent=IntentType.TRANSPORT_INFO,
                target_labels=("TravelInfo", "Location"),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TouristAttraction", "Tour", "Event"),
                fallback_policy="airport_guided_fallback",
                target_class="TravelInfo",
                semantic_category="airport",
                requested_attributes=("description",),
                disable_generic_discovery=True,
                hard_label_contract=True,
                exempt_location_from_grounding_filter=True,
                operation="attribute_lookup",
                operator="airport_lookup",
            )

        # Local Transport (expanded: di chuyen, phuong tien, dao, cau tau, pha)
        transport_signals = [
            "thue xe", "taxi", "grab", "xe om", "xe buyt",
            "di chuyen", "phuong tien", "lam the nao de di",
            "di nhu the nao", "di bang gi", "di bang phuong tien",
            "tau cau", "phà", "pha", "canoe", "thuyen", "tau",
            "cau tau", "ben tau", "bến phà",
        ]
        if ContractValidator._match_any_signal(q_clean, transport_signals) and not is_spatial_routing:
            return ContractPatch(
                contract_name="local_transport",
                intent=IntentType.TRANSPORT_INFO,
                target_labels=("TravelInfo", "Location"),
                forbidden_labels=("Dish", "Restaurant", "Accommodation", "TouristAttraction", "Tour", "Event"),
                fallback_policy="transport_local_guided_fallback",
                target_class="TravelInfo",
                semantic_category="transport",
                requested_attributes=("description",),
                disable_generic_discovery=True,
                hard_label_contract=True,
                exempt_location_from_grounding_filter=True,
                operation="attribute_lookup",
                operator="transport_local_lookup",
            )

        # No contract matched
        return ContractPatch()

    @staticmethod
    def _detect_closed_form_contract(
        q_clean: str, meta: Dict[str, Any], ContractPatch: type
    ) -> "ContractPatch":
        """Detect contracts specific to closed-form queries.

        Closed-form queries (multiple-choice, true/false, fill-blank) need
        special handling:
        - They should NOT be overridden by open-ended contracts
        - They MAY need category/membership/attribute contracts for resolution

        This method only runs when answer_mode is in CLOSED_FORM_MODES.
        """
        # ── Category check ──
        # "đâu là di tích lịch sử" → need BELONGS_TO / type / category
        category_signals = [
            "di tich", "di san", "loai hinh", "phan loai",
            "thuoc loai", "la loai", "danh muc",
        ]
        if ContractValidator._match_any_signal(q_clean, category_signals):
            return ContractPatch(
                contract_name="closed_form_category",
                semantic_category="classification",
            )

        # ── Membership check ──
        # "đâu không nằm trong tour" → need INCLUDES relation
        membership_signals = [
            "nam trong", "khong nam trong", "thuoc tour",
            "gom co", "bao gom", "includes",
        ]
        if ContractValidator._match_any_signal(q_clean, membership_signals):
            return ContractPatch(
                contract_name="closed_form_membership",
                requested_attributes=("description",),
            )

        # ── Attribute check ──
        # "đâu có giá vé rẻ nhất" → need field comparison
        attribute_signals = [
            "gia ve", "gia ca", "gia re", "gia dat",
            "so dien thoai", "dia chi", "gio mo cua",
        ]
        if ContractValidator._match_any_signal(q_clean, attribute_signals):
            return ContractPatch(
                contract_name="closed_form_attribute",
            )

        # ── Negative check ──
        # "địa điểm nào KHÔNG nằm trong..." → negative selection
        negative_signals = ["khong nam", "khong thuoc", "khong phai", "khong co"]
        if ContractValidator._match_any_signal(q_clean, negative_signals):
            return ContractPatch(
                contract_name="closed_form_negative",
            )

        # No closed-form-specific contract needed
        # Return empty patch — the generic closed-form resolver will handle it
        return ContractPatch()
