from __future__ import annotations

from typing import List

from .frame_models import QueryFrame


class QueryFrameValidator:
    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = float(min_confidence)

    def validate(self, frame: QueryFrame) -> QueryFrame:
        errors: List[str] = []
        operator = frame.query_operator

        if frame.confidence < self.min_confidence:
            errors.append("low_confidence")

        if operator == "itinerary_recommendation":
            has_anchor = bool(frame.groundable_mentions or frame.location_scope)
            if not has_anchor:
                errors.append("tour_plan_missing_anchor")

        # tour_availability is a class search — no anchors needed, always valid
        if operator == "tour_availability":
            pass  # no anchor requirement

        if operator == "choice_selection" and len(frame.candidate_entities) < 2:
            errors.append("choice_missing_candidates")

        if operator == "comparison" and len(frame.comparison_subjects) < 2:
            errors.append("comparison_missing_subjects")

        if operator == "dish_to_restaurant" and not frame.retrieval_plan.anchors:
            errors.append("dish_query_missing_dish_anchor")

        if operator == "lodging_near_anchor" and not frame.retrieval_plan.anchors:
            errors.append("lodging_near_missing_anchor")

        if operator == "constrained_nearby_search":
            chain = (frame.retrieval_plan.context_policy or {}).get("chain") or []
            if len(chain) < 2:
                errors.append("chain_too_short")

        for phrase in frame.non_groundable_phrases:
            phrase_norm = phrase.strip().lower()
            for mention in frame.groundable_mentions:
                if mention.text.strip().lower() == phrase_norm:
                    errors.append("non_groundable_used_as_groundable")

        frame.validation_errors = sorted(set(errors))
        frame.valid = not frame.validation_errors
        if not frame.valid:
            frame.fallback_reason = ",".join(frame.validation_errors)
        return frame
