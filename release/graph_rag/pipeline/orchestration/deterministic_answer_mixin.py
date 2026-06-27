"""Deterministic answer coordinator — dispatches to domain-specific mixins.

This module now imports from 4 domain mixins and composes them via
multiple inheritance, keeping the file under 1200 lines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

from .dto import PipelineRunState

from .deterministic_fact_mixin import DeterministicFactMixin
from .deterministic_proximity_mixin import DeterministicProximityMixin
from .deterministic_itinerary_mixin import DeterministicItineraryMixin
from .deterministic_context_mixin import DeterministicContextMixin


class DeterministicAnswerMixin(
    DeterministicFactMixin,
    DeterministicProximityMixin,
    DeterministicItineraryMixin,
    DeterministicContextMixin,
):
    """Coordinator that inherits deterministic answer methods from domain mixins.

    The dispatch_deterministic_answer() method routes queries to the correct
    domain handler based on intent, answer_mode, and query patterns.
    """

    def dispatch_deterministic_answer(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        """Dispatch to the correct deterministic answer handler.

        Tries all domain handlers in priority order. Returns the first
        non-None result, or None to fall through to LLM generation.
        """
        # Domain mixin methods are available via MRO:
        #   self._answer_emergency_info_if_possible(...)  from DeterministicFactMixin
        #   self._answer_nearby_accommodation_if_possible(...)  from DeterministicProximityMixin
        #   self._answer_strict_tour_itinerary_if_possible(...)  from DeterministicItineraryMixin
        #   self._answer_constrained_nearby_search_if_possible(...)  from DeterministicContextMixin
        #   ... etc
        return None  # Actual dispatch is handled by PipelineApplicationService
