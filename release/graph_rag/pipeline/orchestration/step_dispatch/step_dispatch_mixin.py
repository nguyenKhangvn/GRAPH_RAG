from __future__ import annotations
"""Facade composing all step dispatch mixins.

This module re-exports StepDispatchMixin which inherits from all sub-mixins.
application_service.py imports only this class — no API change needed.
"""

from .step1_query_understanding_mixin import Step1QueryUnderstandingMixin
from .step2_grounding_mixin import Step2GroundingMixin
from .step3_expansion_mixin import Step3ExpansionMixin
from .step4_retrieval_mixin import Step4RetrievalMixin
from .step5_generation_mixin import Step5GenerationMixin
from .guided_fallback_mixin import GuidedFallbackMixin
from .discovery_dispatch_mixin import DiscoveryDispatchMixin
from .tour_dispatch_mixin import TourDispatchMixin
from .airport_dispatch_mixin import AirportDispatchMixin
from .pipeline_response_mixin import PipelineResponseMixin
from .v3_structured_generation_mixin import V3StructuredGenerationMixin


class StepDispatchMixin(
    Step1QueryUnderstandingMixin,
    Step2GroundingMixin,
    Step3ExpansionMixin,
    Step4RetrievalMixin,
    Step5GenerationMixin,
    GuidedFallbackMixin,
    DiscoveryDispatchMixin,
    TourDispatchMixin,
    AirportDispatchMixin,
    PipelineResponseMixin,
    V3StructuredGenerationMixin,
):
    """Facade composing all step dispatch mixins.

    application_service.py imports only this class — no API change needed.
    Each sub-mixin is defined in its own file for maintainability.
    """
