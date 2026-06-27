from __future__ import annotations
"""Step 3: Query expansion by appending grounded entity names."""
import logging

logger = logging.getLogger(__name__)


import time



from graph_rag.config import ENABLE_EXPANDED_NAMES_IN_SEARCH_QUERY, MAX_EXPANDED_GROUNDED_NAMES


from ..dto import PipelineRunState


class Step3ExpansionMixin:
    """Mixin providing Step 3 query expansion."""

    def _run_step_3_query_expansion(self, state: PipelineRunState) -> None:
        logger.info("\n [STEP 3] QUERY EXPANSION & ROUTING...")
        step_3_start = time.time()
        if state.grounded_nodes:
            grounded_names = [n.content for n in state.grounded_nodes if n.content]
            logger.info(
                "   -> Grounded Names (preview): "
                f"count={len(grounded_names)}, values={self._preview_list(grounded_names)}"
            )
            if ENABLE_EXPANDED_NAMES_IN_SEARCH_QUERY:
                expanded_names = " ".join(grounded_names[:MAX_EXPANDED_GROUNDED_NAMES])
                if expanded_names:
                    state.search_query = f"{state.search_query} {expanded_names}".strip()
                    logger.info("   -> Expanded Query (flag ON): '%s'", state.search_query)
            else:
                logger.info(
                    "   -> Query expansion by concatenation is disabled (flag OFF); "
                    "using grounded nodes as retrieval seeds only."
                )
        else:
            logger.info("   -> No grounded nodes available for query expansion.")
        logger.info("   -> Search Query used for retrieval: '%s'", state.search_query)
        logger.info("   -> Step 3 completed in %s", self._elapsed(step_3_start))
