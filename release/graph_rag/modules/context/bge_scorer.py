"""BGE cross-encoder scoring for candidate nodes.

Scores each candidate node against the query using a cross-encoder model
(BAAI/bge-reranker-base by default). Writes ``bge_score`` into each
node's ``metadata`` dict so that PolicyRanker can consume it.

Fail-open: any error during scoring is logged and skipped — the pipeline
continues with existing scores unchanged.
"""
from __future__ import annotations

import logging
import time
from typing import List

from graph_rag.core.state import NodeItem

logger = logging.getLogger(__name__)

# Module-level model cache (shared with CrossEncoderTextualReranker)
_model_cache: dict = {}


def _load_model(model_name: str):
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name)
    _model_cache[model_name] = model
    return model


def _build_node_text(node: NodeItem) -> str:
    """Build a text representation of a node for scoring."""
    parts = []
    name = str(node.metadata.get("name") or node.content or "").strip()
    if name:
        parts.append(name)
    desc = str(node.metadata.get("description") or "").strip()
    if desc:
        parts.append(desc[:200])
    loc = str(node.metadata.get("location") or node.metadata.get("address") or "").strip()
    if loc:
        parts.append(loc)
    return " | ".join(parts) if parts else name


def score_candidates_bge(
    query_text: str,
    nodes: List[NodeItem],
    model_name: str = "BAAI/bge-reranker-base",
    weight: float = 2.0,
    timeout_sec: float = 15.0,
) -> int:
    """Score candidate nodes against query using BGE cross-encoder.

    Writes ``node.metadata["bge_score"]`` (weighted) for each node.
    Returns number of nodes scored. Fail-open on any error.
    """
    if not query_text or not nodes:
        return 0

    started = time.time()
    try:
        model = _load_model(model_name)
        texts = [_build_node_text(n) for n in nodes]
        pairs = [(query_text, t) for t in texts]
        raw_scores = model.predict(pairs)

        elapsed_ms = round((time.time() - started) * 1000, 1)
        if elapsed_ms > timeout_sec * 1000:
            logger.warning(
                "BGE candidate scoring timeout: %.0fms > %sms — scores discarded",
                elapsed_ms, timeout_sec * 1000,
            )
            return 0

        scored = 0
        for node, raw_score in zip(nodes, raw_scores):
            weighted = round(float(raw_score) * weight, 3)
            node.metadata["bge_score"] = weighted
            scored += 1

        logger.info(
            "BGE candidate scoring: %d nodes scored in %.0fms (model=%s, weight=%.1f)",
            scored, elapsed_ms, model_name, weight,
        )
        return scored

    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        logger.warning("BGE candidate scoring failed (fail-open): %s", exc)
        return 0
