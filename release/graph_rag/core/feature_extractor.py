"""Feature extractor v1 — keyword-based semantic feature matching.

Determines whether a candidate node matches a semantic feature
(e.g. ``coastal``, ``sunset``, ``island``) using keyword overlap
from ``FEATURE_REGISTRY``.

v1 is fully deterministic: no LLM, no embeddings. Future versions
may add geo-based, attribute-based, or LLM-inferred matching.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set

from graph_rag.core.feature_registry import FEATURE_REGISTRY
from graph_rag.utils.text import normalize_text


class FeatureExtractor:
    """Checks whether a candidate node matches a semantic feature."""

    def __init__(self) -> None:
        # Pre-compile keyword sets per feature for fast lookup.
        self._feature_keywords: Dict[str, List[str]] = {}
        self._feature_labels: Dict[str, Set[str]] = {}
        for feature, spec in FEATURE_REGISTRY.items():
            self._feature_keywords[feature] = list(spec["keywords"])
            self._feature_labels[feature] = set(spec.get("labels") or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_feature(self, node: Any, feature: str) -> bool:
        """Return True if *node* matches the given *feature*.

        Matching is done by normalizing the node's text content
        (name + address + description) and checking for keyword overlap
        from the feature's registry entry.
        """
        return self.match_score(node, feature) > 0.0

    def match_score(self, node: Any, feature: str) -> float:
        """Return a 0.0–1.0 confidence score for *feature* match.

        Scoring heuristic (v1):
        - Base: keyword overlap ratio (matched / total keywords).
        - Bonus +0.2 if any node label intersects the feature's expected labels.
        - Capped at 1.0.
        """
        keywords = self._feature_keywords.get(feature)
        if not keywords:
            return 0.0

        text = self._node_text(node)
        matched = sum(1 for kw in keywords if kw in text)
        if matched == 0:
            return 0.0

        base = matched / len(keywords)

        # Label bonus
        bonus = 0.0
        labels = self._node_labels(node)
        expected = self._feature_labels.get(feature, set())
        if labels & expected:
            bonus = 0.2

        return min(1.0, round(base + bonus, 3))

    def matched_features(self, node: Any) -> List[str]:
        """Return list of all features matched by *node*."""
        return [f for f in self._feature_keywords if self.has_feature(node, f)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _node_text(node: Any) -> str:
        """Extract and normalize text content from various node representations.

        Supports:
        - ``NodeItem`` objects (have ``.metadata`` and ``.content``).
        - Plain dicts (route node format with ``name``, ``attributes``).
        """
        parts: List[str] = []

        if hasattr(node, "metadata"):
            # NodeItem-style
            meta = node.metadata or {}
            parts.append(str(meta.get("name") or ""))
            parts.append(str(meta.get("address") or ""))
            parts.append(str(meta.get("description") or ""))
            parts.append(str(getattr(node, "content", "") or ""))
        elif isinstance(node, dict):
            # Dict-style (route node)
            parts.append(str(node.get("name") or ""))
            attrs = node.get("attributes") or {}
            parts.append(str(attrs.get("address") or node.get("address") or ""))
            parts.append(str(attrs.get("description") or node.get("description") or ""))
        else:
            parts.append(str(node))

        return normalize_text(" ".join(parts), strip_punct=True)

    @staticmethod
    def _node_labels(node: Any) -> Set[str]:
        """Extract labels from various node representations."""
        labels: List[str] = []

        if hasattr(node, "metadata"):
            meta = node.metadata or {}
            labels = meta.get("labels") or []
            if not labels and meta.get("type"):
                labels = [meta.get("type")]
        elif isinstance(node, dict):
            labels = node.get("labels") or []
            if not labels and node.get("type"):
                labels = [node.get("type")]

        return {str(lbl) for lbl in labels if lbl}
