"""Node and seed utility functions — single source of truth."""
from __future__ import annotations

from typing import Any, List


def seed_name(seed: Any) -> str:
    """Extract human-readable name from a seed or node object."""
    meta = getattr(seed, "metadata", {}) or {}
    return str(meta.get("name") or getattr(seed, "content", "") or "").strip()


def get_node_labels(node_or_seed: Any) -> List[str]:
    """Extract label list from a node or seed object.

    Falls back to metadata['type'] if 'labels' is empty/missing.
    """
    meta = getattr(node_or_seed, "metadata", {}) or {}
    labels = meta.get("labels") or []
    if not labels and meta.get("type"):
        labels = [meta.get("type")]
    return [str(label) for label in labels if str(label).strip()]
