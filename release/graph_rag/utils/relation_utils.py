"""Relation type detection — single source of truth."""
from __future__ import annotations

import re
from typing import Optional

from graph_rag.config import RELATIONSHIP_MAP
from graph_rag.utils.text import normalize_text


def detect_relation_type(
    text: str,
    text_norm: Optional[str] = None,
    *,
    normalize_fn=None,
) -> Optional[str]:
    """Detect relation type from context text.

    Args:
        text: Original (un-normalized) text — used for bracket pattern matching.
        text_norm: Pre-normalized text. If None, will be computed via normalize_fn.
        normalize_fn: Custom normalizer. Defaults to graph_rag.utils.text.normalize_text.
    """
    if text_norm is None:
        fn = normalize_fn or normalize_text
        text_norm = fn(text)
    if not text_norm:
        return None

    bracket_match = re.search(r"\[([A-Z_]+)\]\s*->", text)
    if bracket_match:
        return bracket_match.group(1)

    for relation_type, phrase in (RELATIONSHIP_MAP or {}).items():
        phrase_norm = (normalize_fn or normalize_text)(phrase)
        if phrase_norm and f" {phrase_norm} " in f" {text_norm} ":
            return relation_type

    if "lien ket" in text_norm and "buoc" in text_norm:
        return "MULTI_HOP"
    if "thuoc loai" in text_norm:
        return "BELONGS_TO"
    return None
