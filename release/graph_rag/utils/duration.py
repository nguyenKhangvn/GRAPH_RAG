"""Duration extraction utilities — single source of truth.

Consolidates infer_duration (query_fields) and extract_trip_duration (tour_plan_support).
"""
from __future__ import annotations

import re
from typing import Tuple


def infer_duration(q_norm: str) -> Tuple[int, int]:
    """Extract (days, nights) from a normalized query string.

    Handles: "X ngay Y dem", "X ngay", "X dem", accent variants.
    Returns (0, 0) if no duration found.
    """
    # Pattern: "X ngay Y dem"
    m = re.search(r"(\d+)\s*ng[ay]+\s*(\d+)\s*dem", q_norm)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Pattern: "X ngay" only
    m = re.search(r"(\d+)\s*ng[ay]+", q_norm)
    if m:
        days = int(m.group(1))
        return days, max(0, days - 1)
    # Pattern: "X dem" only
    m = re.search(r"(\d+)\s*dem", q_norm)
    if m:
        nights = int(m.group(1))
        return nights + 1, nights
    return 0, 0


def extract_trip_duration(query: str, normalize_fn) -> Tuple[int, int]:
    """Extract trip duration from raw query text.

    Handles additional patterns beyond infer_duration:
    - "nua ngay", "trong ngay", "1 buoi" → (1, 0)
    - "2n1d", "3n2d" compact patterns
    - Falls back to infer_duration for standard patterns
    - Default: (2, 1)
    """
    q = normalize_fn(query)

    # Half-day patterns
    if any(token in q for token in ["nua ngay", "nửa ngày", "trong ngay", "1 buoi", "1 buổi"]):
        return 1, 0

    # Compact: "2n1d", "3n2d"
    compact = re.search(r"\b(\d{1,2})\s*n\s*(\d{1,2})\s*d\b", q)
    if compact:
        days = int(compact.group(1))
        nights = int(compact.group(2))
        return max(1, min(days, 14)), max(0, min(nights, 14))

    # Delegate standard patterns
    result = infer_duration(q)
    if result != (0, 0):
        days, nights = result
        return max(1, min(days, 14)), max(0, min(nights, 14))

    # Special case: "2n1d" without regex boundary
    if "2n1d" in q:
        return 2, 1

    return 2, 1
