"""Semantic feature registry for constraint-driven tour planning.

Each feature defines:
- ``keywords``: normalized (no diacritics) terms used for keyword matching.
- ``labels``:   Neo4j node labels that are *expected* to carry this feature.
- ``weight``:   scoring weight applied in PolicyRanker when a candidate matches.
- ``is_hard``:  if True, the feature is a hard constraint (RouteGate will reject
                routes missing it).

Features: coastal, sunset, island, walking, low_mobility, negative, family_friendly, budget.
Extend by adding entries here — no code changes needed downstream.
"""

from __future__ import annotations

from typing import Any, Dict


FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "coastal": {
        "keywords": [
            "bien", "bai bien", "ky co", "eo gio", "cu lao xanh",
            "hon kho", "hon seo", "trung luong", "quy nhon", "nhon ly",
            "bien dao",
        ],
        "labels": ["Tour", "TouristAttraction"],
        "weight": 1.0,
        "is_hard": True,
    },
    "sunset": {
        "keywords": [
            "hoang hon", "ngam hoang hon", "sunset",
            "eo gio", "ky co",
        ],
        "labels": ["Tour", "TouristAttraction"],
        "weight": 1.0,
        "is_hard": True,
    },
    "island": {
        "keywords": [
            "dao", "cu lao", "cu lao xanh", "hon kho", "ky co",
        ],
        "labels": ["Tour", "TouristAttraction"],
        "weight": 1.0,
        "is_hard": True,
    },
    "walking": {
        "keywords": [
            "di bo", "walking",
        ],
        "labels": [],
        "weight": 1.0,
        "is_hard": False,
    },
    "low_mobility": {
        "keywords": [
            "nguoi gia", "tre em", "gia dinh co tre",
            "phu hop tre em", "nguoi cao tuoi", "gia dinh co nguoi gia",
        ],
        "labels": [],
        "weight": 1.0,
        "is_hard": False,
    },
    "negative": {
        "keywords": [
            "khong muon", "khong thich", "tranh",
            "khong can", "khong can thiet",
        ],
        "labels": [],
        "weight": 0.8,
        "is_hard": False,
        "requires_activity_extraction": True,
    },
    "family_friendly": {
        "keywords": [
            "gia dinh", "tre em", "phu hop cho tre",
            "phu hop gia dinh", "cho tre em", "anh em",
        ],
        "labels": [],
        "weight": 0.8,
        "is_hard": False,
    },
    "budget": {
        "keywords": [
            "gia re", "tiet kiem", "binh dan", "re nhat",
            "re", "hieu qua chi phi", "it ton kem",
        ],
        "labels": [],
        "weight": 0.7,
        "is_hard": False,
    },
}
