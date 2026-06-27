"""Shared constants for the graph_rag package."""
from __future__ import annotations

# Entity types that should NOT be grounded to graph nodes.
# Superset from context_organizer (16 types) and graph_rag_pipeline (11 types).
NON_GROUNDABLE_ENTITY_TYPES: set[str] = {
    "duration",
    "time",
    "groupsize",
    "group_size",
    "personcount",
    "person_count",
    "people",
    "budget",
    "price",
    "number",
    "count",
    "province",
    "city",
    "district",
    "ward",
    "commune",
    "location",
}
