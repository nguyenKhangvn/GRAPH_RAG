"""Region Registry — Single source of truth for all province/region data.

Loads province definitions from region_registry.json and provides
query methods for province resolution, keyword matching, merge mapping,
and region group lookups.

Usage:
    from graph_rag.config.region_registry import region_registry

    # Resolve province from text
    provinces = region_registry.get_province_by_keyword("Quy Nhon")
    # → ["binh_dinh"]

    # Get province info
    p = region_registry.get_province("binh_dinh")
    # → {"display_name": "Bình Định", "region_focus": "coastal", ...}

    # Check merge
    region_registry.get_merge_target("binh_dinh")
    # → "gia_lai"
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_REGISTRY_FILE = "region_registry.json"


def _strip_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _normalize(text: str) -> str:
    """Normalize text for matching: lowercase, strip diacritics, collapse spaces."""
    return " ".join(_strip_diacritics(text).lower().split())


class RegionRegistry:
    """Province/region registry backed by region_registry.json."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            path = _CONFIG_DIR / _REGISTRY_FILE
        with open(path, "r", encoding="utf-8") as f:
            self._data = json.load(f)
        self._provinces: dict[str, dict] = self._data.get("provinces", {})
        self._groups: dict[str, dict] = self._data.get("region_groups", {})
        self._merges: dict[str, dict] = self._data.get("merge_mappings", {})
        # Pre-build lookup indices
        self._keyword_index: dict[str, list[str]] = self._build_keyword_index()
        self._alias_index: dict[str, str] = self._build_alias_index()

    # ── Province Lookups ─────────────────────────────────────────────

    def get_province(self, province_id: str) -> dict | None:
        """Return province data by ID (e.g. 'gia_lai')."""
        return self._provinces.get(province_id)

    def get_province_display_name(self, province_id: str) -> str:
        p = self.get_province(province_id)
        return p["display_name"] if p else province_id

    def get_province_by_keyword(self, text: str) -> list[str]:
        """Return list of province_ids whose keywords appear in text.

        Matching is diacritic-insensitive. Results are ordered by
        longest keyword match first (most specific wins).
        """
        text_norm = _normalize(text)
        matches: list[tuple[int, str]] = []
        for pid, keywords in self._keyword_index.items():
            for kw in keywords:
                if kw in text_norm:
                    matches.append((len(kw), pid))
                    break
        # Sort by longest keyword match (most specific first)
        matches.sort(key=lambda x: x[0], reverse=True)
        return [pid for _, pid in matches]

    def get_province_by_alias(self, name: str) -> str | None:
        """Resolve a province name/alias to province_id."""
        return self._alias_index.get(_normalize(name))

    # ── Region Focus ─────────────────────────────────────────────────

    def get_region_focus(self, province_id: str) -> str:
        """Return region_focus for a province (e.g. 'coastal', 'highland', 'urban')."""
        p = self.get_province(province_id)
        return p.get("region_focus", "unknown") if p else "unknown"

    def get_all_region_focuses(self) -> list[str]:
        """Return all unique region_focus values."""
        return list({p.get("region_focus", "unknown") for p in self._provinces.values()})

    # ── Region Groups ────────────────────────────────────────────────

    def get_region_group(self, province_id: str) -> str:
        """Return region_group for a province (e.g. 'tay_nguyen')."""
        p = self.get_province(province_id)
        return p.get("region_group", "unknown") if p else "unknown"

    def get_provinces_in_group(self, group_id: str) -> list[str]:
        """Return list of province_ids in a region group."""
        g = self._groups.get(group_id)
        return g.get("provinces", []) if g else []

    def get_group_display(self, group_id: str) -> str:
        g = self._groups.get(group_id)
        return g.get("display", group_id) if g else group_id

    def get_all_group_ids(self) -> list[str]:
        return list(self._groups.keys())

    # ── Merge Mappings ───────────────────────────────────────────────

    def get_merge_target(self, province_id: str) -> str | None:
        """If province was merged into another, return target province_id.

        Checks both merge_mappings dict and the province's merged_into field.
        """
        m = self._merges.get(province_id)
        if m:
            return m.get("into")
        # Fallback: check province's own merged_into field
        p = self._provinces.get(province_id)
        return p.get("merged_into") if p else None

    def get_merged_provinces(self, province_id: str) -> list[str]:
        """Return list of province_ids merged INTO this province."""
        return [pid for pid, m in self._merges.items() if m.get("into") == province_id]

    def is_merged(self, province_id: str) -> bool:
        return province_id in self._merges

    def get_merge_year(self, province_id: str) -> int | None:
        m = self._merges.get(province_id)
        return m.get("year") if m else None

    # ── Keyword/Alias Helpers ────────────────────────────────────────

    def get_keywords(self, province_id: str) -> list[str]:
        p = self.get_province(province_id)
        return p.get("keywords", []) if p else []

    def get_aliases(self, province_id: str) -> list[str]:
        p = self.get_province(province_id)
        return p.get("aliases", []) if p else []

    def get_all_keywords_flat(self) -> dict[str, list[str]]:
        """Return {province_id: [normalized_keywords]} for all provinces."""
        return dict(self._keyword_index)

    def get_all_province_ids(self) -> list[str]:
        return list(self._provinces.keys())

    def get_all_display_names(self) -> dict[str, str]:
        """Return {province_id: display_name} for all provinces."""
        return {pid: p["display_name"] for pid, p in self._provinces.items()}

    def get_bounding_box(self, province_id: str) -> dict | None:
        p = self.get_province(province_id)
        return p.get("bounding_box") if p else None

    # ── Internal ─────────────────────────────────────────────────────

    def _build_keyword_index(self) -> dict[str, list[str]]:
        """Build {province_id: [normalized_keywords]} index."""
        index = {}
        for pid, p in self._provinces.items():
            keywords = p.get("keywords", [])
            index[pid] = [_normalize(kw) for kw in keywords]
        return index

    def _build_alias_index(self) -> dict[str, str]:
        """Build {normalized_alias: province_id} index."""
        index = {}
        for pid, p in self._provinces.items():
            for alias in p.get("aliases", []):
                index[_normalize(alias)] = pid
            # Also index the province_id itself
            index[_normalize(pid.replace("_", " "))] = pid
        return index


# ── Module-level singleton ───────────────────────────────────────────
region_registry = RegionRegistry()
