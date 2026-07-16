from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from graph_rag.utils.text import normalize_text


def _build_broad_location_norms() -> set:
    """Build BROAD_LOCATION_NORMS dynamically from RegionRegistry."""
    from graph_rag.config.region_registry import region_registry
    norms = set()
    for pid in region_registry.get_all_province_ids():
        norms.update(region_registry.get_keywords(pid))
        norms.update(a.lower() for a in region_registry.get_aliases(pid))
    return norms


class AdminRegionMappingService:
    """Administrative-name resolver for old/new Vietnamese place names."""

    # Province/city-level broad locations — built dynamically from RegionRegistry.
    # Class-level lazy init to avoid import-time side effects.
    _BROAD_LOCATION_NORMS = None

    @classmethod
    def _get_broad_location_norms(cls) -> set:
        if cls._BROAD_LOCATION_NORMS is None:
            cls._BROAD_LOCATION_NORMS = _build_broad_location_norms()
        return cls._BROAD_LOCATION_NORMS

    @property
    def BROAD_LOCATION_NORMS(self) -> set:
        return self._get_broad_location_norms()

    DEFAULT_MAPPING_PATH = (
        Path(__file__).resolve().parents[3]
        / "graph_rag"
        / "data"
        / "data_location.json"
    )

    BINH_DINH_OLD_DISTRICT_ALIASES = {
        "Quy Nhơn": ["TP Quy Nhơn", "Thành phố Quy Nhơn", "Quy Nhon", "Qui Nhơn"],
        "An Nhơn": ["Thị xã An Nhơn", "TX An Nhơn", "An Nhon"],
        "Hoài Nhơn": ["Thị xã Hoài Nhơn", "TX Hoài Nhơn", "Hoai Nhon", "Bồng Sơn", "Bong Son"],
        "Tây Sơn": ["Huyện Tây Sơn", "Tay Son"],
        "Tuy Phước": ["Huyện Tuy Phước", "Tuy Phuoc"],
        "Phù Cát": ["Huyện Phù Cát", "Phu Cat"],
        "Phù Mỹ": ["Huyện Phù Mỹ", "Phu My"],
        "Vân Canh": ["Huyện Vân Canh", "Van Canh"],
        "Vĩnh Thạnh": ["Huyện Vĩnh Thạnh", "Vinh Thanh"],
        "An Lão": ["Huyện An Lão", "An Lao"],
        "Hoài Ân": ["Huyện Hoài Ân", "Hoai An"],
    }

    GIA_LAI_CORE_ALIASES = {
        "Pleiku": ["TP Pleiku", "Thành phố Pleiku", "Plei Ku"],
        "An Khê": ["Thị xã An Khê", "TX An Khê", "An Khe"],
        "Ayun Pa": ["Thị xã Ayun Pa"],
        "Chư Prông": ["Chu Prong", "Chư Prong"],
        "Chư Sê": ["Chu Se", "Chư Se"],
        "Chư Pưh": ["Chu Puh", "Chư Pưh"],
        "Chư Păh": ["Chu Pah", "Chư Păh"],
        "Ia Grai": ["Iagrai"],
        "Đức Cơ": ["Duc Co"],
        "Đak Đoa": ["Đắk Đoa", "Dak Doa"],
        "Mang Yang": [],
        "Kbang": ["K'Bang"],
        "Kông Chro": ["Kong Chro"],
        "Phú Thiện": ["Phu Thien"],
        "Krông Pa": ["Krong Pa"],
        "Gia Lai cũ": ["Gia Lai cu", "tỉnh Gia Lai cũ", "tinh Gia Lai cu"],
    }

    _ADMIN_REGION_TO_FOCUS = {
        "binh_dinh_old": "coastal_quy_nhon",
        "gia_lai_core": "inland_gia_lai",
        "gia_lai_new": "gia_lai_new",
    }

    @classmethod
    def resolve_region_focus(cls, admin_region_focus: str) -> str:
        """Map admin_region_focus to region_focus for query understanding."""
        return cls._ADMIN_REGION_TO_FOCUS.get(admin_region_focus, "all")

    DISPLAY_BY_REGION = {
        "binh_dinh_old": "Bình Định cũ, nay thuộc tỉnh Gia Lai mới",
        "gia_lai_core": "Gia Lai (Tây Nguyên)",
        "gia_lai_new": "Gia Lai (bao gồm cả khu vực Bình Định/Quy Nhơn)",
    }

    # Khi user query province name, search cả data vùng đã sáp nhập
    # Mapping: region_focus -> list of region_groups to search
    # NOTE: Step 1 maps gia_lai_core → inland_gia_lai as region_focus
    MERGED_REGION_SEARCH = {
        "binh_dinh_old": ["binh_dinh_legacy", "gia_lai_core"],
        "coastal_quy_nhon": ["binh_dinh_legacy", "gia_lai_core"],
        "gia_lai_core": ["gia_lai_core"],
        "gia_lai_new": ["gia_lai_core", "binh_dinh_legacy"],
        "inland_gia_lai": ["gia_lai_core"],
    }

    def __init__(self, mapping_path: str | Path | None = None):
        self.mapping_path = Path(mapping_path) if mapping_path else self.DEFAULT_MAPPING_PATH
        self.lookup: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _region_for_name(self, name: str) -> str:
        normalized = normalize_text(name, strip_punct=True)
        if not normalized:
            return ""

        # Check legacy district aliases (backward compat)
        for canonical, aliases in self.BINH_DINH_OLD_DISTRICT_ALIASES.items():
            names = [canonical, *aliases]
            if any(normalize_text(alias, strip_punct=True) in normalized for alias in names):
                return "binh_dinh_old"

        for canonical, aliases in self.GIA_LAI_CORE_ALIASES.items():
            names = [canonical, *aliases]
            if any(normalize_text(alias, strip_punct=True) in normalized for alias in names):
                return "gia_lai_core"

        # Dynamic: resolve via RegionRegistry
        from graph_rag.config.region_registry import region_registry
        pid = region_registry.get_province_by_alias(name)
        if pid:
            return pid
        matches = region_registry.get_province_by_keyword(name)
        if matches:
            return matches[0]

        # Legacy fallback
        if "binh dinh" in normalized or "quy nhon" in normalized:
            return "binh_dinh_old"
        if "gia lai" in normalized or "pleiku" in normalized:
            return "gia_lai_core"
        return ""

    def _add_lookup(self, alias: str, payload: Dict[str, Any]) -> None:
        key = normalize_text(alias, strip_punct=True)
        if key:
            self.lookup[key] = payload

    def _load(self) -> None:
        if self.mapping_path.exists():
            try:
                rows = json.loads(self.mapping_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                rows = []
            for row in rows if isinstance(rows, list) else []:
                self._load_mapping_row(row)

        # Dynamic: register all provinces from RegionRegistry
        from graph_rag.config.region_registry import region_registry
        for pid in region_registry.get_all_province_ids():
            p = region_registry.get_province(pid)
            if not p:
                continue
            display = p.get("display_name", pid)
            aliases = list(p.get("aliases", []))
            region_group = p.get("region_group", "")
            region_focus = p.get("region_focus", "")
            admin_status = p.get("admin_status", "current")

            target_region_focus = pid
            if pid == "binh_dinh":
                target_region_focus = 'binh_dinh_old'
            elif pid == "gia_lai":
                target_region_focus = "gia_lai_new"
            payload = {
                "matched_alias": display,
                "old_unit": display,
                "new_unit": "",
                "old_province": display,
                "legacy_province": display if admin_status == "merged" else "",
                # gia lai new = bd + gl
                "new_province": "Gia Lai" if pid in ("binh_dinh", "gia_lai") else display,
                "current_province": "Gia Lai" if pid in ("binh_dinh", "gia_lai") else display,
                "region_focus": target_region_focus,  # use province_id as region_focus
                "region_group": region_group,
                "display_region": self.DISPLAY_BY_REGION.get(target_region_focus, display),
                "source": "region_registry",
                "admin_level": "province",
                "admin_status": admin_status,
            }
            for alias in [display, *aliases]:
                self._add_lookup(alias, payload)
            # Also register keywords
            for kw in p.get("keywords", []):
                self._add_lookup(kw, payload)

        # Legacy: keep district aliases for backward compat
        for canonical, aliases in self.BINH_DINH_OLD_DISTRICT_ALIASES.items():
            self._add_region_aliases(canonical, "binh_dinh_old", aliases)
        for canonical, aliases in self.GIA_LAI_CORE_ALIASES.items():
            self._add_region_aliases(canonical, "gia_lai_core", aliases)

    def _load_mapping_row(self, row: Dict[str, Any]) -> None:
        new_unit = str(row.get("new_unit") or "").strip()
        legacy_district = str(row.get("legacy_district") or "").strip()
        legacy_province = str(row.get("legacy_province") or "").strip()
        current_province = str(row.get("current_province") or "").strip()
        region_group = str(row.get("region_group") or "").strip()
        region_focus = self._region_focus_from_group(region_group) or self._region_for_name(
            " ".join([new_unit, legacy_district, legacy_province])
        )
        if not region_focus:
            return

        base_payload = {
            "new_unit": new_unit,
            "legacy_district": legacy_district,
            "old_province": legacy_province or ("Bình Định" if region_focus == "binh_dinh_old" else "Gia Lai"),
            "legacy_province": legacy_province,
            "new_province": current_province or "Gia Lai",
            "current_province": current_province or "Gia Lai",
            "region_focus": region_focus,
            "region_group": region_group or self._region_group_from_focus(region_focus),
            "display_region": self.DISPLAY_BY_REGION.get(region_focus, ""),
            "source": str(self.mapping_path),
        }

        if new_unit:
            self._add_lookup(new_unit, {**base_payload, "matched_alias": new_unit, "old_unit": ""})

        for old_unit in row.get("old_units") or []:
            old_unit = str(old_unit or "").strip()
            if not old_unit:
                continue
            self._add_lookup(old_unit, {**base_payload, "matched_alias": old_unit, "old_unit": old_unit})

    def _add_region_aliases(self, canonical: str, region_focus: str, aliases: List[str]) -> None:
        region_group = self._region_group_from_focus(region_focus)
        payload = {
            "matched_alias": canonical,
            "old_unit": canonical,
            "new_unit": "",
            "old_province": "Bình Định" if region_focus == "binh_dinh_old" else "Gia Lai",
            "legacy_province": "Bình Định" if region_focus == "binh_dinh_old" else "Gia Lai",
            "new_province": "Gia Lai",
            "current_province": "Gia Lai",
            "region_focus": region_focus,
            "region_group": region_group,
            "display_region": self.DISPLAY_BY_REGION.get(region_focus, ""),
            "source": "built_in_alias",
        }
        for alias in [canonical, *aliases]:
            self._add_lookup(alias, payload)

    def resolve(self, text: str, entities: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        parts = [str(text or "")]
        for entity in entities or []:
            if isinstance(entity, dict):
                parts.append(str(entity.get("name") or ""))
        normalized = normalize_text(" ".join(parts), strip_punct=True)
        if not normalized:
            return {}

        matches: List[Dict[str, Any]] = []
        for key, payload in self.lookup.items():
            if re.search(rf"(^|\s){re.escape(key)}($|\s)", normalized):
                matches.append({**payload, "match_key": key})

        if not matches:
            return {}

        # Dynamic multi-province detection: check if matches span multiple provinces
        matched_provinces = set()
        for m in matches:
            prov = m.get("current_province") or m.get("new_province") or ""
            if prov:
                matched_provinces.add(prov)

        if len(matched_provinces) > 1:
            # Multi-region: pick the best match (longest key)
            matches.sort(key=lambda item: len(str(item.get("match_key") or "")), reverse=True)
            best = matches[0]
            best = {**best, "match_key": " + ".join(sorted(matched_provinces))}
        else:
            matches.sort(key=lambda item: len(str(item.get("match_key") or "")), reverse=True)
            best = matches[0]
        region_focus = best.get("region_focus") or ""
        return {
            "matched_alias": best.get("matched_alias") or best.get("old_unit") or "",
            "old_unit": best.get("old_unit") or "",
            "new_unit": best.get("new_unit") or "",
            "old_province": best.get("old_province") or "",
            "new_province": best.get("new_province") or "",
            "legacy_district": best.get("legacy_district") or "",
            "legacy_province": best.get("legacy_province") or best.get("old_province") or "",
            "current_province": best.get("current_province") or best.get("new_province") or "",
            "region_focus": region_focus,
            "region_group": best.get("region_group") or "",
            "display_region": best.get("display_region") or "",
            "source": best.get("source") or "",
            "admin_level": self._infer_admin_level_from_focus(region_focus),
            "admin_status": self._infer_admin_status_from_focus(region_focus),
        }

    def get_merged_province_names(self, region_focus: str) -> list:
        """Return province names to search when region_focus is a merged province.

        E.g., 'gia_lai_core' → ['Gia Lai', 'Bình Định'] because the new Gia Lai
        province includes old Bình Định per the 2025 merger decree.
        """
        from graph_rag.config.region_registry import region_registry
        provinces = []
        # Check if region_focus is a province_id
        p = region_registry.get_province(region_focus)
        if p:
            provinces.append(p.get("display_name", region_focus))
            # Add merged provinces
            for merged_pid in region_registry.get_merged_provinces(region_focus):
                mp = region_registry.get_province(merged_pid)
                if mp:
                    provinces.append(mp.get("display_name", merged_pid))
            return provinces

        # Legacy fallback
        merged_groups = self.MERGED_REGION_SEARCH.get(region_focus, [])
        if region_focus in ("gia_lai_core", "gia_lai_new") or "gia_lai_core" in merged_groups:
            provinces.append("Gia Lai")
        if region_focus == "binh_dinh_old" or "binh_dinh_legacy" in merged_groups:
            provinces.append("Bình Định")
        if not provinces:
            for payload in self.lookup.values():
                prov = payload.get("old_province") or payload.get("new_province") or ""
                if prov and prov not in provinces:
                    provinces.append(prov)
                break
        return provinces

    def get_merged_region_groups_for_province(self, province_name: str) -> Optional[List[str]]:
        """Return merged region_groups for a province name.

        E.g., 'Bình Định' → ['binh_dinh_legacy', 'gia_lai_core']
        Returns None if no merged mapping exists.
        """
        from graph_rag.config.region_registry import region_registry
        # Try RegionRegistry first
        pid = region_registry.get_province_by_alias(province_name)
        if not pid:
            matches = region_registry.get_province_by_keyword(province_name)
            pid = matches[0] if matches else None
        if pid:
            p = region_registry.get_province(pid)
            if not p:
                return None
            # If this province was merged, follow the chain to the target
            target_pid = pid
            if p.get("admin_status") == "merged":
                merge_target = region_registry.get_merge_target(pid)
                if merge_target:
                    target_pid = merge_target
            # Collect region_groups from target + its merged provinces
            target = region_registry.get_province(target_pid)
            if target:
                groups = [target.get("region_group", "")]
                for merged_pid in region_registry.get_merged_provinces(target_pid):
                    mp = region_registry.get_province(merged_pid)
                    if mp:
                        mg = mp.get("region_group", "")
                        if mg and mg not in groups:
                            groups.append(mg)
                if len(groups) > 1:
                    return groups
            return None

        # Legacy fallback
        resolved = self.resolve(province_name)
        if not resolved:
            return None
        region_focus = resolved.get("region_focus") or ""
        merged = self.MERGED_REGION_SEARCH.get(region_focus)
        return merged if merged and len(merged) > 1 else None

    def is_broad_location(self, name: str) -> bool:
        """Check if a location name is a broad administrative area (province/city level).

        Checks both the hardcoded BROAD_LOCATION_NORMS set and the mapping service
        lookup (for district-level names like An Khê, Chư Sê, etc.).
        """
        from graph_rag.utils.text import normalize_text
        name_norm = normalize_text(name, strip_punct=True)
        if name_norm in self.BROAD_LOCATION_NORMS:
            return True
        # Check mapping service for district-level broad locations
        resolved = self.resolve(name)
        if resolved and resolved.get("matched_alias"):
            admin_level = resolved.get("admin_level", "")
            if admin_level in ("province", "area"):
                return True
        return False

    @staticmethod
    def _region_focus_from_group(region_group: str) -> str:
        # Dynamic: find province with this region_group
        from graph_rag.config.region_registry import region_registry
        for pid in region_registry.get_all_province_ids():
            p = region_registry.get_province(pid)
            if p and p.get("region_group") == region_group:
                return pid
        # Legacy fallback
        if region_group == "binh_dinh_legacy":
            return "binh_dinh_old"
        if region_group == "gia_lai_core":
            return "gia_lai_core"
        if region_group == "gia_lai_new":
            return "gia_lai_new"
        return ""

    @staticmethod
    def _infer_admin_level_from_focus(region_focus: str) -> str:
        """Suy ra admin_level từ region_focus."""
        from graph_rag.config.region_registry import region_registry
        p = region_registry.get_province(region_focus)
        if p:
            return "province"
        # Legacy fallback
        if region_focus in ("gia_lai_new", "gia_lai_core", "binh_dinh_old", "inland_gia_lai", "coastal_quy_nhon"):
            return "province"
        return "ward"

    @staticmethod
    def _infer_admin_status_from_focus(region_focus: str) -> str:
        """Suy ra admin_status từ region_focus."""
        from graph_rag.config.region_registry import region_registry
        p = region_registry.get_province(region_focus)
        if p:
            return p.get("admin_status", "current")
        # Legacy fallback
        if region_focus in ("binh_dinh_old", "coastal_quy_nhon"):
            return "merged"
        return "current"

    @staticmethod
    def _region_group_from_focus(region_focus: str) -> str:
        from graph_rag.config.region_registry import region_registry
        p = region_registry.get_province(region_focus)
        if p:
            return p.get("region_group", "")
        # Legacy fallback
        if region_focus in ("binh_dinh_old", "coastal_quy_nhon"):
            return "binh_dinh_legacy"
        if region_focus in ("gia_lai_core", "inland_gia_lai"):
            return "gia_lai_core"
        if region_focus == "gia_lai_new":
            return "gia_lai_new"
        return ""
