from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


import re


from typing import Any, Dict, List


from graph_rag.core import keywords


from graph_rag.utils.text import normalize_text


class QueryPatternMixin:
    def _strip_entity_tail_noise(self, text: str) -> str:
        cleaned = str(text or "").strip(" ,.;:!?")
        if not cleaned:
            return ""
        tail_patterns = [
            r"(?i)\s+(?:v\u00e0|va)\s+(?:c\u00e1c|cac)\b.*$",
            r"(?i)\s+(?:v\u00e0|va)\s+(?:m\u1ed1i|moi)\b.*$",
            r"(?i)\s+(?:m\u1ed1i|moi)\s+quan\s+(?:h\u1ec7|he)\b.*$",
            r"(?i)\s+(?:d\u1ef1a|dua)\s+(?:tr\u00ean|tren)\b.*$",
            r"(?i)\s+(?:theo)\s+(?:th\u00f4ng\s+tin|thong\s+tin|ng\u1eef\s+c\u1ea3nh|ngu\s+canh|context)\b.*$",
            r"(?i)\s+(?:trong)\s+(?:ng\u1eef\s+c\u1ea3nh|ngu\s+canh|context)\b.*$",
            r"(?i)\s+(?:d\u1ecbch\s+v\u1ee5|dich\s+vu|ho\u1ea1t\s+\u0111\u1ed9ng|hoat\s+dong|m\u00f3n|mon|quan\s+h\u1ec7|quan\s+he)\b.*$",
            r"(?i),\s*(?:qu\u00e1n|quan|n\u01a1i|noi|d\u1ecba\s+\u0111i\u1ec3m|dia\s+diem)\b.*$",
            r"(?i),\s*(?:kh\u00e1ch|khach|du\s+kh\u00e1ch|du\s+khach|c\u00f3\s+th\u1ec3|co\s+the)\b.*$",
            r"(?i)\s+(?:c\u00f9ng|cung|c\u1ee5ng|chung)\s*$",
        ]
        for pattern in tail_patterns:
            cleaned = re.sub(pattern, "", cleaned).strip(" ,.;:!?")
        return cleaned

    def _extract_analysis_subject_entity_hint(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw:
            return ""
        analysis_of_match = re.search(
            r"(?i)^\s*(?:phân\s+tích|phan\s+tich).+?\s+(?:của|cua)\s+((?:nhà\s+hàng|nha\s+hang|quán|quan|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi|homestay|resort|tour|khu\s+du\s+lịch|khu\s+du\s+lich)\s+.+?)(?:\s+dựa|\s+dua|,|\.|$)",
            raw,
        )
        if analysis_of_match:
            candidate = self._strip_entity_tail_noise(str(analysis_of_match.group(1) or "")).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        simple_info_match = re.search(
            r"(?i)(?:thông\s+tin\s+về|thong\s+tin\s+ve)\s+(.+?)(?:\s+(?:và|va)\s+(?:các|cac|mối|moi)|,|\.|$)",
            raw,
        )
        if simple_info_match:
            candidate = self._strip_entity_tail_noise(str(simple_info_match.group(1) or "")).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        direct_patterns = [
            r"(?i)\bcủa\s+(.+?)\s+dựa\s+trên\b",
            r"(?i)\bcua\s+(.+?)\s+dua\s+tren\b",
            r"(?i)\bvề\s+(.+?)\s+dựa\s+trên\b",
            r"(?i)\bve\s+(.+?)\s+dua\s+tren\b",
            r"(?i)\bcủa\s+(.+?)\.\s*(?:Giải thích|Giai thich|Hãy|Hay)\b",
            r"(?i)\bcua\s+(.+?)\.\s*(?:Giai thich|Hay)\b",
        ]
        for pattern in direct_patterns:
            direct_match = re.search(pattern, raw)
            if direct_match:
                candidate = self._strip_entity_tail_noise(str(direct_match.group(1) or "")).strip(" ,.;:!?")
                if candidate and not self._is_broad_location_anchor(candidate):
                    return candidate
        info_match = re.search(
            r"(?i)(?:dựa\s+trên\s+thông\s+tin\s+về|dua\s+tren\s+thong\s+tin\s+ve|thông\s+tin\s+về|thong\s+tin\s+ve)\s+(.+?)(?:,|\s+(?:quán|quan|nhà\s+hàng|nha\s+hang|nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san)\s+(?:này|nay)\b|$)",
            raw,
        )
        if info_match:
            candidate = self._strip_entity_tail_noise(str(info_match.group(1) or ""))
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        if not any(token in normalize_text(raw, strip_punct=True) for token in ["phan tich", "chien luoc", "moi quan he", "boi canh"]):
            return ""

        normalized = normalize_text(raw, strip_punct=True)
        normalized_match = re.search(
            r"\bve\s+(.+?)\s+(?:va\s+(?:cac\s+)?moi\s+quan\s+he|dua\s+tren|hay\s+phan\s+tich|phan\s+tich|giai\s+thich)\b",
            normalized,
        )
        if normalized_match:
            normalized_candidate = normalized_match.group(1).strip(" ,.;:!?")
            stop_words = {
                "thong tin",
                "nha nghi nay",
                "khach san nay",
                "dia diem nay",
            }
            normalized_candidate = re.sub(r"^(?:thong tin ve|thong tin)\s+", "", normalized_candidate).strip(" ,.;:!?")
            if normalized_candidate and normalized_candidate not in stop_words and not self._is_broad_location_anchor(normalized_candidate):
                return normalized_candidate

        patterns = [
            r"(?i)(?:của|cua)\s+((?:quán|quan|nhà\s+hàng|nha\s+hang|nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort)\s+.+?)(?:\s+dựa|\s+dua|\s+và\s+mối|\s+va\s+moi|\s+mối\s+quan\s+hệ|\s+moi\s+quan\s+he|,|\.|$)",
            r"(?i)(?:về|ve)\s+((?:quán|quan|nhà\s+hàng|nha\s+hang|nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort)\s+.+?)(?:\s+dựa|\s+dua|\s+và\s+mối|\s+va\s+moi|\s+mối\s+quan\s+hệ|\s+moi\s+quan\s+he|,|\.|$)",
            r"(?i)(?:của|cua)\s+((?:quán|quan|nhà\s+hàng|nha\s+hang|nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort)\s+.+?)(?:\s+dựa|\s+dua|,|\.|$)",
            r"(?i)(?:về|ve)\s+((?:quán|quan|nhà\s+hàng|nha\s+hang|nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort)\s+.+?)(?:\s+dựa|\s+dua|,|\.|$)",
            r"(?i)(?:của|cua)\s+(.+?)(?:\s+như|\s+nhu|\.|\s+dựa|\s+dua|\s+và\s+mối|\s+va\s+moi|\s+mối\s+quan\s+hệ|\s+moi\s+quan\s+he|,|$)",
            r"(?i)(?:về|ve)\s+(.+?)(?:\s+như|\s+nhu|\.|\s+dựa|\s+dua|\s+và\s+mối|\s+va\s+moi|\s+mối\s+quan\s+hệ|\s+moi\s+quan\s+he|,|$)",
            r"(?i)(?:của|cua)\s+(.+?)(?:\.|\s+dựa|\s+dua|\s+và\s+mối|\s+va\s+moi|\s+mối\s+quan\s+hệ|\s+moi\s+quan\s+he|,|$)",
            r"(?i)(?:về|ve)\s+(.+?)(?:\.|\s+dựa|\s+dua|\s+và\s+mối|\s+va\s+moi|\s+mối\s+quan\s+hệ|\s+moi\s+quan\s+he|,|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                candidate = self._strip_entity_tail_noise(str(match.group(1) or ""))
                if candidate and not self._is_broad_location_anchor(candidate):
                    return candidate
        return ""

    def _extract_address_lookup_entity_hint(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw:
            return ""
        patterns = [
            r"(?i)^(?:nhà\s+hàng|nha\s+hang|quán|quan|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi)\s+(.+?)\s+(?:được\s+đặt|duoc\s+dat|đặt|dat)\s+(?:tại|tai|ở|o)\s*_+",
            r"(?i)^(?:nhà\s+hàng|nha\s+hang|quán|quan|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi)\s+(.+?)\s+(?:nằm|nam|tọa\s+lạc|toa\s+lac)\s+(?:tại|tai|ở|o)\s*_+",
            r"(?i)^(.+?)\s+(?:được\s+đặt|duoc\s+dat|đặt|dat|nằm|nam|tọa\s+lạc|toa\s+lac)\s+(?:tại|tai|ở|o)\s*_+",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                return self._canonicalize_entity_name(match.group(1))
        return ""

    def _extract_fill_blank_subject_entity_hint(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw or "___" not in raw:
            return ""
        tour_match = re.search(
            r"(?i)^\s*(tour\s+.+?)\s+(?:có|co)\s+(?:điểm|diem)\s+(?:xuất\s+phát|xuat\s+phat)\b",
            raw,
        )
        if tour_match:
            candidate = str(tour_match.group(1) or "").strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        match = re.search(
            r"(?i)^(.+?)\s+(?:n\u1eb1m\s+c\u00e1ch|nam\s+cach|\u0111\u01b0\u1ee3c\s+m\u1ec7nh\s+danh|duoc\s+menh\s+danh|thu\u1ed9c|thuoc|là|la)\b",
            raw,
        )
        if not match:
            return ""
        candidate = self._canonicalize_entity_name(match.group(1)).strip(" ,.;:!?")
        candidate = self._strip_entity_tail_noise(candidate).strip(" ,.;:!?")
        if candidate and not self._is_broad_location_anchor(candidate):
            return candidate
        return ""

    def _extract_statement_subject_entity_hint(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw:
            return ""

        # Pattern 0: "X tên là Y" / "X có tên là Y" / "X được gọi là Y" / "X mang tên Y"
        # The actual entity name comes AFTER "tên là", not before it.
        # e.g. "quán ăn tên là Bánh xèo tôm nhảy Gia Vỹ địa chỉ ở đâu"
        #   → entity = "Bánh xèo tôm nhảy Gia Vỹ"
        name_patterns = [
            r"(?i)(?:tên|ten)\s+(?:là|la)\s+(.+?)(?:\s+(?:địa\s+chỉ|dia\s+chi|ở|o|nằm|nam|tọa|toa|có|co|được|duoc|và|va|cũng|cung|thì|thi)|[?.!,;:]|$)",
            r"(?i)(?:có|co)\s+(?:tên|ten)\s+(?:là|la)\s+(.+?)(?:\s+(?:địa\s+chỉ|dia\s+chi|ở|o|nằm|nam|tọa|toa|có|co|được|duoc|và|va|cũng|cung|thì|thi)|[?.!,;:]|$)",
            r"(?i)(?:được|duoc)\s+(?:gọi|goi)\s+(?:là|la)\s+(.+?)(?:\s+(?:địa\s+chỉ|dia\s+chi|ở|o|nằm|nam|tọa|toa|có|co|được|duoc|và|va|cũng|cung|thì|thi)|[?.!,;:]|$)",
            r"(?i)(?:mang|ten)\s+(?:tên|ten)\s+(.+?)(?:\s+(?:địa\s+chỉ|dia\s+chi|ở|o|nằm|nam|tọa|toa|có|co|được|duoc|và|va|cũng|cung|thì|thi)|[?.!,;:]|$)",
        ]
        for pat in name_patterns:
            m = re.search(pat, raw)
            if m:
                candidate = self._strip_entity_tail_noise(str(m.group(1) or "")).strip(" ,.;:!?")
                if candidate and len(candidate) >= 3 and not self._is_broad_location_anchor(candidate):
                    logger.info("   -> [NamePattern] Extracted entity from 'tên là' pattern: '%s'", candidate)
                    return candidate

        # Pattern 1: "X là một [type]..."
        match = re.search(
            r"(?i)^(.+?)\s+(?:là|la)\s+(?:một\s+)?(?:sự\s+kiện|su\s+kien|địa\s+điểm|dia\s+diem|tour|nhà\s+hàng|nha\s+hang|quán|quan|danh\s+lam|di\s+tich|khu\s+du|bảo\s+tàng|bao\s+tang|chùa|chua|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi|homestay|resort)\b",
            raw,
        )
        if match:
            candidate = self._strip_entity_tail_noise(str(match.group(1) or "")).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        # Pattern 2: "Nhà hàng/Quán/Khách sạn X, ..." or "X, nằm ở/ở ..."
        # Match entity names that start with known prefixes
        prefix_match = re.search(
            r"(?i)^((?:nhà\s+hàng|nha\s+hang|quán|quan|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi|khu\s+du\s+lịch|khu\s+du\s+lich|bảo\s+tàng|bao\s+tang|chùa|chua|sân\s+van|cong\s+vien|quảng\s+trường|quang\s+truong)\s+[^,]+?)(?:,|\s+(?:nằm|nam|tọa|toa|có|co|được|duoc|phục\s+vụ|phuc\s+vu))\b",
            raw,
        )
        if prefix_match:
            candidate = self._strip_entity_tail_noise(str(prefix_match.group(1) or "")).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        # Pattern 3: "X là một [any noun phrase]" - broader match for entity names
        broad_match = re.search(
            r"(?i)^([A-ZÀ-Ỹ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+)*(?:\s+(?:Gia Lai|Pleiku|Quy Nhơn|Bình Định))?)\s+(?:là|la)\s+(?:một\s+)?",
            raw,
        )
        if broad_match:
            candidate = self._strip_entity_tail_noise(str(broad_match.group(1) or "")).strip(" ,.;:!?")
            if candidate and len(candidate.split()) >= 2 and not self._is_broad_location_anchor(candidate):
                return candidate

        return ""

    def _extract_proximity_anchor_hint(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw:
            return ""
        # Skip proximity extraction for "category noun + như X hay Y" patterns.
        # e.g. "Các điểm tham quan như Biển Hồ T'Nưng hay Biển Hồ Chè"
        # Here "như" lists actual targets, not a proximity anchor.
        _CATEGORY_NOUNS_RE = r"(?:điểm\s+tham\s+quan|địa\s+điểm|điểm\s+du\s+lịch|nơi\s+du\s+lịch|thành\s+phố|tỉnh|khu\s+vực|vùng|hòn\s+đảo|đảo|biển|hồ|thác|núi|chùa|đền|làng|khách\s+sạn|nhà\s+nghỉ|homestay|resort|nhà\s+hàng|quán\s+ăn|món\s+ăn|đặc\s+sản|sự\s+kiện|lễ\s+hội)"
        if re.search(rf"(?i){_CATEGORY_NOUNS_RE}\s+như\s+.+\s+(?:hay|hoặc|và)\s+", raw):
            return ""
        info_match = re.search(
            r"(?i)(?:dựa\s+trên\s+thông\s+tin\s+về|dua\s+tren\s+thong\s+tin\s+ve|thông\s+tin\s+về|thong\s+tin\s+ve)\s+(.+?)(?:,|\s+(?:quán|quan|nhà\s+hàng|nha\s+hang|nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san)\s+(?:này|nay)\b|$)",
            raw,
        )
        if info_match:
            candidate = self._strip_entity_tail_noise(str(info_match.group(1) or ""))
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        q_norm = normalize_text(raw, strip_punct=True)

        # Fill-in-the-blank: "X ... ___" → extract X as entity
        fill_match = re.search(r"^(.+?)\s+(?:nằm cách|được mệnh danh|thuộc|là)\b.*___", raw)
        if fill_match:
            candidate = self._canonicalize_entity_name(fill_match.group(1))
            candidate = self._strip_entity_tail_noise(candidate).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        # Case: "xung quanh/gần <entity> có..." — proximity word comes FIRST
        _PROXIMITY_WORDS_RE = r"(?:xung\s+quanh|quanh|lân\s+cận|lan\s+can|nằm\s+gần|nam\s+gan|ở\s+gần|o\s+gan|gần(?!\s+\d)|gan(?!\s+\d))"
        proximity_first_match = re.search(
            r"(?i)" + _PROXIMITY_WORDS_RE + r"\s+(.+?)(?:\s+(?:có|co|không|khong|thì|thi|và|va|hay|hoac|nào|nao)|[?.!,;:]|$)",
            raw,
        )
        if proximity_first_match:
            candidate = self._canonicalize_entity_name(proximity_first_match.group(1))
            candidate = self._strip_entity_tail_noise(candidate).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        # Case: "<entity> xung quanh/gần/lân cận có chỗ nào..."
        # Use negative lookahead to avoid matching "gần 25 km" (gần + number = "approximately")
        prefix_anchor_match = re.search(
            r"(?i)^(.+?)\s+"
            r"(?:xung\s+quanh|quanh|lân\s+cận|lan\s+can|nằm\s+gần|nam\s+gan|ở\s+gần|o\s+gan|gần(?!\s+\d)|gan(?!\s+\d))\b"
            r".*$",
            raw,
        )
        if prefix_anchor_match:
            raw_prefix = prefix_anchor_match.group(1).strip()
            prefix_norm = normalize_text(raw_prefix, strip_punct=True)
            # Skip if prefix is a multi-choice question phrase, not an entity
            _MULTI_CHOICE_PREFIXES = ["nhung cai nao", "cai nao duoi", "dau la", "dau la", "nhung dia diem nao", "cac dia diem nao"]
            if not any(prefix_norm.startswith(p) for p in _MULTI_CHOICE_PREFIXES):
                candidate = self._canonicalize_entity_name(raw_prefix)
                candidate = self._strip_entity_tail_noise(candidate).strip(" ,.;:!?")
                if candidate and not self._is_broad_location_anchor(candidate):
                    return candidate

        # Chỉ return "" cho câu hỏi quá rộng, không có anchor đứng trước
        if any(token in q_norm for token in ["co cho nao", "co cho nghi", "co noi nao", "co khach san nao", "co nha nghi nao"]):
            return ""
        subject_patterns = [
            r"(?i)^((?:nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort|nhà\s+hàng|nha\s+hang|quán|quan)\s+.+?)\s+(?:nằm\s+gần|nam\s+gan|ở\s+gần|o\s+gan|gần|gan)\b",
            r"(?i)((?:nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort|nhà\s+hàng|nha\s+hang|quán|quan)\s+(?:(?!(?:nằm\s+gần|nam\s+gan|ở\s+gần|o\s+gan|gần|gan|xung\s+quanh|quanh|lân\s+cận|lan\s+can)\b).)+?)(?=\s+(?:nằm\s+gần|nam\s+gan|ở\s+gần|o\s+gan|gần|gan|xung\s+quanh|quanh|lân\s+cận|lan\s+can)\b)",
            r"(?i)(?:nằm\s+gần|nam\s+gan|ở\s+gần|o\s+gan|gần|gan|xung\s+quanh|quanh|lân\s+cận|lan\s+can)\s+(?:(?!(?:nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort|nhà\s+hàng|nha\s+hang|quán|quan)\b).)*((?:nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort|nhà\s+hàng|nha\s+hang|quán|quan)\s+(.+?))(?:\s*(?:,|\(|\.|$|\s+(?:theo|dựa|với|và|là|có|được|nằm|tại|ở)\b))",
            r"(?i)(?:thông\s+tin\s+về|thong\s+tin\s+ve)\s+(.+?)(?:,|\s+(?:quán|quan|nơi\s+này|noi\s+nay|địa\s+điểm\s+này|dia\s+diem\s+nay)\b)",
            r"(?i)(?:về|ve)\s+(.+?)(?:,|\s+(?:quán|quan|nơi\s+này|noi\s+nay|địa\s+điểm\s+này|dia\s+diem\s+nay)\b)",
        ]
        _PROXIMITY_WORDS = r"(?:nằm\s+gần|nam\s+gan|ở\s+gần|o\s+gan|gần|gan|xung\s+quanh|quanh|lân\s+cận|lan\s+can)"
        _LODGING_PATTERN = re.compile(
            r"(?i)(nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort|nhà\s+hàng|nha\s+hang|quán|quan)\s+(.+?)(?:\s*(?:,|\.|$|\s+(?:"
            r"nằm|nam|ở|o|tại|tai|và|va|khách|khach|bạn|ban|có|co|được|duoc|giải|giai|tạo|tao|là|la|tên|ten)\b))",
            re.IGNORECASE,
        )
        # Skip proximity extraction for analysis queries — words like "lân cận"
        # appear in "đối tượng lân cận" which is not a proximity search signal.
        if any(sig in q_norm for sig in keywords.ANALYSIS_SIGNALS):
            return ""
        nearby_collection_match = re.search(
            r"(?iu)(?:thông\s+tin\s+về|thong\s+tin\s+ve)\s+(?:các|cac|những|nhung)?\s*(?:địa\s+điểm|dia\s+diem)\s+(?:gần|gan)\s+(.+?)(?:,|\.|$)",
            raw,
        )
        if nearby_collection_match:
            candidate_raw = str(nearby_collection_match.group(1) or "").strip(" ,.;:!?")
            lodging_match = _LODGING_PATTERN.search(candidate_raw)
            if lodging_match:
                candidate_raw = f"{lodging_match.group(1)} {lodging_match.group(2)}".strip()
            candidate_norm = normalize_text(candidate_raw, strip_punct=True)
            keep_prefix = candidate_norm.startswith(("nha nghi ", "khach san ", "homestay ", "resort "))
            candidate = candidate_raw if keep_prefix else self._canonicalize_entity_name(candidate_raw).strip(" ,.;:!?")
            candidate = self._strip_entity_tail_noise(candidate)
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        if self._is_proximity_query(raw):
            for pattern in subject_patterns:
                match = re.search(pattern, raw)
                if not match:
                    continue
                candidate_raw = str(match.group(1) or "").strip(" ,.;:!?")
                candidate_norm = normalize_text(candidate_raw, strip_punct=True)
                keep_lodging_prefix = candidate_norm.startswith(
                    ("nha nghi ", "khach san ", "homestay ", "resort ")
                )
                candidate = (
                    candidate_raw
                    if keep_lodging_prefix
                    else self._canonicalize_entity_name(candidate_raw).strip(" ,.;:!?")
                )
                candidate = self._strip_entity_tail_noise(candidate)
                if candidate and not self._is_broad_location_anchor(candidate):
                    return candidate
                # If candidate contains proximity words, try extracting the actual entity name
                if candidate:
                    cand_norm = normalize_text(candidate, strip_punct=True)
                    if any(pw in cand_norm for pw in ["gan", "xung quanh", "quanh", "lan can", "cung khu vuc"]):
                        lodging_match = _LODGING_PATTERN.search(candidate)
                        if lodging_match:
                            lodging_entity = f"{lodging_match.group(1)} {lodging_match.group(2)}".strip()
                            lodging_entity = lodging_entity.strip(" ,.;:!?")
                            lodging_entity = self._strip_entity_tail_noise(lodging_entity)
                            if lodging_entity and not self._is_broad_location_anchor(lodging_entity):
                                return lodging_entity
                        after_proximity = re.search(
                            rf"(?i){_PROXIMITY_WORDS}\s+(.+?)(?:\s*(?:,|\.|$))",
                            candidate,
                        )
                        if after_proximity:
                            refined = self._canonicalize_entity_name(after_proximity.group(1)).strip(" ,.;:!?")
                            if refined and not self._is_broad_location_anchor(refined):
                                return refined
        # "X tên là Y" / "X có tên là Y" pattern — extract Y as the entity name
        # e.g. "quán ăn tên là Bánh xèo tôm nhảy Gia Vỹ địa chỉ ở đâu"
        #   → "Bánh xèo tôm nhảy Gia Vỹ"
        _TEN_LA_PATTERNS = [
            r"(?i)(?:tên|ten)\s+(?:là|la)\s+(.+?)(?:\s+(?:địa\s+chỉ|dia\s+chi|ở|o|nằm|nam|tọa|toa|có|co|được|duoc|và|va|cũng|cung|thì|thi|gần|gan)|[?.!,;:]|$)",
            r"(?i)(?:có|co)\s+(?:tên|ten)\s+(?:là|la)\s+(.+?)(?:\s+(?:địa\s+chỉ|dia\s+chi|ở|o|nằm|nam|tọa|toa|có|co|được|duoc|và|va|cũng|cung|thì|thi|gần|gan)|[?.!,;:]|$)",
            r"(?i)(?:được|duoc)\s+(?:gọi|goi)\s+(?:là|la)\s+(.+?)(?:\s+(?:địa\s+chỉ|dia\s+chi|ở|o|nằm|nam|tọa|toa|có|co|được|duoc|và|va|cũng|cung|thì|thi|gần|gan)|[?.!,;:]|$)",
        ]
        for pat in _TEN_LA_PATTERNS:
            m = re.search(pat, raw)
            if m:
                candidate = self._strip_entity_tail_noise(str(m.group(1) or "")).strip(" ,.;:!?")
                if candidate and len(candidate) >= 3 and not self._is_broad_location_anchor(candidate):
                    return candidate

        # Direct lodging/restaurant entity extraction as robust fallback
        lodging_match = _LODGING_PATTERN.search(raw)
        if lodging_match:
            lodging_entity = f"{lodging_match.group(1)} {lodging_match.group(2)}".strip()
            lodging_entity = lodging_entity.strip(" ,.;:!?")
            lodging_entity = self._strip_entity_tail_noise(lodging_entity)
            if lodging_entity and not self._is_broad_location_anchor(lodging_entity):
                return lodging_entity
        # Pronouns/deictics that should never be treated as proximity anchors
        _PRONOUN_ANCHORS = {
            "day", "do", "nay", "kia", "ay", "no", "chung", "toi", "ban",
            "đây", "đó", "này", "kia", "ấy", "nó", "chúng", "tôi", "bạn",
        }
        patterns = [
            r"(?i)(?:nằm\s+gần|nam\s+gan|ở\s+gần|o\s+gan|gần|gan|xung\s+quanh|quanh|lân\s+cận|lan\s+can)\s+(.+?)(?:[?.!,;:]|$)",
            r"(?i)(?:cùng\s+(?:một\s+)?khu\s+vực|cung\s+(?:mot\s+)?khu\s+vuc)\s+(?:với|voi)?\s*(.+?)(?:[?.!,;:]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if not match:
                continue
            raw_match = str(match.group(1) or "").strip(" ,.;:!?")
            # Skip pronouns/deictics — they are not real entity anchors
            if normalize_text(raw_match, strip_punct=True) in _PRONOUN_ANCHORS:
                continue
            raw_match_norm = normalize_text(raw_match, strip_punct=True)
            keep_prefix = raw_match_norm.startswith(
                ("nha nghi ", "khach san ", "homestay ", "resort ")
            )
            candidate = (
                raw_match
                if keep_prefix
                else self._canonicalize_entity_name(raw_match)
            )
            candidate = re.sub(
                r"(?i)\b(?:với|voi|nào|nao|những|nhung|các|cac|địa\s+điểm|dia\s+diem|văn\s+hóa|van\s+hoa|du\s+lịch|du\s+lich|loại\s+hình|loai\s+hinh|điểm\s+đến|diem\s+den|sau\s+đây|sau\s+day|có\s+thể|co\s+the|tạo\s+nên|tao\s+nen|lợi\s+thế|loi\s+the|cạnh\s+tranh|canh\s+tranh|cơ\s+sở|co\s+so|ngành|nganh|theo|dựa\s+trên|dua\s+tren)\b.*$",
                "",
                candidate,
            ).strip(" ,.;:!?")
            candidate = self._strip_entity_tail_noise(candidate)
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        return ""

    def _extract_service_subject_entity_hint(self, query: str) -> str:
        """Extract the business/accommodation entity in service availability questions."""
        raw = str(query or "").strip()
        if not raw:
            return ""
        q_norm = normalize_text(raw, strip_punct=True)
        if not any(signal in q_norm for signal in keywords.SERVICE_SIGNALS):
            return ""

        entity_prefix = (
            r"(?:nhà\s+hàng|nha\s+hang|quán|quan|khách\s+sạn|khach\s+san|"
            r"nhà\s+nghỉ|nha\s+nghi|homestay|resort)"
        )
        patterns = [
            rf"(?i)^({entity_prefix}\s+.+?)\s+(?:ở|o|tại|tai)\s+.+?\s+(?:có|co)\b",
            rf"(?i)^({entity_prefix}\s+.+?)\s+(?:có|co)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if not match:
                continue
            candidate = self._strip_entity_tail_noise(str(match.group(1) or "")).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        head = re.split(r"(?i)\s+(?:có|co)\b", raw, maxsplit=1)[0].strip(" ,.;:!?")
        head = re.sub(r"(?i)\s+(?:ở|o|tại|tai)\s+[^,?.!]+$", "", head).strip(" ,.;:!?")
        if re.match(rf"(?i)^{entity_prefix}\s+", head):
            candidate = self._strip_entity_tail_noise(head).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate
        return ""

    def _is_broad_location_anchor(self, text: str) -> bool:
        q = normalize_text(text, strip_punct=True)
        if not q:
            return True
        admin_prefixes = (
            "phuong ",
            "xa ",
            "thi tran ",
            "thi xa ",
            "huyen ",
            "thanh pho ",
            "tp ",
            "tinh ",
        )
        if q.startswith(admin_prefixes):
            return True
        raw_lower = str(text or "").strip().lower()
        if q.startswith(("quan nao", "quan nhung", "quan cac")) or raw_lower in {"nao", "nào", "nāo"}:
            return True
        return q in keywords.BROAD_LOCATION_NAMES

    def _is_proximity_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        q = re.sub(r"\bgan\s+gui\b", " ", q)
        # Use word-boundary regex for short tokens to avoid false positives.
        # "gan" matches inside "tham quan" → must check as whole word.
        _WHOLE_WORD = ["gan", "quanh"]
        for tok in _WHOLE_WORD:
            if re.search(rf"\b{tok}\b", q):
                return True
        return any(
            token in q
            for token in [
                "nam gan",
                "o gan",
                "xung quanh",
                "lan can",
                "cung khu vuc",
            ]
        )

    def _is_address_lookup_query(self, query: str) -> bool:
        """
        Detect address lookup queries: "ở đâu", "nằm ở đâu", "địa chỉ", "chỗ nào", "vị trí"
        EXCLUDE nearby food/accommodation queries: "gần nhà hàng", "gần quán"
        """
        q = normalize_text(query, strip_punct=True)
        if not q:
            return False
        if any(hint in q for hint in self.ANALYTICAL_LOCATION_HINTS):
            return False
        if any(signal in q for signal in keywords.SERVICE_SIGNALS):
            return False

        # Exclude categorical discovery/recommendation queries (e.g. "ăn ... ở đâu", "mua ... ở đâu", "chơi ... ở đâu")
        # because these represent category/product searches rather than static address lookups.
        if re.search(r"\b(an|mua|choi|uong|tham quan|thuong thuc)\b.*\b(o dau|cho nao)\b", q):
            return False
        if re.search(r"\b(o dau|cho nao)\b.*\b(ban|ngon|re|co)\b", q):
            return False

        # "cho nao" can mean an address lookup ("quan X o cho nao?"), but in
        # proximity + lodging wording it asks for nearby places to stay/rest.
        # Let the accommodation branch handle those queries before the generic
        # address template can answer with only the anchor address.
        if self._is_nearby_accommodation_query(query):
            return False
        # Follow-up queries about sleeping/eating need accommodation handler, not address
        if any(m in q for m in ["ngu o dau", "an o dau", "tim noi ngu", "tim khach san", "tim nha nghi"]):
            return False
        if self._is_proximity_query(query) and any(
            token in q
            for token in ["van hoa", "dia diem", "gan voi", "gan nhung", "gan cac", "gan voi nhung"]
        ):
            return False
        
        # Exclude nearby food/accommodation queries - these should use FOOD/ACCOMMODATION intent
        exclusion_patterns = [
            "gan nha hang",
            "gan quan",
            "o gan nha hang",
            "o gan quan",
            "nam gan nha hang",
            "nam gan quan",
        ]
        is_nearby_food = any(pattern in q for pattern in exclusion_patterns)
        if is_nearby_food:
            return False

        categorical_action_patterns = [
            r"\b(?:mua|an|uong|choi|tham quan|thuong thuc|tim|kiem|mua duoc|an duoc)\b.+\b(?:o dau|cho nao|noi nao)\b",
            r"\b(?:o dau|cho nao|noi nao)\b.+\b(?:mua|an|uong|choi|tham quan|thuong thuc)\b",
        ]
        if any(re.search(pattern, q) for pattern in categorical_action_patterns):
            return False
        
        # Core address lookup signals - these FORCE ENTITY_FACT intent
        signals_normalized = [
            "o dau",
            "nam o dau",
            "dia chi",
            "cho nao",
            "duoc dat tai",
            "dat tai",
            "nam tai",
            "toa lac tai",
        ]
        signals_original = [
            "ở đâu",
            "nằm ở đâu",
            "địa chỉ",
            "chỗ nào",
            "vị trí",
            "được đặt tại",
            "đặt tại",
            "nằm tại",
            "tọa lạc tại",
        ]
        
        is_match_normalized = any(s in q for s in signals_normalized)
        is_match_original = any(s in query for s in signals_original)
        result = is_match_normalized or is_match_original
        
        if result:
            logger.info("   -> Address lookup detected: normalized='%s', original='%s'", q, query)
        
        return result

    def _is_mixed_address_and_description_query(self, query: str) -> bool:
        """Detect queries that ask BOTH for address/location AND descriptive info.

        Example: "Cà phê Pleiku có hương vị như thế nào và có thể mua ở đâu?"
        -> should NOT short-circuit to address-only answer.
        """
        if not self._is_address_lookup_query(query):
            return False

        q = normalize_text(query, strip_punct=True)
        descriptive_signals = [
            "huong vi",
            "nhu the nao",
            "ngon",
            "dac san",
            "gioi thieu",
            "mo ta",
            "danh gia",
            "review",
            "thu vi",
            "mon an",
            "mon nao",
            "co gi",
            "noi tieng",
        ]
        matched = [s for s in descriptive_signals if s in q]
        if matched:
            logger.info("   -> Mixed address+description query detected: signals=%s", matched)
            return True
        return False

    def _is_shared_location_fill_blank_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        if "___" not in str(query or ""):
            return False
        return any(
            signal in q
            for signal in [
                "deu nam o",
                "cung nam o",
                "deu thuoc",
                "cung thuoc",
                "deu toa lac tai",
                "cung toa lac tai",
            ]
        )

    def _extract_shared_location_entity_hints(self, query: str) -> List[str]:
        raw = str(query or "").strip()
        if not raw:
            return []
        head = re.split(
            r"(?i)\s+(?:đều|deu|cùng|cung)\s+(?:nằm|nam|thuộc|thuoc|tọa\s+lạc|toa\s+lac)\b",
            raw,
            maxsplit=1,
        )[0]
        parts = re.split(r"\s+(?:và|va|,)\s+", head)
        hints: List[str] = []
        for part in parts:
            candidate = self._canonicalize_entity_name(part).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                hints.append(candidate)
        return hints[:4]

    def _location_from_node_text(self, node: Any) -> str:
        p = self.pipeline
        texts = [
            str(getattr(node, "metadata", {}).get("address") or ""),
            str(getattr(node, "metadata", {}).get("description") or ""),
            str(getattr(node, "content", "") or ""),
        ]
        normalized = normalize_text(" ".join(texts), strip_punct=True)
        location_candidates = [
            ("Xã An Lão", ["xa an lao", "huyen an lao", "thi tran an lao", "an lao"]),
            ("Phường Quy Nhơn", ["phuong quy nhon", "thanh pho quy nhon", "tp quy nhon", "quy nhon"]),
            ("Phường Pleiku", ["phuong pleiku", "thanh pho pleiku", "tp pleiku", "pleiku"]),
        ]
        for display, aliases in location_candidates:
            if any(alias in normalized for alias in aliases):
                return display

        patterns = [
            r"(?i)(xã|phường|thị trấn|thị xã|huyện|thành phố|tỉnh)\s+([^,.;\n]+)",
            r"(?i)(xa|phuong|thi tran|thi xa|huyen|thanh pho|tinh)\s+([^,.;\n]+)",
        ]
        for text in texts:
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    return f"{match.group(1)} {match.group(2)}".strip()
        return ""

    def _is_nearby_accommodation_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        if not q:
            return False
        if any(sig in q for sig in keywords.ANALYSIS_SIGNALS):
            return False
        has_near_signal = any(token in q for token in keywords.PROXIMITY_SIGNALS)
        has_accommodation_signal = any(token in q for token in keywords.ACCOMMODATION_SIGNALS)
        if not has_accommodation_signal:
            has_accommodation_signal = any(
                token in q
                for token in ["khách sạn", "lưu trú", "nơi ở"]
            )
        # Follow-up queries like "ngủ ở đâu", "ăn ở đâu" don't have explicit "near"
        # but need accommodation/food handler — context comes from conversation history
        if not has_near_signal and not has_accommodation_signal:
            if any(m in q for m in ["ngu o dau", "tim noi ngu", "tim khach san", "tim nha nghi"]):
                return True
        return has_near_signal and has_accommodation_signal

    def _is_nearby_cultural_category_query(self, query: str) -> bool:
        q = normalize_text(query, strip_punct=True)
        if not q:
            return False
        if any(sig in q for sig in keywords.ANALYSIS_SIGNALS):
            return False
        # Skip cultural short-circuit if query also asks about food/restaurants
        # so the full pipeline can include Restaurant/Dish results.
        if any(sig in q for sig in keywords.FOOD_SIGNALS):
            return False
        has_proximity_signal = any(token in q for token in keywords.PROXIMITY_SIGNALS)
        has_category_signal = any(token in q for token in keywords.HERITAGE_SIGNALS | keywords.TOURISM_SIGNALS)
        return has_proximity_signal and has_category_signal

    def _is_strict_tour_itinerary_query(self, query: str) -> bool:
        raw = str(query or "").strip().lower()
        q = normalize_text(query, strip_punct=True)
        if not q and not raw:
            return False

        starts_with_tour = q.startswith("tour ") or raw.startswith("tour ")
        has_tour = ("tour" in q) or ("tour" in raw)
        has_itinerary = any(
            token in q
            for token in ["lich trinh", "the nao", "ra sao", "di dau", "gom nhung", "bao gom"]
        ) or ("lịch trình" in raw)
        has_strict_source = (
            ("theo du lieu he thong" in q)
            or ("theo dữ liệu hệ thống" in raw)
            or ("in nguyen" in q)
            or ("in nguyên" in raw)
            or ("nguyen ban" in q)
            or ("nguyên bản" in raw)
        )
        no_extra = ("khong goi y them diem khac" in q) or ("không gợi ý thêm điểm khác" in raw)
        return has_tour and has_itinerary and (starts_with_tour or has_strict_source or no_extra)

    def _extract_tour_name_hint(self, query: str) -> str:
        raw = str(query or "").strip()
        if not raw:
            return ""

        # Extract from the first "tour ..." chunk, then trim question/control suffixes.
        match = re.search(r"(?i)\btour\b\s+(.+)$", raw)
        if not match:
            return ""
        tail = match.group(1).strip(" .:;-\u2013\u2014")
        if not tail:
            return ""

        candidate = f"Tour {tail}".strip()
        candidate = re.sub(r"(?i)^tour\s+tour\s+", "Tour ", candidate)

        # Trim strict control suffixes that are not part of the tour name.
        stop_markers = [
            "theo dữ liệu hệ thống",
            "theo du lieu he thong",
            "không gợi ý thêm điểm khác",
            "khong goi y them diem khac",
            "(bao gồm",
            "(bao gom",
            "bao gồm điểm đến",
            "bao gom diem den",
            "lịch trình",
            "lich trinh",
            "như thế nào",
            "nhu the nao",
            "thế nào",
            "the nao",
            "ra sao",
            "gồm những",
            "gom nhung",
            "đi đâu",
            "di dau",
        ]
        low = candidate.lower()
        cut = len(candidate)
        for marker in stop_markers:
            idx = low.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        candidate = candidate[:cut].strip(" .:;-\u2013\u2014")
        return candidate

    def _has_two_or_more_explicit_destinations(self, entities: List[Dict[str, Any]] | None) -> bool:
        if not entities:
            return False
        names = set()
        for entity in entities:
            if not self._is_groundable_entity(entity):
                continue
            e_type = str(entity.get("type") or "").lower()
            if e_type == "location":
                continue
            e_name = normalize_text(str(entity.get("name") or ""), strip_punct=True)
            if e_name and len(e_name) >= 3:
                names.add(e_name)
        return len(names) >= 2

    # --- Multi-Select / Multi-Choice / True-or-False entity extraction ---

    _MULTI_CHOICE_PREFIXES = [
        r"(?i)^những\s+cái\s+nào\s+dưới\s+đây\s+la\s+",
        r"(?i)^những\s+địa\s+điểm\s+nào\s+dưới\s+đây\s+la\s+",
        r"(?i)^đâu\s+la\s+",
        r"(?i)^chọn\s+",
        r"(?i)^các\s+đặc\s+điểm\s+nào\s+dưới\s+đây\s+",
    ]

    _CATEGORY_PATTERNS = [
        (r"(?i)địa\s+điểm\s+(?:văn\s+hóa|van\s+hoa)\s*[-–]?\s*(?:tâm\s+linh|tam\s+linh)", "văn hóa tâm linh"),
        (r"(?i)di\s+tích\s+(?:lịch\s+sử|lich\s+su)", "di tích lịch sử"),
        (r"(?i)địa\s+điểm\s+(?:du\s+lịch|du\s+lich)", "điểm du lịch"),
        (r"(?i)loại\s+hình\s+(?:điểm\s+đến|diem\s+den)", "loại hình điểm đến"),
        (r"(?i)đặc\s+điểm\s+(?:hoặc|hay)\s+thông\s+tin", "thông tin"),
        (r"(?i)địa\s+điểm\s+(?:văn\s+hóa|van\s+hoa)\s+(?:hoặc|hay)\s+(?:du\s+lịch|du\s+lich)", "văn hóa hoặc du lịch"),
        (r"(?i)địa\s+điểm\s+(?:văn\s+hóa|van\s+hoa)", "văn hóa"),
        (r"(?i)địa\s+điểm\s+(?:du\s+lịch|du\s+lich)\s+(?:văn\s+hóa|van\s+hoa)\s*[-–]?\s*(?:tâm\s+linh|tam\s+linh)", "văn hóa tâm linh"),
    ]

    def _extract_multi_choice_anchor_hint(self, query: str) -> str:
        """Extract the anchor entity from Multi-Select/Choice questions.

        Handles patterns like:
          "Những cái nào dưới đây là địa điểm văn hóa - tâm linh gần Khách sạn X?"
          "Nhà nghỉ X nằm gần những địa điểm du lịch nào sau đây?"
        """
        raw = str(query or "").strip()
        if not raw:
            return ""
        q_norm = normalize_text(raw, strip_punct=True)

        possessive_anchor_match = re.search(
            r"(?i)(?:của|cua)\s+(.+?)(?:\?|$)",
            raw.splitlines()[0].strip(),
        )
        if possessive_anchor_match:
            candidate = self._strip_entity_tail_noise(str(possessive_anchor_match.group(1) or "")).strip(" ,.;:!?")
            if candidate and len(normalize_text(candidate, strip_punct=True).split()) >= 2 and not self._is_broad_location_anchor(candidate):
                return candidate

        from_anchor_match = re.search(r"(?i)^\s*(?:từ|tu)\s+(.+?)(?:,|\?)", raw)
        if from_anchor_match:
            candidate = self._strip_entity_tail_noise(str(from_anchor_match.group(1) or "")).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        related_entity_match = re.search(
            r"(?i)(?:liên\s+quan\s+trực\s+tiếp\s+đến|lien\s+quan\s+truc\s+tiep\s+den)\s+(?:(?:nhà\s+hàng|nha\s+hang|quán|quan|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi)\s+)?['\"“”‘’]?(.+?)['\"“”‘’]?(?:\s+dựa|\s+dua|\?|$)",
            raw,
        )
        if related_entity_match:
            name = related_entity_match.group(1).strip(" '\"“”‘’.;:,")
            prefix = ""
            prefix_match = re.search(
                r"(?i)(nhà\s+hàng|nha\s+hang|quán|quan|khách\s+sạn|khach\s+san|nhà\s+nghỉ|nha\s+nghi)\s+['\"“”‘’]?"
                + re.escape(name[:20]),
                raw,
            )
            if prefix_match:
                prefix = prefix_match.group(1)
            candidate = f"{prefix} {name}".strip()
            candidate = self._strip_entity_tail_noise(candidate).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        info_anchor_match = re.search(
            r"(?i)(?:thông\s+tin\s+về|thong\s+tin\s+ve)\s+(.+?)(?:,|\s+nếu|\s+neu|\s+dựa|\s+dua|\?|$)",
            raw,
        )
        if info_anchor_match:
            candidate = self._strip_entity_tail_noise(str(info_anchor_match.group(1) or "")).strip(" ,.;:!?")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        first_line = raw.splitlines()[0].strip()
        leading_entity_match = re.search(
            r"(?i)^(.+?)\s+(?:được|duoc|thuộc|thuoc|nằm|nam|là|la)\b",
            first_line,
        )
        if leading_entity_match:
            candidate = self._strip_entity_tail_noise(str(leading_entity_match.group(1) or "")).strip(" ,.;:!?")
            candidate_norm = normalize_text(candidate, strip_punct=True)
            if (
                candidate
                and len(candidate_norm.split()) >= 2
                and not any(candidate_norm.startswith(pfx) for pfx in ["nhung cai nao", "dau la", "chon", "dua tren"])
                and not self._is_broad_location_anchor(candidate)
            ):
                return candidate

        _LODGING_RE = re.compile(
            r"(?i)(nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort|nhà\s+hàng|nha\s+hang|quán|quan|tour|khu\s+du\s+lịch|khu\s+du\s+lich)\s+",
        )
        _STOP_WORDS = re.compile(
            r"(?i)\s+(?:nằm|nam|ở|o|tại|tai|là|la|có|co|được|duoc|và|va|gần|gan|với|voi)\b",
        )

        def _keep_or_canonicalize(text: str) -> str:
            text = self._strip_entity_tail_noise(text).strip(" ,.;:!?")
            if not text:
                return ""
            text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip(" ,.;:!?")
            norm = normalize_text(text, strip_punct=True)
            if re.match(r"(?:nha nghi|khach san|homestay|resort|nha hang|quan|tour|khu du lich)\b", norm):
                return text
            return self._canonicalize_entity_name(text)

        def _extract_lodging_name(text: str) -> str:
            """Extract lodging entity name by splitting at first stop word."""
            m = _LODGING_RE.search(text)
            if not m:
                return ""
            start = m.start()
            rest = text[m.end():]
            stop_m = _STOP_WORDS.search(rest)
            if stop_m:
                name = rest[:stop_m.start()].strip()
            else:
                name = rest.strip(" ,.;:!?")
            if name:
                return f"{m.group(1).strip()} {name}".strip()
            return m.group(1).strip()

        # Pattern 0: Tour entity at start — "Tour X do/bao gồm/..."
        tour_match = re.search(
            r"(?i)^(tour\s+.+?)\s+(?:do\s+(?:công|cong)\s+ty|bao\s+gồm|bao\s+gom|\?|$)",
            raw,
        )
        if tour_match:
            candidate = _keep_or_canonicalize(tour_match.group(1))
            if candidate:
                return candidate

        quoted_entity_match = re.search(
            r"(?i)(?:nhà\s+hàng|nha\s+hang|quán|quan)\s+['\"“”‘’]([^'\"“”‘’]+)['\"“”‘’]",
            raw,
        )
        if quoted_entity_match:
            prefix = "nhà hàng" if re.search(r"(?i)(?:nhà\s+hàng|nha\s+hang)\s+['\"“”‘’]", raw) else "quán"
            candidate = _keep_or_canonicalize(f"{prefix} {quoted_entity_match.group(1)}")
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        # Check if query has proximity signal
        has_proximity = any(
            token in q_norm
            for token in ["gan", "xung quanh", "quanh", "lan can"]
        )

        # Pattern 1: "gần Lodging X?" — entity AFTER "gần"
        _LODGING_PAT = r"(?:nhà\s+nghỉ|nha\s+nghi|khách\s+sạn|khach\s+san|homestay|resort|nhà\s+hàng|nha\s+hang|quán|quan)"
        suffix_with_lodging = re.search(
            r"(?i)(?:gần|gan)\s+(" + _LODGING_PAT + r"\s+.+?)\s*(?:\?|$|\bsau\s+đây|\bnhững|\bcác|\bva\b|\bvà\b)",
            raw,
        )
        if suffix_with_lodging:
            candidate = _keep_or_canonicalize(suffix_with_lodging.group(1))
            if candidate and not self._is_broad_location_anchor(candidate):
                return candidate

        # Pattern 2: lodging entity at start + proximity anywhere in sentence
        if has_proximity:
            lodging = _extract_lodging_name(raw)
            if lodging:
                candidate = _keep_or_canonicalize(lodging)
                if candidate and not self._is_broad_location_anchor(candidate):
                    return candidate

        # Pattern 3: "gần X?" where X is NOT a question word — entity AFTER "gần"
        # Use greedy prefix to match the LAST gần/gan (avoids capturing entire clause)
        suffix_match = re.search(
            r"(?i).*(?:gần|gan)\s+(.+?)\s*(?:\?|$|\bsau\s+đây|\bnhững|\bcác|\bva\b|\bvà\b)",
            raw,
        )
        if suffix_match:
            raw_candidate = suffix_match.group(1).strip(" ,.;:!?")
            # Skip if it's a question word or generic phrase
            cand_norm = normalize_text(raw_candidate, strip_punct=True)
            if not any(cand_norm.startswith(p) for p in ["o dau", "cho nao", "vi tri", "nhung", "cac", "bao xa"]):
                candidate = _keep_or_canonicalize(raw_candidate)
                if candidate and not self._is_broad_location_anchor(candidate):
                    return candidate

        # Pattern 4: "của X" / "thông tin về X"
        for pattern in [
            r"(?i)(?:của|cua)\s+(.+?)\s*(?:\?|$|\bdựa|\btheo|\bsau\s+đây|\bhãy|\bhar|\bcần|\bnếu|\bkhi)",
            r"(?i)(?:thông\s+tin\s+về|thong\s+tin\s+ve)\s+(.+?)\s*(?:\?|$|\bdựa|\btheo|\bnhững|\bcác|\bsau\s+đây|\bhãy|\bhar|\bcần|\bnếu|\bkhi)",
        ]:
            match = re.search(pattern, raw)
            if match:
                candidate = _keep_or_canonicalize(match.group(1))
                if candidate and not self._is_broad_location_anchor(candidate):
                    return candidate

        return ""

    def _extract_multi_choice_target_category(self, query: str) -> str:
        """Extract the target category from Multi-Select/Choice question body.

        Returns e.g. "văn hóa tâm linh", "di tích lịch sử", "điểm du lịch".
        """
        raw = str(query or "").strip()
        if not raw:
            return ""
        for pattern, label in self._CATEGORY_PATTERNS:
            if re.search(pattern, raw):
                return label
        return ""

    _CATEGORY_KEYWORDS = keywords.CATEGORY_KEYWORDS
