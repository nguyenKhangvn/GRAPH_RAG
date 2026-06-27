"""
Deictic Patterns - Cấu hình mẫu đại từ chỉ định cho coreference resolution.

File này chứa tất cả các pattern deictic cần detect để:
1. Không semantic search những phrase này (tránh resolve sai)
2. Fallback về last_active_entity / last_grounded_anchor
3. Trigger entity inheritance từ conversation history

Debug: Thêm pattern mới vào đây, không cần sửa code ở nhiều nơi.
"""

# =============================================================================
# DEICTIC_QUERY_PATTERNS
# Query chứa bất kỳ pattern nào trong list này → được coi là deictic reference.
# Dùng cho: _is_deictic_reference_query(), has_deictic_reference
# =============================================================================

DEICTIC_QUERY_PATTERNS = [
    # ── "đó" forms (chỉ entity đã nhắc trước đó, xa hơn) ──
    "o cho do",      # ở chỗ đó
    "cho do",        # chỗ đó
    "noi do",        # nơi đó
    "o do",          # ở đó
    "khu do",        # khu đó
    "vi tri do",     # vị trí đó
    "diem do",       # điểm đó
    "o cho ay",      # ở chỗ ấy
    "cho ay",        # chỗ ấy
    "noi ay",        # nơi ấy

    # ── "này" forms (chỉ entity vừa nói tới, gần hơn) ──
    "quan nay",      # quán này
    "cho nay",       # chỗ này
    "noi nay",       # nơi này
    "dia diem nay",  # địa điểm này
    "nha hang nay",  # nhà hàng này
    "khach san nay", # khách sạn này
    "homestay nay",  # homestay này
    "resort nay",    # resort này
    "o cho nay",     # ở chỗ này
    "o noi nay",     # ở nơi này
    "tai cho nay",   # tại chỗ này
    "tai noi nay",   # tại nơi này
    "noi day",       # nơi đây
    "cho day",       # chỗ đây

    # ── Pronoun-only references ──
    "no",            # nó (chỉ entity)
    "chung",         # chúng (chỉ entities)

    # ── Follow-up cần context từ hội thoại ──
    "ngu o dau",     # ngủ ở đâu
    "an o dau",      # ăn ở đâu
    "nam o dau",     # nằm ở đâu
    "tim noi ngu",   # tìm nơi ngủ
    "tim khach san", # tìm khách sạn
    "tim nha nghi",  # tìm nhà nghỉ

    # ── Proximity/current-location deictics ──
    "gan day",       # gần đây
    "gan cho toi",   # gần chỗ tôi
    "tu day",        # từ đây
    "tu cho toi",    # từ chỗ tôi
    "quanh day",     # quanh đây
    "quanh vung nay", # quanh vùng này
    "o day",         # ở đây
]

# =============================================================================
# DEICTIC_ENTITY_PHRASES
# Entity name trùng khớp pattern này → KHÔNG được phép semantic search.
# Phải resolve từ conversation state (last_active_entity hoặc entity_memory).
# Dùng cho: guard trước grounding, chặn semantic search.
# =============================================================================

DEICTIC_ENTITY_PHRASES = [
    # "này" forms
    "quan nay", "cho nay", "noi nay", "dia diem nay",
    "nha hang nay", "khach san nay", "homestay nay", "resort nay",
    "o cho nay", "o noi nay", "tai cho nay", "tai noi nay",
    # "đó" forms
    "quan do", "cho do", "noi do", "dia diem do",
    "nha hang do", "khach san do", "homestay do", "resort do",
    "o cho do", "o noi do", "tai cho do", "tai noi do",
    # "ấy" forms
    "quan ay", "cho ay", "noi ay", "dia diem ay",
    # Pronoun-only
    "no", "chung",
]

# =============================================================================
# DEICTIC_PHRASE_TO_TYPE_HINT
# Khi detect deictic phrase, gợi ý entity type để filter entity_memory.
# Ví dụ: "khách sạn này" → filter entity_memory theo type "Accommodation"
# =============================================================================

DEICTIC_PHRASE_TO_TYPE_HINT = {
    "quan":        "Restaurant",
    "nha hang":    "Restaurant",
    "khach san":   "Accommodation",
    "homestay":    "Accommodation",
    "resort":      "Accommodation",
    "villa":       "Accommodation",
    "nha nghi":    "Accommodation",
    "dia diem":    "TouristAttraction",
    "diem tham quan": "TouristAttraction",
    "bai bien":    "TouristAttraction",
    "dao":         "TouristAttraction",
    "thac":        "TouristAttraction",
    "nui":         "TouristAttraction",
    "ho":          "TouristAttraction",
    "cho":         None,  # có thể là Restaurant hoặc TouristAttraction
    "noi":         None,  # generic, không filter type
}


# =============================================================================
# PROXIMITY_DEICTIC_PATTERNS
# Query chứa pattern này → user muốn kết quả quanh vị trí hiện tại.
# current_location phải thắng explicit query region.
# =============================================================================

PROXIMITY_DEICTIC_PATTERNS = [
    "gan day",       # gần đây
    "gan cho toi",   # gần chỗ tôi
    "tu day",        # từ đây
    "tu cho toi",    # từ chỗ tôi
    "quanh day",     # quanh đây
    "quanh vung nay", # quanh vùng này
    "o day",         # ở đây
]


def is_deictic_query(normalized_query: str) -> bool:
    """Kiểm tra query có chứa deictic pattern không."""
    return any(pattern in normalized_query for pattern in DEICTIC_QUERY_PATTERNS)


def is_deictic_entity_phrase(normalized_entity_name: str) -> bool:
    """Kiểm tra entity name có phải deictic phrase không (cấm semantic search)."""
    return normalized_entity_name in DEICTIC_ENTITY_PHRASES


def get_type_hint_for_deictic(normalized_query: str) -> str | None:
    """Lấy type hint từ deictic phrase. Trả về None nếu không có hint.
    Ưu tiên phrase dài hơn (vd: "dia diem nay" > "dia diem").
    """
    best_hint = None
    best_len = 0
    for phrase, type_hint in DEICTIC_PHRASE_TO_TYPE_HINT.items():
        if phrase in normalized_query and len(phrase) > best_len:
            best_hint = type_hint
            best_len = len(phrase)
    return best_hint
