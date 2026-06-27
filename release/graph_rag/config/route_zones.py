"""Route zone definitions for tour plan clustering.

All geographic zones and adjacency rules for daily itinerary planning.
After administrative merger (2025), Gia Lai includes both inland Tây Nguyên
and coastal former Bình Định/Quy Nhơn areas.

To modify zones: edit this file, NOT route_optimizer_service.py.
"""

# ---------------------------------------------------------------------------
# Zone adjacency — which zones can be visited on the same day
# ---------------------------------------------------------------------------

KHUVUC_ADJACENCY = {
    # Quy Nhơn (old Bình Định coastal)
    "quy_nhon": {
        "Khu Trung Tam": {"Khu Trung Tam", "Khu Ban Dao", "Khu Ven Bien Bac", "Khu Ven Bien Nam"},
        "Khu Ban Dao": {"Khu Ban Dao", "Khu Trung Tam", "Khu Ven Bien Nam"},
        "Khu Ven Bien Bac": {"Khu Ven Bien Bac", "Khu Trung Tam"},
        "Khu Ven Bien Nam": {"Khu Ven Bien Nam", "Khu Ban Dao"},
    },
    # Gia Lai inland only (old province)
    "gia_lai": {
        "Khu Trung Tam Pleiku": {"Khu Trung Tam Pleiku", "Khu Ngoai O Pleiku", "Khu Cao Nguyen"},
        "Khu Ngoai O Pleiku": {"Khu Ngoai O Pleiku", "Khu Trung Tam Pleiku", "Khu Cao Nguyen"},
        "Khu Cao Nguyen": {"Khu Cao Nguyen", "Khu Trung Tam Pleiku", "Khu Ngoai O Pleiku"},
    },
    # Merged province: Pleiku inland + Quy Nhơn coastal
    "gia_lai_new": {
        "Khu Trung Tam Pleiku": {"Khu Trung Tam Pleiku", "Khu Ngoai O Pleiku", "Khu Cao Nguyen", "Khu Trung Tam"},
        "Khu Ngoai O Pleiku": {"Khu Ngoai O Pleiku", "Khu Trung Tam Pleiku", "Khu Cao Nguyen"},
        "Khu Cao Nguyen": {"Khu Cao Nguyen", "Khu Trung Tam Pleiku", "Khu Ngoai O Pleiku"},
        "Khu Trung Tam": {"Khu Trung Tam", "Khu Ban Dao", "Khu Ven Bien Bac", "Khu Ven Bien Nam", "Khu Trung Tam Pleiku"},
        "Khu Ban Dao": {"Khu Ban Dao", "Khu Trung Tam", "Khu Ven Bien Nam"},
        "Khu Ven Bien Bac": {"Khu Ven Bien Bac", "Khu Trung Tam"},
        "Khu Ven Bien Nam": {"Khu Ven Bien Nam", "Khu Ban Dao"},
    },
}

# ---------------------------------------------------------------------------
# Zone keywords — assign node to zone by name/address text
# ---------------------------------------------------------------------------

ZONE_KEYWORDS = {
    "gia_lai": {
        "Khu Trung Tam Pleiku": ["pleiku", "bien ho", "dai doan ket", "thien hung tu", "chua minh thanh", "pho thong nhat", "hung vuong", "tran phu", "le duan"],
        "Khu Ngoai O Pleiku": ["ia grai", "ia mo", "chư păh", "chu pah", "kong chro", "kbang", "mo hra", "bien ho che", "nui ham rong", "dong xanh"],
        "Khu Cao Nguyen": ["chu prong", "chu se", "mang yang", "dak doa", "an khe", "dak to", "kon tum", "ngoc linh"],
    },
    "quy_nhon": {
        "Khu Trung Tam": ["ghenh rang", "tuong dai", "trung tam", "quy nhon", "tran hung dao", "le loi", "nguyen hue"],
        "Khu Ban Dao": ["nhon ly", "nhon hai", "merryland", "ky co", "eo gio", "cua bien", "phuong mai", "phuoc mai", "cu lao xanh", "hon kho", "bai xep", "trung luong", "cat tien", "nhon chau"],
        "Khu Ven Bien Bac": ["phu cat", "phu my", "hoai nhon", "bong son", "an nhan"],
        "Khu Ven Bien Nam": ["tuy phuoc", "an nhan", "nhanh", "thi nai"],
    },
}

# ---------------------------------------------------------------------------
# Zone centers — for distance-based fallback assignment
# ---------------------------------------------------------------------------

ZONE_CENTERS = {
    "Khu Trung Tam Pleiku": (13.9730, 108.0120),
    "Khu Ngoai O Pleiku": (13.9300, 108.0500),
    "Khu Cao Nguyen": (14.0500, 108.1000),
    "Khu Trung Tam": (13.7820, 109.2197),
    "Khu Ban Dao": (13.7600, 109.2800),
    "Khu Ven Bien Bac": (13.9000, 109.1500),
    "Khu Ven Bien Nam": (13.7000, 109.1800),
}

# Profile name → which zone keyword/center sets to use
PROFILE_ZONE_MAP = {
    "gia_lai": ["gia_lai"],
    "quy_nhon": ["quy_nhon"],
    "gia_lai_new": ["gia_lai", "quy_nhon"],
}
