"""
Region Patterns - Cấu hình phát hiện vùng miền cho multi-region detection.

File này chứa tất cả keyword/constant dùng để:
1. Phát hiện query đa vùng ("từ A đến B", "A rồi B")
2. Gán region_focus (ALL / inland_gia_lai / coastal_quy_nhon / …)
3. Filter kết quả theo khu vực địa lý

Debug: Thêm / sửa keyword ở đây, không cần sửa code ở nhiều nơi.
"""

from typing import List

# =============================================================================
# MULTI_REGION_NAMES
# Tên địa lý (tỉnh / thành / điểm du lịch) dùng để detect query đa vùng.
# Query cần ≥2 tên trong list này (hoặc ≥1 tên + ≥1 connector) → flagged.
# =============================================================================

MULTI_REGION_NAMES: List[str] = [
    # --- Bình Định ---
    "binh dinh",
    "quy nhon",
    "hon kho",
    "ky co",
    "eo gio",
    # --- Tây Nguyên ---
    "tay nguyen",
    "buon ma thuot",
    # --- Gia Lai (thêm nếu cần detect "Gia Lai + Bình Định") ---
    # "gia lai",   # uncomment nếu muốn "Gia Lai" cũng là region name
    # "pleiku",    # WARNING: "pleiku" là 1 location đơn, KHÔNG phải multi-region marker
]

# =============================================================================
# TYPE_HEADER_MAP
# Map entity type → nhãn tiếng Việt hiển thị trong answer header.
# Dùng cho: _render_discovery_from_context(), _render_discovery_from_candidates()
# =============================================================================

TYPE_HEADER_MAP = {
    "Accommodation": "khách sạn / địa điểm lưu trú",
    "Restaurant": "quán ăn / nhà hàng",
    "TouristAttraction": "địa điểm tham quan",
    "Event": "sự kiện",
    "Tour": "tour du lịch",
    "Dish": "đặc sản / món ngon",
    "TravelInfo": "thông tin di chuyển",
}

# =============================================================================
# DISCOVERY_LOCATION_THRESHOLD
# Độ dài tối thiểu của location string để áp dụng region filter cho DISCOVERY intent.
# Tránh filter khi location quá ngắn (ví dụ "Gia Lai" → 7 chars → OK).
# =============================================================================

DISCOVERY_LOCATION_MIN_LENGTH = 2
