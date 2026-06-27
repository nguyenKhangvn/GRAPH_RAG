import re
from typing import Dict, Optional, Tuple
from graph_rag.core import thresholds
from graph_rag.utils.geo import haversine_km

CITY_CENTER_COORDS = {
    "gia_lai": (13.9833, 108.0000),
    "quy_nhon": (13.7820, 109.2197),
}

CONSTRAINT_PHRASES = {
    "no_cano": [
        "khong cano",
        "ko cano",
        "khong di cano",
        "khong tau cao toc",
        "khong speedboat",
        "khong di tau",
    ],
    "no_climb": [
        "khong leo",
        "ko leo",
        "khong leo nui",
        "khong trekking",
        "khong di bo doc",
        "dau khop",
    ],
    "low_mobility": [
        "nguoi gia",
        "tre em",
        "be nho",
        "di chuyen cham",
        "it di bo",
        "low mobility",
        "han che van dong",
    ],
    "relaxed_route_guardrails": [
        "tha long",
        "bo nguong",
        "bo gioi han",
        "noi nguong",
        "mo rong ban kinh",
        "khong gioi han quang duong",
    ],
}

ACCOMMODATION_TERMS = {
    "khach san",
    "hotel",
    "resort",
    "homestay",
    "villa",
    "hostel",
    "nha nghi",
    "cho nghi",
    "luu tru",
}


def extract_deadline_time(query: str, normalize_text) -> str:
    q = normalize_text(query)
    match = re.search(r"\b(\d{1,2})[:h](\d{1,2})?\b", q)
    if not match:
        match = re.search(r"\b(\d{1,2})\s*gio\b", q)
    if not match:
        return "15:00"

    hour = int(match.group(1))
    minute = int(match.group(2)) if match.lastindex and match.lastindex >= 2 and match.group(2) else 0

    if "chieu" in q and hour < 12:
        hour += 12

    hour = max(0, min(hour, 23))
    minute = max(0, min(minute, 59))
    return f"{hour:02d}:{minute:02d}"


def extract_trip_duration(query: str, normalize_text) -> Tuple[int, int]:
    q = normalize_text(query)

    if any(token in q for token in ["nua ngay", "nửa ngày", "trong ngay", "1 buoi", "1 buổi"]):
        return 1, 0

    compact = re.search(r"\b(\d{1,2})\s*n\s*(\d{1,2})\s*d\b", q)
    if compact:
        days = int(compact.group(1))
        nights = int(compact.group(2))
        return max(1, min(days, 14)), max(0, min(nights, 14))

    day_match = re.search(r"\b(\d{1,2})\s*(?:ngay|nay|ngy)\b", q)
    night_match = re.search(r"\b(\d{1,2})\s*dem\b", q)

    if day_match:
        days = int(day_match.group(1))
        nights = int(night_match.group(1)) if night_match else max(0, days - 1)
        return max(1, min(days, 14)), max(0, min(nights, 14))

    if "2n1d" in q:
        return 2, 1

    return 2, 1


def extract_trip_constraints(query: str, normalize_text) -> Dict[str, bool]:
    q = normalize_text(query)
    return {
        key: any(phrase in q for phrase in phrases)
        for key, phrases in CONSTRAINT_PHRASES.items()
    }


def distance_from_center(
    lat: Optional[float],
    lng: Optional[float],
    detected_location: Optional[str],
    normalize_text,
) -> Optional[float]:
    if lat is None or lng is None:
        return None

    loc = normalize_text(detected_location or "")
    if not loc:
        return None

    center_key = "gia_lai" if ("gia lai" in loc or "pleiku" in loc) else "quy_nhon"
    center_lat, center_lng = CITY_CENTER_COORDS[center_key]

    return haversine_km(center_lat, center_lng, lat, lng)


def infer_location_cluster(
    location: str,
    distance_km: Optional[float],
    detected_location: Optional[str],
    normalize_text,
) -> str:
    loc = normalize_text(location)
    target = normalize_text(detected_location or "")
    city_tokens = ["trung tam"]
    if any(token in target for token in ["gia lai", "pleiku"]):
        city_tokens.extend(["pleiku", "thanh pho pleiku", "tp pleiku", "gia lai"])
    else:
        city_tokens.extend(["tp quy nhon", "thanh pho quy nhon", "quy nhon"])

    if any(token in loc for token in city_tokens):
        if distance_km is None or distance_km <= thresholds.CITY_CENTER_DISTANCE_KM:
            return "city_center"
    if any(token in loc for token in ["nhon ly", "phuong mai", "ky co", "eo gio", "hon kho"]):
        return "far_coastal"
    if any(token in loc for token in ["an khe", "mang yang", "chu prong", "chu se", "dak doa", "ayun pa"]):
        return "far_inland"
    if distance_km is not None and distance_km <= thresholds.CITY_CENTER_FALLBACK_DISTANCE_KM:
        return "city_center"
    if distance_km is not None and distance_km <= thresholds.NEAR_SUBURBAN_DISTANCE_KM:
        return "near_suburban"
    if distance_km is not None:
        return "far_area"
    return "unknown"
