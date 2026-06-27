"""Answer text sanitization — cleanup LLM output before returning to user."""

import re
from typing import List


def format_missing_attributes(attributes: List[str]) -> str:
    """Convert technical attribute names to user-friendly Vietnamese labels."""
    labels = {
        "address": "địa chỉ",
        "phone": "số điện thoại",
        "price": "giá",
        "ticket_price": "giá vé",
        "price_range": "giá phòng",
        "opening_hours": "giờ mở cửa",
        "service_features": "thông tin dịch vụ",
        "room_count": "số lượng phòng",
        "room_type": "loại phòng",
        "amenities": "tiện nghi",
    }
    readable = [labels.get(attr, attr.replace("_", " ")) for attr in attributes]
    return "thông tin " + ", ".join(dict.fromkeys(readable))


def sanitize_answer_text(answer: str) -> str:
    """Clean up LLM-generated answer: replace Chinese chars, strip graph markup, etc."""
    if not answer:
        return ""

    replacements = {
        "确实": "thực sự",
        "远离": "tránh xa",
        "如": " như ",
        "nearby": "gần đó",
        "events": "sự kiện",
        "refresh": "thư giãn",
    }
    cleaned = str(answer)
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    natural_language_replacements = [
        (r"(?i)dựa trên thông tin trong\s+context\b",
         "Dựa trên thông tin hiện có"),
        (r"(?i)dựa trên\s+context\b",
         "Dựa trên thông tin hiện có"),
        (r"(?i)\bcontext\b", "thông tin hiện có"),
    ]
    for pattern, target in natural_language_replacements:
        cleaned = re.sub(pattern, target, cleaned)

    # Clean up coordinate class names
    cleaned = re.sub(r"(?i)(?:có\s+)?tọa\s+độ\s+WGS84Point\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"(?i)WGS84Point\([^)]*\)", "", cleaned)
    cleaned = re.sub(r"(?i)Point\([^)]*\)", "", cleaned)

    # Clean up leaked graph notations
    cleaned = re.sub(
        r"(?i)\s*[\(\[](?:NEAR|LOCATED_IN|BELONGS_TO|HAS|HELD_AT|INCLUDES|OFFERS)[\)\]]\s*",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\s*(?:->|-->|<-|<--)\s*", " ", cleaned)

    # Remove Chinese characters
    cleaned = re.sub(r"[㐀-䶿一-鿿豈-﫿]", "", cleaned)

    # Fix punctuation issues
    cleaned = re.sub(r",\s*\.", ".", cleaned)
    cleaned = re.sub(r",\s*,", ",", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip()
