"""Shared signal lists for ContractValidator detection logic."""

# Advice/tips triggers (normalized, no accents)
ADVICE_SIGNALS = frozenset(
    [
        "kinh nghiem",
        "meo",
        "luu y",
        "nen chuan bi",
        "can biet",
        "tiet kiem",
        "can chuan bi",
        "can mang",
        "can biet truoc",
        "can luu y",
    ]
)

# Booking-related triggers (normalized, no accents)
BOOKING_SIGNALS = frozenset(
    [
        "dat phong",
        "booking",
        "book phong",
        "phong khach san",
        "phong nghi",
        "homestay",
        "resort",
    ]
)
