import re

# Common transport query keywords/signals to stop destination text greediness.
# Matches "mất bao lâu", "phương tiện", "bằng gì", "hết bao nhiêu", "đi bằng", etc.
DESTINATION_STOP_PATTERN = r"\s+(?:mat|bao lau|phuong tien|thuan tien|gia|chi phi|nhu the nao|the nao|khong|ha|đi|di|bang|bằng|het|hết|mat|mất)\b"

# Suffixes representing questions that should be removed from the ends of distance queries.
DISTANCE_TAIL_PATTERNS = [
    r"\s+la\s+bao\s+nhieu\s+km\s*$",
    r"\s+la\s+bao\s+nhieu\s*$",
    r"\s+bao\s+nhieu\s+km\s*$",
    r"\s+bao\s+nhieu\s*$",
    r"\s+là\s+bao\s+nhiêu\s+km\s*$",
    r"\s+là\s+bao\s+nhiêu\s*$",
    r"\s+bao\s+nhiêu\s+km\s*$",
    r"\s+bao\s+nhiêu\s*$",
    r"\s+bao\s+xa\s*$",
    r"\s+mất\s+bao\s+lâu\s*$",
    r"\s+mat\s+bao\s+lau\s*$",
]

# Highly explicit regex patterns to safely extract origin and destination location slots.
# Enforces word boundaries and destination stop pattern constraints.
EXPLICIT_DISTANCE_PATTERNS = [
    # từ A đến B (với stop pattern ở sau) - Hỗ trợ cả có dấu và không dấu hỗn hợp
    r"\b(?:từ|tu)\s+(?P<origin>.+?)\s+(?:đến|tới|den|toi)\s+(?P<destination>.+?)(?:" + DESTINATION_STOP_PATTERN + r"|$)",
    
    # khoảng cách giữa A và B
    r"\b(?:khoảng|khoang)\s+(?:cách|cach)\s+(?:giữa|giua)\s+(?P<origin>.+?)\s+(?:và|va)\s+(?P<destination>.+?)(?:" + DESTINATION_STOP_PATTERN + r"|$)",
    
    # A cách B
    r"\b(?P<origin>.+?)\s+(?:cách|cach)\s+(?P<destination>.+?)(?:" + DESTINATION_STOP_PATTERN + r"|$)",
]

# Verbs/Intents that represent actions rather than actual names.
INTENT_PHRASES_BLACKLIST = {
    "duong di", "đường đi", "duong dan", "đường dẫn", "di toi", "đi tới", 
    "di den", "đi đến", "dan den", "dẫn đến", "dan toi", "dẫn tới", 
    "chi duong", "chỉ đường", "chi dan", "chỉ dẫn", "tim duong", "tìm đường"
}
