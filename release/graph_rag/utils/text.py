import re
import unicodedata
from typing import Set


# Vietnamese stopwords — common function words removed before entity matching.
# These are query-level noise words, not entity-level concepts.
VIETNAMESE_STOPWORDS: Set[str] = {
    # Determiners / demonstratives
    "có", "gần", "là", "ở", "tại", "của", "và", "hoặc", "thì",
    "nào", "đâu", "gì", "ai", "như", "thế", "này", "kia", "đó",
    # Pronouns
    "tôi", "bạn", "mình", "chúng", "họ", "nó", "cho", "tôi",
    # Auxiliaries / modals
    # NOTE: "chưa" excluded — normalizes to "chua" which collides with
    # "chùa" (pagoda), an important concept prefix.
    "muốn", "cần", "thể", "phải", "nên", "được", "sẽ", "đang", "đã",
    "rồi", "không",
    # Quantifiers
    "một", "hai", "các", "những", "nhiều", "ít", "mấy",
    # Prepositions
    "với", "từ", "đến", "về", "qua", "sau", "trước", "trong", "ngoài",
    # Adverbs
    "rất", "lắm", "quá", "thật", "cũng", "hay",
    # Conjunctions
    "nhưng", "mà", "nếu", "khi",
    # Question words (already partially covered above)
    "bao", "nhiêu", "ra sao",
    # Common query verbs
    "biết", "tìm", "xem",
    # Common query phrases (single tokens)
    "giới thiệu", "cho tôi", "có gì",
}

# Lazily-populated expanded stopwords (includes normalized forms).
_NORMALIZED_STOPWORDS: Set[str] | None = None


def remove_vietnamese_stopwords(text: str) -> str:
    """Remove Vietnamese stopwords from text.

    Tokenizes on whitespace and drops tokens present in VIETNAMESE_STOPWORDS.
    Works with both diacritics text (``"có biển"``) and normalized text
    (``"co bien"``) by also checking a normalized copy of each stopword.
    Preserves order of remaining tokens.
    """
    if not text:
        return ""
    # Lazily build an expanded set that includes normalized forms
    global _NORMALIZED_STOPWORDS
    if _NORMALIZED_STOPWORDS is None:
        _NORMALIZED_STOPWORDS = set(VIETNAMESE_STOPWORDS)
        for w in VIETNAMESE_STOPWORDS:
            _NORMALIZED_STOPWORDS.add(normalize_text(w, strip_punct=True))

    words = str(text).lower().split()
    return " ".join(w for w in words if w not in _NORMALIZED_STOPWORDS)


def normalize_text(text: str, *, strip_punct: bool = False) -> str:
    """Chuẩn hoá tiếng Việt: bỏ dấu, lowercase, collapse whitespace.

    Args:
        text: Input string.
        strip_punct: Nếu True, loại bỏ ký tự không phải a-z/0-9/space.
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFKD", str(text))
    norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
    norm = norm.replace("đ", "d").replace("Đ", "D")
    norm = norm.lower()
    if strip_punct:
        norm = re.sub(r"[^a-z0-9\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


def token_overlap(left: str, right: str) -> float:
    """Calculate token overlap ratio between two strings.

    Returns the fraction of left tokens (len >= 2) that also appear in right.
    """
    left_tokens = {tok for tok in left.split() if len(tok) >= 2}
    right_tokens = {tok for tok in right.split() if len(tok) >= 2}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


def clean_query_format(query: str) -> str:
    """Strip JSON-wrapping, string quotes, and pseudo-keys (e.g. 'question:') from query."""
    query = str(query or "").strip()

    # Handle outer quotes if any, e.g. '"..."' or "'...'"
    if len(query) >= 2 and (
        (query.startswith('"') and query.endswith('"'))
        or (query.startswith("'") and query.endswith("'"))
    ):
        query = query[1:-1].strip()

    # 1. Valid JSON object check
    if query.startswith("{") and query.endswith("}"):
        try:
            import json
            parsed = json.loads(query)
            if isinstance(parsed, dict):
                for key in ["question", "query", "content", "text"]:
                    if key in parsed and isinstance(parsed[key], str):
                        return parsed[key].strip()
        except (OSError, FileNotFoundError, json.JSONDecodeError):
            pass

    # 2. Pseudo JSON/Key-Value pattern check (e.g., "question": "..." or 'question': '...')
    m = re.search(r'''(?i)(?:\bquestion\b|\bquery\b)\s*["']?\s*:\s*["'](.+?)["']\s*,?\s*$''', query)
    if m:
        return m.group(1).strip()

    return query

