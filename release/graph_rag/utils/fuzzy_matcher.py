"""Vietnamese fuzzy matching with graph-based dynamic vocabulary.

Replaces hardcoded Vietnamese synonym dictionaries with:
1. Dynamic vocabulary extracted from Neo4j graph entities
2. Concept-based matching (entity name -> concepts -> lookup)
3. Fuzzy matching for diacritics variations and partial matches
4. Stopword removal before matching

Usage::

    from graph_rag.utils.fuzzy_matcher import GraphVocabulary, VietnameseFuzzyMatcher

    vocab = GraphVocabulary(driver)
    matcher = VietnameseFuzzyMatcher(vocab)

    # Match query text to entity names
    entities = matcher.match_query_concepts("có biển Quy Nhơn")

    # Expand normalized entity name with diacritics variants
    variants = matcher.expand_entity_name("bien Quy Nhon")
"""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Set

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

from graph_rag.utils.text import normalize_text, remove_vietnamese_stopwords

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@lru_cache(maxsize=1)
def _load_concept_prefixes() -> Dict[str, str]:
    """Load concept prefix mapping from vietnamese_patterns.json."""
    try:
        path = _CONFIG_DIR / "vietnamese_patterns.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("concept_prefixes", {})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        # Minimal fallback if config file missing
        return {
            "biển": "natural_feature",
            "núi": "natural_feature",
            "hồ": "natural_feature",
            "chùa": "spiritual",
            "nhà hàng": "food_service",
            "khách sạn": "accommodation",
        }


@lru_cache(maxsize=1)
def _load_fuzzy_config() -> dict:
    """Load fuzzy matching config from vietnamese_patterns.json."""
    defaults = {
        "default_threshold": 70,
        "token_match_threshold": 85,
        "max_results": 10,
        "min_token_length": 2,
    }
    try:
        path = _CONFIG_DIR / "vietnamese_patterns.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        defaults.update(data.get("fuzzy_matching", {}))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return defaults


# ---------------------------------------------------------------------------
# Fuzzy scoring — use rapidfuzz if installed, difflib otherwise
# ---------------------------------------------------------------------------

def _fuzzy_ratio(a: str, b: str) -> int:
    """Fuzzy match ratio between two strings (0-100)."""
    try:
        from rapidfuzz.fuzz import ratio  # type: ignore
        return ratio(a, b)
    except ImportError:
        return int(SequenceMatcher(None, a, b).ratio() * 100)


def _partial_ratio(a: str, b: str) -> int:
    """Best substring fuzzy ratio (0-100)."""
    try:
        from rapidfuzz.fuzz import partial_ratio  # type: ignore
        return partial_ratio(a, b)
    except ImportError:
        if not a or not b:
            return 0
        if len(a) > len(b):
            a, b = b, a
        best = 0
        for i in range(len(b) - len(a) + 1):
            sub = b[i : i + len(a)]
            score = int(SequenceMatcher(None, a, sub).ratio() * 100)
            best = max(best, score)
        return best


# ---------------------------------------------------------------------------
# GraphVocabulary — lazy-loaded from Neo4j
# ---------------------------------------------------------------------------


class GraphVocabulary:
    """Dynamic vocabulary extracted from Neo4j graph entities.

    Lazily loads all entity names on first access and builds:
    - ``vocabulary``: {label: [entity_names]}
    - ``concept_map``: {concept_keyword: [entity_names]}
    - ``all_names``: flat list of all entity names

    The vocabulary is cached in memory after the first load.  Call
    :meth:`reload` to refresh from the graph.
    """

    def __init__(self, driver) -> None:
        self.driver = driver
        self._vocabulary: Dict[str, List[str]] = {}
        self._concept_map: Dict[str, List[str]] = {}
        self._all_names: List[str] = []
        self._loaded: bool = False

    # -- public API -----------------------------------------------------------

    def ensure_loaded(self) -> None:
        """Load vocabulary from graph if not already loaded."""
        if not self._loaded:
            self._load_from_graph()
            self._loaded = True

    def reload(self) -> None:
        """Force reload from graph."""
        self._loaded = False
        self._vocabulary.clear()
        self._concept_map.clear()
        self._all_names.clear()
        self.ensure_loaded()

    @property
    def vocabulary(self) -> Dict[str, List[str]]:
        self.ensure_loaded()
        return self._vocabulary

    @property
    def concept_map(self) -> Dict[str, List[str]]:
        self.ensure_loaded()
        return self._concept_map

    @property
    def all_names(self) -> List[str]:
        self.ensure_loaded()
        return self._all_names

    # -- internals ------------------------------------------------------------

    def _load_from_graph(self) -> None:
        """Query Neo4j for all entity names and build concept index."""
        vocabulary: Dict[str, List[str]] = {}
        concept_map: Dict[str, List[str]] = {}

        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (n) WHERE n.name IS NOT NULL "
                    "RETURN DISTINCT n.name AS name, labels(n)[0] AS label"
                )
                for row in result:
                    name = str(row.get("name") or "").strip()
                    label = str(row.get("label") or "Unknown").strip()
                    if not name or len(name) < 2:
                        continue
                    vocabulary.setdefault(label, []).append(name)

                    for concept in self._extract_concepts(name):
                        concept_map.setdefault(concept, []).append(name)

            logger.info(
                "GraphVocabulary loaded: %d entities, %d labels, %d concepts",
                sum(len(v) for v in vocabulary.values()),
                len(vocabulary),
                len(concept_map),
            )
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.warning("GraphVocabulary load failed: %s", exc)

        self._vocabulary = vocabulary
        self._concept_map = concept_map
        self._all_names = [n for names in vocabulary.values() for n in names]

    @staticmethod
    def _extract_concepts(entity_name: str) -> List[str]:
        """Extract concept keywords from a graph entity name.

        Examples::

            "Biển Quy Nhơn"   -> ["biển", "quy nhơn"]
            "Chùa Một Cột"    -> ["chùa", "một cột"]
            "Nhà hàng Hương"  -> ["nhà hàng", "hương"]
            "Phở"             -> ["phở"]

        Note: Stopwords are NOT removed from entity names because they are
        proper nouns — "một" in "Chùa Một Cột" is part of the name.
        """
        name = str(entity_name or "").strip()
        if not name:
            return []

        norm = normalize_text(name, strip_punct=True)
        if not norm:
            return []

        # Do NOT remove stopwords from entity names — they are proper nouns.
        tokens = norm.split()
        concepts: List[str] = []
        prefix_map = _load_concept_prefixes()

        # Check multi-word prefixes first (longest first)
        matched_prefix_len = 0
        for prefix in sorted(prefix_map, key=len, reverse=True):
            prefix_norm = normalize_text(prefix, strip_punct=True)
            if norm.startswith(prefix_norm):
                remaining = norm[len(prefix_norm) :].strip()
                concepts.append(prefix_norm)
                matched_prefix_len = len(prefix_norm)
                if remaining:
                    concepts.append(remaining)
                break

        # If no prefix matched, split into first-token + rest
        if not matched_prefix_len and tokens:
            concepts.append(tokens[0])
            if len(tokens) > 1:
                concepts.append(" ".join(tokens[1:]))

        # Always include full normalized name as a concept
        if norm not in concepts:
            concepts.append(norm)

        min_len = _load_fuzzy_config().get("min_token_length", 2)
        return [c for c in concepts if len(c) >= min_len]


# ---------------------------------------------------------------------------
# VietnameseFuzzyMatcher — the main entry point
# ---------------------------------------------------------------------------


class VietnameseFuzzyMatcher:
    """Vietnamese-aware fuzzy entity matcher using graph vocabulary.

    Matches query terms to graph entities by:
    1. Normalizing and removing stopwords
    2. Looking up concept -> entity mapping (exact)
    3. Fuzzy matching against concepts and entity names
    4. Returning ranked matches above threshold
    """

    def __init__(self, vocabulary: GraphVocabulary) -> None:
        self.vocabulary = vocabulary

    def match_query_concepts(
        self,
        query: str,
        threshold: int | None = None,
        max_results: int | None = None,
    ) -> List[str]:
        """Match query text to graph entity names via concept lookup + fuzzy.

        Args:
            query: User query (e.g. "có biển Quy Nhơn").
            threshold: Minimum fuzzy score (0-100).  Default from config.
            max_results: Maximum results.  Default from config.

        Returns:
            List of matched entity names from the graph.
        """
        self.vocabulary.ensure_loaded()
        cfg = _load_fuzzy_config()
        if threshold is None:
            threshold = cfg.get("default_threshold", 70)
        if max_results is None:
            max_results = cfg.get("max_results", 10)

        q = str(query or "").strip()
        if not q:
            return []

        q_norm = normalize_text(q, strip_punct=True)
        q_cleaned = remove_vietnamese_stopwords(q_norm)
        if not q_cleaned:
            q_cleaned = q_norm

        tokens = q_cleaned.split()
        if not tokens:
            return []

        results: List[str] = []
        seen: Set[str] = set()

        # 1. Exact concept lookup — single tokens
        for token in tokens:
            if token in self.vocabulary.concept_map:
                for entity in self.vocabulary.concept_map[token]:
                    if entity not in seen:
                        seen.add(entity)
                        results.append(entity)

        # 2. Multi-token concept lookup (phrases)
        if len(tokens) > 1:
            for i in range(len(tokens)):
                for j in range(i + 2, min(i + 5, len(tokens) + 1)):
                    phrase = " ".join(tokens[i:j])
                    if phrase in self.vocabulary.concept_map:
                        for entity in self.vocabulary.concept_map[phrase]:
                            if entity not in seen:
                                seen.add(entity)
                                results.append(entity)

        # 3. Fuzzy match against concepts (if no exact matches)
        if not results:
            scored: List[tuple[int, str]] = []
            for concept, entities in self.vocabulary.concept_map.items():
                for token in tokens:
                    score = _fuzzy_ratio(token, concept)
                    if score >= threshold:
                        for entity in entities:
                            if entity not in seen:
                                seen.add(entity)
                                scored.append((score, entity))
            scored.sort(key=lambda x: -x[0])
            results = [name for _, name in scored]

        # 4. Fuzzy match against all entity names (last resort)
        if not results:
            for name in self.vocabulary.all_names:
                if name in seen:
                    continue
                name_norm = normalize_text(name, strip_punct=True)
                for token in tokens:
                    score = _partial_ratio(token, name_norm)
                    if score >= threshold:
                        seen.add(name)
                        results.append(name)
                        break

        return results[:max_results]

    def expand_entity_name(
        self,
        entity_name: str,
        threshold: int | None = None,
    ) -> List[str]:
        """Expand entity name with diacritics variants from graph vocabulary.

        Replaces the old ``_expand_vietnamese_synonyms`` approach.  Instead of
        a hardcoded synonym dictionary, this dynamically finds graph entities
        whose normalized form matches the input.

        Args:
            entity_name: Entity name to expand (e.g. "bien Quy Nhon").
            threshold: Minimum fuzzy score.  Default from config.

        Returns:
            List of expanded name variants with diacritics from graph.
        """
        self.vocabulary.ensure_loaded()
        cfg = _load_fuzzy_config()
        if threshold is None:
            threshold = cfg.get("default_threshold", 70)

        raw = str(entity_name or "").strip()
        if not raw:
            return []

        norm = normalize_text(raw, strip_punct=True)
        if not norm:
            return []

        variants: List[str] = []
        seen: Set[str] = set()

        # 1. Exact normalized match — graph entity has correct diacritics
        for name in self.vocabulary.all_names:
            name_norm = normalize_text(name, strip_punct=True)
            if name_norm == norm and name != raw:
                if name not in seen:
                    seen.add(name)
                    variants.append(name)

        # 2. Substring containment match
        if not variants:
            for name in self.vocabulary.all_names:
                if name in seen or name == raw:
                    continue
                name_norm = normalize_text(name, strip_punct=True)
                if norm in name_norm or name_norm in norm:
                    score = _fuzzy_ratio(norm, name_norm)
                    if score >= threshold:
                        seen.add(name)
                        variants.append(name)

        # 3. Token-level fuzzy match (for partial diacritics differences)
        if not variants:
            token_threshold = cfg.get("token_match_threshold", 85)
            tokens = norm.split()
            for name in self.vocabulary.all_names:
                if name in seen or name == raw:
                    continue
                name_norm = normalize_text(name, strip_punct=True)
                name_tokens = name_norm.split()

                # Check if any query token fuzzy-matches a name token
                matched = False
                for q_tok in tokens:
                    for n_tok in name_tokens:
                        if q_tok == n_tok:
                            continue
                        if _fuzzy_ratio(q_tok, n_tok) >= token_threshold:
                            overall = _fuzzy_ratio(norm, name_norm)
                            if overall >= threshold:
                                seen.add(name)
                                variants.append(name)
                                matched = True
                                break
                    if matched:
                        break

        return variants[:10]
