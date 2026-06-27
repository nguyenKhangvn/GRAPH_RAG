"""Web search fallback service for fetching missing attributes.

When the graph database doesn't have specific information (price, address, phone, etc.),
this service searches the web to fill the gap before passing context to the LLM.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cache: query -> (timestamp, results)
_cache: Dict[str, tuple[float, List[Dict[str, Any]]]] = {}
CACHE_TTL = int(os.getenv("WEB_SEARCH_CACHE_TTL", "3600"))  # 1 hour


class WebSearchService:
    """Searches the web for missing entity attributes."""

    # Attributes that can be fetched from the web
    FETCHABLE_ATTRIBUTES = {
        "address", "phone", "opening_hours", "ticket_price",
        "price", "price_range", "email", "website",
    }

    # Vietnamese search templates for each attribute
    _QUERY_TEMPLATES = {
        "address": "{entity} địa chỉ {location}",
        "phone": "{entity} số điện thoại liên hệ",
        "opening_hours": "{entity} giờ mở cửa thời gian hoạt động",
        "ticket_price": "{entity} giá vé {year}",
        "price": "{entity} giá bao nhiêu",
        "price_range": "{entity} giá phòng",
        "email": "{entity} email liên hệ",
        "website": "{entity} website chính thức",
    }

    def __init__(self, api_key: Optional[str] = None, provider: str = "duckduckgo"):
        self.api_key = api_key or os.getenv("WEB_SEARCH_API_KEY", "")
        self.provider = provider
        self._client = None

    def _get_client(self):
        """Lazy-initialize the search client."""
        if self._client is None:
            if self.provider == "tavily":
                try:
                    from tavily import TavilyClient
                    self._client = TavilyClient(api_key=self.api_key)
                except ImportError:
                    logger.error("tavily-python not installed. Run: pip install tavily-python")
                    return None
                except (ValueError, RuntimeError, OSError) as e:
                    logger.error("Failed to init Tavily client: %s", e)
                    return None
            elif self.provider == "duckduckgo":
                try:
                    from duckduckgo_search import DDGS
                    self._client = DDGS()
                except ImportError:
                    logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
                    return None
                except (ValueError, RuntimeError, OSError) as e:
                    logger.error("Failed to init DuckDuckGo client: %s", e)
                    return None
            else:
                logger.error("Unsupported web search provider: %s", self.provider)
                return None
        return self._client

    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search the web for information.

        Args:
            query: Search query string
            max_results: Maximum number of results to return

        Returns:
            List of dicts with keys: title, snippet, url
        """
        # Check cache
        cache_key = f"{query}:{max_results}"
        if cache_key in _cache:
            ts, results = _cache[cache_key]
            if time.time() - ts < CACHE_TTL:
                logger.info("Web search cache hit for: %s", query[:50])
                return results

        client = self._get_client()
        if not client:
            return []

        try:
            if self.provider == "tavily":
                response = client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="basic",
                    include_answer=True,
                )
                results = []
                # Tavily returns an "answer" field and "results" list
                if response.get("answer"):
                    results.append({
                        "title": "Tóm tắt",
                        "snippet": response["answer"],
                        "url": "",
                        "source": "tavily_answer",
                    })
                for r in response.get("results", [])[:max_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("content", ""),
                        "url": r.get("url", ""),
                        "source": "tavily",
                    })

            elif self.provider == "duckduckgo":
                raw = client.text(query, max_results=max_results)
                results = []
                for r in raw:
                    results.append({
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                        "source": "duckduckgo",
                    })
            else:
                results = []

            # Cache results
            _cache[cache_key] = (time.time(), results)
            logger.info("Web search returned %d results for: %s", len(results), query[:50])
            return results

        except (ValueError, RuntimeError, OSError) as e:
            logger.error("Web search failed for '%s': %s", query[:50], e)
            return []

    def search_entity_attribute(
        self,
        entity: str,
        attribute: str,
        location: str = "",
        year: str = "2025",
    ) -> List[Dict[str, Any]]:
        """Search for a specific attribute of an entity.

        Args:
            entity: Entity name (e.g., "Kỳ Co", "Eo Gió")
            attribute: Attribute to search for (e.g., "ticket_price", "address")
            location: Optional location context
            year: Year for price queries

        Returns:
            List of search results
        """
        template = self._QUERY_TEMPLATES.get(attribute, "{entity} {attribute}")
        query = template.format(
            entity=entity,
            attribute=attribute.replace("_", " "),
            location=location,
            year=year,
        )
        return self.search(query)

    def is_fetchable(self, attribute: str) -> bool:
        """Check if an attribute can be fetched from the web."""
        return attribute.lower() in self.FETCHABLE_ATTRIBUTES

    @staticmethod
    def format_as_context(
        results: List[Dict[str, Any]],
        entity: str,
        attribute: str,
    ) -> List[str]:
        """Format web search results as context lines for LLM.

        Args:
            results: Search results from search_entity_attribute()
            entity: Entity name
            attribute: The attribute being searched

        Returns:
            List of context lines compatible with the pipeline context format
        """
        if not results:
            return []

        lines = []
        attr_vi = {
            "address": "địa chỉ",
            "phone": "số điện thoại",
            "opening_hours": "giờ mở cửa",
            "ticket_price": "giá vé",
            "price": "giá",
            "price_range": "giá phòng",
            "email": "email",
            "website": "website",
        }.get(attribute, attribute)

        for r in results:
            snippet = r.get("snippet", "").strip()
            if not snippet:
                continue
            # Truncate long snippets
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
            url = r.get("url", "")
            source_tag = f"(nguồn: {url})" if url else ""
            lines.append(f"[Web] {entity} — {attr_vi}: {snippet} {source_tag}".strip())

        return lines
