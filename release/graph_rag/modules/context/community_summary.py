from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from graph_rag.core.intents import IntentType
from graph_rag.utils.node_utils import seed_name
from graph_rag.utils.text import normalize_text, token_overlap


OPEN_ENDED_MARKERS = (
    "phan tich",
    "chien luoc",
    "loi the",
    "tiem nang",
    "phu hop voi ai",
    "goi y tong quan",
    "vi sao",
    "tai sao",
    "vai tro",
    "tam quan trong",
    "y nghia",
    "gia tri",
    "de xuat",
)


@dataclass
class CommunitySummary:
    community_id: str
    summary: str
    source_node_ids: List[str] = field(default_factory=list)
    covered_relation_types: List[str] = field(default_factory=list)
    region: str = ""
    topic: str = ""
    keywords: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CommunitySummary":
        return cls(
            community_id=str(data.get("community_id") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            source_node_ids=[
                str(item).strip() for item in data.get("source_node_ids", []) if str(item).strip()
            ],
            covered_relation_types=[
                str(item).strip()
                for item in data.get("covered_relation_types", [])
                if str(item).strip()
            ],
            region=str(data.get("region") or "").strip(),
            topic=str(data.get("topic") or "").strip(),
            keywords=[str(item).strip() for item in data.get("keywords", []) if str(item).strip()],
        )

    def to_debug_dict(self, score: float) -> Dict[str, Any]:
        return {
            "community_id": self.community_id,
            "score": round(float(score or 0.0), 3),
            "region": self.region,
            "topic": self.topic,
            "source_node_count": len(self.source_node_ids),
            "covered_relation_types": self.covered_relation_types,
        }


@dataclass
class CommunitySummaryResult:
    summaries: List[CommunitySummary]
    debug: Dict[str, Any]

    def render(self) -> str:
        if not self.summaries:
            return ""
        lines = ["[COMMUNITY SUMMARY - OFFLINE]"]
        for summary in self.summaries:
            provenance = []
            if summary.community_id:
                provenance.append(f"community_id={summary.community_id}")
            if summary.region:
                provenance.append(f"region={summary.region}")
            if summary.topic:
                provenance.append(f"topic={summary.topic}")
            if summary.covered_relation_types:
                provenance.append("relations=" + ",".join(summary.covered_relation_types[:4]))
            suffix = f" ({'; '.join(provenance)})" if provenance else ""
            lines.append(f"- {summary.summary}{suffix}")
        return "\n".join(lines)


class CommunitySummaryRetriever:
    """Offline community-summary retriever.

    This component never calls an LLM online. It only reads precomputed summary
    documents and appends the best matches for open-ended questions.
    """

    def __init__(
        self,
        path: str | Path,
        normalize_text: Callable[[str], str] | None = None,
        enabled: bool = False,
        top_k: int = 2,
        min_score: float = 0.18,
    ) -> None:
        self.path = Path(path)
        self.normalize_text = normalize_text or self._default_normalize
        self.enabled = bool(enabled)
        self.top_k = max(0, int(top_k or 0))
        self.min_score = float(min_score or 0.0)
        self._cache: List[CommunitySummary] | None = None

    def retrieve(
        self,
        query_text: str,
        primary_intent: str,
        seeds: Sequence[Any] | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> CommunitySummaryResult:
        debug: Dict[str, Any] = {
            "community_summary_enabled": self.enabled,
            "community_summary_applied": False,
            "community_summary_error": "",
            "community_summary_input_count": 0,
            "community_summary_output_count": 0,
        }
        if not self.enabled:
            debug["community_summary_error"] = "disabled"
            return CommunitySummaryResult([], debug)
        if self.top_k <= 0:
            debug["community_summary_error"] = "top_k_zero"
            return CommunitySummaryResult([], debug)
        if not self._should_activate(query_text, primary_intent, metadata or {}):
            debug["community_summary_error"] = "not_open_ended"
            return CommunitySummaryResult([], debug)

        try:
            summaries = self._load()
            debug["community_summary_input_count"] = len(summaries)
            scored = [
                (self._score(summary, query_text, primary_intent, seeds or []), idx, summary)
                for idx, summary in enumerate(summaries)
                if summary.summary
            ]
            selected = [
                (score, summary)
                for score, _, summary in sorted(scored, key=lambda item: (-item[0], item[1]))
                if score >= self.min_score
            ][: self.top_k]
            debug["community_summary_output_count"] = len(selected)
            debug["community_summary_applied"] = bool(selected)
            debug["selected_communities"] = [
                summary.to_debug_dict(score) for score, summary in selected
            ]
            if not selected:
                debug["community_summary_error"] = "no_match"
            return CommunitySummaryResult([summary for _, summary in selected], debug)
        except (ValueError, TypeError, KeyError) as exc:
            debug["community_summary_error"] = str(exc)[:200]
            return CommunitySummaryResult([], debug)

    def _load(self) -> List[CommunitySummary]:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            self._cache = []
            return self._cache
        data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("communities", [])
        self._cache = [
            CommunitySummary.from_dict(item)
            for item in data
            if isinstance(item, dict) and str(item.get("summary") or "").strip()
        ]
        return self._cache

    def _should_activate(self, query_text: str, primary_intent: str, metadata: Dict[str, Any]) -> bool:
        answer_mode = str(metadata.get("answer_mode") or "").lower()
        question_type = str(metadata.get("question_type") or "")
        query_norm = self.normalize_text(query_text)
        if answer_mode == "open_analysis" or question_type == "Open-Ended":
            return True
        if str(primary_intent or "").upper() in {IntentType.DISCOVERY, IntentType.TOUR_PLAN}:
            return any(marker in query_norm for marker in OPEN_ENDED_MARKERS)
        return any(marker in query_norm for marker in OPEN_ENDED_MARKERS)

    def _score(
        self,
        summary: CommunitySummary,
        query_text: str,
        primary_intent: str,
        seeds: Sequence[Any],
    ) -> float:
        query_norm = self.normalize_text(query_text)
        seed_text = " ".join(seed_name(seed) for seed in seeds or [])
        seed_norm = self.normalize_text(seed_text)
        keyword_text = " ".join(summary.keywords + [summary.region, summary.topic, summary.summary])
        keyword_norm = self.normalize_text(keyword_text)

        score = 0.0
        score += 0.55 * token_overlap(query_norm, keyword_norm)
        score += 0.25 * token_overlap(seed_norm, keyword_norm)
        if summary.region and self.normalize_text(summary.region) in query_norm:
            score += 0.12
        if self._topic_matches_intent(summary.topic, primary_intent):
            score += 0.08
        return min(score, 1.0)

    def _topic_matches_intent(self, topic: str, primary_intent: str) -> bool:
        topic_norm = self.normalize_text(topic)
        intent = str(primary_intent or "").upper()
        if intent == IntentType.ACCOMMODATION:
            return any(token in topic_norm for token in ("accommodation", "hotel", "stay", "luu tru"))
        if intent == IntentType.FOOD:
            return any(token in topic_norm for token in ("food", "restaurant", "am thuc"))
        if intent == IntentType.TOURISM:
            return any(token in topic_norm for token in ("tourism", "heritage", "beach", "culture"))
        if intent in {IntentType.DISCOVERY, IntentType.TOUR_PLAN}:
            return True
        return False

    def _default_normalize(self, text: str) -> str:
        return normalize_text(text, strip_punct=True)
