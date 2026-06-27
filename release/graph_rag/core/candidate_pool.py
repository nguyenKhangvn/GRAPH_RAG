from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from graph_rag.utils.node_utils import get_node_labels

if TYPE_CHECKING:
    from graph_rag.core.state import NodeItem
    from graph_rag.core.retrieval_policy import RetrievalPolicyInstance


@dataclass
class CandidateScore:
    """Breakdown of different score components for a candidate node."""

    original_score: float
    policy_score: float
    final_score: float
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_score": self.original_score,
            "policy_score": self.policy_score,
            "final_score": self.final_score,
            "reasons": self.reasons,
        }


@dataclass
class CandidatePool:
    """Wraps a list of NodeItem objects with statistics and score breakdowns.

    This serves purely as a container/data object to maintain clean architecture.
    """

    nodes: List[NodeItem]
    query_state: Any
    policy: RetrievalPolicyInstance
    source_breakdown: Dict[str, int] = field(default_factory=dict)
    label_distribution: Dict[str, int] = field(default_factory=dict)
    score_breakdown: Dict[str, CandidateScore] = field(default_factory=dict)

    @classmethod
    def from_nodes(
        cls,
        nodes: List[NodeItem],
        query_state: Any,
        policy: RetrievalPolicyInstance,
        score_breakdown: Optional[Dict[str, CandidateScore]] = None,
    ) -> CandidatePool:
        """Factory method to build a CandidatePool and compute label/source distributions."""
        source_breakdown: Dict[str, int] = {}
        label_distribution: Dict[str, int] = {}

        for node in nodes:
            # Count source types
            src = node.source_type or "unknown"
            source_breakdown[src] = source_breakdown.get(src, 0) + 1

            # Count labels
            labels = get_node_labels(node)

            for lbl in labels:
                label_distribution[lbl] = label_distribution.get(lbl, 0) + 1

        sb = score_breakdown if score_breakdown is not None else {}
        # Populate default scores if not provided
        if not sb:
            for node in nodes:
                sb[node.id] = CandidateScore(
                    original_score=node.score,
                    policy_score=0.0,
                    final_score=node.score,
                    reasons=["Initial retrieval score"],
                )

        return cls(
            nodes=nodes,
            query_state=query_state,
            policy=policy,
            source_breakdown=source_breakdown,
            label_distribution=label_distribution,
            score_breakdown=sb,
        )

    def top_k(self, k: int) -> List[NodeItem]:
        """Returns the top k nodes sorted by final_score descending, without re-ranking."""
        sorted_nodes = sorted(
            self.nodes,
            key=lambda n: self.score_breakdown.get(n.id, CandidateScore(n.score, 0.0, n.score)).final_score,
            reverse=True
        )
        return sorted_nodes[:k]

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "source_breakdown": self.source_breakdown,
            "label_distribution": self.label_distribution,
            "score_breakdowns": {
                node_id: score.to_dict() for node_id, score in self.score_breakdown.items()
            }
        }
