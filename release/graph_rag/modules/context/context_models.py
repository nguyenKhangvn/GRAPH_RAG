from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MainEntitySelection:
    name: str = ""
    node_id: str = ""
    labels: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = "unresolved"
    hard_keep_enabled: bool = False
    query_mode: str = "unresolved"
    confidence_components: Dict[str, float] = field(default_factory=dict)

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "node_id": self.node_id,
            "labels": self.labels,
            "confidence": round(float(self.confidence or 0.0), 3),
            "reason": self.reason,
            "hard_keep_enabled": bool(self.hard_keep_enabled),
            "query_mode": self.query_mode,
            "confidence_components": {
                key: round(float(value or 0.0), 3)
                for key, value in (self.confidence_components or {}).items()
            },
        }


@dataclass
class ContextItem:
    id: str
    kind: str
    text: str
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    source_label: Optional[str] = None
    target_label: Optional[str] = None
    relation_type: Optional[str] = None
    retrieval_source: str = "graph_traversal"
    score: Optional[float] = None
    must_keep: bool = False
    selection_reason: Optional[str] = None
    confidence: Optional[float] = None

    def to_debug_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "relation_type": self.relation_type,
            "must_keep": self.must_keep,
            "selection_reason": self.selection_reason,
            "confidence": self.confidence,
        }


@dataclass
class ContextOrganizationResult:
    final_context: str
    structural_items: List[ContextItem]
    textual_items: List[ContextItem]
    kept_structural_items: List[ContextItem]
    selected_textual_context: str
    main_entity: MainEntitySelection
    debug: Dict[str, Any] = field(default_factory=dict)
