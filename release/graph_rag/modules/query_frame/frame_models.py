from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Mention:
    text: str
    role: str
    type_hint: str = ""
    groundability: str = "groundable"
    required: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "role": self.role,
            "type_hint": self.type_hint,
            "groundability": self.groundability,
            "required": self.required,
            "confidence": self.confidence,
        }


@dataclass
class RetrievalPlan:
    mode: str = "single_anchor"
    anchors: List[Mention] = field(default_factory=list)
    candidate_entities: List[Mention] = field(default_factory=list)
    required_attributes: List[str] = field(default_factory=list)
    required_relations: List[str] = field(default_factory=list)
    context_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "anchors": [item.to_dict() for item in self.anchors],
            "candidate_entities": [item.to_dict() for item in self.candidate_entities],
            "required_attributes": list(self.required_attributes),
            "required_relations": list(self.required_relations),
            "context_policy": dict(self.context_policy),
        }


@dataclass
class QueryFrame:
    query_operator: str = "fact_lookup"
    answer_mode: str = ""
    question_type: str = ""
    location_scope: str = ""
    groundable_mentions: List[Mention] = field(default_factory=list)
    candidate_entities: List[Mention] = field(default_factory=list)
    comparison_subjects: List[Mention] = field(default_factory=list)
    answer_set_variables: List[Dict[str, Any]] = field(default_factory=list)
    requested_attributes: List[str] = field(default_factory=list)
    requested_relations: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    non_groundable_phrases: List[str] = field(default_factory=list)
    retrieval_plan: RetrievalPlan = field(default_factory=RetrievalPlan)
    confidence: float = 0.0
    valid: bool = False
    validation_errors: List[str] = field(default_factory=list)
    fallback_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_operator": self.query_operator,
            "answer_mode": self.answer_mode,
            "question_type": self.question_type,
            "location_scope": self.location_scope,
            "groundable_mentions": [item.to_dict() for item in self.groundable_mentions],
            "candidate_entities": [item.to_dict() for item in self.candidate_entities],
            "comparison_subjects": [item.to_dict() for item in self.comparison_subjects],
            "answer_set_variables": list(self.answer_set_variables),
            "requested_attributes": list(self.requested_attributes),
            "requested_relations": list(self.requested_relations),
            "constraints": dict(self.constraints),
            "non_groundable_phrases": list(self.non_groundable_phrases),
            "retrieval_plan": self.retrieval_plan.to_dict(),
            "confidence": self.confidence,
            "valid": self.valid,
            "validation_errors": list(self.validation_errors),
            "fallback_reason": self.fallback_reason,
        }

