from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Any
from graph_rag.core.intents import IntentType
from graph_rag.core.state import QuestionShape
from graph_rag.utils.text import normalize_text

if TYPE_CHECKING:
    pass

class RetrievalPolicyInstance:
    def __init__(
        self,
        primary_labels: List[str],
        allowed_labels: List[str],
        relax_labels: List[str],
        blocked_labels: List[str],
        context_budget: Dict[str, float],
    ):
        self.primary_labels = primary_labels
        self.allowed_labels = allowed_labels
        self.relax_labels = relax_labels
        self.blocked_labels = blocked_labels
        self.context_budget = context_budget

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_labels": self.primary_labels,
            "allowed_labels": self.allowed_labels,
            "relax_labels": self.relax_labels,
            "blocked_labels": self.blocked_labels,
            "context_budget": self.context_budget,
        }


class RetrievalPolicy:
    _COMPANION_LABELS: Dict[str, List[str]] = {
        "Restaurant": ["Dish", "Specialty"],
        "Accommodation": [],
        "TouristAttraction": ["Tour"],
        "Event": [],
        "Dish": ["Restaurant", "Specialty"],
        "Specialty": ["Dish", "Restaurant"],
    }

    BASE_POLICIES = {
        IntentType.ACCOMMODATION: {
            "primary_labels": ["Accommodation"],
            "allowed_labels": ["Accommodation", "TouristAttraction", "Restaurant", "Location"],
            "relax_labels": ["TouristAttraction"],
            "blocked_labels": ["Event", "Tour", "Dish"],
            "context_budget": {"Accommodation": 0.70, "TouristAttraction": 0.15, "Restaurant": 0.10, "Location": 0.05}
        },
        IntentType.FOOD: {
            "primary_labels": ["Restaurant", "Dish", "Specialty"],
            "allowed_labels": ["Restaurant", "Dish", "Specialty", "TouristAttraction", "Location"],
            "relax_labels": ["TouristAttraction"],
            "blocked_labels": ["Accommodation", "Event", "Tour"],
            "context_budget": {"Restaurant": 0.40, "Dish": 0.25, "Specialty": 0.15, "TouristAttraction": 0.10, "Location": 0.10}
        },
        IntentType.TOURISM: {
            "primary_labels": ["TouristAttraction"],
            "allowed_labels": ["TouristAttraction", "Event", "Restaurant", "Accommodation", "Tour", "Location"],
            "relax_labels": ["Event", "Tour", "Restaurant", "Accommodation"],
            "blocked_labels": [],
            "context_budget": {"TouristAttraction": 0.60, "Event": 0.10, "Restaurant": 0.10, "Accommodation": 0.10, "Location": 0.10}
        },
        IntentType.EVENT: {
            "primary_labels": ["Event"],
            "allowed_labels": ["Event", "TouristAttraction", "Location"],
            "relax_labels": ["TouristAttraction"],
            "blocked_labels": ["Restaurant", "Dish", "Accommodation", "Tour"],
            "context_budget": {"Event": 0.70, "TouristAttraction": 0.20, "Location": 0.10}
        },
        IntentType.TOUR_PLAN: {
            "primary_labels": ["TouristAttraction", "Restaurant", "Accommodation"],
            "allowed_labels": ["TouristAttraction", "Restaurant", "Dish", "Accommodation", "Tour", "Event", "Location"],
            "relax_labels": ["TouristAttraction", "Restaurant", "Accommodation", "Tour", "Event"],
            "blocked_labels": [],
            "context_budget": {"TouristAttraction": 0.35, "Event": 0.15, "Dish": 0.10, "Restaurant": 0.15, "Accommodation": 0.15, "Location": 0.10}
        },
        IntentType.DISTANCE: {
            "primary_labels": ["TouristAttraction", "Restaurant", "Accommodation", "Event", "Location"],
            "allowed_labels": ["TouristAttraction", "Restaurant", "Accommodation", "Event", "Location"],
            "relax_labels": ["TouristAttraction", "Location"],
            "blocked_labels": [],
            "context_budget": {"TouristAttraction": 0.30, "Restaurant": 0.20, "Accommodation": 0.20, "Location": 0.30}
        },
        IntentType.DISCOVERY: {
            "primary_labels": ["TouristAttraction", "Restaurant", "Accommodation", "Event", "Tour", "Dish"],
            "allowed_labels": ["TouristAttraction", "Restaurant", "Dish", "Accommodation", "Tour", "Event", "Location", "TravelInfo"],
            "relax_labels": ["TouristAttraction", "Restaurant", "Accommodation", "Event", "TravelInfo"],
            "blocked_labels": [],
            "context_budget": {"TouristAttraction": 0.35, "Restaurant": 0.12, "Accommodation": 0.12, "Event": 0.08, "Tour": 0.08, "Location": 0.10, "TravelInfo": 0.15}
        },
        IntentType.ENTITY_FACT: {
            "primary_labels": ["TouristAttraction", "Restaurant", "Accommodation", "Event", "Tour", "Dish"],
            "allowed_labels": ["TouristAttraction", "Restaurant", "Dish", "Accommodation", "Tour", "Event", "Location", "TravelInfo"],
            "relax_labels": ["TouristAttraction", "Restaurant", "Accommodation", "Event", "TravelInfo"],
            "blocked_labels": [],
            "context_budget": {"TouristAttraction": 0.35, "Restaurant": 0.12, "Accommodation": 0.12, "Event": 0.08, "Tour": 0.08, "Location": 0.10, "TravelInfo": 0.15}
        },
        IntentType.TRAVEL_ADVICE: {
            "primary_labels": ["TravelInfo"],
            "allowed_labels": ["TravelInfo", "Accommodation", "Location"],
            "relax_labels": ["TravelInfo"],
            "blocked_labels": ["Dish", "Restaurant", "Event", "Tour"],
            "context_budget": {"TravelInfo": 0.70, "Accommodation": 0.20, "Location": 0.10}
        },
        IntentType.TRANSPORT_INFO: {
            "primary_labels": ["TravelInfo"],
            "allowed_labels": ["TravelInfo", "Location"],
            "relax_labels": ["TravelInfo"],
            "blocked_labels": ["Dish", "Restaurant", "Accommodation", "TouristAttraction", "Tour", "Event"],
            "context_budget": {"TravelInfo": 0.80, "Location": 0.20}
        },
        IntentType.EMERGENCY_SUPPORT: {
            "primary_labels": ["TravelInfo"],
            "allowed_labels": ["TravelInfo"],
            "relax_labels": ["TravelInfo"],
            "blocked_labels": ["Dish", "Restaurant", "Accommodation", "TouristAttraction", "Event", "Tour"],
            "context_budget": {"TravelInfo": 1.0}
        },
        IntentType.CASHLESS_PAYMENT: {
            "primary_labels": ["TravelInfo"],
            "allowed_labels": ["TravelInfo"],
            "relax_labels": ["TravelInfo"],
            "blocked_labels": ["Dish", "Restaurant", "Accommodation", "TouristAttraction", "Event", "Tour"],
            "context_budget": {"TravelInfo": 1.0}
        }
    }

    @classmethod
    def resolve_policy(
        cls,
        primary_intent: str,
        intents: List[str],
        user_query: str,
    ) -> RetrievalPolicyInstance:
        # 1. Detect query keyword signals
        norm = normalize_text(user_query, strip_punct=True)
        keyword_to_labels = {
            "Event": ["le hoi", "su kien", "festival", "hoat dong"],
            "Accommodation": ["khach san", "nha nghi", "homestay", "resort", "luu tru"],
            "Restaurant": ["nha hang", "quan an", "mon an", "am thuc", "an gi"],
            "Dish": ["dac san", "mon ngon", "mon dac trung", "am thuc dia phuong"],
            "Tour": ["lich trinh", "tour", "lo trinh"],
            "TouristAttraction": ["diem choi", "tham quan", "check in", "di dau"],
            "TravelInfo": ["khan cap", "cap cuu", "duong day nong", "su co", "cuu ho",
                           "san bay", "thoi tiet", "thue xe", "taxi", "thanh toan",
                           "tiem phong", "chi phi du lich", "gia ve may bay",
                           "kinh nghiem", "meo", "luu y", "nen chuan bi", "can biet",
                           "dat phong the nao", "tiet kiem", "tranh bi"],
        }
        signaled_labels = set()
        for label, keywords in keyword_to_labels.items():
            if any(kw in norm for kw in keywords):
                signaled_labels.add(label)
                if label == "Restaurant":
                    signaled_labels.add("Dish")

        # 2. Get base policy for primary intent
        base = cls.BASE_POLICIES.get(primary_intent, cls.BASE_POLICIES[IntentType.DISCOVERY])
        
        primary_labels = set(base["primary_labels"])
        allowed_labels = set(base["allowed_labels"])
        relax_labels = set(base["relax_labels"])
        blocked_labels = set(base["blocked_labels"])
        context_budget = dict(base["context_budget"])

        # 3. Union/Intersect with other intents
        other_allowed = set()
        other_primaries = set()
        for intent in (intents or []):
            if intent == primary_intent:
                continue
            other_base = cls.BASE_POLICIES.get(intent)
            if other_base:
                other_allowed.update(other_base["allowed_labels"])
                other_primaries.update(other_base["primary_labels"])

        # Unblock labels allowed by other intents or query signals
        unblock_targets = other_allowed | signaled_labels
        blocked_labels = blocked_labels - unblock_targets

        # Add newly allowed labels to allowed_labels
        allowed_labels.update(unblock_targets)

        # Update relax_labels if they are now allowed and part of secondary intents
        for intent in (intents or []):
            other_base = cls.BASE_POLICIES.get(intent)
            if other_base:
                relax_labels.update(other_base["relax_labels"])
        relax_labels.update(signaled_labels)
        relax_labels = relax_labels & allowed_labels

        # 4. Adjust budget and labels for food_specialty queries (đặc sản → prioritize Dish)
        if "Dish" in signaled_labels and primary_intent == IntentType.FOOD:
            # Boost Dish priority when "đặc sản" keywords detected
            context_budget["Dish"] = 0.50
            context_budget["Restaurant"] = 0.25
            context_budget["Specialty"] = 0.15
            # Move Restaurant to relax_labels (lower priority)
            if "Restaurant" in primary_labels:
                primary_labels.remove("Restaurant")
                relax_labels.add("Restaurant")

        # 5. Dynamically calculate budget
        if "Location" not in context_budget:
            context_budget["Location"] = 0.05

        for lbl in allowed_labels:
            if lbl not in context_budget:
                if lbl in primary_labels or lbl in other_primaries:
                    context_budget[lbl] = 0.25
                else:
                    context_budget[lbl] = 0.15

        # Remove blocked labels from budget
        for lbl in list(context_budget.keys()):
            if lbl in blocked_labels or lbl not in allowed_labels:
                context_budget.pop(lbl, None)

        # Normalize budget to sum to 1.0
        total_weight = sum(context_budget.values())
        if total_weight > 0:
            context_budget = {k: round(v / total_weight, 3) for k, v in context_budget.items()}

        return RetrievalPolicyInstance(
            primary_labels=sorted(list(primary_labels)),
            allowed_labels=sorted(list(allowed_labels)),
            relax_labels=sorted(list(relax_labels)),
            blocked_labels=sorted(list(blocked_labels)),
            context_budget=context_budget,
        )

    @classmethod
    def resolve_policy_from_query_plan(
        cls,
        query_plan: Any,
        intents: List[str] | None = None,
    ) -> RetrievalPolicyInstance:
        """Shape-aware policy resolution driven by a canonical QueryPlan (Phase 3)."""
        primary_intent = query_plan.intent
        if intents is None:
            intents = [primary_intent]
        user_query = query_plan.query

        # Start from the standard intent-based policy
        policy = cls.resolve_policy(primary_intent, intents, user_query)

        primary_labels: List[str] = list(policy.primary_labels)
        allowed_labels: List[str] = list(policy.allowed_labels)
        relax_labels: List[str] = list(policy.relax_labels)
        blocked_labels: List[str] = list(policy.blocked_labels)
        context_budget: Dict[str, float] = dict(policy.context_budget)

        # Refinement 1 – target_class label focus
        target_class = query_plan.target_class
        target_class_confidence = query_plan.target_class_confidence

        if target_class and target_class_confidence >= 0.8:
            companions = cls._COMPANION_LABELS.get(target_class, [])
            for lbl in [target_class] + companions:
                if lbl in blocked_labels:
                    blocked_labels.remove(lbl)
                if lbl not in allowed_labels:
                    allowed_labels.append(lbl)
            if target_class not in primary_labels:
                primary_labels.insert(0, target_class)
            elif primary_labels[0] != target_class:
                primary_labels.remove(target_class)
                primary_labels.insert(0, target_class)

        # Refinement 1b – semantic category filtering
        semantic_category = query_plan.semantic_category
        if semantic_category in {"natural_landmark", "heritage"} and target_class in {"Dish", "Restaurant", "Specialty", "Event", "Tour"}:
            semantic_category = None
        if semantic_category in {"natural_landmark", "heritage"}:
            _block_labels = ["Restaurant", "Dish", "Accommodation", "Tour", "Event"]
            if semantic_category == "natural_landmark":
                _block_labels.append("Museum")
            for lbl in _block_labels:
                if lbl not in blocked_labels:
                    blocked_labels.append(lbl)
                if lbl in allowed_labels:
                    allowed_labels.remove(lbl)
                if lbl in primary_labels:
                    primary_labels.remove(lbl)
            for lbl in ["TouristAttraction", "Location"]:
                if lbl in blocked_labels:
                    blocked_labels.remove(lbl)
                if lbl not in allowed_labels:
                    allowed_labels.append(lbl)
            context_budget = {
                "TouristAttraction": 0.80,
                "Location": 0.20,
            }

        # Refinement 2 – shape-aware context_budget override
        shape = query_plan.question_shape or QuestionShape.UNKNOWN
        _skip_budget_override = bool(semantic_category)

        if not _skip_budget_override and shape == QuestionShape.ITINERARY:
            context_budget = {
                "TouristAttraction": 0.35,
                "Restaurant": 0.25,
                "Accommodation": 0.25,
                "Location": 0.10,
                "Event": 0.05,
            }
            for lbl in ["TouristAttraction", "Restaurant", "Accommodation", "Event", "Location"]:
                if lbl not in allowed_labels:
                    allowed_labels.append(lbl)
                if lbl in blocked_labels:
                    blocked_labels.remove(lbl)

        elif not _skip_budget_override and shape in (QuestionShape.LIST_RANKING, QuestionShape.RECOMMENDATION_LIST):
            if target_class and target_class in allowed_labels:
                companions = cls._COMPANION_LABELS.get(target_class, [])
                context_budget = {target_class: 0.70}
                if companions:
                    comp_budget = 0.15 / len(companions)
                    for comp in companions:
                        context_budget[comp] = comp_budget
                    if "Location" not in context_budget:
                        context_budget["Location"] = 0.10
                    if "TouristAttraction" not in context_budget:
                        context_budget["TouristAttraction"] = 0.05
                else:
                    if "Location" not in context_budget:
                        context_budget["Location"] = 0.15
                    if "TouristAttraction" not in context_budget:
                        context_budget["TouristAttraction"] = 0.15

        elif not _skip_budget_override and shape == QuestionShape.SINGLE_FACT:
            if target_class and target_class in allowed_labels:
                context_budget = {
                    target_class: 0.80,
                    "Location": 0.20,
                }
            else:
                if primary_labels:
                    context_budget = {primary_labels[0]: 0.80, "Location": 0.20}

        elif not _skip_budget_override and shape == QuestionShape.COMPARISON:
            if primary_labels:
                per = round(0.80 / len(primary_labels), 3)
                context_budget = {lbl: per for lbl in primary_labels}
                context_budget["Location"] = round(0.20, 3)

        elif not _skip_budget_override and shape == QuestionShape.YES_NO:
            if target_class and target_class in allowed_labels:
                context_budget = {target_class: 0.85, "Location": 0.15}
            elif primary_labels:
                context_budget = {primary_labels[0]: 0.85, "Location": 0.15}

        # Normalize budget to sum 1.0
        context_budget = {k: v for k, v in context_budget.items() if k in set(allowed_labels + ["Location"])}
        total = sum(context_budget.values())
        if total > 0:
            context_budget = {k: round(v / total, 3) for k, v in context_budget.items()}

        # Refinement 4 – forbidden_fallbacks from intent_policy.json
        # Skip labels needed by secondary intents (e.g. TouristAttraction
        # when FOOD is primary but TOURISM is secondary).
        try:
            import json
            from pathlib import Path
            _policy_path = Path(__file__).resolve().parent.parent / "config" / "intent_policy.json"
            if _policy_path.exists():
                with open(_policy_path, "r", encoding="utf-8") as f:
                    _intent_config = json.load(f)
                _forbidden = _intent_config.get("forbidden_fallbacks", {})
                _intent_key = primary_intent
                _forbidden_labels = _forbidden.get(_intent_key, [])
                # Collect labels that secondary intents need
                _secondary_allowed = set()
                for other_intent in (intents or []):
                    if other_intent == primary_intent:
                        continue
                    other_base = cls.BASE_POLICIES.get(other_intent)
                    if other_base:
                        _secondary_allowed.update(other_base.get("allowed_labels", set()))
                for lbl in _forbidden_labels:
                    if lbl in _secondary_allowed:
                        continue  # secondary intent needs this label
                    if lbl in allowed_labels:
                        allowed_labels.remove(lbl)
                    if lbl not in blocked_labels:
                        blocked_labels.append(lbl)
        except (ValueError, KeyError, TypeError):
            pass

        # Refinement 5 – Enforced contract forbidden_labels
        forbidden_labels_from_contract = set(query_plan.forbidden_labels)
        if forbidden_labels_from_contract:
            allowed_labels = [lbl for lbl in allowed_labels if lbl not in forbidden_labels_from_contract]
            primary_labels = [lbl for lbl in primary_labels if lbl not in forbidden_labels_from_contract]
            relax_labels = [lbl for lbl in relax_labels if lbl not in forbidden_labels_from_contract]
            for lbl in forbidden_labels_from_contract:
                if lbl not in blocked_labels:
                    blocked_labels.append(lbl)
            context_budget = {k: v for k, v in context_budget.items() if k not in forbidden_labels_from_contract}
            total = sum(context_budget.values())
            if total > 0:
                context_budget = {k: round(v / total, 3) for k, v in context_budget.items()}

        return RetrievalPolicyInstance(
            primary_labels=primary_labels,
            allowed_labels=sorted(set(allowed_labels)),
            relax_labels=sorted(set(relax_labels) & set(allowed_labels)),
            blocked_labels=sorted(set(blocked_labels) - set(allowed_labels)),
            context_budget=context_budget,
        )


