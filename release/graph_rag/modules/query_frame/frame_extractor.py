from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Set

from graph_rag.config import cfg as _cfg
from graph_rag.core.intents import IntentType
from graph_rag.core import keywords as _core_kw
from .frame_models import Mention, QueryFrame, RetrievalPlan

_kw = _cfg.keywords()

class QueryFrameExtractor:
    """Rule-based QueryFrame extractor.

    The extractor is intentionally conservative. It proposes role metadata only
    when common domain patterns are present; a validator decides whether the
    frame can affect grounding/retrieval.

    All keyword lists are loaded from domain_keywords.json (via _kw or _core_kw)
    so that modifying keywords does not require changing Python code.
    """

    ENTITY_PREFIXES = _kw.get("entity_prefixes", [])

    QUERY_OPERATOR_PATTERNS = _kw.get("query_operator_patterns", {})

    CONSTRAINED_NEARBY_PATTERNS = _core_kw.CONSTRAINED_NEARBY_PATTERNS

    NON_GROUNDABLE_GENERIC_PHRASES = _core_kw.NON_GROUNDABLE_GENERIC_PHRASES

    NON_GROUNDABLE_STARTS = _core_kw.NON_GROUNDABLE_STARTS

    LOCATION_TERMS = list(_core_kw.BROAD_LOCATION_NAMES)

    CONSTRAINT_TERMS = _core_kw.CONSTRAINT_TERMS

    def __init__(self, normalizer=None):
        self._external_normalizer = normalizer

    def normalize(self, text: str) -> str:
        if self._external_normalizer:
            return self._external_normalizer(text)
        norm = unicodedata.normalize("NFKD", str(text or ""))
        norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
        norm = norm.replace("\u0111", "d").replace("\u0110", "d")
        return re.sub(r"\s+", " ", norm.lower()).strip()

    def extract(self, query: str, metadata: Dict[str, Any] | None = None, primary_intent: str = "") -> QueryFrame:
        metadata = metadata or {}
        llm_entities = metadata.get("entities", [])
        q_norm = self.normalize(query)
        question_type = str(metadata.get("question_type") or "")
        answer_mode = str(metadata.get("answer_mode") or "")
        operator = self._infer_operator(q_norm, question_type, primary_intent)

        # Chain detection: constrained_nearby_search overrides before
        # lodging_near_anchor / dish_to_restaurant
        chain = self._extract_chain(q_norm)
        if chain and operator not in {"comparison", "choice_selection"}:
            operator = "constrained_nearby_search"

        non_groundable = self._extract_non_groundable_phrases(query, q_norm, operator)
        candidates = self._extract_choice_candidates(query)
        comparison_subjects = self._extract_comparison_subjects(query, q_norm, llm_entities=llm_entities)
        entity_mentions = self._extract_groundable_mentions(query, metadata)
        
        # dish_mentions now come from LLM entities
        dish_mentions = [m for m in entity_mentions if m.type_hint == "Dish"]
        for m in dish_mentions:
            m.role = "dish"

        lodging_near_anchors = self._extract_lodging_near_anchors(query, q_norm, llm_entities=llm_entities)
        if dish_mentions:
            comparison_subjects = [
                mention for mention in comparison_subjects
                if not self._is_generic_dish_comparison_subject(mention.text, dish_mentions)
            ]
            entity_mentions = self._dedupe_mentions(list(entity_mentions) + dish_mentions)
            if (
                self._is_dish_to_restaurant_query(q_norm, operator, candidates, comparison_subjects)
            ):
                operator = "dish_to_restaurant"
        if lodging_near_anchors and operator != "constrained_nearby_search":
            entity_mentions = self._dedupe_mentions(list(entity_mentions) + lodging_near_anchors)
            if operator != "comparison":
                operator = "lodging_near_anchor"
        location_scope = self._extract_location_scope(query, metadata)
        constraints = self._extract_frame_topics(q_norm)
        answer_set_variables = self._extract_answer_set_variables(q_norm, operator, constraints)

        if operator == "choice_selection" and candidates:
            required_relations = list(metadata.get("requested_relations") or [])
            if dish_mentions and "HAS" not in required_relations:
                required_relations.append("HAS")
            plan = RetrievalPlan(
                mode="multi_candidate",
                candidate_entities=candidates,
                required_attributes=list(metadata.get("requested_attributes") or []),
                required_relations=required_relations,
                context_policy={
                    "render_by_candidate": True,
                    "dish_constraints": [m.text for m in dish_mentions],
                },
            )
        elif operator == "dish_to_restaurant":
            required_relations = list(metadata.get("requested_relations") or [])
            for rel in ["HAS", "LOCATED_IN", "NEAR"]:
                if rel not in required_relations:
                    required_relations.append(rel)
            plan = RetrievalPlan(
                mode="dish_to_restaurant",
                anchors=dish_mentions,
                required_attributes=list(metadata.get("requested_attributes") or []),
                required_relations=required_relations,
                context_policy={
                    "answer_set_from_relation": "HAS",
                    "answer_set_label": "Restaurant",
                    "dish_constraints": [m.text for m in dish_mentions],
                },
            )
        elif operator == "constrained_nearby_search":
            # Multi-hop chain reasoning plan
            answer_set_label = self._infer_chain_answer_label(q_norm)
            plan = RetrievalPlan(
                mode="constrained_nearby_search",
                anchors=[],  # No single anchor — chain defines the path
                required_relations=["NEAR", "HAS", "LOCATED_IN"],
                context_policy={
                    "answer_set_label": answer_set_label,
                    "chain": chain,
                    "location_scope": location_scope,
                },
            )
        elif operator == "lodging_near_anchor":
            required_relations = list(metadata.get("requested_relations") or [])
            for rel in ["NEAR", "LOCATED_IN", "BELONGS_TO"]:
                if rel not in required_relations:
                    required_relations.append(rel)
            plan = RetrievalPlan(
                mode="lodging_near_anchor",
                anchors=lodging_near_anchors,
                required_attributes=list(metadata.get("requested_attributes") or []),
                required_relations=required_relations,
                context_policy={
                    "answer_set_label": "Accommodation",
                    "anchor_relation": "NEAR",
                    "prefer_nearby_lodging": True,
                },
            )
        elif operator == "comparison" and comparison_subjects:
            required_relations = list(metadata.get("requested_relations") or [])
            for rel in ["NEAR", "LOCATED_IN", "BELONGS_TO", "HAS", "OFFERS", "INCLUDES"]:
                if rel not in required_relations:
                    required_relations.append(rel)
            if dish_mentions and "HAS" not in required_relations:
                required_relations.append("HAS")
            required_attributes = list(metadata.get("requested_attributes") or [])
            if "description" not in required_attributes:
                required_attributes.append("description")
            plan = RetrievalPlan(
                mode="comparison",
                anchors=comparison_subjects,
                required_attributes=required_attributes,
                required_relations=required_relations,
                context_policy={
                    "render_comparison": True,
                    "dish_constraints": [m.text for m in dish_mentions],
                },
            )
        elif operator == "tour_availability":
            # Tour availability: find Tour nodes matching constraints, no route building
            plan = RetrievalPlan(
                mode="class_search",
                anchors=[],
                required_relations=["INCLUDES", "OFFERS"],
                context_policy={
                    "answer_set_label": "Tour",
                    "target_class": "Tour",
                    "constraints": constraints,
                },
            )
        elif operator == "itinerary_recommendation":
            anchors = [m for m in entity_mentions if m.role in {"origin_accommodation", "anchor_entity", "tour"}]
            plan = RetrievalPlan(
                mode="tour_plan",
                anchors=anchors or entity_mentions[:1],
                required_relations=["NEAR"],
                context_policy={"prefer_structural": True, "constraints": constraints},
            )
        elif operator == "global_discovery":
            plan = RetrievalPlan(
                mode="global_discovery",
                anchors=entity_mentions[:1],
                required_attributes=list(metadata.get("requested_attributes") or []),
                required_relations=list(metadata.get("requested_relations") or []),
                context_policy={"avoid_single_main_entity": True},
            )
        else:
            plan = RetrievalPlan(
                mode="single_anchor",
                anchors=entity_mentions[:1],
                required_attributes=list(metadata.get("requested_attributes") or []),
                required_relations=list(metadata.get("requested_relations") or []),
            )

        confidence = self._estimate_confidence(operator, entity_mentions, candidates, comparison_subjects)
        return QueryFrame(
            query_operator=operator,
            answer_mode=answer_mode,
            question_type=question_type,
            location_scope=location_scope,
            groundable_mentions=entity_mentions,
            candidate_entities=candidates,
            comparison_subjects=comparison_subjects,
            answer_set_variables=answer_set_variables,
            requested_attributes=list(metadata.get("requested_attributes") or []),
            requested_relations=list(metadata.get("requested_relations") or []),
            constraints=constraints,
            non_groundable_phrases=non_groundable,
            retrieval_plan=plan,
            confidence=confidence,
        )

    def _infer_operator(self, q_norm: str, question_type: str, primary_intent: str) -> str:
        # Tour availability must be checked BEFORE itinerary — both contain "tour"
        if self._contains_any(q_norm, self.QUERY_OPERATOR_PATTERNS["tour_availability"]):
            return "tour_availability"
        if self._contains_any(q_norm, self.QUERY_OPERATOR_PATTERNS["itinerary_recommendation"]):
            return "itinerary_recommendation"
        qt_norm = self.normalize(question_type)
        if self._contains_any(q_norm, self.QUERY_OPERATOR_PATTERNS["comparison"]):
            return "comparison"
        if qt_norm in {"multiple-choice", "multi-choice", "multi-select"} or self._contains_any(q_norm, self.QUERY_OPERATOR_PATTERNS["choice_selection"]):
            return "choice_selection"
        # constrained_nearby_search is detected via regex later in extract(),
        # but we also check here for early return when the chain pattern is
        # unambiguous.
        if self._is_constrained_nearby_query(q_norm):
            return "constrained_nearby_search"
        if any(marker in q_norm for marker in ["thuoc the loai", "thuoc loai", "loai hinh nao", "phan loai nao"]):
            return "fact_lookup"
        if str(primary_intent) == IntentType.TOUR_PLAN:
            return "itinerary_recommendation"
        if self._contains_any(q_norm, self.QUERY_OPERATOR_PATTERNS["global_discovery"]):
            return "global_discovery"
        if str(primary_intent) == IntentType.DISCOVERY:
            return "global_discovery"
        return "fact_lookup"

    def _extract_non_groundable_phrases(self, query: str, q_norm: str, operator: str) -> List[str]:
        phrases: List[str] = []
        for pattern in self.NON_GROUNDABLE_STARTS:
            if pattern in q_norm:
                phrases.append(pattern)
        if operator == "itinerary_recommendation":
            phrases.extend(["goi y mot lich trinh", "goi y lich trinh", "du khach luu tru"])
        elif operator == "choice_selection":
            phrases.extend(["dia diem nao sau day", "trong cac lua chon", "phuong an nao"])
        elif operator == "comparison":
            phrases.extend(["so sanh cac", "so sanh vi tri"])
        elif operator == "constrained_nearby_search":
            # All generic category words in chain queries are non-groundable
            for gp in self.NON_GROUNDABLE_GENERIC_PHRASES:
                if gp in q_norm:
                    phrases.append(gp)
        return self._dedupe_strings(phrases)

    def _extract_choice_candidates(self, query: str) -> List[Mention]:
        candidates: List[Mention] = []
        for match in re.finditer(r"(?im)^\s*([A-D])\s*[\).:-]\s+(.+?)\s*$", query or ""):
            text = self._clean_entity_text(match.group(2))
            if self._is_non_groundable_choice(text):
                continue
            if text:
                candidates.append(Mention(
                    text=text,
                    role="choice_candidate",
                    type_hint=self._infer_type_hint(text),
                    groundability="groundable",
                    required=True,
                    confidence=0.9,
                ))
        return self._dedupe_mentions(candidates)

    def _extract_lodging_near_anchors(self, query: str, q_norm: str, llm_entities: List[Dict] = None) -> List[Mention]:
        if not any(term in q_norm for term in ["nha nghi", "khach san", "homestay", "resort", "luu tru"]):
            return []
        if " gan " not in f" {q_norm} ":
            return []
        # Extract anchor text from pattern
        pattern = r"(?is)\b(?:gần|gan)\s+(.+?)(?=\s+(?:và|va|để|de|có thể|co the|trong|ở|o|tại|tai)\b|[?.]\s*|$)"
        match = re.search(pattern, query or "")
        if not match:
            return []
        anchor_text = self._clean_entity_text(match.group(1))
        if not anchor_text or self._is_non_groundable_text(anchor_text):
            return []
        # Try to match with LLM entities first
        if llm_entities:
            anchor_norm = self.normalize(anchor_text)
            for e in llm_entities:
                e_name = str(e.get("name") or "")
                if self.normalize(e_name) in anchor_norm or anchor_norm in self.normalize(e_name):
                    return [Mention(
                        text=e_name,
                        role="proximity_anchor",
                        type_hint=str(e.get("type") or self._infer_type_hint(e_name)),
                        groundability="groundable",
                        required=True,
                        confidence=0.84,
                    )]
        # Fallback: use raw anchor text
        return [Mention(
            text=anchor_text,
            role="proximity_anchor",
            type_hint=self._infer_type_hint(anchor_text),
            groundability="groundable",
            required=True,
            confidence=0.74,
        )]

    def _is_dish_to_restaurant_query(
        self,
        q_norm: str,
        operator: str,
        candidates: List[Mention],
        comparison_subjects: List[Mention],
    ) -> bool:
        if not ("mon" in q_norm and any(term in q_norm for term in ["quan", "nha hang", "ca phe", "am thuc"])):
            return False
        if candidates:
            return False
        if operator == "comparison" and len(comparison_subjects) >= 2:
            return False
        return operator in {"fact_lookup", "comparison", "global_discovery", "choice_selection"}

    def _is_generic_dish_comparison_subject(self, text: str, dish_mentions: List[Mention]) -> bool:
        value_norm = self.normalize(text)
        if value_norm in {
            "cac quan an",
            "cac nha hang",
            "nhung quan an",
            "nhung nha hang",
            "quan an",
            "nha hang",
        }:
            return True
        dish_norms = {self.normalize(mention.text) for mention in dish_mentions}
        return bool(value_norm and value_norm in dish_norms)

    def _extract_comparison_subjects(self, query: str, q_norm: str, llm_entities: List[Dict] = None) -> List[Mention]:
        if "so sanh" not in q_norm and "khac biet" not in q_norm and "tuong dong" not in q_norm:
            return []
        mentions = (
            self._extract_listed_subjects(query)
            or self._extract_generic_comparison_subjects(query)
            or self._extract_paired_subjects(query)
        )
        # Fallback to LLM entities if regex found nothing
        if not mentions and llm_entities:
            mentions = [e["name"] for e in llm_entities if e.get("name")]
        return [
            Mention(
                text=m,
                role="comparison_subject",
                type_hint=self._infer_type_hint(m),
                groundability="groundable",
                required=True,
                confidence=0.78,
            )
            for m in mentions[:4]
            if not self._is_non_groundable_text(m)
        ]

    def _extract_listed_subjects(self, query: str) -> List[str]:
        match = re.search(r"(?is)\b(?:trong\s+ba|ba|cac|các)\s+(?:quan|quán|nha\s+hang|nhà\s+hàng)[:：]\s*(.+?)(?:,\s*(?:quan|quán|nha\s+hang|nhà\s+hàng)\s+nao|\?)", query or "")
        if not match:
            return []
        segment = match.group(1)
        parts = re.split(r"\s*(?:,|\s+và\s+|\s+va\s+)\s*", segment)
        cleaned: List[str] = []
        for part in parts:
            value = self._clean_entity_text(part)
            if not value:
                continue
            value_norm = self.normalize(value)
            if not value_norm.startswith(("quan ", "nha hang ", "quán ", "nhà hàng ")) and "coffee" not in value_norm and "bê thui" not in value_norm:
                value = f"Quán {value}"
            cleaned.append(value)
        return self._dedupe_strings(cleaned)

    def _extract_generic_comparison_subjects(self, query: str) -> List[str]:
        text = str(query or "")
        paired_match = re.search(
            r"(?is)\b(?:giữa|giua)\s+(.+?)\s+(?:và|va)\s+(.+?)(?:,|\s+(?:và|va)\s+(?:giải|giai)|[?.]|$)",
            text,
        )
        if paired_match:
            first = self._clean_comparison_subject_text(paired_match.group(1))
            second = self._clean_comparison_subject_text(paired_match.group(2))
            cleaned = [
                value for value in [first, second]
                if value and not self._is_non_groundable_text(value)
            ]
            if len(cleaned) >= 2:
                return self._dedupe_strings(cleaned)
        patterns = [
            r"(?is)\b(?:của|cua|giữa|giua)\s+(.+?)\s+(?:về|ve|dựa|dua|đồng thời|dong thoi|theo)\b",
            r"(?is)\bso\s+sánh\s+(?:đặc\s+điểm\s+)?(?:của|cua)?\s*(.+?)\s+(?:về|ve|dựa|dua|đồng thời|dong thoi|theo)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if not match:
                continue
            segment = match.group(1)
            parts = re.split(r"\s*(?:,|\s+và\s+|\s+va\s+)\s*", segment)
            cleaned: List[str] = []
            for part in parts:
                value = self._clean_comparison_subject_text(part)
                if not value or self._is_non_groundable_text(value):
                    continue
                norm = self.normalize(value)
                if norm in {"cac nha hang", "nhung nha hang", "nha hang", "cac dia diem"}:
                    continue
                cleaned.append(value)
            if len(cleaned) >= 2:
                return self._dedupe_strings(cleaned)
        return []

    def _extract_paired_subjects(self, query: str) -> List[str]:
        prefix = r"(?:Khach san|Khách sạn|Nha nghi|Nhà nghỉ|Homestay|Resort|Nha hang|Nhà hàng|Quán|Bien|Biển|Đập|Dap|Thác|Thac|Suối|Suoi|Làng nghề|Lang nghe|Lễ hội|Le hoi|Khu du lich|Khu du lịch|Di tich|Di tích|Bao tang|Bảo tàng)"
        pattern = rf"(?i)\b({prefix}\s+.+?)\s+(?:va|và)\s+({prefix}\s+.+?)(?:\s+(?:dua tren|dựa trên|theo|co|có|nam|nằm)|[?.!,]|$)"
        match = re.search(pattern, query or "")
        if not match:
            name_prefix = r"(?:Nha hang|Nhà hàng|Khach san|Khách sạn|Nha nghi|Nhà nghỉ|Quán|Resort|Tour|Đập|Dap|Thác|Thac|Suối|Suoi|Làng nghề|Lang nghe|Lễ hội|Le hoi)"
            loose = re.findall(rf"(?i)\b({name_prefix}\s+[^,?.!]+?)(?=\s+(?:va|và|dua tren|dựa trên|theo)|[?.!,]|$)", query or "")
            return self._dedupe_strings([self._clean_comparison_subject_text(item) for item in loose])
        return self._dedupe_strings([
            self._clean_comparison_subject_text(match.group(1)),
            self._clean_comparison_subject_text(match.group(2)),
        ])

    def _extract_groundable_mentions(self, query: str, metadata: Dict[str, Any]) -> List[Mention]:
        mentions: List[Mention] = []
        for entity in metadata.get("entities") or []:
            if not isinstance(entity, dict):
                continue
            name = self._clean_entity_text(str(entity.get("name") or ""))
            if not name or self._is_non_groundable_text(name):
                continue
            mentions.append(Mention(
                text=name,
                role=self._infer_role(name, query),
                type_hint=str(entity.get("type") or self._infer_type_hint(name)),
                groundability="groundable",
                required=False,
                confidence=0.72,
            ))
        return self._dedupe_mentions(mentions)

    def _extract_location_scope(self, query: str, metadata: Dict[str, Any]) -> str:
        explicit = str(metadata.get("detected_location") or metadata.get("geo_anchor_location") or "").strip()
        if explicit:
            return explicit
        q_norm = self.normalize(query)
        for term in self.LOCATION_TERMS:
            if term in q_norm:
                return term
        return ""

    def _extract_frame_topics(self, q_norm: str) -> Dict[str, Any]:
        constraints: Dict[str, Any] = {}
        wanted: List[str] = []
        for key, terms in self.CONSTRAINT_TERMS.items():
            if any(term in q_norm for term in terms):
                wanted.append(key)
        if wanted:
            constraints["candidate_topics"] = wanted
        if "nua ngay" in q_norm:
            constraints["duration"] = "half_day"
        elif "mot ngay" in q_norm or "1 ngay" in q_norm:
            constraints["duration"] = "one_day"
        else:
            # Multi-day duration: "2 ngay 1 dem", "3n2d", etc.
            multi_day = re.search(r'\b(\d{1,2})\s*(?:ngay|ngày)\b', q_norm)
            if multi_day:
                constraints["duration"] = f"{multi_day.group(1)}_days"
            elif re.search(r'\b(\d{1,2})\s*n\s*(\d{1,2})\s*d\b', q_norm):
                m = re.search(r'\b(\d{1,2})\s*n\s*(\d{1,2})\s*d\b', q_norm)
                constraints["duration"] = f"{m.group(1)}_days"
        return constraints

    def _extract_answer_set_variables(self, q_norm: str, operator: str, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        variables: List[Dict[str, Any]] = []
        if operator == "itinerary_recommendation":
            variables.append({
                "role": "candidate_attractions",
                "relation": "NEAR",
                "constraints": constraints.get("candidate_topics") or [],
            })
        elif operator == "choice_selection":
            variables.append({"role": "selected_choice"})
        elif "nao" in q_norm or "nhung" in q_norm:
            variables.append({"role": "answer_set"})
        return variables

    def _infer_role(self, text: str, query: str) -> str:
        norm = self.normalize(text)
        q_norm = self.normalize(query)
        if any(term in norm for term in ["khach san", "nha nghi", "homestay", "resort"]):
            if any(term in q_norm for term in ["luu tru", "nghi", "o tai", "tai"]):
                return "origin_accommodation"
            return "accommodation"
        if any(term in norm for term in ["nha hang", "quan "]):
            return "restaurant"
        if norm.startswith("tour "):
            return "tour"
        return "anchor_entity"

    def _infer_type_hint(self, text: str) -> str:
        norm = self.normalize(text)
        # Duration patterns should be typed as "duration" so downstream filters can skip them
        if re.search(r'\b\d+\s*(?:ngay|ngày|dem|đêm|n\s*\d+\s*d)\b', norm):
            return "duration"
        if any(term in norm for term in ["khach san", "nha nghi", "homestay", "resort", "hotel"]):
            return "Accommodation"
        if any(term in norm for term in ["nha hang", "quan "]):
            return "Restaurant"
        if any(term in norm for term in ["ca phe", "bun", "mi ", "lau", "thit", "tom", "ga ", "ca ", "cari", "ca ri"]):
            return "Dish"
        if any(term in norm for term in ["tour "]):
            return "Tour"
        if any(term in norm for term in ["le hoi"]):
            return "Event"
        if any(term in norm for term in ["dap ", "thac ", "suoi ", "lang nghe"]):
            return "TouristAttraction"
        if norm in self.LOCATION_TERMS:
            return "Location"
        return "Place"

    def _estimate_confidence(
        self,
        operator: str,
        mentions: List[Mention],
        candidates: List[Mention],
        comparison_subjects: List[Mention],
    ) -> float:
        if operator == "constrained_nearby_search":
            return 0.88  # Chain detected by regex — high confidence
        if operator == "dish_to_restaurant":
            return 0.86 if any(m.role == "dish" or m.type_hint == "Dish" for m in mentions) else 0.45
        if operator == "lodging_near_anchor":
            return 0.84 if mentions else 0.45
        if operator == "choice_selection":
            return 0.9 if len(candidates) >= 2 else 0.45
        if operator == "comparison":
            return 0.85 if len(comparison_subjects) >= 2 else 0.5
        if operator == "itinerary_recommendation":
            if any(m.role == "origin_accommodation" for m in mentions):
                return 0.9
            return 0.65 if mentions else 0.45
        if mentions:
            return 0.72
        return 0.4

    def _is_non_groundable_text(self, text: str) -> bool:
        norm = self.normalize(text)
        if not norm:
            return True
        # Duration patterns are not groundable entities
        if re.search(r'\b\d+\s*(?:ngay|ngày|dem|đêm|n\s*\d+\s*d)\b', norm):
            return True
        if any(norm.startswith(pattern) for pattern in self.NON_GROUNDABLE_STARTS):
            return True
        # Generic category phrases must never become grounding targets
        if norm in self.NON_GROUNDABLE_GENERIC_PHRASES:
            return True
        if norm in {
            "khach san",
            "nha nghi",
            "homestay",
            "resort",
            "nha hang",
            "quan an",
            "bien",
            "bao tang",
            "di tich",
            "di tich lich su",
            "thac nuoc",
            "lang chai",
            "danh thang",
            "diem tham quan",
        }:
            return True
        if len(norm.split()) > 12 and not any(prefix in norm for prefix in self.ENTITY_PREFIXES):
            return True
        variable_terms = ["dia diem nao", "nha hang nao", "quan nao", "quan an nao", "thuc the nao", "phuong an nao"]
        return any(term in norm for term in variable_terms)

    def _is_non_groundable_choice(self, text: str) -> bool:
        norm = self.normalize(text)
        if not norm:
            return True
        return norm.startswith(("ca hai", "tat ca", "khong ", "khong co", "khong quan", "khong dia diem"))

    def _is_generic_dish_phrase(self, text: str) -> bool:
        norm = self.normalize(text)
        if not norm:
            return True

        # Reject phrases consisting purely of question particles and functional grammatical helper words
        norm_tokens = norm.split()
        question_helpers = {
            "nao", "gi", "khac", "nua", "khong", "a", "nhi", "the", "vay", 
            "chua", "sao", "dau", "ai", "co", "con", "de", "cho", "la", "lam",
            "ra", "di", "thoi", "nhe", "nha", "chut", "it", "ti", "tieu",
            "them", "bot", "giup", "ho", "voi", "mang", "dem", "lay", "muon",
            "ngon", "re", "doc", "la", "hay", "tot", "truyen", "thong", "co", "truyen"
        }
        if norm_tokens and all(token in question_helpers for token in norm_tokens):
            return True

        generic = {
            "mon",
            "mon an",
            "mon an dac trung",
            "an dac trung",
            "an khac",
            "cac mon an dac trung",
            "dia diem gan do",
            "cac dia diem lan can",
            "dac san",
            "thong tin",
        }
        return norm in generic or norm.startswith(("cac mon an", "nhung mon an", "cac dia diem", "dia diem gan"))

    def _clean_entity_text(self, text: str) -> str:
        value = re.sub(r"\s+", " ", str(text or "")).strip(" ,.;:!?")
        tails = [
            r"\s+(?:muon|muốn|uu tien|ưu tiên|can|cần|hay|hãy|va|và)\b.*$",
            r"\s+(?:nhu|như)\b.*$",
            r"\s+(?:dua tren|dựa trên)\b.*$",
            r"\s+(?:la|là)\s+(?:gi|gì)\b.*$",
            r"\s+(?:thuoc|thuộc)\s+(?:loai|loại|the|thể)\b.*$",
            r"\s+(?:co|có)\s+(?:mon|món|so|số)\b.*$",
            r"\s+(?:o|ở|tai|tại)\s+(?:Quy Nhon|Quy Nhơn|Gia Lai|Binh Dinh|Bình Định)\b.*$",
        ]
        for pattern in tails:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip(" ,.;:!?")
        value = re.sub(r"\s+(?:co|có)\s+(?:bao gom|bao gồm)\b.*$", "", value, flags=re.IGNORECASE).strip(" ,.;:!?")
        if value.endswith(")") and "(" in value and value.count("(") == value.count(")"):
            return value
        if "(" in value and ")" not in value:
            value = value.split("(")[0].strip(" ,.;:!?")
        return value

    def _clean_comparison_subject_text(self, text: str) -> str:
        value = self._clean_entity_text(text)
        tails = [
            r"\s+(?:doi voi|đối với)\b.*$",
            r"\s+(?:dua tren|dựa trên|theo)\b.*$",
            r"\s+(?:neu|nếu)\b.*$",
            r"\s+(?:ve|về)\b.*$",
            r"\s+(?:co|có)\s+(?:nhung|những|cac|các)\b.*$",
        ]
        for pattern in tails:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE).strip(" ,.;:!?")
        return value

    # ── Constrained nearby search helpers ─────────────────────────────

    def _is_constrained_nearby_query(self, q_norm: str) -> bool:
        """Return True if q_norm matches any CONSTRAINED_NEARBY_PATTERNS regex."""
        return any(re.search(pat, q_norm) for pat in self.CONSTRAINED_NEARBY_PATTERNS)

    def _extract_chain(self, q_norm: str) -> List[Dict[str, str]]:
        """Extract a multi-hop chain from the normalized query.

        Returns a list of dicts like:
          [{"from": "Accommodation", "rel": "NEAR", "to": "TouristAttraction"},
           {"from": "TouristAttraction", "rel": "HAS", "to": "Dish"}]

        The chain is built by walking the query left→right and mapping
        each \"gần\" / \"có\" / \"đi qua\" to a relationship hop.
        """
        if not self._is_constrained_nearby_query(q_norm):
            return []

        # Determine the *first* entity class mentioned (answer set)
        first_label = self._label_from_prefix(q_norm)
        if not first_label:
            return []

        # Walk the query for relationship+label pairs
        chain: List[Dict[str, str]] = []
        current_label = first_label

        # Regex to find relationship triggers followed by a category
        hop_pattern = re.compile(
            r"\b(gan|di qua|qua)\b[^,?.]*?\b"
            r"(dia diem du lich|diem du lich|dia diem tham quan|diem tham quan|dia diem|diem"
            r"|khach san|nha nghi|homestay|resort"
            r"|nha hang|quan an"
            r"|le hoi|su kien"
            r"|tour)"
        )
        for m in hop_pattern.finditer(q_norm):
            rel_word = m.group(1)
            target_word = m.group(2)
            rel = "NEAR" if rel_word in {"gan"} else "PASSES_THROUGH"
            target_label = self._category_to_label(target_word)
            if target_label and target_label != current_label:
                chain.append({"from": current_label, "rel": rel, "to": target_label})
                current_label = target_label

        # Look for "có" after the last NEAR hop
        # HAS for Restaurant -> Dish; Dish cũng có SPECIALTY_OF -> Location
        has_pattern = re.compile(
            r"\bco\b[^,?.]*?\b"
            r"(mon an dac san|mon dac san|dac san|am thuc|mon an|mon ngon"
            r"|nha hang|quan an"
            r"|le hoi|su kien"
            r"|khach san|nha nghi)"
        )
        for m in has_pattern.finditer(q_norm):
            target_word = m.group(1)
            target_label = self._category_to_label(target_word)
            if target_label and target_label != current_label:
                chain.append({"from": current_label, "rel": "HAS", "to": target_label})
                current_label = target_label

        return chain if len(chain) >= 2 else []

    def _label_from_prefix(self, q_norm: str) -> str:
        """Determine the answer-set label from the query prefix."""
        prefix_map = [
            (["khach san", "nha nghi", "homestay", "resort"], "Accommodation"),
            (["nha hang", "quan an"], "Restaurant"),
            (["tour"], "Tour"),
            (["dia diem", "diem du lich"], "TouristAttraction"),
        ]
        for markers, label in prefix_map:
            for marker in markers:
                if marker in q_norm[:30]:
                    return label
        return ""

    def _category_to_label(self, category: str) -> str:
        """Map a normalized Vietnamese category phrase to a Neo4j label."""
        mapping = {
            "dia diem du lich": "TouristAttraction",
            "diem du lich": "TouristAttraction",
            "dia diem tham quan": "TouristAttraction",
            "diem tham quan": "TouristAttraction",
            "dia diem": "TouristAttraction",
            "diem": "TouristAttraction",
            "khach san": "Accommodation",
            "nha nghi": "Accommodation",
            "homestay": "Accommodation",
            "resort": "Accommodation",
            "nha hang": "Restaurant",
            "quan an": "Restaurant",
            "le hoi": "Event",
            "su kien": "Event",
            "tour": "Tour",
            "mon an dac san": "Dish",
            "mon dac san": "Dish",
            "dac san": "Dish",
            "am thuc": "Dish",
            "mon an": "Dish",
            "mon ngon": "Dish",
        }
        return mapping.get(category, "")

    def _infer_chain_answer_label(self, q_norm: str) -> str:
        """The answer set is the first entity class in the query."""
        return self._label_from_prefix(q_norm) or "Accommodation"

    def _contains_any(self, text: str, terms: Iterable[str]) -> bool:
        return any(term in text for term in terms)

    def _dedupe_strings(self, values: Iterable[str]) -> List[str]:
        result: List[str] = []
        seen: Set[str] = set()
        for value in values:
            key = self.normalize(value)
            if key and key not in seen:
                seen.add(key)
                result.append(str(value).strip())
        return result

    def _dedupe_mentions(self, mentions: Iterable[Mention]) -> List[Mention]:
        result: List[Mention] = []
        seen: Set[str] = set()
        for mention in mentions:
            key = self.normalize(mention.text)
            if key and key not in seen:
                seen.add(key)
                result.append(mention)
        return result
