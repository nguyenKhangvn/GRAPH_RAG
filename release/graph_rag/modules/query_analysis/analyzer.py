# graph_rag/modules/query_analysis/analyzer.py

import json
import re
import logging
from typing import Dict, Any, List
from graph_rag.services.ai_model import LLMService
from graph_rag.core.intents import IntentType
from graph_rag.core import keywords, business_rules
from graph_rag.utils.text import normalize_text

logger = logging.getLogger("graph_rag.query_analysis")

class QueryAnalyzer:
    MAX_HISTORY_CHARS = 700
    MAX_SCHEMA_CHARS = 900
    MAX_HISTORY_TURNS = 5

    # Centralized keyword references
    CURRENT_LOCATION_ROUTE_HINTS = keywords.LOCATION_ROUTE_SIGNALS
    ADMIN_CENTER_HINTS = keywords.ADMIN_CENTER_HINTS
    LOCATION_LEAKAGE_HINTS = keywords.LOCATION_LEAKAGE_HINTS
    TYPO_NORMALIZATION = keywords.TYPO_NORMALIZATION
    ATTRIBUTE_HINTS = keywords.ATTRIBUTE_HINTS
    RELATION_HINTS = keywords.RELATION_HINTS
    TOURISM_ANALYSIS_HINTS = keywords.TOURISM_ANALYSIS_HINTS
    REAL_FOOD_HINTS = keywords.REAL_FOOD_HINTS
    _QUESTION_PARTICLES = keywords.QUESTION_PARTICLES
    _QUESTION_BIGRAMS = keywords.QUESTION_BIGRAMS
    _BROAD_LOCATION_NAMES = keywords.BROAD_LOCATION_NAMES

    # Build set of all known location names from region_aliases for substitution detection
    _KNOWN_LOCATION_NAMES: set = set()
    for _aliases in business_rules.REGION_ALIASES.values():
        for _alias in _aliases:
            _KNOWN_LOCATION_NAMES.add(normalize_text(_alias))

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        # Lazy load recovery configuration and blacklist from domain_keywords.json
        self.recovery_config = keywords._kw.get("entity_recovery", {})
        self.blacklist = keywords._kw.get("blacklist", {})
        
        # Flatten dictionary to support Longest-Match Matching
        self.flat_aliases = []
        for c_name, meta in self.recovery_config.items():
            all_variants = [c_name] + meta.get("aliases", [])
            for variant in all_variants:
                self.flat_aliases.append({
                    "raw_variant": variant,
                    "norm_variant": normalize_text(variant).strip(),
                    "canonical_name": meta.get("canonical_name", c_name),
                    "type": meta.get("type", "TouristAttraction")
                })
        # Sort by length descending to prioritize Longest Match
        self.flat_aliases.sort(key=lambda x: len(x["norm_variant"]), reverse=True)

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if not text:
            return ""
        text = str(text)
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + " ..."

    def _format_recent_history(self, history: List[Dict[str, Any]]) -> str:
        """Keep only recent compact turns for coreference, not full conversation memory."""
        turns: list[str] = []
        for item in list(history or [])[-self.MAX_HISTORY_TURNS:]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip() or "unknown"
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            content = re.sub(r"\s+", " ", content)
            turns.append(f"{role}: {content[:180]}")
        return self._truncate_text("\n".join(turns), self.MAX_HISTORY_CHARS)

    def _compact_schema_context(self) -> str:
        """Minimal schema contract for routing; detailed schema is handled downstream."""
        return self._truncate_text(
            "\n".join(
                [
                    "Entity types: Location, TouristAttraction, Restaurant, Dish, Accommodation, Event, Tour, TravelInfo, TravelAgency, Duration, GroupSize.",
                    "Relations: NEAR, LOCATED_IN, BELONGS_TO, HAS, HELD_AT, INCLUDES, OFFERS.",
                    "Intents: ACCOMMODATION_RECOMMENDATION, FOOD_RECOMMENDATION, TOURISM_RECOMMENDATION, EVENT_RECOMMENDATION, TOUR_PLAN, DISTANCE_QUERY, ENTITY_FACT_QUERY, DISCOVERY_SEARCH, TRAVEL_ADVICE.",
                    "Attributes: address, phone, price, ticket_price, price_range, opening_hours, description, service_features.",
                ]
            ),
            self.MAX_SCHEMA_CHARS,
        )

    def _is_route_from_current_location_query(self, query: str) -> bool:
        q = normalize_text(query)
        if not q:
            return False
        return any(hint in q for hint in self.CURRENT_LOCATION_ROUTE_HINTS)

    # Regex to match example clauses: "(Ví dụ: X, Y)" or "Ví dụ: X, Y"
    _EXAMPLE_CLAUSE_RE = re.compile(
        r"(?i)\s*(?:\(|\s)(?:ví\s+dụ|vi\s+du|VD|vd|chẳng\s+hạn|chang\s+han|ví\s+dụ\s+như|vi\s+du\s+như)\s*[:：]?\s*[^)]*?(?:\)|$)",
        re.DOTALL,
    )

    def _strip_example_clauses(self, query: str) -> str:
        """Strip 'Ví dụ: X, Y' clauses that are examples, not query targets."""
        cleaned = self._EXAMPLE_CLAUSE_RE.sub("", str(query or "")).strip()
        # Clean up empty parentheses left behind
        cleaned = re.sub(r"\(\s*\)", "", cleaned).strip()
        return cleaned if cleaned else str(query or "")

    def _normalize_typo_query(self, query: str) -> str:
        text = str(query or "")
        if not text:
            return ""
        normalized = text
        for typo, fixed in self.TYPO_NORMALIZATION.items():
            normalized = re.sub(rf"\b{re.escape(typo)}\b", fixed, normalized, flags=re.IGNORECASE)
        return normalized

    def _has_explicit_location(self, entities: List[Dict[str, Any]]) -> bool:
        for entity in entities or []:
            if not isinstance(entity, dict):
                continue
            if str(entity.get("type") or "").strip().lower() == "location":
                return True
        return False

    def _sanitize_rewritten_query(self, original_query: str, rewritten_query: str) -> str:
        if not rewritten_query:
            return original_query

        original_norm = normalize_text(original_query)
        rewritten_norm = normalize_text(rewritten_query)
        if not original_norm or not rewritten_norm:
            return rewritten_query

        is_admin_query = any(hint in original_norm for hint in self.ADMIN_CENTER_HINTS)
        is_route_query = self._is_route_from_current_location_query(original_query)
        if is_admin_query or is_route_query:
            return rewritten_query

        introduced_location_hints = [
            hint for hint in self.LOCATION_LEAKAGE_HINTS
            if hint in rewritten_norm and hint not in original_norm
        ]
        if introduced_location_hints:
            logger.info(
                " Query Analyzer Guardrail: rewritten_query introduced out-of-query locations "
                f"{introduced_location_hints}. Fallback to original query."
            )
            return original_query

        # Layer 2: Detect location entity substitution (e.g. "binh dinh" → "gia lai")
        # If a known location in the original was replaced by a different known location, reject.
        original_locations = {loc for loc in self._KNOWN_LOCATION_NAMES if loc in original_norm}
        rewritten_locations = {loc for loc in self._KNOWN_LOCATION_NAMES if loc in rewritten_norm}
        added_locations = rewritten_locations - original_locations
        removed_locations = original_locations - rewritten_locations
        if added_locations and removed_locations:
            logger.info(
                " Query Analyzer Guardrail: rewritten_query substituted locations "
                f"{removed_locations} -> {added_locations}. Fallback to original query."
            )
            return original_query

        return rewritten_query

    def _is_tourism_analysis_query(self, query: str) -> bool:
        q = normalize_text(self._normalize_typo_query(query))
        if not q:
            return False
        has_analysis_signal = any(hint in q for hint in self.TOURISM_ANALYSIS_HINTS)
        has_real_food_signal = any(hint in q for hint in self.REAL_FOOD_HINTS)
        return has_analysis_signal and not has_real_food_signal

    def _normalize_intents_by_query(self, intents: List[str], query: str) -> List[str]:
        normalized = [str(intent or "").strip().upper() for intent in intents if str(intent or "").strip()]

        # Advice/tips queries → TRAVEL_ADVICE (not recommendation)
        _ADVICE_SIGNALS = ["kinh nghiem", "meo", "luu y", "nen chuan bi", "can biet",
                           "dat phong the nao", "tiet kiem", "tranh bi"]
        q_norm = normalize_text(query, strip_punct=True)
        if any(s in q_norm for s in _ADVICE_SIGNALS):
            return [IntentType.TRAVEL_ADVICE]

        if self._is_tourism_analysis_query(query):
            normalized = [
                intent for intent in normalized
                if intent not in {IntentType.FOOD, IntentType.ACCOMMODATION}
            ]
            if IntentType.TOURISM not in normalized:
                normalized.insert(0, IntentType.TOURISM)
            if IntentType.DISCOVERY not in normalized:
                normalized.append(IntentType.DISCOVERY)
        return normalized or [IntentType.DISCOVERY]

    def _sanitize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out invalid entity names using general Vietnamese linguistic heuristics."""
        cleaned = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "")).strip()
            if not name:
                continue

            # Strip surrounding quotes
            name = name.strip('"\'""''')
            if not name:
                continue

            # Hard char-length cap: reject names that are clearly not entities
            # (e.g. LLM returning the entire query as entity name)
            if len(name) > 60:
                continue

            # General check: name contains Vietnamese question particles
            # These are fundamental grammar markers — any name containing them
            # is a question phrase, not a proper entity
            name_lower = name.lower()
            name_tokens = set(name_lower.split())

            # 1. Check if name contains question particles (single words)
            if name_tokens & self._QUESTION_PARTICLES:
                continue

            # 2. Check if name contains question bigrams (multi-word phrases)
            if any(bigram in name_lower for bigram in self._QUESTION_BIGRAMS):
                continue

            # 3. Skip name too short (1 char)
            if len(name) <= 1:
                continue

            # 4. Check entity type — "Place" is too generic, likely not a real entity
            ent_type = str(ent.get("type", "")).strip()
            if ent_type.lower() in {"place", "location", "unknown", ""}:
                # Only keep generic types if name looks like a proper noun
                # (starts with uppercase in Vietnamese)
                if name[0].islower():
                    continue

            # 5. Correct known location names misclassified by LLM as Dish/Restaurant/etc.
            #    "Gia Lai", "Pleiku", "Quy Nhon" etc. are always Location, never food.
            name_norm = normalize_text(name, strip_punct=True)
            if name_norm in self._BROAD_LOCATION_NAMES and ent_type.lower() not in {"location", "touristattraction"}:
                ent["type"] = "Location"
                logger.debug("[Sanitize] Corrected entity type for '%s': %s -> Location", name, ent_type)

            ent["name"] = name
            cleaned.append(ent)
        return cleaned

    def _recover_missed_entities(self, query: str, current_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scan query for configured domain aliases and inject them if missed by LLM."""
        normalized_query = normalize_text(query)
        existing_norms = {normalize_text(e.get("name", "")) for e in current_entities}
        updated_entities = list(current_entities)
        
        temp_query = normalized_query
        for item in self.flat_aliases:
            norm_v = item["norm_variant"]
            canonical = item["canonical_name"]
            # If the alias is in query and neither its variant nor canonical name has been extracted
            if norm_v in temp_query and canonical not in [e.get("name") for e in updated_entities]:
                # Inject with higher confidence to prioritize recovery representation
                updated_entities.append({
                    "name": canonical,
                    "type": item["type"],
                    "confidence": 0.7,
                    "source": "guardrail_recovery"
                })
                logger.info("[Guardrail Recovery] Injected missed anchor: %s (Confidence: 0.7)", canonical)
                temp_query = temp_query.replace(norm_v, " ")
                
        return updated_entities

    def _extract_entity_from_fact_pattern(self, query: str, current_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Regex fallback: extract entity from 'X thuộc loại hình du lịch nào?' patterns.

        When LLM fails to extract the entity, this method catches structured
        Vietnamese classification query patterns and injects the entity.
        Always runs when classification pattern matches — overrides low-quality LLM entities.
        """
        q_norm = normalize_text(query, strip_punct=True)
        # Only activate for classification query patterns
        classification_signals = ["thuoc loai", "loai hinh", "phan loai", "the loai", "thuoc nhom"]
        if not any(sig in q_norm for sig in classification_signals):
            return current_entities

        # Pattern: "<entity> thuộc loại hình du lịch nào?"
        # Match the original (non-normalized) query to preserve diacritics
        # Handle both diacritized and non-diacritized Vietnamese
        patterns = [
            r"(.+?)\s+(?:thuộc|thuoc|la)\s+(?:loại\s+hình|loai\s+hinh|loại|loai|nhóm|nhom)\s+(?:du\s+lịch\s+|du\s+lich\s+)?(?:nào|nao|gì|gi)",
            r"(.+?)\s+(?:phân\s+loại|phan\s+loai|thể\s+loại|the\s+loai)",
        ]
        for pat in patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                entity_name = m.group(1).strip()
                # Clean up common prefixes
                for prefix in ["các ", "những ", "một ", "con ", "chiếc "]:
                    if entity_name.lower().startswith(prefix):
                        entity_name = entity_name[len(prefix):].strip()
                if entity_name and len(entity_name) >= 3:
                    logger.warning("[EntityFallback] Extracted entity from classification pattern: '%s'", entity_name)
                    return [{
                        "name": entity_name,
                        "type": "TouristAttraction",
                        "confidence": 0.75,
                        "source": "classification_pattern_fallback",
                    }]
        return current_entities

    def _preserve_comparison_subjects(self, query: str, current_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract and preserve comparison subjects from query.

        For queries like "so sánh X và Y", ensure both X and Y are in entities
        with trusted=True to prevent sanitize from removing them.
        """
        q_norm = normalize_text(query, strip_punct=True)

        # Only activate for comparison queries
        if "so sanh" not in q_norm and "khac biet" not in q_norm and "tuong dong" not in q_norm:
            return current_entities

        existing_norms = {normalize_text(e.get("name", "")) for e in current_entities}

        # Pattern 1: "X và Y" after "so sánh"
        # Handle "so sánh vị trí của X và Y"
        comparison_patterns = [
            r"(?:so\s+sánh|khác\s+biệt|tương\s+đồng).*?(?:của|giữa)\s+(.+?)\s+(?:và|va)\s+(.+?)(?:\s+(?:dựa|dua|theo|về|ve|dựa\s+trên)|[?.]|$)",
            r"(?:so\s+sánh|khác\s+biệt|tương\s+đồng)\s+(.+?)\s+(?:và|va)\s+(.+?)(?:\s+(?:dựa|dua|theo|về|ve)|[?.]|$)",
        ]

        subjects = []
        for pat in comparison_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                subj_a = m.group(1).strip()
                subj_b = m.group(2).strip()
                # Clean trailing words
                for trailing in ["dựa trên", "dua tren", "theo", "về", "ve", "dựa"]:
                    subj_a = re.sub(rf"\s+{re.escape(trailing)}\s*$", "", subj_a, flags=re.IGNORECASE).strip()
                    subj_b = re.sub(rf"\s+{re.escape(trailing)}\s*$", "", subj_b, flags=re.IGNORECASE).strip()
                if subj_a and subj_b and len(subj_a) >= 3 and len(subj_b) >= 3:
                    subjects = [subj_a, subj_b]
                    break

        if not subjects:
            return current_entities

        # Inject comparison subjects as trusted entities
        for i, subj in enumerate(subjects):
            subj_norm = normalize_text(subj, strip_punct=True)
            if subj_norm not in existing_norms:
                logger.info("[ComparisonSubject] Preserved subject %s: '%s'", i+1, subj)
                current_entities.append({
                    "name": subj,
                    "type": "TouristAttraction",  # Default type, will be corrected later
                    "confidence": 0.85,
                    "source": "comparison_subject",
                    "trusted": True,
                    "slot": "A" if i == 0 else "B",
                })
                existing_norms.add(subj_norm)

        return current_entities

    def _sanitize_and_dedup_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out blacklisted names and deduplicate entities keeping canonical/recovered ones."""
        seen_norms = set()
        clean_entities = []
        
        # Load greetings and time constraints from blacklist
        blacklist_greetings = self.blacklist.get("greetings", [])
        blacklist_constraints = self.blacklist.get("time_constraints", [])
        
        stop_words = {normalize_text(w) for w in (blacklist_greetings + blacklist_constraints)}
        regex_patterns = [re.compile(p, re.IGNORECASE) for p in self.blacklist.get("regex_patterns", [])]
        
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            if not name:
                continue

            # Strip surrounding quotes
            name = name.strip('"\'""''')
            if not name:
                continue

            norm_name = normalize_text(name).strip()

            # Trusted entities (comparison subjects, pattern-extracted) skip all filters
            is_trusted = entity.get("trusted") is True
            is_pattern_extracted = entity.get("source") in ("classification_pattern_fallback", "guardrail_recovery", "comparison_subject")

            # 1. Sanitize Filters — trusted entities bypass all filters
            if not is_trusted:
                if not is_pattern_extracted and len(norm_name.split()) > 5:
                    continue
                if norm_name in stop_words:
                    continue
                if any(pattern.search(norm_name) for pattern in regex_patterns):
                    continue
                # Filter duration patterns (e.g., "2 ngày 1 đêm", "3n2d", "1 ngay")
                if re.search(r'\b\d+\s*(?:ngay|ngày|dem|đêm|n\s*\d+\s*d)\b', norm_name):
                    continue
                if len(norm_name) <= 1:
                    continue

                # Filter question particles & bigrams
                name_tokens = set(norm_name.split())
                if name_tokens & self._QUESTION_PARTICLES:
                    continue
                if any(bigram in norm_name for bigram in self._QUESTION_BIGRAMS):
                    continue

                # Filter place/location names if lowercase
                ent_type = str(entity.get("type", "")).strip()
                if ent_type.lower() in {"place", "location", "unknown", ""}:
                    if name[0].islower():
                        continue

            # Correct known location names misclassified as Dish/Restaurant/etc.
            # "Bình Định", "Gia Lai", "Pleiku" etc. are always Location, never food.
            name_norm = normalize_text(name, strip_punct=True)
            if name_norm in self._BROAD_LOCATION_NAMES and ent_type.lower() not in {"location", "touristattraction"}:
                entity["type"] = "Location"
                logger.info("   -> [Sanitize] Corrected '%s' type from '%s' to 'Location'", name, ent_type)

            # 2. Deduplication using confidence
            entity["confidence"] = entity.get("confidence", 0.5)
            entity["name"] = name
            
            if norm_name in seen_norms:
                for existing in clean_entities:
                    if normalize_text(existing["name"]).strip() == norm_name:
                        # Keep the one with higher confidence
                        if entity["confidence"] > existing.get("confidence", 0.5):
                            existing["name"] = entity["name"]
                            existing["type"] = entity["type"]
                            existing["confidence"] = entity["confidence"]
                            if "source" in entity:
                                existing["source"] = entity["source"]
                continue
                
            seen_norms.add(norm_name)
            clean_entities.append(entity)
            
        return clean_entities

    def _infer_anchor_type_from_query(self, query: str) -> str:
        """Infer default entity type for v3_router anchors from query context."""
        q = normalize_text(query, strip_punct=True)
        q_bounded = f" {q} "

        # Short words that match as substrings in unrelated Vietnamese words.
        # Require word-boundary matching for these.
        _SHORT_FOOD_SIGNALS = {"pho", "bun", "com", "banh", "nem", "che", "lau", "tra", "nuoc"}
        _SHORT_FOOD_FP = {
            "pho": ["thanh pho", "pho dong", "pho tay", "pho bien", "pho co", "pho di bo", "pho phuong"],
            "com": ["com phai", "com bang"],
            "banh": ["banh rang", "banh tay"],
        }
        # Multi-word food signals (safe to use substring match)
        _MULTI_FOOD_SIGNALS = [
            "mon an", "an gi", "quan an", "nha hang", "dac san", "am thuc",
            "mon ngon", "mon dac san", "an sang", "an trua", "an toi", "an vat",
            "nuong", "hai san", "thuc an", "do an", "do uong", "ca phe",
        ]
        # Check multi-word signals first
        if any(s in q for s in _MULTI_FOOD_SIGNALS):
            return "Dish"
        # Check short signals with word-boundary + false-positive guard
        for sig in _SHORT_FOOD_SIGNALS:
            if f" {sig} " in q_bounded:
                # Reject if in a known false-positive context
                is_fp = any(fp in q for fp in _SHORT_FOOD_FP.get(sig, []))
                if not is_fp:
                    return "Dish"
        # Accommodation signals
        _ACCOM_SIGNALS = [
            "nha nghi", "khach san", "homestay", "resort", "luu tru", "cho nghi",
        ]
        if any(s in q for s in _ACCOM_SIGNALS):
            return "Accommodation"
        # Restaurant signals
        _REST_SIGNALS = ["nha hang", "quan an", "quan nhau", "an uong"]
        if any(s in q for s in _REST_SIGNALS):
            return "Restaurant"
        # Event signals
        _MULTI_EVENT_SIGNALS = [
            "le hoi", "su kien",
            "giai chay", "marathon", "to chuc", "cuoc thi",
            "giai dau", "giao luu", "le ky niem", "dien ra",
        ]
        if any(s in q for s in _MULTI_EVENT_SIGNALS):
            return "Event"
        # Short event signals need word-boundary matching to avoid false positives
        # e.g. "hoi" matching inside "thoi" (thời gian), "le" inside "tople"
        _SHORT_EVENT_SIGNALS = {"le", "hoi"}
        _SHORT_EVENT_FP = {
            "hoi": ["thoi", "khoi", "hoi chu", "hoi thao", "hoi nghi"],
            "le": ["tople", "role", "mile"],
        }
        for sig in _SHORT_EVENT_SIGNALS:
            if f" {sig} " in q_bounded:
                is_fp = any(fp in q for fp in _SHORT_EVENT_FP.get(sig, []))
                if not is_fp:
                    return "Event"
        return "TouristAttraction"  # default

    def _merge_and_route_sources(self, query: str, step1_entities: List[Dict[str, Any]], router_anchors: List[str]) -> List[Dict[str, Any]]:
        """
        Chủ động hợp nhất dữ liệu từ LLM Analyzer và V3 Router.
        Áp dụng khôi phục và làm sạch thực thể.
        """
        merged_pool = []

        # 1. Chuẩn hóa cấu trúc dữ liệu cho Router Anchors
        # Infer type from query context instead of hardcoding TouristAttraction
        default_anchor_type = self._infer_anchor_type_from_query(query)

        # Pre-compute analyzer entity norms for cross-validation
        analyzer_norms = set()
        for entity in (step1_entities or []):
            if isinstance(entity, dict):
                name_val = str(entity.get("name") or "").strip()
                norm = normalize_text(name_val).strip()
                if norm:
                    analyzer_norms.add(norm)

        temp_anchors = []
        for anchor in (router_anchors or []):
            if anchor and isinstance(anchor, str):
                # Cross-validate: if V3 anchor is NOT in analyzer entities,
                # lower confidence (analyzer didn't recognize it as an entity).
                anchor_norm = normalize_text(anchor).strip()
                conf = 0.85 if anchor_norm in analyzer_norms else 0.6
                temp_anchors.append({
                    "name": anchor,
                    "type": default_anchor_type,
                    "confidence": conf,
                    "source": "v3_router"
                })
        
        sanitized_anchors = self._sanitize_and_dedup_entities(temp_anchors)
        merged_pool.extend(sanitized_anchors)
        
        # Router anchor normalized names for primary checks
        router_norms = {normalize_text(a["name"]).strip() for a in sanitized_anchors}
            
        # 2. Đưa các entity từ Step 1 vào pool
        for entity in (step1_entities or []):
            if isinstance(entity, dict):
                ent = dict(entity)
                ent["source"] = ent.get("source", "llm_analyzer")
                ent["confidence"] = ent.get("confidence", 0.5)
                
                name_val = str(ent.get("name") or "").strip()
                norm_name = normalize_text(name_val).strip()

                # LLM entities > 5 tokens bị loại (trừ pattern-extracted và trusted)
                is_trusted = ent.get("trusted") is True
                is_pattern_extracted = ent.get("source") in ("classification_pattern_fallback", "guardrail_recovery", "comparison_subject")
                if not is_trusted and not is_pattern_extracted and len(norm_name.split()) > 5:
                    continue
                    
                # Nếu trùng với router anchor, bỏ qua vì router anchors là PRIMARY
                if norm_name in router_norms:
                    continue
                    
                merged_pool.append(ent)
            
        # 3. Tiến hành khôi phục và khử trùng lặp
        recovered_pool = self._recover_missed_entities(query, merged_pool)
        final_clean_entities = self._sanitize_and_dedup_entities(recovered_pool)
        
        return final_clean_entities



    def _normalize_output(self, raw: Dict[str, Any], query: str) -> Dict[str, Any]:
        data = raw if isinstance(raw, dict) else {}
        query = self._normalize_typo_query(query)

        intents = data.get("intents")
        if not isinstance(intents, list) or not intents:
            single_intent = data.get("intent")
            if isinstance(single_intent, str) and single_intent.strip():
                intents = [single_intent.strip()]
            else:
                intents = [IntentType.DISCOVERY]
        intents = self._normalize_intents_by_query(intents, query)

        rewritten_query = data.get("rewritten_query")
        if not isinstance(rewritten_query, str) or not rewritten_query.strip():
            rewritten_query = query
        rewritten_query = self._sanitize_rewritten_query(query, rewritten_query)

        entities = data.get("entities")
        if not isinstance(entities, list):
            entities = []

        resolved_entities = data.get("resolved_entities")
        if not isinstance(resolved_entities, list):
            resolved_entities = []
        if not entities and resolved_entities:
            entities = resolved_entities

        # Recover missed entities from user query
        entities = self._recover_missed_entities(query, entities)
        resolved_entities = self._recover_missed_entities(query, resolved_entities)

        # Fallback: extract entity from classification patterns if LLM missed it
        entities = self._extract_entity_from_fact_pattern(query, entities)

        # Preserve comparison subjects (e.g., "so sánh X và Y")
        entities = self._preserve_comparison_subjects(query, entities)

        # Sanitize and deduplicate entities
        entities = self._sanitize_and_dedup_entities(entities)
        resolved_entities = self._sanitize_and_dedup_entities(resolved_entities)

        detected_location = data.get("detected_location")
        if not isinstance(detected_location, str) or not detected_location.strip():
            detected_location = None

        search_keywords = data.get("search_keywords")
        if not isinstance(search_keywords, list) or not search_keywords:
            search_keywords = [query]

        constraints = data.get("constraints")
        if not isinstance(constraints, dict):
            constraints = {}
        optimize_distance = bool(constraints.get("optimize_distance", False))

        coreference_confidence = data.get("coreference_confidence")
        try:
            coreference_confidence = float(coreference_confidence)
        except (ValueError, TypeError):
            coreference_confidence = 0.5
        coreference_confidence = max(0.0, min(1.0, coreference_confidence))

        needs_clarification = bool(data.get("needs_clarification", False))
        requested_attributes = list(data.get("requested_attributes") or [])
        requested_relations = list(data.get("requested_relations") or [])

        return {
            "intents": intents,
            "rewritten_query": rewritten_query,
            "entities": entities,
            "resolved_entities": resolved_entities,
            "has_explicit_location": self._has_explicit_location(entities),
            "detected_location": detected_location,
            "search_keywords": search_keywords,
            "constraints": {"optimize_distance": optimize_distance},
            "coreference_confidence": coreference_confidence,
            "needs_clarification": needs_clarification,
            "requested_attributes": requested_attributes,
            "requested_relations": requested_relations,
            "is_follow_up": bool(data.get("is_follow_up", False)),
            "dialog_act": str(data.get("dialog_act") or "NEW_QUERY").strip(),
            "proximity_anchor_type": str(data.get("proximity_anchor_type") or "").strip() or None,
        }

    def analyze(self, query: str, history: List[Dict], current_location: str) -> Dict[str, Any]:
        """
        Phân tích intent và viết lại query dựa trên ngữ cảnh và Schema.
        """
        query = self._normalize_typo_query(query)
        # Strip example clauses ("Ví dụ: X, Y") before entity extraction
        # so the LLM doesn't treat examples as query targets
        query_for_extraction = self._strip_example_clauses(query)
        history_txt = self._format_recent_history(history)
        schema_context = self._compact_schema_context()

        allow_location_context = self._is_route_from_current_location_query(query)
        prompt_location_context = current_location if allow_location_context else ""

        system_prompt = f"""
        You are a Semantic Query Router for a Vietnam Tourism Knowledge Graph.

        ### {business_rules.SYSTEM_FACTS}

        CONTEXT:
        - Current Location: "{prompt_location_context}"
        - Chat History:
        {history_txt}

        ### SCHEMA KNOWLEDGE:
        {schema_context}

        ### GUIDELINES:
        1. REWRITE QUERY: Auto-correct typos, teenspeak (e.g. khum->không, ks->khách sạn) and resolve deictic pronouns using chat history. Keep name keywords (e.g. Quy Nhơn, Pleiku). NEVER substitute one location name for another — if the user says "Bình Định", keep "Bình Định", do NOT replace with "Gia Lai".
        2. INTENT & ENTITY EXTRACTION: Map query to standard intents (ACCOMMODATION_RECOMMENDATION, FOOD_RECOMMENDATION, TOURISM_RECOMMENDATION, EVENT_RECOMMENDATION, TOUR_PLAN, DISTANCE_QUERY, ENTITY_FACT_QUERY, DISCOVERY_SEARCH). Extract specific locations/attractions/dishes/events. Maps "lễ hội", "sự kiện", "hoạt động diễn ra" to EVENT_RECOMMENDATION. Maps food/dish queries ("món X ở đâu bán", "ăn gì ngon", "quán nào bán X") to FOOD_RECOMMENDATION — extract dish name as entity type="Dish".
        3. EXTRACTION SAFETY: Do NOT copy comparative operators or function words (e.g., "so sánh", "khác nhau", "hay hơn", "nửa ngày", "ngày", "đêm", "Xin chào") as entities. Filter out entities longer than 5 words.
        4. IGNORE EXAMPLES: Do NOT extract entities from example clauses introduced by "Ví dụ:", "VD:", "chẳng hạn như:", "ví dụ như:", "(Ví dụ: ...)" — these are user-provided examples, not the actual query targets.
        5. MULTI-INTENT PRIORITY: When query contains food keywords ("quán ăn", "bán", "món", "phở", "đặc sản", "ăn") AND accommodation/event keywords, prioritize FOOD_RECOMMENDATION as primary intent. Always extract dish names as entity type="Dish" when food keywords are present, regardless of other references in the query.
        6. LOCATION ADMIN LEVEL: For Location entities, ALWAYS include "admin_level" field: "province" (tỉnh/thành phố trực thuộc TW) or "ward" (phường/xã/thị trấn — sau sáp nhập 2025, quận/huyện bị bỏ, chỉ còn province→ward). E.g., "Gia Lai" → admin_level="province", "Bình Định" → admin_level="province", "Pleiku" → admin_level="ward", "Quy Nhơn" → admin_level="ward", "Chư Sê" → admin_level="ward".

        FEW-SHOT EXAMPLES:

        Example 1: Direct comparison with greeting & accentless names
        - User: "Xin chào, so sánh Kỳ Co và Eo Gió cái nào đi quan trọng trong nửa ngày"
        - Output JSON:
        {{
            "intents": ["TOURISM_RECOMMENDATION"],
            "rewritten_query": "So sánh bãi biển Kỳ Co và thắng cảnh Eo Gió địa điểm nào nổi bật nên đi trong nửa ngày",
            "entities": [
                {{"name": "Kỳ Co", "type": "TouristAttraction"}},
                {{"name": "Eo Gió", "type": "TouristAttraction"}}
            ],
            "resolved_entities": [
                {{"name": "Kỳ Co", "type": "TouristAttraction", "source": "query"}},
                {{"name": "Eo Gió", "type": "TouristAttraction", "source": "query"}}
            ],
            "detected_location": "Quy Nhơn",
            "search_keywords": ["Kỳ Co", "Eo Gió"],
            "requested_attributes": ["description"],
            "requested_relations": [],
            "constraints": {{
                "optimize_distance": false
            }},
            "coreference_confidence": 1.0,
            "needs_clarification": false,
            "is_follow_up": false,
            "dialog_act": "NEW_QUERY"
        }}

        Example 2: Hidden choice without "so sánh" keyword
        - User: "nen di hon kho hay cu lao xanh"
        - Output JSON:
        {{
            "intents": ["TOURISM_RECOMMENDATION"],
            "rewritten_query": "Nên đi Hòn Khô hay Cù Lao Xanh",
            "entities": [
                {{"name": "Hòn Khô", "type": "TouristAttraction"}},
                {{"name": "Cù Lao Xanh", "type": "TouristAttraction"}}
            ],
            "resolved_entities": [
                {{"name": "Hòn Khô", "type": "TouristAttraction", "source": "query"}},
                {{"name": "Cù Lao Xanh", "type": "TouristAttraction", "source": "query"}}
            ],
            "detected_location": "Quy Nhơn",
            "search_keywords": ["Hòn Khô", "Cù Lao Xanh"],
            "requested_attributes": ["description"],
            "requested_relations": [],
            "constraints": {{
                "optimize_distance": false
            }},
            "coreference_confidence": 1.0,
            "needs_clarification": false,
            "is_follow_up": false,
            "dialog_act": "NEW_QUERY"
        }}

        Example 3: Negative Contrastive (CRITICAL - DO NOT extract comparative phrase as entity)
        - User: "so sanh bien ho va thac phu cuong"
        - Output JSON:
        {{
            "intents": ["TOURISM_RECOMMENDATION"],
            "rewritten_query": "So sánh Biển Hồ và Thác Phú Cường",
            "entities": [
                {{"name": "Biển Hồ", "type": "TouristAttraction"}},
                {{"name": "Thác Phú Cường", "type": "TouristAttraction"}}
            ],
            "resolved_entities": [
                {{"name": "Biển Hồ", "type": "TouristAttraction", "source": "query"}},
                {{"name": "Thác Phú Cường", "type": "TouristAttraction", "source": "query"}}
            ],
            "detected_location": "Gia Lai",
            "search_keywords": ["Biển Hồ", "Thác Phú Cường"],
            "requested_attributes": ["description"],
            "requested_relations": [],
            "constraints": {{
                "optimize_distance": false
            }},
            "coreference_confidence": 1.0,
            "needs_clarification": false,
            "is_follow_up": false,
            "dialog_act": "NEW_QUERY"
        }}

        USER QUERY: "{query}"

        OUTPUT JSON format only:
        {{
            "intents": ["...", "..."],
            "rewritten_query": "...",
            "entities": [
                {{"name": "...", "type": "Location|TouristAttraction|Restaurant|Dish|Duration|GroupSize|Event|..."}}
            ],
            "resolved_entities": [
                {{"name": "...", "type": "Location|TouristAttraction|Restaurant|Dish|Duration|GroupSize|Event|...", "source": "history|query"}}
            ],
            "detected_location": "..." or null,
            "search_keywords": ["..."],
            "requested_attributes": ["address|phone|price|ticket_price|price_range|opening_hours|description|service_features"],
            "requested_relations": ["NEAR|LOCATED_IN|BELONGS_TO|HAS|HELD_AT|INCLUDES|OFFERS"],
            "constraints": {{
                "optimize_distance": true or false
            }},
            "coreference_confidence": 0.0 to 1.0,
            "needs_clarification": true or false,
            "is_follow_up": true or false,
            "dialog_act": "NEW_QUERY|REQUEST_MORE|CLARIFICATION|FOLLOW_UP|SWITCH_TOPIC",
            "proximity_anchor_type": "generic_feature|named_entity" or null
        }}
        """

        system_prompt = f"""
        You are a lightweight Semantic Query Router for a Vietnam Tourism Knowledge Graph.

        FACTS:
        {business_rules.SYSTEM_FACTS}

        CONTEXT:
        - Current Location: "{prompt_location_context}"
        - Recent Chat History:
        {history_txt}

        SCHEMA:
        {schema_context}

        RULES:
        - Return JSON only. Do not explain.
        - Rewrite only to fix typos/teenspeak and resolve clear references from recent history.
        - NEVER substitute one location/province/city name for another. If the user says "Binh Dinh", keep "Binh Dinh" in rewritten_query. If the user says "Quy Nhon", keep "Quy Nhon". Do NOT replace location names even if they are administratively related.
        - Extract only concrete entities from the query or resolved history. Do not extract operators, greetings, durations, or generic phrases as entities.
        - NAME EXTRACTION: When the query has pattern "X tên là Y", "X có tên là Y", "X được gọi là Y", or "X mang tên Y" — the entity name is Y (after "tên là"), NOT X. Do NOT include category words like "quán ăn", "nhà hàng", "khách sạn" as part of the entity name. Example: "quán ăn tên là Bánh xèo tôm nhảy Gia Vỹ" → entity = "Bánh xèo tôm nhảy Gia Vỹ", type = "Restaurant".
        - Ignore examples introduced by "Vi du", "VD", "chang han", or parenthesized example clauses.
        - Map food/dish/restaurant/dining questions to FOOD_RECOMMENDATION (e.g. "món X ở đâu bán", "ăn gì ngon", "quán nào bán X", "ở đâu có X ngon"). Extract the dish/food name as entity type="Dish". buying raw seafood/specialties/souvenirs or local market shopping questions to TRAVEL_ADVICE; hotel/lodging to ACCOMMODATION_RECOMMENDATION; festival/event/race/schedule to EVENT_RECOMMENDATION; route/distance/how-to-go to DISTANCE_QUERY; address/phone/ticket/opening-hours facts to ENTITY_FACT_QUERY; itinerary/plan/schedule/lịch trình/kế hoạch/lộ trình/half-day/nửa ngày to TOUR_PLAN.
        - Use requested_attributes only for explicit facts: address, phone, ticket_price, opening_hours, price, price_range, description, service_features.
        - If pronouns like "cho do", "o do", "noi nay" cannot be resolved, set needs_clarification=true and keep entities empty.
        - Keep entity names short, usually under 5 words, unless it is an official event/tour name.
        - For Location entities, include "admin_level": "province" (tỉnh), "city" (thành phố trực thuộc tỉnh), "district" (quận/huyện/thị xã), "ward" (phường/xã/thị trấn). Use your knowledge of Vietnamese administrative divisions. E.g., "Gia Lai"→province, "Pleiku"→city, "Quy Nhơn"→city, "Chư Sê"→district, "An Nhơn"→city.
        - If the query contains a proximity phrase like "gần X", classify X: if X is a generic geographic/terrain feature (biển, sông, núi, hồ, thác, chợ, công viên, bãi biển, bến cảng, sân bay) set proximity_anchor_type to "generic_feature"; if X is a specific named landmark or place (Eo Gió, Tháp Đôi, Kỳ Co, Biển Hồ T'Nưng) set proximity_anchor_type to "named_entity". If no proximity phrase, omit the field.

        FEW-SHOT EXAMPLES:
        - User: "Khách sạn tốt ở Pleiku, Gia Lai?"
        - Output JSON:
        {{
            "intents": ["ACCOMMODATION_RECOMMENDATION"],
            "rewritten_query": "Khách sạn tốt ở Pleiku, Gia Lai",
            "entities": [{{"name": "Pleiku", "type": "Location", "admin_level": "ward"}}, {{"name": "Gia Lai", "type": "Location", "admin_level": "province"}}],
            "resolved_entities": [{{"name": "Pleiku", "type": "Location", "admin_level": "ward", "source": "query"}}, {{"name": "Gia Lai", "type": "Location", "admin_level": "province", "source": "query"}}],
            "detected_location": "Pleiku, Gia Lai",
            "search_keywords": ["Khách sạn Pleiku Gia Lai"],
            "requested_attributes": ["address", "description"],
            "requested_relations": [],
            "constraints": {{"optimize_distance": false}},
            "coreference_confidence": 1.0,
            "needs_clarification": false,
            "is_follow_up": false,
            "dialog_act": "NEW_QUERY"
        }}

        - User: "Mua hải sản tươi sống ở đâu Quy Nhơn ngon rẻ"
        - Output JSON:
        {{
            "intents": ["TRAVEL_ADVICE"],
            "rewritten_query": "Mua hải sản tươi sống ở đâu tại Quy Nhơn ngon rẻ",
            "entities": [],
            "resolved_entities": [],
            "detected_location": "Quy Nhơn",
            "search_keywords": ["Mua hải sản tươi sống Quy Nhơn"],
            "requested_attributes": ["address", "description"],
            "requested_relations": [],
            "constraints": {{
                "optimize_distance": false
            }},
            "coreference_confidence": 1.0,
            "needs_clarification": false,
            "is_follow_up": false,
            "dialog_act": "NEW_QUERY"
        }}

        - User: "quán ăn tên là Bánh xèo tôm nhảy Gia Vỹ địa chỉ ở đâu"
        - Output JSON:
        {{
            "intents": ["ENTITY_FACT_QUERY"],
            "rewritten_query": "Địa chỉ quán ăn Bánh xèo tôm nhảy Gia Vỹ",
            "entities": [{{"name": "Bánh xèo tôm nhảy Gia Vỹ", "type": "Restaurant"}}],
            "resolved_entities": [{{"name": "Bánh xèo tôm nhảy Gia Vỹ", "type": "Restaurant", "source": "query"}}],
            "detected_location": null,
            "search_keywords": ["Bánh xèo tôm nhảy Gia Vỹ"],
            "requested_attributes": ["address"],
            "requested_relations": [],
            "constraints": {{"optimize_distance": false}},
            "coreference_confidence": 1.0,
            "needs_clarification": false,
            "is_follow_up": false,
            "dialog_act": "NEW_QUERY"
        }}

        - User: "Gợi ý lịch trình nửa ngày ở Quy Nhơn cho du khách lưu trú tại Khách sạn Hữu Phước"
        - Output JSON:
        {{
            "intents": ["TOUR_PLAN"],
            "rewritten_query": "Gợi ý lịch trình nửa ngày ở Quy Nhơn cho du khách lưu trú tại Khách sạn Hữu Phước",
            "entities": [{{"name": "Quy Nhơn", "type": "Location", "admin_level": "city"}}, {{"name": "Khách sạn Hữu Phước", "type": "Accommodation"}}],
            "resolved_entities": [{{"name": "Quy Nhơn", "type": "Location", "admin_level": "city", "source": "query"}}, {{"name": "Khách sạn Hữu Phước", "type": "Accommodation", "source": "query"}}],
            "detected_location": "Quy Nhơn",
            "search_keywords": ["lịch trình nửa ngày Quy Nhơn"],
            "requested_attributes": ["description"],
            "requested_relations": ["NEAR", "LOCATED_IN"],
            "constraints": {{"optimize_distance": true}},
            "coreference_confidence": 1.0,
            "needs_clarification": false,
            "is_follow_up": false,
            "dialog_act": "NEW_QUERY"
        }}

        USER QUERY: "{query}"

        OUTPUT JSON:
        {{
            "intents": ["..."],
            "rewritten_query": "...",
            "entities": [
                {{"name": "...", "type": "Location|TouristAttraction|Restaurant|Dish|Accommodation|Event|Tour|TravelInfo|Duration|GroupSize", "admin_level": "province|ward|null"}}
            ],
            "resolved_entities": [
                {{"name": "...", "type": "Location|TouristAttraction|Restaurant|Dish|Accommodation|Event|Tour|TravelInfo|Duration|GroupSize", "admin_level": "province|ward|null", "source": "history|query"}}
            ],
            "detected_location": "..." or null,
            "search_keywords": ["..."],
            "requested_attributes": ["address|phone|price|ticket_price|price_range|opening_hours|description|service_features"],
            "requested_relations": ["NEAR|LOCATED_IN|BELONGS_TO|HAS|HELD_AT|INCLUDES|OFFERS"],
            "constraints": {{"optimize_distance": true or false}},
            "coreference_confidence": 0.0 to 1.0,
            "needs_clarification": true or false,
            "is_follow_up": true or false,
            "dialog_act": "NEW_QUERY|REQUEST_MORE|CLARIFICATION|FOLLOW_UP|SWITCH_TOPIC",
            "proximity_anchor_type": "generic_feature|named_entity" or null
        }}
        """

        try:
            raw = self.llm_service.generate_json(system_prompt, query_for_extraction)
            return self._normalize_output(raw, query)
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
            logger.error("Query Analyzer LLM failed: %s", e)
            return {
                "intents": [IntentType.DISCOVERY],
                "rewritten_query": query,
                "entities": [],
                "resolved_entities": [],
                "has_explicit_location": False,
                "detected_location": None,
                "search_keywords": [query],
                "constraints": {"optimize_distance": False},
                "coreference_confidence": 0.0,
                "needs_clarification": False,
                "requested_attributes": [],
                "requested_relations": [],
                "is_follow_up": False,
                "dialog_act": "NEW_QUERY",
                "_llm_failed": True,
            }
