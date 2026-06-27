from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Dict, Any, Optional
from graph_rag.config import cfg
from graph_rag.core.intents import IntentType
from graph_rag.core.state import QuestionShape

from graph_rag.core.candidate_pool import CandidatePool, CandidateScore
from graph_rag.core.feature_extractor import FeatureExtractor
from graph_rag.utils.node_utils import get_node_labels
from graph_rag.utils.text import normalize_text

if TYPE_CHECKING:
    from graph_rag.core.state import NodeItem
    from graph_rag.pipeline.orchestration.exclusion_context import ExclusionContext


class PolicyRanker:
    """Applies shape-aware and policy-aware scoring rules to a CandidatePool.

    Calculates adjustments (policy_score) without mutating the original_score of NodeItem.
    """

    CONSTRAINT_BOOST: float = 3.0       # Boost for candidate matching a hard constraint
    HARD_MISS_PENALTY: float = 1.5      # Penalty for backbone candidate missing a hard constraint

    def __init__(self) -> None:
        self._feature_extractor = FeatureExtractor()
        self._config = cfg

    def rank(
        self,
        pool: CandidatePool,
        exclusion_set: set = None,
        entities: List[Dict[str, Any]] = None,
        region_focus: str = None,
        detected_location: str = None,
        grounded_anchor_nodes: List[Any] = None,
        exclusion_context: Optional["ExclusionContext"] = None,
    ) -> CandidatePool:
        """Ranks candidates inside the pool and returns a new CandidatePool with updated scores.

        Args:
            pool: The candidate pool to rank.
            exclusion_set: Set of normalized entity names to penalize (for follow-up "còn ... khác").
                Deprecated: prefer ``exclusion_context``.
            entities: Runtime list of entities (optional, overrides pool.query_state.metadata).
            region_focus: Runtime region focus (optional, overrides pool.query_state.metadata).
            detected_location: Runtime location context (optional, overrides pool.query_state.metadata).
            grounded_anchor_nodes: Runtime grounded nodes (optional, overrides pool.query_state.metadata).
            exclusion_context: ExclusionContext carrying entity names and flags.
                When provided, takes precedence over ``exclusion_set``.
        """
        if not pool.nodes:
            return pool

        # Prefer ExclusionContext over raw exclusion_set
        if exclusion_context is not None:
            exclusion_set = exclusion_context.entity_names
        else:
            exclusion_set = exclusion_set or set()
        query = pool.query_state.query
        query_norm = normalize_text(query, strip_punct=True)
        policy_dict = pool.policy.to_dict()

        # Resolve parameters dynamically (for compatibility with QueryPlan and legacy QueryState)
        if entities is None:
            entities = getattr(pool.query_state, "entities", None)
            if entities is None and hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
                entities = pool.query_state.metadata.get("entities")
            entities = entities or []
        if region_focus is None:
            region_focus = getattr(pool.query_state, "region_focus", None)
            if region_focus is None and hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
                region_focus = pool.query_state.metadata.get("region_focus")
            region_focus = region_focus or "all"
        if detected_location is None:
            detected_location = ""
            if hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
                detected_location = (
                    pool.query_state.metadata.get("detected_location")
                    or pool.query_state.metadata.get("geo_anchor_location")
                    or ""
                )
            if not detected_location and hasattr(pool.query_state, "get_location_display"):
                detected_location = pool.query_state.get_location_display()
        if grounded_anchor_nodes is None:
            grounded_anchor_nodes = []
            if hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
                grounded_anchor_nodes = pool.query_state.metadata.get("grounded_anchor_nodes") or []

        # 1. Use pool.nodes directly — PolicyRanker applies its own scoring below
        base_ranked = pool.nodes

        # Map to track modifications
        score_breakdown: Dict[str, CandidateScore] = {}

        # 2. Extract refined policy attributes
        primary_labels = pool.policy.primary_labels
        blocked_labels = pool.policy.blocked_labels
        allowed_labels = pool.policy.allowed_labels
        context_budget = pool.policy.context_budget or {}

        # Pre-compute grounded anchor names for anchor boost
        grounded_anchor_names: set = set()
        for anchor in grounded_anchor_nodes:
            if isinstance(anchor, dict):
                anchor_name = normalize_text(str(anchor.get("name") or "").strip(), strip_punct=True)
            else:
                anchor_name = normalize_text(str(getattr(anchor, "content", "") or "").strip(), strip_punct=True)
            if anchor_name:
                grounded_anchor_names.add(anchor_name)
        # Also check target_entity
        target_entity_val = getattr(pool.query_state, "target_entity", None)
        if target_entity_val is None and hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
            target_entity_val = pool.query_state.metadata.get("target_entity")
        target_entity_norm = normalize_text(str(target_entity_val or "").strip(), strip_punct=True)

        shape = getattr(pool.query_state, "question_shape", None) or QuestionShape.UNKNOWN

        for node in base_ranked:
            original_score = next((n.score for n in pool.nodes if n.id == node.id), node.score)
            policy_score = 0.0
            reasons: List[str] = []

            # Extract node labels
            labels = set(get_node_labels(node))

            # A0. Grounded anchor — small tiebreaker only.
            # Semantic relevance is handled by BGE score (Block F below).
            node_name_norm = normalize_text(
                str(node.metadata.get("name") or node.content or "").strip(), strip_punct=True
            )
            is_grounded_anchor = False
            if node_name_norm and grounded_anchor_names:
                for anchor_norm in grounded_anchor_names:
                    if anchor_norm == node_name_norm or anchor_norm in node_name_norm or node_name_norm in anchor_norm:
                        is_grounded_anchor = True
                        break
            if not is_grounded_anchor and target_entity_norm and node_name_norm:
                if target_entity_norm == node_name_norm or target_entity_norm in node_name_norm or node_name_norm in target_entity_norm:
                    is_grounded_anchor = True
            if is_grounded_anchor:
                # When BGE is enabled, anchor is just a tiebreaker (BGE handles semantic relevance).
                # When BGE is disabled, anchor gets a stronger boost to compensate.
                anchor_boost = 0.3 if self._config.is_bge_candidate_scoring_enabled() else 3.0
                reasons.append(f"Grounded anchor tiebreaker (+{anchor_boost})")
                policy_score += anchor_boost

            # A. Label weight boost/penalty
            # Check if any label is blocked
            is_blocked = any(lbl in blocked_labels for lbl in labels)
            if is_blocked:
                policy_score -= 2.0
                reasons.append("Blocked label penalty (-2.0)")

            # Check if off-intent (not allowed)
            is_allowed = any(lbl in allowed_labels for lbl in labels)
            if not is_allowed:
                policy_score -= 1.0
                reasons.append("Off-intent label penalty (-1.0)")

            # A0b. Target class boost — when query has a clear target class
            # (e.g. Accommodation for "nhà nghỉ/hostel"), boost matching labels.
            # Also applies to SINGLE_FACT when intent is EVENT — event queries often
            # use question words ("khi nào", "thời gian") that produce SINGLE_FACT shape,
            # but the target_class (Event) should still be boosted.
            target_class = getattr(pool.query_state, "target_class", None)
            _boost_shapes = {QuestionShape.LIST, QuestionShape.LIST_RANKING,
                             QuestionShape.RECOMMENDATION_LIST, QuestionShape.DISCOVERY}
            primary_intent_val = getattr(pool.query_state, "intent", None)
            if primary_intent_val is None and hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
                primary_intent_val = pool.query_state.metadata.get("intent")
            primary_intent = str(primary_intent_val or "").upper()
            if target_class and (shape in _boost_shapes or primary_intent.startswith("EVENT")):
                if target_class in labels:
                    policy_score += 1.5
                    reasons.append(f"Target class match '{target_class}' (+1.5)")
                else:
                    policy_score -= 0.8
                    reasons.append(f"Non-target label demotion (-0.8)")

            # A0c. Intent-based label boost — when intent clearly signals a label
            # (e.g. EVENT_RECOMMENDATION → Event). This is more reliable than
            # target_class which may be mispredicted by QueryState.
            intent_to_label = {
                IntentType.EVENT: "Event",
                IntentType.ACCOMMODATION: "Accommodation",
                IntentType.FOOD: "Restaurant",
            }
            primary_intent_val = getattr(pool.query_state, "original_intent", None) or getattr(pool.query_state, "intent", None)
            if primary_intent_val is None and hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
                primary_intent_val = pool.query_state.metadata.get("original_intent") or pool.query_state.metadata.get("intent")
            primary_intent = str(primary_intent_val or "").upper()
            expected_label = intent_to_label.get(primary_intent)
            if expected_label and shape in (QuestionShape.LIST, QuestionShape.LIST_RANKING, QuestionShape.RECOMMENDATION_LIST, QuestionShape.DISCOVERY):
                if expected_label in labels:
                    policy_score += 1.0
                    reasons.append(f"Intent-label match '{expected_label}' (+1.0)")
                # Only demote if target_class didn't already handle it
                elif target_class not in labels:
                    policy_score -= 0.5
                    reasons.append(f"Intent-label mismatch (-0.5)")

            # Budget weight boost
            # If label matches context budget, add a slight weight proportional to the budget
            budget_boost = 0.0
            for lbl in labels:
                if lbl in context_budget:
                    budget_boost = max(budget_boost, context_budget[lbl] * 1.5)
            if budget_boost > 0:
                policy_score += budget_boost
                reasons.append(f"Label context budget boost (+{budget_boost:.3f})")

            # B0. Semantic category boost — nature keywords for natural_landmark queries
            semantic_category = pool.query_state.semantic_category
            if semantic_category == "natural_landmark":
                node_name = str(node.metadata.get("name") or node.content or "").strip()
                node_name_norm = normalize_text(node_name, strip_punct=True)
                nature_kws = {"ho", "nui", "thac", "bien", "dao", "dam", "suoi", "ghenh", "eo gio", "nui lua", "ho boi", "ho nuoc", "nui lua chua", "ho tnung", "ho bien"}
                if any(kw in node_name_norm for kw in nature_kws):
                    policy_score += 1.5
                    reasons.append(f"Natural landmark keyword boost (+1.5)")
                # Demote non-nature entities
                demote_kws = {
                    "bao tang", "nha hang", "quan", "khach san", "mon an", "tour",
                    "am thuc", "dac san", "pho", "bun", "com", "hoc vien",
                    "truong", "co so", "cong ty", "nha may", "xa", "phuong",
                    "thi tran", "huyen", "tinh", "khu pho",
                }
                if any(kw in node_name_norm for kw in demote_kws):
                    policy_score -= 2.0
                    reasons.append("Non-nature entity demotion (-2.0)")

            # B0c. Accommodation keyword boost — when target_class is Accommodation,
            # boost accommodation names and demote unrelated types
            if target_class == "Accommodation" and "Accommodation" in labels:
                node_name = str(node.metadata.get("name") or node.content or "").strip()
                node_name_norm = normalize_text(node_name, strip_punct=True)
                accom_kws = {"khach san", "nha nghi", "hostel", "homestay", "resort", "nha khach", "nha tro", "motel", "guest house"}
                if any(kw in node_name_norm for kw in accom_kws):
                    policy_score += 1.0
                    reasons.append("Accommodation keyword boost (+1.0)")

            # B0b. Example anchor boost — user-given examples are strong signals
            example_entities = [e for e in (entities or []) if e.get("example_origin")]
            if example_entities:
                node_name = str(node.metadata.get("name") or node.content or "").strip()
                node_norm = normalize_text(node_name, strip_punct=True)
                for ex in example_entities:
                    ex_norm = normalize_text(str(ex.get("name") or ""), strip_punct=True)
                    if ex_norm and (ex_norm in node_norm or node_norm in ex_norm):
                        policy_score += 2.0
                        reasons.append(f"Example anchor match '{ex.get('name')}' (+2.0)")
                        break

            # B. Exact Match / Proximity Match
            # Exact Match Boost for SINGLE_FACT
            if shape == QuestionShape.SINGLE_FACT:
                target_entity_val = getattr(pool.query_state, "target_entity", None)
                if target_entity_val is None and hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
                    target_entity_val = pool.query_state.metadata.get("target_entity")
                target_entity = target_entity_val or ""
                target_norm = normalize_text(target_entity)
                node_name = str(node.metadata.get("name") or node.content or "").strip()
                node_norm = normalize_text(node_name)

                if target_norm and node_norm == target_norm:
                    policy_score += 1.0
                    reasons.append("Single-fact target entity exact match (+1.0)")
                
                if "exact" in str(node.source_type).lower():
                    policy_score += 0.5
                    reasons.append("Exact search source type boost (+0.5)")

            # Location Match Boost
            # If query location matches node location
            query_loc = ""
            if hasattr(pool.query_state, "metadata") and isinstance(pool.query_state.metadata, dict):
                query_loc = pool.query_state.metadata.get("detected_location") or pool.query_state.metadata.get("geo_anchor_location") or ""
            if not query_loc and hasattr(pool.query_state, "get_location_display"):
                query_loc = pool.query_state.get_location_display()
            query_loc_norm = normalize_text(query_loc)
            if query_loc_norm:
                node_loc = str(node.metadata.get("location") or node.metadata.get("province") or "").strip()
                node_loc_norm = normalize_text(node_loc)
                if node_loc_norm and (node_loc_norm == query_loc_norm or query_loc_norm in node_loc_norm):
                    policy_score += 0.5
                    reasons.append("Location match boost (+0.5)")

            # Relation Match Boost
            if shape == QuestionShape.RECOMMENDATION_LIST:
                if node.source_type == "graph" or "graph" in str(node.source_type).lower():
                    policy_score += 0.3
                    reasons.append("Graph relation match boost (+0.3)")

            # C. Rating / Review Evidence Boost (Quantitative only!)
            if shape in (QuestionShape.LIST_RANKING, QuestionShape.RECOMMENDATION_LIST):
                rating_score, rating_reasons = self._calculate_rating_reviews_boost(node.metadata)
                if rating_score > 0.0:
                    policy_score += rating_score
                    reasons.extend(rating_reasons)

            # C2. Star Rating Boost — when query has quality signals ("tốt", "chất lượng")
            _QUALITY_SIGNALS = {
                "tot", "tot nhat", "chat luong", "uy tin", "danh gia cao",
                "hang sang", "cao cap", "dep", "xinh", "tuyet voi",
                "noi tieng", "tot nhat", "dep nhat",
            }
            if any(sig in query_norm for sig in _QUALITY_SIGNALS):
                star_val = node.metadata.get("star_rating")
                if star_val is not None:
                    try:
                        star = int(star_val)
                        if star >= 4:
                            policy_score += 2.0
                            reasons.append(f"Quality query + star_rating={star} (+2.0)")
                        elif star == 3:
                            policy_score += 0.5
                            reasons.append(f"Quality query + star_rating=3 (+0.5)")
                        elif star <= 2:
                            policy_score -= 1.5
                            reasons.append(f"Quality query + star_rating={star} (-1.5)")
                    except (ValueError, TypeError):
                        pass

            # D. Follow-up exclusion penalty
            if exclusion_set:
                node_name = str(node.metadata.get("name") or node.content or "").strip()
                node_norm = normalize_text(node_name, strip_punct=True)
                if node_norm in exclusion_set:
                    policy_score -= 5.0
                    reasons.append(f"Follow-up exclusion penalty (-5.0): '{node_name}' was already answered")

            # E. Generic constraint scoring (replaces hardcoded coastal/sunset/island)
            if shape == QuestionShape.ITINERARY:
                qs = pool.query_state
                backbone_labels = {"Tour", "TouristAttraction"}
                support_labels = {"Accommodation", "Restaurant", "Dish", "Specialty"}
                is_backbone = bool(labels & backbone_labels)

                for constraint in (qs.constraints or []):
                    feature = getattr(constraint, "feature", str(constraint))
                    weight = getattr(constraint, "weight", 1.0)
                    is_hard = getattr(constraint, "is_hard", False)

                    matched = self._feature_extractor.has_feature(node, feature)
                    if matched:
                        boost = weight * self.CONSTRAINT_BOOST
                        policy_score += boost
                        reasons.append(f"Constraint '{feature}' match (+{boost})")
                    elif is_hard and is_backbone:
                        # Backbone node that doesn't match a hard constraint → penalize
                        policy_score -= self.HARD_MISS_PENALTY
                        reasons.append(f"Constraint '{feature}' hard miss (-{self.HARD_MISS_PENALTY})")

                # Demote support labels when hard constraints exist
                if any(getattr(c, "is_hard", False) for c in (qs.constraints or [])):
                    if any(lbl in labels for lbl in support_labels):
                        policy_score -= 1.0
                        reasons.append("Non-backbone demote (-1.0)")

            # F. BGE score is applied ONLY in bge_scorer.py (single scoring point).
            # No additional BGE weighting here to avoid triple-weighting.

            final_score = round(node.score + policy_score, 3)
            score_breakdown[node.id] = CandidateScore(
                original_score=original_score,
                policy_score=policy_score,
                final_score=final_score,
                reasons=reasons
            )

        # 2b. Filter out candidates with negative final scores.
        # These are topically irrelevant (e.g., "Khách sạn 4-5 sao" for
        # a "xe máy Cù Lao Xanh" query).  Anchor entities typically have
        # high positive scores so they are preserved.
        MIN_CANDIDATE_SCORE = 0.0
        base_ranked = [
            n for n in base_ranked
            if (score_breakdown.get(n.id) or CandidateScore(n.id, n.score, n.score, n.score, {})).final_score >= MIN_CANDIDATE_SCORE
        ]

        # 3. Apply shape-aware diversity sorting
        if shape == QuestionShape.ITINERARY:
            # Let's adjust scores dynamically to enforce category diversity
            sorted_nodes = self._apply_itinerary_diversity(base_ranked, score_breakdown)
        elif shape in (QuestionShape.LIST, QuestionShape.LIST_RANKING, QuestionShape.RECOMMENDATION_LIST, QuestionShape.DISCOVERY):
            # Apply list diversification
            sorted_nodes = self._apply_list_diversity(base_ranked, score_breakdown)
        else:
            # Otherwise sort by final score descending
            sorted_nodes = sorted(
                base_ranked,
                key=lambda n: (score_breakdown.get(n.id) or CandidateScore(n.id, n.score, n.score, n.score, {})).final_score,
                reverse=True
            )

        # Mutate the scores on the nodes themselves in the returned pool so they carry final scores,
        # but we also keep original_score in CandidateScore
        for node in sorted_nodes:
            entry = score_breakdown.get(node.id)
            node.score = entry.final_score if entry else node.score

        return CandidatePool.from_nodes(
            nodes=sorted_nodes,
            query_state=pool.query_state,
            policy=pool.policy,
            score_breakdown=score_breakdown
        )

    @classmethod
    def _calculate_rating_reviews_boost(cls, metadata: Dict[str, Any]) -> tuple[float, List[str]]:
        boost = 0.0
        reasons = []
        
        # 1. rating_evidence_boost (Numeric rating: scale of 1-5 or 1-10)
        rating_keys = ["rating", "stars", "rate", "rating_score"]
        rating_val = None
        for key in rating_keys:
            val = metadata.get(key)
            if val is not None:
                try:
                    f_val = float(val)
                    if 0.0 <= f_val <= 5.0:
                        rating_val = f_val
                        break
                    elif 0.0 <= f_val <= 10.0:
                        rating_val = f_val / 2.0
                        break
                except (ValueError, TypeError):
                    pass
        
        if rating_val is not None:
            # Boost score based on numeric rating: rating * 0.1 (max 0.5 boost)
            rating_boost = round(rating_val * 0.1, 3)
            boost += rating_boost
            reasons.append(f"Rating evidence boost: {rating_val}/5 stars (+{rating_boost})")

        # 2. review_count_boost (Numeric review count)
        review_count_keys = ["reviews_count", "rating_count", "review_count", "num_reviews"]
        review_count = None
        for key in review_count_keys:
            val = metadata.get(key)
            if val is not None:
                try:
                    i_val = int(val)
                    if i_val >= 0:
                        review_count = i_val
                        break
                except (ValueError, TypeError):
                    pass

        if review_count is not None and review_count > 0:
            # Logarithmic review boost to avoid giant numbers skewing too much
            import math
            review_boost = round(math.log10(review_count + 1) * 0.1, 3)
            boost += review_boost
            reasons.append(f"Review count boost: {review_count} reviews (+{review_boost})")

        # 3. reputation_text_boost (Very mild boost for textual descriptors, e.g. "nổi tiếng", "yêu thích")
        desc = str(metadata.get("description") or "").lower()
        reputation_keywords = ["nổi tiếng", "noi tieng", "yêu thích", "yeu thich", "đánh giá cao", "danh gia cao"]
        if any(kw in desc for kw in reputation_keywords):
            reputation_boost = 0.05
            boost += reputation_boost
            reasons.append(f"Reputation text boost (+{reputation_boost})")

        return boost, reasons

    @staticmethod
    def _apply_diversity(
        nodes: List[NodeItem],
        score_breakdown: Dict[str, CandidateScore],
        penalty_fn: Callable[[NodeItem, List[NodeItem]], float],
        penalty_label: str = "diversity",
    ) -> List[NodeItem]:
        """Greedy diversity selection: pick best effective score after penalty.

        Args:
            nodes: Nodes to rank.
            score_breakdown: Mutable score tracking dict.
            penalty_fn: Computes penalty for a candidate given already-ranked nodes.
            penalty_label: Label for the penalty reason string.
        """
        ranked: List[NodeItem] = []
        remaining = list(nodes)

        while remaining:
            best_idx = 0
            best_eff_score = -999999.0
            for idx, node in enumerate(remaining):
                penalty = penalty_fn(node, ranked)
                entry = score_breakdown.get(node.id)
                final_score = entry.final_score if entry else node.score
                eff_score = final_score - penalty
                if eff_score > best_eff_score:
                    best_eff_score = eff_score
                    best_idx = idx

            selected_node = remaining.pop(best_idx)
            entry = score_breakdown.get(selected_node.id)
            original_score = entry.final_score if entry else selected_node.score
            penalty_val = original_score - best_eff_score
            if penalty_val > 0.0 and entry:
                entry.reasons.append(f"{penalty_label.title()} penalty (-{penalty_val:.2f})")
                entry.final_score = round(best_eff_score, 3)

            ranked.append(selected_node)
        return ranked

    @staticmethod
    def _apply_itinerary_diversity(nodes: List[NodeItem], score_breakdown: Dict[str, CandidateScore]) -> List[NodeItem]:
        def _itinerary_penalty(node: NodeItem, ranked: List[NodeItem]) -> float:
            labels = set(get_node_labels(node))
            penalty = 0.0
            for r_node in ranked:
                r_labels = set(get_node_labels(r_node))
                if labels & r_labels:
                    penalty += 0.4
            return penalty

        return PolicyRanker._apply_diversity(nodes, score_breakdown, _itinerary_penalty, "itinerary category diversity")

    @staticmethod
    def _apply_list_diversity(nodes: List[NodeItem], score_breakdown: Dict[str, CandidateScore]) -> List[NodeItem]:
        _parent_keys = ["parent", "parent_id", "parent_entity", "belongs_to", "located_in", "province", "district"]

        def _list_penalty(node: NodeItem, ranked: List[NodeItem]) -> float:
            labels = set(get_node_labels(node))
            parent_vals = set()
            for p_key in _parent_keys:
                val = node.metadata.get(p_key)
                if val:
                    parent_vals.add(str(val).strip().lower())

            penalty = 0.0
            for r_node in ranked:
                r_labels = set(get_node_labels(r_node))
                if labels & r_labels:
                    penalty += 0.3
                r_parent_vals = set()
                for p_key in _parent_keys:
                    val = r_node.metadata.get(p_key)
                    if val:
                        r_parent_vals.add(str(val).strip().lower())
                if parent_vals & r_parent_vals:
                    penalty += 0.3
            return penalty

        return PolicyRanker._apply_diversity(nodes, score_breakdown, _list_penalty, "list diversity")
