from __future__ import annotations

from typing import Any, Dict, List, Sequence
from graph_rag.utils.relation_utils import detect_relation_type
from graph_rag.utils.text import normalize_text


class FactEvidenceReranker:

    def rerank(
        self,
        raw_context: Sequence[str],
        seeds: Sequence[Any],
        query_text: str,
        primary_intent: str,
        relation_priority: Sequence[str],
        metadata: Dict[str, Any] = None,
    ) -> List[str]:
        metadata = metadata or {}

        # 1. Prepare Anchor Names
        anchors = []
        for anchor_name in (metadata.get("query_frame_anchor_names") or []):
            if str(anchor_name).strip():
                anchors.append(str(anchor_name).strip())
        for anchor_item in (metadata.get("anchors") or []):
            if str(anchor_item).strip():
                anchors.append(str(anchor_item).strip())
        intent_data = metadata.get("v3_intent_data") or {}
        for anchor_item in (intent_data.get("anchors") or []):
            if str(anchor_item).strip():
                anchors.append(str(anchor_item).strip())
        anchors = list(dict.fromkeys(anchors))

        # 2. Prepare Seed Names
        seed_names = []
        for seed in (seeds or []):
            meta = getattr(seed, "metadata", {}) or {}
            name = str(meta.get("name") or getattr(seed, "content", "") or "").strip()
            if name:
                seed_names.append(name)

        # 3. Detect Requested Categories from Query
        query_norm = normalize_text(query_text)
        query_categories = []
        category_aliases = [
            ("Di tích lịch sử - Văn hóa", ["di tich lich su", "lich su van hoa"]),
            ("Di tích lịch sử", ["di tich lich su", "lich su van hoa"]),
            ("Danh lam thắng cảnh", ["danh lam thang canh", "danh lam", "danh thang"]),
            ("Làng nghề truyền thống", ["lang nghe truyen thong", "lang nghe"]),
        ]
        for category, aliases in category_aliases:
            if any(alias in query_norm for alias in aliases):
                query_categories.append(category)

        # 4. Get active region info
        legacy_province = metadata.get("legacy_province") or ""
        region_group = metadata.get("region_group") or ""
        relation_priority = list(relation_priority or [])
        retrieval_mode = str(metadata.get("retrieval_plan_mode") or intent_data.get("intent_mode") or "")

        scored_facts = []
        for fact in raw_context:
            fact_str = str(fact or "").strip()
            if not fact_str:
                continue
            score = self.score_fact(
                fact=fact_str,
                query_text=query_text,
                anchors=anchors,
                seed_names=seed_names,
                query_categories=query_categories,
                relation_priority=relation_priority,
                legacy_province=legacy_province,
                region_group=region_group,
                primary_intent=primary_intent,
            )
            scored_facts.append((score, fact_str))

        # Stable sort by score descending
        scored_facts.sort(key=lambda row: row[0], reverse=True)

        # Distribute quota-aware prioritization. This protects comparison/category
        # questions from being dominated by one high-degree anchor.
        prioritized = []
        remaining = list(scored_facts)
        selected_texts = set()

        def add_to_prioritized(score_item):
            text = score_item[1]
            if text not in selected_texts:
                prioritized.append(text)
                selected_texts.add(text)
                if score_item in remaining:
                    remaining.remove(score_item)

        # 0. Ensure at least 1 mandatory evidence fact for each active anchor
        for anchor in anchors:
            anchor_norm = normalize_text(anchor)
            if not anchor_norm:
                continue
            best_fact_item = None
            best_score = -9999.0
            for item in remaining:
                fact_str_norm = normalize_text(item[1])
                if anchor_norm in fact_str_norm:
                    if item[0] > best_score:
                        best_score = item[0]
                        best_fact_item = item
            if best_fact_item:
                add_to_prioritized(best_fact_item)

        # 1. Multi-anchor quota (for comparison or tour_plan)
        if len(anchors) >= 2:
            for anchor in anchors:
                anchor_norm = normalize_text(anchor)
                quota_count = 0
                for item in list(remaining):
                    if quota_count >= 3:
                        break
                    fact_norm = normalize_text(item[1])
                    if anchor_norm and anchor_norm in fact_norm:
                        add_to_prioritized(item)
                        quota_count += 1

        # 2. Category listing quota
        if query_categories:
            for cat in query_categories:
                cat_norm = normalize_text(cat)
                quota_count = 0
                for item in list(remaining):
                    if quota_count >= 3:
                        break
                    fact_norm = normalize_text(item[1])
                    relation = detect_relation_type(item[1], fact_norm)
                    if cat_norm and cat_norm in fact_norm and relation == "BELONGS_TO":
                        add_to_prioritized(item)
                        quota_count += 1
                if quota_count == 0:
                    for item in list(remaining):
                        if quota_count >= 2:
                            break
                        fact_norm = normalize_text(item[1])
                        if cat_norm and cat_norm in fact_norm:
                            add_to_prioritized(item)
                            quota_count += 1

        # 3. For choice/comparison questions, make direct NEAR evidence visible early.
        if retrieval_mode in {"comparison", "multi_candidate", "multi_entity_nearby"}:
            for item in list(remaining):
                if len(prioritized) >= max(6, len(anchors) * 3):
                    break
                fact_norm = normalize_text(item[1])
                if detect_relation_type(item[1], fact_norm) == "NEAR":
                    if any(normalize_text(anchor) in fact_norm for anchor in anchors):
                        add_to_prioritized(item)

        # Append remaining sorted facts
        final_list = prioritized + [item[1] for item in remaining]
        return final_list

    def score_fact(
        self,
        fact: str,
        query_text: str,
        anchors: List[str],
        seed_names: List[str],
        query_categories: List[str],
        relation_priority: Sequence[str],
        legacy_province: str = "",
        region_group: str = "",
        primary_intent: str = "",
    ) -> float:
        score = 0.0
        fact_norm = normalize_text(fact)

        # Rule 1: Contains asked anchors: +40
        for anchor in anchors:
            anchor_norm = normalize_text(anchor)
            if anchor_norm and anchor_norm in fact_norm:
                score += 40.0

        # Rule 2: Contains retrieved seeds: +30
        for seed_name in seed_names:
            seed_norm = normalize_text(seed_name)
            if seed_norm and seed_norm in fact_norm:
                score += 30.0

        # Rule 3: Relation match intent priority relations: +20
        fact_relation = detect_relation_type(fact, fact_norm)
        if fact_relation and fact_relation in relation_priority:
            score += 20.0

        # Rule 4: Contains category from query: +35
        for cat in query_categories:
            cat_norm = normalize_text(cat)
            if cat_norm and cat_norm in fact_norm:
                score += 35.0
                if fact_relation == "BELONGS_TO":
                    score += 15.0
                break

        # Mandatory Evidence Boost:
        boost = 0.0
        q_norm = normalize_text(query_text or "", strip_punct=True)
        
        # 1. ticket_price queries: Facts containing ticket_price, price, or giá vé related to the anchor node receive a boost
        price_signals = ["gia ve", "ve vao", "phi tham quan", "ve tham quan", "ve cong", "ticket_price", "price"]
        if any(sig in q_norm for sig in price_signals):
            fact_price_signals = ["ticket_price", "price", "gia ve", "ve vao", "ve cong", "ve tham quan", "chi phi"]
            if any(sig in fact_norm for sig in fact_price_signals):
                boost += 50.0
                
        # 2. weather queries: TravelInfo facts containing weather, thời tiết, nắng, mưa for the relevant region receive a boost
        weather_signals = ["thoi tiet", "nhiet do", "mua", "nang", "weather", "temperature"]
        if any(sig in q_norm for sig in weather_signals):
            fact_weather_signals = ["weather", "thoi tiet", "nhiet do", "mua", "nang", "khi hau", "bao"]
            if any(sig in fact_norm for sig in fact_weather_signals):
                boost += 50.0
                
        # 3. tour_plan queries: Prioritize facts containing description, address, and order/route for the selected itinerary seeds
        if primary_intent == "TOUR_PLAN":
            tour_signals = ["description", "address", "order", "route", "mo ta", "dia chi", "lich trinh", "ngay 1", "ngay 2", "ngay 3"]
            if any(sig in fact_norm for sig in tour_signals):
                boost += 50.0
                
        # 4. food queries: Boost menu/tag/dish facts for the restaurant nodes
        if primary_intent == "FOOD" or "FOOD" in str(primary_intent).upper():
            food_signals = ["menu", "tag", "dish", "mon an", "thuc don", "dac san", "quan an", "nha hang"]
            if any(sig in fact_norm for sig in food_signals):
                boost += 50.0

        score += boost

        # Rule 5: Positive region signal: +20
        if legacy_province:
            prov_norm = normalize_text(legacy_province)
            if prov_norm and prov_norm in fact_norm:
                score += 20.0
        if region_group:
            group_norm = normalize_text(region_group)
            if group_norm and group_norm in fact_norm:
                score += 20.0

        # Rule 6: Region mismatch penalty: -40
        if legacy_province:
            prov_norm = normalize_text(legacy_province)
            # Check if mentions a different known province without mentioning the correct one
            known_provinces = {
                "binh dinh": ["binh dinh", "quy nhon", "tay son", "phu cat", "an nhon", "nhon hai", "nhon ly", "hoai nhon"],
                "gia lai": ["gia lai", "pleiku", "an khe", "chư prong", "chư se", "kbang", "dak doa", "xuan thuy"],
            }
            target_prov_key = None
            for key in known_provinces:
                if key in prov_norm:
                    target_prov_key = key
                    break

            if target_prov_key:
                mismatch_found = False
                for other_key, variations in known_provinces.items():
                    if other_key == target_prov_key:
                        continue
                    if any(var in fact_norm for var in variations):
                        # check if correct province or its variations are also in the fact to prevent false mismatch
                        if not any(var in fact_norm for var in known_provinces[target_prov_key]):
                            mismatch_found = True
                            break
                if mismatch_found:
                    score -= 40.0

        generic_markers = ["thuoc loai accommodation", "thuoc loai restaurant", "type:", "du lieu thieu"]
        if any(marker in fact_norm for marker in generic_markers):
            score -= 10.0

        return score
