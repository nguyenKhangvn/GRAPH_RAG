from __future__ import annotations
"""Text-to-Cypher Retriever — converts natural language queries to Cypher queries.

Strategy 6 in the retrieval pipeline. Uses LLM to generate Cypher queries
from natural language, executes them safely against Neo4j, and returns
NodeItem results.

Safety guards:
- READ-only operations (no CREATE/DELETE/SET/REMOVE/MERGE)
- Query timeout (configurable)
- Result limit (configurable)
- Schema-guided generation (LLM sees only valid node types and relationships)
"""


import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph_rag.core.state import NodeItem

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safety Constants
# ---------------------------------------------------------------------------

ALLOWED_CLAUSES = {
    "MATCH", "WHERE", "RETURN", "ORDER BY", "LIMIT", "SKIP",
    "WITH", "UNWIND", "OPTIONAL MATCH", "CALL", "YIELD",
    "DISTINCT", "AS", "AND", "OR", "NOT", "IN", "CONTAINS",
    "STARTS WITH", "ENDS WITH", "=~", "IS NULL", "IS NOT NULL",
    "CASE", "WHEN", "THEN", "ELSE", "END",
}

FORBIDDEN_CLAUSES = {
    "CREATE", "DELETE", "DETACH DELETE", "SET", "REMOVE",
    "MERGE", "FOREACH", "LOAD CSV", "IMPORT",
}

DEFAULT_MAX_RESULTS = 20
DEFAULT_TIMEOUT_MS = 5000


# ---------------------------------------------------------------------------
# Schema Context for LLM
# ---------------------------------------------------------------------------

_CYPHER_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts" / "cypher"

def _load_json_prompt(filename: str, fallback: dict) -> dict:
    try:
        path = _CYPHER_PROMPTS_DIR / filename
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load prompt json %s: %s", filename, e)
    return fallback

def _load_text_prompt(filename: str, fallback: str) -> str:
    try:
        path = _CYPHER_PROMPTS_DIR / filename
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except OSError as e:
        logger.error("Failed to load prompt text %s: %s", filename, e)
    return fallback

FULL_SCHEMA_CONTEXT = _load_text_prompt("full_schema_context.txt", """## Neo4j Schema (2026-06-14 — khớp DB thực tế: 1,158 nodes)

### Node Types (11 loại):
- TouristAttraction (id, name, description, address, location, category, ticket_price, opening_hours, phone, province, enriched_rating, enriched_rating_source, embedding) — 201 nodes
- Restaurant (id, name, address, location, phone, type, tags, opening_hours, province, email, enriched_rating, enriched_rating_source, embedding) — 132 nodes
- Accommodation (id, name, description, address, location, phone, type, price_range, amenities, capacity, villa_segment, province, enriched_rating, enriched_rating_source, embedding) — 360 nodes. KHÔNG có email property.
- Event (id, name, address, category, month, activities, province, description, image, year, embedding) — 18 nodes. KHÔNG có location property. Dùng province hoặc address để filter.
- Tour (id, name, description, price, duration, start_location, embedding) — 36 nodes
- Dish (id, name, description, category, location, region_group, province, embedding) — 151 nodes. Nhiều node thiếu id/category/location (chỉ ~32%).
- Specialty (id, name, description, category, location, region_group, province, embedding) — 49 nodes
- Location (id, name, region_group, admin_level, admin_status, legacy_district, current_province, old_units, aliases) — 89 nodes
- Category (name) — 10 nodes. Node RIÊNG, kết nối qua BELONGS_TO.
- TravelAgency (id, name, address, phone, email, website, embedding) — 73 nodes
- TravelInfo (id, name, description, topic, location, province) — 32 nodes

### Relationships (13 loại):
- (TouristAttraction)-[:LOCATED_IN]->(Location)      — 511 rels
- (Restaurant)-[:LOCATED_IN]->(Location)               — 316 rels
- (Accommodation)-[:LOCATED_IN]->(Location)            — 868 rels
- (Restaurant)-[:NEAR]->(TouristAttraction)            — 849 rels
- (Accommodation)-[:NEAR]->(TouristAttraction)         — 2856 rels
- (Restaurant)-[:HAS]->(Dish)                          — 132 rels
- (Restaurant)-[:HAS]->(Specialty)                     — 37 rels
- (TouristAttraction)-[:BELONGS_TO]->(Category)        — 192 rels
- (Tour)-[:INCLUDES]->(TouristAttraction)              — 206 rels
- (TravelAgency)-[:OFFERS]->(Tour)                     — 36 rels
- (Event)-[:HELD_AT]->(TouristAttraction)              — 21 rels
- (TravelInfo)-[:Guide_for]->(Location)                — 32 rels
- (Location)-[:SUPERSEDED_BY]->(Location)              — 1 rel (post-2025 merger)

### Location filtering — QUAN TRỌNG:
- TouristAttraction/Restaurant/Accommodation: dùng LOCATED_IN → Location.name
  MATCH (n)-[:LOCATED_IN]->(l:Location) WHERE l.name CONTAINS 'tên tỉnh'
  HOẶC: WHERE n.address CONTAINS 'tên tỉnh'
  HOẶC kết hợp: WHERE l.name CONTAINS '...' OR n.address CONTAINS '...'
- LƯU Ý admin_level: Location có admin_level = ward (thành phố/xã) hoặc province (tỉnh)
  * Pleiku, Quy Nhơn, An Nhơn → admin_level=ward, Location.name = 'Pleiku'
  * Gia Lai, Bình Định → admin_level=province, Location.name = 'Gia Lai'
  * Khi user hỏi 'Pleiku': dùng l.name CONTAINS 'Pleiku', KHÔNG dùng l.name CONTAINS 'Gia Lai'
  * Nếu 0 rows: thử l.current_province CONTAINS 'Gia Lai' hoặc n.address CONTAINS 'Pleiku'
- Event: location suy ra qua HELD_AT → TouristAttraction → LOCATED_IN → Location:
  MATCH (e:Event)-[:HELD_AT]->(t:TouristAttraction)-[:LOCATED_IN]->(l:Location)
  WHERE l.name CONTAINS 'tên tỉnh'
  LƯU Ý: e.address có thể dùng để fallback.
- TravelInfo: dùng Guide_for → Location:
  MATCH (t:TravelInfo)-[:Guide_for]->(l:Location) WHERE l.name CONTAINS 'tên tỉnh'
  Fallback: MATCH (t:TravelInfo) WHERE t.location CONTAINS 'tên tỉnh' OR t.province CONTAINS 'tên tỉnh'
- Specialty/Dish: có location/region_group/province properties → query TRỰC TIẾP:
  MATCH (s:Specialty) WHERE s.location CONTAINS 'Gia Lai' OR s.province CONTAINS 'Gia Lai'
  MATCH (d:Dish) WHERE d.location CONTAINS 'Gia Lai' OR d.province CONTAINS 'Gia Lai'
  HOẶC: (Dish)-[:SPECIALTY_OF]->(Location)

### Location hierarchy:
- Location nodes represent provinces/cities AND xã/phường với: name, admin_level (ward=83, province=3, area=3)
- Location-[:SUPERSEDED_BY]->Location links old provinces to new ones (post-2025 merger)
- Location.name chứa tên tỉnh/thành phố (ví dụ: 'Gia Lai', 'Pleiku', 'Quy Nhơn')
- admin_status: current (50), merged (39)
- region_group: gia_lai_core (40), binh_dinh_legacy (39)

### Rules:
- Use CONTAINS for Vietnamese text matching (not exact match)
- PRESERVE Vietnamese diacritics in search terms (e.g., 'Đại ngàn' not 'Dai ngan')
- Do NOT use toLower() on search terms (it breaks Vietnamese diacritics)
- Split long search phrases into individual words and use OR conditions
  Example: "Lễ hội Tinh hoa Đại ngàn" →
  WHERE e.name CONTAINS 'Tinh hoa' OR e.name CONTAINS 'Đại ngàn'
- Always add LIMIT to prevent large result sets
- Use OPTIONAL MATCH when relationships may not exist
- For province-level queries: do NOT add narrow keyword filters (e.g., CONTAINS 'làng') on attraction names — this misses most results. Only use keyword filters when the user specifically asks for a category (e.g., "làng văn hóa" → filter by name/description)
- IMPORTANT: Category là NODE RIÊNG, KHÔNG phải property. Để lọc theo category, dùng:
  MATCH (ta:TouristAttraction)-[:BELONGS_TO]->(c:Category) WHERE c.name CONTAINS 'Thắng cảnh thiên nhiên'
  KHÔNG dùng ta.category (không tồn tại)
- Category names thực tế trong graph (dùng CONTAINS để match):
  * "Thắng cảnh thiên nhiên" — địa danh tự nhiên, núi, thác, hồ, rừng
  * "Danh lam thắng cảnh" — danh lam, thắng cảnh nổi tiếng
  * "Di tích lịch sử - Văn hóa" / "Di tích lịch sử - văn hóa" — di tích lịch sử (LƯ Ý: viết hoa khác nhau)
  * "Làng văn hóa" — làng văn hóa dân tộc
  * "Làng nghề truyền thống" / "Làng nghề - Văn hóa" / "Làng nghề - Nông nghiệp" — làng nghề
  * "Điểm tham quan" / "Điểm check-in" — điểm tham quan chung
- IMPORTANT: Specialty là node RIÊNG, KHÔNG có address/location. Specialty kết nối qua:
  * (Restaurant)-[:HAS]->(Specialty) — quán phục vụ đặc sản
  KHÔNG dùng Specialty.address hoặc Specialty-[:LOCATED_IN] (không tồn tại).
  Dùng s.location, s.province properties để filter theo vùng.
- IMPORTANT: Dish có thể thiếu id/category/location (chỉ ~32% có). Khi query Dish, dùng OPTIONAL MATCH cho properties.
- Accommodation KHÔNG có email property trong DB (0% coverage).
- Restaurant.email chỉ có 25%, phone chỉ 27% — không nên rely vào.
- IMPORTANT: Khi dùng UNION giữa nhiều label KHÁC NHAU, TẤT CẢ sub-query PHẢI có RETURN clause với CÙNG tên cột (dùng AS alias).
  Nếu label KHÔNG có property nào đó, dùng '' AS alias hoặc NULL AS alias. KHÔNG được đổi tên cột giữa các sub-query.
  Ví dụ ĐÚNG:
    MATCH (d:Dish) WHERE d.name CONTAINS 'x'
    RETURN d.id AS id, d.name AS name, d.description AS description, d.location AS location, d.province AS province, labels(d) AS labels
    UNION
    MATCH (r:Restaurant)-[:HAS]->(d:Dish) WHERE d.name CONTAINS 'x'
    RETURN r.id AS id, r.name AS name, '' AS description, r.address AS location, r.province AS province, labels(r) AS labels
  Ví dụ SAI (KHÔNG được làm):
    RETURN r.id AS id, r.name AS name, r.address AS address, d.name AS dish_name, labels(r) AS labels  ← tên cột khác với sub-query trên!
""")

NODE_SCHEMAS = _load_json_prompt("node_schemas.json", {
    "TouristAttraction": "- TouristAttraction (id, name, description, address, location, category, ticket_price, opening_hours, phone, province, enriched_rating, enriched_rating_source, embedding) — 201 nodes",
    "Restaurant": "- Restaurant (id, name, address, location, phone, type, tags, opening_hours, province, email, enriched_rating, enriched_rating_source, embedding) — 132 nodes",
    "Accommodation": "- Accommodation (id, name, description, address, location, phone, type, price_range, amenities, capacity, villa_segment, province, enriched_rating, enriched_rating_source, embedding) — 360 nodes. KHÔNG có email property.",
    "Event": "- Event (id, name, address, category, month, activities, province, description, image, year, embedding) — 18 nodes. KHÔNG có location property. Dùng province hoặc address để filter.",
    "Tour": "- Tour (id, name, description, price, duration, start_location, embedding) — 36 nodes",
    "Dish": "- Dish (id, name, description, category, location, region_group, province, embedding) — 151 nodes. Nhiều node thiếu id/category/location (chỉ ~32%).",
    "Specialty": "- Specialty (id, name, description, category, location, region_group, province, embedding) — 49 nodes",
    "Location": "- Location (id, name, region_group, admin_level, admin_status, legacy_district, current_province, old_units, aliases) — 89 nodes",
    "Category": "- Category (name) — 10 nodes. Node RIÊNG, kết nối qua BELONGS_TO.",
    "TravelAgency": "- TravelAgency (id, name, address, phone, email, website, embedding) — 73 nodes",
    "TravelInfo": "- TravelInfo (id, name, description, topic, location, province) — 32 nodes\n    Kết nối Location qua [:Guide_for] — thay thế LOCATED_IN cũ.",
})

RELATIONSHIP_SCHEMAS = _load_json_prompt("relationship_schemas.json", {
    "TouristAttraction_LOCATED_IN_Location": "- (TouristAttraction)-[:LOCATED_IN]->(Location)      — 511 rels",
    "Restaurant_LOCATED_IN_Location": "- (Restaurant)-[:LOCATED_IN]->(Location)               — 316 rels",
    "Accommodation_LOCATED_IN_Location": "- (Accommodation)-[:LOCATED_IN]->(Location)            — 868 rels",
    "Restaurant_NEAR_TouristAttraction": "- (Restaurant)-[:NEAR]->(TouristAttraction)            — 849 rels",
    "Accommodation_NEAR_TouristAttraction": "- (Accommodation)-[:NEAR]->(TouristAttraction)         — 2856 rels",
    "Restaurant_HAS_Dish": "- (Restaurant)-[:HAS]->(Dish)                          — 169 rels",
    "Dish_SPECIALTY_OF_Location": "- (Dish)-[:SPECIALTY_OF]->(Location)                  — 48 rels\n  (Dish/Specialty cũng có location/region_group/province properties để fallback)",
    "TouristAttraction_BELONGS_TO_Category": "- (TouristAttraction)-[:BELONGS_TO]->(Category)        — 192 rels",
    "Tour_INCLUDES_TouristAttraction": "- (Tour)-[:INCLUDES]->(TouristAttraction)              — 206 rels",
    "TravelAgency_OFFERS_Tour": "- (TravelAgency)-[:OFFERS]->(Tour)                     — 36 rels",
    "Event_HELD_AT_TouristAttraction": "- (Event)-[:HELD_AT]->(TouristAttraction)              — 21 rels",
    "TravelInfo_Guide_for_Location": "- (TravelInfo)-[:Guide_for]->(Location)                — 32 rels\n  (Event location qua HELD_AT → TouristAttraction → LOCATED_IN → Location)",
    "Location_SUPERSEDED_BY_Location": "- (Location)-[:SUPERSEDED_BY]->(Location)              — 1 rel (post-2025 merger)\n  (Event location suy ra qua HELD_AT → TouristAttraction → LOCATED_IN → Location)",
})

LOCATION_FILTER_RULES = _load_json_prompt("location_filter_rules.json", {
    "TouristAttraction": "- TouristAttraction: dùng LOCATED_IN → Location.name\n  MATCH (n)-[:LOCATED_IN]->(l:Location) WHERE l.name CONTAINS 'tên tỉnh'\n  HOẶC: WHERE n.address CONTAINS 'tên tỉnh'\n  HOẶC kết hợp: WHERE l.name CONTAINS '...' OR n.address CONTAINS '...'\n  LƯU Ý: Pleiku/Quy Nhơn là ward (l.name='Pleiku'), Gia Lai/Bình Định là province. Dùng đúng tên Location.name.",
    "Restaurant": "- Restaurant: dùng LOCATED_IN → Location.name\n  MATCH (n)-[:LOCATED_IN]->(l:Location) WHERE l.name CONTAINS 'tên tỉnh'\n  HOẶC: WHERE n.address CONTAINS 'tên tỉnh'\n  HOẶC kết hợp: WHERE l.name CONTAINS '...' OR n.address CONTAINS '...'\n  LƯU Ý: Pleiku/Quy Nhơn là ward (l.name='Pleiku'), Gia Lai/Bình Định là province. Nếu 0 rows, thử l.current_province.",
    "Accommodation": "- Accommodation: dùng LOCATED_IN → Location.name\n  MATCH (n)-[:LOCATED_IN]->(l:Location) WHERE l.name CONTAINS 'tên tỉnh'\n  HOẶC: WHERE n.address CONTAINS 'tên tỉnh'\n  HOẶC kết hợp: WHERE l.name CONTAINS '...' OR n.address CONTAINS '...'\n  LƯU Ý: Pleiku/Quy Nhơn là ward (l.name='Pleiku'), Gia Lai/Bình Định là province.",
    "Event": "- Event: location suy ra qua HELD_AT → TouristAttraction → LOCATED_IN → Location:\n  MATCH (e:Event)-[:HELD_AT]->(t:TouristAttraction)-[:LOCATED_IN]->(l:Location) WHERE l.name CONTAINS 'tên tỉnh'\n  Fallback: MATCH (e:Event) WHERE e.address CONTAINS 'tên tỉnh'",
    "TravelInfo": "- TravelInfo: dùng Guide_for → Location (ưu tiên):\n  MATCH (t:TravelInfo)-[:Guide_for]->(l:Location) WHERE l.name CONTAINS 'tên tỉnh'\n  Fallback: MATCH (t:TravelInfo) WHERE t.location CONTAINS 'tên tỉnh' OR t.province CONTAINS 'tên tỉnh'",
    "Specialty": "- Specialty: có location/region_group/province properties → query TRỰC TIẾP:\n  MATCH (s:Specialty) WHERE s.location CONTAINS 'Gia Lai' OR s.province CONTAINS 'Gia Lai'",
    "Dish": "- Dish: query qua location/region_group/province properties hoặc SPECIALTY_OF:\n  MATCH (d:Dish)-[:SPECIALTY_OF]->(l:Location) WHERE l.name CONTAINS 'tên tỉnh'\n  Fallback: MATCH (d:Dish) WHERE d.location CONTAINS 'tên tỉnh' OR d.province CONTAINS 'tên tỉnh'",
})

LOCATION_HIERARCHY = _load_text_prompt("location_hierarchy.txt", """### Location hierarchy:
- Location nodes represent provinces/cities AND xã/phường với: name, admin_level (ward=83, province=3, area=3)
- Location-[:SUPERSEDED_BY]->Location links old provinces to new ones (post-2025 merger)
- Location.name chứa tên tỉnh/thành phố (ví dụ: 'Gia Lai', 'Pleiku', 'Quy Nhơn')
- admin_status: current (50), merged (39)
- region_group: gia_lai_core (40), binh_dinh_legacy (39)

### Location filtering theo admin_level — QUAN TRỌNG:
- Khi user hỏi về thành phố (Pleiku, Quy Nhơn, An Nhơn):
  Location.name = tên thành phố (admin_level=ward)
  → MATCH (n)-[:LOCATED_IN]->(l:Location) WHERE l.name CONTAINS 'Pleiku'
  HOẶC nếu cần cả tỉnh: WHERE l.name CONTAINS 'Pleiku' OR l.current_province CONTAINS 'Gia Lai'
- Khi user hỏi về tỉnh (Gia Lai, Bình Định):
  Location.name = tên tỉnh (admin_level=province)
  → MATCH (n)-[:LOCATED_IN]->(l:Location) WHERE l.name CONTAINS 'Gia Lai'
  HOẶC dùng region_group: WHERE l.region_group = 'gia_lai_core'
- KHÔNG dùng l.name CONTAINS 'Gia Lai' khi user nói 'Pleiku' — Pleiku là ward, KHÔNG phải province name
- Nếu query trả 0 rows với Location.name, thử dùng n.address CONTAINS hoặc l.current_province CONTAINS""")

GENERAL_RULES = _load_text_prompt("general_rules.txt", """- Use CONTAINS for Vietnamese text matching (not exact match)
- PRESERVE Vietnamese diacritics in search terms (e.g., 'Đại ngàn' not 'Dai ngan')
- Do NOT use toLower() on search terms (it breaks Vietnamese diacritics)
- Split long search phrases into individual words and use OR conditions
  Example: "Lễ hội Tinh hoa Đại ngàn" →
  WHERE e.name CONTAINS 'Tinh hoa' OR e.name CONTAINS 'Đại ngàn'
- Always add LIMIT to prevent large result sets
- Use OPTIONAL MATCH when relationships may not exist
- For province-level queries: do NOT add narrow keyword filters (e.g., CONTAINS 'làng') on attraction names — this misses most results. Only use keyword filters when the user specifically asks for a category (e.g., "làng văn hóa" → filter by name/description)
- IMPORTANT: Category là NODE RIÊNG, KHÔNG phải property. Để lọc theo category, dùng:
  MATCH (ta:TouristAttraction)-[:BELONGS_TO]->(c:Category) WHERE c.name CONTAINS 'Thắng cảnh thiên nhiên'
  KHÔNG dùng ta.category (không tồn tại)
- Category names thực tế trong graph (dùng CONTAINS để match):
  * "Thắng cảnh thiên nhiên" — địa danh tự nhiên, núi, thác, hồ, rừng
  * "Danh lam thắng cảnh" — danh lam, thắng cảnh nổi tiếng
  * "Di tích lịch sử - Văn hóa" / "Di tích lịch sử - văn hóa" — di tích lịch sử (LƯ Ý: viết hoa khác nhau)
  * "Làng văn hóa" — làng văn hóa dân tộc
  * "Làng nghề truyền thống" / "Làng nghề - Văn hóa" / "Làng nghề - Nông nghiệp" — làng nghề
  * "Điểm tham quan" / "Điểm check-in" — điểm tham quan chung
- IMPORTANT: Specialty là node RIÊNG, KHÔNG có address/location. Specialty kết nối qua:
  * (Restaurant)-[:HAS]->(Specialty) — quán phục vụ đặc sản
  KHÔNG dùng Specialty.address hoặc Specialty-[:LOCATED_IN] (không tồn tại).
  Dùng s.location, s.province properties để filter theo vùng.
- IMPORTANT: Dish có thể thiếu id/category/location (chỉ ~32% có). Khi query Dish, dùng OPTIONAL MATCH cho properties.
- Accommodation KHÔNG có email property trong DB (0% coverage).
- Restaurant.email chỉ có 25%, phone chỉ 27% — không nên rely vào.""")

INTENT_SCHEMA_MAP = {
    "EVENT_RECOMMENDATION": {
        "nodes": ["Event", "TouristAttraction", "Location"],
        "relationships": ["Event_HELD_AT_TouristAttraction", "TouristAttraction_LOCATED_IN_Location"],
        "location_rules": ["Event", "TouristAttraction"],
    },
    "FOOD_RECOMMENDATION": {
        "nodes": ["Restaurant", "Dish", "Specialty", "Location"],
        "relationships": ["Restaurant_LOCATED_IN_Location", "Restaurant_HAS_Dish", "Dish_SPECIALTY_OF_Location"],
        "location_rules": ["Restaurant", "Dish", "Specialty"],
    },
    "ACCOMMODATION_RECOMMENDATION": {
        "nodes": ["Accommodation", "Location", "TouristAttraction"],
        "relationships": ["Accommodation_LOCATED_IN_Location", "Accommodation_NEAR_TouristAttraction"],
        "location_rules": ["Accommodation"],
    },
    "TOURISM_RECOMMENDATION": {
        "nodes": ["TouristAttraction", "Location", "Category"],
        "relationships": ["TouristAttraction_LOCATED_IN_Location", "TouristAttraction_BELONGS_TO_Category"],
        "location_rules": ["TouristAttraction"],
    },
    "TOUR_PLAN": {
        "nodes": ["Tour", "TouristAttraction", "TravelAgency", "Location"],
        "relationships": [
            "Tour_INCLUDES_TouristAttraction",
            "TravelAgency_OFFERS_Tour",
            "TouristAttraction_LOCATED_IN_Location"
        ],
        "location_rules": ["TouristAttraction"],
    },
    "TRAVEL_AGENCY": {
        "nodes": ["TravelAgency", "Tour", "Location"],
        "relationships": ["TravelAgency_OFFERS_Tour"],
        "location_rules": [],
    },
    "TRAVEL_ADVICE": {
        "nodes": ["TravelInfo", "TouristAttraction", "Restaurant", "Specialty", "Location"],
        "relationships": ["TouristAttraction_LOCATED_IN_Location", "Restaurant_LOCATED_IN_Location", "TravelInfo_Guide_for_Location"],
        "location_rules": ["TravelInfo", "TouristAttraction", "Restaurant"],
    },
    "DISCOVERY_SEARCH": {
        "nodes": ["TouristAttraction", "Restaurant", "Accommodation", "Event", "TravelInfo", "Location"],
        "relationships": ["TouristAttraction_LOCATED_IN_Location", "Restaurant_LOCATED_IN_Location", "Accommodation_LOCATED_IN_Location", "Event_HELD_AT_TouristAttraction", "TravelInfo_Guide_for_Location"],
        "location_rules": ["TouristAttraction", "Restaurant", "Accommodation"],
    },
}

def build_schema_context(intent: str = "") -> str:
    """Build minimised schema context dynamically based on active query intent."""
    intent_upper = str(intent).strip().upper()
    if intent_upper in INTENT_SCHEMA_MAP:
        config = INTENT_SCHEMA_MAP[intent_upper]
        nodes = config["nodes"]
        rels = config["relationships"]
        loc_rules_keys = config["location_rules"]

        # Build nodes block
        node_lines = [NODE_SCHEMAS[node] for node in nodes if node in NODE_SCHEMAS]
        node_block = "### Node Types:\n" + "\n".join(node_lines)

        # Build relationships block
        rel_lines = [RELATIONSHIP_SCHEMAS[rel] for rel in rels if rel in RELATIONSHIP_SCHEMAS]
        rel_block = "### Relationships:\n" + "\n".join(rel_lines)

        # Build location rules block
        loc_rule_lines = [LOCATION_FILTER_RULES[k] for k in loc_rules_keys if k in LOCATION_FILTER_RULES]
        loc_rules_block = "### Location filtering — QUAN TRỌNG:\n" + "\n".join(loc_rule_lines)

        # Filter relevant rules from GENERAL_RULES
        relevant_rules = []
        for line in GENERAL_RULES.splitlines():
            line_strip = line.strip()
            if not line_strip:
                continue

            contains_unrelated = False
            for node_name in NODE_SCHEMAS.keys():
                if node_name not in nodes and f"{node_name}." in line_strip:
                    contains_unrelated = True
                    break

            if not contains_unrelated:
                relevant_rules.append(line)

        rules_block = "### Rules:\n" + "\n".join(relevant_rules)

        parts = [
            f"## Neo4j Schema (Minimized for {intent_upper})",
            node_block,
            rel_block,
            loc_rules_block,
            LOCATION_HIERARCHY,
            rules_block
        ]
        return "\n\n".join(parts)

    return FULL_SCHEMA_CONTEXT


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------

CYPHER_PROMPT = """Bạn là chuyên gia Neo4j Cypher. Hãy chuyển câu hỏi của người dùng thành truy vấn Cypher.

{schema_context}

## Câu hỏi của người dùng:
{query}

{context_hint}

## Yêu cầu:
1. Chỉ tạo truy vấn READ (MATCH, RETURN, WHERE, ORDER BY, LIMIT)
2. KHÔNG tạo truy vấn WRITE (CREATE, DELETE, SET, REMOVE, MERGE)
3. Luôn thêm LIMIT (tối đa {max_results})
4. Dùng CONTAINS cho text matching (tiếng Việt)
5. Dùng OPTIONAL MATCH khi relationship có thể không tồn tại
6. Nếu có thông tin khu vực/location, BẮT BUỘC thêm điều kiện WHERE để lọc theo Location.name (CHỨA tên tỉnh) hoặc node.address. Dùng Location.name CONTAINS 'tên tỉnh' để match Location node theo tỉnh.
7. Nếu câu hỏi là follow-up (ví dụ: "còn món nào khác", "có gì nữa không"), hãy truy vấn các entity KHÁC với kết quả trước đó
8. BẮT BUỘC luôn dùng ALIAS và luôn trả về id của node chính trong RETURN clause: RETURN n.id AS id, n.name AS name, n.description AS description, labels(n) AS labels. Luôn include labels(n) AS labels để xác định loại node.
9. Khi dùng UNION giữa nhiều label KHÁC NHAU, TẤT CẢ sub-query PHẢI có RETURN clause với CÙNG tên cột (dùng AS alias). Nếu label KHÔNG có property nào đó, dùng '' AS alias hoặc NULL AS alias. KHÔNG được đổi tên cột giữa các sub-query.
10. Trả về JSON với format:
{{"cypher": "MATCH ... RETURN ...", "explanation": "Giải thích ngắn gọn"}}

## Cypher query:"""


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CypherResult:
    """Result from text-to-cypher generation."""
    cypher: str
    explanation: str
    success: bool
    error: Optional[str] = None


@dataclass
class CypherRow:
    """Single row from Cypher execution."""
    data: Dict[str, Any]


# ---------------------------------------------------------------------------
# TextToCypherGenerator
# ---------------------------------------------------------------------------

class TextToCypherGenerator:
    """Generates Cypher queries from natural language using LLM."""

    def __init__(self, llm_service):
        self.llm = llm_service

    def generate(
        self,
        query: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        context_hint: str = "",
        intent: str = "",
    ) -> CypherResult:
        """Generate Cypher query from natural language.

        Args:
            query: Natural language query in Vietnamese
            max_results: Maximum number of results to return
            context_hint: Additional context (location, intent, follow-up info)
            intent: Business intent for schema filtering

        Returns:
            CypherResult with generated Cypher query
        """
        system_prompt = "Bạn là chuyên gia Neo4j Cypher. Chỉ trả về JSON hợp lệ."
        schema_context = build_schema_context(intent)
        user_prompt = CYPHER_PROMPT.format(
            schema_context=schema_context,
            query=query,
            max_results=max_results,
            context_hint=context_hint,
        )

        try:
            # Use generate_json for structured output
            response = self.llm.generate_json(system_prompt, user_prompt)
            if isinstance(response, dict):
                cypher = str(response.get("cypher", "")).strip()
                explanation = str(response.get("explanation", "")).strip()
                if cypher:
                    return CypherResult(
                        cypher=cypher,
                        explanation=explanation,
                        success=True,
                    )
            # Fallback: try text generation
            response_text = self.llm.generate_text(system_prompt, user_prompt)
            return self._parse_response(str(response_text))
        except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
            logger.error("Text-to-Cypher generation failed: %s", e)
            return CypherResult(
                cypher="",
                explanation="",
                success=False,
                error=str(e),
            )

    def _parse_response(self, response: str) -> CypherResult:
        """Parse LLM response to extract Cypher query."""
        import json

        # Try JSON parse first
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*"cypher"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                cypher = str(data.get("cypher", "")).strip()
                explanation = str(data.get("explanation", "")).strip()
                if cypher:
                    return CypherResult(
                        cypher=cypher,
                        explanation=explanation,
                        success=True,
                    )
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: extract Cypher from code blocks
        code_match = re.search(r'```(?:cypher)?\s*(.*?)```', response, re.DOTALL)
        if code_match:
            cypher = code_match.group(1).strip()
            return CypherResult(
                cypher=cypher,
                explanation="",
                success=True,
            )

        # Fallback: look for MATCH statement
        match_match = re.search(r'(MATCH\s+.*?RETURN\s+.*?)(?:\n|$)', response, re.DOTALL | re.IGNORECASE)
        if match_match:
            cypher = match_match.group(1).strip()
            return CypherResult(
                cypher=cypher,
                explanation="",
                success=True,
            )

        return CypherResult(
            cypher="",
            explanation="",
            success=False,
            error="Could not parse Cypher from LLM response",
        )


# ---------------------------------------------------------------------------
# CypherExecutor
# ---------------------------------------------------------------------------

class CypherExecutor:
    """Safely executes Cypher queries against Neo4j."""

    def __init__(self, driver, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self.driver = driver
        self.timeout_ms = timeout_ms

    @staticmethod
    def _fix_exists_syntax(cypher: str) -> str:
        """Auto-rewrite old-style EXISTS((pattern) WHERE ...) to Neo4j 5.x EXISTS { MATCH ... WHERE ... }.

        The LLM sometimes generates the deprecated form:
            EXISTS((r)-[:LOCATED_IN]->(:Location) WHERE l.name CONTAINS 'X')
        Correct Neo4j 5.x syntax:
            EXISTS { MATCH (r)-[:LOCATED_IN]->(l:Location) WHERE l.name CONTAINS 'X' }
        """
        result = cypher
        i = 0
        while i < len(result):
            m = re.search(r'EXISTS\s*\(', result[i:], re.IGNORECASE)
            if not m:
                break
            start = i + m.start()
            paren_start = i + m.end() - 1  # position of opening (

            # Find matching closing paren by counting depth
            depth = 0
            j = paren_start
            while j < len(result):
                if result[j] == '(':
                    depth += 1
                elif result[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1

            if depth != 0:
                i = j + 1
                continue

            inner = result[paren_start + 1:j]
            end = j + 1

            # Only rewrite old syntax: has WHERE but not Neo4j 5.x { MATCH ... }
            inner_upper = inner.upper().strip()
            if 'WHERE' not in inner_upper or inner_upper.startswith('{'):
                i = end
                continue

            # Split pattern and WHERE clause
            # Find WHERE that's not inside nested parens/brackets
            where_idx = -1
            depth2 = 0
            for k, ch in enumerate(inner):
                if ch in '([':
                    depth2 += 1
                elif ch in ')]':
                    depth2 -= 1
                elif depth2 == 0 and inner[k:k+5].upper() == 'WHERE':
                    where_idx = k
                    break

            if where_idx < 0:
                i = end
                continue

            pattern = inner[:where_idx].strip()
            where_clause = inner[where_idx + 5:].strip()

            # Fix unbound node variables: (:Label) → (v:Label)
            def _bind_node(nm: re.Match) -> str:
                var = nm.group(1)
                label = nm.group(2)
                if not var:
                    v = label[0].lower() if label else "n"
                    return f"({v}:{label})"
                return nm.group(0)

            pattern_fixed = re.sub(
                r'\(\s*(\w*)\s*:\s*(\w+)\s*\)',
                _bind_node,
                pattern
            )

            replacement = f"EXISTS {{ MATCH {pattern_fixed} WHERE {where_clause} }}"
            result = result[:start] + replacement + result[end:]
            i = start + len(replacement)

        return result

    # Properties per label — loaded from label_properties.json for easy editing.
    _LABEL_PROPERTIES: dict[str, list[str]] = _load_json_prompt("label_properties.json", {})

    @classmethod
    def _detect_label(cls, part: str) -> Optional[str]:
        """Detect the primary node label from a MATCH clause."""
        m = re.search(r'MATCH\s+\(\s*\w+\s*:\s*(\w+)', part, re.IGNORECASE)
        if m:
            label = m.group(1)
            if label in cls._LABEL_PROPERTIES:
                return label
        return None

    @classmethod
    def _fix_union_columns(cls, cypher: str) -> str:
        """Auto-fix UNION column mismatches.

        Neo4j requires all UNION sub-queries to have identical RETURN column names.
        If the LLM generates mismatched columns, this method attempts to unify them
        by aliasing missing columns with NULL/empty string — but uses the actual
        node properties when possible instead of always falling back to NULL.
        """
        # Split by UNION (case-insensitive)
        parts = re.split(r'\bUNION\b', cypher, flags=re.IGNORECASE)
        if len(parts) <= 1:
            return cypher

        # Extract RETURN column aliases from each part
        def extract_aliases(part: str) -> list[str]:
            ret_match = re.findall(r'\bRETURN\s+(.+?)(?:\s+ORDER\b|\s+LIMIT\b|\s+$)',
                                   part, re.IGNORECASE | re.DOTALL)
            if not ret_match:
                return []
            ret_clause = ret_match[-1]
            aliases = re.findall(r'\bAS\s+(\w+)', ret_clause, re.IGNORECASE)
            return aliases

        all_aliases = [extract_aliases(p) for p in parts]

        if not all_aliases or any(len(a) == 0 for a in all_aliases):
            return cypher

        # Check if all already match
        first = all_aliases[0]
        if all(a == first for a in all_aliases):
            return cypher

        # Build unified column list
        unified = first[:]
        for aliases in all_aliases[1:]:
            for alias in aliases:
                if alias not in unified:
                    unified.append(alias)

        logger.warning("UNION column mismatch detected. Unifying to: %s", unified)

        # Detect label for each sub-query
        labels = [cls._detect_label(p) for p in parts]

        # Rewrite each sub-query's RETURN clause
        fixed_parts = []
        for i, part in enumerate(parts):
            aliases = all_aliases[i]
            if aliases == unified:
                fixed_parts.append(part)
                continue

            ret_match = re.search(r'\bRETURN\s+', part, re.IGNORECASE)
            if not ret_match:
                fixed_parts.append(part)
                continue

            ret_start = ret_match.end()
            rest = part[ret_start:]
            end_match = re.search(r'\s+(ORDER|LIMIT)\b', rest, re.IGNORECASE)
            ret_end = ret_start + (end_match.start() if end_match else len(rest))
            original_ret = part[ret_start:ret_end]

            # Determine the primary node variable (first variable in MATCH)
            node_var_match = re.search(r'MATCH\s+\(\s*(\w+)\s*:', part, re.IGNORECASE)
            node_var = node_var_match.group(1) if node_var_match else "n"

            label = labels[i]
            label_props = cls._LABEL_PROPERTIES.get(label, []) if label else []

            new_cols = []
            for alias in unified:
                if alias in aliases:
                    # Keep original expression
                    col_match = re.search(
                        rf'([^,]*?\bAS\s+{re.escape(alias)}\b)',
                        original_ret, re.IGNORECASE
                    )
                    if col_match:
                        new_cols.append(col_match.group(1).strip())
                    else:
                        new_cols.append(f"NULL AS {alias}")
                else:
                    # Column missing — try to use actual property from the node
                    if alias in label_props:
                        new_cols.append(f"{node_var}.{alias} AS {alias}")
                    else:
                        new_cols.append(f"NULL AS {alias}")

            new_ret = ", ".join(new_cols)
            fixed_part = part[:ret_start] + new_ret + part[ret_end:]
            fixed_parts.append(fixed_part)

        return " UNION ".join(fixed_parts)

    def validate(self, cypher: str) -> tuple[bool, Optional[str]]:
        """Validate Cypher query for safety.

        Returns:
            (is_valid, error_message)
        """
        cypher_upper = cypher.upper().strip()

        # Check for forbidden clauses
        for clause in FORBIDDEN_CLAUSES:
            # Use word boundary to avoid false positives
            if re.search(rf'\b{re.escape(clause)}\b', cypher_upper):
                return False, f"Forbidden clause: {clause}"

        # Must have MATCH and RETURN
        if "MATCH" not in cypher_upper:
            return False, "Missing MATCH clause"
        if "RETURN" not in cypher_upper:
            return False, "Missing RETURN clause"

        # Note: LIMIT auto-add is handled in execute(), not here
        return True, None

    def execute(self, cypher: str, params: Optional[Dict] = None) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute Cypher query safely.

        Returns:
            (rows, error_message)
        """
        # Auto-fix common LLM syntax errors before validation
        cypher = self._fix_exists_syntax(cypher)
        cypher = self._fix_union_columns(cypher)

        # Validate
        is_valid, error = self.validate(cypher)
        if not is_valid:
            return [], error

        # Add LIMIT if not present
        if "LIMIT" not in cypher.upper():
            cypher = cypher.rstrip(";") + f" LIMIT {DEFAULT_MAX_RESULTS}"

        try:
            with self.driver.session() as session:
                result = session.run(cypher, **(params or {}))
                rows = [dict(record) for record in result]
                if not rows:
                    logger.info("Cypher returned 0 rows. Query: %s", cypher[:200])
                return rows, None
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.error("Cypher execution failed: %s", e)
            return [], str(e)


# ---------------------------------------------------------------------------
# TextToCypherRetriever
# ---------------------------------------------------------------------------

class TextToCypherRetriever:
    """Combines generator + executor to retrieve data via text-to-cypher.

    Usage:
        retriever = TextToCypherRetriever(driver, llm_service)
        results = retriever.retrieve("Lễ hội Tinh hoa Đại ngàn", intent="EVENT_RECOMMENDATION")
    """

    def __init__(self, driver, llm_service, timeout_ms: int = DEFAULT_TIMEOUT_MS):
        self.generator = TextToCypherGenerator(llm_service)
        self.executor = CypherExecutor(driver, timeout_ms)

    def retrieve(
        self,
        query: str,
        intent: str = "",
        max_results: int = DEFAULT_MAX_RESULTS,
        allowed_labels: Optional[List[str]] = None,
        location: str = "",
        is_follow_up: bool = False,
        exclude_entities: Optional[List[str]] = None,
        location_aliases: Optional[List[str]] = None,
    ) -> List[NodeItem]:
        """Retrieve data via text-to-cypher.

        Args:
            query: Natural language query
            intent: Intent type (for context)
            max_results: Maximum results
            allowed_labels: Filter by node labels
            location: Location context for filtering
            is_follow_up: Whether this is a follow-up query
            exclude_entities: Entity names to exclude (for follow-up "còn ... khác")
            location_aliases: Additional province names to search (for merged provinces)

        Returns:
            List of NodeItem results
        """
        # Build context hint for the LLM
        context_parts = []
        if location:
            # Build location search terms: primary + merged province aliases
            all_locations = [location] + [a for a in (location_aliases or []) if a != location]
            if len(all_locations) > 1:
                context_parts.append(f"Khu vực: {', '.join(all_locations)} (tỉnh mới sáp nhập)")
            else:
                context_parts.append(f"Khu vực: {location}")
            # Build WHERE clauses for all locations
            loc_clauses = " OR ".join(
                f"l.name CONTAINS '{loc}'" for loc in all_locations
            )
            addr_clauses = " OR ".join(
                f"n.address CONTAINS '{loc}'" for loc in all_locations
            )
            prov_clauses = " OR ".join(
                f"n.province CONTAINS '{loc}'" for loc in all_locations
            )
            # Build current_province clauses for ward→province fallback
            cur_prov_clauses = " OR ".join(
                f"l.current_province CONTAINS '{loc}'" for loc in all_locations
            )
            context_parts.append(
                f"IMPORTANT: "
                f"Để tìm nodes, dùng MỘT TRONG các cách: "
                f"(1) MATCH (n)-[:LOCATED_IN]->(l:Location) WHERE {loc_clauses} "
                f"(2) MATCH (n) WHERE {prov_clauses} "
                f"(3) MATCH (n) WHERE {addr_clauses} "
                f"(4) MATCH (n)-[:LOCATED_IN]->(l:Location) WHERE {cur_prov_clauses} "
                f"Ưu tiên (1) cho TouristAttraction/Restaurant/Accommodation. "
                f"Ưu tiên (2) cho Event/Dish/Specialty. "
                f"Dùng (4) khi (1) trả 0 rows (location là ward, không phải province). "
                f"KHÔNG filter theo keyword hẹp — chỉ dùng khi user yêu cầu cụ thể."
            )
        if intent:
            context_parts.append(f"Intent: {intent}")
        if allowed_labels:
            context_parts.append(f"Loại node cần tìm: {', '.join(allowed_labels)}")
        if is_follow_up:
            context_parts.append("Đây là câu hỏi follow-up, cần tìm entity KHÁC với kết quả trước")
        if exclude_entities:
            context_parts.append(f"Loại trừ các entity đã trả lời: {', '.join(exclude_entities[:5])}")
        context_hint = "## Context:\n" + "\n".join(context_parts) if context_parts else ""

        # Generate Cypher
        result = self.generator.generate(query, max_results, context_hint=context_hint, intent=intent)
        if not result.success:
            logger.warning("Text-to-Cypher generation failed: %s", result.error)
            return []

        logger.info("Generated Cypher: %s", result.cypher)
        if result.explanation:
            logger.info("Explanation: %s", result.explanation)

        # Execute
        rows, error = self.executor.execute(result.cypher)
        if error:
            logger.warning("Text-to-Cypher execution failed: %s", error)

        # Fallback: if LLM query returned 0 rows and we have a location, try simpler query
        if not rows and location:
            logger.info("Text-to-Cypher returned 0 rows, trying fallback query for location '%s'", location)
            # Try dish-aware fallback first if food-related
            if allowed_labels and ("Restaurant" in allowed_labels or "Dish" in allowed_labels):
                rows = self._fallback_dish_query(location, max_results, location_aliases)
            if not rows:
                rows = self._fallback_location_query(location, allowed_labels, max_results, location_aliases)

        if not rows:
            logger.info("Text-to-Cypher returned 0 rows")
            return []

        # Convert to NodeItem
        items = self._rows_to_node_items(rows, allowed_labels)

        # Resolve missing real IDs and coordinates from Neo4j for the retrieved items
        names_to_resolve = [item.content for item in items]
        if names_to_resolve and self.executor.driver:
            try:
                with self.executor.driver.session() as session:
                    cypher = """
                    MATCH (n)
                    WHERE n.name IN $names
                    RETURN n.name AS name,
                           n.id AS id,
                           CASE 
                             WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' 
                             THEN n.location.latitude 
                             ELSE n.lat
                           END AS lat,
                           CASE 
                             WHEN n.location IS NOT NULL AND toLower(toString(n.location)) STARTS WITH 'point' 
                             THEN n.location.longitude 
                             ELSE n.lng
                           END AS lng
                    """
                    db_rows = session.run(cypher, names=names_to_resolve).data()
                    db_map = {r["name"]: r for r in db_rows if r.get("name")}
                    for item in items:
                        if item.content in db_map:
                            db_info = db_map[item.content]
                            real_id = db_info.get("id")
                            if real_id:
                                item.id = str(real_id)
                                item.metadata["id"] = str(real_id)
                                if "attributes" in item.metadata and isinstance(item.metadata["attributes"], dict):
                                    item.metadata["attributes"]["id"] = str(real_id)
                            lat = db_info.get("lat")
                            lng = db_info.get("lng")
                            if lat is not None and lng is not None:
                                item.metadata["lat"] = lat
                                item.metadata["lng"] = lng
                                # Also update lat and lng in metadata attributes if attributes dict exists
                                if "attributes" in item.metadata and isinstance(item.metadata["attributes"], dict):
                                    item.metadata["attributes"]["lat"] = lat
                                    item.metadata["attributes"]["lng"] = lng
            except (Neo4jClientError, ServiceUnavailable) as e:
                logger.warning("Failed to resolve missing IDs and coordinates for cypher items: %s", e)

        return items

    def _fallback_location_query(
        self,
        location: str,
        allowed_labels: Optional[List[str]] = None,
        max_results: int = DEFAULT_MAX_RESULTS,
        location_aliases: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback: simple query without keyword filters when LLM query returns 0 rows."""
        # Build label filter
        label_filter = ""
        if allowed_labels:
            label_conditions = [f"'{lbl}' IN labels(n)" for lbl in allowed_labels if lbl != "Location"]
            if label_conditions:
                label_filter = f"AND ({' OR '.join(label_conditions)})"

        all_locations = [location] + [a for a in (location_aliases or []) if a != location]
        params = {f"loc_{i}": loc for i, loc in enumerate(all_locations)}
        params["limit"] = max_results

        # Try 1: LOCATED_IN → Location.name (works for ward-level: Pleiku, Quy Nhơn)
        loc_in_conditions = " OR ".join(
            f"l.name CONTAINS $loc_{i}" for i in range(len(all_locations))
        )
        fallback_cypher = f"""
        MATCH (n)-[:LOCATED_IN]->(l:Location)
        WHERE ({loc_in_conditions})
              {label_filter}
        RETURN n.id AS id, n.name AS name, n.description AS description,
               n.address AS address, labels(n) AS labels
        LIMIT $limit
        """
        try:
            rows, error = self.executor.execute(fallback_cypher, params)
            if rows:
                logger.info("Fallback (LOCATED_IN) found %d results for '%s'", len(rows), location)
                return rows
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.warning("Fallback LOCATED_IN query failed: %s", e)

        # Try 2: LOCATED_IN → Location.current_province (ward→province fallback)
        cur_prov_conditions = " OR ".join(
            f"l.current_province CONTAINS $loc_{i}" for i in range(len(all_locations))
        )
        fallback_cypher2 = f"""
        MATCH (n)-[:LOCATED_IN]->(l:Location)
        WHERE ({cur_prov_conditions})
              {label_filter}
        RETURN n.id AS id, n.name AS name, n.description AS description,
               n.address AS address, labels(n) AS labels
        LIMIT $limit
        """
        try:
            rows, error = self.executor.execute(fallback_cypher2, params)
            if rows:
                logger.info("Fallback (current_province) found %d results for '%s'", len(rows), location)
                return rows
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.warning("Fallback current_province query failed: %s", e)

        # Try 3: Direct property match (n.province, n.address, n.name)
        prop_conditions = " OR ".join(
            f"n.province CONTAINS $loc_{i} OR n.address CONTAINS $loc_{i} OR n.name CONTAINS $loc_{i}"
            for i in range(len(all_locations))
        )
        fallback_cypher3 = f"""
        MATCH (n)
        WHERE ({prop_conditions})
              {label_filter}
        RETURN n.id AS id, n.name AS name, n.description AS description,
               n.address AS address, labels(n) AS labels
        LIMIT $limit
        """
        try:
            rows, error = self.executor.execute(fallback_cypher3, params)
            if rows:
                logger.info("Fallback (properties) found %d results for '%s' (+ aliases: %s)", len(rows), location, location_aliases)
            return rows or []
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.warning("Fallback properties query failed: %s", e)
            return []

    def _fallback_dish_query(
        self,
        location: str,
        max_results: int = DEFAULT_MAX_RESULTS,
        location_aliases: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fallback for food queries: find Restaurants that serve Dishes/Specialties in the location."""
        all_locations = [location] + [a for a in (location_aliases or []) if a != location]
        loc_conditions = " OR ".join(
            f"l.name CONTAINS $loc_{i}" for i in range(len(all_locations))
        )
        params = {f"loc_{i}": loc for i, loc in enumerate(all_locations)}
        params["limit"] = max_results

        # Query: Restaurant-[:LOCATED_IN]->Location + Restaurant-[:HAS]->Dish/Specialty
        fallback_cypher = f"""
        MATCH (r:Restaurant)-[:LOCATED_IN]->(l:Location)
        WHERE ({loc_conditions})
        MATCH (r)-[:HAS]->(d)
        RETURN r.id AS id, r.name AS name, r.address AS address,
               r.phone AS phone, r.tags AS tags, r.type AS type,
               r.enriched_rating AS rating,
               collect(d.name) AS dishes,
               labels(r) AS labels
        LIMIT $limit
        """
        try:
            rows, error = self.executor.execute(fallback_cypher, params)
            if rows:
                logger.info("Dish fallback found %d results for '%s'", len(rows), location)
            return rows or []
        except (Neo4jClientError, ServiceUnavailable) as e:
            logger.warning("Dish fallback query failed: %s", e)
            return []

    def _rows_to_node_items(
        self,
        rows: List[Dict[str, Any]],
        allowed_labels: Optional[List[str]] = None,
    ) -> List[NodeItem]:
        """Convert Cypher result rows to NodeItem list."""
        items = []

        for row in rows:
            # Handle Neo4j Node objects
            node_data = self._extract_node_from_row(row)
            if node_data:
                node_id = node_data.get("id", "")
                name = node_data.get("name", "")
                description = node_data.get("description", "")
                label = node_data.get("label", "")
            else:
                # Try to extract from flat dict
                node_id = self._extract_field(row, ["id", "nodeId", "elementId"])
                name = self._extract_field(row, ["name", "e.name", "r.name", "t.name", "n.name", "s.name"])
                description = self._extract_field(row, ["description", "e.description", "r.description", "s.description"])
                label = self._extract_field(row, ["label", "labels", "e.labels", "type"])

            if not name:
                # Try any key ending with .name (e.g., s.name, a.name)
                for key, value in row.items():
                    if key.endswith(".name") and isinstance(value, str) and value:
                        name = value
                        break
            if not name:
                # Try to find any string value as name
                for key, value in row.items():
                    if isinstance(value, str) and len(value) > 2:
                        name = value
                        break

            if not name:
                continue

            # Determine label
            if not label:
                label = self._infer_label(row, allowed_labels)

            # Filter by allowed labels
            if allowed_labels and label and label not in allowed_labels:
                continue

            # Build metadata
            metadata = {
                "name": name,
                "label": label or "Unknown",
                "source": "text_to_cypher",
            }
            # Add all row fields to metadata
            for key, value in row.items():
                if value is not None and key not in metadata:
                    metadata[key] = value

            items.append(NodeItem(
                id=str(node_id or name),
                content=name,
                metadata=metadata,
                score=1.0,
                source_type="text_to_cypher",
            ))

        return items

    def _extract_node_from_row(self, row: Dict) -> Optional[Dict[str, Any]]:
        """Extract node data from row that may contain Neo4j Node objects."""
        for key, value in row.items():
            # Check if value is a Neo4j Node (has .get() method and labels)
            if hasattr(value, "get") and hasattr(value, "labels"):
                try:
                    return {
                        "id": str(value.get("id", "") or value.get("elementId", "") or ""),
                        "name": str(value.get("name", "") or ""),
                        "description": str(value.get("description", "") or ""),
                        "label": list(value.labels)[0] if value.labels else "",
                        "properties": dict(value),
                    }
                except (Neo4jClientError, ServiceUnavailable, AttributeError, TypeError):
                    pass
        return None

    def _extract_field(self, row: Dict, keys: List[str]) -> Optional[str]:
        """Extract a field value from row, trying multiple key names."""
        for key in keys:
            if key in row and row[key] is not None:
                val = row[key]
                # Handle lists (e.g., labels() returns a list)
                if isinstance(val, list):
                    return str(val[0]).strip() if val else None
                return str(val).strip()
        return None

    def _infer_label(self, row: Dict, allowed_labels: Optional[List[str]]) -> str:
        """Infer node label from row data."""
        # Check for label field
        for key in ["label", "labels", "type"]:
            if key in row and row[key]:
                val = row[key]
                # Handle actual list (from Neo4j labels(n))
                if isinstance(val, list):
                    return str(val[0]) if val else ""
                # Handle string format
                val = str(val).strip("[]'\"")
                if val:
                    return val

        # No label info in row — return empty to avoid wrong label assignment
        return ""
