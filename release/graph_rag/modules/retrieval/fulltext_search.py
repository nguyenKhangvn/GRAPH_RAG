import logging

logger = logging.getLogger(__name__)

from typing import List, Dict, Optional
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
import re
from graph_rag.config import FULLTEXT_INDEXES
from graph_rag.utils.text import normalize_text


# Module-level lazy singleton for AdminRegionMappingService
_admin_region_mapping_svc = None
def _get_admin_mapping():
    global _admin_region_mapping_svc
    if _admin_region_mapping_svc is None:
        from graph_rag.modules.pipeline_support.admin_region_mapping_service import AdminRegionMappingService
        _admin_region_mapping_svc = AdminRegionMappingService()
    return _admin_region_mapping_svc


def _resolve_legacy_province(city_name: str) -> Optional[str]:
    """Resolve a city/province name to legacy_province using AdminRegionMappingService."""
    if not city_name:
        return None
    svc = _get_admin_mapping()
    resolved = svc.resolve(city_name)
    if resolved and resolved.get("legacy_province"):
        return resolved["legacy_province"]
    return None

# Vietnamese stop words — common function words that add noise to Lucene queries
_VN_STOP_WORDS = {
    "và", "của", "có", "là", "được", "cho", "với", "từ", "đến", "trong",
    "trên", "dưới", "về", "theo", "những", "các", "này", "đó", "khi",
    "nếu", "nhưng", "hoặc", "hay", "cũng", "không", "đã", "sẽ", "đang",
    "thì", "mà", "như", "lại", "nữa", "rằng", "tại", "do", "bởi",
    "sau", "trước", "giữa", "ngoài", "qua", "lên", "xuống", "ra", "vào",
    "hãy", "xin", "vui lòng", "giúp", "tôi", "bạn", "mình", "chúng",
    "nào", "gì", "ai", "đâu", "bao giờ", "thế nào", "tại sao",
    "so sánh", "phân tích", "dựa trên", "lần lượt", "bạn có thể",
    "có thể", "cần", "nên", "muốn", "thích", "gần", "xa",
    "đây", "đó", "kia", "nọ",
}

# Maximum number of tokens in a Lucene query to avoid TooManyNestedClauses
_MAX_LUCENE_TOKENS = 8


def sanitize_lucene_query(text: str) -> str:
    """Làm sạch và tối ưu chuỗi tìm kiếm cho Lucene.

    Phase 10 improvements:
    - Vietnamese compound word splitting (e.g., "khách sạn" → "khách" AND "sạn")
    - Diacritics-stripped variant generation for fallback matching
    - OR-based matching for better recall (fuzzy ~ handles precision)
    - Filter Vietnamese stop words to reduce noise
    - Cap token count at _MAX_LUCENE_TOKENS to prevent TooManyNestedClauses
    - Prioritize longer, more meaningful tokens
    """
    if not text:
        return ""

    clean_text = str(text)
    # Xử lý các toán tử nhiều ký tự trước
    clean_text = clean_text.replace("&&", " ").replace("||", " ")
    # Xóa tất cả ký tự đặc biệt của Lucene (bao gồm cả / và \)
    clean_text = re.sub(r'[+\-&|!(){}\[\]^"~*?:\\/]', ' ', clean_text)
    # Chuẩn hóa khoảng trắng
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    tokens = clean_text.split()
    # Chỉ fuzzy từ có độ dài > 2 để tránh nhiễu
    safe_tokens = [t for t in tokens if len(t) > 2]
    # Filter stop words
    filtered = [t for t in safe_tokens if t.lower() not in _VN_STOP_WORDS]
    # If filtering removed everything, keep original safe_tokens
    if not filtered:
        filtered = safe_tokens
    # Cap token count to avoid TooManyNestedClauses (>1024 in Lucene)
    # Prioritize longer tokens (more meaningful)
    if len(filtered) > _MAX_LUCENE_TOKENS:
        filtered = sorted(filtered, key=len, reverse=True)[:_MAX_LUCENE_TOKENS]

    # Phase 10: Use OR instead of AND for better recall.
    # Fuzzy matching (~) already handles precision; OR ensures we don't miss
    # results that match some but not all tokens.
    # For queries with <= 2 tokens, keep AND for precision.
    if len(filtered) <= 2:
        return " AND ".join([f"{t}~" for t in filtered])
    else:
        return " OR ".join([f"{t}~" for t in filtered])


def _region_group_for_cypher(region_group):
    """Extract Cypher-safe region_group value (str or None).

    When region_group is a list (merged regions), pass None to Cypher
    and rely on application-level _matches_region for filtering.
    """
    if isinstance(region_group, (list, set)):
        logger.debug("[DEBUG-CYPHER-PARAM] region_group is %s (%s) -> None for Cypher", type(region_group).__name__, region_group)
        return None
    if region_group and not isinstance(region_group, str):
        logger.warning("[DEBUG-CYPHER-PARAM] unexpected region_group type: %s = %s", type(region_group).__name__, region_group)
    return region_group


def _matches_location(record: Dict, filter_city: Optional[str]) -> bool:
    if not filter_city:
        return True
    needle = normalize_text(filter_city, strip_punct=True)
    haystack = " ".join(
        [
            normalize_text(record.get("address"), strip_punct=True),
            normalize_text(record.get("commune_name"), strip_punct=True),
            normalize_text(record.get("name"), strip_punct=True),
            normalize_text(record.get("entity_location"), strip_punct=True),
        ]
    )
    return bool(needle) and needle in haystack


def _matches_region(record: Dict, region_group, legacy_province: Optional[str]) -> bool:
    # Check both Location node fields and entity node fields (hybrid location)
    # Exclude merged Location nodes unless explicitly querying that legacy province
    admin_status = str(record.get("admin_status") or "").strip().lower()
    if admin_status == "merged" and not legacy_province:
        return False

    if region_group:
        # Handle merged regions (list of region_groups)
        if isinstance(region_group, list):
            expected_groups = set(region_group)
        else:
            # Dynamic: resolve region_focus to region_groups via RegionRegistry
            from graph_rag.config.region_registry import region_registry
            provinces = region_registry.get_provinces_in_group(region_group)
            if provinces:
                expected_groups = {region_registry.get_region_group(pid) for pid in provinces}
                expected_groups.add(region_group)  # include the group itself
            else:
                expected_groups = {region_group}
        record_rg = str(record.get("region_group") or "").strip()
        record_rf = str(record.get("entity_region_focus") or "").strip()
        if record_rg in expected_groups:
            return True
        if record_rf in expected_groups:
            return True
        # When region_group filter is active, nodes without any region data should be excluded
        logger.debug("[DEBUG-MATCHES-REGION] REJECTED: name=%s record_rg='%s' record_rf='%s' expected=%s",
                     record.get("name", "?"), record_rg, record_rf, expected_groups)
        return False
    if legacy_province:
        lp_norm = normalize_text(legacy_province, strip_punct=True)
        if normalize_text(record.get("legacy_province"), strip_punct=True) == lp_norm:
            return True
        if normalize_text(record.get("entity_legacy_province"), strip_punct=True) == lp_norm:
            return True
        # Check Dish.location TEXT property (e.g. "Tỉnh Bình Định")
        if lp_norm in normalize_text(record.get("entity_location"), strip_punct=True):
            return True
        return False
    # No region filter: allow all
    return True


def search_fulltext_loop(driver, search_text: str, k: int,
                         filter_labels: List[str] = None,
                         filter_city: str = None,
                         region_group=None,
                         legacy_province: str = None) -> List[Dict]:
    """
    Duyệt qua danh sách Fulltext Index với FILTER.
    """
    all_results = []

    # Normalize compound location như 'Pleiku, Gia Lai' → lấy phần đầu tiên
    if filter_city and ',' in filter_city:
        filter_city = filter_city.split(',')[0].strip()
    # Nếu filter_city là tên tỉnh → set legacy_province để filter Restaurant/TouristAttraction
    # Giữ filter_city để Cypher match node.location TEXT trên Dish nodes
    if filter_city and not (region_group or legacy_province):
        legacy_province = _resolve_legacy_province(filter_city.strip())

    # Khi region_group đã có, bỏ legacy_province (nodes có NULL legacy_province)
    if region_group and legacy_province:
        legacy_province = None

    # Always sanitize to avoid Lucene lexical errors from user text.
    lucene_query = sanitize_lucene_query(search_text)

    if not lucene_query:
        return []

    # Phase 10: Also prepare diacritics-stripped query for fallback search.
    # Vietnamese users often type without diacritics (e.g., "Bien Ho" instead
    # of "Biển Hồ"), and the fulltext index may not match both variants.
    search_text_stripped = normalize_text(search_text, strip_punct=True)
    lucene_query_stripped = sanitize_lucene_query(search_text_stripped) if search_text_stripped != search_text else ""

    # Tăng candidate_k đáng kể khi có region filter để không bỏ sót nodes đúng region
    # Phase 10: Increased multipliers for better recall (more candidates before filtering)
    if region_group or legacy_province:
        candidate_k = k * 20
    elif filter_labels or filter_city:
        candidate_k = k * 5
    else:
        candidate_k = k * 2

    # Cypher query — hỗ trợ hybrid location: check cả entity node fields lẫn Location node
    # Dish/Specialty: dùng location/province properties hoặc SPECIALTY_OF -> Location
    cypher = """
    CALL db.index.fulltext.queryNodes($index_name, $lucene_query, {limit: $candidate_k})
    YIELD node, score

    // Match Location via LOCATED_IN (Restaurant, TouristAttraction, Accommodation)
    OPTIONAL MATCH (node)-[:LOCATED_IN]->(loc:Location)

    WITH node, score, loc,
         coalesce(node.region_focus, '') as entity_region_focus,
         coalesce(node.legacy_province, '') as entity_legacy_province,
         CASE WHEN node.location IS NOT NULL THEN toString(node.location) ELSE '' END as entity_location,
         coalesce(loc.region_focus, '') as loc_region_focus,
         coalesce(loc.legacy_province, '') as loc_legacy_province,
         coalesce(loc.admin_status, '') as admin_status
    WHERE
        // 1. LỌC THEO LABEL
        ($filter_labels IS NULL OR size([lbl IN labels(node) WHERE lbl IN $filter_labels]) > 0)

        AND

        // 2. LỌC THEO LOCATION
        // Check: address, Location.name, node.location (POINT toString hoặc TEXT)
        ($filter_city IS NULL OR
         toLower(node.address) CONTAINS toLower($filter_city) OR
          toLower(coalesce(loc.name, '')) CONTAINS toLower($filter_city) OR
          CASE WHEN node.location IS NOT NULL THEN toLower(toString(node.location)) ELSE '' END CONTAINS toLower($filter_city)
        )

        AND

        // 3. LỌC THEO REGION (hybrid: check entity fields OR Location node fields)
        ($region_group IS NULL OR
         coalesce(loc.region_group, '') = $region_group OR
         entity_region_focus = $region_group
        )

        AND

        // 4. LỌC THEO LEGACY PROVINCE (hybrid: check entity fields OR Location node fields)
        // Thêm check node.location TEXT cho Dish nodes (e.g. "Tỉnh Bình Định")
        ($legacy_province IS NULL OR
         (loc IS NULL AND entity_region_focus = '' AND entity_legacy_province = '' AND entity_location = '') OR
         coalesce(loc.legacy_province, '') = $legacy_province OR
         toLower(entity_legacy_province) = toLower($legacy_province) OR
         toLower(entity_location) CONTAINS toLower($legacy_province)
        )

    RETURN
        node.id as id,
        labels(node)[0] as type,
        node.name as name,
        coalesce(node.address, '') as address,
        coalesce(node.description, '') as description,
        coalesce(node.topic, '') as topic,
        coalesce(node.category, '') as category,
        coalesce(node.star_rating, 0) as star_rating,
        coalesce(node.price_range, '') as price_range,
        coalesce(loc.name, '') as commune_name,
        coalesce(loc.region_group, '') as region_group,
        coalesce(loc.legacy_province, '') as legacy_province,
        coalesce(loc.admin_status, '') as admin_status,
        coalesce(entity_legacy_province, '') as entity_legacy_province,
        coalesce(entity_location, '') as entity_location,
        case when node.location is not null and toString(node.location) <> node.location then node.location.latitude else toFloat(node.lat) end as lat,
        case when node.location is not null and toString(node.location) <> node.location then node.location.longitude else toFloat(node.lng) end as lng,
        score,
        'fulltext' as found_by
    LIMIT $candidate_k
    """
    
    with driver.session() as session:
        for index_name in FULLTEXT_INDEXES:
            try:
                result = session.run(
                    cypher,
                    index_name=index_name,
                    lucene_query=lucene_query,
                    candidate_k=candidate_k,
                    k=k,
                    filter_labels=filter_labels,
                    filter_city=filter_city,
                    region_group=_region_group_for_cypher(region_group),
                    legacy_province=legacy_province
                )
                data = [record.data() for record in result]
                _cypher_count = len(data)
                if filter_city:
                    data = [row for row in data if _matches_location(row, filter_city)]
                if region_group or legacy_province:
                    data = [row for row in data if _matches_region(row, region_group, legacy_province)]
                _after_filter = len(data)
                if _cypher_count > 0 and _after_filter == 0:
                    logger.info("[DEBUG-FULLTEXT] index=%s cypher=%d -> after_filter=%d (ALL FILTERED OUT! region_group=%s legacy_province=%s)",
                                index_name, _cypher_count, _after_filter, region_group, legacy_province)
                elif _cypher_count > 0:
                    logger.debug("[DEBUG-FULLTEXT] index=%s cypher=%d -> after_filter=%d", index_name, _cypher_count, _after_filter)
                all_results.extend(data)
            except (Neo4jClientError, ServiceUnavailable) as e:
                logger.warning("Fulltext warning: %s", e)
                # pass

    # Phase 10: Fallback search with diacritics-stripped query.
    # If main search returned few results and we have a different stripped query,
    # retry with the stripped version to catch nodes that match the diacritics-free form.
    if len(all_results) < k and lucene_query_stripped and lucene_query_stripped != lucene_query:
        logger.info("[DEBUG-FULLTEXT] Few results (%d), retrying with stripped query: '%s'", len(all_results), lucene_query_stripped)
        with driver.session() as session:
            for index_name in FULLTEXT_INDEXES:
                try:
                    result = session.run(
                        cypher,
                        index_name=index_name,
                        lucene_query=lucene_query_stripped,
                        candidate_k=candidate_k,
                        k=k,
                        filter_labels=filter_labels,
                        filter_city=filter_city,
                        region_group=_region_group_for_cypher(region_group),
                        legacy_province=legacy_province
                    )
                    data = [record.data() for record in result]
                    if filter_city:
                        data = [row for row in data if _matches_location(row, filter_city)]
                    if region_group or legacy_province:
                        data = [row for row in data if _matches_region(row, region_group, legacy_province)]
                    # Deduplicate against existing results
                    existing_ids = {r.get("id") for r in all_results}
                    new_data = [r for r in data if r.get("id") not in existing_ids]
                    if new_data:
                        logger.info("[DEBUG-FULLTEXT] Stripped query found %d new results from index %s", len(new_data), index_name)
                    all_results.extend(new_data)
                except (Neo4jClientError, ServiceUnavailable) as e:
                    logger.warning("Fulltext stripped fallback warning: %s", e)

    return all_results
