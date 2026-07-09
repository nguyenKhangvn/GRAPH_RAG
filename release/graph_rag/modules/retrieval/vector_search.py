from typing import List, Dict, Optional
import logging
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
from graph_rag.config import VECTOR_INDEXES
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)


# Module-level lazy singleton for AdminRegionMappingService
_admin_region_mapping_svc = None
def _get_admin_mapping():
    global _admin_region_mapping_svc
    if _admin_region_mapping_svc is None:
        from graph_rag.modules.pipeline_support.admin_region_mapping_service import AdminRegionMappingService
        _admin_region_mapping_svc = AdminRegionMappingService()
    return _admin_region_mapping_svc


def _resolve_legacy_province(city_name: str) -> Optional[str]:
    """Resolve a city/province name to legacy_province using AdminRegionMappingService.

    Replaces the hardcoded _PROVINCE_TO_LEGACY dict.
    """
    if not city_name:
        return None
    svc = _get_admin_mapping()
    resolved = svc.resolve(city_name)
    if resolved and resolved.get("legacy_province"):
        return resolved["legacy_province"]
    return None


def _region_group_for_cypher(region_group):
    """Extract Cypher-safe region_group value (str or None).

    When region_group is a list (merged regions), pass None to Cypher
    and rely on application-level _matches_region for filtering.
    """
    if isinstance(region_group, list):
        return None
    # Map registry region groups to DB region groups
    if region_group == "tay_nguyen":
        return "gia_lai_core"
    if region_group == "duyen_hai_nam_trung_bo":
        return "binh_dinh_legacy"
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
                expected_groups.add(region_group)
            else:
                expected_groups = {region_group}
        # Add database-specific region_group mapped values for hybrid matching
        if "tay_nguyen" in expected_groups:
            expected_groups.add("gia_lai_core")
        if "duyen_hai_nam_trung_bo" in expected_groups:
            expected_groups.add("binh_dinh_legacy")
        record_rg = str(record.get("region_group") or "").strip()
        record_rf = str(record.get("entity_region_focus") or "").strip()
        if record_rg in expected_groups:
            return True
        if record_rf in expected_groups:
            return True
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


def search_vector_loop(driver, query_vector: List[float], k: int,
                       filter_labels: List[str] = None,
                       filter_city: str = None,
                       region_group=None,
                       legacy_province: str = None) -> List[Dict]:
    """
    Duyệt qua danh sách Vector Index và tìm kiếm có FILTER.
    """
    all_results = []

    # Normalize compound location như 'Pleiku, Gia Lai' → lấy phần đầu tiên
    if filter_city and ',' in filter_city:
        filter_city = filter_city.split(',')[0].strip()
    # Nếu filter_city là tên tỉnh → set legacy_province để filter Restaurant/TouristAttraction
    # Giữ filter_city để Cypher match node.location TEXT trên Dish nodes
    if filter_city and not (region_group or legacy_province):
        resolved_lp = _resolve_legacy_province(filter_city.strip())
        if resolved_lp:
            legacy_province = resolved_lp

    orig_legacy_province = legacy_province

    # Khi region_group đã có, bỏ legacy_province vì:
    # 1. DB nodes không có legacy_province property (NULL) → filter loại hết
    # 2. region_group đã đủ xác định vùng, không cần double-filter
    if region_group and legacy_province:
        logger.debug("[VECTOR] region_group=%s provided, dropping legacy_province=%s (nodes have NULL legacy_province)", region_group, legacy_province)
        legacy_province = None

    # [CHIẾN THUẬT] Nếu có filter, lấy nhiều candidate hơn rồi lọc lại
    # Tăng candidate_k đáng kể khi có region filter để không bỏ sót nodes đúng region
    if region_group or legacy_province:
        candidate_k = k * 20  # Lấy nhiều hơn vì region filter sẽ loại bỏ大部分
    elif filter_labels or filter_city:
        candidate_k = k * 5
    else:
        candidate_k = k

    # Cypher query — dùng Location node thay vì Commune/Province (đã bị xóa khỏi schema)
    # Hỗ trợ hybrid location: check cả entity node fields lẫn Location node
    # Dish/Specialty: dùng location/province properties hoặc SPECIALTY_OF -> Location
    cypher = """
    CALL db.index.vector.queryNodes($index_name, $candidate_k, $embedding)
    YIELD node, score

    // Match Location via LOCATED_IN (Restaurant, TouristAttraction, Accommodation)
    OPTIONAL MATCH (node)-[:LOCATED_IN]->(loc:Location)

    WITH node, score, loc,
         coalesce(node.region_focus, '') as entity_region_focus,
         coalesce(node.legacy_province, '') as entity_legacy_province,
         CASE WHEN node.location IS NOT NULL THEN toString(node.location) ELSE '' END as entity_location,
         coalesce(loc.region_focus, '') as loc_region_focus,
         coalesce(loc.legacy_province, loc.current_province, '') as loc_legacy_province,
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
        // Khi có region_group filter, chỉ lấy nodes có relationship tới Location đúng region
        ($region_group IS NULL OR
         coalesce(loc.region_group, '') = $region_group OR
         entity_region_focus = $region_group
        )

        AND

        // 4. LỌC THEO LEGACY PROVINCE (hybrid: check entity fields OR Location node fields)
        // Thêm check node.location TEXT cho Dish nodes (e.g. "Tỉnh Bình Định")
        ($legacy_province IS NULL OR
         loc_legacy_province = $legacy_province OR
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
        coalesce(loc_legacy_province, '') as legacy_province,
        coalesce(loc.admin_status, '') as admin_status,
        coalesce(entity_legacy_province, '') as entity_legacy_province,
        coalesce(entity_location, '') as entity_location,
        case when node.location is not null and toString(node.location) <> node.location then node.location.latitude else toFloat(node.lat) end as lat,
        case when node.location is not null and toString(node.location) <> node.location then node.location.longitude else toFloat(node.lng) end as lng,
        score,
        'vector' as found_by
    LIMIT $candidate_k
    """
    
    with driver.session() as session:
        for index_name in VECTOR_INDEXES:
            try:
                result = session.run(
                    cypher,
                    index_name=index_name,
                    candidate_k=candidate_k,
                    k=k,
                    embedding=query_vector,
                    filter_labels=filter_labels,
                    filter_city=filter_city,
                    region_group=_region_group_for_cypher(region_group),
                    legacy_province=legacy_province
                )
                data = [record.data() for record in result]
                _cypher_count = len(data)
                if filter_city:
                    data = [row for row in data if _matches_location(row, filter_city)]
                if region_group or orig_legacy_province:
                    data = [row for row in data if _matches_region(row, region_group, orig_legacy_province)]
                _after_filter = len(data)
                if _cypher_count > 0 and _after_filter == 0:
                    logger.info("[DEBUG-VECTOR] index=%s cypher=%d -> after_filter=%d (ALL FILTERED OUT! region_group=%s legacy_province=%s)",
                                index_name, _cypher_count, _after_filter, region_group, legacy_province)
                elif _cypher_count > 0:
                    logger.debug("[DEBUG-VECTOR] index=%s cypher=%d -> after_filter=%d", index_name, _cypher_count, _after_filter)
                all_results.extend(data)
            except (Neo4jClientError, ServiceUnavailable) as e:
                logger.warning("Vector index '%s' failed: %s", index_name, e, exc_info=True)
                
    return all_results
