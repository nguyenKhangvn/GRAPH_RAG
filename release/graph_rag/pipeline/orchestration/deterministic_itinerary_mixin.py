"""Itinerary and tourism analysis mixin — tour plans, lodging/heritage, spatial analysis."""

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
import logging
import re
from typing import Any, Dict, List, Optional

from graph_rag.core.intents import IntentType
from graph_rag.core import keywords
from graph_rag.utils.text import normalize_text

logger = logging.getLogger(__name__)

from .dto import PipelineRunState

# Hard label sets — only these count as lodging/heritage.
LODGING_LABELS: set = keywords.LODGING_LABELS
HERITAGE_LABELS: set = keywords.HERITAGE_LABELS


class DeterministicItineraryMixin:
    """Mixin for itinerary, tour, and tourism analysis deterministic answers."""

    def _clean_display_name(self, name: str) -> str:
        """Remove technical labels like (Accommodation) from display names."""
        cleaned = re.sub(
            r"\s*\((?:Accommodation|TouristAttraction|Restaurant|Hotel|Homestay|"
            r"GuestHouse|Resort|TravelAgency|HeritageSite|HistoricalSite|"
            r"ArchaeologicalSite|Event|Dish|Tour)\)\s*",
            "",
            str(name or ""),
        ).strip()
        return cleaned or str(name or "").strip()

    def _fetch_tour_itinerary_from_graph(
        self, tour_name_hint: str
    ) -> Dict[str, Any] | None:
        """Query tour itinerary from the graph database."""
        p = self.pipeline
        if not tour_name_hint:
            return None

        cypher = """
        MATCH (t:Tour)
        WHERE toLower(t.name) = toLower($name)
           OR toLower(t.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(t.name)
        OPTIONAL MATCH (t)-[r:INCLUDES]->(a:TouristAttraction)
        RETURN t.id AS tour_id,
               t.name AS tour_name,
               t.duration AS tour_duration,
               t.price AS tour_price,
               t.description AS tour_full_content,
               a.id AS poi_id,
               a.name AS poi_name,
               a.address AS poi_address,
               r.day AS rel_day,
               r.order AS rel_order,
               r.activity AS rel_activity,
               CASE WHEN a.location IS NOT NULL AND toLower(toString(a.location)) STARTS WITH 'point'
                    AND a.location.latitude IS NOT NULL THEN a.location.latitude
                    ELSE toFloat(a.lat) END AS poi_lat,
               CASE WHEN a.location IS NOT NULL AND toLower(toString(a.location)) STARTS WITH 'point'
                    AND a.location.longitude IS NOT NULL THEN a.location.longitude
                    ELSE toFloat(a.lng) END AS poi_lng,
               CASE WHEN toLower(t.name) = toLower($name) THEN 0 ELSE 1 END AS exact_rank
        ORDER BY exact_rank ASC, t.name ASC, rel_day ASC, rel_order ASC
        LIMIT 120
        """

        try:
            with p.driver.session() as session:
                rows = session.run(cypher, name=tour_name_hint).data()
        except (Neo4jClientError, ServiceUnavailable) as exc:
            p.logger.warning("strict_tour_itinerary_query_failed: %s", str(exc))
            return None

        if not rows:
            return None

        selected_tour_id = str(rows[0].get("tour_id") or "")
        selected = [r for r in rows if str(r.get("tour_id") or "") == selected_tour_id]
        if not selected:
            return None

        head = selected[0]
        points = []
        for row in selected:
            poi_name = str(row.get("poi_name") or "").strip()
            if not poi_name:
                continue
            points.append(
                {
                    "id": str(row.get("poi_id") or ""),
                    "name": poi_name,
                    "address": str(row.get("poi_address") or "").strip(),
                    "day": int(row.get("rel_day") or 1),
                    "order": int(row.get("rel_order") or 999),
                    "activity": str(row.get("rel_activity") or "").strip(),
                    "lat": float(row.get("poi_lat")) if row.get("poi_lat") is not None else None,
                    "lng": float(row.get("poi_lng")) if row.get("poi_lng") is not None else None,
                }
            )

        points.sort(key=lambda x: (x["day"], x["order"], x["name"]))
        return {
            "tour_id": selected_tour_id,
            "tour_name": str(head.get("tour_name") or "").strip(),
            "tour_duration": head.get("tour_duration"),
            "tour_price": head.get("tour_price"),
            "tour_full_content": str(head.get("tour_full_content") or "").strip(),
            "points": points,
        }

    def _answer_strict_tour_itinerary_if_possible(
        self, state: PipelineRunState
    ) -> Dict[str, Any] | None:
        if state.primary_intent != IntentType.TOUR_PLAN:
            return None
        if not self._is_strict_tour_itinerary_query(state.user_query):
            return None

        tour_hint = self._extract_tour_name_hint(
            state.search_query or state.user_query
        )
        if not tour_hint:
            return None

        tour_data = self._fetch_tour_itinerary_from_graph(tour_hint)
        if not tour_data:
            return None

        lines = [
            f"Lịch trình gốc theo dữ liệu hệ thống: {tour_data['tour_name']}"
        ]
        if tour_data.get("tour_duration"):
            lines.append(f"Thời lượng: {tour_data['tour_duration']}")
        if tour_data.get("tour_price") is not None:
            lines.append(f"Giá tham khảo: {tour_data['tour_price']}")

        points = tour_data.get("points") or []
        if points:
            current_day = None
            for point in points:
                day = int(point.get("day") or 1)
                if day != current_day:
                    current_day = day
                    lines.append(f"Ngày {current_day}:")
                order = int(point.get("order") or 999)
                activity = point.get("activity") or ""
                address = point.get("address") or ""
                segment = f"- ({order}) {point.get('name') or ''}"
                if activity:
                    segment += f" | Hoạt động: {activity}"
                if address:
                    segment += f" | Địa chỉ: {address}"
                lines.append(segment)
        elif tour_data.get("tour_full_content"):
            lines.append("Chi tiết nội dung tour:")
            lines.append(tour_data["tour_full_content"])
        else:
            lines.append(
                "Chưa có dữ liệu chi tiết lịch trình theo điểm đến trong quan hệ INCLUDES."
            )

        lines.append("Nguồn: dữ liệu lịch trình tour hiện có trong hệ thống.")

        intent = state.query_plan.intent if state.query_plan else state.primary_intent
        p = self.pipeline
        state.runtime.metadata["intent"] = intent
        state.runtime.metadata["strict_tour_mode"] = True
        state.runtime.metadata["strict_tour_name"] = tour_data.get("tour_name")
        state.runtime.metadata["strict_tour_points"] = points
        state.runtime.metadata["detected_location"] = state.location

        from types import SimpleNamespace

        tour_nodes = [
            SimpleNamespace(
                id=pt.get("id"),
                content=pt.get("name", ""),
                metadata={
                    "name": pt.get("name"),
                    "labels": ["TouristAttraction"],
                    "lat": pt.get("lat"),
                    "lng": pt.get("lng"),
                    "address": pt.get("address"),
                },
            )
            for pt in points
            if pt.get("lat") is not None and pt.get("lng") is not None
        ]

        map_seeds = tour_nodes if tour_nodes else state.all_seeds
        state.runtime.metadata["seed_nodes"] = self._build_seed_metadata(map_seeds)
        state.runtime.metadata["route_seed_nodes"] = []
        state.runtime.metadata["graph"] = p._build_graph_payload(
            map_seeds, [], intent=intent
        )

        return {"answer": "\n".join(lines), "metadata": state.runtime.metadata}

    def _answer_attraction_classification_analysis_if_possible(
        self, state: PipelineRunState
    ) -> str | None:
        q = normalize_text(state.user_query, strip_punct=True)
        if not any(
            token in q
            for token in ["phan tich", "y nghia", "gia tri", "quan ly", "phat trien"]
        ):
            return None
        if not any(
            token in q
            for token in [
                "touristattraction",
                "diem du lich",
                "danh lam",
                "phan loai",
                "danh muc",
            ]
        ):
            return None

        context_lines: List[str] = []
        for item in state.raw_context or []:
            context_lines.extend(
                line.strip() for line in str(item or "").splitlines() if line.strip()
            )
        if state.clean_context:
            context_lines.extend(
                line.strip()
                for line in str(state.clean_context or "").splitlines()
                if line.strip()
            )
        context_lines = list(dict.fromkeys(context_lines))
        if not context_lines:
            return None

        subject = ""
        subject_type = ""
        address = ""
        location_text = ""
        located_in = ""
        category = ""
        for line in context_lines:
            main_match = re.search(
                r"\*\*(?:THỰC THỂ CHÍNH|THUC THE CHINH|TH.+?C TH.+? CH.+?NH):\*\*\s*(.+?)\s*"
                r"\((?:Loại|Loai|Lo.+?i):\s*(.+?)\)",
                line,
            )
            if main_match:
                subject = main_match.group(1).strip()
                subject_type = main_match.group(2).strip()
                continue
            addr_match = re.search(r"^-\s*address:\s*(.+?)\s*$", line)
            if addr_match:
                address = addr_match.group(1).strip()
                continue
            loc_attr_match = re.search(r"^-\s*location:\s*(.+?)\s*$", line)
            if loc_attr_match:
                location_text = loc_attr_match.group(1).strip()
                continue
            located_match = re.search(
                r"^-\s*(.+?)\s+\[LOCATED_IN\]\s*->\s*(.+?)\s*$", line
            )
            if located_match:
                subject = subject or located_match.group(1).strip()
                located_in = located_match.group(2).strip()
                continue
            category_match = re.search(
                r"^-\s*(.+?)\s+\[BELONGS_TO\]\s*->\s*(.+?)\s*$", line
            )
            if category_match:
                subject = subject or category_match.group(1).strip()
                category = category_match.group(2).strip()

        if not subject:
            for node in state.grounded_nodes or state.all_seeds or []:
                subject = str(
                    (getattr(node, "metadata", {}) or {}).get("name")
                    or getattr(node, "content", "")
                    or ""
                ).strip()
                labels = (getattr(node, "metadata", {}) or {}).get("labels") or []
                subject_type = str(labels[0]) if labels else subject_type
                break

        if not subject or not (located_in or address) or not category:
            return None

        area = located_in or address
        subject_label = (
            f"{subject} ({subject_type})" if subject_type else subject
        )
        lines = [
            f"{subject_label} được xác định trong dữ liệu là một điểm du lịch "
            f"tại {area} và thuộc danh mục {category}.",
            f"Ý nghĩa chính của việc phân loại này là làm rõ bản chất tài nguyên du lịch "
            f"của {subject}: đây không chỉ là một địa danh có tọa độ, mà là một tài nguyên "
            f"cảnh quan/di sản có thể được đưa vào hệ thống quản lý điểm đến.",
            f"Quan hệ LOCATED_IN với {area} giúp địa phương quản lý theo không gian hành chính: "
            f"lập bản đồ, điều phối hạ tầng, kết nối tuyến tham quan và phân quyền "
            f"cập nhật dữ liệu.",
            f"Quan hệ BELONGS_TO với {category} giúp chuẩn hóa loại hình sản phẩm du lịch. "
            f"Nhờ đó, hệ thống có thể nhóm {subject} với các điểm danh lam thắng cảnh khác, "
            f"phục vụ tìm kiếm, thống kê, quảng bá và thiết kế tuyến theo chủ đề.",
        ]
        if location_text:
            lines.append(
                f"Tọa độ {location_text} bổ sung giá trị định vị, hỗ trợ dẫn đường "
                f"và phân tích khoảng cách khi xây dựng lịch trình."
            )
        lines.append(
            "Với phát triển du lịch địa phương, các quan hệ này quan trọng vì chúng "
            "biến thông tin rời rạc thành dữ liệu có cấu trúc: biết điểm nằm ở đâu, "
            "thuộc loại gì, và nên được kết nối với nhóm sản phẩm nào. Đây là nền tảng "
            "để quản lý tài nguyên, ưu tiên đầu tư, truyền thông điểm đến và xây dựng "
            "trải nghiệm phù hợp cho du khách."
        )
        return "\n".join(lines)

    def _answer_lodging_heritage_strategy_if_possible(
        self, state: PipelineRunState
    ) -> str | None:
        q = normalize_text(state.user_query, strip_punct=True)
        if not any(
            token in q
            for token in ["phan tich", "chien luoc", "tiem nang", "phat trien"]
        ):
            return None
        if not any(
            token in q
            for token in ["nha nghi", "khach san", "luu tru", "nghi duong"]
        ):
            return None
        if not any(
            token in q
            for token in ["di san", "lich su", "dia ly", "van hoa", "ket hop"]
        ):
            return None

        context_lines: List[str] = []
        for item in state.raw_context or []:
            context_lines.extend(
                line.strip() for line in str(item or "").splitlines() if line.strip()
            )
        if state.clean_context:
            context_lines.extend(
                line.strip()
                for line in str(state.clean_context or "").splitlines()
                if line.strip()
            )
        context_lines = list(dict.fromkeys(context_lines))

        subject = ""
        subject_type = ""
        location_text = ""
        near_places: List[str] = []
        for line in context_lines:
            main_match = re.search(
                r"\*\*(?:THỰC THỂ CHÍNH|THUC THE CHINH|TH.+?C TH.+? CH.+?NH):\*\*\s*(.+?)\s*"
                r"\((?:Loại|Loai|Lo.+?i):\s*(.+?)\)",
                line,
            )
            if main_match:
                subject = main_match.group(1).strip()
                subject_type = main_match.group(2).strip()
                continue
            loc_match = re.search(r"^-\s*location:\s*(.+?)\s*$", line)
            if loc_match:
                location_text = loc_match.group(1).strip()
                continue
            near_match = re.search(
                r"^-\s*(.+?)\s+\[NEAR\]\s*->\s*(.+?)\s*$", line
            )
            if near_match:
                left = near_match.group(1).strip()
                right = near_match.group(2).strip()
                if not subject:
                    subject = left
                subject_norm = normalize_text(subject, strip_punct=True)
                left_norm = normalize_text(left, strip_punct=True)
                place = right if subject_norm == left_norm else left
                if place:
                    near_places.append(place)

        if not subject:
            for node in state.grounded_nodes or state.all_seeds or []:
                node_name = str(
                    (getattr(node, "metadata", {}) or {}).get("name")
                    or getattr(node, "content", "")
                    or ""
                ).strip()
                labels = (getattr(node, "metadata", {}) or {}).get("labels") or []
                label_set = set(labels)
                if label_set & LODGING_LABELS:
                    subject = node_name
                    subject_type = str(list(label_set & LODGING_LABELS)[0])
                    break

        near_places = list(dict.fromkeys([p for p in near_places if p]))

        grounded_accommodations: List[str] = []
        grounded_heritage: List[str] = []
        for node in state.grounded_nodes or state.all_seeds or []:
            node_name = str(
                (getattr(node, "metadata", {}) or {}).get("name")
                or getattr(node, "content", "")
                or ""
            ).strip()
            labels = (getattr(node, "metadata", {}) or {}).get("labels") or []
            label_set = set(labels)
            if not node_name:
                continue
            if label_set & LODGING_LABELS:
                grounded_accommodations.append(node_name)
            elif label_set & HERITAGE_LABELS:
                grounded_heritage.append(node_name)

        # Level 1: Rich context
        if subject and len(near_places) >= 2:
            return self._format_lodging_heritage_rich(
                subject, subject_type, location_text, near_places
            )

        # Level 2: Thin context
        if subject:
            augmented_near = near_places or grounded_heritage
            if len(augmented_near) >= 1:
                return self._format_lodging_heritage_thin(
                    subject,
                    subject_type,
                    location_text,
                    augmented_near,
                    grounded_accommodations,
                )

        # Level 3: Minimal context
        if grounded_accommodations and grounded_heritage:
            return self._format_lodging_heritage_minimal(
                grounded_accommodations, grounded_heritage
            )

        return None

    def _format_lodging_heritage_rich(
        self,
        subject: str,
        subject_type: str,
        location_text: str,
        near_places: List[str],
    ) -> str:
        heritage_places = []
        for place in near_places:
            n = normalize_text(place, strip_punct=True)
            if any(
                token in n
                for token in ["di tich", "khao co", "tham sat", "lich su", "di san"]
            ):
                heritage_places.append(place)

        display_name = self._clean_display_name(subject)
        place_names = [self._clean_display_name(p) for p in near_places]
        heritage_names = [
            self._clean_display_name(p) for p in (heritage_places or near_places)
        ]

        lines = ["**Phân tích tiềm năng tuyến lưu trú - di sản:**", ""]
        lines.append(f"**Cơ sở lưu trú:** {display_name}")
        if location_text:
            lines.append(f"- Vị trí: {location_text}")
        lines.append(f"- Gần: {', '.join(place_names)}")
        lines.append("")
        lines.append(f"**Điểm di tích/di sản:** {', '.join(heritage_names)}")
        lines.append("")
        lines.append(
            "**Đánh giá:** Tiềm năng phát triển mô hình lưu trú kết hợp tham quan di sản. "
            "Phù hợp gói nghỉ ngắn ngày kết hợp tìm hiểu lịch sử địa phương."
        )
        return "\n".join(lines)

    def _format_lodging_heritage_thin(
        self,
        subject: str,
        subject_type: str,
        location_text: str,
        heritage_sites: List[str],
        all_accommodations: List[str],
    ) -> str:
        display_name = self._clean_display_name(subject)
        heritage_text = ", ".join(heritage_sites[:5])

        lines = ["**Phân tích tiềm năng tuyến lưu trú - di sản:**", ""]
        lines.append(f"**Cơ sở lưu trú:** {display_name}")
        if location_text:
            lines.append(f"- Vị trí: {location_text}")
        if all_accommodations:
            acc_names = [self._clean_display_name(a) for a in all_accommodations[:5]]
            lines.append(f"- Cơ sở lưu trú khác: {', '.join(acc_names)}")
        lines.append("")
        lines.append(f"**Điểm di tích/di sản:** {heritage_text}")
        lines.append("")
        lines.append(
            "**Đánh giá:** Tiềm năng kết hợp lưu trú - di sản là khả thi. "
            "Dữ liệu chưa có thông tin khoảng cách cụ thể giữa nhà nghỉ và di tích."
        )
        return "\n".join(lines)

    def _format_lodging_heritage_minimal(
        self, accommodations: List[str], heritage_sites: List[str]
    ) -> str:
        acc_names = [self._clean_display_name(a) for a in accommodations[:5]]
        heritage_names = [self._clean_display_name(h) for h in heritage_sites[:5]]

        lines = ["**Phân tích tiềm năng tuyến lưu trú - di sản:**", ""]
        lines.append("**Dữ liệu hiện có:**")
        lines.append(f"- {len(accommodations)} cơ sở lưu trú: {', '.join(acc_names)}")
        lines.append(f"- {len(heritage_sites)} điểm di tích: {', '.join(heritage_names)}")
        lines.append("")
        lines.append(
            "**Đánh giá:** Tiềm năng kết hợp là khả thi. "
            "Dữ liệu chưa có thông tin khoảng cách cụ thể giữa nhà nghỉ và di tích."
        )
        return "\n".join(lines)

    def _answer_spatial_strategy_analysis_from_context_if_possible(
        self, state: PipelineRunState
    ) -> str | None:
        q = normalize_text(state.user_query, strip_punct=True)
        if not any(
            token in q
            for token in [
                "phan tich",
                "chien luoc",
                "loi the",
                "tiem nang",
                "dinh vi",
                "phat trien",
            ]
        ):
            return None
        if not any(
            token in q
            for token in [
                "khong gian",
                "vi tri",
                "gan",
                "lan can",
                "xung quanh",
                "moi quan he",
                "dia ly",
                "lich su",
                "di san",
            ]
        ):
            return None

        context_lines: List[str] = []
        for item in state.raw_context or []:
            context_lines.extend(
                line.strip() for line in str(item or "").splitlines() if line.strip()
            )
        if state.clean_context:
            context_lines.extend(
                line.strip()
                for line in str(state.clean_context or "").splitlines()
                if line.strip()
            )
        context_lines = list(dict.fromkeys(context_lines))
        if not context_lines:
            return None

        subject = ""
        subject_type = ""
        location_text = ""
        near_places: List[str] = []
        for line in context_lines:
            main_match = re.search(
                r"\*\*(?:THỰC THỂ CHÍNH|THUC THE CHINH):\*\*\s*(.+?)\s*"
                r"\((?:Loại|Loai):\s*(.+?)\)",
                line,
            )
            if main_match:
                subject = main_match.group(1).strip()
                subject_type = main_match.group(2).strip()
                continue
            loc_match = re.search(r"^-\s*location:\s*(.+?)\s*$", line)
            if loc_match:
                location_text = loc_match.group(1).strip()
                continue
            near_match = re.search(
                r"^-\s*(.+?)\s+\[NEAR\]\s*->\s*(.+?)\s*$", line
            )
            if near_match:
                left = near_match.group(1).strip()
                right = near_match.group(2).strip()
                if not subject:
                    subject = left
                if normalize_text(subject, strip_punct=True) in {
                    normalize_text(left, strip_punct=True),
                    normalize_text(right, strip_punct=True),
                }:
                    place = (
                        right
                        if normalize_text(subject, strip_punct=True)
                        == normalize_text(left, strip_punct=True)
                        else left
                    )
                else:
                    place = right
                if place:
                    near_places.append(place)

        if not subject:
            for node in state.grounded_nodes or state.all_seeds or []:
                subject = str(
                    (getattr(node, "metadata", {}) or {}).get("name")
                    or getattr(node, "content", "")
                    or ""
                ).strip()
                labels = (getattr(node, "metadata", {}) or {}).get("labels") or []
                subject_type = str(labels[0]) if labels else subject_type
                break
        near_places = list(dict.fromkeys([p for p in near_places if p]))
        if not subject or len(near_places) < 2:
            return None

        def classify(place: str) -> str:
            n = normalize_text(place, strip_punct=True)
            if any(token in n for token in keywords.CLASSIFY_HERITAGE_KEYWORDS):
                return "văn hóa - lịch sử"
            if any(token in n for token in keywords.CLASSIFY_SPIRITUAL_KEYWORDS):
                return "tâm linh - kiến trúc"
            if any(token in n for token in keywords.CLASSIFY_CRAFT_KEYWORDS):
                return "văn hóa bản địa - làng nghề"
            if any(token in n for token in keywords.CLASSIFY_PUBLIC_SPACE_KEYWORDS):
                return "không gian công cộng - thư giãn"
            if any(token in n for token in keywords.CLASSIFY_NATURE_KEYWORDS):
                return "thiên nhiên - cảnh quan"
            return "tham quan tổng hợp"

        groups: Dict[str, List[str]] = {}
        for place in near_places:
            groups.setdefault(classify(place), []).append(place)

        subject_label = (
            f"{subject} ({subject_type})" if subject_type else subject
        )
        lines = [
            f"Dựa trên các quan hệ không gian được cung cấp, {subject_label} "
            f"có lợi thế ở vai trò điểm lưu trú/điểm xuất phát gần một cụm "
            f"điểm tham quan đa dạng."
        ]
        if location_text:
            lines.append(f"Vị trí tham chiếu trong dữ liệu là {location_text}.")

        lines.append(
            "Các điểm lân cận gồm: " + ", ".join(near_places) + "."
        )
        lines.append("Có thể phân nhóm giá trị du lịch như sau:")
        for group, places in groups.items():
            lines.append(f"- {group}: {', '.join(places)}")

        advantage_parts = []
        if "văn hóa - lịch sử" in groups:
            advantage_parts.append(
                "tiếp cận nhanh các điểm tìm hiểu lịch sử và văn hóa địa phương"
            )
        if "tâm linh - kiến trúc" in groups:
            advantage_parts.append(
                "kết hợp tham quan chùa/không gian tâm linh trong cùng hành trình"
            )
        if "văn hóa bản địa - làng nghề" in groups:
            advantage_parts.append(
                "mở rộng trải nghiệm sang văn hóa bản địa và thủ công truyền thống"
            )
        if "không gian công cộng - thư giãn" in groups:
            advantage_parts.append(
                "có điểm nghỉ chân, dạo bộ hoặc sinh hoạt cộng đồng gần nơi lưu trú"
            )
        if "thiên nhiên - cảnh quan" in groups:
            advantage_parts.append(
                "kết hợp tham quan cảnh quan thiên nhiên"
            )
        if not advantage_parts:
            advantage_parts.append(
                "giảm thời gian di chuyển giữa nhiều điểm tham quan gần nhau"
            )

        lines.append(
            "Lợi thế chính là " + "; ".join(advantage_parts) + "."
        )

        has_cultural_or_heritage = "văn hóa - lịch sử" in groups
        has_indigenous = "văn hóa bản địa - làng nghề" in groups
        has_spiritual = "tâm linh - kiến trúc" in groups
        has_nature = "thiên nhiên - cảnh quan" in groups

        if has_cultural_or_heritage and not has_indigenous and not has_spiritual:
            tourism_type = "du lịch di sản - lịch sử kết hợp nghỉ dưỡng"
        elif has_cultural_or_heritage or has_indigenous or has_spiritual:
            tourism_type = (
                "du lịch văn hóa - lịch sử kết hợp tâm linh và trải nghiệm bản địa"
            )
        elif has_nature:
            tourism_type = "du lịch nghỉ dưỡng kết hợp khám phá thiên nhiên"
        else:
            tourism_type = "du lịch tham quan đô thị ngắn ngày"

        lines.append(
            f"Vì vậy, loại hình phù hợp nhất là {tourism_type}. {subject} "
            f"có thể được định vị như một điểm lưu trú thuận tiện cho du khách "
            f"muốn đi nhiều điểm trong một ngày mà vẫn quay lại nghỉ ngơi dễ dàng."
        )

        return "\n".join(lines)
