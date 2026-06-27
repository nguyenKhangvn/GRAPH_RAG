from __future__ import annotations
from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable
"""Tour list, tour plan, and lodging itinerary dispatch."""
import logging

logger = logging.getLogger(__name__)


import re





from graph_rag.utils.text import normalize_text


from ..dto import PipelineRunState


class TourDispatchMixin:
    """Mixin providing tour availability, tour plan, and lodging itinerary dispatch."""

    def _dispatch_tour_list(self, state: PipelineRunState, generator_candidates: list, full_generator_candidates: list) -> str:
        """Tour availability: list/rank Tour nodes matching user constraints. No route optimization."""
        from graph_rag.utils.text import normalize_text
        import re as _re
        p = self.pipeline
        qs = state.query_plan

        # 1. Filter Tour nodes from candidates
        tour_candidates = []
        for c in (full_generator_candidates or []):
            labels = set()
            if isinstance(c, dict):
                labels = {str(l).lower() for l in (c.get("labels") or [])}
                if not labels and c.get("type"):
                    labels = {str(c["type"]).lower()}
            else:
                meta_labels = getattr(c, "metadata", {}) or {}
                labels = {str(l).lower() for l in (meta_labels.get("labels") or [])}
                if not labels:
                    t = meta_labels.get("type")
                    if t:
                        labels = {str(t).lower()}
            if "tour" in labels:
                tour_candidates.append(c)

        # Also try to fetch Tour nodes from graph if few candidates
        if len(tour_candidates) < 3 and hasattr(p, 'driver') and p.driver:
            try:
                from graph_rag.core.state import NodeItem
                with p.driver.session() as session:
                    result = session.run(
                        """
                        MATCH (t:Tour)
                        WHERE t.name IS NOT NULL
                        RETURN t.id AS id, t.name AS name, t.description AS description,
                               t.price AS price, t.duration AS duration, labels(t) AS labels
                        LIMIT 20
                        """
                    )
                    existing_ids = {getattr(c, "id", None) or (c.get("id") if isinstance(c, dict) else None) for c in tour_candidates}
                    for record in result:
                        if record["id"] in existing_ids:
                            continue
                        tour_candidates.append(NodeItem(
                            id=record["id"],
                            content=record["name"] or "",
                            score=0.8,
                            source_type="graph_scan",
                            metadata={
                                "name": record["name"],
                                "description": record.get("description"),
                                "price": record.get("price"),
                                "duration": record.get("duration"),
                                "labels": record["labels"] or ["Tour"],
                                "type": "Tour",
                            },
                        ))
            except (ValueError, RuntimeError, OSError) as e:
                logger.error("   -> [TourList] Graph scan error: %s", e)

        if not tour_candidates:
            return "Hiện tại tôi chưa tìm thấy tour nào phù hợp trong dữ liệu."

        # 2. TourMatcher helpers
        def _tour_field_text(tour, *keys: str) -> str:
            values = []
            if isinstance(tour, dict):
                attrs = tour.get("attributes") or {}
                for key in keys:
                    values.append(tour.get(key, ""))
                    values.append(attrs.get(key, ""))
            else:
                meta = getattr(tour, "metadata", {}) or {}
                for key in keys:
                    values.append(meta.get(key, ""))
                    values.append(getattr(tour, key, ""))
            return normalize_text(" ".join(str(v or "") for v in values), strip_punct=True)

        def _extract_tour_days(text: str) -> float:
            raw = str(text or "")
            t = normalize_text(raw, strip_punct=True)
            raw_norm = normalize_text(raw, strip_punct=False)
            if any(marker in t for marker in ["nua ngay", "ban ngay", "half day"]):
                return 0.5
            if _re.search(r"\b1\s*/\s*2\s*ngay\b", raw_norm, flags=_re.IGNORECASE):
                return 0.5
            if _re.search(r"\b1\s+2\s*ngay\b", t):
                return 0.5
            match = _re.search(r"(?<!/)\b(\d+)\s*(?:ngay|nay|ngy|n)\b", t)
            return float(match.group(1)) if match else 0.0

        def _contains_any(text: str, terms: set[str]) -> bool:
            return any(term in text for term in terms)

        # Vietnamese labels for constraint keys — only show constraints user asked about
        _LABEL_MAP = {
            "duration": "thời lượng",
            "coastal": "biển/đảo",
            "sunset": "hoàng hôn",
            "island": "đảo",
        }
        _active_keys = set()
        if qs:
            if qs.duration_days > 0:
                _active_keys.add("duration")
            if qs.coastal_required:
                _active_keys.add("coastal")
            if qs.sunset_required:
                _active_keys.add("sunset")
            if qs.island_required:
                _active_keys.add("island")

        def _label(key: str) -> str:
            return _LABEL_MAP.get(key, key)

        # 3. TourMatcher: check constraint coverage
        matched = []
        partial = []

        for tour in tour_candidates:
            tour_name = ""
            tour_desc = ""
            tour_duration = ""
            if isinstance(tour, dict):
                tour_name = str(tour.get("name", ""))
                tour_desc = str((tour.get("attributes") or {}).get("description", ""))
                tour_duration = str((tour.get("attributes") or {}).get("duration", ""))
            else:
                meta = getattr(tour, "metadata", {}) or {}
                tour_name = str(meta.get("name", ""))
                tour_desc = str(meta.get("description", ""))
                tour_duration = str(meta.get("duration", ""))

            # Separate text layers for strict vs loose matching
            identity_text = _tour_field_text(tour, "name", "duration", "title")
            itinerary_text = _tour_field_text(tour, "name", "duration", "itinerary", "included_points", "highlights", "summary")
            full_text = _tour_field_text(tour, "name", "duration", "itinerary", "included_points", "highlights", "summary", "description", "content")

            checks = {}

            # Duration check — strict: must match exactly
            if qs and qs.duration_days > 0:
                tour_days = _extract_tour_days(f"{tour_name} {tour_duration}")
                checks["duration"] = tour_days == qs.duration_days
            else:
                checks["duration"] = True

            # Coastal check — strong terms in itinerary, weak terms in identity
            if qs and qs.coastal_required:
                coastal_strong = {"ky co", "eo gio", "cu lao xanh", "hon kho", "nhon ly", "bai bien", "bien dao", "trung luong", "cat tien", "bai xep"}
                coastal_weak = {"bien", "dao"}
                checks["coastal"] = _contains_any(itinerary_text, coastal_strong) or _contains_any(identity_text, coastal_weak)
            else:
                checks["coastal"] = True

            # Sunset check — STRICT: require explicit sunset evidence in itinerary/highlights
            # Place names (Kỳ Co, Eo Gió) alone are NOT enough
            if qs and qs.sunset_required:
                sunset_evidence = {"hoang hon", "ngam hoang hon", "sunset", "chieu ta", "chieu toi", "mặt trời lặn"}
                checks["sunset"] = _contains_any(itinerary_text, sunset_evidence)
            else:
                checks["sunset"] = True

            # Island check
            if qs and qs.island_required:
                island_terms = {"cu lao xanh", "hon kho", "ky co", "dao"}
                checks["island"] = _contains_any(itinerary_text, island_terms)
            else:
                checks["island"] = True

            # Classification: full vs partial
            # Full: ALL constraints satisfied
            # Partial: must match at least coastal OR sunset (not just island/duration),
            #          and at least 2 hard constraints total
            if all(checks.values()):
                matched.append((tour, checks))
            else:
                # Only count constraints the user actually asked about
                active_checks = {k: v for k, v in checks.items() if k in _active_keys}
                satisfied_active = [k for k, v in active_checks.items() if v]
                has_coastal_or_sunset = checks.get("coastal") or checks.get("sunset")
                num_satisfied = len(satisfied_active)

                # Partial rule: must match coastal OR sunset, and at least 2 active constraints
                if has_coastal_or_sunset and num_satisfied >= 2:
                    partial.append((tour, checks))

        # 4. Build answer
        def _format_tour_entry(tour, checks, match_status):
            name = ""
            desc = ""
            price = ""
            duration = ""
            if isinstance(tour, dict):
                name = tour.get("name", "")
                attrs = tour.get("attributes") or {}
                desc = attrs.get("description", "")
                price = attrs.get("price", "")
                duration = attrs.get("duration", "")
            else:
                meta = getattr(tour, "metadata", {}) or {}
                name = meta.get("name", "")
                desc = meta.get("description", "")
                price = meta.get("price", "")
                duration = meta.get("duration", "")

            lines = [f"**{name}**"]
            if duration:
                lines.append(f"  - Thời gian: {duration}")
            if price:
                lines.append(f"  - Giá: {price}")
            if desc:
                short_desc = desc[:200] + "..." if len(desc) > 200 else desc
                lines.append(f"  - {short_desc}")

            # Only show constraints the user asked about, in Vietnamese
            active_checks = {k: v for k, v in checks.items() if k in _active_keys}
            satisfied = [_label(k) for k, v in active_checks.items() if v]
            unsatisfied = [_label(k) for k, v in active_checks.items() if not v]
            if match_status == "full":
                lines.append(f"  - ✅ Đáp ứng: {', '.join(satisfied)}")
            else:
                if satisfied:
                    lines.append(f"  - ✅ Đáp ứng: {', '.join(satisfied)}")
                if unsatisfied:
                    lines.append(f"  - ❌ Chưa đáp ứng: {', '.join(unsatisfied)}")
            return "\n".join(lines)

        parts = []
        total_partial = len(partial)
        if total_partial > 5:
            partial = partial[:5]
        if matched:
            parts.append(f"Tìm thấy **{len(matched)} tour** phù hợp đầy đủ:\n")
            for i, (tour, checks) in enumerate(matched, 1):
                parts.append(f"{i}. {_format_tour_entry(tour, checks, 'full')}\n")
        if partial:
            parts.append(f"\nCó **{len(partial)} tour** gần phù hợp (chưa đáp ứng đầy đủ):\n")
            for i, (tour, checks) in enumerate(partial, 1):
                parts.append(f"{i}. {_format_tour_entry(tour, checks, 'partial')}\n")
        if not matched and not partial:
            parts.append("Không tìm thấy tour nào phù hợp với yêu cầu.")

        return "\n".join(parts)

    def _dispatch_tour_plan(self, state: PipelineRunState, generator_candidates: list, full_generator_candidates: list) -> str:
        """Tour plan: use existing tour plan generation logic."""
        p = self.pipeline

        # If SanityGate blocked generation, return informative fallback instead of wrong itinerary
        if (state.metadata or {}).get("route_constraint_blocked"):
            qs = state.query_plan
            missing = []
            if getattr(qs, "coastal_required", False):
                missing.append("biển/đảo (Kỳ Co, Eo Gió, Cù Lao Xanh, Hòn Khô)")
            if getattr(qs, "sunset_required", False):
                missing.append("điểm ngắm hoàng hôn")
            if getattr(qs, "island_required", False):
                missing.append("đảo (Cù Lao Xanh, Hòn Khô, Kỳ Co)")
            missing_text = ", ".join(missing)
            return (
                f"Xin lỗi, hiện tại tôi chưa tìm đủ điểm {missing_text} "
                f"để xây dựng lịch trình phù hợp với yêu cầu của bạn. "
                f"Bạn có thể thử mở rộng phạm vi tìm kiếm hoặc giảm bớt yêu cầu ràng buộc."
            )

        # Inject constraint hints into context
        context = state.clean_context or ""
        constraint_hint = (state.metadata or {}).get("route_constraint_hint")
        if constraint_hint:
            context = f"**YÊU CẦU BẮT BUỘC:** {constraint_hint}\n\n{context}"
        plan = state.query_plan
        intent = plan.intent if plan else state.primary_intent
        answer = p.generator.generate(
            user_query=state.user_query,
            context_text=context,
            intent=intent,
            detected_location=state.location,
            candidate_nodes=full_generator_candidates,
            strict_route_nodes=state.runtime.metadata.get("route_seed_nodes"),
            dropped_route_points=state.runtime.metadata.get("dropped_route_points"),
            daily_cluster_plan=state.runtime.metadata.get("daily_cluster_plan"),
            route_optimizer_metrics=state.runtime.metadata.get("route_optimizer_metrics"),
            lodging_suggestions=state.runtime.metadata.get("lodging_suggestions"),
            query_state=state.query_plan,
        )
        answer = answer or ""
        if self._context_has_facts(state.raw_context):
            cleaned = self._remove_tour_plan_apology_caveat(answer)
            if cleaned != answer:
                state.runtime.metadata["tour_plan_apology_caveat_removed"] = True
            answer = cleaned
        return answer


    def _answer_lodging_cultural_itinerary_if_possible(self, state: PipelineRunState) -> str:
        q_norm = normalize_text(state.user_query, strip_punct=True)
        if not all(marker in q_norm for marker in ["luu tru", "lich trinh"]):
            return ""
        if not any(marker in q_norm for marker in ["van hoa", "tam linh", "di tich", "lich su", "thac", "gan noi o", "gan cho o", "gan khach san", "gan nha nghi"]):
            return ""
        # Skip for comparison questions — let comparison logic handle it
        if "so sanh" in q_norm:
            return ""
        frame = (state.metadata or {}).get("query_frame") or {}
        if frame.get("query_operator") == "comparison":
            return ""

        lodging_name = self._extract_lodging_anchor_from_query(state.user_query)
        if not lodging_name:
            for entity in state.entities or []:
                if not isinstance(entity, dict):
                    continue
                e_type = str(entity.get("type") or "").strip().lower()
                e_name = str(entity.get("name") or "").strip()
                if e_name and e_type in {"accommodation", "hotel", "lodging"}:
                    lodging_name = e_name
                    break
        if not lodging_name:
            return ""

        location_hint = normalize_text(state.location or "", strip_punct=True)
        # Extract city-level keyword for matching (e.g., "quy nhon" from "phuong quy nhon")
        location_city = location_hint
        for prefix in ["phuong ", "xa ", "thi xa ", "thanh pho ", "quan ", "huyen "]:
            if location_hint.startswith(prefix):
                location_city = location_hint[len(prefix):]
                break
        cypher = """
        MATCH (lodging:Accommodation)
        WHERE toLower(lodging.name) = toLower($name)
           OR toLower(lodging.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(lodging.name)
        OPTIONAL MATCH (lodging)-[:LOCATED_IN]->(lodging_admin)
        OPTIONAL MATCH (lodging)-[:NEAR]-(poi:TouristAttraction)
        WHERE poi IS NULL
           OR $location_city = ""
           OR toLower(coalesce(poi.address, '')) CONTAINS toLower($location_city)
           OR toLower(coalesce(poi.name, '')) CONTAINS toLower($location_city)
        OPTIONAL MATCH (poi)-[:BELONGS_TO]->(cat)
        OPTIONAL MATCH (poi)-[:LOCATED_IN]->(poi_admin)
        WITH lodging, poi, cat, lodging_admin, poi_admin
        WHERE poi IS NULL
           OR lodging_admin IS NULL
           OR poi_admin IS NULL
           OR lodging_admin.name = poi_admin.name
           OR toLower(coalesce(poi.address, '')) CONTAINS toLower(coalesce(lodging.address, ''))
           OR toLower(coalesce(lodging.address, '')) CONTAINS toLower(coalesce(poi.address, ''))
        ORDER BY
          CASE
            WHEN cat.name CONTAINS 'Di tích' THEN 0
            WHEN cat.name CONTAINS 'Làng nghề' THEN 1
            WHEN cat.name CONTAINS 'Danh lam' THEN 2
            ELSE 3
          END,
          poi.name
        RETURN lodging.name AS lodging,
               lodging.address AS lodging_address,
               collect(DISTINCT {
                 name: poi.name,
                 address: poi.address,
                 category: cat.name
               }) AS pois
        LIMIT 1
        """
        try:
            with self.pipeline.driver.session() as session:
                row = session.run(cypher, name=lodging_name, location_city=location_city).single()
        except (Neo4jClientError, ServiceUnavailable) as exc:
            self.pipeline.logger.warning("lodging_cultural_itinerary_query_failed: %s", str(exc))
            return ""
        if not row:
            # Cypher returned no row — accommodation may exist but has no NEAR edges.
            # Still return a message with the lodging name.
            return f"Hiện tại dữ liệu chưa ghi nhận điểm văn hóa/tâm linh gần {lodging_name} để lập lịch trình."

        resolved_lodging = str(row.get("lodging") or lodging_name).strip()
        raw_pois = row.get("pois") or []
        preferred = []
        seen = set()
        for item in raw_pois:
            name = str((item or {}).get("name") or "").strip()
            category = str((item or {}).get("category") or "").strip()
            if not name or name in seen:
                continue
            cat_norm = normalize_text(category, strip_punct=True)
            name_norm = normalize_text(name, strip_punct=True)
            if any(marker in cat_norm for marker in ["di tich", "lich su", "lang nghe", "danh lam", "bien", "vinh", "san khau", "cong vien"]) or any(
                marker in name_norm for marker in ["chua", "bao tang", "nha lao", "quang truong", "bien", "vinh", "bai", "ho", "thac"]
            ):
                preferred.append({
                    "name": name,
                    "address": str((item or {}).get("address") or "").strip(),
                    "category": category or "điểm văn hóa/tâm linh",
                })
                seen.add(name)
            if len(preferred) >= 3:
                break
        if not preferred:
            return f"Hiện tại dữ liệu chưa ghi nhận điểm văn hóa/tâm linh gần {resolved_lodging} để lập lịch trình."

        first = preferred[0]
        second = preferred[1] if len(preferred) > 1 else None
        is_full_day = any(m in q_norm for m in ["trong ngay", "mot ngay", "1 ngay", "ca ngay"])
        if is_full_day:
            lines = [
                f"Dựa trên dữ liệu NEAR trong hệ thống, mình gợi ý lịch trình trong ngày bắt đầu từ {resolved_lodging}:",
                "",
                "Buổi sáng:",
                f"- 08:00: Xuất phát từ {resolved_lodging}.",
                f"- 08:15 - 10:00: Tham quan {first['name']} ({first['category']}).",
            ]
            if second:
                lines.append(f"- 10:15 - 11:30: Ghé {second['name']} ({second['category']}).")
            elif len(preferred) == 1:
                lines.append("- 10:15 - 11:30: Dành thêm thời gian tham quan/chụp ảnh tại điểm chính.")
            lines.extend([
                "- 11:30 - 13:00: Nghỉ trưa, ăn uống.",
                "",
                "Buổi chiều:",
            ])
            third = preferred[2] if len(preferred) > 2 else None
            if third:
                lines.append(f"- 13:30 - 15:00: Tham quan {third['name']} ({third['category']}).")
            else:
                lines.append(f"- 13:30 - 15:30: Tiếp tục khám phá {first['name']} hoặc tham quan tự do quanh khu vực.")
            lines.extend([
                "- 16:00: Quay lại nơi lưu trú.",
                "",
                "Lưu ý: Lịch trình dựa trên dữ liệu NEAR trong hệ thống. Nếu du khách muốn mở rộng sang thác nước hoặc điểm xa hơn, cần thêm thông tin về khoảng cách và phương tiện di chuyển.",
            ])
        else:
            lines = [
                f"Dựa trên dữ liệu NEAR trong hệ thống, mình gợi ý lịch trình nửa ngày bắt đầu từ {resolved_lodging}:",
                "",
                f"- 08:00: Xuất phát từ {resolved_lodging}.",
                f"- 08:15 - 09:30: Tham quan {first['name']} ({first['category']}).",
            ]
            if second:
                lines.append(f"- 09:45 - 11:00: Ghé {second['name']} ({second['category']}).")
            elif len(preferred) == 1:
                lines.append("- 09:45 - 11:00: Dành thêm thời gian tham quan/chụp ảnh tại điểm chính vì dữ liệu chỉ ghi nhận một điểm phù hợp gần nơi lưu trú.")
            lines.append(f"- 11:15: Quay lại {resolved_lodging} hoặc nghỉ trưa.")
        lines.append("")
        lines.append(
            "Các điểm được chọn vì có quan hệ gần nơi lưu trú trong dữ liệu và thuộc nhóm văn hóa/tâm linh hoặc di sản."
        )
        return "\n".join(lines)


    def _extract_lodging_anchor_from_query(self, query: str) -> str:
        raw = str(query or "").strip().strip(" \"'“”.,")
        raw = re.sub(r'(?is)^\s*["\']?question["\']?\s*:\s*["\']?', "", raw).strip()
        patterns = [
            r"(?i)lưu\s+trú\s+tại\s+(.+?)(?:,|\.|\s+ưu\s+tiên|\s+muốn|\s+ở\s+pleiku|$)",
            r"(?i)luu\s+tru\s+tai\s+(.+?)(?:,|\.|\s+uu\s+tien|\s+muon|\s+o\s+pleiku|$)",
            r"(?i)tại\s+((?:nhà\s+nghỉ|khách\s+sạn|resort|homestay)\s+.+?)(?:,|\.|\s+ưu\s+tiên|\s+muốn|$)",
            r"(?i)tai\s+((?:nha\s+nghi|khach\s+san|resort|homestay)\s+.+?)(?:,|\.|\s+uu\s+tien|\s+muon|$)",
            r"(?i)(?:gần|gan|của|cua|từ|tu)\s+((?:nhà\s+nghỉ|khách\s+sạn|resort|homestay)\s+.+?)(?:,|\.|\s+ưu\s+tiên|\s+muốn|\s+hãy|\s+va|$)",
            r"(?i)((?:nhà\s+nghỉ|khách\s+sạn|resort|homestay)\s+[A-ZÀ-Ỵ][a-zà-ỹ]*(?:\s+[A-ZÀ-Ỵ][a-zà-ỹ]*)*)(?:,|\.|\s+ưu\s+tiên|\s+muốn|\s+hãy|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, raw)
            if match:
                value = str(match.group(1) or "").strip(" \"'“”.,")
                if value:
                    return value
        return ""


    def _remove_tour_plan_apology_caveat(self, answer: str) -> str:
        """Drop only the apology paragraph about a missing secondary tour.

        Tour-plan questions often ask for a schedule around a hotel plus an
        optional tour name. If hotel/NEAR context is present, the answer should
        provide the feasible nearby schedule instead of embedding an apology
        about the optional tour not being grounded.
        """
        text = str(answer or "").strip()
        if not text:
            return ""
        norm = normalize_text(text, strip_punct=True)
        if "xin loi" not in norm and "khong tim thay" not in norm and "chua co du thong tin" not in norm:
            return text
        if not any(marker in norm for marker in ["lich trinh", "7:00", "8:00", "buoi sang", "nua ngay"]):
            return text

        paragraphs = re.split(r"\n\s*\n", text)
        kept: list[str] = []
        for paragraph in paragraphs:
            p_norm = normalize_text(paragraph, strip_punct=True)
            is_missing_tour_caveat = (
                any(marker in p_norm for marker in ["xin loi", "khong tim thay", "chua co du thong tin", "he thong du lieu"])
                and any(marker in p_norm for marker in ["tour", "con chim", "cheo sup"])
            )
            if is_missing_tour_caveat:
                continue
            kept.append(paragraph)

        cleaned = "\n\n".join(part for part in kept if part.strip()).strip()
        cleaned = re.sub(
            r"(?i)\bDù vậy,\s*mình có thể\s+",
            "Dựa trên các điểm gần khách sạn, mình có thể ",
            cleaned,
        )
        cleaned = re.sub(
            r"(?i)\bNếu bạn có thông tin chi tiết về tour, mình sẵn sàng hỗ trợ thêm nhé!?\s*",
            "",
            cleaned,
        ).strip()
        return cleaned or text
