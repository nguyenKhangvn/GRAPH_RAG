"""Prompt builder — system and user prompt assembly for LLM generation."""

import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import graph_rag.modules.generation.tour_plan_support as tour_plan_support
from graph_rag.core import keywords, thresholds, business_rules
from graph_rag.core.intents import IntentType
from graph_rag.core.schema import GraphSchema
from graph_rag.core.state import QuestionShape
from graph_rag.utils.text import normalize_text

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
logger = logging.getLogger(__name__)


def load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


class PromptBuilder:
    """Assembles system and user prompts for various answer modes."""

    @staticmethod
    def critical_system_facts_block() -> str:
        return business_rules.SYSTEM_FACTS

    @staticmethod
    def few_shot_examples_block() -> str:
        return business_rules.FEW_SHOT_EXAMPLES

    @classmethod
    def build_system_prompt(
        cls,
        answer_mode: Optional[str] = None,
        intent: Optional[str] = None,
        query_state: Optional[Any] = None,
    ) -> str:
        """Build system prompt with schema context, critical facts, and few-shot examples."""
        schema_context = GraphSchema.get_system_prompt_context()
        critical_facts = cls.critical_system_facts_block()
        few_shot = cls.few_shot_examples_block()

        template_name = "system_default.txt"
        if answer_mode:
            from graph_rag.core.answer_mode import AnswerMode
            from graph_rag.core.intents import IntentType

            is_travel_agency_query = False
            if query_state and getattr(query_state, "target_class", None) == "TravelAgency":
                is_travel_agency_query = True

            if is_travel_agency_query:
                template_name = "system_travel_agency.txt"
            elif answer_mode == AnswerMode.FACT_ANSWER:
                template_name = "system_fact.txt"
            elif answer_mode == AnswerMode.PARTIAL_FACT_ANSWER:
                template_name = "system_partial_fact.txt"
            elif answer_mode == AnswerMode.DISCOVERY_LIST:
                if intent == IntentType.FOOD:
                    template_name = "system_curated_recommendation.txt"
                elif intent == IntentType.TOURISM:
                    template_name = "system_tourism_recommendation.txt"
                elif intent == IntentType.EVENT:
                    template_name = "system_event_recommendation.txt"
                elif intent == IntentType.ACCOMMODATION:
                    template_name = "system_accommodation_recommendation.txt"
                elif intent in {IntentType.TRAVEL_ADVICE, IntentType.TRANSPORT_INFO}:
                    template_name = "system_travel_advice.txt"
                else:
                    template_name = "system_discovery.txt"
            elif answer_mode == AnswerMode.OPEN_ANALYSIS:
                template_name = "system_analysis.txt"
            elif AnswerMode.is_closed_form(answer_mode):
                template_name = "system_closed_form.txt"
            elif answer_mode == AnswerMode.DISTANCE:
                template_name = "system_distance.txt"
            elif answer_mode == AnswerMode.CURATED_RECOMMENDATION:
                if intent == IntentType.TOURISM:
                    template_name = "system_tourism_recommendation.txt"
                elif intent == IntentType.EVENT:
                    template_name = "system_event_recommendation.txt"
                elif intent == IntentType.ACCOMMODATION:
                    template_name = "system_accommodation_recommendation.txt"
                else:
                    template_name = "system_curated_recommendation.txt"
            elif intent in {IntentType.TRAVEL_ADVICE, IntentType.TRANSPORT_INFO}:
                template_name = "system_travel_advice.txt"

        template = load_prompt(template_name)
        return template.format(
            critical_facts=critical_facts,
            few_shot=few_shot,
            schema_context=schema_context,
        )

    @classmethod
    def build_user_prompt(
        cls,
        query: str,
        context: str,
        context_validation: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        detected_location: Optional[str] = None,
        query_state: Optional[Any] = None,
        validation_feedback: Optional[str] = None,
        partial_answer_mode: bool = False,
        missing_attrs_text: str = "",
    ) -> str:
        """Build user prompt with context, hints, shape guidelines, and validation."""
        validation_block = ""
        if context_validation:
            validation_block = f"""
        CONTEXT VALIDATION:
        {context_validation}

        Nếu validation báo thiếu thuộc tính/quan hệ mà câu hỏi yêu cầu, KHÔNG được thay thế bằng mô tả chung.
        """

        hint_parts = []
        if detected_location:
            hint_parts.append(f"- Khu vực phát hiện: {detected_location}")
        if intent:
            hint_parts.append(f"- Loại câu hỏi: {intent}")

        if intent and str(intent).upper() in {"ENTITY_FACT", "TOURISM", "ACCOMMODATION", "FOOD", "EVENT"}:
            relations_in_context = re.findall(
                r"^-\s*(.+?)\s+\[(\w+)\]\s*->\s*(.+?)$",
                str(context or ""),
                re.MULTILINE,
            )
            if relations_in_context:
                rel_summary = []
                for left, rel_type, right in relations_in_context[:8]:
                    rel_summary.append(f"  {left} [{rel_type}] {right}")
                hint_parts.append(
                    "- Các quan hệ trong Context (BẮT BUỘC đề cập tất cả):\n"
                    + "\n".join(rel_summary)
                )

        shape_guideline = ""
        if query_state:
            guidelines = []
            if query_state.question_shape == QuestionShape.COMPARISON:
                subjects = query_state.comparison_subjects
                if not subjects:
                    subjects = (
                        query_state.metadata.get("query_frame_anchor_names")
                        or query_state.metadata.get("comparison_subjects_expected")
                        or []
                    )
                if subjects:
                    guidelines.append(
                        f"- Đây là câu hỏi so sánh. BẮT BUỘC phải đề cập và "
                        f"so sánh rõ ràng giữa các thực thể sau: {', '.join(subjects)}."
                    )
            elif query_state.question_shape == QuestionShape.ITINERARY:
                guidelines.append(
                    "- Đây là lịch trình du lịch. BẮT BUỘC phải trình bày "
                    "theo mốc thời gian rõ ràng (ngày, các buổi sáng/chiều/tối)."
                )

            if query_state.requested_attributes:
                guidelines.append(
                    f"- Trả lời tập trung vào các thuộc tính được hỏi: "
                    f"{', '.join(query_state.requested_attributes)}."
                )
                guidelines.append(
                    "- Nếu dữ liệu Context không có thông tin về thuộc tính được hỏi "
                    "(ví dụ: số điện thoại, giờ mở cửa, giá cả), hãy trả lời rõ ràng "
                    "là 'chưa có dữ liệu', TUYỆT ĐỐI KHÔNG tự bịa ra giá trị."
                )

            if guidelines:
                shape_guideline = (
                    "\n        YÊU CẦU CẤU TRÚC TRẢ LỜI (SHAPE-AWARE GUIDELINES):\n        "
                    + "\n        ".join(guidelines)
                    + "\n"
                )

        feedback_block = ""
        if validation_feedback:
            feedback_block = f"""
        [CẢNH BÁO SỬA LỖI TỪ HỆ THỐNG]: Lần trả lời trước của bạn đã bị lỗi kiểm định:
        {validation_feedback}
        Hãy đọc kỹ phản hồi trên và sinh lại câu trả lời chính xác hơn, tránh tuyệt đối các lỗi bị bắt gặp.
        """

        hint_block = ""
        if hint_parts:
            hint_block = (
                "\n        GỢI Ý:\n        "
                + "\n        ".join(hint_parts)
                + "\n"
            )

        partial_guideline = ""
        if partial_answer_mode and missing_attrs_text:
            partial_guideline = f"""
        [CHẾ ĐỘ TRẢ LỜI MỘT PHẦN - PARTIAL ANSWER MODE]:
        Hệ thống phát hiện dữ liệu thiếu một số thuộc tính người dùng hỏi ({missing_attrs_text}).
        YÊU CẦU:
        1. Hãy trả lời đầy đủ các thông tin hiện có trong dữ liệu Context (ví dụ: địa chỉ, mô tả, v.v.).
        2. Ở cuối câu trả lời, hãy ghi chú rõ ràng rằng các thông tin ({missing_attrs_text})
           hiện chưa có dữ liệu trong hệ thống (ghi chú rõ: 'Lưu ý thiếu dữ liệu: ...').
        """

        return f"""
        THÔNG TIN NGỮ CẢNH (CONTEXT):
        ---------------------
        {context}
        ---------------------
        {hint_block}
        {shape_guideline}
        {partial_guideline}
        {feedback_block}
        CÂU HỎI CỦA NGƯỜI DÙNG:
        {validation_block}

        "{query}"

        HÃY TRẢ LỜI NGAY BÂY GIỜ:
        """

    @classmethod
    def build_tour_plan_system_prompt(cls) -> str:
        critical_facts = cls.critical_system_facts_block()
        few_shot = cls.few_shot_examples_block()
        template = load_prompt("system_tour_plan.txt")
        return template.format(
            critical_facts=critical_facts,
            few_shot=few_shot,
        )

    @classmethod
    def build_tour_plan_user_prompt(
        cls,
        query: str,
        context: str,
        target_location: str,
        days: int,
        nights: int,
        constraints: Dict[str, bool],
        verifier_feedback: Optional[str] = None,
        strict_route_nodes: Optional[List[Dict[str, Any]]] = None,
        dropped_route_points: Optional[List[str]] = None,
        daily_cluster_plan: Optional[List[Dict[str, Any]]] = None,
        lodging_suggestions: Optional[List[Dict[str, Any]]] = None,
        skeleton: str = "",
    ) -> str:
        retry_block = ""
        if verifier_feedback:
            retry_block = (
                "\n\n⚠️ VERIFIER FEEDBACK (MUST FIX ALL):\n"
                f"{verifier_feedback}\n"
            )

        hard_constraints: List[str] = []
        if constraints.get("no_cano"):
            hard_constraints.append("- CẤM cano/tàu cao tốc/di chuyển biển gây sốc.")
        if constraints.get("no_climb"):
            hard_constraints.append("- CẤM leo núi, leo dốc dài, nhiều bậc thang.")
        if constraints.get("low_mobility"):
            hard_constraints.append("- Nhịp độ chậm, nghỉ thường xuyên, ưu tiên điểm bằng phẳng.")
        hard_constraint_block = (
            "\n".join(hard_constraints) if hard_constraints
            else "- Không có ràng buộc cứng bổ sung."
        )

        allowed_route_names = [
            str(n.get("name") or "").strip()
            for n in (strict_route_nodes or [])
            if str(n.get("name") or "").strip()
        ]
        dropped_names = [
            str(name).strip()
            for name in (dropped_route_points or [])
            if str(name).strip()
        ]
        allowed_route_block = (
            "\n".join([f"- {name}" for name in allowed_route_names])
            if allowed_route_names
            else "- (Không có danh sách Route Optimizer)"
        )
        dropped_route_block = (
            "\n".join([f"- {name}" for name in dropped_names])
            if dropped_names
            else "- (Trống)"
        )

        cluster_lines: List[str] = []
        for plan in (daily_cluster_plan or []):
            day = plan.get("day")
            areas = plan.get("areas") or []
            points = plan.get("point_names") or []
            region = plan.get("region_label") or ""
            region_str = f" [Vùng: {region}]" if region else ""
            cluster_lines.append(
                f"- Ngày {day}{region_str}: "
                f"KhuVuc={', '.join(areas) if areas else '(chưa có)'} | "
                f"Điểm={', '.join(points) if points else '(chưa có)'}"
            )
        cluster_block = (
            "\n".join(cluster_lines) if cluster_lines
            else "- Chưa có phân cụm theo ngày."
        )

        lodging_lines: List[str] = []
        for i, lodge in enumerate(lodging_suggestions or [], start=1):
            name = lodge.get("name") or ""
            address = lodge.get("address") or ""
            phone = lodge.get("phone") or ""
            if name:
                line = f"- Đêm {i}: {name}"
                if address:
                    line += f" - {address}"
                if phone:
                    line += f" (ĐT: {phone})"
                lodging_lines.append(line)
        lodging_block = (
            "\n".join(lodging_lines) if lodging_lines
            else "- Chưa có dữ liệu khách sạn cụ thể"
        )

        skeleton_block = ""
        if skeleton:
            skeleton_block = f"""
            <route_skeleton_distribution>
            {skeleton}
            </route_skeleton_distribution>
            """

        return f"""
        <input_data>
            <user_query>{query}</user_query>
            <knowledge_graph_context>
            {context}
            </knowledge_graph_context>
            <allowed_points>
            {allowed_route_block}
            </allowed_points>
            <dropped_points>
            {dropped_route_block}
            </dropped_points>
            <daily_cluster_plan>
            {cluster_block}
            </daily_cluster_plan>
            <lodging_suggestions>
            {lodging_block}
            </lodging_suggestions>
            <hard_constraints>
            {hard_constraint_block}
            </hard_constraints>
            {skeleton_block}
            {retry_block}
        </input_data>

        --------------------------------------------------
        NHIỆM VỤ:
        Dựa trên thông tin được cung cấp trong thẻ <input_data>, hãy tạo lịch trình du lịch {days} ngày {nights} đêm tại {target_location}.
        Viết theo phong cách blogger du lịch: tự nhiên, có cảm xúc, có trải nghiệm cụ thể cho mỗi địa điểm.

        YÊU CẦU BẮT BUỘC:
        1. **Mỗi ngày bắt buộc phải có ít nhất 1-2 điểm tham quan (TouristAttraction)** từ danh sách `<allowed_points>` hoặc `<daily_cluster_plan>`. 
           TUYỆT ĐỐI không được tạo một ngày chỉ bao gồm hoạt động ăn uống (Restaurant/Dish) hoặc nghỉ ngơi/lưu trú (Accommodation) mà không có điểm tham quan nào.
           *Mẹo giải quyết xung đột:* Nếu phân bổ ngày từ `<daily_cluster_plan>` hoặc `<route_skeleton_distribution>` không có điểm tham quan nào cho ngày đó, bạn ĐƯỢC PHÉP tự động chọn bổ sung 1 điểm tham quan nổi bật trong danh sách `<allowed_points>` của khu vực đó để đưa vào buổi sáng hoặc buổi chiều của ngày hôm đó nhằm đảm bảo chất lượng lịch trình.
        2. Format output dạng Markdown với heading và emoji.
        3. Mỗi ngày phải có mô tả TRẢI NGHIỆM chi tiết cho từng điểm (không chỉ ghi tên và giờ).
        4. Giữ nguyên cấu trúc ngày từ `<route_skeleton_distribution>` nếu có (ngoại trừ trường hợp cần bổ sung điểm tham quan như đã nêu ở mục 1).
        5. Nếu thiếu thông tin cần thiết → hãy ghi nhận rõ ràng vào phần "Lưu ý thực tế", TUYỆT ĐỐI không được tự ý bịa thêm dữ liệu.
        6. Đối với tour từ 2 ngày trở lên: phần "Gợi ý nghỉ đêm" phải tách biệt thành một mục riêng ở cuối bài viết, KHÔNG lồng chi tiết (inline) vào lịch trình chi tiết hàng ngày.
        7. Mốc thời gian và tên địa điểm (ví dụ: `- **07:30 - 10:00** 📍 **[Tên điểm]**`) BẮT BUỘC phải viết trên cùng 1 dòng, không được xuống dòng ở giữa chúng.
        8. Trong phần "⚠️ Ràng buộc đã tuân thủ", chỉ ghi nhận các gạch đầu dòng ngắn gọn, thân thiện (ví dụ: "✅ Không dùng cano", "✅ Giữ lịch trình nhịp độ chậm"). 
           TUYỆT ĐỐI KHÔNG ghi các câu lệnh chỉ dẫn kỹ thuật hoặc ghi chú hệ thống (như "riêng, không inline", "XML", v.v.) vào mục này.
        """

    @classmethod
    def build_transfer_route_system_prompt(cls) -> str:
        critical_facts = cls.critical_system_facts_block()
        few_shot = cls.few_shot_examples_block()
        return f"""
                        You are a local travel assistant for practical same-day routing.

                            {critical_facts}

                            {few_shot}

                        Rules:
                        - Reply in Vietnamese.
                        - Use ONLY places present in context.
                        - Do NOT invent or rename locations.
                        - This is NOT a multi-day itinerary.
                        - Focus on one transfer route with one quick stop (about 45-75 minutes) and one lunch stop.
                        - Keep conservative time buffers so user arrives early at airport.
                        """

    @classmethod
    def build_transfer_route_user_prompt(
        cls,
        user_query: str,
        context_block: str,
        deadline: str,
    ) -> str:
        return f"""
                        User query:
                        {user_query}

                        Context:
                        {context_block}

                        Task:
                        Create a practical route for today from current point to airport with these constraints:
                        - 1 quick sightseeing stop around 1 hour
                        - 1 lunch stop with local specialty
                        - arrival at airport before deadline ({deadline})

                        Output structure (markdown):
                        1. TÓM TẮT LỘ TRÌNH
                        2. LỊCH TRÌNH THEO GIỜ
                        3. ĐIỂM GHÉ NHANH (1 GIỜ)
                        4. ĂN TRƯA ĐẶC SẢN
                        5. ĐƯỜNG ĐI GỢI Ý (text route)
                        6. PHƯƠNG ÁN DỰ PHÒNG (nếu trễ giờ)
                        """
