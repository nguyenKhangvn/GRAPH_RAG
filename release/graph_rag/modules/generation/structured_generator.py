from __future__ import annotations

from typing import Any, Dict


class StructuredAnswerGenerator:
    def __init__(self, llm_service: Any):
        self.llm = llm_service

    def generate(
        self,
        question: str,
        structured_context: str,
        intent_data: Dict[str, Any],
        validation: Dict[str, Any],
        extra_instruction: str = "",
    ) -> str:
        context_state = (validation or {}).get("context_state")
        if context_state == "NO_CANDIDATE":
            return self._no_candidate_answer(intent_data, validation)

        system_prompt = self._system_prompt(
            intent_data,
            validation,
            extra_instruction=extra_instruction,
        )
        user_prompt = self._user_prompt(question, structured_context)
        raw = self.llm.generate_text(system_prompt, user_prompt)
        return str(raw or "").strip()

    def _system_prompt(
        self,
        intent_data: Dict[str, Any],
        validation: Dict[str, Any],
        extra_instruction: str = "",
    ) -> str:
        intent_mode = intent_data.get("intent_mode") or "single_anchor"
        context_state = (validation or {}).get("context_state") or ""

        base = (
            "DIRECT NEAR RULE: If a DIRECT NEAR EVIDENCE section is present, only those listed pairs may be described as near/lan can. Do not infer NEAR through a shared third place.\n"
            "Bạn là trợ lý thông tin du lịch đáng tin cậy. Chỉ trả lời dựa trên STRUCTURED CONTEXT.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Chỉ dùng thông tin có trong context; không bịa khoảng cách, thời gian di chuyển, tiện ích, giá cả, giờ mở cửa, hoặc quan hệ gần nếu context không nêu.\n"
            "2. Chỉ được nói A gần B khi context có quan hệ NEAR trực tiếp giữa A và B. Không tự suy diễn gần nhau chỉ vì hai địa điểm cùng xuất hiện trong context.\n"
            "3. Nếu context có dữ liệu, phải dùng dữ liệu đó; không trả lời chung chung.\n"
            "4. Không lặp lại các nhãn kỹ thuật/nội bộ như 'Dữ liệu có', 'Dữ liệu thiếu', 'Dữ liệu cho thực thể', 'Type:', 'Anchor', 'PARTIAL', 'Context', 'Schema'. TUYỆT ĐỐI KHÔNG đưa các nhãn quan hệ viết hoa (NEAR, LOCATED_IN, BELONGS_TO, HAS, HELD_AT, INCLUDES, OFFERS) hay ký hiệu lập trình (->, [, ]) lên câu trả lời cuối cùng, hãy viết tự nhiên cho người dùng.\n"
            "5. Nếu câu hỏi hỏi về một thuộc tính cụ thể (ví dụ: website, số điện thoại, giá vé, giờ mở cửa...) mà context không có, bạn phải trả lời rõ ràng dữ liệu hệ thống chưa cập nhật thông tin này, không tìm cách lấp liếm bằng các thông tin khác.\n"
            "6. Nếu thiếu dữ liệu quan trọng, hãy nêu ngắn gọn trong phần kết luận, không biến thành đoạn mở đầu dài dòng.\n"
            "7. FORMAT TRẢ LỜI (BẮT BUỘC): Sử dụng markdown. Khi liệt kê nhiều mục, BẮT BUỘC phải dùng danh sách xuống dòng của Markdown (dùng ký tự gạch đầu dòng '-' hoặc '*', ví dụ: '- Tên mục: mô tả'), mỗi mục nằm trên một dòng riêng biệt. TUYỆT ĐỐI KHÔNG viết các mục liệt kê dính liền trên cùng một dòng bằng các ký tự dấu chấm hoặc bullet như '•' mà không xuống dòng. Phân nhóm rõ ràng (📍địa điểm, 🏨lưu trú, 🍜ẩm thực, 🎉sự kiện, 🗺️lịch trình). KHÔNG viết câu trả lời dài thành 1 đoạn liền. Nếu nội dung trên 3 câu, phải chia thành đoạn hoặc danh sách."
        )

        templates = {
            "comparison": (
                base
                + "\n\nCOMPARISON CONTRACT:\n"
                "- Trả lời lần lượt từng thực thể được hỏi.\n"
                "- Nếu câu hỏi hỏi điểm chung, chỉ liệt kê điểm chung có trong context cho tất cả thực thể liên quan.\n"
                "- Nếu câu hỏi hỏi lựa chọn phù hợp hơn, chọn một phương án chỉ khi có bằng chứng (evidence) trực tiếp trong context; nếu không, nêu rõ không đủ dữ liệu.\n"
                "- Kết luận phải dựa trên bằng chứng trực tiếp, không dựa trên phân loại tự suy diễn."
            ),
            "tour_plan": (
                base
                + "\n\nTOUR PLAN CONTRACT:\n"
                "- Gợi ý nơi lưu trú và điểm tham quan/hoạt động chỉ khi chúng xuất hiện trong context.\n"
                "- Chỉ nói hai điểm gần nhau/cùng khu vực khi context có quan hệ NEAR/LOCATED_IN hoặc địa chỉ rõ ràng hỗ trợ.\n"
                "- Không tự chèn thêm tour, nhà cung cấp, resort, khung giờ, chi phí nếu không có trong context.\n"
                "- KHÔNG hoạt động ngoài trời (tham quan, đi biển, trekking) vào giờ trưa nắng (11:30-14:00). "
                "Buổi trưa nên ăn uống hoặc nghỉ ngơi. "
                "Điểm biển/đảo nên buổi sáng sớm hoặc buổi chiều (sau 15:00). "
                "Phải có ít nhất 1 bữa ăn (trưa hoặc tối) trong lịch trình.\n"
                "- FORMAT BẮT BUỘC: Dùng markdown. In đậm tên tour. Liệt kê điểm đến bằng danh sách xuống dòng của Markdown (dùng ký tự gạch đầu dòng '-' hoặc '*', ví dụ: '- 📍 Tên điểm đến: mô tả'), mỗi điểm đến nằm trên một dòng riêng biệt. Sử dụng emoji phù hợp (🏝️đảo, 📍địa điểm, 🏨lưu trú, 🍜ẩm thực). Nếu có giá tour thì ghi riêng 1 dòng. Nếu thiếu thông tin hoạt động thì ghi rõ ở cuối, không nhồi vào giữa."
            ),
            "constraint_matching": (
                base
                + "\n\nCONSTRAINT MATCHING CONTRACT:\n"
                "- Kiểm tra từng điều kiện theo bằng chứng.\n"
                "- Chỉ chọn ứng viên đáp ứng đủ điều kiện; không chọn ứng viên gần đúng."
            ),
            "multi_entity_nearby": (
                base
                + "\n\nNEARBY CONTRACT:\n"
                "- Liệt kê điểm lân cận riêng cho từng anchor.\n"
                "- Nếu cần điểm chung, chỉ lấy giao của các danh sách NEAR trong context."
            ),
            "dish_to_restaurant": (
                base
                + "\n\nDISH TO RESTAURANT CONTRACT:\n"
                "- Nêu rõ món ăn, nhà hàng phục vụ và địa chỉ/liên hệ nếu context có.\n"
                "- Không tự suy ra nhà hàng phục vụ món nếu context không có quan hệ HAS trực tiếp."
            ),
            "single_anchor": base + "\n\nSINGLE ANCHOR CONTRACT: Tóm tắt đúng thông tin của anchor và phân loại (category)/quan hệ có trong context.",
            "negative": base + "\n\nNEGATIVE CONTRACT: Từ chối rõ ràng nếu dữ liệu không đáp ứng.",
        }

        prompt = templates.get(intent_mode, base)

        if context_state == "PARTIAL":
            anchors = intent_data.get("anchors") or []
            missing_list = validation.get("missing") or []
            present_anchors = []
            missing_anchors = []
            for anchor in anchors:
                if f"FACTS:{anchor}" in missing_list or any(
                    f":{anchor}" in item for item in missing_list if str(item).startswith("FACTS:")
                ):
                    missing_anchors.append(anchor)
                else:
                    present_anchors.append(anchor)
            present_str = ", ".join(present_anchors) if present_anchors else "một số đối tượng"
            missing_str = ", ".join(missing_anchors) if missing_anchors else "một số thông tin"
            prompt += (
                "\n\nPARTIAL CONTEXT: Hãy trả lời tự nhiên dựa trên phần có dữ liệu "
                f"({present_str}). Chỉ nêu ngắn gọn phần thiếu ({missing_str}) khi nó ảnh hưởng trực tiếp đến kết luận. "
                "Tuyệt đối không mở đầu bằng mẫu 'Dữ liệu có... Dữ liệu thiếu...'."
            )

        if extra_instruction:
            prompt += "\n" + str(extra_instruction).strip()
        return prompt

    def _user_prompt(self, question: str, structured_context: str) -> str:
        return (
            "STRUCTURED CONTEXT - MUST USE:\n"
            f"{structured_context}\n\n"
            "QUESTION:\n"
            f"{question}\n\n"
            "Trả lời bằng tiếng Việt có dấu, tự nhiên, ngắn gọn, bao phủ các anchor/constraint bắt buộc."
        )

    def _no_candidate_answer(self, intent_data: Dict[str, Any], validation: Dict[str, Any]) -> str:
        matrix = validation.get("candidate_matrix") or []
        if not matrix:
            return "Không có ứng viên đáp ứng đủ điều kiện trong dữ liệu hiện có."
        lines = ["Không có ứng viên đáp ứng đủ tất cả điều kiện trong dữ liệu hiện có. Chi tiết:"]
        for row in matrix:
            candidate = row.get("candidate") or "Ứng viên"
            failures = [
                key.replace("has_", "").upper()
                for key, value in row.items()
                if key.startswith("has_") and not value
            ]
            if failures:
                lines.append(f"- {candidate}: thiếu bằng chứng cho {', '.join(failures)}.")
            else:
                lines.append(f"- {candidate}: thiếu dữ liệu xác minh.")
        lines.append("Vì vậy hệ thống không chọn một ứng viên gần đúng.")
        return "\n".join(lines)
