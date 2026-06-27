from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from graph_rag.core.state import QuestionShape
from graph_rag.core.thresholds import MIN_CONTEXT_LENGTH
from graph_rag.utils.text import normalize_text

@dataclass
class ValidationIssue:
    code: str
    severity: str  # "info" | "warning" | "error"
    message: str

@dataclass
class ValidationResult:
    passed: bool
    issues: List[ValidationIssue]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnswerContract:
    question_shape: QuestionShape
    target_class: Optional[str] = None
    requested_attributes: List[str] = field(default_factory=list)
    comparison_subjects: List[str] = field(default_factory=list)
    target_entity: Optional[str] = None
    
    # Evidence fields derived from Context
    context_entity_names: List[str] = field(default_factory=list)
    context_has_rating_evidence: bool = False
    context_has_review_evidence: bool = False
    context_sufficient: bool = True
    unsupported_attributes: List[str] = field(default_factory=list)
    allow_apology: bool = False
    clean_context: str = ""

    @classmethod
    def from_query_plan(
        cls,
        query_plan: Any,
        clean_context: str,
        entities: List[Dict[str, Any]] = None,
        context_validation_ok: bool = True,
        seed_nodes: List[Any] = None,
    ) -> AnswerContract:
        """Build contract from query plan and clean context string."""
        context_norm = normalize_text(clean_context)
        
        # 1. Detect rating evidence in context
        has_rating = False
        if "rating" in context_norm or "sao" in context_norm or "star" in context_norm:
            if re.search(r"\b[0-5](\.\d)?\s*(sao|star)\b", context_norm) or re.search(r"rating\s*[:\-\s]\s*[0-5](\.\d)?", context_norm):
                has_rating = True

        # 2. Detect review evidence in context
        has_review = False
        if "review" in context_norm or "danh gia" in context_norm:
            if re.search(r"\b\d+\s*(luot\s*)?(danh\s*gia|review|nhan\s*xet)\b", context_norm) or "review_count" in context_norm:
                has_review = True

        # 3. Extract entity names from context
        entity_names = []
        for match in re.finditer(r"\*\*(.*?)\*\*", clean_context):
            name = match.group(1).strip()
            if name and len(name.split()) >= 2 and not any(kw in name.lower() for kw in ["tóm tắt", "thông tin", "kết quả"]):
                entity_names.append(name)

        # Also add grounded entity names from entities
        for entity in (entities or []):
            if isinstance(entity, dict):
                ent_name = str(entity.get("name") or "").strip()
                if ent_name and ent_name not in entity_names:
                    entity_names.append(ent_name)
        if query_plan.target_entity:
            target = str(query_plan.target_entity).strip()
            if target and target not in entity_names:
                entity_names.append(target)

        # Also add entity names from seed nodes (retrieval candidates)
        for seed in (seed_nodes or []):
            seed_name = ""
            if hasattr(seed, "metadata") and isinstance(seed.metadata, dict):
                seed_name = str(seed.metadata.get("name") or "").strip()
            if not seed_name:
                seed_name = str(getattr(seed, "content", "") or "").strip()
            if seed_name and seed_name not in entity_names:
                entity_names.append(seed_name)

        # 4. Extract comparison subjects
        comparison_subjects = []
        if query_plan.question_shape == QuestionShape.COMPARISON:
            comparison_subjects = list(query_plan.anchors)

        # 5. Find unsupported attributes
        unsupported_attributes = []
        for attr in query_plan.requested_attributes:
            attr_norm = attr.lower()
            attr_keywords = {
                "phone": ["sdt", "so dien thoai", "hotline", "lien he", "phone"],
                "opening_hours": ["gio mo cua", "gio hoat dong", "open", "hour", "khung gio"],
                "price": ["gia", "chi phi", "gia ve", "gia phong", "price", "vnd", "dong"],
                "rating": ["rating", "sao", "star", "danh gia"],
                "review": ["review", "danh gia", "nhan xet", "luot"],
                "address": ["dia chi", "duong", "quan", "address"],
                "name": ["ten", "festival", "le hoi", "su kien", "event"],
                "month": ["thang", "month", "to chuc"],
                "year": ["nam", "year"],
                "activities": ["hoat dong", "trinh dien", "giao luu", "activities"],
                "description": ["mo ta", "gioi thieu", "description", "dip", "khong gian"],
            }
            keywords = attr_keywords.get(attr_norm, [attr_norm])
            found = False
            for kw in keywords:
                if kw in context_norm:
                    found = True
                    break
            if not found:
                unsupported_attributes.append(attr)

        # 6. Context Sufficiency
        context_sufficient = context_validation_ok
        if len(clean_context.strip()) < MIN_CONTEXT_LENGTH:
            context_sufficient = False
        
        allow_apology = not context_sufficient

        return cls(
            question_shape=query_plan.question_shape,
            target_class=query_plan.target_class,
            requested_attributes=list(query_plan.requested_attributes),
            comparison_subjects=comparison_subjects,
            target_entity=query_plan.target_entity,
            context_entity_names=list(set(entity_names)),
            context_has_rating_evidence=has_rating,
            context_has_review_evidence=has_review,
            context_sufficient=context_sufficient,
            unsupported_attributes=unsupported_attributes,
            allow_apology=allow_apology,
            clean_context=clean_context,
        )


class AnswerValidator:
    """Validator to verify generated answers against AnswerContract constraints."""

    COMMON_APOLOGY_KEYWORDS = [
        "xin loi", "khong tim thay", "khong co thong tin", "chua co thong tin",
        "chua cap nhat", "khong duoc cung cap", "khong the tra loi", "khong du thong tin"
    ]

    COMMON_STOP_WORDS = {
        "ban co", "tuy nhien", "ngoai ra", "doi voi", "hien tai", "trong do", "he thong",
        "du lich", "xin loi", "cam on", "chung toi", "quy khach", "duoi day", "theo du lieu"
    }

    def validate(self, answer: str, contract: AnswerContract) -> ValidationResult:
        issues: List[ValidationIssue] = []
        answer_norm = normalize_text(answer)

        # 1. Rating/Review Hallucination Guard
        if not contract.context_has_rating_evidence:
            # Check for rating numeric patterns
            has_numeric_rating = (
                re.search(r"\b([0-5]\.\d|[0-5])\s*(\/\s*5)?\s*(sao|star)\b", answer_norm) or 
                re.search(r"\brating\s*(la|dat)?\s*([0-5]\.\d|[0-5])\b", answer_norm)
            )
            # Check for qualitative rating statements
            qualitative_rating_keywords = [
                "danh gia cao", "rating tot", "danh gia tot", "rating cao", 
                "danh gia tich cuc", "rating tuyet voi", "nhieu sao"
            ]
            has_qualitative_rating = any(kw in answer_norm for kw in qualitative_rating_keywords)
            
            if has_numeric_rating or has_qualitative_rating:
                issues.append(
                    ValidationIssue(
                        code="rating_hallucination",
                        severity="error",
                        message="Câu trả lời bịa đặt điểm đánh giá (rating) hoặc mức độ đánh giá cao khi dữ liệu context không có.",
                    )
                )

        if not contract.context_has_review_evidence:
            has_numeric_review = re.search(r"\b\d+\s*(luot\s*)?(danh\s*gia|review|nhan\s*xet)\b", answer_norm)
            qualitative_review_keywords = ["nhieu review", "nhieu luot review", "nhieu nhan xet", "nhieu danh gia"]
            has_qualitative_review = any(kw in answer_norm for kw in qualitative_review_keywords)
            
            if has_numeric_review or has_qualitative_review:
                # Distinguish from qualitative rating if already flagged, but review counts must be flagged
                issues.append(
                    ValidationIssue(
                        code="review_hallucination",
                        severity="error",
                        message="Câu trả lời bịa đặt số lượng đánh giá/review khi dữ liệu context không có.",
                    )
                )

        # 3. Unsupported Attribute Guard
        for attr in contract.unsupported_attributes:
            if attr == "phone":
                # Check VN phone format (e.g. 0912345678, +84..., 024-...)
                if re.search(r"\b(0|84|\+84)[35789]\d{8}\b", answer_norm) or re.search(r"\b\d{3,4}[\.\s-]\d{3,4}[\.\s-]\d{3,4}\b", answer_norm):
                    issues.append(
                        ValidationIssue(
                            code="phone_hallucination",
                            severity="error",
                            message="Câu trả lời bịa số điện thoại liên hệ dù dữ liệu context không có.",
                        )
                    )
            elif attr == "opening_hours":
                # Check opening hour pattern (e.g. 8h-22h, 08:00, mở cửa)
                if re.search(r"\b\d{1,2}h(\d{2})?\b", answer_norm) or re.search(r"\b\d{1,2}:\d{2}\b", answer_norm):
                    # Only flag if it asserts a specific hour block
                    if any(kw in answer_norm for kw in ["mo cua", "dong cua", "hoat dong tu"]):
                        issues.append(
                            ValidationIssue(
                                code="opening_hours_hallucination",
                                severity="error",
                                message="Câu trả lời bịa giờ mở/đóng cửa dù dữ liệu context không có.",
                            )
                        )
            elif attr == "price":
                # Check price patterns (e.g. 100.000đ, 50k, 200k,...)
                if re.search(r"\b\d+([.,]\d+)?\s*(dong|d|vnd|k|nghin|trieu)\b", answer_norm):
                    issues.append(
                        ValidationIssue(
                            code="price_hallucination",
                            severity="error",
                            message="Câu trả lời tự bịa mức giá/vé/phòng dù dữ liệu context không có.",
                        )
                    )

        # 4. Entity Grounding Check
        # Extract capitalized entities in answer (ignoring first word of sentences)
        # We can find all sequences of words starting with capital letters
        capital_sequences = re.findall(r"\b[A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶEÈÉẺẼẸÊỀẾỂỄỆIÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸY][a-zàáảãạâầấẩẫậăằắẳẵặeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹy]*+(?:\s+[A-ZÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶEÈÉẺẼẸÊỀẾỂỄỆIÌÍỈĨỊOÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢUÙÚỦŨỤƯỪỨỬỮỰYỲÝỶỸY][a-zàáảãạâầấẩẫậăằắẳẵặeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹy]*)*\b", answer)
        
        context_entities_norm = [normalize_text(name) for name in contract.context_entity_names]
        clean_context_norm = normalize_text(contract.clean_context) if contract.clean_context else ""
        
        ungrounded_entities = []
        for seq in capital_sequences:
            seq_norm = normalize_text(seq)
            if len(seq_norm.split()) < 2:
                continue
            if seq_norm in self.COMMON_STOP_WORDS:
                continue
            # Skip general location/province keywords if they are not in the query
            if seq_norm in ["gia lai", "binh dinh", "quy nhon", "viet nam"]:
                continue
            # Skip common ethnic/cultural group names that may appear in descriptions
            if seq_norm in ["ba na", "bana", "e de", "ede", "giao rai", "jarai", "co ho", "koho",
                            "tay nguyen", "tay nguyen", "dong nam a", "khu vuc"]:
                continue

            # Check if this sequence is grounded using multiple strategies
            is_grounded = False
            seq_tokens = set(seq_norm.split())

            # Strategy 0: Direct substring check in context to avoid false positives for contextual terms
            if clean_context_norm and seq_norm in clean_context_norm:
                is_grounded = True

            for ctx_ent in context_entities_norm:
                # Strategy 1: Substring containment (original)
                if seq_norm in ctx_ent or ctx_ent in seq_norm:
                    is_grounded = True
                    break

                # Strategy 2: Token overlap — if 80%+ of answer tokens appear in context entity
                ctx_tokens = set(ctx_ent.split())
                if seq_tokens and ctx_tokens:
                    overlap = seq_tokens & ctx_tokens
                    if len(overlap) >= max(1, int(len(seq_tokens) * 0.8)):
                        is_grounded = True
                        break

                # Strategy 3: Last meaningful segment match (e.g., "Mơ Hra" matches "Làng Du lịch cộng đồng Mơ Hra")
                # Extract last 2-3 words from context entity and check if answer seq matches
                ctx_words = ctx_ent.split()
                for seg_len in [2, 3]:
                    if len(ctx_words) >= seg_len:
                        last_segment = " ".join(ctx_words[-seg_len:])
                        if seq_norm == last_segment or last_segment == seq_norm:
                            is_grounded = True
                            break
                if is_grounded:
                    break

            if not is_grounded and seq_norm not in ungrounded_entities:
                ungrounded_entities.append(seq)
                
        for ent in ungrounded_entities:
            issues.append(
                ValidationIssue(
                    code="ungrounded_entity",
                    severity="warning",
                    message=f"Câu trả lời đề cập đến thực thể '{ent}' không xuất hiện trong dữ liệu context.",
                )
            )

        # Escalate to error if many ungrounded entities (mass hallucination)
        if len(ungrounded_entities) >= 3:
            issues.append(
                ValidationIssue(
                    code="mass_ungrounded_entities",
                    severity="error",
                    message=f"Câu trả lời có {len(ungrounded_entities)} thực thể không grounded trong context — nghi ngờ hallucination hàng loạt.",
                )
            )

        # 5. Shape Validation
        # 5a. COMPARISON
        if contract.question_shape == QuestionShape.COMPARISON and contract.comparison_subjects:
            for sub in contract.comparison_subjects:
                sub_norm = normalize_text(sub)
                if sub_norm not in answer_norm:
                    issues.append(
                        ValidationIssue(
                            code="missing_comparison_subject",
                            severity="error",
                            message=f"Câu hỏi so sánh thiếu đối tượng so sánh '{sub}' trong câu trả lời.",
                        )
                    )

        # 5b. ITINERARY
        if contract.question_shape == QuestionShape.ITINERARY:
            # Check for schedule indicators
            itinerary_markers = ["ngay", "buoi", "sang", "chieu", "toi", "lich trinh", "lo trinh", "diem dung"]
            matches = [m for m in itinerary_markers if m in answer_norm]
            if len(matches) < 2:
                issues.append(
                    ValidationIssue(
                        code="itinerary_missing_structure",
                        severity="error",
                        message="Câu trả lời về lịch trình (itinerary) thiếu các mốc thời gian hoặc cấu trúc lịch trình phù hợp (ngày/buổi/sáng/chiều/tối).",
                    )
                )

        passed = all(issue.severity != "error" for issue in issues)
        return ValidationResult(passed=passed, issues=issues)
