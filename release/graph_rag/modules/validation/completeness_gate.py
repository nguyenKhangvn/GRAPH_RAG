from typing import Any, Dict, List, Tuple
from graph_rag.utils.text import normalize_text


class CompletenessGate:
    def validate(self, intent_data: Dict[str, Any], grouped_facts: Dict[str, Any]) -> Dict[str, Any]:
        intent_mode = str(intent_data.get("intent_mode") or "single_anchor")
        anchors = [str(anchor or "").strip() for anchor in (intent_data.get("anchors") or []) if str(anchor or "").strip()]
        constraints = intent_data.get("constraints") or {}
        required_conditions = constraints.get("required_conditions") or []
        required_relations = constraints.get("relations") or []

        missing: List[str] = []
        candidate_matrix: List[Dict[str, Any]] = []
        warnings: List[str] = []

        anchor_fact_flags: List[Tuple[str, bool, Dict[str, Any]]] = []
        for anchor in anchors:
            facts = grouped_facts.get(anchor, {}) if isinstance(grouped_facts, dict) else {}
            relations = facts.get("relations") or {}
            attributes = facts.get("attributes") or {}
            raw_facts = facts.get("raw_facts") or []
            has_any = bool(raw_facts or attributes or any(relations.values()))
            anchor_fact_flags.append((anchor, has_any, relations))

        if intent_mode == "comparison":
            missing_anchors = [anchor for anchor, has_facts, _ in anchor_fact_flags if not has_facts]
            if not anchors:
                context_state = "INSUFFICIENT_EVIDENCE"
                missing.append("ANCHORS")
            elif missing_anchors:
                context_state = "PARTIAL" if len(missing_anchors) < len(anchors) else "INSUFFICIENT_EVIDENCE"
                missing.extend(f"FACTS:{anchor}" for anchor in missing_anchors)
            else:
                required = required_conditions or required_relations
                required_rel_types = self._normalize_required_relations(required)
                missing_relations = self._missing_required_relations(anchor_fact_flags, required_rel_types)
                if missing_relations:
                    context_state = "PARTIAL"
                    missing.extend(missing_relations)
                else:
                    context_state = "COMPLETE"
        elif intent_mode == "constraint_matching":
            required = required_conditions or required_relations
            required_rel_types = self._normalize_required_relations(required)
            for anchor, has_facts, relations in anchor_fact_flags:
                row = {"candidate": anchor, "has_facts": has_facts}
                valid = has_facts
                for rel in required_rel_types:
                    has_rel = bool(relations.get(rel))
                    row[f"has_{rel.lower()}"] = has_rel
                    if not has_rel:
                        valid = False
                row["valid"] = valid
                candidate_matrix.append(row)
            valid_candidates = [row for row in candidate_matrix if row.get("valid")]
            if not valid_candidates:
                context_state = "NO_CANDIDATE" if candidate_matrix else "INSUFFICIENT_EVIDENCE"
            else:
                context_state = "COMPLETE"
        elif intent_mode == "negative":
            has_any = any(has_facts for _, has_facts, _ in anchor_fact_flags)
            context_state = "COMPLETE" if has_any else "NO_CANDIDATE"
            if not has_any:
                warnings.append("negative_intent_no_evidence")
        else:
            has_any = any(has_facts for _, has_facts, _ in anchor_fact_flags)
            context_state = "COMPLETE" if has_any else "INSUFFICIENT_EVIDENCE"

        return {
            "context_state": context_state,
            "missing": missing,
            "candidate_matrix": candidate_matrix,
            "warnings": warnings,
        }

    def _missing_required_relations(
        self,
        anchor_fact_flags: List[Tuple[str, bool, Dict[str, Any]]],
        required_rel_types: List[str],
    ) -> List[str]:
        if not required_rel_types:
            return []
        missing: List[str] = []
        for anchor, has_facts, relations in anchor_fact_flags:
            if not has_facts:
                continue
            for rel in required_rel_types:
                if not relations.get(rel):
                    missing.append(f"{rel}:{anchor}")
        return missing

    def _normalize_required_relations(self, required_conditions: List[str]) -> List[str]:
        rels: List[str] = []
        explicit_relations = {"NEAR", "LOCATED_IN", "BELONGS_TO", "HAS", "INCLUDES", "OFFERS"}
        for cond in required_conditions or []:
            raw = str(cond or "").strip()
            text = normalize_text(raw, strip_punct=True)
            if raw.upper() in explicit_relations:
                rels.append(raw.upper())
            elif any(token in text for token in ["near", "gan", "xung quanh", "lan can", "diem gan"]):
                rels.append("NEAR")
            elif any(token in text for token in ["located", "dia chi", "khu vuc", "o dau", "nam tai"]):
                rels.append("LOCATED_IN")
            elif any(token in text for token in ["belongs", "loai", "phan loai", "thuoc loai"]):
                rels.append("BELONGS_TO")
            elif any(token in text for token in ["mon", "dish", "phuc vu", "has"]):
                rels.append("HAS")
        return list(dict.fromkeys(rels))

