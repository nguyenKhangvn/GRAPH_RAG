from __future__ import annotations

from typing import Any, Dict, List


class StructuredContextBuilder:
    def build(
        self,
        intent_data: Dict[str, Any],
        grouped_facts: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> str:
        sections: List[str] = []
        intent_mode = intent_data.get("intent_mode") or "single_anchor"
        sections.append("## INTENT\n" + str(intent_mode))

        anchors = intent_data.get("anchors") or []
        for idx, anchor in enumerate(anchors, start=1):
            facts = grouped_facts.get(anchor, {}) if isinstance(grouped_facts, dict) else {}
            entity = facts.get("entity") or {"name": anchor}
            relations = facts.get("relations") or {}
            attributes = facts.get("attributes") or {}

            header = f"## ANCHOR {idx}: {entity.get('name') or anchor}"
            lines = [header]
            labels = entity.get("labels") or []
            if labels:
                lines.append(f"- Type: {', '.join(str(label) for label in labels)}")
            address = entity.get("address") or attributes.get("dia_chi") or attributes.get("dia chi")
            if address:
                lines.append(f"- Address: {address}")

            description = entity.get("description") or attributes.get("description") or attributes.get("mo_ta") or attributes.get("mo ta")
            if description:
                desc_text = str(description).strip()
                if len(desc_text) > 500:
                    desc_text = desc_text[:500] + "..."
                lines.append(f"- Description: {desc_text}")

            for rel_type, rel_items in relations.items():
                if not rel_items:
                    continue
                rel_title = rel_type.replace("_", " ")
                rel_list = ", ".join(rel_items[:10])
                lines.append(f"- {rel_title}: {rel_list}")

            sections.append("\n".join(lines))

        validation_lines = ["## VALIDATION"]
        if validation:
            validation_lines.append(f"- Context state: {validation.get('context_state')}")
            if validation.get("missing"):
                missing = ", ".join(str(item) for item in validation.get("missing"))
                validation_lines.append(f"- Missing: {missing}")
            if validation.get("warnings"):
                warnings = ", ".join(str(item) for item in validation.get("warnings"))
                validation_lines.append(f"- Warnings: {warnings}")
        sections.append("\n".join(validation_lines))

        contract_lines = ["## ANSWER CONTRACT"]
        contract = intent_data.get("answer_contract") or {}
        for key, value in contract.items():
            contract_lines.append(f"- {key}: {value}")
        sections.append("\n".join(contract_lines))

        return "\n\n".join(sections)
