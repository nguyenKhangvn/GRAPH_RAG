from __future__ import annotations
"""Deterministic airport info renderer."""

from ..dto import PipelineRunState


class AirportDispatchMixin:
    """Mixin providing deterministic airport information dispatch."""

    def _dispatch_airport_info(self, state: PipelineRunState, candidates: list) -> str:
        """Deterministic airport info renderer from TravelInfo nodes."""
        # Find TravelInfo nodes with topic='airport' in candidates
        airport_nodes = []
        for c in (candidates or []):
            node_type = ""
            if hasattr(c, "metadata"):
                node_type = str(c.metadata.get("type") or "")
                topic = str(c.metadata.get("topic") or "")
            elif isinstance(c, dict):
                node_type = str(c.get("type") or c.get("labels", [""])[0] if c.get("labels") else "")
                topic = str(c.get("topic") or "")
            else:
                continue

            if "TravelInfo" in node_type and topic == "airport":
                airport_nodes.append(c)

        # Also check grounded_nodes for TravelInfo
        for c in (state.grounded_nodes or []):
            if hasattr(c, "metadata"):
                topic = str(c.metadata.get("topic") or "")
                node_type = str(c.metadata.get("type") or "")
            elif isinstance(c, dict):
                topic = str(c.get("topic") or "")
                node_type = str(c.get("type") or c.get("labels", [""])[0] if c.get("labels") else "")
            else:
                continue

            if "TravelInfo" in node_type and topic == "airport":
                if c not in airport_nodes:
                    airport_nodes.append(c)

        if not airport_nodes:
            return ""  # Fall through to LLM generation

        # Render deterministic response from TravelInfo data
        lines = ["## Thông tin sân bay\n"]
        for node in airport_nodes:
            if hasattr(node, "content"):
                lines.append(node.content)
            elif hasattr(node, "metadata"):
                name = node.metadata.get("name", "")
                desc = node.metadata.get("description", "")
                contact = node.metadata.get("contact", "")
                lines.append(f"**{name}**")
                if desc:
                    lines.append(desc)
                if contact:
                    lines.append(f"Liên hệ: {contact}")
            elif isinstance(node, dict):
                name = node.get("name", "")
                desc = node.get("description", "")
                contact = node.get("contact", "")
                lines.append(f"**{name}**")
                if desc:
                    lines.append(desc)
                if contact:
                    lines.append(f"Liên hệ: {contact}")

        return "\n\n".join(lines)
