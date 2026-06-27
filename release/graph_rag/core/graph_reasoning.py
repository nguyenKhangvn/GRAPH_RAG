"""GraphReasoningExecutor — deterministic multi-hop chain answerer.

Given a chain like:
  [Accommodation -NEAR-> TouristAttraction -HAS-> Dish]

The executor builds a single Cypher query that walks the chain
and returns the answer-set nodes with evidence from each hop.

This module is intentionally self-contained so it can be tested
independently of the full RAG pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from neo4j.exceptions import ClientError as Neo4jClientError, ServiceUnavailable

logger = logging.getLogger(__name__)


# ────────────────────── Data classes ──────────────────────


@dataclass
class ChainHop:
    """One hop in a reasoning chain."""
    from_label: str
    rel: str
    to_label: str


@dataclass
class ChainResult:
    """Result of executing a chain query."""
    answer_nodes: List[Dict[str, Any]] = field(default_factory=list)
    evidence_paths: List[str] = field(default_factory=list)
    cypher_query: str = ""
    hit_count: int = 0
    fallback: bool = False
    error: Optional[str] = None


# ────────────────────── Executor ──────────────────────


class GraphReasoningExecutor:
    """Execute multi-hop chain queries against Neo4j.

    Usage:
        executor = GraphReasoningExecutor(neo4j_driver)
        result = executor.execute_chain(
            answer_set_label="Accommodation",
            chain=[
                {"from": "Accommodation", "rel": "NEAR", "to": "TouristAttraction"},
                {"from": "Restaurant", "rel": "HAS", "to": "Dish"},
            ],
            location_scope="Quy Nhơn",
        )
    """

    # Relationship direction map:
    #   "undirected" → (a)-[:REL]-(b)
    #   "outbound"   → (a)-[:REL]->(b)
    #   "inbound"    → (a)<-[:REL]-(b)
    REL_DIRECTION = {
        "NEAR": "undirected",
        "LOCATED_IN": "outbound",
        "BELONGS_TO": "outbound",
        "HAS": "outbound",
        "OFFERS": "outbound",
        "INCLUDES": "outbound",
        "HELD_AT": "outbound",
        "Guide_for": "outbound",
        "SUPERSEDED_BY": "outbound",
    }

    def __init__(self, neo4j_driver):
        self._driver = neo4j_driver

    def execute_chain(
        self,
        answer_set_label: str,
        chain: List[Dict[str, str]],
        location_scope: str = "",
        limit: int = 7,
    ) -> ChainResult:
        """Build and execute a Cypher query for the given chain."""
        if not chain or len(chain) < 2:
            return ChainResult(error="Chain must have at least 2 hops", fallback=True)

        hops = [ChainHop(h["from"], h["rel"], h["to"]) for h in chain]
        cypher, params = self._build_cypher(answer_set_label, hops, location_scope, limit)

        logger.info("[GraphReasoning] Cypher:\n%s\nParams: %s", cypher, params)

        try:
            with self._driver.session() as session:
                records = list(session.run(cypher, **params))
        except (Neo4jClientError, ServiceUnavailable) as exc:
            logger.error("[GraphReasoning] Cypher execution failed: %s", exc)
            return ChainResult(error=str(exc), fallback=True, cypher_query=cypher)

        # Fallback: if no results and chain has HAS->Dish, try expanding
        # through intermediate Restaurant node:
        #   Accommodation-[:NEAR]-TouristAttraction<-[:NEAR]-Restaurant-[:HAS]->Dish
        if not records:
            has_dish_hop = any(
                h.to_label == "Dish" and h.rel == "HAS"
                for h in hops
            )
            if has_dish_hop:
                logger.info("[GraphReasoning] No results. Trying intermediate Restaurant fallback.")
                expanded = []
                for h in hops:
                    if h.to_label == "Dish" and h.rel == "HAS":
                        expanded.append(ChainHop(h.from_label, "NEAR", "Restaurant"))
                        expanded.append(ChainHop("Restaurant", "HAS", "Dish"))
                    else:
                        expanded.append(h)
                fb_cypher, fb_params = self._build_cypher(answer_set_label, expanded, location_scope, limit)
                logger.info("[GraphReasoning] Fallback Cypher:\n%s", fb_cypher)
                try:
                    with self._driver.session() as session:
                        records = list(session.run(fb_cypher, **fb_params))
                    if records:
                        cypher, params = fb_cypher, fb_params
                        hops = expanded
                except (Neo4jClientError, ServiceUnavailable) as exc:
                    logger.error("[GraphReasoning] Fallback also failed: %s", exc)

        if not records:
            return ChainResult(
                cypher_query=cypher,
                hit_count=0,
                fallback=True,
                error="no_results",
            )

        answer_nodes, evidence_paths = self._parse_records(records, hops)

        return ChainResult(
            answer_nodes=answer_nodes,
            evidence_paths=evidence_paths,
            cypher_query=cypher,
            hit_count=len(answer_nodes),
        )

    # ─── Cypher builder ───────────────────────────────────────────

    def _build_cypher(
        self,
        answer_set_label: str,
        hops: List[ChainHop],
        location_scope: str,
        limit: int,
    ) -> tuple[str, Dict[str, Any]]:
        """Build a Cypher MATCH chain.

        All hops use MATCH (not WHERE pattern expressions) to avoid
        "PatternExpressions are not allowed to introduce new variables" errors.

        Example for [Accommodation-NEAR->Restaurant-HAS->Dish]:
            MATCH (n0:Accommodation)-[:NEAR]-(n1:Restaurant)
            MATCH (n1)-[:HAS]->(n2:Dish)
            WHERE n0.address CONTAINS $loc ...
            RETURN DISTINCT n0, n1, n2
            LIMIT $limit
        """
        params: Dict[str, Any] = {"limit": limit}
        match_lines: List[str] = []
        where_parts: List[str] = []
        return_vars: List[str] = ["n0"]

        # First hop
        first = hops[0]
        rel_pat = self._rel_pattern("n0", first.rel, "n1", first.to_label)
        match_lines.append(f"MATCH (n0:{first.from_label}){rel_pat}")
        return_vars.append("n1")

        # All subsequent hops: MATCH (not WHERE pattern expression)
        for i, hop in enumerate(hops[1:], start=1):
            prev_var = f"n{i}"
            next_var = f"n{i + 1}"
            rel_pat = self._rel_pattern(prev_var, hop.rel, next_var, hop.to_label)
            match_lines.append(f"MATCH ({prev_var}){rel_pat}")
            return_vars.append(next_var)

        # Location filter — apply to location-aware nodes only (skip Dish/Event).
        # n0 is always the answer-set node; subsequent nodes come from hops.
        if location_scope:
            loc_norm = location_scope.strip()
            params["loc"] = loc_norm.lower()
            # Build label list: n0 = first hop's from_label, n{i} = hops[i-1].to_label
            node_labels = [hops[0].from_label] + [h.to_label for h in hops]
            skip_labels = {"Dish", "Specialty", "Event"}
            for i, label in enumerate(node_labels):
                if label in skip_labels:
                    continue
                var = f"n{i}"
                loc_var = f"loc_{var}"
                where_parts.append(
                    f"(toLower({var}.address) CONTAINS $loc "
                    f"OR EXISTS {{ MATCH ({var})-[:LOCATED_IN]->({loc_var}:Location) "
                    f"WHERE toLower({loc_var}.name) CONTAINS $loc }})"
                )

        # Assemble
        cypher_lines = match_lines.copy()
        if where_parts:
            cypher_lines.append("WHERE " + "\n  AND ".join(where_parts))
        # Ranking: prefer nodes with specific address, then rating
        cypher_lines.append(
            f"RETURN DISTINCT {', '.join(return_vars)}, "
            f"CASE WHEN n0.address IS NOT NULL THEN 0 ELSE 1 END AS _addr_rank, "
            f"coalesce(n0.star_rating, n0.rating, 0) AS _rating "
            f"ORDER BY _addr_rank ASC, _rating DESC"
        )
        cypher_lines.append(f"LIMIT $limit")
        return "\n".join(cypher_lines), params

    def _rel_pattern(self, src_var: str, rel: str, dst_var: str, dst_label: str) -> str:
        """Build a relationship pattern fragment with correct direction."""
        dst = f"({dst_var}:{dst_label})" if dst_var else f"(:{dst_label})"
        cypher_rel = rel
        direction = self.REL_DIRECTION.get(rel, "undirected")
        if direction == "inbound":
            return f"<-[:{cypher_rel}]-{dst}"
        elif direction == "outbound":
            return f"-[:{cypher_rel}]->{dst}"
        else:
            return f"-[:{cypher_rel}]-{dst}"

    # ─── Result parsing ───────────────────────────────────────────

    def _parse_records(
        self,
        records: list,
        hops: List[ChainHop],
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Parse Neo4j records into answer nodes and evidence paths."""
        answer_nodes: List[Dict[str, Any]] = []
        evidence_paths: List[str] = []
        seen_ids: set = set()

        for record in records:
            n0 = record.get("n0")
            if n0 is None:
                continue
            node_id = n0.element_id if hasattr(n0, "element_id") else str(n0.get("name", id(n0)))
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)

            # Build answer node dict
            props = dict(n0) if hasattr(n0, "__iter__") else {}
            answer_nodes.append({
                "name": props.get("name", ""),
                "label": hops[0].from_label,
                "properties": props,
            })

            # Build evidence path string
            path_parts = [f"[{hops[0].from_label}] {props.get('name', '?')}"]
            for i, hop in enumerate(hops):
                ni = record.get(f"n{i + 1}")
                if ni is not None:
                    ni_props = dict(ni) if hasattr(ni, "__iter__") else {}
                    path_parts.append(
                        f"--[{hop.rel}]--> [{hop.to_label}] {ni_props.get('name', '?')}"
                    )
            evidence_paths.append(" ".join(path_parts))

        return answer_nodes, evidence_paths


# ────────────────────── Answer Renderer ──────────────────────


def render_chain_answer(
    query: str,
    result: ChainResult,
    answer_set_label: str,
    location_scope: str = "",
) -> str:
    """Render a user-friendly answer from chain results.

    Shows top 5 items with per-item evidence, then mentions remaining count.
    """
    if result.error and not result.answer_nodes:
        return (
            f"Xin lỗi, tôi không tìm thấy {answer_set_label.lower()} nào "
            f"phù hợp với tiêu chí của bạn"
            f"{' tại ' + location_scope if location_scope else ''}."
        )

    label_vn = _label_to_vietnamese(answer_set_label)
    scope_text = f" tại {location_scope}" if location_scope else ""

    total = result.hit_count
    show_count = min(5, total)
    remaining = total - show_count

    if remaining > 0:
        lines = [f"Dựa trên dữ liệu, có **{total}** {label_vn}{scope_text} phù hợp. Hiển thị {show_count} kết quả tốt nhất:\n"]
    else:
        lines = [f"Dựa trên dữ liệu, có **{total}** {label_vn}{scope_text} phù hợp:\n"]

    # Build a map from node name → evidence path for per-item evidence
    evidence_by_name = {}
    for path in result.evidence_paths:
        # Extract the main entity name from first segment: "[Accommodation] COBE Homestay --..."
        first_seg = path.split(" --[")[0].strip()
        if "]" in first_seg:
            name = first_seg.split("]")[-1].strip()
            evidence_by_name[name] = path

    for i, node in enumerate(result.answer_nodes[:show_count], 1):
        props = node.get("properties", {})
        name = props.get("name", "Không rõ tên")
        address = props.get("address") or props.get("location") or ""
        rating = props.get("rating") or props.get("averageRating") or ""

        line = f"**{i}. {name}**"
        details = []
        if address:
            details.append(f"📍 {address}")
        if rating:
            details.append(f"⭐ {rating}")
        if details:
            line += "\n   " + " | ".join(details)

        # Per-item evidence
        evidence_path = evidence_by_name.get(name)
        if evidence_path:
            narrative = _build_evidence_narrative([evidence_path])
            if narrative:
                line += f"\n   💡 {narrative}"

        lines.append(line)

    if remaining > 0:
        lines.append(f"\n*và {remaining} kết quả khác...*")

    return "\n".join(lines)


def _build_evidence_narrative(paths: List[str]) -> str:
    """Convert technical evidence paths into natural Vietnamese sentences.

    Input:  "[Accommodation] COBE Homestay --[NEAR]--> [TouristAttraction] Biển Quy Hòa --[NEAR]--> [Restaurant] Nhà hàng Mộc Việt --[HAS]--> [Dish] Cơm niêu"
    Output: "COBE Homestay nằm gần Biển Quy Hòa; khu vực này có Nhà hàng Mộc Việt phục vụ món Cơm niêu."
    """
    sentences = []
    for path in paths[:3]:  # limit to 3 most relevant
        parts = path.split(" --[")
        if len(parts) < 2:
            continue

        # Extract the main entity name from first segment
        first = parts[0].strip()
        main_name = first.split("]")[-1].strip() if "]" in first else first

        # Build narrative from hops
        hop_texts = []
        for part in parts[1:]:
            rel_end = part.find("]-->")
            if rel_end < 0:
                continue
            rel = part[:rel_end].strip()
            rest = part[rel_end + 4:].strip()
            target_name = rest.split("]")[-1].strip() if "]" in rest else rest

            if rel == "NEAR":
                hop_texts.append(f"gần {target_name}")
            elif rel == "HAS":
                hop_texts.append(f"phục vụ món {target_name}")
            elif rel == "LOCATED_IN":
                hop_texts.append(f"nằm tại {target_name}")
            elif rel == "INCLUDES":
                hop_texts.append(f"bao gồm {target_name}")
            else:
                hop_texts.append(f"liên kết với {target_name}")

        if hop_texts:
            sentence = f"**{main_name}** — {', '.join(hop_texts)}."
            sentences.append(sentence)

    return " ".join(sentences) if sentences else ""


def _label_to_vietnamese(label: str) -> str:
    """Map Neo4j labels to Vietnamese display names."""
    mapping = {
        "Accommodation": "khách sạn/nhà nghỉ",
        "Restaurant": "nhà hàng/quán ăn",
        "TouristAttraction": "địa điểm du lịch",
        "Tour": "tour",
        "Dish": "món ăn",
        "Event": "lễ hội/sự kiện",
    }
    return mapping.get(label, label.lower())
