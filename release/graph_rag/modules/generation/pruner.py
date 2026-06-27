# graph_rag/modules/generation/pruner.py
import logging

logger = logging.getLogger(__name__)

from typing import List, Optional
import numpy as np
import re
import time
import unicodedata


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Tính cosine similarity giữa 2 vectors (list of float)."""
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for near-duplicate detection: strip accents, lowercase, collapse whitespace."""
    norm = unicodedata.normalize("NFKD", str(text))
    norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
    norm = norm.lower().replace("đ", "d")
    norm = re.sub(r"[^a-z0-9\s]", " ", norm)
    return re.sub(r"\s+", " ", norm).strip()


class ContextPruner:
    """
    Làm sạch, lọc trùng và chọn Context tối ưu bằng MMR (Maximal Marginal Relevance).

    MMR cân bằng 2 tiêu chí:
      - Relevance   : fact phải liên quan đến query (cosine similarity với query embedding)
      - Diversity   : fact không được quá trùng lặp với những fact đã chọn

    Công thức: MMR(d) = λ · sim(d, query) − (1 − λ) · max_{s ∈ S} sim(d, s)
      λ = 0.7  →  70% relevance, 30% diversity

    Fallback: Nếu không có query_embedding hoặc embedding_service, dùng lại logic
    deduplicate + top-K cũ (không crash pipeline).
    """

    def __init__(self, lambda_param: float = 0.7):
        self.lambda_ = lambda_param

    _embedding_cache = {}
    _embedding_cache_ttl_sec = 900
    _embedding_cache_max_items = 2000

    @staticmethod
    def _normalize_for_tokens(text: str) -> List[str]:
        if not text:
            return []
        cleaned = re.sub(r"[^\w\s]", " ", str(text).lower(), flags=re.UNICODE)
        return [t for t in cleaned.split() if len(t) >= 2]

    @staticmethod
    def _lexical_preselect_facts(facts: List[str], query_text: str, limit: int) -> List[str]:
        if not facts or not query_text or len(facts) <= limit:
            return facts

        query_tokens = set(ContextPruner._normalize_for_tokens(query_text))
        if not query_tokens:
            return facts[:limit]

        scored = []
        for idx, fact in enumerate(facts):
            fact_tokens = set(ContextPruner._normalize_for_tokens(fact))
            overlap = len(query_tokens.intersection(fact_tokens))
            scored.append((overlap, -idx, fact))

        scored.sort(reverse=True)
        selected = [item[2] for item in scored[:limit]]
        return selected

    @staticmethod
    def _prune_embedding_cache():
        cache = ContextPruner._embedding_cache
        now = time.time()
        expired_keys = [k for k, v in cache.items() if (now - v["ts"]) > ContextPruner._embedding_cache_ttl_sec]
        for k in expired_keys:
            cache.pop(k, None)

        # Soft cap with oldest-first eviction
        if len(cache) > ContextPruner._embedding_cache_max_items:
            sorted_items = sorted(cache.items(), key=lambda kv: kv[1]["ts"])
            drop_count = len(cache) - ContextPruner._embedding_cache_max_items
            for i in range(drop_count):
                cache.pop(sorted_items[i][0], None)

    # ------------------------------------------------------------------
    # PUBLIC API (giữ tương thích ngược: vẫn là static method đơn giản)
    # ------------------------------------------------------------------

    @staticmethod
    def prune(
        context_list: List[str],
        max_items: int = 20,
        query_embedding: Optional[List[float]] = None,
        embedding_service=None,
        query_text: str = "",
    ) -> str:
        """
        Prune context với MMR nếu có query_embedding + embedding_service.
        Fallback về top-K deduplication nếu thiếu.

        Args:
            context_list      : danh sách fact thô từ GraphTraverser
            max_items         : số fact tối đa giữ lại
            query_embedding   : vector của user query (list of float)
            embedding_service : LocalEmbeddingService, có method embed_query(str)->list
        """
        if not context_list:
            return "Không có thông tin ngữ cảnh."

        # Deduplicate (preserve insertion order) trước MMR — normalized comparison
        seen_norm: set = set()
        unique_context: List[str] = []
        for item in context_list:
            norm = _normalize_for_dedup(item)
            if norm and norm not in seen_norm:
                seen_norm.add(norm)
                unique_context.append(item)

        # Reduce candidate set before MMR on large context.
        # Phase 10: Increased from 3x to 4x to preserve more candidates
        # before MMR selection (improves retrieval recall).
        pre_mmr_limit = max(max_items * 4, 80)
        if len(unique_context) > pre_mmr_limit:
            unique_context = ContextPruner._lexical_preselect_facts(
                unique_context,
                query_text=query_text,
                limit=pre_mmr_limit,
            )

        # Nếu đủ điều kiện → chạy MMR
        if query_embedding and embedding_service and len(unique_context) > max_items:
            selected = ContextPruner._mmr_select(
                facts=unique_context,
                query_embedding=query_embedding,
                embedding_service=embedding_service,
                max_items=max_items,
                lambda_param=0.7,
            )
        else:
            # Fallback: giữ thứ tự gốc (đã được Retriever/Traverser sort theo score)
            selected = unique_context[:max_items]

        return "\n".join([f"- {line}" for line in selected])

    # ------------------------------------------------------------------
    # MMR ALGORITHM
    # ------------------------------------------------------------------

    @staticmethod
    def _mmr_select(
        facts: List[str],
        query_embedding: List[float],
        embedding_service,
        max_items: int,
        lambda_param: float,
    ) -> List[str]:
        """
        Chọn tối đa max_items facts theo thuật toán MMR.

        Bước 1: Embed toàn bộ facts (batch encode để tăng tốc).
        Bước 2: Lặp greedy: mỗi vòng chọn fact có MMR score cao nhất.
        """
        # --- Bước 1: Embed facts (batch) ---
        try:
            ContextPruner._prune_embedding_cache()
            fact_embeddings: List[List[float]] = [None] * len(facts)

            missing_indices = []
            missing_facts = []
            for idx, fact in enumerate(facts):
                cached = ContextPruner._embedding_cache.get(fact)
                if cached is not None:
                    fact_embeddings[idx] = cached["vec"]
                else:
                    missing_indices.append(idx)
                    missing_facts.append(fact)

            if missing_facts:
                if hasattr(embedding_service, "embed_texts"):
                    new_vectors = embedding_service.embed_texts(missing_facts)
                else:
                    new_vectors = [embedding_service.embed_query(f) for f in missing_facts]

                now = time.time()
                for idx, fact, vec in zip(missing_indices, missing_facts, new_vectors):
                    fact_embeddings[idx] = vec
                    ContextPruner._embedding_cache[fact] = {"vec": vec, "ts": now}
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.warning("       MMR embedding warning (fallback to top-K): %s", e)
            return facts[:max_items]

        # --- Bước 2: Tính relevance scores một lần ---
        relevance = [
            _cosine_similarity(query_embedding, emb) for emb in fact_embeddings
        ]

        selected_indices: List[int] = []
        remaining_indices = list(range(len(facts)))

        # Minimum MMR score threshold — stop selecting when the best
        # remaining candidate is no longer relevant enough.  This prevents
        # "best of the worst" selections (e.g., picking facts with 2.4%
        # relevance for a "khinh khí cầu" query).
        # FIX: Lowered from 0.15 to 0.08 to avoid over-pruning list queries.
        # Phase 10: Lowered from 0.08 to 0.04 to improve retrieval recall
        # (56 zero-recall questions in deepeval v3 — many valid facts were
        # pruned because embeddings had low similarity scores).
        MMR_MIN_SCORE = 0.04

        while len(selected_indices) < max_items and remaining_indices:
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining_indices:
                rel_score = relevance[idx]

                # Độ giống nhau tối đa với các fact đã chọn
                if selected_indices:
                    max_sim = max(
                        _cosine_similarity(fact_embeddings[idx], fact_embeddings[sel])
                        for sel in selected_indices
                    )
                else:
                    max_sim = 0.0

                mmr_score = lambda_param * rel_score - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            # Stop if the best remaining candidate is below threshold
            if best_score < MMR_MIN_SCORE:
                break

            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [facts[i] for i in selected_indices]