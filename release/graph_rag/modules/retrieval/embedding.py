import os
import logging
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from graph_rag.config import EMBEDDING_MODEL_NAME
import numpy as np

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    def __init__(self):
        logger.info("Loading embedding model: %s...", EMBEDDING_MODEL_NAME)
        self.model = self._load_model(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded.")

    @staticmethod
    def _load_model(model_name: str) -> SentenceTransformer:
        """Load SentenceTransformer with offline fallback.

        1. If HF_HUB_OFFLINE=1 → load from local cache only (no network).
        2. Otherwise try normal online load first.
        3. On network/DNS failure, retry with local_files_only=True as fallback.
        4. If both fail, raise with a clear message telling the user what to do.
        """
        offline = os.getenv("HF_HUB_OFFLINE", "0") == "1"

        def load_from_local_cache() -> SentenceTransformer:
            local_path = snapshot_download(model_name, local_files_only=True)
            logger.info("Loading embedding model from local snapshot: %s", local_path)
            return SentenceTransformer(local_path, local_files_only=True)

        # Explicit offline mode requested
        if offline:
            logger.info("HF_HUB_OFFLINE=1 — loading model from local cache only")
            return load_from_local_cache()

        # Try normal (online) load
        try:
            return SentenceTransformer(model_name)
        except (OSError, IOError, ConnectionError, TimeoutError, RuntimeError, ValueError) as online_err:
            err_str = str(online_err).lower()
            # Network / DNS errors — retry from local cache
            network_keywords = [
                "failed to resolve", "nameresolutionerror",
                "getaddrinfo", "connecttimeout", "connectionerror",
                "urlopen error", "temporary failure",
                "nodename nor servname", "network is unreachable",
            ]
            is_network = any(kw in err_str for kw in network_keywords)

            if is_network:
                logger.warning(
                    "Network error loading model from HuggingFace: %s\n"
                    "Retrying with local cache (local_files_only=True)...",
                    online_err,
                )
                try:
                    return load_from_local_cache()
                except (OSError, IOError, RuntimeError, ValueError) as local_err:
                    raise RuntimeError(
                        f"Cannot load embedding model '{model_name}' — "
                        f"network unavailable and no local cache found.\n"
                        f"Network error: {online_err}\n"
                        f"Local cache error: {local_err}\n\n"
                        f"FIX: Run this once with internet access to cache the model:\n"
                        f"  python -c \"from sentence_transformers import SentenceTransformer; "
                        f"SentenceTransformer('{model_name}')\"\n\n"
                        f"Or set HF_HUB_OFFLINE=1 after the model is cached."
                    ) from local_err

            # Non-network error — raise directly
            raise

    def embed_query(self, text: str) -> list:
        """
        Chuyển text thành vector (list of floats).
        """
        if not text:
            return []
        
        # Model trả về numpy array, cần convert sang list python chuẩn
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed texts to reduce latency and suppress noisy progress output."""
        if not texts:
            return []
        embeddings = self.model.encode(texts, show_progress_bar=False)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [np.asarray(vec).tolist() for vec in embeddings]
