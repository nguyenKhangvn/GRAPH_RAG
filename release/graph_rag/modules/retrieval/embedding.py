import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

class LocalEmbeddingService:
    def __init__(self):
        self.use_api = os.getenv("USE_HF_INFERENCE_API", "false").lower() == "true"
        self.hf_token = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")
        self.model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip().strip('"').strip("'")
        
        if self.use_api:
            logger.info("Using Hugging Face Inference API via InferenceClient (Model: %s)", self.model_name)
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(api_key=self.hf_token)
        else:
            logger.info("Loading embedding model local: %s...", self.model_name)
            self.model = self._load_model(self.model_name)
            logger.info("Embedding model loaded.")

    @staticmethod
    def _load_model(model_name: str):
        """Load SentenceTransformer with offline fallback."""
        from sentence_transformers import SentenceTransformer
        from huggingface_hub import snapshot_download
        
        offline = os.getenv("HF_HUB_OFFLINE", "0") == "1"

        def load_from_local_cache() -> SentenceTransformer:
            local_path = snapshot_download(model_name, local_files_only=True)
            logger.info("Loading embedding model from local snapshot: %s", local_path)
            return SentenceTransformer(local_path, local_files_only=True)

        if offline:
            logger.info("HF_HUB_OFFLINE=1 — loading model from local cache only")
            return load_from_local_cache()

        try:
            return SentenceTransformer(model_name)
        except (OSError, IOError, ConnectionError, TimeoutError, RuntimeError, ValueError) as online_err:
            err_str = str(online_err).lower()
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
            raise

    def _call_api(self, inputs):
        """Call Hugging Face Inference API via InferenceClient with retries."""
        import time
        
        last_error = None
        for attempt in range(5):
            try:
                res = self.client.feature_extraction(inputs, model=self.model_name)
                
                # Convert numpy array to list if needed
                if hasattr(res, "tolist"):
                    res = res.tolist()
                
                # Normalize output structure
                if isinstance(inputs, str):
                    if res and isinstance(res, list) and isinstance(res[0], list):
                        res = res[0]
                else:
                    if res and isinstance(res, list) and not isinstance(res[0], list):
                        res = [res]
                        
                return res
            except Exception as e:
                last_error = e
                logger.warning("Hugging Face Inference API error on attempt %d: %s", attempt + 1, e)
                # Exponential backoff or constant sleep before retrying
                time.sleep(2)
        
        logger.error("Hugging Face Inference API failed after all retries. Last error: %s", last_error)
        raise RuntimeError(f"Hugging Face Inference API failed. Error: {last_error}") from last_error

    def embed_query(self, text: str) -> list:
        """Chuyển text thành vector (list of floats)."""
        if not text:
            return []
        
        if self.use_api:
            return self._call_api(text)
        
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-embed texts to reduce latency."""
        if not texts:
            return []
        
        if self.use_api:
            return self._call_api(texts)
            
        embeddings = self.model.encode(texts, show_progress_bar=False)
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [np.asarray(vec).tolist() for vec in embeddings]

