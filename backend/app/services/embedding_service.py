"""
Embedding Service

Handles text embedding generation using Sentence Transformers.
Implements caching and fallback mechanisms for production reliability.
"""

from typing import List, Optional, Dict
from pathlib import Path
import hashlib
import json
import time
from functools import lru_cache

from loguru import logger

from ..core.config import settings, Settings
from ..core.logging_config import get_logger

# Lazy import to avoid loading models at import time
_sentence_transformer = None


class EmbeddingService:
    """
    Service for generating text embeddings using Sentence Transformers.

    Features:
    - Primary and fallback model support
    - Embedding caching to disk
    - Batch processing for efficiency
    - Dimension validation
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the embedding service.

        Args:
            settings: Application settings (uses global settings if not provided)
        """
        self.settings = settings or settings
        self._model = None
        self._fallback_model = None
        self._model_name = None
        self._embedding_dim = None
        self._cache_dir = Path("./data/embedding_cache")
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(__name__)

    def _load_model(self, model_name: str, is_fallback: bool = False) -> bool:
        """
        Load a sentence transformer model.

        Args:
            model_name: Name of the model to load
            is_fallback: Whether this is the fallback model

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.logger.info(f"Loading embedding model: {model_name}")
            start_time = time.time()

            model = SentenceTransformer(model_name)
            elapsed = time.time() - start_time

            self.logger.info(f"Model loaded successfully in {elapsed:.2f}s")

            if is_fallback:
                self._fallback_model = model
                self._fallback_model_name = model_name
            else:
                self._model = model
                self._model_name = model_name
                self._embedding_dim = model.get_sentence_embedding_dimension()
                self.logger.info(f"Embedding dimension: {self._embedding_dim}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False

    def load_models(self) -> bool:
        """
        Load primary and fallback embedding models.

        Returns:
            True if at least one model loaded successfully
        """
        # Try primary model
        primary_loaded = self._load_model(self.settings.embedding_model)

        if not primary_loaded:
            self.logger.warning("Primary model failed, trying fallback...")
            # Try fallback model
            fallback_loaded = self._load_model(self.settings.fallback_embedding_model, is_fallback=True)

            if fallback_loaded:
                # Use fallback as primary
                self._model = self._fallback_model
                self._model_name = self._fallback_model_name
                self._embedding_dim = self._fallback_model.get_sentence_embedding_dimension()
                return True
            return False

        # Try to load fallback as well for redundancy
        self._load_model(self.settings.fallback_embedding_model, is_fallback=True)

        return True

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get current model name."""
        return self._model_name

    @property
    def embedding_dimension(self) -> Optional[int]:
        """Get embedding dimension."""
        return self._embedding_dim

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get path to cache file."""
        return self._cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, text: str) -> Optional[List[float]]:
        """Load embedding from cache if available."""
        if not self.settings.cache_embeddings:
            return None

        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    if data.get("model") == self._model_name:
                        return data.get("embedding")
            except Exception as e:
                self.logger.warning(f"Cache read error: {e}")

        return None

    def _save_to_cache(self, text: str, embedding: List[float]) -> None:
        """Save embedding to cache."""
        if not self.settings.cache_embeddings:
            return

        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "w") as f:
                json.dump({"model": self._model_name, "embedding": embedding}, f)
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}")

    def generate_embedding(self, text: str, batch_size: int = 32) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            batch_size: Batch size for processing

        Returns:
            Embedding vector as list of floats

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Embedding model not loaded. Call load_models() first.")

        # Check cache
        cached = self._load_from_cache(text)
        if cached is not None:
            self.logger.debug(f"Cache hit for text (length: {len(text)})")
            return cached

        # Generate embedding
        self.logger.debug(f"Generating embedding for text (length: {len(text)})")
        start_time = time.time()

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        elapsed = time.time() - start_time
        self.logger.debug(f"Embedding generated in {elapsed:.4f}s")

        # Save to cache
        self._save_to_cache(text, embedding)

        return embedding

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded:
            raise RuntimeError("Embedding model not loaded. Call load_models() first.")

        if not texts:
            return []

        batch_size = batch_size or self.settings.batch_size

        self.logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
        start_time = time.time()

        # Check cache for each text
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_process: List[str] = []
        indices_to_process: List[int] = []

        for i, text in enumerate(texts):
            cached = self._load_from_cache(text)
            if cached is not None:
                embeddings[i] = cached
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)

        # Generate embeddings for non-cached texts
        if texts_to_process:
            self.logger.info(
                f"Cache miss: {len(texts_to_process)}/{len(texts)} texts need processing"
            )

            batch_embeddings = self._model.encode(
                texts_to_process,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress,
            ).tolist()

            # Fill in embeddings and cache
            for idx, (text_idx, embedding) in enumerate(zip(indices_to_process, batch_embeddings)):
                embeddings[text_idx] = embedding
                self._save_to_cache(texts_to_process[idx], embedding)

        elapsed = time.time() - start_time
        self.logger.info(f"Batch embedding completed in {elapsed:.2f}s")

        # Type assertion - all embeddings should be filled
        return [emb for emb in embeddings if emb is not None]

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        import numpy as np

        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Normalize if not already normalized
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        similarity = np.dot(vec1, vec2)
        return float(similarity)


# Global instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(settings: Optional[Settings] = None) -> EmbeddingService:
    """
    Get or create the embedding service singleton.

    Args:
        settings: Optional settings override

    Returns:
        EmbeddingService instance
    """
    global _embedding_service

    if _embedding_service is None:
        _embedding_service = EmbeddingService(settings)

    return _embedding_service
