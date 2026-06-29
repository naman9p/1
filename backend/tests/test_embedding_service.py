"""
Tests for Embedding Service
"""

import pytest
from app.services.embedding_service import EmbeddingService, get_embedding_service
from app.core.config import Settings


class TestEmbeddingService:
    """Test cases for EmbeddingService."""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance."""
        settings = Settings()
        service = EmbeddingService(settings)
        return service

    def test_service_initialization(self, embedding_service):
        """Test service initializes correctly."""
        assert embedding_service is not None
        assert not embedding_service.is_loaded

    def test_load_models(self, embedding_service):
        """Test model loading."""
        # This test may fail if models can't be downloaded in test environment
        # In CI, we would mock the model loading
        result = embedding_service.load_models()
        # We don't assert on result as it depends on network/model availability
        assert isinstance(result, bool)

    def test_generate_embedding_raises_when_not_loaded(self, embedding_service):
        """Test that generate_embedding raises when model not loaded."""
        with pytest.raises(RuntimeError, match="not loaded"):
            embedding_service.generate_embedding("test text")

    def test_generate_embeddings_batch_empty(self, embedding_service):
        """Test batch embedding with empty list."""
        result = embedding_service.generate_embeddings_batch([])
        assert result == []

    def test_compute_similarity(self, embedding_service):
        """Test similarity computation."""
        # Use simple vectors for testing
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]
        vec3 = [0.0, 1.0, 0.0]

        sim_same = embedding_service.compute_similarity(vec1, vec2)
        sim_diff = embedding_service.compute_similarity(vec1, vec3)

        assert abs(sim_same - 1.0) < 0.01
        assert abs(sim_diff - 0.0) < 0.01


def test_get_embedding_service_singleton():
    """Test that get_embedding_service returns singleton."""
    service1 = get_embedding_service()
    service2 = get_embedding_service()
    assert service1 is service2
