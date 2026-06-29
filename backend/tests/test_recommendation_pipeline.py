"""
Tests for Recommendation Pipeline
"""

import pytest
from app.services.recommendation_pipeline import RecommendationPipeline, get_recommendation_pipeline
from app.schemas.job import JobDescriptionInput
from app.schemas.candidate import CandidateProfile
from app.schemas.recommendation import RecommendationRequest


class TestRecommendationPipeline:
    """Test cases for RecommendationPipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        return RecommendationPipeline()

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initializes correctly."""
        assert pipeline is not None
        assert not pipeline.is_initialized

    def test_initialize(self, pipeline):
        """Test pipeline initialization."""
        # This may fail in test environment without models
        result = pipeline.initialize()
        assert isinstance(result, bool)

    def test_process_job(self, pipeline):
        """Test job processing."""
        if not pipeline.is_initialized:
            pipeline.initialize()

        job_input = JobDescriptionInput(
            job_id="test_job",
            title="Test Engineer",
            description="Looking for a test engineer with Python skills.",
        )

        job_description = pipeline.process_job(job_input)

        assert job_description is not None
        assert job_description.parsed_job.job_id == "test_job"
        assert job_description.embedding is not None
        assert len(job_description.embedding) > 0

    def test_index_candidates(self, pipeline):
        """Test candidate indexing."""
        if not pipeline.is_initialized:
            pipeline.initialize()

        candidates = [
            CandidateProfile(
                candidate_id=f"test_cand_{i}",
                name=f"Test Candidate {i}",
                headline="Engineer",
                skills=["Python"],
                total_years_experience=5,
            )
            for i in range(3)
        ]

        indexed, failed = pipeline.index_candidates(candidates)

        assert indexed + failed == len(candidates)
        assert len(pipeline._candidates) > 0

    def test_recommend(self, pipeline):
        """Test recommendation generation."""
        if not pipeline.is_initialized:
            pipeline.initialize()

        # Setup job
        job_input = JobDescriptionInput(
            job_id="test_job",
            title="Test Engineer",
            description="Looking for a test engineer with Python skills.",
        )
        pipeline.process_job(job_input)

        # Setup candidates
        candidates = [
            CandidateProfile(
                candidate_id=f"test_cand_{i}",
                name=f"Test Candidate {i}",
                headline="Engineer",
                skills=["Python", "Testing"],
                total_years_experience=5,
            )
            for i in range(3)
        ]
        pipeline.index_candidates(candidates)

        # Generate recommendations
        request = RecommendationRequest(
            job_id="test_job",
            top_k=10,
            include_reasoning=True,
            use_reranking=False,  # Skip cross-encoder in tests
        )

        result = pipeline.recommend(request)

        assert result is not None
        assert result.job_id == "test_job"
        assert result.total_candidates_processed == 3
        assert len(result.recommendations) <= 3

    def test_export_to_csv(self, pipeline, tmp_path):
        """Test CSV export."""
        if not pipeline.is_initialized:
            pipeline.initialize()

        # Setup job
        job_input = JobDescriptionInput(
            job_id="test_job",
            title="Test Engineer",
            description="Test description",
        )
        pipeline.process_job(job_input)

        # Setup candidates
        candidates = [
            CandidateProfile(
                candidate_id="test_cand_1",
                name="Test Candidate",
                headline="Engineer",
                skills=["Python"],
                total_years_experience=5,
            )
        ]
        pipeline.index_candidates(candidates)

        # Generate and export
        request = RecommendationRequest(
            job_id="test_job",
            top_k=10,
            include_reasoning=True,
            use_reranking=False,
        )
        result = pipeline.recommend(request)

        output_file = tmp_path / "test_output.csv"
        output_path = pipeline.export_to_csv(result, str(output_file))

        assert output_path is not None
        assert output_file.exists()

    def test_get_candidate(self, pipeline):
        """Test getting candidate by ID."""
        if not pipeline.is_initialized:
            pipeline.initialize()

        candidate_profile = CandidateProfile(
            candidate_id="test_cand",
            name="Test Candidate",
            headline="Engineer",
        )
        pipeline.index_candidates([candidate_profile])

        candidate = pipeline.get_candidate("test_cand")
        assert candidate is not None
        assert candidate.profile.candidate_id == "test_cand"

    def test_get_candidate_not_found(self, pipeline):
        """Test getting non-existent candidate."""
        if not pipeline.is_initialized:
            pipeline.initialize()

        candidate = pipeline.get_candidate("non_existent")
        assert candidate is None

    def test_get_metrics(self, pipeline):
        """Test getting metrics."""
        if not pipeline.is_initialized:
            pipeline.initialize()

        metrics = pipeline.get_metrics()

        assert isinstance(metrics, dict)
        assert "total_candidates" in metrics
        assert "total_jobs" in metrics
        assert "vector_store_size" in metrics


def test_get_recommendation_pipeline_singleton():
    """Test that get_recommendation_pipeline returns singleton."""
    pipeline1 = get_recommendation_pipeline()
    pipeline2 = get_recommendation_pipeline()
    assert pipeline1 is pipeline2
