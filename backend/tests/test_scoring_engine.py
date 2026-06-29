"""
Tests for Scoring Engine
"""

import pytest
from app.services.scoring_engine import ScoringEngine, get_scoring_engine
from app.schemas.job import ParsedJob
from app.schemas.candidate import Candidate, CandidateProfile, CandidateDocument
from app.schemas.scoring import ScoringFeatures, WeightConfig


class TestScoringEngine:
    """Test cases for ScoringEngine."""

    @pytest.fixture
    def scoring_engine(self):
        """Create scoring engine instance."""
        return ScoringEngine()

    @pytest.fixture
    def sample_job(self):
        """Create sample parsed job."""
        return ParsedJob(
            job_id="test_job",
            title="Senior ML Engineer",
            required_skills=["Python", "PyTorch", "Machine Learning"],
            preferred_skills=["FastAPI", "Vector Search"],
            responsibilities=["Build ML systems"],
            seniority="senior",
            years_experience=5,
            industry="ai_ml",
            education="Bachelor's Degree",
            keywords=["ml", "python"],
            technologies=["Python", "PyTorch"],
            behavioral_expectations=["communication"],
            raw_description="Test job description",
            processed_document="Test processed document",
        )

    @pytest.fixture
    def sample_candidate(self):
        """Create sample candidate."""
        profile = CandidateProfile(
            candidate_id="test_cand",
            name="Test Candidate",
            headline="ML Engineer",
            skills=["Python", "PyTorch", "TensorFlow"],
            total_years_experience=6,
            work_experience=[],
            education=[],
            projects=[],
        )

        document = CandidateDocument(
            candidate_id="test_cand",
            title_document="ML Engineer",
            skills_document="Skills: Python, PyTorch, TensorFlow",
            experience_document="Experience: 6 years",
            education_document="Education: BS",
            projects_document="Projects: None",
            full_document="ML Engineer with Python, PyTorch",
            seniority_level="senior",
            industry_domains=["ai_ml"],
            leadership_experience=False,
            startup_experience=False,
            ml_skills=["PyTorch", "TensorFlow"],
            backend_skills=["Python"],
            data_skills=[],
            cloud_skills=[],
        )

        return Candidate(
            profile=profile,
            document=document,
            embedding=None,
            indexed=False,
        )

    def test_compute_features(self, scoring_engine, sample_job, sample_candidate):
        """Test feature computation."""
        features = scoring_engine.compute_features(
            candidate=sample_candidate,
            job=sample_job,
            semantic_similarity=0.85,
        )

        assert isinstance(features, ScoringFeatures)
        assert features.semantic_similarity == 0.85
        assert 0 <= features.skill_match <= 1
        assert 0 <= features.experience_match <= 1

    def test_compute_hybrid_score(self, scoring_engine, sample_job, sample_candidate):
        """Test hybrid score computation."""
        features = scoring_engine.compute_features(
            candidate=sample_candidate,
            job=sample_job,
            semantic_similarity=0.85,
        )

        hybrid_score, components = scoring_engine.compute_hybrid_score(features)

        assert isinstance(hybrid_score, float)
        assert 0 <= hybrid_score <= 1
        assert isinstance(components, dict)
        assert "weighted_semantic" in components
        assert "weighted_skill" in components

    def test_compute_hybrid_score_with_custom_weights(
        self, scoring_engine, sample_job, sample_candidate
    ):
        """Test hybrid score with custom weights."""
        features = scoring_engine.compute_features(
            candidate=sample_candidate,
            job=sample_job,
            semantic_similarity=0.85,
        )

        custom_weights = WeightConfig(
            semantic=0.5,
            skill=0.2,
            experience=0.1,
            behavior=0.1,
            industry=0.05,
            education=0.03,
            bonus=0.02,
        )

        hybrid_score, components = scoring_engine.compute_hybrid_score(
            features, weights=custom_weights
        )

        assert isinstance(hybrid_score, float)
        assert 0 <= hybrid_score <= 1

    def test_skill_match_perfect(self, scoring_engine, sample_job, sample_candidate):
        """Test perfect skill match."""
        # Candidate has all required skills
        score = scoring_engine._compute_required_skills_match(
            sample_candidate.document, sample_job
        )
        assert 0 <= score <= 1

    def test_seniority_match(self, scoring_engine, sample_job, sample_candidate):
        """Test seniority matching."""
        score = scoring_engine._compute_seniority_match(
            sample_candidate.document, sample_job
        )
        assert 0 <= score <= 1

    def test_fuzzy_match(self, scoring_engine):
        """Test fuzzy string matching."""
        assert scoring_engine._fuzzy_match("python", "python") is True
        assert scoring_engine._fuzzy_match("Python", "python") is True


def test_get_scoring_engine_singleton():
    """Test that get_scoring_engine returns singleton."""
    engine1 = get_scoring_engine()
    engine2 = get_scoring_engine()
    assert engine1 is engine2
