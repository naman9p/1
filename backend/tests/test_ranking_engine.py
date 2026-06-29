"""
Tests for Ranking Engine
"""

import pytest
from app.services.ranking_engine import RankingEngine, get_ranking_engine
from app.schemas.job import ParsedJob
from app.schemas.candidate import Candidate, CandidateProfile, CandidateDocument
from app.schemas.scoring import ScoringResult, ScoringFeatures


class TestRankingEngine:
    """Test cases for RankingEngine."""

    @pytest.fixture
    def ranking_engine(self):
        """Create ranking engine instance."""
        engine = RankingEngine()
        return engine

    @pytest.fixture
    def sample_job(self):
        """Create sample parsed job."""
        return ParsedJob(
            job_id="test_job",
            title="Senior ML Engineer",
            required_skills=["Python", "PyTorch"],
            preferred_skills=["FastAPI"],
            responsibilities=["Build ML systems"],
            seniority="senior",
            years_experience=5,
            industry="ai_ml",
            education="Bachelor's Degree",
            keywords=["ml", "python"],
            technologies=["Python", "PyTorch"],
            behavioral_expectations=["communication"],
            raw_description="Test job description",
            processed_document="Senior ML Engineer with Python, PyTorch",
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
            full_document="ML Engineer with Python, PyTorch, TensorFlow",
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

    def test_engine_initialization(self, ranking_engine):
        """Test engine initializes correctly."""
        assert ranking_engine is not None
        assert not ranking_engine.is_loaded

    def test_load_model(self, ranking_engine):
        """Test model loading."""
        # This test may fail if model can't be downloaded
        result = ranking_engine.load_model()
        assert isinstance(result, bool)

    def test_generate_reasoning(self, ranking_engine, sample_job, sample_candidate):
        """Test reasoning generation."""
        features = ScoringFeatures(
            semantic_similarity=0.85,
            skill_match=0.8,
            required_skills_match=0.85,
            experience_match=0.75,
            behavior_score=0.7,
            ml_experience=0.9,
            python_expertise=0.8,
        )

        reasoning = ranking_engine.generate_reasoning(
            candidate=sample_candidate,
            job=sample_job,
            features=features,
            final_score=0.82,
        )

        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        # Reasoning should mention some positive aspects
        assert any(
            keyword in reasoning.lower()
            for keyword in ["alignment", "match", "experience", "skill"]
        )

    def test_generate_reasoning_low_score(self, ranking_engine, sample_job, sample_candidate):
        """Test reasoning generation for low score."""
        features = ScoringFeatures(
            semantic_similarity=0.4,
            skill_match=0.3,
            required_skills_match=0.35,
            experience_match=0.4,
            behavior_score=0.3,
        )

        reasoning = ranking_engine.generate_reasoning(
            candidate=sample_candidate,
            job=sample_job,
            features=features,
            final_score=0.35,
        )

        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_rerank_empty_list(self, ranking_engine, sample_job):
        """Test re-ranking with empty list."""
        result = ranking_engine.rerank(
            job=sample_job,
            candidates=[],
            top_k=10,
        )
        assert result == []


def test_get_ranking_engine_singleton():
    """Test that get_ranking_engine returns singleton."""
    engine1 = get_ranking_engine()
    engine2 = get_ranking_engine()
    assert engine1 is engine2
