"""
Tests for Behavioral Signal Parser
"""

import pytest
from datetime import datetime, timedelta
from app.services.behavioral_parser import BehavioralSignalParser, get_behavioral_parser
from app.schemas.candidate import CandidateProfile, WorkExperience


class TestBehavioralSignalParser:
    """Test cases for BehavioralSignalParser."""

    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return BehavioralSignalParser()

    @pytest.fixture
    def complete_profile(self):
        """Create a complete candidate profile."""
        return CandidateProfile(
            candidate_id="test_cand",
            name="Test User",
            headline="Software Engineer",
            summary="Experienced engineer with strong background in...",
            location="San Francisco, CA",
            skills=["Python", "Java", "AWS", "Docker"],
            total_years_experience=5,
            work_experience=[
                WorkExperience(
                    company="Tech Corp",
                    title="Senior Engineer",
                    start_date="2020-01",
                    end_date=None,
                    duration_months=48,
                    is_current=True,
                )
            ],
            education=[],
            projects=[],
        )

    @pytest.fixture
    def sparse_profile(self):
        """Create a sparse candidate profile."""
        return CandidateProfile(
            candidate_id="test_cand_sparse",
            name=None,
            headline=None,
            summary=None,
            location=None,
            skills=[],
            total_years_experience=None,
            work_experience=[],
            education=[],
            projects=[],
        )

    def test_parse_complete_profile(self, parser, complete_profile):
        """Test parsing complete profile."""
        signals = parser.parse(complete_profile)

        assert signals.profile_completeness > 0.5
        assert signals.recency_score > 0
        assert 0 <= signals.behavior_score <= 1

    def test_parse_sparse_profile(self, parser, sparse_profile):
        """Test parsing sparse profile."""
        signals = parser.parse(sparse_profile)

        assert signals.profile_completeness < 0.5
        assert 0 <= signals.behavior_score <= 1

    def test_normalize(self, parser):
        """Test value normalization."""
        assert parser._normalize(0.5) == 0.5
        assert parser._normalize(1.5) == 1.0
        assert parser._normalize(-0.5) == 0.0

    def test_compute_recency_score_recent(self, parser):
        """Test recency score for recent update."""
        recent = datetime.utcnow() - timedelta(days=5)
        score = parser._compute_recency_score(recent)
        assert score >= 0.9

    def test_compute_recency_score_old(self, parser):
        """Test recency score for old update."""
        old = datetime.utcnow() - timedelta(days=400)
        score = parser._compute_recency_score(old)
        assert score <= 0.3

    def test_compute_recency_score_none(self, parser):
        """Test recency score for None."""
        score = parser._compute_recency_score(None)
        assert score == 0.3

    def test_compute_profile_completeness(self, parser, complete_profile):
        """Test profile completeness computation."""
        score = parser._compute_profile_completeness(complete_profile)
        assert 0 <= score <= 1

    def test_compute_trust_score(self, parser, complete_profile):
        """Test trust score computation."""
        signals = parser.parse(complete_profile)
        trust = parser._compute_trust_score(signals)
        assert 0 <= trust <= 1


def test_get_behavioral_parser_singleton():
    """Test that get_behavioral_parser returns singleton."""
    parser1 = get_behavioral_parser()
    parser2 = get_behavioral_parser()
    assert parser1 is parser2
