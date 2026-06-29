"""
Pydantic Schemas Module

Data validation and serialization models for the application.
"""

from .job import JobDescription, JobDescriptionInput, ParsedJob
from .candidate import (
    Candidate,
    CandidateProfile,
    CandidateInput,
    CandidateDocument,
    BehavioralSignals,
)
from .recommendation import (
    RecommendationRequest,
    RecommendationResponse,
    RankedCandidate,
    RecommendationResult,
)
from .scoring import ScoringFeatures, ScoringResult, WeightConfig
from .api import (
    HealthResponse,
    MetricsResponse,
    UploadJobResponse,
    IndexCandidatesResponse,
    ErrorResponse,
)

__all__ = [
    # Job schemas
    "JobDescription",
    "JobDescriptionInput",
    "ParsedJob",
    # Candidate schemas
    "Candidate",
    "CandidateProfile",
    "CandidateInput",
    "CandidateDocument",
    "BehavioralSignals",
    # Recommendation schemas
    "RecommendationRequest",
    "RecommendationResponse",
    "RankedCandidate",
    "RecommendationResult",
    # Scoring schemas
    "ScoringFeatures",
    "ScoringResult",
    "WeightConfig",
    # API schemas
    "HealthResponse",
    "MetricsResponse",
    "UploadJobResponse",
    "IndexCandidatesResponse",
    "ErrorResponse",
]
