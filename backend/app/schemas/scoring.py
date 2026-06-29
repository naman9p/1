"""
Scoring Schemas

Pydantic models for scoring features and results.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class ScoringFeatures(BaseModel):
    """
    Individual scoring features for a candidate.
    All scores are normalized between 0 and 1.
    """

    # Semantic similarity (from vector search)
    semantic_similarity: float = Field(default=0.0, ge=0, le=1, description="Semantic similarity score")

    # Skill matching
    skill_match: float = Field(default=0.0, ge=0, le=1, description="Skill match percentage")
    required_skills_match: float = Field(default=0.0, ge=0, le=1, description="Required skills match")
    preferred_skills_match: float = Field(default=0.0, ge=0, le=1, description="Preferred skills match")

    # Experience matching
    experience_match: float = Field(default=0.0, ge=0, le=1, description="Experience level match")
    years_experience_match: float = Field(default=0.0, ge=0, le=1, description="Years of experience match")
    seniority_match: float = Field(default=0.0, ge=0, le=1, description="Seniority level match")

    # Behavioral signals
    behavior_score: float = Field(default=0.0, ge=0, le=1, description="Behavioral signals score")

    # Industry and domain
    industry_match: float = Field(default=0.0, ge=0, le=1, description="Industry domain match")

    # Education
    education_match: float = Field(default=0.0, ge=0, le=1, description="Education level match")

    # Bonus features
    leadership_score: float = Field(default=0.0, ge=0, le=1, description="Leadership experience score")
    startup_experience: float = Field(default=0.0, ge=0, le=1, description="Startup experience flag")
    ml_experience: float = Field(default=0.0, ge=0, le=1, description="ML/AI experience score")
    llm_experience: float = Field(default=0.0, ge=0, le=1, description="LLM experience score")
    vector_search_experience: float = Field(default=0.0, ge=0, le=1, description="Vector search experience")
    fastapi_experience: float = Field(default=0.0, ge=0, le=1, description="FastAPI experience")
    python_expertise: float = Field(default=0.0, ge=0, le=1, description="Python expertise score")
    open_source_contribution: float = Field(default=0.0, ge=0, le=1, description="Open source contributions")
    project_quality: float = Field(default=0.0, ge=0, le=1, description="Project quality score")
    employment_stability: float = Field(default=0.0, ge=0, le=1, description="Employment stability score")
    recent_activity: float = Field(default=0.0, ge=0, le=1, description="Recent activity score")

    # Cross-encoder score (for re-ranking)
    cross_encoder_score: Optional[float] = Field(None, ge=0, le=1, description="Cross-encoder re-ranking score")

    class Config:
        json_schema_extra = {
            "example": {
                "semantic_similarity": 0.89,
                "skill_match": 0.85,
                "required_skills_match": 0.90,
                "experience_match": 0.80,
                "behavior_score": 0.75,
                "industry_match": 0.85,
                "education_match": 0.70,
                "ml_experience": 0.95,
                "python_expertise": 0.90,
            }
        }


class ScoringResult(BaseModel):
    """
    Final scoring result for a candidate.
    """

    candidate_id: str = Field(..., description="Candidate identifier")

    # Individual feature scores
    features: ScoringFeatures = Field(..., description="Individual scoring features")

    # Weighted scores
    weighted_semantic: float = Field(default=0.0, description="Weighted semantic score")
    weighted_skill: float = Field(default=0.0, description="Weighted skill score")
    weighted_experience: float = Field(default=0.0, description="Weighted experience score")
    weighted_behavior: float = Field(default=0.0, description="Weighted behavior score")
    weighted_industry: float = Field(default=0.0, description="Weighted industry score")
    weighted_education: float = Field(default=0.0, description="Weighted education score")
    weighted_bonus: float = Field(default=0.0, description="Weighted bonus score")

    # Final scores
    hybrid_score: float = Field(default=0.0, ge=0, le=1, description="Hybrid weighted score")
    rerank_score: Optional[float] = Field(None, ge=0, le=1, description="Re-ranked score")
    final_score: float = Field(default=0.0, ge=0, le=1, description="Final combined score")

    # Ranking
    initial_rank: Optional[int] = Field(None, description="Initial retrieval rank")
    rerank_rank: Optional[int] = Field(None, description="Post re-ranking rank")
    final_rank: Optional[int] = Field(None, description="Final rank")

    class Config:
        json_schema_extra = {
            "example": {
                "candidate_id": "cand_001",
                "hybrid_score": 0.847,
                "rerank_score": 0.892,
                "final_score": 0.870,
                "final_rank": 1,
            }
        }


class WeightConfig(BaseModel):
    """
    Configuration for scoring weights.
    """

    semantic: float = Field(default=0.40, ge=0, le=1, description="Semantic similarity weight")
    skill: float = Field(default=0.20, ge=0, le=1, description="Skill match weight")
    experience: float = Field(default=0.10, ge=0, le=1, description="Experience match weight")
    behavior: float = Field(default=0.10, ge=0, le=1, description="Behavioral signals weight")
    industry: float = Field(default=0.10, ge=0, le=1, description="Industry match weight")
    education: float = Field(default=0.05, ge=0, le=1, description="Education match weight")
    bonus: float = Field(default=0.05, ge=0, le=1, description="Bonus features weight")

    class Config:
        json_schema_extra = {
            "example": {
                "semantic": 0.40,
                "skill": 0.20,
                "experience": 0.10,
                "behavior": 0.10,
                "industry": 0.10,
                "education": 0.05,
                "bonus": 0.05,
            }
        }

    @property
    def total(self) -> float:
        """Sum of all weights."""
        return (
            self.semantic
            + self.skill
            + self.experience
            + self.behavior
            + self.industry
            + self.education
            + self.bonus
        )

    def is_valid(self) -> bool:
        """Check if weights sum to approximately 1.0."""
        return abs(self.total - 1.0) < 0.01

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "semantic": self.semantic,
            "skill": self.skill,
            "experience": self.experience,
            "behavior": self.behavior,
            "industry": self.industry,
            "education": self.education,
            "bonus": self.bonus,
        }
