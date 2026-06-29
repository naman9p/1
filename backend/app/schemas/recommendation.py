"""
Recommendation Schemas

Pydantic models for recommendation requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class RecommendationRequest(BaseModel):
    """
    Request schema for getting recommendations.
    """

    job_id: str = Field(..., description="Job identifier to get recommendations for")
    top_k: int = Field(default=100, ge=1, le=100, description="Number of recommendations to return")
    include_reasoning: bool = Field(default=True, description="Include reasoning for each recommendation")
    use_reranking: bool = Field(default=True, description="Use cross-encoder re-ranking")
    min_score: Optional[float] = Field(None, ge=0, le=1, description="Minimum score threshold")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "top_k": 100,
                "include_reasoning": True,
                "use_reranking": True,
            }
        }


class RankedCandidate(BaseModel):
    """
    A single ranked candidate in the recommendation results.
    """

    rank: int = Field(..., description="Rank position")
    candidate_id: str = Field(..., description="Candidate identifier")
    candidate_name: Optional[str] = Field(None, description="Candidate name")
    candidate_headline: Optional[str] = Field(None, description="Candidate headline")

    # Scores
    semantic_score: float = Field(..., description="Semantic similarity score")
    skill_score: float = Field(..., description="Skill match score")
    experience_score: float = Field(..., description="Experience match score")
    behavior_score: float = Field(..., description="Behavioral signals score")
    hybrid_score: float = Field(..., description="Hybrid weighted score")
    rerank_score: Optional[float] = Field(None, description="Cross-encoder re-rank score")
    final_score: float = Field(..., description="Final combined score")

    # Key attributes
    total_years_experience: Optional[float] = Field(None, description="Years of experience")
    seniority_level: Optional[str] = Field(None, description="Seniority level")
    top_skills: List[str] = Field(default_factory=list, description="Top matching skills")
    industry: Optional[str] = Field(None, description="Primary industry")

    # Reasoning
    reasoning: Optional[str] = Field(None, description="Explanation for ranking")

    class Config:
        json_schema_extra = {
            "example": {
                "rank": 1,
                "candidate_id": "cand_001",
                "candidate_name": "John Doe",
                "candidate_headline": "Senior ML Engineer",
                "semantic_score": 0.89,
                "skill_score": 0.85,
                "experience_score": 0.80,
                "behavior_score": 0.75,
                "hybrid_score": 0.847,
                "rerank_score": 0.892,
                "final_score": 0.870,
                "total_years_experience": 7,
                "seniority_level": "senior",
                "top_skills": ["Python", "PyTorch", "FastAPI", "Vector Search"],
                "reasoning": "Excellent semantic alignment with production ML requirements...",
            }
        }


class RecommendationResult(BaseModel):
    """
    Complete recommendation result.
    """

    job_id: str = Field(..., description="Job identifier")
    job_title: Optional[str] = Field(None, description="Job title")
    total_candidates_processed: int = Field(..., description="Total candidates in database")
    candidates_retrieved: int = Field(..., description="Candidates retrieved from vector search")
    candidates_reranked: int = Field(..., description="Candidates re-ranked with cross-encoder")
    final_recommendations: int = Field(..., description="Final number of recommendations")

    # Recommendations
    recommendations: List[RankedCandidate] = Field(
        default_factory=list, description="List of ranked candidates"
    )

    # Metadata
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    embedding_model: str = Field(..., description="Embedding model used")
    cross_encoder_model: Optional[str] = Field(None, description="Cross-encoder model used")
    weights_used: Dict[str, float] = Field(default_factory=dict, description="Weights used for scoring")
    generated_at: datetime = Field(default_factory=datetime.utcnow, description="Generation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "job_title": "Senior ML Engineer",
                "total_candidates_processed": 5000,
                "candidates_retrieved": 300,
                "candidates_reranked": 100,
                "final_recommendations": 100,
                "execution_time_ms": 2450.5,
                "embedding_model": "BAAI/bge-large-en-v1.5",
            }
        }


class RecommendationResponse(BaseModel):
    """
    API response for recommendation endpoint.
    """

    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    result: Optional[RecommendationResult] = Field(None, description="Recommendation result")
    error: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully generated 100 recommendations",
                "result": {
                    "job_id": "job_001",
                    "final_recommendations": 100,
                },
            }
        }
