"""
API Response Schemas

Pydantic models for API responses.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    components: Dict[str, str] = Field(default_factory=dict, description="Component health status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "components": {
                    "database": "healthy",
                    "embedding_model": "loaded",
                    "vector_store": "connected",
                },
            }
        }


class MetricsResponse(BaseModel):
    """
    System metrics response.
    """

    # System info
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")

    # Database metrics
    total_candidates: int = Field(..., description="Total candidates indexed")
    total_jobs: int = Field(..., description="Total jobs processed")
    vector_store_size: int = Field(..., description="Number of vectors in store")

    # Model info
    embedding_model: str = Field(..., description="Current embedding model")
    cross_encoder_model: Optional[str] = Field(None, description="Current cross-encoder model")
    model_loaded: bool = Field(..., description="Whether models are loaded")

    # Performance metrics
    avg_recommendation_time_ms: Optional[float] = Field(None, description="Average recommendation time")
    total_recommendations_generated: int = Field(default=0, description="Total recommendations generated")

    # Configuration
    weights: Dict[str, float] = Field(default_factory=dict, description="Current scoring weights")
    retrieval_top_k: int = Field(..., description="Retrieval top-k setting")
    rerank_top_k: int = Field(..., description="Re-ranking top-k setting")

    class Config:
        json_schema_extra = {
            "example": {
                "version": "1.0.0",
                "uptime_seconds": 3600.5,
                "total_candidates": 5000,
                "total_jobs": 10,
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "model_loaded": True,
                "avg_recommendation_time_ms": 2450.5,
            }
        }


class UploadJobResponse(BaseModel):
    """
    Response for job upload endpoint.
    """

    success: bool = Field(..., description="Whether upload was successful")
    message: str = Field(..., description="Response message")
    job_id: str = Field(..., description="Job identifier")
    job_title: Optional[str] = Field(None, description="Job title")
    parsed: bool = Field(..., description="Whether job was parsed successfully")
    embedding_generated: bool = Field(..., description="Whether embedding was generated")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Job uploaded and processed successfully",
                "job_id": "job_001",
                "job_title": "Senior ML Engineer",
                "parsed": True,
                "embedding_generated": True,
            }
        }


class IndexCandidatesResponse(BaseModel):
    """
    Response for candidate indexing endpoint.
    """

    success: bool = Field(..., description="Whether indexing was successful")
    message: str = Field(..., description="Response message")
    total_processed: int = Field(..., description="Total candidates processed")
    total_indexed: int = Field(..., description="Total candidates indexed")
    total_failed: int = Field(..., description="Total candidates failed")
    embedding_model: str = Field(..., description="Embedding model used")
    execution_time_seconds: float = Field(..., description="Total execution time")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Successfully indexed 5000 candidates",
                "total_processed": 5000,
                "total_indexed": 4998,
                "total_failed": 2,
                "embedding_model": "BAAI/bge-large-en-v1.5",
                "execution_time_seconds": 125.5,
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response.
    """

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "error": "Job not found",
                "error_code": "JOB_NOT_FOUND",
                "details": {"job_id": "job_001"},
            }
        }


class CandidateResponse(BaseModel):
    """
    Response for getting a specific candidate.
    """

    success: bool = Field(..., description="Whether request was successful")
    candidate_id: str = Field(..., description="Candidate identifier")
    name: Optional[str] = Field(None, description="Candidate name")
    headline: Optional[str] = Field(None, description="Candidate headline")
    location: Optional[str] = Field(None, description="Location")
    total_years_experience: Optional[float] = Field(None, description="Years of experience")
    skills: List[str] = Field(default_factory=list, description="Top skills")
    current_company: Optional[str] = Field(None, description="Current company")
    current_title: Optional[str] = Field(None, description="Current title")
    education_summary: Optional[str] = Field(None, description="Education summary")
    behavioral_score: Optional[float] = Field(None, description="Behavioral score")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "candidate_id": "cand_001",
                "name": "John Doe",
                "headline": "Senior ML Engineer",
                "skills": ["Python", "PyTorch", "Machine Learning"],
                "total_years_experience": 7,
            }
        }
