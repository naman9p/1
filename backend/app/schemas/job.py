"""
Job Description Schemas

Pydantic models for job description parsing and representation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class JobDescriptionInput(BaseModel):
    """
    Input schema for raw job description text.
    """

    job_id: str = Field(..., description="Unique identifier for the job")
    title: str = Field(..., description="Job title")
    description: str = Field(..., description="Full job description text")
    company: Optional[str] = Field(None, description="Company name")
    location: Optional[str] = Field(None, description="Job location")
    job_type: Optional[str] = Field(None, description="Job type (full-time, contract, etc.)")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "title": "Senior Machine Learning Engineer",
                "description": "We are seeking a Senior ML Engineer...",
                "company": "TechCorp",
                "location": "San Francisco, CA",
                "job_type": "full-time",
            }
        }


class ParsedJob(BaseModel):
    """
    Structured representation of a parsed job description.
    Contains extracted entities and requirements.
    """

    job_id: str = Field(..., description="Unique identifier for the job")
    title: str = Field(..., description="Job title")
    company: Optional[str] = Field(None, description="Company name")
    location: Optional[str] = Field(None, description="Job location")

    # Extracted requirements
    required_skills: List[str] = Field(default_factory=list, description="Required technical skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred technical skills")
    responsibilities: List[str] = Field(default_factory=list, description="Key responsibilities")

    # Job attributes
    industry: Optional[str] = Field(None, description="Industry domain")
    seniority: Optional[str] = Field(None, description="Seniority level")
    years_experience: Optional[int] = Field(None, description="Required years of experience")
    education: Optional[str] = Field(None, description="Education requirements")

    # Keywords and technologies
    keywords: List[str] = Field(default_factory=list, description="Important keywords")
    technologies: List[str] = Field(default_factory=list, description="Required technologies")

    # Behavioral expectations
    behavioral_expectations: List[str] = Field(default_factory=list, description="Behavioral expectations")

    # Raw content for semantic matching
    raw_description: str = Field(..., description="Original job description text")
    processed_document: str = Field(..., description="Processed document for embedding")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_001",
                "title": "Senior Machine Learning Engineer",
                "required_skills": ["Python", "PyTorch", "Machine Learning", "Deep Learning"],
                "preferred_skills": ["FastAPI", "Vector Databases", "MLOps"],
                "seniority": "senior",
                "years_experience": 5,
                "technologies": ["Python", "PyTorch", "TensorFlow", "FastAPI"],
            }
        }


class JobDescription(BaseModel):
    """
    Complete job description with embedding.
    """

    parsed_job: ParsedJob = Field(..., description="Parsed job information")
    embedding: List[float] = Field(..., description="Semantic embedding vector")
    embedding_model: str = Field(..., description="Model used for embedding")

    class Config:
        arbitrary_types_allowed = True
