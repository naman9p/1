"""
Candidate Profile Schemas

Pydantic models for candidate profile representation and processing.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class WorkExperience(BaseModel):
    """
    Work experience entry.
    """

    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    start_date: Optional[str] = Field(None, description="Start date")
    end_date: Optional[str] = Field(None, description="End date")
    duration_months: Optional[int] = Field(None, description="Duration in months")
    description: Optional[str] = Field(None, description="Job description")
    is_current: bool = Field(default=False, description="Whether this is current position")


class Education(BaseModel):
    """
    Education entry.
    """

    institution: str = Field(..., description="Institution name")
    degree: str = Field(..., description="Degree type")
    field_of_study: Optional[str] = Field(None, description="Field of study")
    graduation_year: Optional[int] = Field(None, description="Graduation year")


class Certification(BaseModel):
    """
    Certification entry.
    """

    name: str = Field(..., description="Certification name")
    issuing_organization: Optional[str] = Field(None, description="Issuing organization")
    issue_date: Optional[str] = Field(None, description="Issue date")
    expiry_date: Optional[str] = Field(None, description="Expiry date")


class Project(BaseModel):
    """
    Project entry.
    """

    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(None, description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used")
    role: Optional[str] = Field(None, description="Role in project")
    is_open_source: bool = Field(default=False, description="Whether project is open source")
    url: Optional[str] = Field(None, description="Project URL")


class BehavioralSignals(BaseModel):
    """
    Behavioral signals from recruiter platform interactions.
    Normalized scores between 0 and 1.
    """

    # Engagement metrics
    recruiter_response_rate: float = Field(default=0.0, ge=0, le=1, description="Response rate to recruiters")
    profile_completeness: float = Field(default=0.0, ge=0, le=1, description="Profile completeness score")
    activity_score: float = Field(default=0.0, ge=0, le=1, description="Platform activity score")

    # Interaction metrics
    hiring_interaction: float = Field(default=0.0, ge=0, le=1, description="Previous hiring interactions")
    message_acceptance: float = Field(default=0.0, ge=0, le=1, description="Message acceptance rate")
    application_quality: float = Field(default=0.0, ge=0, le=1, description="Application quality score")

    # Trust metrics
    recency_score: float = Field(default=0.0, ge=0, le=1, description="Recency of activity")
    consistency_score: float = Field(default=0.0, ge=0, le=1, description="Profile consistency score")
    trust_score: float = Field(default=0.0, ge=0, le=1, description="Overall trust score")

    # Computed behavior score
    behavior_score: float = Field(default=0.0, ge=0, le=1, description="Computed behavior score")

    class Config:
        json_schema_extra = {
            "example": {
                "recruiter_response_rate": 0.85,
                "profile_completeness": 0.92,
                "activity_score": 0.78,
                "hiring_interaction": 0.65,
                "message_acceptance": 0.88,
                "application_quality": 0.90,
                "recency_score": 0.95,
                "consistency_score": 0.87,
                "trust_score": 0.85,
                "behavior_score": 0.85,
            }
        }


class CandidateProfile(BaseModel):
    """
    Raw candidate profile from JSONL input.
    """

    candidate_id: str = Field(..., description="Unique candidate identifier")
    name: Optional[str] = Field(None, description="Candidate name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location")
    headline: Optional[str] = Field(None, description="Professional headline")
    summary: Optional[str] = Field(None, description="Professional summary")

    # Experience
    work_experience: List[WorkExperience] = Field(default_factory=list, description="Work history")
    total_years_experience: Optional[float] = Field(None, description="Total years of experience")

    # Skills
    skills: List[str] = Field(default_factory=list, description="Technical skills")
    endorsements: Optional[Dict[str, int]] = Field(None, description="Skill endorsements count")

    # Education
    education: List[Education] = Field(default_factory=list, description="Education history")

    # Certifications
    certifications: List[Certification] = Field(default_factory=list, description="Certifications")

    # Projects
    projects: List[Project] = Field(default_factory=list, description="Projects")

    # Additional info
    languages: List[str] = Field(default_factory=list, description="Languages")
    publications: List[Dict[str, Any]] = Field(default_factory=list, description="Publications")
    patents: List[Dict[str, Any]] = Field(default_factory=list, description="Patents")

    # Behavioral signals
    behavioral_signals: Optional[BehavioralSignals] = Field(None, description="Behavioral signals")

    # Metadata
    last_updated: Optional[datetime] = Field(None, description="Last profile update")
    source: Optional[str] = Field(None, description="Profile source")

    class Config:
        json_schema_extra = {
            "example": {
                "candidate_id": "cand_001",
                "name": "John Doe",
                "headline": "Senior ML Engineer",
                "skills": ["Python", "PyTorch", "Machine Learning", "Deep Learning"],
                "total_years_experience": 7,
            }
        }


class CandidateDocument(BaseModel):
    """
    Processed candidate document for embedding and search.
    Combines all relevant information into a searchable text.
    """

    candidate_id: str = Field(..., description="Unique candidate identifier")

    # Processed text fields
    title_document: str = Field(..., description="Title and headline document")
    skills_document: str = Field(..., description="Skills document")
    experience_document: str = Field(..., description="Experience document")
    education_document: str = Field(..., description="Education document")
    projects_document: str = Field(..., description="Projects document")

    # Combined document for embedding
    full_document: str = Field(..., description="Combined document for semantic search")

    # Extracted features
    seniority_level: Optional[str] = Field(None, description="Inferred seniority level")
    industry_domains: List[str] = Field(default_factory=list, description="Industry domains")
    leadership_experience: bool = Field(default=False, description="Has leadership experience")
    startup_experience: bool = Field(default=False, description="Has startup experience")

    # Skill categories
    ml_skills: List[str] = Field(default_factory=list, description="ML/AI skills")
    backend_skills: List[str] = Field(default_factory=list, description="Backend development skills")
    data_skills: List[str] = Field(default_factory=list, description="Data engineering skills")
    cloud_skills: List[str] = Field(default_factory=list, description="Cloud platform skills")

    class Config:
        json_schema_extra = {
            "example": {
                "candidate_id": "cand_001",
                "full_document": "Senior ML Engineer with 7 years experience...",
                "seniority_level": "senior",
                "ml_skills": ["PyTorch", "TensorFlow", "Transformers"],
            }
        }


class Candidate(BaseModel):
    """
    Complete candidate with embedding and processed data.
    """

    profile: CandidateProfile = Field(..., description="Original candidate profile")
    document: CandidateDocument = Field(..., description="Processed candidate document")
    embedding: Optional[List[float]] = Field(None, description="Semantic embedding vector")
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")
    indexed: bool = Field(default=False, description="Whether candidate is indexed in vector DB")

    class Config:
        arbitrary_types_allowed = True
