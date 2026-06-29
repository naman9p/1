"""
Candidate Processor Service

Processes candidate profiles into searchable documents.
"""

from typing import List, Optional, Dict, Any
import re
from datetime import datetime

from loguru import logger

from ..core.config import settings, Settings
from ..core.logging_config import get_logger
from ..schemas.candidate import (
    CandidateProfile,
    CandidateDocument,
    WorkExperience,
    Education,
    Project,
)


class CandidateProcessor:
    """
    Service for processing candidate profiles into searchable documents.

    Features:
    - Document generation for embedding
    - Feature extraction (seniority, skills, etc.)
    - Skill categorization
    - Experience calculation
    """

    # Skill category mappings
    ML_SKILLS = {
        "pytorch",
        "tensorflow",
        "keras",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "hugging face",
        "transformers",
        "machine learning",
        "deep learning",
        "neural networks",
        "nlp",
        "computer vision",
        "reinforcement learning",
        "mlops",
        "model deployment",
    }

    BACKEND_SKILLS = {
        "python",
        "java",
        "go",
        "rust",
        "node.js",
        "fastapi",
        "django",
        "flask",
        "spring",
        "express",
        "graphql",
        "rest api",
        "microservices",
        "api design",
        "backend development",
    }

    DATA_SKILLS = {
        "pandas",
        "numpy",
        "spark",
        "kafka",
        "airflow",
        "sql",
        "postgresql",
        "mongodb",
        "redis",
        "etl",
        "data pipeline",
        "data engineering",
        "data warehouse",
    }

    CLOUD_SKILLS = {
        "aws",
        "azure",
        "gcp",
        "google cloud",
        "kubernetes",
        "docker",
        "terraform",
        "cloud infrastructure",
        "devops",
        "ci/cd",
    }

    VECTOR_SEARCH_SKILLS = {
        "chromadb",
        "pinecone",
        "weaviate",
        "milvus",
        "qdrant",
        "vector database",
        "vector search",
        "embedding",
        "similarity search",
        "ann",
        "approximate nearest neighbor",
    }

    LLM_SKILLS = {
        "llm",
        "large language model",
        "gpt",
        "bert",
        "rag",
        "retrieval augmented generation",
        "langchain",
        "llama",
        "generative ai",
        "prompt engineering",
    }

    # Seniority keywords in titles
    SENIORITY_KEYWORDS = {
        "entry": ["entry", "junior", "associate", "intern"],
        "mid": ["mid", "middle", "ii", "2"],
        "senior": ["senior", "sr", "iii", "3"],
        "staff": ["staff", "principal", "iv", "4"],
        "lead": ["lead", "tech lead", "engineering lead"],
        "executive": ["head", "director", "vp", "cto", "ceo", "chief"],
    }

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the candidate processor.

        Args:
            settings: Application settings
        """
        self.settings = settings or settings
        self.logger = get_logger(__name__)

    def process(self, profile: CandidateProfile) -> CandidateDocument:
        """
        Process a candidate profile into a searchable document.

        Args:
            profile: Raw candidate profile

        Returns:
            Processed candidate document
        """
        self.logger.debug(f"Processing candidate: {profile.candidate_id}")

        # Generate document sections
        title_document = self._generate_title_document(profile)
        skills_document = self._generate_skills_document(profile)
        experience_document = self._generate_experience_document(profile)
        education_document = self._generate_education_document(profile)
        projects_document = self._generate_projects_document(profile)

        # Combine into full document
        full_document = " | ".join([
            title_document,
            skills_document,
            experience_document,
            education_document,
            projects_document,
        ])

        # Extract features
        seniority_level = self._infer_seniority(profile)
        industry_domains = self._infer_industries(profile)
        leadership_experience = self._has_leadership_experience(profile)
        startup_experience = self._has_startup_experience(profile)

        # Categorize skills
        ml_skills = self._categorize_skills(profile.skills, self.ML_SKILLS)
        backend_skills = self._categorize_skills(profile.skills, self.BACKEND_SKILLS)
        data_skills = self._categorize_skills(profile.skills, self.DATA_SKILLS)
        cloud_skills = self._categorize_skills(profile.skills, self.CLOUD_SKILLS)

        return CandidateDocument(
            candidate_id=profile.candidate_id,
            title_document=title_document,
            skills_document=skills_document,
            experience_document=experience_document,
            education_document=education_document,
            projects_document=projects_document,
            full_document=full_document,
            seniority_level=seniority_level,
            industry_domains=industry_domains,
            leadership_experience=leadership_experience,
            startup_experience=startup_experience,
            ml_skills=ml_skills,
            backend_skills=backend_skills,
            data_skills=data_skills,
            cloud_skills=cloud_skills,
        )

    def _generate_title_document(self, profile: CandidateProfile) -> str:
        """Generate title/headline document."""
        parts = []

        if profile.headline:
            parts.append(f"Headline: {profile.headline}")

        if profile.summary:
            parts.append(f"Summary: {profile.summary}")

        if profile.location:
            parts.append(f"Location: {profile.location}")

        if profile.work_experience:
            current = next(
                (exp for exp in profile.work_experience if exp.is_current), None
            )
            if current:
                parts.append(f"Current: {current.title} at {current.company}")

        return " | ".join(parts) if parts else "No title information"

    def _generate_skills_document(self, profile: CandidateProfile) -> str:
        """Generate skills document."""
        if profile.skills:
            return f"Skills: {', '.join(profile.skills)}"
        return "Skills: Not specified"

    def _generate_experience_document(self, profile: CandidateProfile) -> str:
        """Generate experience document."""
        if not profile.work_experience:
            return "Experience: Not specified"

        parts = []
        for exp in profile.work_experience[:5]:  # Top 5 experiences
            duration = ""
            if exp.duration_months:
                years = exp.duration_months / 12
                duration = f" ({years:.1f} years)"

            desc = ""
            if exp.description:
                desc = f" - {exp.description[:100]}"

            parts.append(f"{exp.title} at {exp.company}{duration}{desc}")

        if profile.total_years_experience:
            parts.insert(0, f"Total Experience: {profile.total_years_experience} years")

        return "Experience: " + " | ".join(parts)

    def _generate_education_document(self, profile: CandidateProfile) -> str:
        """Generate education document."""
        if not profile.education:
            return "Education: Not specified"

        parts = []
        for edu in profile.education[:3]:  # Top 3 education entries
            field = f" in {edu.field_of_study}" if edu.field_of_study else ""
            year = f" ({edu.graduation_year})" if edu.graduation_year else ""
            parts.append(f"{edu.degree}{field} from {edu.institution}{year}")

        return "Education: " + " | ".join(parts)

    def _generate_projects_document(self, profile: CandidateProfile) -> str:
        """Generate projects document."""
        if not profile.projects:
            return "Projects: Not specified"

        parts = []
        for proj in profile.projects[:5]:  # Top 5 projects
            tech = f" ({', '.join(proj.technologies)})" if proj.technologies else ""
           开源 = " [Open Source]" if proj.is_open_source else ""
            desc = f" - {proj.description[:50]}" if proj.description else ""
            parts.append(f"{proj.name}{tech}{开源}{desc}")

        return "Projects: " + " | ".join(parts)

    def _infer_seniority(self, profile: CandidateProfile) -> Optional[str]:
        """Infer seniority level from profile."""
        # Check headline
        if profile.headline:
            headline_lower = profile.headline.lower()
            for level, keywords in self.SENIORITY_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in headline_lower:
                        return level

        # Check work experience titles
        for exp in profile.work_experience:
            title_lower = exp.title.lower()
            for level, keywords in self.SENIORITY_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in title_lower:
                        return level

        # Infer from years of experience
        if profile.total_years_experience:
            years = profile.total_years_experience
            if years < 2:
                return "entry"
            elif years < 5:
                return "mid"
            elif years < 8:
                return "senior"
            elif years < 12:
                return "staff"
            else:
                return "executive"

        return None

    def _infer_industries(self, profile: CandidateProfile) -> List[str]:
        """Infer industry domains from profile."""
        industries = []

        # Check company names and descriptions
        industry_keywords = {
            "technology": ["tech", "software", "saas", "platform", "startup"],
            "finance": ["fintech", "finance", "banking", "trading", "investment"],
            "healthcare": ["health", "medical", "healthcare", "pharma", "biotech"],
            "ecommerce": ["ecommerce", "retail", "marketplace", "shopping"],
            "ai_ml": ["ai", "machine learning", "ml", "deep learning", "neural"],
            "data": ["data", "analytics", "big data", "business intelligence"],
            "automotive": ["automotive", "auto", "vehicle", "transportation"],
        }

        all_text = " ".join([
            profile.headline or "",
            profile.summary or "",
            " ".join([exp.company for exp in profile.work_experience]),
            " ".join([exp.description or "" for exp in profile.work_experience]),
        ]).lower()

        for industry, keywords in industry_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    industries.append(industry)
                    break

        return list(dict.fromkeys(industries))

    def _has_leadership_experience(self, profile: CandidateProfile) -> bool:
        """Check if candidate has leadership experience."""
        leadership_keywords = [
            "lead",
            "manager",
            "director",
            "head",
            "principal",
            "staff",
            "architect",
            "mentor",
            "team",
            "supervise",
            "manage",
        ]

        for exp in profile.work_experience:
            title_lower = exp.title.lower()
            desc_lower = (exp.description or "").lower()

            for keyword in leadership_keywords:
                if keyword in title_lower or keyword in desc_lower:
                    return True

        return False

    def _has_startup_experience(self, profile: CandidateProfile) -> bool:
        """Check if candidate has startup experience."""
        startup_keywords = ["startup", "venture", "seed", "series", "founder", "co-founder"]

        for exp in profile.work_experience:
            company_lower = exp.company.lower()
            desc_lower = (exp.description or "").lower()

            for keyword in startup_keywords:
                if keyword in company_lower or keyword in desc_lower:
                    return True

        return False

    def _categorize_skills(
        self, skills: List[str], category_set: set
    ) -> List[str]:
        """Categorize skills into a specific category."""
        matched = []
        skills_lower = [s.lower() for s in skills]

        for category_skill in category_set:
            for skill in skills_lower:
                if category_skill in skill or skill in category_skill:
                    # Get original case skill
                    original_idx = skills_lower.index(skill)
                    matched.append(skills[original_idx])
                    break

        return list(dict.fromkeys(matched))


# Global instance
_candidate_processor: Optional[CandidateProcessor] = None


def get_candidate_processor(settings: Optional[Settings] = None) -> CandidateProcessor:
    """
    Get or create the candidate processor singleton.

    Args:
        settings: Optional settings override

    Returns:
        CandidateProcessor instance
    """
    global _candidate_processor

    if _candidate_processor is None:
        _candidate_processor = CandidateProcessor(settings)

    return _candidate_processor
