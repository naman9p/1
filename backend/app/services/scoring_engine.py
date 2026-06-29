"""
Scoring Engine

Computes hybrid scores for candidates based on multiple factors.
"""

from typing import List, Optional, Dict, Any, Tuple
import re

from loguru import logger

from ..core.config import settings, Settings
from ..core.logging_config import get_logger
from ..schemas.job import ParsedJob
from ..schemas.candidate import Candidate, CandidateDocument, BehavioralSignals
from ..schemas.scoring import ScoringFeatures, ScoringResult, WeightConfig


class ScoringEngine:
    """
    Engine for computing hybrid candidate scores.

    Features:
    - Multi-factor scoring
    - Configurable weights
    - Skill matching with fuzzy comparison
    - Experience level matching
    - Industry alignment
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the scoring engine.

        Args:
            settings: Application settings
        """
        self.settings = settings or settings
        self.logger = get_logger(__name__)

    def compute_features(
        self,
        candidate: Candidate,
        job: ParsedJob,
        semantic_similarity: float,
    ) -> ScoringFeatures:
        """
        Compute individual scoring features for a candidate.

        Args:
            candidate: Candidate with profile and document
            job: Parsed job description
            semantic_similarity: Pre-computed semantic similarity score

        Returns:
            ScoringFeatures with all individual scores
        """
        features = ScoringFeatures()

        # Semantic similarity (from vector search)
        features.semantic_similarity = semantic_similarity

        # Skill matching
        features.skill_match = self._compute_skill_match(
            candidate.document, job
        )
        features.required_skills_match = self._compute_required_skills_match(
            candidate.document, job
        )
        features.preferred_skills_match = self._compute_preferred_skills_match(
            candidate.document, job
        )

        # Experience matching
        features.experience_match = self._compute_experience_match(
            candidate.document, job
        )
        features.years_experience_match = self._compute_years_experience_match(
            candidate.profile, job
        )
        features.seniority_match = self._compute_seniority_match(
            candidate.document, job
        )

        # Behavioral signals
        features.behavior_score = self._get_behavior_score(candidate.profile)

        # Industry match
        features.industry_match = self._compute_industry_match(
            candidate.document, job
        )

        # Education match
        features.education_match = self._compute_education_match(
            candidate.profile, job
        )

        # Bonus features
        features.leadership_score = 1.0 if candidate.document.leadership_experience else 0.0
        features.startup_experience = 1.0 if candidate.document.startup_experience else 0.0
        features.ml_experience = self._compute_ml_experience(candidate.document)
        features.llm_experience = self._compute_llm_experience(candidate.document)
        features.vector_search_experience = self._compute_vector_search_experience(
            candidate.document
        )
        features.fastapi_experience = self._compute_fastapi_experience(candidate.document)
        features.python_expertise = self._compute_python_expertise(candidate.document)
        features.open_source_contribution = self._compute_open_source_score(
            candidate.profile
        )
        features.project_quality = self._compute_project_quality(candidate.profile)
        features.employment_stability = self._compute_employment_stability(
            candidate.profile
        )
        features.recent_activity = self._compute_recent_activity(candidate.profile)

        return features

    def compute_hybrid_score(
        self,
        features: ScoringFeatures,
        weights: Optional[WeightConfig] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute weighted hybrid score from features.

        Args:
            features: Scoring features
            weights: Weight configuration (uses settings if not provided)

        Returns:
            Tuple of (hybrid_score, weighted_components)
        """
        if weights is None:
            weights = WeightConfig(
                semantic=self.settings.weight_semantic,
                skill=self.settings.weight_skill,
                experience=self.settings.weight_experience,
                behavior=self.settings.weight_behavior,
                industry=self.settings.weight_industry,
                education=self.settings.weight_education,
                bonus=self.settings.weight_bonus,
            )

        # Compute component scores
        skill_score = (features.required_skills_match * 0.7 +
                      features.preferred_skills_match * 0.3)

        experience_score = (features.experience_match * 0.4 +
                          features.years_experience_match * 0.3 +
                          features.seniority_match * 0.3)

        bonus_score = (
            features.ml_experience * 0.25 +
            features.llm_experience * 0.20 +
            features.vector_search_experience * 0.15 +
            features.fastapi_experience * 0.15 +
            features.python_expertise * 0.15 +
            features.leadership_score * 0.05 +
            features.startup_experience * 0.05
        )

        # Compute weighted hybrid score
        hybrid_score = (
            features.semantic_similarity * weights.semantic +
            skill_score * weights.skill +
            experience_score * weights.experience +
            features.behavior_score * weights.behavior +
            features.industry_match * weights.industry +
            features.education_match * weights.education +
            bonus_score * weights.bonus
        )

        weighted_components = {
            "weighted_semantic": features.semantic_similarity * weights.semantic,
            "weighted_skill": skill_score * weights.skill,
            "weighted_experience": experience_score * weights.experience,
            "weighted_behavior": features.behavior_score * weights.behavior,
            "weighted_industry": features.industry_match * weights.industry,
            "weighted_education": features.education_match * weights.education,
            "weighted_bonus": bonus_score * weights.bonus,
        }

        return max(0.0, min(1.0, hybrid_score)), weighted_components

    def _compute_skill_match(
        self,
        document: CandidateDocument,
        job: ParsedJob,
    ) -> float:
        """Compute overall skill match score."""
        all_job_skills = set(job.required_skills + job.preferred_skills)
        if not all_job_skills:
            return 0.5

        candidate_skills = set(
            s.lower() for s in document.skills_document.split(": ")[-1].split(", ")
            if ": " in document.skills_document
        )

        matches = 0
        for job_skill in all_job_skills:
            job_skill_lower = job_skill.lower()
            for cand_skill in candidate_skills:
                if self._fuzzy_match(job_skill_lower, cand_skill):
                    matches += 1
                    break

        return matches / len(all_job_skills)

    def _compute_required_skills_match(
        self,
        document: CandidateDocument,
        job: ParsedJob,
    ) -> float:
        """Compute required skills match score."""
        if not job.required_skills:
            return 0.5

        candidate_skills = set(
            s.lower() for s in document.skills_document.split(": ")[-1].split(", ")
            if ": " in document.skills_document
        )

        matches = 0
        for skill in job.required_skills:
            skill_lower = skill.lower()
            for cand_skill in candidate_skills:
                if self._fuzzy_match(skill_lower, cand_skill):
                    matches += 1
                    break

        return matches / len(job.required_skills)

    def _compute_preferred_skills_match(
        self,
        document: CandidateDocument,
        job: ParsedJob,
    ) -> float:
        """Compute preferred skills match score."""
        if not job.preferred_skills:
            return 0.5

        candidate_skills = set(
            s.lower() for s in document.skills_document.split(": ")[-1].split(", ")
            if ": " in document.skills_document
        )

        matches = 0
        for skill in job.preferred_skills:
            skill_lower = skill.lower()
            for cand_skill in candidate_skills:
                if self._fuzzy_match(skill_lower, cand_skill):
                    matches += 1
                    break

        return matches / len(job.preferred_skills)

    def _fuzzy_match(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """Check if two strings fuzzy match."""
        if str1 == str2:
            return True

        # Check substring match
        if str1 in str2 or str2 in str1:
            return True

        # Try using rapidfuzz if available
        try:
            from rapidfuzz import fuzz
            score = fuzz.ratio(str1, str2) / 100
            return score >= threshold
        except ImportError:
            # Fallback to simple comparison
            return False

    def _compute_experience_match(
        self,
        document: CandidateDocument,
        job: ParsedJob,
    ) -> float:
        """Compute experience match score."""
        # Extract years from experience document
        exp_text = document.experience_document.lower()

        # Check for experience keywords matching job seniority
        if job.seniority:
            if job.seniority in exp_text:
                return 1.0

        # Check for years mention
        years_match = re.search(r"(\d+)\s*years?", exp_text)
        if years_match and job.years_experience:
            candidate_years = int(years_match.group(1))
            if candidate_years >= job.years_experience:
                return 1.0
            elif candidate_years >= job.years_experience * 0.8:
                return 0.8
            else:
                return candidate_years / job.years_experience

        return 0.5

    def _compute_years_experience_match(
        self,
        profile,
        job: ParsedJob,
    ) -> float:
        """Compute years of experience match."""
        if not job.years_experience:
            return 0.5

        candidate_years = profile.total_years_experience or 0

        if candidate_years >= job.years_experience:
            return 1.0
        elif candidate_years >= job.years_experience * 0.8:
            return 0.8
        else:
            return max(0.0, candidate_years / job.years_experience)

    def _compute_seniority_match(
        self,
        document: CandidateDocument,
        job: ParsedJob,
    ) -> float:
        """Compute seniority level match."""
        if not job.seniority or not document.seniority_level:
            return 0.5

        seniority_order = ["entry", "mid", "senior", "staff", "executive"]

        job_idx = seniority_order.index(job.seniority) if job.seniority in seniority_order else 2
        cand_idx = seniority_order.index(document.seniority_level) if document.seniority_level in seniority_order else 2

        diff = abs(job_idx - cand_idx)

        if diff == 0:
            return 1.0
        elif diff == 1:
            return 0.8
        elif diff == 2:
            return 0.5
        else:
            return 0.2

    def _get_behavior_score(self, profile) -> float:
        """Get behavior score from profile."""
        if profile.behavioral_signals:
            return profile.behavioral_signals.behavior_score
        return 0.5

    def _compute_industry_match(
        self,
        document: CandidateDocument,
        job: ParsedJob,
    ) -> float:
        """Compute industry match score."""
        if not job.industry:
            return 0.5

        if not document.industry_domains:
            return 0.3

        if job.industry in document.industry_domains:
            return 1.0

        # Check for related industries
        related = {
            "ai_ml": ["technology", "data"],
            "technology": ["ai_ml", "data", "ecommerce"],
            "data": ["technology", "ai_ml", "finance"],
            "finance": ["technology", "data"],
        }

        if job.industry in related:
            for related_industry in related[job.industry]:
                if related_industry in document.industry_domains:
                    return 0.7

        return 0.3

    def _compute_education_match(self, profile, job: ParsedJob) -> float:
        """Compute education match score."""
        if not job.education:
            return 0.5

        if not profile.education:
            return 0.3

        education_levels = {
            "high school": 1,
            "bachelor": 2,
            "master": 3,
            "phd": 4,
        }

        job_level = 2  # Default to bachelor
        for level_str, level_num in education_levels.items():
            if level_str in job.education.lower():
                job_level = level_num
                break

        # Check candidate's highest education
        max_candidate_level = 0
        for edu in profile.education:
            degree_lower = edu.degree.lower()
            for level_str, level_num in education_levels.items():
                if level_str in degree_lower:
                    max_candidate_level = max(max_candidate_level, level_num)

        if max_candidate_level >= job_level:
            return 1.0
        elif max_candidate_level >= job_level - 1:
            return 0.7
        else:
            return 0.4

    def _compute_ml_experience(self, document: CandidateDocument) -> float:
        """Compute ML experience score."""
        if not document.ml_skills:
            return 0.0

        ml_keywords = ["pytorch", "tensorflow", "machine learning", "deep learning"]
        matches = sum(1 for skill in document.ml_skills
                     if any(kw in skill.lower() for kw in ml_keywords))

        return min(1.0, matches / 3)

    def _compute_llm_experience(self, document: CandidateDocument) -> float:
        """Compute LLM experience score."""
        llm_keywords = ["llm", "gpt", "bert", "transformers", "langchain", "rag"]
        all_skills = document.ml_skills + document.backend_skills

        matches = sum(1 for skill in all_skills
                     if any(kw in skill.lower() for kw in llm_keywords))

        return min(1.0, matches / 2)

    def _compute_vector_search_experience(self, document: CandidateDocument) -> float:
        """Compute vector search experience score."""
        vector_keywords = ["chromadb", "pinecone", "vector", "embedding", "similarity"]
        all_text = document.full_document.lower()

        matches = sum(1 for kw in vector_keywords if kw in all_text)

        return min(1.0, matches / 2)

    def _compute_fastapi_experience(self, document: CandidateDocument) -> float:
        """Compute FastAPI experience score."""
        all_text = document.full_document.lower()

        if "fastapi" in all_text:
            return 1.0
        elif "api" in all_text and ("python" in all_text or "backend" in all_text):
            return 0.5

        return 0.0

    def _compute_python_expertise(self, document: CandidateDocument) -> float:
        """Compute Python expertise score."""
        all_text = document.full_document.lower()

        if "python" not in all_text:
            return 0.0

        # Check for Python-related keywords
        python_keywords = ["django", "flask", "fastapi", "pandas", "numpy", "scikit"]
        matches = sum(1 for kw in python_keywords if kw in all_text)

        return min(1.0, 0.5 + (matches * 0.1))

    def _compute_open_source_score(self, profile) -> float:
        """Compute open source contribution score."""
        if not profile.projects:
            return 0.0

        open_source_projects = sum(1 for p in profile.projects if p.is_open_source)

        if open_source_projects == 0:
            return 0.0
        elif open_source_projects >= 3:
            return 1.0
        else:
            return open_source_projects / 3

    def _compute_project_quality(self, profile) -> float:
        """Compute project quality score."""
        if not profile.projects:
            return 0.3

        quality_factors = 0

        for project in profile.projects[:5]:
            if project.description and len(project.description) > 50:
                quality_factors += 0.2
            if project.technologies and len(project.technologies) > 2:
                quality_factors += 0.2
            if project.is_open_source:
                quality_factors += 0.2
            if project.url:
                quality_factors += 0.1

        return min(1.0, quality_factors)

    def _compute_employment_stability(self, profile) -> float:
        """Compute employment stability score."""
        if not profile.work_experience:
            return 0.5

        # Check for long tenures
        long_tenures = sum(1 for exp in profile.work_experience
                         if exp.duration_months and exp.duration_months > 24)

        if long_tenures >= 2:
            return 1.0
        elif long_tenures >= 1:
            return 0.7
        else:
            return 0.4

    def _compute_recent_activity(self, profile) -> float:
        """Compute recent activity score."""
        if not profile.work_experience:
            return 0.3

        # Check if has current position
        has_current = any(exp.is_current for exp in profile.work_experience)

        if has_current:
            return 1.0

        # Check for recent end date
        recent_exp = any(
            exp.end_date and "202" in exp.end_date
            for exp in profile.work_experience
        )

        if recent_exp:
            return 0.7

        return 0.4


# Global instance
_scoring_engine: Optional[ScoringEngine] = None


def get_scoring_engine(settings: Optional[Settings] = None) -> ScoringEngine:
    """
    Get or create the scoring engine singleton.

    Args:
        settings: Optional settings override

    Returns:
        ScoringEngine instance
    """
    global _scoring_engine

    if _scoring_engine is None:
        _scoring_engine = ScoringEngine(settings)

    return _scoring_engine
