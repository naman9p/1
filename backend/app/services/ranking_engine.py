"""
Ranking Engine

Handles candidate re-ranking using cross-encoder models.
"""

from typing import List, Optional, Dict, Any, Tuple
import time

from loguru import logger

from ..core.config import settings, Settings
from ..core.logging_config import get_logger
from ..schemas.job import ParsedJob
from ..schemas.candidate import Candidate
from ..schemas.scoring import ScoringResult, ScoringFeatures


class RankingEngine:
    """
    Engine for re-ranking candidates using cross-encoder models.

    Features:
    - Cross-encoder based re-ranking
    - Score combination with hybrid scores
    - Batch processing for efficiency
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the ranking engine.

        Args:
            settings: Application settings
        """
        self.settings = settings or settings
        self._cross_encoder = None
        self._model_name = None
        self.logger = get_logger(__name__)

    def load_model(self) -> bool:
        """
        Load the cross-encoder model.

        Returns:
            True if model loaded successfully
        """
        try:
            from sentence_transformers import CrossEncoder

            self.logger.info(f"Loading cross-encoder model: {self.settings.cross_encoder_model}")
            start_time = time.time()

            self._cross_encoder = CrossEncoder(self.settings.cross_encoder_model)
            self._model_name = self.settings.cross_encoder_model

            elapsed = time.time() - start_time
            self.logger.info(f"Cross-encoder loaded in {elapsed:.2f}s")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load cross-encoder: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._cross_encoder is not None

    @property
    def model_name(self) -> Optional[str]:
        """Get model name."""
        return self._model_name

    def rerank(
        self,
        job: ParsedJob,
        candidates: List[Tuple[Candidate, ScoringResult]],
        top_k: int,
    ) -> List[Tuple[Candidate, ScoringResult, float]]:
        """
        Re-rank candidates using cross-encoder.

        Args:
            job: Parsed job description
            candidates: List of (candidate, scoring_result) tuples
            top_k: Number of top candidates to return

        Returns:
            List of (candidate, scoring_result, cross_encoder_score) tuples
        """
        if not candidates:
            return []

        if not self.is_loaded:
            self.logger.warning("Cross-encoder not loaded, using hybrid scores only")
            return [(c, s, s.hybrid_score) for c, s in candidates[:top_k]]

        self.logger.info(f"Re-ranking {len(candidates)} candidates")
        start_time = time.time()

        # Prepare pairs for cross-encoder
        pairs = []
        for candidate, _ in candidates:
            query = job.processed_document
            passage = candidate.document.full_document
            pairs.append([query, passage])

        # Get cross-encoder scores in batches
        batch_size = 32
        cross_scores = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            batch_scores = self._cross_encoder.predict(batch)
            cross_scores.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else list(batch_scores))

        # Normalize cross-encoder scores to 0-1 range
        if cross_scores:
            min_score = min(cross_scores)
            max_score = max(cross_scores)
            if max_score > min_score:
                cross_scores = [(s - min_score) / (max_score - min_score) for s in cross_scores]
            else:
                cross_scores = [0.5] * len(cross_scores)

        # Combine with hybrid scores
        results = []
        for i, (candidate, scoring_result) in enumerate(candidates):
            cross_score = cross_scores[i] if i < len(cross_scores) else 0.5

            # Update scoring result with cross-encoder score
            scoring_result.rerank_score = cross_score

            # Combine scores (weighted average)
            # Give more weight to cross-encoder for final ranking
            combined_score = (scoring_result.hybrid_score * 0.4 + cross_score * 0.6)
            scoring_result.final_score = combined_score

            results.append((candidate, scoring_result, cross_score))

        # Sort by combined score
        results.sort(key=lambda x: x[2], reverse=True)

        # Take top_k
        top_results = results[:top_k]

        elapsed = time.time() - start_time
        self.logger.info(f"Re-ranking completed in {elapsed:.2f}s")

        return top_results

    def generate_reasoning(
        self,
        candidate: Candidate,
        job: ParsedJob,
        features: ScoringFeatures,
        final_score: float,
    ) -> str:
        """
        Generate human-readable reasoning for candidate ranking.

        Args:
            candidate: Candidate profile
            job: Parsed job description
            features: Scoring features
            final_score: Final combined score

        Returns:
            Reasoning string
        """
        reasons = []

        # Semantic alignment
        if features.semantic_similarity >= 0.8:
            reasons.append("Excellent semantic alignment with job requirements")
        elif features.semantic_similarity >= 0.6:
            reasons.append("Good semantic alignment with job requirements")

        # Skill match
        if features.required_skills_match >= 0.8:
            reasons.append(f"Strong match on required skills ({features.required_skills_match:.0%})")
        elif features.required_skills_match >= 0.5:
            reasons.append(f"Moderate match on required skills ({features.required_skills_match:.0%})")

        # Experience
        if features.years_experience_match >= 0.8:
            reasons.append("Meets or exceeds experience requirements")
        elif features.years_experience_match >= 0.5:
            reasons.append("Partially meets experience requirements")

        # ML/AI expertise
        if features.ml_experience >= 0.7:
            reasons.append("Strong ML/AI background")
        if features.llm_experience >= 0.5:
            reasons.append("LLM/Generative AI experience")

        # Technical skills
        if features.vector_search_experience >= 0.5:
            reasons.append("Vector search/embedding experience")
        if features.fastapi_experience >= 0.5:
            reasons.append("FastAPI/backend development experience")
        if features.python_expertise >= 0.7:
            reasons.append("Strong Python expertise")

        # Behavioral
        if features.behavior_score >= 0.7:
            reasons.append("High recruiter engagement and platform activity")
        elif features.behavior_score >= 0.5:
            reasons.append("Good platform engagement")

        # Industry
        if features.industry_match >= 0.8:
            reasons.append(f"Relevant industry experience ({job.industry or 'tech'})")

        # Leadership
        if features.leadership_score >= 0.8:
            reasons.append("Leadership experience")

        # Startup
        if features.startup_experience >= 0.8:
            reasons.append("Startup experience")

        # Open source
        if features.open_source_contribution >= 0.7:
            reasons.append("Active open source contributor")

        # Project quality
        if features.project_quality >= 0.7:
            reasons.append("High-quality project portfolio")

        # Combine reasons
        if not reasons:
            reasons.append("Moderate overall fit for the position")

        reasoning = ". ".join(reasons) + "."

        # Add score context
        if final_score >= 0.8:
            reasoning = f"Highly recommended. {reasoning}"
        elif final_score >= 0.6:
            reasoning = f"Recommended. {reasoning}"
        else:
            reasoning = f"Potential fit. {reasoning}"

        return reasoning


# Global instance
_ranking_engine: Optional[RankingEngine] = None


def get_ranking_engine(settings: Optional[Settings] = None) -> RankingEngine:
    """
    Get or create the ranking engine singleton.

    Args:
        settings: Optional settings override

    Returns:
        RankingEngine instance
    """
    global _ranking_engine

    if _ranking_engine is None:
        _ranking_engine = RankingEngine(settings)

    return _ranking_engine
