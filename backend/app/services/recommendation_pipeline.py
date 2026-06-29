"""
Recommendation Pipeline

Orchestrates the complete recommendation workflow.
"""

from typing import List, Optional, Dict, Any, Tuple
import time
import csv
from pathlib import Path
from datetime import datetime

from loguru import logger

from ..core.config import settings, Settings
from ..core.logging_config import get_logger
from ..schemas.job import JobDescriptionInput, ParsedJob, JobDescription
from ..schemas.candidate import Candidate, CandidateProfile, CandidateDocument
from ..schemas.recommendation import (
    RecommendationRequest,
    RecommendationResult,
    RankedCandidate,
)
from ..schemas.scoring import ScoringResult, ScoringFeatures, WeightConfig

from .embedding_service import EmbeddingService, get_embedding_service
from .vector_store import VectorStoreService, get_vector_store
from .job_parser import JobParserService, get_job_parser
from .candidate_processor import CandidateProcessor, get_candidate_processor
from .behavioral_parser import BehavioralSignalParser, get_behavioral_parser
from .scoring_engine import ScoringEngine, get_scoring_engine
from .ranking_engine import RankingEngine, get_ranking_engine


class RecommendationPipeline:
    """
    Main pipeline for candidate recommendations.

    Orchestrates:
    - Job parsing and embedding
    - Candidate indexing
    - Semantic retrieval
    - Hybrid scoring
    - Cross-encoder re-ranking
    - CSV output generation
    """

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the recommendation pipeline.

        Args:
            settings: Application settings
        """
        self.settings = settings or settings
        self.logger = get_logger(__name__)

        # Initialize services
        self.embedding_service = get_embedding_service(settings)
        self.vector_store = get_vector_store(settings)
        self.job_parser = get_job_parser(settings)
        self.candidate_processor = get_candidate_processor(settings)
        self.behavioral_parser = get_behavioral_parser(settings)
        self.scoring_engine = get_scoring_engine(settings)
        self.ranking_engine = get_ranking_engine(settings)

        # State
        self._jobs: Dict[str, JobDescription] = {}
        self._candidates: Dict[str, Candidate] = {}
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize all services and load models.

        Returns:
            True if initialization successful
        """
        self.logger.info("Initializing recommendation pipeline...")
        start_time = time.time()

        # Load embedding models
        if not self.embedding_service.load_models():
            self.logger.error("Failed to load embedding models")
            return False

        # Initialize vector store
        if not self.vector_store.initialize():
            self.logger.error("Failed to initialize vector store")
            return False

        # Load cross-encoder
        if not self.ranking_engine.load_model():
            self.logger.warning("Cross-encoder not loaded, will use hybrid scoring only")

        self._initialized = True

        elapsed = time.time() - start_time
        self.logger.info(f"Pipeline initialized in {elapsed:.2f}s")

        return True

    @property
    def is_initialized(self) -> bool:
        """Check if pipeline is initialized."""
        return self._initialized

    def process_job(self, job_input: JobDescriptionInput) -> JobDescription:
        """
        Process a job description.

        Args:
            job_input: Raw job description input

        Returns:
            JobDescription with embedding
        """
        self.logger.info(f"Processing job: {job_input.job_id}")
        start_time = time.time()

        # Parse job
        parsed_job = self.job_parser.parse(job_input)

        # Generate embedding
        embedding = self.embedding_service.generate_embedding(
            parsed_job.processed_document
        )

        # Create job description object
        job_description = JobDescription(
            parsed_job=parsed_job,
            embedding=embedding,
            embedding_model=self.embedding_service.model_name or "unknown",
        )

        # Store job
        self._jobs[job_input.job_id] = job_description

        elapsed = time.time() - start_time
        self.logger.info(f"Job processed in {elapsed:.2f}s")

        return job_description

    def index_candidates(
        self,
        candidates: List[CandidateProfile],
        batch_size: int = 100,
    ) -> Tuple[int, int]:
        """
        Index candidate profiles.

        Args:
            candidates: List of candidate profiles
            batch_size: Batch size for indexing

        Returns:
            Tuple of (indexed_count, failed_count)
        """
        self.logger.info(f"Indexing {len(candidates)} candidates")
        start_time = time.time()

        indexed = 0
        failed = 0

        # Process and embed candidates
        candidates_to_index = []

        for i, profile in enumerate(candidates):
            try:
                # Process profile
                document = self.candidate_processor.process(profile)

                # Parse behavioral signals
                behavioral_signals = self.behavioral_parser.parse(profile)
                profile.behavioral_signals = behavioral_signals

                # Generate embedding
                embedding = self.embedding_service.generate_embedding(
                    document.full_document
                )

                # Create candidate object
                candidate = Candidate(
                    profile=profile,
                    document=document,
                    embedding=embedding,
                    embedding_model=self.embedding_service.model_name,
                    indexed=True,
                )

                # Store candidate
                self._candidates[profile.candidate_id] = candidate

                # Prepare for vector store
                metadata = {
                    "seniority_level": document.seniority_level,
                    "total_years_experience": profile.total_years_experience,
                    "behavior_score": behavioral_signals.behavior_score,
                }

                candidates_to_index.append((
                    profile.candidate_id,
                    embedding,
                    document,
                    metadata,
                ))

                indexed += 1

                if (i + 1) % 500 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(candidates)} candidates")

            except Exception as e:
                self.logger.error(f"Failed to process candidate {profile.candidate_id}: {e}")
                failed += 1

        # Index in vector store
        if candidates_to_index:
            vector_indexed, vector_failed = self.vector_store.upsert_candidates_batch(
                candidates_to_index,
                batch_size=batch_size,
            )
            indexed = vector_indexed
            failed = vector_failed

        elapsed = time.time() - start_time
        self.logger.info(
            f"Indexing completed: {indexed} indexed, {failed} failed in {elapsed:.2f}s"
        )

        return indexed, failed

    def recommend(
        self,
        request: RecommendationRequest,
    ) -> RecommendationResult:
        """
        Generate recommendations for a job.

        Args:
            request: Recommendation request

        Returns:
            Recommendation result with ranked candidates
        """
        self.logger.info(f"Generating recommendations for job: {request.job_id}")
        start_time = time.time()

        # Get job
        if request.job_id not in self._jobs:
            raise ValueError(f"Job not found: {request.job_id}")

        job_description = self._jobs[request.job_id]
        job = job_description.parsed_job

        # Step 1: Semantic retrieval
        self.logger.info("Step 1: Semantic retrieval")
        retrieval_results = self.vector_store.search(
            query_embedding=job_description.embedding,
            top_k=self.settings.retrieval_top_k,
        )

        # Get candidates from retrieval
        retrieved_candidates = []
        for result in retrieval_results:
            candidate_id = result["candidate_id"]
            if candidate_id in self._candidates:
                # Compute semantic similarity from distance
                distance = result.get("distance", 0.5)
                semantic_similarity = 1.0 - distance  # Convert distance to similarity

                retrieved_candidates.append((
                    self._candidates[candidate_id],
                    semantic_similarity,
                ))

        self.logger.info(f"Retrieved {len(retrieved_candidates)} candidates")

        # Step 2: Compute hybrid scores
        self.logger.info("Step 2: Computing hybrid scores")
        scored_candidates = []

        for candidate, semantic_sim in retrieved_candidates:
            # Compute features
            features = self.scoring_engine.compute_features(
                candidate=candidate,
                job=job,
                semantic_similarity=semantic_sim,
            )

            # Compute hybrid score
            hybrid_score, weighted_components = self.scoring_engine.compute_hybrid_score(
                features=features,
            )

            # Create scoring result
            scoring_result = ScoringResult(
                candidate_id=candidate.profile.candidate_id,
                features=features,
                weighted_semantic=weighted_components["weighted_semantic"],
                weighted_skill=weighted_components["weighted_skill"],
                weighted_experience=weighted_components["weighted_experience"],
                weighted_behavior=weighted_components["weighted_behavior"],
                weighted_industry=weighted_components["weighted_industry"],
                weighted_education=weighted_components["weighted_education"],
                weighted_bonus=weighted_components["weighted_bonus"],
                hybrid_score=hybrid_score,
            )

            scored_candidates.append((candidate, scoring_result))

        # Sort by hybrid score
        scored_candidates.sort(key=lambda x: x[1].hybrid_score, reverse=True)

        # Step 3: Re-ranking with cross-encoder
        if request.use_reranking and self.ranking_engine.is_loaded:
            self.logger.info("Step 3: Cross-encoder re-ranking")
            reranked = self.ranking_engine.rerank(
                job=job,
                candidates=scored_candidates[:self.settings.rerank_top_k],
                top_k=request.top_k,
            )

            # Update scoring results with rerank scores
            final_candidates = []
            for i, (candidate, scoring_result, cross_score) in enumerate(reranked):
                scoring_result.final_rank = i + 1
                final_candidates.append((candidate, scoring_result))
        else:
            self.logger.info("Step 3: Using hybrid scores only (no re-ranking)")
            final_candidates = scored_candidates[:request.top_k]
            for i, (candidate, scoring_result) in enumerate(final_candidates):
                scoring_result.final_rank = i + 1
                scoring_result.final_score = scoring_result.hybrid_score

        # Step 4: Generate recommendations
        self.logger.info("Step 4: Generating final recommendations")
        recommendations = []

        for i, (candidate, scoring_result) in enumerate(final_candidates):
            # Generate reasoning if requested
            reasoning = None
            if request.include_reasoning:
                reasoning = self.ranking_engine.generate_reasoning(
                    candidate=candidate,
                    job=job,
                    features=scoring_result.features,
                    final_score=scoring_result.final_score,
                )

            # Get top matching skills
            top_skills = candidate.document.ml_skills[:5]
            if not top_skills:
                top_skills = candidate.profile.skills[:5]

            ranked_candidate = RankedCandidate(
                rank=i + 1,
                candidate_id=candidate.profile.candidate_id,
                candidate_name=candidate.profile.name,
                candidate_headline=candidate.profile.headline,
                semantic_score=scoring_result.features.semantic_similarity,
                skill_score=scoring_result.features.skill_match,
                experience_score=scoring_result.features.experience_match,
                behavior_score=scoring_result.features.behavior_score,
                hybrid_score=scoring_result.hybrid_score,
                rerank_score=scoring_result.rerank_score,
                final_score=scoring_result.final_score,
                total_years_experience=candidate.profile.total_years_experience,
                seniority_level=candidate.document.seniority_level,
                top_skills=top_skills,
                industry=job.industry,
                reasoning=reasoning,
            )

            recommendations.append(ranked_candidate)

        # Apply minimum score filter if specified
        if request.min_score is not None:
            recommendations = [
                r for r in recommendations if r.final_score >= request.min_score
            ]

        elapsed = time.time() - start_time
        self.logger.info(f"Recommendations generated in {elapsed:.2f}s")

        # Create result
        result = RecommendationResult(
            job_id=request.job_id,
            job_title=job.title,
            total_candidates_processed=len(self._candidates),
            candidates_retrieved=len(retrieved_candidates),
            candidates_reranked=len(scored_candidates[:self.settings.rerank_top_k]),
            final_recommendations=len(recommendations),
            recommendations=recommendations,
            execution_time_ms=elapsed * 1000,
            embedding_model=self.embedding_service.model_name or "unknown",
            cross_encoder_model=self.ranking_engine.model_name,
            weights_used=self.settings.weights,
        )

        return result

    def export_to_csv(
        self,
        result: RecommendationResult,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export recommendations to CSV.

        Args:
            result: Recommendation result
            output_path: Output file path (uses settings if not provided)

        Returns:
            Path to exported file
        """
        output_path = output_path or self.settings.output_file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Exporting recommendations to: {output_file}")

        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "candidate_id",
                "rank",
                "score",
                "reasoning",
                "name",
                "headline",
                "years_experience",
                "seniority_level",
                "semantic_score",
                "skill_score",
                "experience_score",
                "behavior_score",
                "hybrid_score",
                "rerank_score",
                "top_skills",
            ])

            # Write data
            for rec in result.recommendations:
                writer.writerow([
                    rec.candidate_id,
                    rec.rank,
                    f"{rec.final_score:.4f}",
                    rec.reasoning or "",
                    rec.candidate_name or "",
                    rec.candidate_headline or "",
                    rec.total_years_experience or "",
                    rec.seniority_level or "",
                    f"{rec.semantic_score:.4f}",
                    f"{rec.skill_score:.4f}",
                    f"{rec.experience_score:.4f}",
                    f"{rec.behavior_score:.4f}",
                    f"{rec.hybrid_score:.4f}",
                    f"{rec.rerank_score:.4f}" if rec.rerank_score else "",
                    "; ".join(rec.top_skills),
                ])

        self.logger.info(f"Exported {len(result.recommendations)} recommendations")
        return str(output_file)

    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        """
        Get a candidate by ID.

        Args:
            candidate_id: Candidate identifier

        Returns:
            Candidate or None
        """
        return self._candidates.get(candidate_id)

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get pipeline metrics.

        Returns:
            Metrics dictionary
        """
        return {
            "total_candidates": len(self._candidates),
            "total_jobs": len(self._jobs),
            "vector_store_size": self.vector_store.collection_size,
            "embedding_model": self.embedding_service.model_name,
            "cross_encoder_model": self.ranking_engine.model_name,
            "models_loaded": self.embedding_service.is_loaded,
        }


# Global instance
_recommendation_pipeline: Optional[RecommendationPipeline] = None


def get_recommendation_pipeline(settings: Optional[Settings] = None) -> RecommendationPipeline:
    """
    Get or create the recommendation pipeline singleton.

    Args:
        settings: Optional settings override

    Returns:
        RecommendationPipeline instance
    """
    global _recommendation_pipeline

    if _recommendation_pipeline is None:
        _recommendation_pipeline = RecommendationPipeline(settings)

    return _recommendation_pipeline
