"""
API Routes

FastAPI endpoint definitions.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Body, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import json
from datetime import datetime

from ..core.config import settings
from ..core.logging_config import get_logger
from ..schemas.job import JobDescriptionInput
from ..schemas.candidate import CandidateProfile
from ..schemas.recommendation import RecommendationRequest, RecommendationResponse
from ..schemas.api import (
    HealthResponse,
    MetricsResponse,
    UploadJobResponse,
    IndexCandidatesResponse,
    ErrorResponse,
    CandidateResponse,
)
from ..services.recommendation_pipeline import get_recommendation_pipeline

from .. import __version__

logger = get_logger(__name__)

router = APIRouter()

# Store startup time for uptime calculation
_startup_time = datetime.utcnow()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and component health.
    """
    pipeline = get_recommendation_pipeline()

    components = {}

    # Check embedding service
    if pipeline.embedding_service.is_loaded:
        components["embedding_model"] = "loaded"
    else:
        components["embedding_model"] = "not_loaded"

    # Check vector store
    if pipeline.vector_store.is_initialized:
        components["vector_store"] = "connected"
    else:
        components["vector_store"] = "disconnected"

    # Check cross-encoder
    if pipeline.ranking_engine.is_loaded:
        components["cross_encoder"] = "loaded"
    else:
        components["cross_encoder"] = "not_loaded"

    # Overall status
    all_healthy = all(v in ["loaded", "connected"] for v in components.values())
    status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        timestamp=datetime.utcnow(),
        components=components,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["Health"])
async def get_metrics():
    """
    Get system metrics.

    Returns detailed metrics about the system state.
    """
    pipeline = get_recommendation_pipeline()
    metrics = pipeline.get_metrics()

    uptime = (datetime.utcnow() - _startup_time).total_seconds()

    return MetricsResponse(
        version=__version__,
        uptime_seconds=uptime,
        total_candidates=metrics["total_candidates"],
        total_jobs=metrics["total_jobs"],
        vector_store_size=metrics["vector_store_size"],
        embedding_model=metrics["embedding_model"] or "not_loaded",
        cross_encoder_model=metrics["cross_encoder_model"],
        model_loaded=metrics["models_loaded"],
        weights=settings.weights,
        retrieval_top_k=settings.retrieval_top_k,
        rerank_top_k=settings.rerank_top_k,
    )


@router.post("/upload-job", response_model=UploadJobResponse, tags=["Jobs"])
async def upload_job(
    job_id: str = Body(..., description="Job identifier"),
    title: str = Body(..., description="Job title"),
    description: str = Body(..., description="Job description text"),
    company: Optional[str] = Body(None, description="Company name"),
    location: Optional[str] = Body(None, description="Job location"),
    job_type: Optional[str] = Body(None, description="Job type"),
):
    """
    Upload and process a job description.

    Parses the job description and generates embeddings for semantic matching.
    """
    try:
        pipeline = get_recommendation_pipeline()

        if not pipeline.is_initialized:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Create job input
        job_input = JobDescriptionInput(
            job_id=job_id,
            title=title,
            description=description,
            company=company,
            location=location,
            job_type=job_type,
        )

        # Process job
        job_description = pipeline.process_job(job_input)

        return UploadJobResponse(
            success=True,
            message="Job uploaded and processed successfully",
            job_id=job_id,
            job_title=title,
            parsed=True,
            embedding_generated=True,
        )

    except Exception as e:
        logger.error(f"Failed to upload job: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/index-candidates", response_model=IndexCandidatesResponse, tags=["Candidates"])
async def index_candidates(
    file: UploadFile = File(..., description="JSONL file with candidate profiles"),
):
    """
    Index candidate profiles from a JSONL file.

    Expects a JSONL file where each line is a candidate profile JSON object.
    """
    try:
        pipeline = get_recommendation_pipeline()

        if not pipeline.is_initialized:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Read and parse JSONL file
        content = await file.read()
        lines = content.decode("utf-8").strip().split("\n")

        candidates = []
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                profile_data = json.loads(line)
                profile = CandidateProfile(**profile_data)
                candidates.append(profile)
            except Exception as e:
                logger.warning(f"Failed to parse line {i}: {e}")

        if not candidates:
            raise HTTPException(status_code=400, detail="No valid candidates found in file")

        # Index candidates
        indexed, failed = pipeline.index_candidates(candidates)

        return IndexCandidatesResponse(
            success=True,
            message=f"Successfully indexed {indexed} candidates",
            total_processed=len(candidates),
            total_indexed=indexed,
            total_failed=failed,
            embedding_model=pipeline.embedding_service.model_name or "unknown",
            execution_time_seconds=0.0,  # Will be updated by pipeline
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to index candidates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend(
    request: RecommendationRequest,
):
    """
    Generate candidate recommendations for a job.

    Returns ranked list of candidates with scores and reasoning.
    """
    try:
        pipeline = get_recommendation_pipeline()

        if not pipeline.is_initialized:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Generate recommendations
        result = pipeline.recommend(request)

        # Export to CSV
        csv_path = pipeline.export_to_csv(result)
        logger.info(f"Recommendations exported to: {csv_path}")

        return RecommendationResponse(
            success=True,
            message=f"Successfully generated {len(result.recommendations)} recommendations",
            result=result,
        )

    except ValueError as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to generate recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candidate/{candidate_id}", response_model=CandidateResponse, tags=["Candidates"])
async def get_candidate(candidate_id: str):
    """
    Get details for a specific candidate.

    Returns candidate profile information.
    """
    try:
        pipeline = get_recommendation_pipeline()

        candidate = pipeline.get_candidate(candidate_id)

        if not candidate:
            raise HTTPException(status_code=404, detail=f"Candidate not found: {candidate_id}")

        # Get top skills
        top_skills = candidate.document.ml_skills[:10]
        if not top_skills:
            top_skills = candidate.profile.skills[:10]

        # Get current position
        current_company = None
        current_title = None
        for exp in candidate.profile.work_experience:
            if exp.is_current:
                current_company = exp.company
                current_title = exp.title
                break

        # Education summary
        education_summary = None
        if candidate.profile.education:
            edu = candidate.profile.education[0]
            education_summary = f"{edu.degree} from {edu.institution}"

        return CandidateResponse(
            success=True,
            candidate_id=candidate_id,
            name=candidate.profile.name,
            headline=candidate.profile.headline,
            location=candidate.profile.location,
            total_years_experience=candidate.profile.total_years_experience,
            skills=top_skills,
            current_company=current_company,
            current_title=current_title,
            education_summary=education_summary,
            behavioral_score=(
                candidate.profile.behavioral_signals.behavior_score
                if candidate.profile.behavioral_signals
                else None
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get candidate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend/export", tags=["Recommendations"])
async def export_recommendations(
    job_id: str = Body(..., description="Job identifier"),
    top_k: int = Body(100, description="Number of recommendations"),
    include_reasoning: bool = Body(True, description="Include reasoning"),
    use_reranking: bool = Body(True, description="Use cross-encoder re-ranking"),
):
    """
    Generate and export recommendations to CSV.

    Returns the path to the exported CSV file.
    """
    try:
        pipeline = get_recommendation_pipeline()

        if not pipeline.is_initialized:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        # Create request
        request = RecommendationRequest(
            job_id=job_id,
            top_k=top_k,
            include_reasoning=include_reasoning,
            use_reranking=use_reranking,
        )

        # Generate recommendations
        result = pipeline.recommend(request)

        # Export to CSV
        csv_path = pipeline.export_to_csv(result)

        return {
            "success": True,
            "message": f"Recommendations exported to {csv_path}",
            "file_path": csv_path,
            "total_recommendations": len(result.recommendations),
        }

    except Exception as e:
        logger.error(f"Failed to export recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reset", tags=["Administration"])
async def reset_system():
    """
    Reset the system (clear all data).

    Use with caution - this will delete all indexed candidates and jobs.
    """
    try:
        pipeline = get_recommendation_pipeline()

        # Reset vector store
        pipeline.vector_store.reset()

        # Clear in-memory state
        pipeline._jobs.clear()
        pipeline._candidates.clear()

        return {
            "success": True,
            "message": "System reset successfully",
        }

    except Exception as e:
        logger.error(f"Failed to reset system: {e}")
        raise HTTPException(status_code=500, detail=str(e))
