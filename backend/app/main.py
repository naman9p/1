"""
Main FastAPI Application

Entry point for the AI Candidate Recommendation Engine.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from .core.config import settings
from .core.logging_config import setup_logging, get_logger
from .api.routes import router
from .services.recommendation_pipeline import get_recommendation_pipeline

from . import __version__

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI Candidate Recommendation Engine...")
    start_time = time.time()

    # Setup logging
    setup_logging()

    # Initialize pipeline
    pipeline = get_recommendation_pipeline()
    if not pipeline.initialize():
        logger.error("Failed to initialize pipeline")
        yield
        return

    elapsed = time.time() - start_time
    logger.info(f"Application started in {elapsed:.2f}s")

    yield

    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## AI Candidate Recommendation Engine

A production-ready system for semantically matching job descriptions with candidate profiles.

### Features

- **Semantic Search**: Uses Sentence Transformers for deep semantic understanding
- **Hybrid Scoring**: Combines multiple factors (skills, experience, behavior, etc.)
- **Cross-Encoder Re-ranking**: Advanced re-ranking with transformer models
- **Behavioral Signals**: Incorporates recruiter interaction data
- **Configurable Weights**: All scoring weights are configurable
- **CSV Export**: Easy export of recommendations

### API Endpoints

- **POST /upload-job**: Upload and process a job description
- **POST /index-candidates**: Index candidate profiles from JSONL
- **POST /recommend**: Generate recommendations
- **GET /candidate/{id}**: Get candidate details
- **GET /health**: Health check
- **GET /metrics**: System metrics
    """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api")


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.

    Returns basic API information.
    """
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
