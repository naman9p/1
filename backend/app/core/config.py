"""
Application Configuration Module

Centralized configuration management using Pydantic Settings.
All weights and parameters are configurable via environment variables or YAML config.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from pathlib import Path
import os


class Settings(BaseSettings):
    """
    Application settings with sensible defaults.
    All values can be overridden via environment variables.
    """

    # Application Settings
    app_name: str = Field(default="AI Candidate Recommendation Engine", description="Application name")
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=True, description="Debug mode flag")
    log_level: str = Field(default="INFO", description="Logging level")

    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # AI Model Settings
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="Primary sentence transformer model for embeddings"
    )
    fallback_embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Fallback embedding model if primary fails"
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking"
    )

    # Vector Database Settings
    chroma_db_path: str = Field(default="./data/chroma_db", description="Path to ChromaDB storage")
    chroma_collection: str = Field(default="candidates", description="ChromaDB collection name")

    # Scoring Weights (must sum to 1.0)
    weight_semantic: float = Field(default=0.40, ge=0, le=1, description="Weight for semantic similarity")
    weight_skill: float = Field(default=0.20, ge=0, le=1, description="Weight for skill match")
    weight_experience: float = Field(default=0.10, ge=0, le=1, description="Weight for experience match")
    weight_behavior: float = Field(default=0.10, ge=0, le=1, description="Weight for behavioral signals")
    weight_industry: float = Field(default=0.10, ge=0, le=1, description="Weight for industry match")
    weight_education: float = Field(default=0.05, ge=0, le=1, description="Weight for education match")
    weight_bonus: float = Field(default=0.05, ge=0, le=1, description="Weight for bonus skills")

    # Retrieval Settings
    retrieval_top_k: int = Field(default=300, ge=10, le=1000, description="Number of candidates to retrieve initially")
    rerank_top_k: int = Field(default=100, ge=10, le=500, description="Number of candidates to re-rank")
    final_output_k: int = Field(default=100, ge=1, le=100, description="Final number of recommendations")

    # Data Paths
    data_dir: str = Field(default="./data", description="Base directory for data files")
    candidates_file: str = Field(default="./data/candidates.jsonl", description="Path to candidates JSONL file")
    job_description_file: str = Field(default="./data/job_description.txt", description="Path to job description file")
    output_file: str = Field(default="./output/recommendations.csv", description="Path to output CSV file")

    # Performance Settings
    max_workers: int = Field(default=4, description="Maximum number of parallel workers")
    batch_size: int = Field(default=32, description="Batch size for embedding generation")
    cache_embeddings: bool = Field(default=True, description="Cache embeddings to disk")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def weights(self) -> dict:
        """Return scoring weights as a dictionary."""
        return {
            "semantic": self.weight_semantic,
            "skill": self.weight_skill,
            "experience": self.weight_experience,
            "behavior": self.weight_behavior,
            "industry": self.weight_industry,
            "education": self.weight_education,
            "bonus": self.weight_bonus,
        }

    @property
    def weights_sum(self) -> float:
        """Return sum of all weights (should be 1.0)."""
        return sum(self.weights.values())

    def validate_weights(self) -> bool:
        """Validate that weights sum to approximately 1.0."""
        return abs(self.weights_sum - 1.0) < 0.01


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings
