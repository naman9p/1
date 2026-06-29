"""
Pytest Configuration

Fixtures and configuration for tests.
"""

import pytest
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set test environment
os.environ["APP_ENV"] = "test"
os.environ["DEBUG"] = "true"
os.environ["LOG_LEVEL"] = "WARNING"
os.environ["CHROMA_DB_PATH"] = "./data/test_chroma_db"


@pytest.fixture(scope="session")
def test_settings():
    """Create test settings."""
    from app.core.config import Settings

    return Settings(
        app_env="test",
        debug=True,
        log_level="WARNING",
        chroma_db_path="./data/test_chroma_db",
    )


@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Cleanup test data after tests."""
    yield
    # Cleanup test ChromaDB
    import shutil
    test_db_path = Path("./data/test_chroma_db")
    if test_db_path.exists():
        shutil.rmtree(test_db_path)
