"""
Logging Configuration Module

Centralized logging setup using Loguru with structured output.
"""

import sys
from loguru import logger
from pathlib import Path
from typing import Optional

from .config import settings


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
    debug: Optional[bool] = None,
) -> None:
    """
    Configure application logging with Loguru.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        debug: Debug mode flag
    """
    # Remove default handler
    logger.remove()

    # Get settings
    level = log_level or settings.log_level
    is_debug = debug if debug is not None else settings.debug

    # Console handler with color and structured format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=is_debug,
        backtrace=is_debug,
        diagnose=is_debug,
    )

    # File handler for persistent logs
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            log_path,
            format=file_format,
            level=level,
            rotation="10 MB",
            retention="7 days",
            compression="zip",
            backtrace=is_debug,
            diagnose=is_debug,
        )

    logger.info(f"Logging initialized at level: {level}")


def get_logger(name: str = __name__):
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Loguru logger instance
    """
    return logger.bind(name=name)
