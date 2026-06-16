"""
src/utils/logger.py
Centralised Loguru logger setup.
Import `logger` from here in every module.
"""

import sys
from pathlib import Path
from loguru import logger as _logger


def setup_logger(log_level: str = "INFO", log_dir: str = "./logs") -> None:
    """Configure logger with console + rotating file handlers."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    _logger.remove()  # Remove the default handler

    # Console — human-readable with colours
    _logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File — JSON-structured for log aggregation tools
    _logger.add(
        f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
        level=log_level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        serialize=True,  # JSON format
    )


# Initialise with defaults so the logger is usable on import
setup_logger()

logger = _logger