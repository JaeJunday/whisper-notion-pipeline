import sys
from pathlib import Path

from loguru import logger

from .config import settings


def setup_logger():
    """Configure logger based on settings"""
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stderr,
        format=settings.logging.format,
        level=settings.logging.level,
        colorize=True,
    )
    
    # Add file logger
    log_path = Path(settings.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        settings.logging.file,
        format=settings.logging.format,
        level=settings.logging.level,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        compression=settings.logging.compression,
        encoding="utf-8",
    )
    
    return logger


# Initialize logger
logger = setup_logger()