"""
Logging utilities for FinSentNet.
Provides structured logging with financial-context-aware formatting.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "finsent",
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure structured logger with file + console output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger  # already configured

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_path / f"{name}_{timestamp}.log")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def log_config(logger: logging.Logger, config: dict, indent: int = 0) -> None:
    """Recursively log configuration dict."""
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info("%s%s:", " " * indent, key)
            log_config(logger, value, indent + 2)
        else:
            logger.info("%s%s: %s", " " * indent, key, value)
