"""Shared logging setup for notebooks and modules."""

from __future__ import annotations

import logging
import sys


def setup_logger(name: str = "rag_academy", level: int = logging.INFO) -> logging.Logger:
    """Create or return a configured logger with standard stream handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s : %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger
