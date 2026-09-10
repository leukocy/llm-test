"""
Module-level logger factory for consistent logging across modules.

Usage:
    from utils.get_logger import get_logger
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
"""

import logging
import sys

from .log_sanitizer import SanitizingFormatter

_LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"


def _make_handler() -> logging.StreamHandler:
    """stdout handler + 脱敏 formatter(API key/换行/ANSI 清洗, 审查 #9)。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(SanitizingFormatter(fmt=_LOG_FORMAT, datefmt="%H:%M:%S"))
    return handler


# Configure root logger once
def _setup_logging():
    """Configure the root logger for the application."""
    root = logging.getLogger("llm_test")
    if not root.handlers:
        root.addHandler(_make_handler())
        root.setLevel(logging.INFO)  # Default level, can be overridden
    return root


_root_logger = _setup_logging()


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger instance for the current module.

    Args:
        name: Module name (use __name__ when calling)
        level: Logging level (default: INFO)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> from utils.get_logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.debug("This is a debug message")
        >>> logger.info("This is an info message")
        >>> logger.warning("This is a warning")
        >>> logger.error("This is an error")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    # Add sanitizing handler if not already present
    if not logger.handlers:
        logger.addHandler(_make_handler())
    return logger


