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
from typing import Optional

# Configure root logger once
def _setup_logging():
    """Configure the root logger for the application."""
    root = logging.getLogger("llm_test")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
        root.setLevel(logging.INFO)  # Default level, can be overridden
    return root

_root_logger = _setup_logging()


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
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
    # Add handler if not already present
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_global_level(level: int):
    """
    Set the global logging level.

    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)

    Example:
        >>> from utils.get_logger import set_global_level
        >>> import logging
        >>> set_global_level(logging.DEBUG)
    """
    _root_logger.setLevel(level)
    for handler in _root_logger.handlers:
        handler.setLevel(level)
