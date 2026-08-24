"""
Log sanitization utilities to prevent log injection attacks.

This module provides functions to sanitize user input before logging,
preventing log injection attacks through newline characters, ANSI codes,
and other control sequences.
"""

import logging
import re
from typing import Any


def sanitize_log_message(message: Any, max_length: int = 10000) -> str:
    """
    Sanitize a message for safe logging.

    Removes or escapes characters that could be used for log injection:
    - Newlines (\n, \r)
    - ANSI escape codes
    - Control characters

    Args:
        message: The message to sanitize
        max_length: Maximum length of sanitized message (default: 10000)

    Returns:
        Sanitized string safe for logging
    """
    if message is None:
        return ""

    # Convert to string
    message_str = str(message)

    # Remove newlines and carriage returns (log injection)
    message_str = message_str.replace("\n", "\\n").replace("\r", "\\r")

    # Remove ANSI escape codes
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    message_str = ansi_escape.sub("", message_str)

    # Remove other control characters (except tab)
    message_str = "".join(char for char in message_str if char == "\t" or char.isprintable())

    # Limit length to prevent log flooding
    if len(message_str) > max_length:
        message_str = message_str[:max_length] + "... (truncated)"

    return message_str


def sanitize_api_key(message: str) -> str:
    """
    Remove API keys from a message.

    Detects and redacts common API key patterns:
    - OpenAI: sk-...
    - Gemini: AIza...
    - Generic: api_key=..., token=...

    Args:
        message: Message that may contain API keys

    Returns:
        Message with API keys redacted
    """
    if not message:
        return message

    result = message

    # Redact OpenAI-style keys (sk- followed by 20+ alphanumeric chars)
    result = re.sub(r"sk-[a-zA-Z0-9]{20,}", "sk-[REDACTED]", result)

    # Redact Gemini keys (AIza followed by 30+ alphanumeric chars)
    result = re.sub(r"AIza[a-zA-Z0-9_-]{30,}", "AIza[REDACTED]", result)

    # Redact Bearer tokens
    result = re.sub(r"Bearer\s+[a-zA-Z0-9\-._~+/]{15,}", "Bearer [REDACTED]", result)

    # Redact api_key= patterns
    result = re.sub(
        r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{10,}',
        "api_key=[REDACTED]",
        result,
        flags=re.IGNORECASE,
    )

    # Redact token= patterns
    result = re.sub(
        r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-._~+/]{15,}',
        "token=[REDACTED]",
        result,
        flags=re.IGNORECASE,
    )

    return result




class SanitizingFormatter(logging.Formatter):
    """
    A logging formatter that sanitizes log records.

    Usage:
        import logging
        from utils.log_sanitizer import SanitizingFormatter

        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(SanitizingFormatter())
        logger.addHandler(handler)
    """

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        max_length: int = 10000,
    ):
        """
        Initialize the formatter.

        Args:
            fmt: Log format string (same as logging.Formatter)
            datefmt: Date format string
            max_length: Maximum message length
        """
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.max_length = max_length

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with sanitization.

        Args:
            record: Log record to format

        Returns:
            Formatted and sanitized log message
        """
        # Sanitize the message (log injection + API key 清洗, 审查 #9)
        record.msg = sanitize_api_key(sanitize_log_message(record.msg, self.max_length))

        # Sanitize args if present
        if record.args:
            sanitized_args = tuple(
                sanitize_log_message(arg, self.max_length) for arg in record.args
            )
            record.args = sanitized_args

        # Use default formatting
        return super().format(record)




# Import logging at the end to avoid circular dependency
import logging
