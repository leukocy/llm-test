"""
Log sanitization utilities to prevent log injection attacks.

This module provides functions to sanitize user input before logging,
preventing log injection attacks through newline characters, ANSI codes,
and other control sequences.
"""

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
    message_str = message_str.replace('\n', '\\n').replace('\r', '\\r')

    # Remove ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    message_str = ansi_escape.sub('', message_str)

    # Remove other control characters (except tab)
    message_str = ''.join(
        char for char in message_str
        if char == '\t' or char.isprintable()
    )

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
    result = re.sub(r'sk-[a-zA-Z0-9]{20,}', 'sk-[REDACTED]', result)

    # Redact Gemini keys (AIza followed by 30+ alphanumeric chars)
    result = re.sub(r'AIza[a-zA-Z0-9_-]{30,}', 'AIza[REDACTED]', result)

    # Redact Bearer tokens
    result = re.sub(r'Bearer\s+[a-zA-Z0-9\-._~+/]{15,}', 'Bearer [REDACTED]', result)

    # Redact api_key= patterns
    result = re.sub(r'api[_-]?key["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-_]{10,}', 'api_key=[REDACTED]', result, flags=re.IGNORECASE)

    # Redact token= patterns
    result = re.sub(r'token["\']?\s*[:=]\s*["\']?[a-zA-Z0-9\-._~+/]{15,}', 'token=[REDACTED]', result, flags=re.IGNORECASE)

    return result


def sanitize_error_response(error_text: str, api_key: str = None) -> str:
    """
    Remove API key from error messages.

    Args:
        error_text: Error response text
        api_key: API key to redact (optional)

    Returns:
        Sanitized error text
    """
    if not error_text:
        return error_text

    result = error_text

    # Remove specific API key if provided
    if api_key and api_key in result:
        result = result.replace(api_key, '***REDACTED***')

    # Also redact common key patterns
    result = sanitize_api_key(result)

    return result


class SanitizingFormatter:
    """
    A logging formatter that sanitizes log records.

    Usage:
        import logging
        from utils.log_sanitizer import SanitizingFormatter

        logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        handler.setFormatter(SanizingFormatter())
        logger.addHandler(handler)
    """

    def __init__(self, max_length: int = 10000):
        """
        Initialize the formatter.

        Args:
            max_length: Maximum message length
        """
        self.max_length = max_length

    def format(self, record) -> str:
        """
        Format a log record with sanitization.

        Args:
            record: Log record to format

        Returns:
            Formatted and sanitized log message
        """
        # Sanitize the message
        record.msg = sanitize_log_message(record.msg, self.max_length)

        # Sanitize args if present
        if record.args:
            sanitized_args = tuple(
                sanitize_log_message(arg, self.max_length)
                for arg in record.args
            )
            record.args = sanitized_args

        # Use default formatting
        return logging.Formatter().format(record)


def safe_log(logger, level: str, message: Any, *args, **kwargs):
    """
    Safely log a message with sanitization.

    Args:
        logger: Logger instance
        level: Log level ('debug', 'info', 'warning', 'error', 'critical')
        message: Message to log
        *args: Additional args (will be sanitized)
        **kwargs: Additional kwargs
    """
    sanitized_message = sanitize_log_message(message)
    sanitized_args = tuple(sanitize_log_message(arg) for arg in args)

    log_func = getattr(logger, level.lower(), None)
    if log_func:
        log_func(sanitized_message, *sanitized_args, **kwargs)


# Import logging at the end to avoid circular dependency
import logging
