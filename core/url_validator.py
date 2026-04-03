"""
URL validation utilities to prevent SSRF (Server-Side Request Forgery) attacks.
"""

import re
from typing import Optional, Tuple
from urllib.parse import urlparse, ParseResult


class SSRFError(Exception):
    """Raised when URL validation fails due to SSRF risk."""
    pass


# Default allowlist of known safe API providers
DEFAULT_SAFE_DOMAINS = {
    'api.openai.com',
    'api.deepseek.com',
    'open.bigmodel.cn',
    'dashscope.aliyuncs.com',
    'api.minimax.chat',
    'api.siliconflow.cn',
    'openrouter.ai',
    'ark.cn-beijing.volces.com',
    'api.moonshot.cn',
    'generativelanguage.googleapis.com',
    'xiaomimimo.com',
    # Local development servers
    'localhost',
    '127.0.0.1',
}

# Blocklist of dangerous IP ranges (CIDR notation converted to regex)
BLOCKED_IP_PATTERNS = [
    r'192\.168\.\d+\.\d+',      # Private network (Class C)
    r'10\.\d+\.\d+\.\d+',        # Private network (Class A)
    r'172\.(1[6-9]|2[0-9]|3[01])\.\d+\.\d+',  # Private network (Class B)
    r'127\.\d+\.\d+\.\d+',       # Loopback (beyond localhost)
    r'0\.\d+\.\d+\.\d+',         # Invalid
    r'169\.254\.\d+\.\d+',       # Link-local
    r'224\.\d+\.\d+\.\d+',       # Multicast
    r'240\.\d+\.\d+\.\d+',       # Reserved
]


def is_safe_url(
    url: str,
    allow_private: bool = False,
    custom_safe_domains: Optional[set[str]] = None
) -> Tuple[bool, Optional[str]]:
    """
    Validate a URL to prevent SSRF attacks.

    Args:
        url: The URL to validate
        allow_private: Whether to allow private/internal IPs (default: False)
        custom_safe_domains: Additional domains to allow

    Returns:
        Tuple of (is_safe: bool, error_message: Optional[str])
    """
    if not url or not isinstance(url, str):
        return False, "URL must be a non-empty string"

    try:
        parsed: ParseResult = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"

    # Check scheme (protocol)
    if parsed.scheme not in ('http', 'https'):
        return False, f"Unsupported protocol: {parsed.scheme}. Only http and https are allowed."

    # Check hostname exists
    if not parsed.hostname:
        return False, "URL must have a valid hostname"

    hostname = parsed.hostname.lower()

    # Check for IP-based SSRF
    # First, check if it's an IP address
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(ip_pattern, hostname):
        # Check against blocked IP patterns
        for blocked_pattern in BLOCKED_IP_PATTERNS:
            if re.match(blocked_pattern, hostname):
                if not allow_private:
                    return False, f"Private/internal IP addresses are not allowed: {hostname}"

        # Validate IP octets are all <= 255
        octets = hostname.split('.')
        for octet in octets:
            if int(octet) > 255:
                return False, f"Invalid IP address: {hostname}"

    # Check for private/internal hostnames
    private_patterns = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
    if hostname in private_patterns and not allow_private:
        return False, f"Local addresses are not allowed: {hostname}"

    # Check for metadata endpoints (AWS, GCP, Azure)
    if 'metadata' in hostname.lower():
        return False, "Metadata endpoints are not allowed"

    # Check against allowlist (if provided)
    safe_domains = DEFAULT_SAFE_DOMAINS.copy()
    if custom_safe_domains:
        safe_domains.update(custom_safe_domains)

    # If hostname is in allowlist, it's safe
    if hostname in safe_domains:
        return True, None

    # Check if it's a subdomain of a safe domain
    for safe_domain in safe_domains:
        if hostname.endswith('.' + safe_domain):
            return True, None

    # For unknown domains, warn but allow (with logging recommendation)
    # In production, you might want to be more strict
    import warnings
    warnings.warn(
        f"Unrecognized domain: {hostname}. Ensure this is a trusted API provider.",
        UserWarning,
        stacklevel=2
    )

    return True, None


def validate_and_normalize_url(
    url: str,
    allow_private: bool = False,
    require_https: bool = False
) -> str:
    """
    Validate and normalize a URL.

    Args:
        url: The URL to validate
        allow_private: Whether to allow private IPs
        require_https: Whether to require HTTPS (recommended for production)

    Returns:
        Normalized URL

    Raises:
        SSRFError: If URL is invalid or unsafe
    """
    is_safe, error = is_safe_url(url, allow_private=allow_private)
    if not is_safe:
        raise SSRFError(f"URL validation failed: {error}")

    parsed = urlparse(url)

    # Enforce HTTPS if required
    if require_https and parsed.scheme != 'https':
        raise SSRFError("HTTPS is required for API calls")

    # Normalize: ensure path doesn't end with duplicate slashes
    normalized = url.rstrip('/')
    if not normalized.endswith('/v1') and not normalized.endswith('/v1/') and '/' not in normalized.split('://', 1)[1]:
        # Add trailing slash for consistency if no path
        normalized = normalized + '/'

    return normalized


def is_port_safe(url: str) -> Tuple[bool, Optional[str]]:
    """
    Check if the port in the URL is safe.

    Args:
        url: The URL to check

    Returns:
        Tuple of (is_safe, error_message)
    """
    try:
        parsed = urlparse(url)
        if parsed.port is None:
            return True, None

        port = parsed.port

        # Block dangerous ports
        dangerous_ports = {
            22,    # SSH
            23,    # Telnet
            25,    # SMTP
            53,    # DNS
            135,   # Windows RPC
            139,   # NetBIOS
            445,   # SMB
            3389,  # RDP
            5432,  # PostgreSQL (unless explicitly allowed)
            3306,  # MySQL (unless explicitly allowed)
            6379,  # Redis
            27017, # MongoDB
        }

        if port in dangerous_ports:
            return False, f"Dangerous port not allowed: {port}"

        # Allow common HTTP ports and high ports
        if port in (80, 443, 8080, 8443, 3000, 5000, 8000, 10814):
            return True, None

        # For other high ports, warn but allow
        if port > 1024:
            return True, None

        return False, f"Port {port} is not allowed"

    except Exception as e:
        return False, f"Error checking port: {e}"
