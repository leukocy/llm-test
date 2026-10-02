"""
URL validation utilities to prevent SSRF (Server-Side Request Forgery) attacks.
"""

import ipaddress
import socket
from urllib.parse import ParseResult, urlparse


class SSRFError(Exception):
    """Raised when URL validation fails due to SSRF risk."""

    pass


# Default allowlist of known safe API providers
DEFAULT_SAFE_DOMAINS = {
    "api.openai.com",
    "api.anthropic.com",
    "api.together.ai",
    "api.together.xyz",
    "api.deepseek.com",
    "open.bigmodel.cn",
    "dashscope.aliyuncs.com",
    "api.minimax.chat",
    "api.siliconflow.cn",
    "openrouter.ai",
    "ark.cn-beijing.volces.com",
    "api.moonshot.cn",
    "generativelanguage.googleapis.com",
    "xiaomimimo.com",
}


def parse_trusted_private_networks(
    value: str,
) -> set[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse administrator CIDRs, rejecting global and special-use networks."""
    networks: set[ipaddress.IPv4Network | ipaddress.IPv6Network] = set()
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            network = ipaddress.ip_network(entry, strict=True)
        except ValueError as exc:
            raise SSRFError("Invalid trusted API network CIDR") from exc
        if (
            not network.is_private
            or network.is_loopback
            or network.is_link_local
            or network.is_multicast
            or network.is_reserved
        ):
            raise SSRFError("Trusted API networks must be private, non-special-use CIDRs")
        networks.add(network)
    return networks


def is_safe_url(
    url: str,
    allow_private: bool = False,
    custom_safe_domains: set[str] | None = None,
    trusted_private_networks: set[ipaddress.IPv4Network | ipaddress.IPv6Network] | None = None,
) -> tuple[bool, str | None]:
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
    if parsed.scheme not in ("http", "https"):
        return (
            False,
            f"Unsupported protocol: {parsed.scheme}. Only http and https are allowed.",
        )

    # Check hostname exists
    if not parsed.hostname:
        return False, "URL must have a valid hostname"

    if parsed.username or parsed.password:
        return False, "Credentials in provider URLs are not allowed"
    try:
        _ = parsed.port
    except ValueError:
        return False, "URL has an invalid port"

    hostname = parsed.hostname.lower()
    if "metadata" in hostname:
        return False, "Metadata endpoints are not allowed"

    # getaddrinfo()/HTTP clients accept integer, hex, octal and shortened IPv4
    # forms. Reject them before applying an allowlist to textual hostnames.
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
        try:
            socket.inet_aton(hostname)
        except OSError:
            pass
        else:
            return False, f"Non-canonical IP address is not allowed: {hostname}"

    if address is not None:
        if address.is_link_local or address.is_multicast or address.is_unspecified:
            return False, f"Special-use IP address is not allowed: {hostname}"
        if not address.is_global:
            if allow_private and (
                address.is_loopback
                or (custom_safe_domains and hostname in custom_safe_domains)
                or any(
                    address.version == network.version and address in network
                    for network in trusted_private_networks or ()
                )
            ):
                return True, None
            return False, f"Private/internal IP address must be explicitly trusted: {hostname}"
        if custom_safe_domains and hostname in custom_safe_domains:
            return True, None
        return False, f"IP endpoint must be explicitly trusted: {hostname}"

    if hostname == "localhost":
        if allow_private:
            return True, None
        return False, "Local addresses are not allowed: localhost"

    if any(
        hostname == domain or hostname.endswith("." + domain) for domain in DEFAULT_SAFE_DOMAINS
    ):
        return True, None

    if custom_safe_domains and hostname in custom_safe_domains:
        try:
            resolved = {
                ipaddress.ip_address(info[4][0])
                for info in socket.getaddrinfo(
                    hostname, parsed.port or 443, type=socket.SOCK_STREAM
                )
            }
        except (OSError, ValueError) as exc:
            return False, f"Trusted endpoint could not be resolved: {exc}"
        if not resolved:
            return False, "Trusted endpoint has no resolved address"
        if any(ip.is_link_local or ip.is_multicast or ip.is_unspecified for ip in resolved):
            return False, "Trusted endpoint resolves to a special-use address"
        if any(not ip.is_global for ip in resolved) and not allow_private:
            return False, "Trusted endpoint resolves to a private/internal address"
        return True, None

    return False, f"Endpoint host is not trusted: {hostname}"


def validate_and_normalize_url(
    url: str,
    allow_private: bool = False,
    require_https: bool = False,
    custom_safe_domains: set[str] | None = None,
    trusted_private_networks: set[ipaddress.IPv4Network | ipaddress.IPv6Network] | None = None,
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
    is_safe, error = is_safe_url(
        url,
        allow_private=allow_private,
        custom_safe_domains=custom_safe_domains,
        trusted_private_networks=trusted_private_networks,
    )
    if not is_safe:
        raise SSRFError(f"URL validation failed: {error}")

    parsed = urlparse(url)

    # Enforce HTTPS if required
    if require_https and parsed.scheme != "https":
        raise SSRFError("HTTPS is required for API calls")

    # Normalize: ensure path doesn't end with duplicate slashes
    normalized = url.rstrip("/")
    if (
        not normalized.endswith("/v1")
        and not normalized.endswith("/v1/")
        and "/" not in normalized.split("://", 1)[1]
    ):
        # Add trailing slash for consistency if no path
        normalized = normalized + "/"

    return normalized
