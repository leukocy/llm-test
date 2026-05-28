"""
Secure configuration management for API keys and sensitive data.
Loads from environment variables with fallback to .env file.
"""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: Optional[str] = None) -> None:
    """
    Load environment variables from .env file.
    This is a simplified version - for production, consider using python-dotenv.

    Args:
        env_path: Path to .env file. If None, looks in project root.
    """
    if env_path is None:
        # Get project root (3 levels up from this file)
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / '.env'

    if not env_path.exists():
        return

    with open(env_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())


def get_api_key(service: str, key: Optional[str] = None) -> Optional[str]:
    """
    Get API key from environment with optional fallback.

    Args:
        service: Service name (e.g., 'aliyun', 'gemini')
        key: Direct key value (for backward compatibility, deprecated)

    Returns:
        API key from environment or provided key
    """
    # Try environment variable first
    env_key = os.getenv(f'{service.upper()}_API_KEY')
    if env_key:
        return env_key

    # Fallback to provided key (for backward compatibility)
    if key:
        import warnings
        warnings.warn(
            f"Using hardcoded API key for {service}. "
            f"Set {service.upper()}_API_KEY environment variable instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return key

    return None


# Initialize environment on module load
load_env_file()
