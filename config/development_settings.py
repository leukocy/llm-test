"""
Development-specific settings with internal servers.

WARNING: This module should ONLY be used for local development and NEVER committed.
Internal IP addresses should not be exposed in production configuration.
"""

import os


def get_development_providers():
    """
    Get development-specific provider configurations.

    These are NOT included in PROVIDER_OPTIONS to prevent accidental exposure.
    To use internal servers, users must configure them manually via the UI or environment variables.

    Returns:
        dict: Development provider options if in dev mode, empty dict otherwise
    """
    # Check if we're in development mode
    IS_DEVELOPMENT = os.getenv("LLM_TEST_DEV", "false").lower() == "true"

    if not IS_DEVELOPMENT:
        return {}

    # Internal providers - only loaded in development mode
    # Use environment variables when possible
    DEVELOPMENT_PROVIDERS = {
        name: url
        for name, url in {
            f"Dev{i}": os.getenv(f"DEV{i}_URL") for i in range(1, 10)
        }.items()
        if url
    }

    return DEVELOPMENT_PROVIDERS


def load_development_settings():
    """
    Load development settings into the main configuration.

    This function should be called at app startup if LLM_TEST_DEV=true.
    """
    providers = get_development_providers()
    if providers:
        # Import here to avoid circular dependency
        from .settings import PROVIDER_OPTIONS

        PROVIDER_OPTIONS.update(providers)
    return providers
