"""Administrator-owned configuration for the single-tenant control plane."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from core.providers.factory import _validate_base_url

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Endpoint:
    id: str
    label: str
    provider: str
    api_base_url: str
    model_id: str
    api_key_env: str
    tokenizer_option: str = "auto"

    def public(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "provider": self.provider,
            "model_id": self.model_id,
        }

    def api_key(self) -> str:
        value = os.getenv(self.api_key_env, "")
        if not value:
            raise RuntimeError(f"Credential environment variable {self.api_key_env} is unset")
        return value


@dataclass(frozen=True)
class Settings:
    api_token: str
    db_path: Path
    artifact_root: Path
    endpoints: dict[str, Endpoint]
    worker_poll_seconds: float = 2.0

    @classmethod
    def from_env(cls) -> Settings:
        token = os.getenv("LLM_TEST_API_TOKEN", "")
        if len(token) < 32:
            raise ValueError("LLM_TEST_API_TOKEN must contain at least 32 characters")
        endpoint_file = Path(os.getenv("LLM_TEST_ENDPOINTS_FILE", "config/endpoints.json"))
        try:
            data = json.loads(endpoint_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load endpoint configuration: {endpoint_file}") from exc
        if not isinstance(data, list) or not data:
            raise ValueError("Endpoint configuration must be a nonempty JSON array")
        endpoints: dict[str, Endpoint] = {}
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Each endpoint must be a JSON object")
            endpoint = Endpoint(**item)
            if not _IDENTIFIER.fullmatch(endpoint.id) or endpoint.id in endpoints:
                raise ValueError(f"Invalid or duplicate endpoint ID: {endpoint.id}")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", endpoint.api_key_env):
                raise ValueError(f"Invalid credential environment variable: {endpoint.id}")
            if (
                not endpoint.model_id
                or len(endpoint.model_id) > 200
                or not re.fullmatch(r"[A-Za-z0-9._/@:+-]+", endpoint.model_id)
                or endpoint.model_id.startswith("/")
                or any(part in {"", ".", ".."} for part in endpoint.model_id.split("/"))
            ):
                raise ValueError(f"Invalid model ID: {endpoint.id}")
            if not endpoint.api_base_url:
                raise ValueError(f"Empty API URL: {endpoint.id}")
            _validate_base_url(endpoint.api_base_url)
            endpoints[endpoint.id] = endpoint
        return cls(
            api_token=token,
            db_path=Path(os.getenv("LLM_TEST_DB_PATH", "data/benchmark.db")),
            artifact_root=Path(os.getenv("LLM_TEST_ARTIFACT_ROOT", "results/jobs")),
            endpoints=endpoints,
        )
