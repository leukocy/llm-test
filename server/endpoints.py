"""Persistent measurement API endpoints shared by the control API and worker."""

from __future__ import annotations

import base64
import hashlib
import os
import sqlite3
import time
import uuid
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

from server.settings import Endpoint, Settings, validate_endpoint
from server.store import JobStore


class EndpointNotFound(LookupError):
    pass


class EndpointConflict(ValueError):
    pass


class EndpointCredentialUnavailable(RuntimeError):
    pass


_ACTIVE_STATES = ("created", "queued", "running", "pausing", "paused", "cancelling")
_MANAGED_KEY_ENV = "LLM_TEST_MANAGED_CREDENTIAL"


class EndpointRegistry:
    """Keep credentials encrypted in the shared database and out of job records."""

    def __init__(self, settings: Settings, store: JobStore) -> None:
        self.settings = settings
        self.store = store
        secret = os.getenv("LLM_TEST_ENDPOINT_ENCRYPTION_KEY") or settings.api_token
        if len(secret) < 32:
            raise ValueError("LLM_TEST_ENDPOINT_ENCRYPTION_KEY must contain at least 32 characters")
        key_material = hashlib.sha256(
            b"llm-test/endpoint-credentials/v1\0" + secret.encode("utf-8")
        ).digest()
        self._cipher = Fernet(base64.urlsafe_b64encode(key_material))

    def _public_managed(self, row: sqlite3.Row) -> dict[str, str | bool]:
        try:
            self._cipher.decrypt(row["credential_ciphertext"].encode("ascii"))
            credential_configured = True
        except (InvalidToken, UnicodeError):
            credential_configured = False
        return {
            "id": row["endpoint_id"],
            "label": row["label"],
            "provider": row["provider"],
            "api_base_url": row["api_base_url"],
            "model_id": row["model_id"],
            "tokenizer_option": row["tokenizer_option"],
            "credential_configured": credential_configured,
            "source": "managed",
        }

    def list_public(self) -> list[dict[str, str | bool]]:
        with self.store._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM control_endpoints ORDER BY updated_at DESC, endpoint_id"
            ).fetchall()
        return [self._public_managed(row) for row in rows] + [
            endpoint.public() for endpoint in self.settings.endpoints.values()
        ]

    def get(self, endpoint_id: str) -> Endpoint:
        static = self.settings.endpoints.get(endpoint_id)
        if static is not None:
            return static
        with self.store._connection() as conn:
            row = conn.execute(
                "SELECT * FROM control_endpoints WHERE endpoint_id = ?", (endpoint_id,)
            ).fetchone()
        if row is None:
            raise EndpointNotFound(endpoint_id)
        try:
            api_key = self._cipher.decrypt(row["credential_ciphertext"].encode("ascii")).decode(
                "utf-8"
            )
        except (InvalidToken, UnicodeError) as exc:
            raise EndpointCredentialUnavailable(
                "Stored endpoint credential cannot be decrypted"
            ) from exc
        endpoint = Endpoint(
            id=row["endpoint_id"],
            label=row["label"],
            provider=row["provider"],
            api_base_url=row["api_base_url"],
            model_id=row["model_id"],
            api_key_env=_MANAGED_KEY_ENV,
            tokenizer_option=row["tokenizer_option"],
            api_key_value=api_key,
        )
        validate_endpoint(endpoint)
        return endpoint

    @staticmethod
    def _check_unused(conn: sqlite3.Connection, endpoint_id: str) -> None:
        placeholders = ",".join("?" for _ in _ACTIVE_STATES)
        active = conn.execute(
            f"SELECT 1 FROM control_jobs WHERE endpoint_id = ? AND status IN ({placeholders}) LIMIT 1",
            (endpoint_id, *_ACTIVE_STATES),
        ).fetchone()
        if active:
            raise EndpointConflict("该端点有待执行或运行中的任务，完成后再修改")

    def save(
        self,
        *,
        label: str,
        provider: str,
        api_base_url: str,
        model_id: str,
        tokenizer_option: str = "auto",
        api_key: str | None = None,
        endpoint_id: str | None = None,
    ) -> dict[str, Any]:
        identifier = endpoint_id or f"api-{uuid.uuid4().hex[:12]}"
        if identifier in self.settings.endpoints:
            raise EndpointConflict("文件预设端点只能在管理员配置文件中修改")
        endpoint = Endpoint(
            identifier,
            label.strip(),
            provider,
            api_base_url.strip(),
            model_id.strip(),
            _MANAGED_KEY_ENV,
            tokenizer_option.strip() or "auto",
        )
        validate_endpoint(endpoint)
        if api_key is not None:
            api_key = api_key.strip()
            if not api_key or len(api_key) > 4096 or any(char in api_key for char in "\r\n\0"):
                raise ValueError("Invalid API credential")
            ciphertext = self._cipher.encrypt(api_key.encode("utf-8")).decode("ascii")
        else:
            ciphertext = None
        now = time.time()
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT * FROM control_endpoints WHERE endpoint_id = ?", (identifier,)
            ).fetchone()
            if endpoint_id and existing is None:
                raise EndpointNotFound(identifier)
            if existing is not None:
                self._check_unused(conn, identifier)
                ciphertext = ciphertext or existing["credential_ciphertext"]
                conn.execute(
                    """UPDATE control_endpoints SET label = ?, provider = ?, api_base_url = ?,
                       model_id = ?, tokenizer_option = ?, credential_ciphertext = ?, updated_at = ?
                       WHERE endpoint_id = ?""",
                    (
                        endpoint.label,
                        endpoint.provider,
                        endpoint.api_base_url,
                        endpoint.model_id,
                        endpoint.tokenizer_option,
                        ciphertext,
                        now,
                        identifier,
                    ),
                )
            else:
                if not ciphertext:
                    raise ValueError("API key is required for a new endpoint")
                conn.execute(
                    """INSERT INTO control_endpoints
                       (endpoint_id, label, provider, api_base_url, model_id,
                        tokenizer_option, credential_ciphertext, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        identifier,
                        endpoint.label,
                        endpoint.provider,
                        endpoint.api_base_url,
                        endpoint.model_id,
                        endpoint.tokenizer_option,
                        ciphertext,
                        now,
                        now,
                    ),
                )
            row = conn.execute(
                "SELECT * FROM control_endpoints WHERE endpoint_id = ?", (identifier,)
            ).fetchone()
            conn.commit()
        if row is None:
            raise EndpointNotFound(identifier)
        return self._public_managed(row)

    def delete(self, endpoint_id: str) -> None:
        if endpoint_id in self.settings.endpoints:
            raise EndpointConflict("文件预设端点只能在管理员配置文件中删除")
        with self.store._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._check_unused(conn, endpoint_id)
            cursor = conn.execute(
                "DELETE FROM control_endpoints WHERE endpoint_id = ?", (endpoint_id,)
            )
            if cursor.rowcount == 0:
                raise EndpointNotFound(endpoint_id)
            conn.commit()
