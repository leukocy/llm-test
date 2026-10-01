"""Canonical, credential-free evidence for isolated code grading."""

import hashlib
import json
import re
from typing import Any

VERSION = "sandbox-environment-v1"
FIELDS = ("version", "image_id", "policy_sha256", "runtime", "cpu_sha256")


def identity_digest(evidence: dict[str, Any]) -> str:
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {*FIELDS, "sha256"}:
        raise ValueError("missing sandbox environment evidence")
    if value["version"] != VERSION or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", str(value["image_id"])
    ):
        raise ValueError("invalid sandbox environment version/image")
    for field in ("policy_sha256", "cpu_sha256", "sha256"):
        if not isinstance(value[field], str) or not re.fullmatch(r"[0-9a-f]{64}", value[field]):
            raise ValueError("invalid sandbox environment digest")
    runtime = value["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "ServerVersion",
        "KernelVersion",
        "Architecture",
        "OperatingSystem",
    }:
        raise ValueError("invalid sandbox runtime evidence")
    if any(not isinstance(item, str) or not item or len(item) > 256 for item in runtime.values()):
        raise ValueError("invalid sandbox runtime value")
    evidence = {field: value[field] for field in FIELDS}
    if value["sha256"] != identity_digest(evidence):
        raise ValueError("sandbox evidence checksum mismatch")
    return {**evidence, "sha256": value["sha256"]}
