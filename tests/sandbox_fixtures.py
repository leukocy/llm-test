"""Synthetic environment evidence for tests that never launch containers."""

from core.sandbox_identity import VERSION, identity_digest


def sandbox_identity(image="a"):
    evidence = {
        "version": VERSION,
        "image_id": "sha256:" + image * 64,
        "policy_sha256": "b" * 64,
        "cpu_sha256": "c" * 64,
        "runtime": {
            "ServerVersion": "test-engine",
            "KernelVersion": "test-kernel",
            "Architecture": "test-arch",
            "OperatingSystem": "test-os",
        },
    }
    return {**evidence, "sha256": identity_digest(evidence)}
