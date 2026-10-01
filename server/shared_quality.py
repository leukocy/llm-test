"""Ordered A/B quality jobs reuse the first side's frozen, hashed sample scopes."""

from __future__ import annotations

import hashlib

from server.checkpoints import CheckpointConflict, JobJournal, _decode, _encode

PROTOCOL = "shared-quality-v1"


class SharedQualityJournal(JobJournal):
    requires_shared_source = True

    def __init__(self, store, job, worker_id, endpoint):
        tag = job["parameters"].get("_comparison") or {}
        if tag.get("protocol") != PROTOCOL or not job.get("parent_job_id"):
            raise CheckpointConflict("Missing shared comparison identity")
        batch = store.get_batch(job["parent_job_id"])
        with store._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM control_jobs WHERE parent_job_id=? ORDER BY created_at,rowid",
                (job["parent_job_id"],),
            ).fetchall()
        members = [store._as_job(row) for row in rows]
        if (
            batch["max_parallel"] != 1
            or len(members) != 2
            or any(member["test_type"] != "quality" for member in members)
        ):
            raise CheckpointConflict("Shared comparison requires two ordered quality jobs")
        self.leader_id = members[0]["job_id"]
        self.role = "A" if job["job_id"] == self.leader_id else "B"
        if tag.get("role") != self.role or endpoint.model_id != job["model_id"]:
            raise CheckpointConflict("Comparison role or model changed")
        snapshot = {
            field: getattr(endpoint, field)
            for field in ("model_id", "provider", "api_base_url", "tokenizer_option")
        }
        if tag.get("endpoint_snapshot") != snapshot:
            raise CheckpointConflict("Comparison endpoint identity changed after submission")
        common = [
            {key: value for key, value in member["parameters"].items() if key != "_comparison"}
            for member in members
        ]
        if common[0] != common[1]:
            raise CheckpointConflict("Comparison sides have different quality parameters")
        super().__init__(store, job, worker_id, endpoint)

    def prepare_scope(self, key, factory, *, allow_empty=False):
        if self.role == "A":

            def freeze():
                metadata, inputs = factory()
                digest = hashlib.sha256(_encode(metadata).encode())
                for item in inputs:
                    raw = _encode(item).encode()
                    digest.update(len(raw).to_bytes(8, "big"))
                    digest.update(raw)
                return {
                    **metadata,
                    "shared_plan": {
                        "protocol": PROTOCOL,
                        "batch_id": self.job["parent_job_id"],
                        "leader_job_id": self.leader_id,
                        "sha256": digest.hexdigest(),
                    },
                }, inputs

            return super().prepare_scope(key, freeze, allow_empty=allow_empty)

        def reuse():
            with self.store._connection() as conn:
                self._lease(conn)
                scope = conn.execute(
                    "SELECT * FROM checkpoint_scopes WHERE job_id=? AND scope_key=?",
                    (self.leader_id, key),
                ).fetchone()
                if scope is None:
                    raise CheckpointConflict(
                        "Model A did not freeze the complete comparison plan; no Model B requests will be issued"
                    )
                metadata = _decode(scope["metadata_json"], scope["metadata_sha256"])
                rows = conn.execute(
                    "SELECT * FROM checkpoint_units WHERE job_id=? AND scope_key=? ORDER BY unit_index",
                    (self.leader_id, key),
                ).fetchall()
            if [row["unit_index"] for row in rows] != list(range(scope["units"])):
                raise CheckpointConflict("Shared sample membership changed")
            inputs = [_decode(row["input_json"], row["input_sha256"]) for row in rows]
            plan = metadata.get("shared_plan") or {}
            if (
                plan.get("leader_job_id") != self.leader_id
                or plan.get("batch_id") != self.job["parent_job_id"]
                or plan.get("protocol") != PROTOCOL
            ):
                raise CheckpointConflict("Shared plan identity changed")
            digest = hashlib.sha256(
                _encode(
                    {name: value for name, value in metadata.items() if name != "shared_plan"}
                ).encode()
            )
            for item in inputs:
                raw = _encode(item).encode()
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
            if digest.hexdigest() != plan.get("sha256"):
                raise CheckpointConflict("Shared plan content changed")
            return metadata, inputs

        return super().prepare_scope(key, reuse, allow_empty=allow_empty)
