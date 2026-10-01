"""The parity gate must reject omissions and stale proof, not just count rows."""

import csv
import hashlib
import json
from subprocess import CompletedProcess

import pytest

from docs.ui_parity.check_acceptance import DIRECTORY, audit


@pytest.fixture
def ledger(tmp_path):
    with (DIRECTORY / "acceptance.csv").open() as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["status"] = "pending"
    path = tmp_path / "acceptance.csv"

    def write(items):
        with path.open("w") as target:
            writer = csv.DictWriter(target, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(items)

    return path, rows, write


def test_complete_inventory_is_not_complete_acceptance(ledger, tmp_path):
    path, rows, write = ledger
    write(rows)
    result = audit(path, tmp_path)
    assert result["ledger_valid"] and result["inventory_total"] == 511
    assert not result["all_verified"] and result["statuses"] == {"pending": 511}


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "metadata"])
def test_invalid_ledger_cannot_pass(ledger, tmp_path, mutation):
    path, rows, write = ledger
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "unknown":
        rows[0]["id"] = "IC-9999"
    else:
        rows[0]["original_source"] = "wrong.py:1"
    write(rows)
    assert not audit(path, tmp_path)["ledger_valid"]


def test_stale_runtime_evidence_is_not_complete(ledger, tmp_path, monkeypatch):
    path, rows, write = ledger
    row = rows[0]
    row.update(
        status="verified",
        acceptance="specific behavior",
        candidate_sources="frontend/src/api.ts",
        evidence_file="proof.json",
        evidence_check=json.dumps(["passed"]),
        tested_commit="a" * 40,
    )
    (tmp_path / "proof.json").write_text(
        json.dumps({"checks": ["passed"], "errors": [], "external_requests": []})
    )

    row["evidence_sha256"] = hashlib.sha256((tmp_path / "proof.json").read_bytes()).hexdigest()

    def git_result(args, **kwargs):
        return CompletedProcess(args, 0, "frontend/src/api.ts\n" if args[1] == "diff" else "", "")

    monkeypatch.setattr("docs.ui_parity.check_acceptance.subprocess.run", git_result)
    write(rows)
    result = audit(path, tmp_path)
    assert result["ledger_valid"] and result["stale_ids"] == [row["id"]]
    assert not result["all_verified"]


def test_evidence_cannot_escape_the_selected_root(ledger, tmp_path):
    path, rows, write = ledger
    rows[0].update(
        status="verified",
        acceptance="specific behavior",
        evidence_file="../outside.json",
        evidence_check=json.dumps(["passed"]),
    )
    write(rows)
    assert not audit(path, tmp_path)["ledger_valid"]
