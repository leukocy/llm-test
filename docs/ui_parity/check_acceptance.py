"""Audit every original ID and evidence freshness without running models or deploying."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess  # nosec B404
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
INVENTORIES = ("controls", "outputs", "html_actions")


def inventory() -> dict[str, dict[str, str]]:
    records = {}
    for name in INVENTORIES:
        with (DIRECTORY / f"first_commit_{name}.csv").open() as source:
            for row in csv.DictReader(source):
                if row["id"] in records:
                    raise ValueError(f"Duplicate original ID: {row['id']}")
                records[row["id"]] = row
    return records


def audit(ledger: Path, evidence_root: Path) -> dict:
    original = inventory()
    with ledger.open() as source:
        rows = list(csv.DictReader(source))
    counts = Counter(row["id"] for row in rows)
    missing = sorted(set(original) - set(counts))
    unknown = sorted(set(counts) - set(original))
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    errors = []
    statuses: Counter[str] = Counter()
    stale = []
    for row in rows:
        key = row["id"]
        if key not in original:
            continue
        if (
            row["original_source"] != original[key]["source"]
            or row["original_reachability"] != original[key]["reachability"]
        ):
            errors.append(f"{key}: original metadata changed")
        status = row["status"]
        if status not in {"pending", "verified", "gap"}:
            errors.append(f"{key}: invalid status")
        statuses[status] += 1
        paths = row["candidate_sources"].split(";") if row["candidate_sources"] else []
        for path in paths:
            resolved = (ROOT / path).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.is_file():
                errors.append(f"{key}: invalid source pointer")
        if status != "verified":
            continue
        if not paths or not row["acceptance"] or not row["evidence_check"]:
            errors.append(f"{key}: incomplete acceptance")
            continue
        try:
            proof = (evidence_root / row["evidence_file"]).resolve()
            if not proof.is_relative_to(evidence_root.resolve()) or proof.suffix != ".json":
                raise ValueError("evidence must be a JSON file under the evidence root")
            raw_evidence = proof.read_bytes()
            if hashlib.sha256(raw_evidence).hexdigest() != row["evidence_sha256"]:
                raise ValueError(
                    "runtime evidence changed; review and bind the new proof explicitly"
                )
            evidence = json.loads(raw_evidence)
            required_checks = json.loads(row["evidence_check"])
            if (
                not isinstance(required_checks, list)
                or not required_checks
                or not all(isinstance(check, str) for check in required_checks)
            ):
                raise ValueError("acceptance requires a nonempty list of runtime checks")
            if (
                evidence.get("errors") != []
                or evidence.get("external_requests") != []
                or not all(check in evidence.get("checks", []) for check in required_checks)
            ):
                raise ValueError("runtime evidence does not prove the acceptance")
            commit = row["tested_commit"]
            if not commit or any(char not in "0123456789abcdef" for char in commit):
                raise ValueError("invalid tested commit")
            # Fixed executable, argument list, local Git reads only.
            ancestor = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                cwd=ROOT,
                capture_output=True,
                check=False,
            )  # nosec B603 B607
            if ancestor.returncode:
                raise ValueError("tested commit is not an ancestor of HEAD")
            changed = subprocess.run(
                ["git", "diff", "--name-only", commit, "--", *paths],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )  # nosec B603 B607
            untracked = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            )  # nosec B603 B607
            if changed.stdout.strip() or untracked.stdout.strip():
                stale.append(key)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            errors.append(f"{key}: {exc}")
    valid = not (missing or unknown or duplicates or errors)
    return {
        "inventory_total": len(original),
        "ledger_total": len(rows),
        "statuses": dict(statuses),
        "missing_ids": missing,
        "unknown_ids": unknown,
        "duplicate_ids": duplicates,
        "errors": errors,
        "stale_ids": stale,
        "ledger_valid": valid,
        "all_verified": valid and not stale and statuses["verified"] == len(original),
        "note": "Evidence presence and unchanged source establish freshness, not independent proof of test adequacy. Each acceptance still requires human review; dynamic choices have a separate audit.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DIRECTORY / "acceptance.csv")
    parser.add_argument("--evidence-root", type=Path, default=Path.home() / "llm-perf/evaluation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.ledger, args.evidence_root)
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    passed = result["all_verified"] if args.require_complete else result["ledger_valid"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
