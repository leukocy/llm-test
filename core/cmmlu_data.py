"""Read the pinned official CMMLU CSV archive without executing dataset scripts."""

import csv
import io
import json
import re
import zipfile
from pathlib import Path

MAX_ARCHIVE_BYTES = 64 * 1024**2


def prepare_archive(archive_path: str, target: Path, progress=None) -> None:
    if Path(archive_path).stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("CMMLU archive exceeds 64 MiB")
    merged: dict[str, list[dict]] = {"test": [], "dev": []}
    subjects: dict[str, set[str]] = {"test": set(), "dev": set()}
    with zipfile.ZipFile(archive_path) as archive:
        members = [member for member in archive.infolist() if not member.is_dir()]
        if sum(member.file_size for member in members) > MAX_ARCHIVE_BYTES:
            raise ValueError("CMMLU expanded data exceeds 64 MiB")
        for index, member in enumerate(members):
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("CMMLU archive contains a symbolic link")
            # v1.0.1 ships a 158-byte test/.zip artifact. Never open or extract it.
            if member.filename == "test/.zip":
                continue
            match = re.fullmatch(r"(test|dev)/([a-z][a-z0-9_]*)\.csv", member.filename)
            if not match:
                raise ValueError("CMMLU archive contains unsupported entries")
            split, subject = match.groups()
            if subject in subjects[split]:
                raise ValueError("Duplicate CMMLU subject file")
            subjects[split].add(subject)
            reader = csv.DictReader(io.StringIO(archive.read(member).decode("utf-8-sig")))
            if reader.fieldnames != ["", "Question", "A", "B", "C", "D", "Answer"]:
                raise ValueError("Invalid CMMLU CSV header")
            records = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("Invalid CMMLU CSV row")
                identifier = row.pop("")
                records.append({**row, "id": identifier, "subject": subject})
            from evaluators.cmmlu_evaluator import CMMLUEvaluator

            CMMLUEvaluator.normalize(records, split)
            if not records or (split == "dev" and len(records) != 5):
                raise ValueError("CMMLU requires scoring rows and five dev examples per subject")
            merged[split].extend(records)
            if progress:
                progress(0.1 + 0.8 * (index + 1) / len(members), f"CMMLU {split}/{subject}")
    if subjects["test"] != subjects["dev"] or len(subjects["test"]) != 67:
        raise ValueError("CMMLU requires matching test/dev files for all 67 subjects")
    for split, rows in merged.items():
        (target / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False), encoding="utf-8"
        )
    (target / "metadata.json").write_text(
        json.dumps({"name": "cmmlu", "subjects": sorted(subjects["test"]), "splits": list(merged)}),
        encoding="utf-8",
    )
