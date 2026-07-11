"""Pure services for saved benchmark result history."""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config.test_types import DEFAULT_TEST_TYPE, normalize_test_type

LAST_RESULT_FILENAME = ".last_result.json"

_KNOWN_TEST_TYPES = {
    "Concurrency_Test": "concurrency",
    "Prefill_Stress_Test": "prefill",
    "Segmented_Context_Test": "segmented",
    "Long_Context_Test": "long_context",
    "Concurrency_Context_Matrix_Test": "matrix",
    "Custom_Text_Test": "custom",
    "All_Tests": "all",
    "Stability_Test": "stability",
}


def _last_result_path(base_dir: str | os.PathLike[str]) -> Path:
    return Path(base_dir) / LAST_RESULT_FILENAME


def _metadata_path(csv_path: str | os.PathLike[str]) -> Path:
    path = Path(csv_path)
    return path.with_suffix(path.suffix + ".meta.json")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _newest_csv(base_dir: str | os.PathLike[str]) -> Path | None:
    pattern = str(Path(base_dir) / "**" / "*.csv")
    csv_files = [Path(path) for path in glob.glob(pattern, recursive=True)]
    csv_files = [path for path in csv_files if path.is_file()]
    if not csv_files:
        return None
    return max(csv_files, key=lambda path: path.stat().st_mtime)


def _infer_model_id(csv_path: Path, base_dir: str | os.PathLike[str]) -> str:
    try:
        relative = csv_path.resolve().relative_to(Path(base_dir).resolve())
        if len(relative.parts) > 1:
            return relative.parts[0]
    except ValueError:
        pass
    return ""


def _infer_test_type(csv_path: Path) -> str:
    filename = csv_path.stem
    for raw, label in sorted(
        _KNOWN_TEST_TYPES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if raw in filename:
            return label

    match = re.search(r"benchmark_results_.+?_(.+)_\d{8}_\d{6}$", filename)
    if match:
        raw_type = match.group(1).replace("_", " ")
        return normalize_test_type(raw_type)

    return DEFAULT_TEST_TYPE


def _build_inferred_snapshot(
    csv_path: str | os.PathLike[str],
    base_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    path = Path(csv_path)
    return {
        "csv_path": str(path),
        "test_type": _infer_test_type(path),
        "model_id": _infer_model_id(path, base_dir),
        "provider": "Unknown",
        "duration": 0.0,
        "test_config": {},
        "system_info": {},
    }


def _snapshot_for_csv(
    csv_path: str | os.PathLike[str],
    base_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    path = Path(csv_path)
    snapshot = _read_json(_metadata_path(path))
    if not snapshot:
        snapshot = _build_inferred_snapshot(path, base_dir)
    snapshot["csv_path"] = str(path)
    snapshot.setdefault("test_type", _infer_test_type(path))
    snapshot.setdefault("model_id", _infer_model_id(path, base_dir))
    snapshot.setdefault("provider", "Unknown")
    snapshot.setdefault("duration", 0.0)
    snapshot.setdefault("test_config", {})
    snapshot.setdefault("system_info", {})
    snapshot["test_type"] = normalize_test_type(snapshot.get("test_type"))
    return snapshot


def _load_snapshot(
    base_dir: str | os.PathLike[str],
    *,
    fallback_to_latest_csv: bool = True,
) -> dict[str, Any]:
    snapshot_path = _last_result_path(base_dir)
    if snapshot_path.exists():
        snapshot = _read_json(snapshot_path)
        csv_path = Path(snapshot.get("csv_path", ""))
        if csv_path.exists():
            return _snapshot_for_csv(csv_path, base_dir) | snapshot

    if not fallback_to_latest_csv:
        return {}

    csv_path = _newest_csv(base_dir)
    if not csv_path:
        return {}

    return _snapshot_for_csv(csv_path, base_dir)


def _restore_snapshot_to_session(
    session_state: Any, snapshot: dict[str, Any], csv_path: Path
) -> bool:
    try:
        df = pd.read_csv(csv_path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return False

    if df.empty:
        return False

    session_state["results_df"] = df
    session_state["current_csv_file"] = str(csv_path)
    session_state["current_test_type"] = normalize_test_type(
        snapshot.get("test_type") or _infer_test_type(csv_path)
    )
    session_state["current_model_id"] = snapshot.get("model_id") or ""
    session_state["current_provider"] = snapshot.get("provider") or "Unknown"
    session_state["test_duration"] = float(snapshot.get("duration") or 0)
    session_state["test_config"] = snapshot.get("test_config") or {}
    session_state["system_info"] = snapshot.get("system_info") or {}
    session_state["restored_result_context"] = {
        "test_type": session_state["current_test_type"],
        "model_id": session_state["current_model_id"],
        "provider": session_state["current_provider"],
        "duration": session_state["test_duration"],
        "test_config": session_state["test_config"],
        "system_info": session_state["system_info"],
    }
    session_state["restored_from_csv"] = True
    return True


def save_last_result_snapshot(
    *,
    csv_path: str | os.PathLike[str],
    test_type: str,
    model_id: str,
    provider: str,
    duration: float,
    test_config: dict[str, Any] | None = None,
    system_info: dict[str, Any] | None = None,
    base_dir: str | os.PathLike[str] = "raw_data",
) -> None:
    """Save a small pointer to the latest run so the UI can rebuild charts after refresh."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "csv_path": str(Path(csv_path)),
        "test_type": normalize_test_type(test_type),
        "model_id": model_id,
        "provider": provider,
        "duration": float(duration or 0),
        "test_config": test_config or {},
        "system_info": system_info or {},
    }

    with _last_result_path(base).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with _metadata_path(csv_path).open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def list_saved_results(
    *,
    base_dir: str | os.PathLike[str] = "raw_data",
) -> list[dict[str, Any]]:
    """List saved CSV result files with available metadata, newest first."""
    pattern = str(Path(base_dir) / "**" / "*.csv")
    csv_files = [Path(path) for path in glob.glob(pattern, recursive=True)]
    rows = []
    for csv_path in csv_files:
        if not csv_path.is_file():
            continue
        snapshot = _snapshot_for_csv(csv_path, base_dir)
        stat = csv_path.stat()
        rows.append(
            {
                "path": str(csv_path),
                "display_name": csv_path.name,
                "model_id": snapshot.get("model_id", ""),
                "test_type": snapshot.get("test_type", ""),
                "provider": snapshot.get("provider", "Unknown"),
                "modified_time": stat.st_mtime,
                "size_kb": stat.st_size / 1024,
            }
        )

    rows.sort(key=lambda item: (item["modified_time"], item["path"]), reverse=True)
    return rows


def _write_last_snapshot_for_latest(base_dir: str | os.PathLike[str]) -> None:
    base = Path(base_dir)
    latest_csv = _newest_csv(base)
    last_path = _last_result_path(base)
    if not latest_csv:
        try:
            last_path.unlink()
        except FileNotFoundError:
            pass
        return

    snapshot = _snapshot_for_csv(latest_csv, base)
    base.mkdir(parents=True, exist_ok=True)
    with last_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)


def delete_saved_result(
    csv_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] = "raw_data",
) -> bool:
    """Delete one saved result CSV and its metadata, then repair latest-result pointer."""
    path = Path(csv_path)
    deleted = False

    try:
        path.unlink()
        deleted = True
    except FileNotFoundError:
        pass
    except OSError:
        return False

    try:
        _metadata_path(path).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        return False

    _write_last_snapshot_for_latest(base_dir)
    return deleted


def restore_result_file_to_session(
    session_state: Any,
    csv_path: str | os.PathLike[str],
    *,
    base_dir: str | os.PathLike[str] = "raw_data",
) -> bool:
    """Restore one selected CSV result file into session state."""
    path = Path(csv_path)
    if not path.exists():
        return False
    snapshot = _snapshot_for_csv(path, base_dir)
    return _restore_snapshot_to_session(session_state, snapshot, path)


def restore_last_results_to_session(
    session_state: Any,
    *,
    base_dir: str | os.PathLike[str] = "raw_data",
    fallback_to_latest_csv: bool = True,
) -> bool:
    """Restore latest saved results into a dict-like Streamlit session state."""
    snapshot = _load_snapshot(base_dir, fallback_to_latest_csv=fallback_to_latest_csv)
    csv_path_raw = snapshot.get("csv_path")
    if not csv_path_raw:
        return False

    csv_path = Path(csv_path_raw)
    if not csv_path.exists():
        return False

    return _restore_snapshot_to_session(session_state, snapshot, csv_path)
