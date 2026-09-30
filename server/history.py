"""Bounded, stateless reports over saved CSV files; no measurement jobs are fabricated."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from config.test_types import TEST_TYPE_SPECS
from server.analytics import GROUP_AXIS, GROUP_FIELDS, NUMERIC_FIELDS, describe_observations

MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_METADATA_BYTES = 65536
MAX_ROWS = 20000
MAX_ENTRIES = 5000
MAX_GROUPS = 128
_SECRET = re.compile(
    r"api.?key|credential|secret|password|authorization|bearer|access.?token", re.I
)
_TYPES = {
    "concurrency": "Concurrency_Test",
    "prefill": "Prefill_Stress_Test",
    "segmented_prefill": "Segmented_Context_Test",
    "long_context": "Long_Context_Test",
    "throughput_matrix": "Concurrency_Context_Matrix_Test",
    "custom_text": "Custom_Text_Test",
    "stability": "Stability_Test",
}


class HistoryError(ValueError):
    """Safe messages for invalid history input."""


class HistoryConflict(RuntimeError):
    """The file changed after it was selected."""


class HistoryNotFound(LookupError):
    """No saved file has this opaque ID."""


def _clean(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if isinstance(value, dict):
        return {
            str(key)[:120]: _clean(item, depth + 1)
            for key, item in list(value.items())[:80]
            if not _SECRET.search(str(key)) and "url" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_clean(item, depth + 1) for item in value[:80]]
    if isinstance(value, str):
        return value[:2000]
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return None


def _kind(value: str) -> str:
    key = value.strip().casefold().replace("-", "_").replace(" ", "_")
    for spec in TEST_TYPE_SPECS:
        if key in {
            candidate.casefold().replace("-", "_").replace(" ", "_")
            for candidate in (spec.id, spec.label, *spec.aliases)
        }:
            normalized = {
                "segmented": "segmented_prefill",
                "matrix": "throughput_matrix",
                "custom": "custom_text",
            }.get(spec.id, spec.id)
            return normalized if normalized in _TYPES else "unknown"
    return "unknown"


def _bounded_read(path: Path, limit: int) -> bytes:
    # Do not follow a replaced final file symlink between discovery and opening.
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise HistoryError("历史数据必须是普通文件")
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise HistoryError("文件超过读取上限：CSV 10 MiB，元数据 64 KiB")
    return data


def _parse(
    raw: bytes,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, Any]], list[str], str]:
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        columns = list(reader.fieldnames or [])
        if (
            not 1 <= len(columns) <= 80
            or len(set(columns)) != len(columns)
            or any(
                not name or len(name) > 120 or any(ord(c) < 32 for c in name) for name in columns
            )
            or not set(columns).intersection(NUMERIC_FIELDS)
        ):
            raise HistoryError(
                "需要唯一列名及至少一个原始性能指标列（如 ttft 或 tps）；不自动猜测单位"
            )
        raw_rows, rows, warnings, versions = [], [], set(), set()
        for index, item in enumerate(reader, 1):
            if index > MAX_ROWS:
                raise HistoryError("CSV 超过 20,000 行，未计算截断统计")
            if None in item or any(value is None for value in item.values()):
                raise HistoryError(f"CSV 第 {index} 行列数不一致")
            raw_rows.append(item)
            extra = json.loads(item.get("extra_metrics") or "{}")
            if not isinstance(extra, dict):
                raise HistoryError(f"CSV 第 {index} 行指标元数据必须是 JSON 对象")
            explicit, embedded = (
                item.get("metric_contract_version", ""),
                extra.get("metric_contract_version", ""),
            )
            if explicit and embedded and explicit != embedded:
                raise HistoryError("CSV 的指标契约互相冲突")
            version = explicit or embedded or "legacy-unversioned"
            if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", version):
                raise HistoryError("CSV 的指标契约格式无效")
            versions.add(version)
            if len(versions) > 1:
                raise HistoryError("CSV 混用了不同指标契约，不能汇总")
            error = item.get("error", "").strip()
            error = "" if error.lower() in {"", "none", "null", "nan"} else error
            success = item.get("success", "").strip().lower()
            if success and success not in {"1", "0", "true", "false"}:
                raise HistoryError(f"CSV 第 {index} 行 success 必须是 1/0 或 true/false")
            if success in {"1", "true"} and error:
                raise HistoryError(f"CSV 第 {index} 行成功状态与错误字段冲突")
            outcome = (
                "failed"
                if error or success in {"0", "false"}
                else "success"
                if success or "error" in columns
                else "unknown"
            )
            row: dict[str, Any] = {
                "id": index,
                "outcome": outcome,
                "error": error or ("CSV success=0" if outcome == "failed" else None),
            }
            if item.get("concurrency_level", "").strip() and item.get("concurrency", "").strip():
                try:
                    same = float(item["concurrency_level"]) == float(item["concurrency"])
                except ValueError:
                    same = False
                if not same:
                    raise HistoryError(f"CSV 第 {index} 行的两个并发字段互相冲突")
            for name in (
                *NUMERIC_FIELDS,
                "prefill_tokens",
                "decode_tokens",
                "input_tokens_target",
                "context_length_target",
                "concurrency_level",
            ):
                original = item.get(
                    name, item.get("concurrency", "") if name == "concurrency_level" else ""
                )
                integer = name not in NUMERIC_FIELDS
                try:
                    number = float(original) if original.strip() else None
                    if number is not None and (
                        not math.isfinite(number)
                        or number < 0
                        or (integer and not number.is_integer())
                    ):
                        raise ValueError
                    row[name] = int(number) if integer and number is not None else number
                except (ValueError, OverflowError):
                    row[name] = None
                    warnings.add(f"{name} 包含无效或非有限数值，已标为未采集；未以零填充。")
            for name in ("token_source", "token_calc_method"):
                row[name] = item.get(name, "")[:200] or None
            rows.append(row)
        if not rows:
            raise HistoryError("CSV 只有表头或没有请求记录")
    except (UnicodeError, csv.Error, json.JSONDecodeError, RecursionError) as exc:
        raise HistoryError("CSV 必须是有效 UTF-8，且每行指标元数据必须是有效 JSON") from exc
    return columns, raw_rows, rows, sorted(warnings), versions.pop()


def _describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["outcome"] != "unknown"]
    result = describe_observations(known)
    result.update(requests=len(rows), unknown_outcomes=len(rows) - len(known))
    return result


class SavedCsvHistory:
    def __init__(self, root: Path):
        self.root = Path(os.path.realpath(root))
        self._lock = threading.RLock()

    def _scan(self) -> tuple[list[Path], bool, int]:
        found: list[Path] = []
        pending, scanned, skipped = [(self.root, 0)], 0, 0
        if not self.root.exists():
            return [], False, 0
        while pending:
            directory, depth = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    scanned += 1
                    if scanned > MAX_ENTRIES:
                        return found, True, skipped
                    if entry.name.startswith(".") or entry.is_symlink():
                        skipped += int(entry.is_symlink())
                        continue
                    if entry.is_dir(follow_symlinks=False) and depth < 4:
                        pending.append((Path(entry.path), depth + 1))
                    elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".csv"):
                        found.append(Path(entry.path))
        return found, False, skipped

    def _id(self, path: Path) -> str:
        return hashlib.sha256(path.relative_to(self.root).as_posix().encode()).hexdigest()

    def _safe(self, path: Path) -> Path:
        normalized = os.path.realpath(path)
        if not normalized.startswith(str(self.root).rstrip(os.sep) + os.sep) or path.is_symlink():
            raise HistoryError("历史文件路径越界或使用了软链接")
        return Path(normalized)

    def _resolve(self, identifier: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", identifier):
            raise HistoryNotFound(identifier)
        files, _, _ = self._scan()
        for path in files:
            if self._id(path) == identifier:
                return self._safe(path)
        raise HistoryNotFound(identifier)

    def _metadata_path(self, path: Path) -> Path:
        return self._safe(path.with_suffix(".csv.meta.json"))

    def _revision(self, path: Path) -> str:
        states: list[tuple[int, int, int, int, int] | None] = []
        for target in (path, self._metadata_path(path)):
            try:
                stat = target.stat()
                states.append(
                    (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                )
            except FileNotFoundError:
                states.append(None)
        return hashlib.sha256(json.dumps(states).encode()).hexdigest()

    def _metadata(self, path: Path) -> tuple[dict[str, Any], bytes, list[str]]:
        target = self._metadata_path(path)
        if not target.exists():
            return {}, b"", ["未提供配套元数据；模型和测试类型可能仅从文件名推断。"]
        try:
            raw = _bounded_read(target, MAX_METADATA_BYTES)
        except HistoryError:
            return {}, b"", ["配套元数据超过 64 KiB，未读取或用于推断测量条件。"]
        try:
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError
        except (ValueError, UnicodeError, RecursionError):
            return {}, raw, ["配套元数据无效，未将其中内容作为测量条件。"]
        return _clean(result), raw, []

    def _entry(self, path: Path) -> dict[str, Any]:
        metadata, _, warnings = self._metadata(path)
        display_name = str(metadata.get("display_name") or path.name)[:240]
        test_type = _kind(str(metadata.get("test_type") or ""))
        if test_type == "unknown":
            test_type = next(
                (kind for kind, marker in _TYPES.items() if marker in display_name), "unknown"
            )
        inferred_model = (
            path.parent.name
            if path.parent != self.root and path.parent.parent != self.root / "uploads"
            else "未知模型"
        )
        stat = path.stat()
        return {
            "id": self._id(path),
            "filename": display_name,
            "model_id": str(metadata.get("model_id") or inferred_model)[:200],
            "provider": str(metadata.get("provider") or "未记录")[:120],
            "test_type": test_type,
            "modified_at": stat.st_mtime,
            "bytes": stat.st_size,
            "revision": self._revision(path),
            "warnings": warnings,
        }

    def catalog(self) -> dict[str, Any]:
        with self._lock:
            files, truncated, skipped = self._scan()
            items = []
            for file in files:
                try:
                    items.append(self._entry(self._safe(file)))
                except (OSError, HistoryError):
                    skipped += 1
            items.sort(key=lambda item: (item["modified_at"], item["id"]), reverse=True)
            return {
                "items": items,
                "truncated": truncated,
                "skipped": skipped,
                "scan_limit": MAX_ENTRIES,
            }

    def _snapshot(
        self, identifier: str, revision: str | None = None
    ) -> tuple[dict, dict, bytes, bytes, list[str]]:
        path = self._resolve(identifier)
        before = self._revision(path)
        entry = self._entry(path)
        if revision and revision != entry["revision"]:
            raise HistoryConflict("所选文件或元数据已变化，请刷新后重新加载或确认")
        raw = _bounded_read(path, MAX_CSV_BYTES)
        metadata, meta_raw, warnings = self._metadata(path)
        if before != entry["revision"] or entry["revision"] != self._revision(path):
            raise HistoryConflict("读取期间文件已变化，请重新加载")
        return entry, metadata, raw, meta_raw, warnings

    def load(
        self, identifier: str, *, revision: str | None = None, offset: int = 0, limit: int = 50
    ) -> dict[str, Any]:
        with self._lock:
            entry, metadata, raw, meta_raw, warnings = self._snapshot(identifier, revision)
            columns, raw_rows, rows, invalid, version = _parse(raw)
        fields = GROUP_FIELDS.get(
            entry["test_type"], ("concurrency_level", "context_length_target")
        )
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for row in rows:
            grouped[tuple(row[name] for name in fields)].append(row)
        if len(grouped) > MAX_GROUPS:
            raise HistoryError("条件切片超过 128 组，未生成截断报告")
        slices = [
            {
                "label": " / ".join(
                    f"{value:,} {'并发' if field == 'concurrency_level' else 'tokens'}"
                    if value is not None
                    else "未记录"
                    for field, value in zip(fields, key, strict=True)
                ),
                **_describe(group),
            }
            for key, group in grouped.items()
        ]
        warnings += invalid
        token_sources = sorted({row["token_source"] for row in rows if row["token_source"]})
        token_methods = sorted(
            {row["token_calc_method"] for row in rows if row["token_calc_method"]}
        )
        if len(token_sources) > 1 or len(token_methods) > 1:
            warnings.append("历史文件混用了不同的 token 来源或算法；比较前需核对口径。")
        if any(row["outcome"] == "unknown" for row in rows):
            warnings.append("部分请求缺少成功状态和错误列；不计入成功率及成功请求的性能统计。")
        warnings.append(
            "本次只重算已有观测，不调用受测 API；计划请求、预热和执行条件未经控制队列核验。"
        )
        origin = {
            "kind": "saved_csv",
            "filename": entry["filename"],
            "csv_sha256": hashlib.sha256(raw).hexdigest(),
            "metadata_sha256": hashlib.sha256(meta_raw).hexdigest() if meta_raw else None,
            "metadata": {
                key: metadata[key]
                for key in ("duration", "test_config", "system_info")
                if key in metadata
            },
        }
        job = {
            "job_id": identifier,
            "model_id": entry["model_id"],
            "test_type": entry["test_type"],
            "status": "历史文件",
        }
        summary = {
            "metric_contract_version": version,
            "integrity": {
                "verified": False,
                "recorded_requests": len(rows),
                "expected_requests": None,
                "reasons": ["来源为历史 CSV，未核验原始执行状态、计划请求数及测量口径。"],
            },
            "run": {
                "test_id": identifier,
                "test_type": entry["test_type"],
                "model_id": entry["model_id"],
                "provider": entry["provider"],
            },
            "overall": _describe(rows),
            "groups": slices,
            "group_axis": " × ".join(GROUP_AXIS[name] for name in fields),
            "measurement_protocol": None,
            "execution_control": {},
            "origin": origin,
            "provenance": {
                "token_sources": token_sources,
                "token_methods": token_methods,
            },
            "data_quality": {
                "warnings": warnings
                + [
                    f"{group['label']} 样本数为 {group['requests']}，尾部分位数分辨率有限。"
                    for group in slices
                    if group["requests"] < 20
                ]
            },
            "notes": [
                "延迟以秒、TPS 以 token/s 展示；不自动换算未声明的其他单位。",
                "性能统计仅使用已知成功、有限且大于零的值，缺失或无效值不以零填充。",
                "分位数采用线性插值，成功率区间采用双侧 95% Wilson score。",
                "历史数据仅作描述性分析；不据此推断差异显著性或原始测量完整性。",
            ],
        }
        preview = [
            {key: "[已隐去]" if _SECRET.search(key) else value[:2000] for key, value in row.items()}
            for row in raw_rows[offset : offset + limit]
        ]
        return {
            "entry": entry,
            "job": job,
            "summary": summary,
            "preview": {
                "columns": columns,
                "items": preview,
                "total": len(rows),
                "offset": offset,
                "limit": limit,
            },
        }

    def export_csv(self, identifier: str, revision: str | None = None) -> str:
        with self._lock:
            _, _, raw, _, _ = self._snapshot(identifier, revision)
            columns, raw_rows, _, _, _ = _parse(raw)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            "'" + name if name.lstrip(" ").startswith(("=", "+", "-", "@")) else name
            for name in columns
        )
        for row in raw_rows:
            values = []
            for key in columns:
                value = "[已隐去]" if _SECRET.search(key) else row[key]
                if value.lstrip(" ").startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
                    value = "'" + value
                values.append(value)
            writer.writerow(values)
        return buffer.getvalue()

    def upload(self, raw: bytes, filename: str, meta_raw: bytes = b"") -> dict[str, Any]:
        if (
            not filename.lower().endswith(".csv")
            or len(raw) > MAX_CSV_BYTES
            or len(meta_raw) > MAX_METADATA_BYTES
        ):
            raise HistoryError("仅支持最多 10 MiB 的 CSV 与最多 64 KiB 的 JSON 元数据")
        _, _, rows, _, _ = _parse(raw)
        try:
            metadata = json.loads(meta_raw) if meta_raw else {}
            if not isinstance(metadata, dict):
                raise ValueError
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise HistoryError("配套元数据必须是 UTF-8 JSON 对象") from exc
        metadata = _clean(metadata)
        metadata["display_name"] = filename.replace("\\", "/").rsplit("/", 1)[-1][:240]
        kind = _kind(str(metadata.get("test_type") or ""))
        if kind == "unknown":
            kind = next((name for name, marker in _TYPES.items() if marker in filename), "unknown")
        fields = GROUP_FIELDS.get(kind, ("concurrency_level", "context_length_target"))
        if len({tuple(row[field] for field in fields) for row in rows}) > MAX_GROUPS:
            raise HistoryError("条件切片超过 128 组，未保存或生成截断报告")
        meta_bytes = json.dumps(
            metadata, ensure_ascii=False, allow_nan=False, sort_keys=True
        ).encode()
        if len(meta_bytes) > MAX_METADATA_BYTES:
            raise HistoryError("整理后的元数据超过 64 KiB，未保存文件")
        with self._lock:
            uploads = self._safe(self.root / "uploads")
            directory = self._safe(uploads / hashlib.sha256(raw + b"\0" + meta_bytes).hexdigest())
            path = directory / "result.csv"
            if directory.exists():
                if (
                    _bounded_read(self._safe(path), MAX_CSV_BYTES) != raw
                    or _bounded_read(self._metadata_path(path), MAX_METADATA_BYTES) != meta_bytes
                ):
                    raise HistoryConflict("同一导入记录已被修改，请检查现有文件后再导入")
                return self._entry(path)
            directory.mkdir(parents=True, mode=0o700)
            try:
                path.write_bytes(raw)
                self._metadata_path(path).write_bytes(meta_bytes)
                return self._entry(path)
            except Exception:
                shutil.rmtree(directory)
                raise

    def delete(self, identifier: str, revision: str) -> dict[str, Any]:
        with self._lock:
            path = self._resolve(identifier)
            if revision != self._revision(path):
                raise HistoryConflict("所选文件或元数据已变化，请重新确认删除")
            metadata = self._metadata_path(path)
            trash_root = self._safe(self.root / ".trash")
            trash = self._safe(trash_root / uuid.uuid4().hex)
            trash.mkdir(parents=True, mode=0o700)
            destination = trash / "result.csv"
            path.rename(destination)
            try:
                if metadata.exists():
                    metadata.rename(trash / "result.csv.meta.json")
            except OSError:
                destination.rename(path)
                trash.rmdir()
                raise
            return {"deleted": identifier, "backup_retained": True}
