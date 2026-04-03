"""
Request Logs记录器模块

记录每次 API 调用完整请求and响应信息到 JSON 文件。

功能特性:
- 按日期分目录存储 (如 api_logs/2026-02-12/)
- 按Test Type标记 (concurrency/prefill/long_contextetc.)
- 按succeeded/失败Status标记 (success/error)
- CreateIndex文件支持快速Query
- Log总量大小限制
"""

import json
import os
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# defaultLog总大小限制 (MB)
DEFAULT_MAX_TOTAL_SIZE_MB = 500


class TestType(Enum):
    """Test Type枚举"""

    CONCURRENCY = "concurrency"  # ConcurrencyTest
    PREFILL = "prefill"  # Prefill Test
    SEGMENTED_PREFILL = "segmented_prefill"  # 分段 Prefill Test
    LONG_CONTEXT = "long_context"  # Long Context Test
    THROUGHPUT_MATRIX = "throughput_matrix"  # Throughput矩阵Test
    STABILITY = "stability"  # Stability Test
    CUSTOM = "custom"  # CustomTest
    QUALITY = "quality"  # Quality Assessment
    BATCH = "batch"  # Batch Test
    UNKNOWN = "unknown"  # 未知类型


class LogStatus(Enum):
    """LogStatus枚举"""

    SUCCESS = "success"
    ERROR = "error"


@dataclass
class RequestLogEntry:
    """Request Logs条目Data结构"""

    log_id: str  # 唯一ID: {timestamp}_{session_id}
    created_at: str  # ISO 时间戳
    session_id: str  # will话ID
    test_type: str  # Test类型
    status: str  # Status: success/error
    config: Dict[str, Any]  # ModelConfigure
    request: Dict[str, Any]  # 请求详情
    response: Dict[str, Any]  # 响应详情
    metrics: Dict[str, Any]  # Performance Metrics
    error: Optional[str] = None  # Error信息


@dataclass
class LogIndexEntry:
    """LogIndex条目"""

    log_id: str
    filepath: str  # 相对路径
    created_at: str
    date: str  # 日期 YYYY-MM-DD
    test_type: str
    status: str
    session_id: str
    provider: str
    model: str
    ttft: Optional[float]
    total_time: Optional[float]
    error: Optional[str]


@dataclass
class LogStats:
    """LogStatistics信息"""

    total_files: int  # 文件总数
    total_size_mb: float  # 总大小 (MB)
    oldest_file: Optional[str]  # 最旧Filename
    newest_file: Optional[str]  # 最新Filename
    log_dir: str  # Log目录路径
    by_status: Dict[str, int] = field(default_factory=dict)  # 按StatusStatistics
    by_test_type: Dict[str, int] = field(default_factory=dict)  # 按Test TypeStatistics
    by_date: Dict[str, int] = field(default_factory=dict)  # 按日期Statistics


class LogIndex:
    """LogIndex管理器"""

    INDEX_FILENAME = "_index.json"

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.index_file = os.path.join(log_dir, self.INDEX_FILENAME)
        self._lock = threading.Lock()
        self._entries: Dict[str, LogIndexEntry] = {}
        self._load()

    def _load(self):
        """从文件LoadIndex"""
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("entries", []):
                        entry = LogIndexEntry(**item)
                        self._entries[entry.log_id] = entry
                logger.debug(f"Loaded {len(self._entries)} index entries")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self._entries = {}

    def _save(self):
        """SaveIndex到文件"""
        try:
            data = {
                "version": 1,
                "updated_at": datetime.now().isoformat(),
                "entries": [asdict(entry) for entry in self._entries.values()],
            }
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save index: {e}")

    def add(self, entry: LogIndexEntry):
        """AddIndex条目"""
        with self._lock:
            self._entries[entry.log_id] = entry
            self._save()

    def remove(self, log_id: str):
        """DeleteIndex条目"""
        with self._lock:
            if log_id in self._entries:
                del self._entries[log_id]
                self._save()

    def remove_by_filepath(self, filepath: str):
        """based onFile pathDeleteIndex条目"""
        with self._lock:
            to_remove = [
                log_id
                for log_id, entry in self._entries.items()
                if entry.filepath == filepath
            ]
            for log_id in to_remove:
                del self._entries[log_id]
            if to_remove:
                self._save()

    def query(
        self,
        date: Optional[str] = None,
        test_type: Optional[str] = None,
        status: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[LogIndexEntry]:
        """
        QueryIndex

        Args:
            date: 日期Filter (YYYY-MM-DD)
            test_type: Test TypeFilter
            status: StatusFilter (success/error)
            provider: Provider Filter
            model: ModelFilter
            session_id: will话IDFilter
            limit: Return数量限制

        Returns:
            匹配Index条目列表
        """
        results = []
        for entry in self._entries.values():
            if date and entry.date != date:
                continue
            if test_type and entry.test_type != test_type:
                continue
            if status and entry.status != status:
                continue
            if provider and entry.provider != provider:
                continue
            if model and entry.model != model:
                continue
            if session_id and entry.session_id != session_id:
                continue
            results.append(entry)
            if len(results) >= limit:
                break

        # 按Created at倒序排列
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """GetIndexStatistics"""
        stats = {
            "total": len(self._entries),
            "by_status": {},
            "by_test_type": {},
            "by_date": {},
        }

        for entry in self._entries.values():
            stats["by_status"][entry.status] = stats["by_status"].get(entry.status, 0) + 1
            stats["by_test_type"][entry.test_type] = stats["by_test_type"].get(entry.test_type, 0) + 1
            stats["by_date"][entry.date] = stats["by_date"].get(entry.date, 0) + 1

        return stats

    def rebuild(self, log_dir: str) -> int:
        """
        重建Index

        Args:
            log_dir: Log根目录

        Returns:
            Index条目数量
        """
        with self._lock:
            self._entries = {}
            count = 0

            # 遍历所has日期子目录
            for date_dir in sorted(os.listdir(log_dir)):
                date_path = os.path.join(log_dir, date_dir)
                if not os.path.isdir(date_path):
                    continue

                # Checkis否is日期格式目录 (YYYY-MM-DD)
                if len(date_dir) != 10 or date_dir[4] != "-" or date_dir[7] != "-":
                    continue

                # 遍历目录in JSON 文件
                for filename in os.listdir(date_path):
                    if not filename.endswith(".json"):
                        continue

                    filepath = os.path.join(date_path, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        entry = LogIndexEntry(
                            log_id=data.get("log_id", ""),
                            filepath=os.path.join(date_dir, filename),
                            created_at=data.get("created_at", ""),
                            date=date_dir,
                            test_type=data.get("test_type", "unknown"),
                            status=data.get("status", "unknown"),
                            session_id=data.get("session_id", ""),
                            provider=data.get("config", {}).get("provider", ""),
                            model=data.get("config", {}).get("model", ""),
                            ttft=data.get("metrics", {}).get("ttft"),
                            total_time=data.get("metrics", {}).get("total_time"),
                            error=data.get("error"),
                        )
                        self._entries[entry.log_id] = entry
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to index {filepath}: {e}")

            self._save()
            return count


class RequestLogger:
    """Request Logs记录器"""

    def __init__(
        self,
        log_dir: str = "api_logs",
        enabled: bool = True,
        mask_api_key: bool = True,
        max_stream_samples: int = 5,
        max_total_size_mb: float = DEFAULT_MAX_TOTAL_SIZE_MB,
    ):
        """
        InitializeRequest Logs记录器

        Args:
            log_dir: Log文件存储根目录
            enabled: is否启用Log记录
            mask_api_key: is否隐藏 API 密钥
            max_stream_samples: Save最大流式响应Sample count
            max_total_size_mb: Log总大小on限 (MB)，超过时自动Delete最旧Log。
                              Setis 0 表示not限制。
        """
        self.log_dir = log_dir
        self.enabled = enabled
        self.mask_api_key = mask_api_key
        self.max_stream_samples = max_stream_samples
        self.max_total_size_mb = max_total_size_mb
        self._lock = threading.Lock()
        self._index: Optional[LogIndex] = None

        if self.enabled:
            if not os.path.exists(self.log_dir):
                try:
                    os.makedirs(self.log_dir, exist_ok=True)
                    logger.info(f"Created log directory: {self.log_dir}")
                except Exception as e:
                    logger.warning(f"Failed to create log directory: {e}")
                    self.enabled = False
                    return

            # InitializeIndex
            self._index = LogIndex(self.log_dir)

    def _get_date_dir(self, created_at: float) -> str:
        """Get日期目录路径"""
        date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, date_str)

    def _generate_filename(
        self,
        provider: str,
        model_id: str,
        session_id: str,
        created_at: float,
        test_type: str,
        status: str,
    ) -> str:
        """
        GenerateLogFilename

        格式: {HHMMSS}_{毫seconds}_{status}_{test_type}_{provider}_{model}_{session_id}.json
        """
        dt = datetime.fromtimestamp(created_at)
        time_str = dt.strftime("%H%M%S")
        milliseconds = f"{int((created_at % 1) * 1000):03d}"

        # Cleanup model_id in特殊字符
        safe_model_id = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in model_id
        )

        # Cleanup provider 名称
        safe_provider = "".join(
            c if c.isalnum() or c in "-_" else "_" for c in provider
        )

        return f"{time_str}_{milliseconds}_{status}_{test_type}_{safe_provider}_{safe_model_id}_{session_id}.json"

    def _mask_headers(self, headers: Dict[str, Any]) -> Dict[str, Any]:
        """隐藏敏感 API 密钥"""
        if not self.mask_api_key:
            return headers

        masked = {}
        for key, value in headers.items():
            key_lower = key.lower()
            if key_lower in ["authorization", "api-key", "x-api-key"]:
                masked[key] = "*****"
            else:
                masked[key] = value
        return masked

    def _get_all_log_files(self) -> List[Tuple[str, float, int]]:
        """
        Get所hasLog文件信息，按修改时间Sort（最旧in前）

        Returns:
            List of (filepath, mtime, size) tuples
        """
        if not os.path.exists(self.log_dir):
            return []

        files = []

        # 遍历所has日期子目录
        for date_dir in os.listdir(self.log_dir):
            date_path = os.path.join(self.log_dir, date_dir)
            if not os.path.isdir(date_path):
                continue

            for filename in os.listdir(date_path):
                if filename.endswith(".json") and not filename.startswith("_"):
                    filepath = os.path.join(date_path, filename)
                    try:
                        stat = os.stat(filepath)
                        files.append((filepath, stat.st_mtime, stat.st_size))
                    except OSError:
                        continue

        # 按修改时间Sort，最旧in前
        files.sort(key=lambda x: x[1])
        return files

    def _get_total_size_bytes(self) -> int:
        """GetLog目录总大小（字节）"""
        files = self._get_all_log_files()
        return sum(size for _, _, size in files)

    def _cleanup_old_logs(self, needed_space_bytes: int = 0) -> int:
        """
        Cleanup旧Log文件，isneed新Log腾出空间

        Args:
            needed_space_bytes: need腾出空间（字节）

        Returns:
            Delete文件数量
        """
        if self.max_total_size_mb <= 0:
            return 0

        max_bytes = self.max_total_size_mb * 1024 * 1024
        current_size = self._get_total_size_bytes()
        target_size = max_bytes - needed_space_bytes

        if current_size <= target_size:
            return 0

        files = self._get_all_log_files()
        deleted_count = 0
        freed_bytes = 0

        for filepath, _, size in files:
            if current_size - freed_bytes <= target_size:
                break

            try:
                os.remove(filepath)
                freed_bytes += size
                deleted_count += 1
                logger.debug(f"Deleted old log file: {filepath}")

                # 从IndexinDelete
                if self._index:
                    rel_path = os.path.relpath(filepath, self.log_dir)
                    self._index.remove_by_filepath(rel_path)
            except OSError as e:
                logger.warning(f"Failed to delete log file {filepath}: {e}")

        if deleted_count > 0:
            freed_mb = freed_bytes / (1024 * 1024)
            logger.info(
                f"Cleaned up {deleted_count} old log files, freed {freed_mb:.2f} MB"
            )

        return deleted_count

    def get_stats(self) -> LogStats:
        """
        GetLogStatistics信息

        Returns:
            LogStats 实例
        """
        files = self._get_all_log_files()

        if not files:
            return LogStats(
                total_files=0,
                total_size_mb=0.0,
                oldest_file=None,
                newest_file=None,
                log_dir=self.log_dir,
            )

        total_size = sum(size for _, _, size in files)
        oldest = files[0][0] if files else None
        newest = files[-1][0] if files else None

        stats = LogStats(
            total_files=len(files),
            total_size_mb=total_size / (1024 * 1024),
            oldest_file=oldest,
            newest_file=newest,
            log_dir=self.log_dir,
        )

        # 从IndexGet详细Statistics
        if self._index:
            index_stats = self._index.get_stats()
            stats.by_status = index_stats.get("by_status", {})
            stats.by_test_type = index_stats.get("by_test_type", {})
            stats.by_date = index_stats.get("by_date", {})

        return stats

    def query(
        self,
        date: Optional[str] = None,
        test_type: Optional[str] = None,
        status: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        QueryLog

        Args:
            date: 日期Filter (YYYY-MM-DD)
            test_type: Test TypeFilter
            status: StatusFilter (success/error)
            provider: Provider Filter
            model: ModelFilter
            session_id: will话IDFilter
            limit: Return数量限制

        Returns:
            匹配Log条目列表
        """
        if not self._index:
            return []

        entries = self._index.query(
            date=date,
            test_type=test_type,
            status=status,
            provider=provider,
            model=model,
            session_id=session_id,
            limit=limit,
        )

        return [asdict(entry) for entry in entries]

    def rebuild_index(self) -> int:
        """
        重建LogIndex

        Returns:
            Index条目数量
        """
        if not self._index:
            return 0
        return self._index.rebuild(self.log_dir)

    def cleanup(self) -> int:
        """
        手动TriggerLogCleanup

        Returns:
            Delete文件数量
        """
        with self._lock:
            return self._cleanup_old_logs()

    def log_request(
        self,
        session_id: str,
        provider: str,
        model_id: str,
        platform: str,
        api_base_url: str,
        headers: Dict[str, Any],
        payload: Dict[str, Any],
        test_type: str = "unknown",
        thinking_enabled: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        full_response_content: str = "",
        reasoning_content: str = "",
        usage_info: Optional[Dict[str, Any]] = None,
        raw_stream_chunks: Optional[List[Dict[str, Any]]] = None,
        created_at: Optional[float] = None,
        start_time: Optional[float] = None,
        first_token_time: Optional[float] = None,
        end_time: Optional[float] = None,
        token_timestamps: Optional[List[float]] = None,
        error: Optional[str] = None,
    ) -> Optional[str]:
        """
        记录Request Logs

        Args:
            session_id: will话ID
            provider: Provider 名称
            model_id: ModelID
            platform: 平台名称
            api_base_url: API Base URL
            headers: 请求头
            payload: 请求负载
            test_type: Test Type (concurrency/prefill/long_contextetc.)
            thinking_enabled: is否启用Thinking mode
            thinking_budget: Thinking budget
            reasoning_effort: 推理强度
            full_response_content: 完整响应内容
            reasoning_content: 推理内容
            usage_info: Token use信息
            raw_stream_chunks: 原始流式响应块
            created_at: Created at戳
            start_time: Start time
            first_token_time: 首 Token 时间
            end_time: End time
            token_timestamps: Token 时间戳列表
            error: Error message

        Returns:
            LogFile path，if禁用or失败则Return None
        """
        if not self.enabled:
            return None

        try:
            # use当前时间if未提供
            if created_at is None:
                created_at = datetime.now().timestamp()

            # 确定Status
            status = LogStatus.ERROR.value if error else LogStatus.SUCCESS.value

            # Generate唯一 ID
            log_id = f"{int(created_at * 1000)}_{session_id}"

            # BuildConfigure信息
            config = {
                "provider": provider,
                "model": model_id,
                "platform": platform,
                "api_base_url": api_base_url,
                "thinking_enabled": thinking_enabled,
                "thinking_budget": thinking_budget,
                "reasoning_effort": reasoning_effort,
            }

            # Build请求信息
            request_info = {
                "url": f"{api_base_url}/chat/completions",
                "headers": self._mask_headers(headers),
                "payload": payload,
            }

            # CalculatePerformance Metrics
            metrics = {}
            if start_time is not None and end_time is not None:
                metrics["total_time"] = round(end_time - start_time, 3)
            if start_time is not None and first_token_time is not None:
                metrics["ttft"] = round(first_token_time - start_time, 3)
            if token_timestamps:
                metrics["token_count"] = len(token_timestamps)
                if metrics.get("total_time") and metrics["token_count"] > 0:
                    # Calculate TPOT (Time Per Output Token)
                    if first_token_time is not None:
                        generation_time = end_time - first_token_time
                        if generation_time > 0:
                            metrics["tpot_avg"] = round(
                                generation_time / metrics["token_count"], 3
                            )

            # Build响应信息
            response_info = {
                "full_content": full_response_content,
                "reasoning_content": reasoning_content,
                "usage": usage_info or {},
            }

            # Add流式响应样本（只Save前几 chunk）
            if raw_stream_chunks:
                response_info["stream_sample"] = raw_stream_chunks[
                    : self.max_stream_samples
                ]

            # CreateLog条目
            entry = RequestLogEntry(
                log_id=log_id,
                created_at=datetime.fromtimestamp(created_at).isoformat(),
                session_id=str(session_id),
                test_type=test_type,
                status=status,
                config=config,
                request=request_info,
                response=response_info,
                metrics=metrics,
                error=error,
            )

            # Get日期目录
            date_dir = self._get_date_dir(created_at)

            # GenerateFilename
            filename = self._generate_filename(
                provider, model_id, str(session_id), created_at, test_type, status
            )

            # 完整File path
            filepath = os.path.join(date_dir, filename)

            # Serialize JSON 内容，用于Calculate大小
            json_content = json.dumps(asdict(entry), ensure_ascii=False, indent=2)
            new_file_size = len(json_content.encode("utf-8"))

            # Thread安全写入（包含大小CheckandCleanup）
            with self._lock:
                # Create日期目录
                os.makedirs(date_dir, exist_ok=True)

                # Check总大小限制，如hasneed则Cleanup旧Log
                self._cleanup_old_logs(needed_space_bytes=new_file_size)

                # 写入Log文件
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(json_content)

                # Add到Index
                if self._index:
                    date_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")
                    index_entry = LogIndexEntry(
                        log_id=log_id,
                        filepath=os.path.join(date_str, filename),
                        created_at=entry.created_at,
                        date=date_str,
                        test_type=test_type,
                        status=status,
                        session_id=str(session_id),
                        provider=provider,
                        model=model_id,
                        ttft=metrics.get("ttft"),
                        total_time=metrics.get("total_time"),
                        error=error,
                    )
                    self._index.add(index_entry)

            logger.debug(f"Request log saved: {filepath}")
            return filepath

        except Exception as e:
            logger.warning(f"Failed to log request: {e}")
            return None


# 全局Singleton
_request_logger: Optional[RequestLogger] = None


def init_request_logger(
    log_dir: str = "api_logs", enabled: bool = True, **kwargs
) -> RequestLogger:
    """
    Initialize全局Request Logs记录器

    Args:
        log_dir: Log目录
        enabled: is否启用
        **kwargs: 传递给 RequestLogger other参数

    Returns:
        RequestLogger 实例
    """
    global _request_logger
    _request_logger = RequestLogger(log_dir=log_dir, enabled=enabled, **kwargs)
    return _request_logger


def get_request_logger() -> Optional[RequestLogger]:
    """Get全局Request Logs记录器实例"""
    return _request_logger
