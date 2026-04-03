"""
响应缓存系统 (Response Cache)

功能：
1. 基于 prompt hash 响应缓存，避免重复请求
2. 支持断点续评（SaveandRestore评估Status）
3. 缓存Statisticsand管理
4. 自动过期and容量限制

use方式：
    from core.response_cache import ResponseCache, get_cache

    # Get全局缓存实例
    cache = get_cache()

    # use缓存
    prompt = "What is 2+2?"
    cached = cache.get(prompt, model_id="gpt-4")
    if cached:
        response = cached
    else:
        response = await call_api(prompt)
        cache.set(prompt, response, model_id="gpt-4")

    # 断点续评
    cache.save_checkpoint(evaluator_state, "mmlu_checkpoint")
    state = cache.load_checkpoint("mmlu_checkpoint")
"""

import gzip
import hashlib
import json
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    """缓存条目"""
    prompt_hash: str
    model_id: str
    response: str
    timestamp: float
    ttl_seconds: int = 86400 * 7  # default7天过期
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() > (self.timestamp + self.ttl_seconds)


@dataclass
class CacheStats:
    """缓存Statistics"""
    total_entries: int = 0
    total_hits: int = 0
    total_misses: int = 0
    total_bytes: int = 0
    oldest_entry: float | None = None
    newest_entry: float | None = None

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0.0


class ResponseCache:
    """
    响应缓存系统

    use SQLite 作is后端存储，支持：
    - 基于 prompt+model 响应缓存
    - 自动过期
    - 容量限制
    - 断点续评
    - Statistics信息
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        max_size_mb: int = 500,
        default_ttl_seconds: int = 86400 * 7,  # 7天
        enable_compression: bool = True
    ):
        """
        Initialize缓存

        Args:
            cache_dir: 缓存目录
            max_size_mb: 最大缓存大小 (MB)
            default_ttl_seconds: default过期时间 (seconds)
            enable_compression: is否启用压缩
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds
        self.enable_compression = enable_compression

        # Data库文件
        self.db_path = self.cache_dir / "response_cache.db"
        self.checkpoint_dir = self.cache_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Statistics
        self._stats = CacheStats()
        self._lock = threading.Lock()

        # InitializeDatabase
        self._init_db()
        self._load_stats()

    def _init_db(self):
        """InitializeDatabase表"""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prompt_hash TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    response BLOB NOT NULL,
                    timestamp REAL NOT NULL,
                    ttl_seconds INTEGER NOT NULL,
                    hit_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    size_bytes INTEGER,
                    UNIQUE(prompt_hash, model_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_prompt_model
                ON cache(prompt_hash, model_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON cache(timestamp)
            """)
            conn.commit()

    def _load_stats(self):
        """Load缓存Statistics"""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(hit_count) as hits,
                    SUM(size_bytes) as bytes,
                    MIN(timestamp) as oldest,
                    MAX(timestamp) as newest
                FROM cache
            """)
            row = cursor.fetchone()
            if row:
                self._stats.total_entries = row[0] or 0
                self._stats.total_hits = row[1] or 0
                self._stats.total_bytes = row[2] or 0
                self._stats.oldest_entry = row[3]
                self._stats.newest_entry = row[4]

    def _compute_hash(self, prompt: str, model_id: str = "") -> str:
        """Calculate prompt 哈希值"""
        content = f"{model_id}:{prompt}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:32]

    def _compress(self, data: str) -> bytes:
        """压缩Data"""
        if self.enable_compression:
            return gzip.compress(data.encode('utf-8'))
        return data.encode('utf-8')

    def _decompress(self, data: bytes) -> str:
        """解压Data"""
        if self.enable_compression:
            try:
                return gzip.decompress(data).decode('utf-8')
            except:
                return data.decode('utf-8')
        return data.decode('utf-8')

    def get(
        self,
        prompt: str,
        model_id: str = "",
        include_expired: bool = False
    ) -> str | None:
        """
        Get缓存响应

        Args:
            prompt: 输入 prompt
            model_id: Model ID
            include_expired: is否包含过期条目

        Returns:
            缓存响应，if未命inReturn None
        """
        prompt_hash = self._compute_hash(prompt, model_id)

        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                    SELECT response, timestamp, ttl_seconds
                    FROM cache
                    WHERE prompt_hash = ? AND model_id = ?
                """, (prompt_hash, model_id))

            row = cursor.fetchone()
            if row:
                response_data, timestamp, ttl = row

                # Checkis否过期
                if not include_expired and time.time() > (timestamp + ttl):
                    self._stats.total_misses += 1
                    return None

                # Update命in计数
                conn.execute("""
                        UPDATE cache
                        SET hit_count = hit_count + 1
                        WHERE prompt_hash = ? AND model_id = ?
                    """, (prompt_hash, model_id))
                conn.commit()

                self._stats.total_hits += 1
                return self._decompress(response_data)

            self._stats.total_misses += 1
            return None

    def set(
        self,
        prompt: str,
        response: str,
        model_id: str = "",
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None
    ) -> bool:
        """
        Set缓存

        Args:
            prompt: 输入 prompt
            response: Model响应
            model_id: Model ID
            ttl_seconds: 过期时间 (seconds)
            metadata: 额外元Data

        Returns:
            is否succeeded
        """
        prompt_hash = self._compute_hash(prompt, model_id)
        response_data = self._compress(response)
        ttl = ttl_seconds or self.default_ttl
        metadata_json = json.dumps(metadata or {})

        with self._lock:
            # Check容量
            self._ensure_capacity(len(response_data))

            with sqlite3.connect(str(self.db_path)) as conn:
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO cache
                        (prompt_hash, model_id, response, timestamp, ttl_seconds, metadata, size_bytes)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        prompt_hash, model_id, response_data,
                        time.time(), ttl, metadata_json, len(response_data)
                    ))
                    conn.commit()

                    self._stats.total_entries += 1
                    self._stats.total_bytes += len(response_data)
                    return True
                except Exception as e:
                    print(f"缓存写入失败: {e}")
                    return False

    def _ensure_capacity(self, new_bytes: int):
        """确保has足够容量"""
        if self._stats.total_bytes + new_bytes <= self.max_size_bytes:
            return

        # needCleanup旧条目
        target_bytes = self.max_size_bytes * 0.8  # Cleanup到 80%

        with sqlite3.connect(str(self.db_path)) as conn:
            # Delete过期条目
            conn.execute("""
                DELETE FROM cache
                WHERE timestamp + ttl_seconds < ?
            """, (time.time(),))

            # if还not够，按 LRU Delete
            while self._stats.total_bytes > target_bytes:
                cursor = conn.execute("""
                    SELECT id, size_bytes FROM cache
                    ORDER BY hit_count ASC, timestamp ASC
                    LIMIT 100
                """)
                rows = cursor.fetchall()
                if not rows:
                    break

                for row_id, size in rows:
                    conn.execute("DELETE FROM cache WHERE id = ?", (row_id,))
                    self._stats.total_bytes -= size
                    self._stats.total_entries -= 1

            conn.commit()

    def delete(self, prompt: str, model_id: str = "") -> bool:
        """Delete指定缓存"""
        prompt_hash = self._compute_hash(prompt, model_id)

        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                    SELECT size_bytes FROM cache
                    WHERE prompt_hash = ? AND model_id = ?
                """, (prompt_hash, model_id))
            row = cursor.fetchone()
            if row:
                conn.execute("""
                        DELETE FROM cache
                        WHERE prompt_hash = ? AND model_id = ?
                    """, (prompt_hash, model_id))
                conn.commit()
                self._stats.total_entries -= 1
                self._stats.total_bytes -= row[0]
                return True
            return False

    def clear(self, model_id: str | None = None):
        """
        清空缓存

        Args:
            model_id: if指定，只清除该Model缓存
        """
        with self._lock:
            with sqlite3.connect(str(self.db_path)) as conn:
                if model_id:
                    conn.execute("DELETE FROM cache WHERE model_id = ?", (model_id,))
                else:
                    conn.execute("DELETE FROM cache")
                conn.commit()

            self._load_stats()

    def cleanup_expired(self) -> int:
        """Cleanup过期条目，ReturnDelete数量"""
        with self._lock, sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.execute("""
                    SELECT COUNT(*) FROM cache
                    WHERE timestamp + ttl_seconds < ?
                """, (time.time(),))
            count = cursor.fetchone()[0]

            if count > 0:
                conn.execute("""
                        DELETE FROM cache
                        WHERE timestamp + ttl_seconds < ?
                    """, (time.time(),))
                conn.commit()
                self._load_stats()

            return count

    def get_stats(self) -> CacheStats:
        """Get缓存Statistics"""
        self._load_stats()
        return self._stats

    # ========== 断点续评功能 ==========

    def save_checkpoint(
        self,
        state: dict[str, Any],
        checkpoint_name: str,
        metadata: dict[str, Any] | None = None
    ) -> str:
        """
        Save评估Check点

        Args:
            state: 评估Status (包含Completed样本Resultetc.)
            checkpoint_name: Check点名称
            metadata: 额外元Data

        Returns:
            Check点File path
        """
        checkpoint_data = {
            "state": state,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "created_at": datetime.now().isoformat()
        }

        filepath = self.checkpoint_dir / f"{checkpoint_name}.pkl.gz"

        with gzip.open(filepath, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        return str(filepath)

    def load_checkpoint(self, checkpoint_name: str) -> dict[str, Any] | None:
        """
        Load评估Check点

        Args:
            checkpoint_name: Check点名称

        Returns:
            Check点Data，ifnot存inReturn None
        """
        filepath = self.checkpoint_dir / f"{checkpoint_name}.pkl.gz"

        if not filepath.exists():
            return None

        try:
            with gzip.open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"LoadCheck点失败: {e}")
            return None

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """列出所hasCheck点"""
        checkpoints = []

        for filepath in self.checkpoint_dir.glob("*.pkl.gz"):
            try:
                with gzip.open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    checkpoints.append({
                        "name": filepath.stem.replace('.pkl', ''),
                        "path": str(filepath),
                        "timestamp": data.get("timestamp"),
                        "created_at": data.get("created_at"),
                        "metadata": data.get("metadata", {})
                    })
            except:
                pass

        return sorted(checkpoints, key=lambda x: x.get("timestamp", 0), reverse=True)

    def delete_checkpoint(self, checkpoint_name: str) -> bool:
        """DeleteCheck点"""
        filepath = self.checkpoint_dir / f"{checkpoint_name}.pkl.gz"
        if filepath.exists():
            filepath.unlink()
            return True
        return False


# ========== 全局缓存实例 ==========

_global_cache: ResponseCache | None = None


def get_cache(
    cache_dir: str = "cache",
    max_size_mb: int = 500,
    **kwargs
) -> ResponseCache:
    """Get全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = ResponseCache(
            cache_dir=cache_dir,
            max_size_mb=max_size_mb,
            **kwargs
        )
    return _global_cache


def reset_cache():
    """Reset全局缓存实例"""
    global _global_cache
    _global_cache = None


# ========== Decorator ==========

def cached_response(model_id: str = "", ttl_seconds: int = None):
    """
    缓存Decorator

    Usage:
        @cached_response(model_id="gpt-4", ttl_seconds=3600)
        async def get_completion(prompt: str) -> str:
            return await api.complete(prompt)
    """
    def decorator(func):
        async def wrapper(prompt: str, *args, **kwargs):
            cache = get_cache()

            # 尝试从缓存Get
            cached = cache.get(prompt, model_id=model_id)
            if cached is not None:
                return cached

            # 调用原函数
            response = await func(prompt, *args, **kwargs)

            # 缓存Result
            cache.set(prompt, response, model_id=model_id, ttl_seconds=ttl_seconds)

            return response

        return wrapper
    return decorator
