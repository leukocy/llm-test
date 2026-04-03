"""
Database Backup Module

提供自动BackupandRestore功能。
"""

import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class DatabaseBackup:
    """
    DatabaseBackup管理器

    功能：
    - 自动Backup（按时间/大小Trigger）
    - Backup轮转（保留最近 N ）
    - RestoreBackup
    """

    def __init__(self, db_path: str = "data/benchmark.db", backup_dir: str = "data/backups"):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # defaultConfigure
        self.max_backups = 10  # 保留最近 10 Backup
        self.min_interval_hours = 1  # 最小Backup间隔（hours）
        self.max_size_change_mb = 50  # Data库变化超过 50MB 时自动Backup

    def create_backup(self, reason: str = "manual") -> Optional[Path]:
        """
        CreateBackup

        Args:
            reason: Backup原因

        Returns:
            BackupFile path，失败Return None
        """
        if not self.db_path.exists():
            logger.warning(f"Database文件not存in: {self.db_path}")
            return None

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"benchmark_{timestamp}_{reason}.db"
            backup_path = self.backup_dir / backup_name

            # use SQLite in线Backup API
            self._backup_sqlite(backup_path)

            logger.info(f"Database已Backup: {backup_path}")

            # Cleanup旧Backup
            self._cleanup_old_backups()

            return backup_path

        except Exception as e:
            logger.error(f"Backup失败: {e}")
            return None

    def _backup_sqlite(self, backup_path: Path):
        """use SQLite API 进行Backup"""
        source = sqlite3.connect(str(self.db_path))
        dest = sqlite3.connect(str(backup_path))

        try:
            source.backup(dest)
        finally:
            source.close()
            dest.close()

    def restore_backup(self, backup_path: Path) -> bool:
        """
        RestoreBackup

        Args:
            backup_path: BackupFile path

        Returns:
            is否succeeded
        """
        if not backup_path.exists():
            logger.error(f"Backup文件not存in: {backup_path}")
            return False

        try:
            # 先Backup当前Database
            if self.db_path.exists():
                self.create_backup("pre_restore")

            # RestoreBackup
            shutil.copy2(backup_path, self.db_path)
            logger.info(f"Database已Restore: {backup_path}")
            return True

        except Exception as e:
            logger.error(f"Restore失败: {e}")
            return False

    def list_backups(self) -> List[dict]:
        """
        列出所hasBackup

        Returns:
            Backup信息列表
        """
        backups = []
        for f in self.backup_dir.glob("benchmark_*.db"):
            stat = f.stat()
            backups.append({
                "path": str(f),
                "name": f.name,
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })

        # 按时间倒序排列
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        return backups

    def _cleanup_old_backups(self):
        """Cleanup旧Backup"""
        backups = self.list_backups()

        if len(backups) > self.max_backups:
            # Delete最旧Backup
            for backup in backups[self.max_backups:]:
                try:
                    Path(backup['path']).unlink()
                    logger.info(f"已Delete旧Backup: {backup['name']}")
                except Exception as e:
                    logger.warning(f"DeleteBackup失败: {e}")

    def should_backup(self) -> bool:
        """
        Checkis否应该自动Backup

        Returns:
            is否needBackup
        """
        backups = self.list_backups()

        if not backups:
            return True

        # Check时间间隔
        last_backup_time = datetime.fromisoformat(backups[0]['created_at'])
        hours_since_backup = (datetime.now() - last_backup_time).total_seconds() / 3600

        if hours_since_backup >= self.min_interval_hours:
            return True

        # CheckDatabase大小变化
        if self.db_path.exists():
            current_size = self.db_path.stat().st_size
            last_backup_size = backups[0]['size_mb'] * 1024 * 1024
            size_change_mb = abs(current_size - last_backup_size) / 1024 / 1024

            if size_change_mb >= self.max_size_change_mb:
                return True

        return False

    def auto_backup_if_needed(self) -> Optional[Path]:
        """
        ifneed则自动Backup

        Returns:
            Backup路径，未BackupReturn None
        """
        if self.should_backup():
            return self.create_backup("auto")
        return None

    def get_backup_summary(self) -> dict:
        """GetBackup摘要"""
        backups = self.list_backups()
        total_size = sum(b['size_mb'] for b in backups)

        return {
            "count": len(backups),
            "total_size_mb": round(total_size, 2),
            "latest": backups[0] if backups else None,
            "oldest": backups[-1] if backups else None,
        }
