"""
基础 Repository 类
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Dict, Any, Tuple

from core.database.connection import Database, db


T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """
    基础 Repository 类

    提供通用 CRUD 操作andQuery方法。
    """

    def __init__(self, database: Database = None):
        self.db = database or db
        self._table_name: str = ""

    @abstractmethod
    def _from_row(self, row: Dict[str, Any]) -> T:
        """
        从Database行ConvertisModel

        Args:
            row: Database行字典

        Returns:
            Model实例
        """
        pass

    def find_by_id(self, id: int) -> Optional[T]:
        """
        based on ID 查找

        Args:
            id: 主键 ID

        Returns:
            Model实例or None
        """
        sql = f"SELECT * FROM {self._table_name} WHERE id = ?"
        row = self.db.fetch_one(sql, (id,))
        return self._from_row(row) if row else None

    def find_all(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "id DESC"
    ) -> List[T]:
        """
        查找所has记录

        Args:
            limit: 限制数量
            offset: 偏移量
            order_by: Sort字段

        Returns:
            Model实例列表
        """
        sql = f"SELECT * FROM {self._table_name} ORDER BY {order_by} LIMIT ? OFFSET ?"
        rows = self.db.fetch_all(sql, (limit, offset))
        return [self._from_row(r) for r in rows]

    def find_by(
        self,
        where: str,
        params: Tuple = (),
        limit: int = 100,
        order_by: str = "id DESC"
    ) -> List[T]:
        """
        条件Query

        Args:
            where: WHERE 子句
            params: 参数元组
            limit: 限制数量
            order_by: Sort字段

        Returns:
            Model实例列表
        """
        sql = f"SELECT * FROM {self._table_name} WHERE {where} ORDER BY {order_by} LIMIT ?"
        rows = self.db.fetch_all(sql, params + (limit,))
        return [self._from_row(r) for r in rows]

    def find_one_by(self, where: str, params: Tuple = ()) -> Optional[T]:
        """
        条件Query单条

        Args:
            where: WHERE 子句
            params: 参数元组

        Returns:
            Model实例or None
        """
        sql = f"SELECT * FROM {self._table_name} WHERE {where} LIMIT 1"
        row = self.db.fetch_one(sql, params)
        return self._from_row(row) if row else None

    def count(self, where: str = "", params: Tuple = ()) -> int:
        """
        计数

        Args:
            where: WHERE 子句（optional）
            params: 参数元组

        Returns:
            记录数
        """
        return self.db.count(self._table_name, where, params)

    def exists(self, where: str, params: Tuple = ()) -> bool:
        """
        Check记录is否存in

        Args:
            where: WHERE 子句
            params: 参数元组

        Returns:
            is否存in
        """
        return self.count(where, params) > 0

    def delete_by_id(self, id: int) -> bool:
        """
        based on ID Delete

        Args:
            id: 主键 ID

        Returns:
            is否succeeded
        """
        return self.db.delete(self._table_name, "id = ?", (id,)) > 0

    def delete_by(self, where: str, params: Tuple = ()) -> int:
        """
        条件Delete

        Args:
            where: WHERE 子句
            params: 参数元组

        Returns:
            Delete记录数
        """
        return self.db.delete(self._table_name, where, params)

    def update_by(self, data: Dict[str, Any], where: str, where_params: Tuple = ()) -> int:
        """
        条件Update

        Args:
            data: UpdateData
            where: WHERE 子句
            where_params: WHERE 参数

        Returns:
            Update记录数
        """
        return self.db.update(self._table_name, data, where, where_params)

    def insert_raw(self, data: Dict[str, Any]) -> int:
        """
        原始Insert

        Args:
            data: Data字典

        Returns:
            新记录 ID
        """
        return self.db.insert(self._table_name, data)

    def paginate(
        self,
        page: int = 1,
        page_size: int = 20,
        where: str = "",
        params: Tuple = (),
        order_by: str = "id DESC"
    ) -> Dict[str, Any]:
        """
        分页Query

        Args:
            page: 页码（从1开始）
            page_size: 每页数量
            where: WHERE 子句
            params: 参数元组
            order_by: Sort字段

        Returns:
            分页Result字典
        """
        offset = (page - 1) * page_size

        # Query总数
        total = self.count(where, params)

        # QueryData
        if where:
            sql = f"SELECT * FROM {self._table_name} WHERE {where} ORDER BY {order_by} LIMIT ? OFFSET ?"
            rows = self.db.fetch_all(sql, params + (page_size, offset))
        else:
            sql = f"SELECT * FROM {self._table_name} ORDER BY {order_by} LIMIT ? OFFSET ?"
            rows = self.db.fetch_all(sql, (page_size, offset))

        items = [self._from_row(r) for r in rows]

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size if page_size > 0 else 0,
        }
