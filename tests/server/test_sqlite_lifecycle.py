"""Verify real connection closure while retaining references to defeat GC cleanup."""

import sqlite3

import pytest

from core.database.backup import DatabaseBackup
from core.database.connection import Database
from core.database.schema import create_tables
from server.store import JobStore
from server.warehouse import WarehouseReader, WarehouseRunNotFound, WarehouseSelection


@pytest.fixture
def connections(monkeypatch):
    original = sqlite3.connect
    retained = []

    class TrackedConnection(sqlite3.Connection):
        closed = False
        fail_setup = False

        def close(self):
            super().close()
            self.closed = True

        def execute(self, sql, *args, **kwargs):
            if self.fail_setup and sql.startswith("PRAGMA"):
                raise sqlite3.OperationalError("synthetic setup failure")
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        conn = original(*args, factory=TrackedConnection, **kwargs)
        retained.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    return retained, TrackedConnection, original


def test_schema_connection_closes_and_pool_connection_remains_usable(tmp_path, connections):
    retained, _, original = connections
    db = object.__new__(Database)
    db._init(str(tmp_path / "data.db"))
    assert retained[0].closed
    db.execute("CREATE TABLE lifecycle_test(value TEXT)")
    db.execute("INSERT INTO lifecycle_test VALUES (?)", ("persisted",))
    assert not retained[-1].closed
    db.close()
    assert all(conn.closed for conn in retained)
    with original(tmp_path / "data.db") as reader:
        assert reader.execute("SELECT value FROM lifecycle_test").fetchone()[0] == "persisted"
    reader.close()


def test_warehouse_reads_and_not_found_close_each_connection(tmp_path, connections):
    retained, _, original = connections
    path = tmp_path / "warehouse.db"
    with original(path) as conn:
        create_tables(conn)
    conn.close()
    reader = WarehouseReader(path)
    assert reader.window(WarehouseSelection()).matched_total == 0
    assert reader.filter_options()["model_id"] == []
    with pytest.raises(WarehouseRunNotFound):
        reader.detail("missing")
    assert len(retained) == 3 and all(conn.closed for conn in retained)


@pytest.mark.parametrize("owner", ["queue", "database"])
def test_setup_failure_closes_new_connection(tmp_path, connections, owner):
    retained, connection_type, _ = connections
    connection_type.fail_setup = True
    with pytest.raises(sqlite3.OperationalError, match="synthetic"):
        if owner == "queue":
            JobStore(tmp_path / "queue.db")
        else:
            db = object.__new__(Database)
            db._init(str(tmp_path / "data.db"))
    assert len(retained) == 1 and retained[0].closed


def test_schema_failure_closes_connection(tmp_path, connections, monkeypatch):
    retained, _, _ = connections

    def fail(conn):
        raise sqlite3.OperationalError("synthetic migration failure")

    monkeypatch.setattr("core.database.connection.create_tables", fail)
    db = object.__new__(Database)
    with pytest.raises(sqlite3.OperationalError):
        db._init(str(tmp_path / "schema.db"))
    assert retained[0].closed


def test_backup_destination_failure_closes_source(tmp_path, connections, monkeypatch):
    retained, _, _ = connections
    connect = sqlite3.connect

    def fail_destination(path, *args, **kwargs):
        if str(path).endswith("destination.db"):
            raise sqlite3.OperationalError("synthetic destination failure")
        return connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", fail_destination)
    backup = DatabaseBackup(str(tmp_path / "source.db"), str(tmp_path / "backups"))
    with pytest.raises(sqlite3.OperationalError):
        backup._backup_sqlite(tmp_path / "destination.db")
    assert len(retained) == 1 and retained[0].closed
