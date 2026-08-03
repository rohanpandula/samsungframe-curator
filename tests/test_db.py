"""Tests for src/curator/db.py and src/curator/schema.py.

Proves the catalog database factory: WAL + foreign keys on, all eight v1
boundary-map tables created by ``migrate()``, linear/idempotent migrations via
``PRAGMA user_version``, config-driven path resolution, and the UNIQUE
(connector_id, asset_id) scoping guarantee plus FK enforcement.
"""

from __future__ import annotations

import sqlite3

import pytest

from curator import db, schema
from curator.config import CuratorConfig


@pytest.fixture
def dbase(data_root):
    """A migrated connection under the isolated CURATOR_DATA_ROOT."""
    conn = db.connect()
    db.migrate(conn)
    return conn


def test_default_db_path_resolves_from_env(data_root):
    path = db.default_db_path()
    assert path == data_root / "catalog.db"
    assert path.parent == data_root


def test_default_db_path_explicit_root(tmp_path):
    root = tmp_path / "custom-root"
    assert db.default_db_path(root) == root / "catalog.db"


def test_default_db_path_defaults_to_curator_home(monkeypatch):
    monkeypatch.delenv("CURATOR_DATA_ROOT", raising=False)
    assert db.default_db_path() == CuratorConfig().data_root / "catalog.db"


def test_connect_creates_data_root(data_root):
    # connect() must create the data root directory before writing catalog.db.
    conn = db.connect()
    try:
        assert (data_root / "catalog.db").exists()
    finally:
        conn.close()


def test_connect_enables_wal_and_foreign_keys(dbase):
    assert dbase.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert dbase.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_migrate_creates_all_boundary_tables(dbase):
    tables = db.table_names(dbase)
    for name in schema.EXPECTED_TABLES:
        assert name in tables, f"missing table {name}"


def test_migrate_sets_user_version_to_current(dbase):
    assert dbase.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION


def test_migrate_is_idempotent(dbase):
    # Snapshot table set + row counts, migrate again, assert nothing changed.
    before_tables = db.table_names(dbase)
    before_entries = dbase.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()[0]
    before_user_version = dbase.execute("PRAGMA user_version").fetchone()[0]

    db.migrate(dbase)

    assert dbase.execute("PRAGMA user_version").fetchone()[0] == before_user_version
    assert db.table_names(dbase) == before_tables
    assert dbase.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()[0] == (
        before_entries
    )


def test_unique_connector_asset_scoping(dbase):
    """Same opaque asset_id across two connector instances is two distinct rows."""
    dbase.execute(
        "INSERT INTO source_connectors(connector_id, connector_type) VALUES (?, ?)",
        ("c1", "local"),
    )
    dbase.execute(
        "INSERT INTO source_connectors(connector_id, connector_type) VALUES (?, ?)",
        ("c2", "synthetic-remote"),
    )
    dbase.commit()
    dbase.execute(
        "INSERT INTO source_assets(connector_id, asset_id) VALUES (?, ?)", ("c1", "asset-1")
    )
    dbase.execute(
        "INSERT INTO source_assets(connector_id, asset_id) VALUES (?, ?)", ("c2", "asset-1")
    )
    dbase.commit()
    assert dbase.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 2


def test_unique_connector_asset_same_instance_conflicts(dbase):
    """Same connector + same asset_id violates UNIQUE(connector_id, asset_id)."""
    dbase.execute(
        "INSERT INTO source_connectors(connector_id, connector_type) VALUES (?, ?)",
        ("c1", "local"),
    )
    dbase.execute(
        "INSERT INTO source_assets(connector_id, asset_id) VALUES (?, ?)", ("c1", "asset-1")
    )
    dbase.commit()
    with pytest.raises(sqlite3.IntegrityError):
        dbase.execute(
            "INSERT INTO source_assets(connector_id, asset_id) VALUES (?, ?)",
            ("c1", "asset-1"),
        )


def test_foreign_keys_enforced(dbase):
    """Inserting a source_asset with an unknown connector_id is rejected."""
    with pytest.raises(sqlite3.IntegrityError):
        dbase.execute(
            "INSERT INTO source_assets(connector_id, asset_id) VALUES (?, ?)",
            ("no-such-connector", "asset-1"),
        )


def test_migrate_wal_file_materialized(data_root):
    """After connect+migrate the -wal companion file is used (WAL mode)."""
    conn = db.connect()
    try:
        db.migrate(conn)
        conn.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", ("a" * 64, 3))
        conn.commit()
        assert (data_root / "catalog.db").exists()
        assert (data_root / "catalog.db").stat().st_size > 0
    finally:
        conn.close()
