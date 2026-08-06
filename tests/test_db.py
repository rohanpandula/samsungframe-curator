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


def test_migration_v2_adds_dedup_columns(dbase):
    """v2 adds the four dedup/consolidation columns to catalog_entries."""
    cols = {r[1] for r in dbase.execute("PRAGMA table_info(catalog_entries)")}
    for name in ("cluster_id", "dupe_of", "quality_flags", "best_original"):
        assert name in cols, f"missing v2 column {name}"


def test_migration_v2_creates_content_image(dbase):
    """v2 ships the per-content-hash image-signature table with its columns."""
    cols = {r[1] for r in dbase.execute("PRAGMA table_info(content_image)")}
    assert {"sha256", "width", "height", "phash"} <= cols


def test_upgrade_from_v1_to_v2(data_root):
    """A v1-only DB upgrades in place to v2 via the linear migration runner."""
    conn = db.connect()
    try:
        # Force a v1-only database: apply migration 1 and pin user_version to 1.
        conn.executescript(schema.MIGRATIONS[0][1])
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(catalog_entries)")}
        assert "cluster_id" not in cols
        assert "content_image" not in db.table_names(conn)

        db.migrate(conn)

        assert (
            conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(catalog_entries)")}
        assert "cluster_id" in cols
        assert "content_image" in db.table_names(conn)
    finally:
        conn.close()


def test_migration_v3_expands_consolidation_journal(dbase):
    """v3 adds the per-file state-machine columns to consolidation_journal."""
    cols = {r[1] for r in dbase.execute("PRAGMA table_info(consolidation_journal)")}
    for name in (
        "connector_id",
        "asset_id",
        "sha256",
        "error",
        "started_at",
        "finished_at",
    ):
        assert name in cols, f"missing v3 column {name}"
    # v1 run-level columns are preserved for backward compatibility.
    for name in ("id", "status", "note", "created_at"):
        assert name in cols, f"v1 column lost {name}"


def test_migration_v3_creates_resume_indexes(dbase):
    """v3 ships the (connector_id, asset_id) + status resume indexes."""
    indexes = {
        r[0]
        for r in dbase.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type='index' AND tbl_name='consolidation_journal'"
        )
    }
    assert "idx_consolidation_journal_conn_asset" in indexes
    assert "idx_consolidation_journal_status" in indexes


def test_migrate_is_idempotent_v3(dbase):
    """Re-running migrate after v3 re-applies nothing (pure idempotence)."""
    cols_before = {
        r[1] for r in dbase.execute("PRAGMA table_info(consolidation_journal)")
    }
    db.migrate(dbase)
    cols_after = {
        r[1] for r in dbase.execute("PRAGMA table_info(consolidation_journal)")
    }
    assert cols_before == cols_after
    assert dbase.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION


def test_upgrade_from_v2_to_v3(data_root):
    """A v2-only DB upgrades in place to v3 via the linear migration runner."""
    conn = db.connect()
    try:
        # Force a v2-only database: apply migrations 1+2 and pin user_version to 2.
        conn.executescript(schema.MIGRATIONS[0][1])
        conn.executescript(schema.MIGRATIONS[1][1])
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(consolidation_journal)")}
        assert "asset_id" not in cols  # v2 does not have the per-file columns

        db.migrate(conn)

        assert (
            conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(consolidation_journal)")}
        assert "asset_id" in cols
        assert "finished_at" in cols
    finally:
        conn.close()


def test_migration_v16_adds_vote_pairing_columns(dbase):
    """v16 adds vote_group + retracted_at to taste_preferences (M009/S01)."""
    cols = {r[1] for r in dbase.execute("PRAGMA table_info(taste_preferences)")}
    for name in ("vote_group", "retracted_at"):
        assert name in cols, f"missing v16 column {name}"
    # v13 columns are preserved for backward compatibility.
    for name in ("id", "profile_id", "catalog_entry_id", "preference", "note", "created_at"):
        assert name in cols, f"v13 column lost {name}"


def test_migration_v16_creates_vote_group_index(dbase):
    """v16 ships the vote_group lookup index on taste_preferences."""
    indexes = {
        r[0]
        for r in dbase.execute(
            "SELECT name FROM sqlite_master"
            " WHERE type='index' AND tbl_name='taste_preferences'"
        )
    }
    assert "idx_taste_preferences_vote_group" in indexes


def test_migration_v16_new_columns_nullable_no_default(dbase):
    """v16 columns are nullable with no DEFAULT — a pre-M009 4-column INSERT works."""
    dbase.execute(
        "INSERT INTO taste_preferences(profile_id, catalog_entry_id, preference, note)"
        " VALUES (1, 1, 1, 'legacy fixture row')"
    )
    dbase.commit()
    row = dbase.execute(
        "SELECT vote_group, retracted_at FROM taste_preferences WHERE note = ?",
        ("legacy fixture row",),
    ).fetchone()
    assert row == (None, None)


def test_upgrade_from_v15_to_v16(data_root):
    """A v15-only DB upgrades in place to v16 via the linear migration runner."""
    conn = db.connect()
    try:
        # Force a v15-only database: apply migrations 1-15 and pin user_version to 15.
        for _version, ddl in schema.MIGRATIONS[:15]:
            conn.executescript(ddl)
        conn.execute("PRAGMA user_version = 15")
        conn.commit()
        cols = {r[1] for r in conn.execute("PRAGMA table_info(taste_preferences)")}
        assert "vote_group" not in cols  # v15 does not have the vote-pairing columns

        db.migrate(conn)

        assert (
            conn.execute("PRAGMA user_version").fetchone()[0] == schema.SCHEMA_VERSION
        )
        cols = {r[1] for r in conn.execute("PRAGMA table_info(taste_preferences)")}
        assert "vote_group" in cols
        assert "retracted_at" in cols
    finally:
        conn.close()
