"""Tests for src/curator/cli.py.

Proves the minimal headless CLI (T06): ``curator catalog init`` creates
CURATOR_DATA_ROOT/catalog.db in WAL mode with all v1 tables and reports the path;
``curator catalog add <file>`` writes a catalog_entries row with the correct SHA-256
and connector-scoped source identity; re-adding the same file is idempotent (one
row); the CLI honors the CURATOR_DATA_ROOT env var; and failures (missing file)
surface as a non-zero exit code with a stderr message. Subcommands are invoked
in-process via :func:`curator.cli.main` under a monkeypatched env.
"""

from __future__ import annotations

from curator import cli, db, schema
from curator.hashing import sha256_hex
from fixture_library import CORRUPT_FILENAME, RAW_FILENAMES, build_fixture


def test_init_creates_migrated_wal_db(data_root, capsys):
    rc = cli.main(["catalog", "init"])
    assert rc == 0

    out = capsys.readouterr().out
    assert str(data_root / "catalog.db") in out
    assert (data_root / "catalog.db").exists()

    conn = db.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        tables = db.table_names(conn)
        for name in schema.EXPECTED_TABLES:
            assert name in tables, f"missing table {name}"
    finally:
        conn.close()


def test_init_is_idempotent(data_root):
    assert cli.main(["catalog", "init"]) == 0
    assert cli.main(["catalog", "init"]) == 0
    # Still just the one database file.
    assert (data_root / "catalog.db").exists()


def test_add_writes_entry_with_sha_and_identity(data_root, tmp_path, capsys):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"curator-cli-art")
    digest = sha256_hex(b"curator-cli-art")

    assert cli.main(["catalog", "init"]) == 0
    rc = cli.main(["catalog", "add", str(f)])
    assert rc == 0

    out = capsys.readouterr().out
    assert digest in out

    conn = db.connect()
    try:
        row = conn.execute(
            "SELECT connector_id, asset_id, sha256 FROM catalog_entries"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "cli-local"
    # Connector-scoped source identity is the normalized absolute path.
    assert row[1] == str(f.resolve())
    assert row[2] == digest


def test_readd_same_file_is_idempotent(data_root, tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"idempotent-cli")

    assert cli.main(["catalog", "init"]) == 0
    assert cli.main(["catalog", "add", str(f)]) == 0
    assert cli.main(["catalog", "add", str(f)]) == 0

    conn = db.connect()
    try:
        n = conn.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()[0]
    finally:
        conn.close()
    assert n == 1


def test_add_honors_custom_data_root(tmp_path, monkeypatch):
    """The CLI resolves CURATOR_DATA_ROOT; a different root gets its own db."""
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"custom-root-cli")
    digest = sha256_hex(b"custom-root-cli")

    monkeypatch.setenv("CURATOR_DATA_ROOT", str(root_a))
    assert cli.main(["catalog", "init"]) == 0
    assert cli.main(["catalog", "add", str(f)]) == 0
    # Database materialized only under root_a, not root_b.
    assert (root_a / "catalog.db").exists()
    assert not (root_b / "catalog.db").exists()

    conn = db.connect()
    try:
        sha = conn.execute("SELECT sha256 FROM catalog_entries").fetchone()[0]
    finally:
        conn.close()
    assert sha == digest


def test_add_missing_file_returns_error(data_root, capsys):
    assert cli.main(["catalog", "init"]) == 0
    rc = cli.main(["catalog", "add", "/no/such/file.jpg"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_catalog_add_requires_file_argument(data_root):
    # Missing the required positional renders a usage error (non-zero).
    try:
        rc = cli.main(["catalog", "add"])
    except SystemExit as exc:  # argparse exits 2 on missing required arg
        rc = exc.code
    assert rc != 0


# ---------------------------------------------------------------------------
# T05: `curator ingest PATH` CLI
# ---------------------------------------------------------------------------


def test_ingest_reports_30_clusters(data_root, tmp_path, capsys):
    folder = build_fixture(tmp_path / "fixture").root
    rc = cli.main(["ingest", str(folder)])
    assert rc == 0

    out = capsys.readouterr().out
    # Demo surface: 50 files -> 30 unique clusters, exact/near counts.
    assert "total enumerated : 50" in out
    assert "indexed          : 47" in out
    assert "unique clusters  : 30  (exact=5, near=8)" in out
    assert "unsupported      : 2" in out
    assert "corrupt          : 1" in out
    # RAW explicit-unsupported surface (R003).
    assert "explicit-unsupported (RAW):" in out
    assert RAW_FILENAMES[0] in out and RAW_FILENAMES[1] in out
    # Corrupt error preserved in the report.
    assert "corrupt:" in out
    assert CORRUPT_FILENAME in out
    assert "failed to decode" in out
    # At least one best-original flag printed with phash distance.
    assert "best=" in out
    assert "phash_dist=0" in out


def test_ingest_persists_catalog_and_journal(data_root, tmp_path):
    folder = build_fixture(tmp_path / "fixture").root
    assert cli.main(["ingest", str(folder)]) == 0

    conn = db.connect()
    try:
        entries = conn.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()[0]
        distinct = conn.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM catalog_entries"
            " WHERE cluster_id IS NOT NULL"
        ).fetchone()[0]
        journal = conn.execute("SELECT COUNT(*) FROM ingest_journal").fetchone()[0]
        unsup = conn.execute(
            "SELECT COUNT(*) FROM ingest_journal WHERE status='unsupported'"
        ).fetchone()[0]
        corrupt = conn.execute(
            "SELECT COUNT(*) FROM ingest_journal WHERE status='corrupt'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert entries == 47
    assert distinct == 30
    assert journal == 50
    assert unsup == 2
    assert corrupt == 1


def test_ingest_resume_is_idempotent(data_root, tmp_path, capsys):
    folder = build_fixture(tmp_path / "fixture").root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()  # drain

    assert cli.main(["ingest", "--resume", str(folder)]) == 0
    out = capsys.readouterr().out
    assert "unique clusters  : 30" in out

    conn = db.connect()
    try:
        entries = conn.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()[0]
    finally:
        conn.close()
    assert entries == 47  # unchanged after resume re-ingest


def test_ingest_missing_dir_returns_error(data_root, capsys):
    rc = cli.main(["ingest", "/no/such/dir"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "error" in err.lower()
