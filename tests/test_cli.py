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

import json

from consolidate_fixture import (
    CONSOLIDATED_FILES,
    UNIQUE_LIBRARY_FILES,
    build_consolidation_fixture,
)
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
    assert rc == 2
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
    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()


# ---------------------------------------------------------------------------
# T4: `curator consolidate` (dry-run plan / execute / resume / archive)
# ---------------------------------------------------------------------------


def _legacy_ssd(tmp_path):
    """Build the deterministic consolidation fixture; return its source folder."""
    return build_consolidation_fixture(tmp_path / "src").root


PLAN_LABELS = (
    "exact_dupes",
    "near_dupes",
    "higher_res_originals",
    "filename_collisions",
    "panels",
    "sidecars",
    "corrupt",
    "missing_date",
)


def test_consolidate_dry_run_reports_all_eight_groups(data_root, tmp_path, capsys):
    src = _legacy_ssd(tmp_path)
    rc = cli.main(["consolidate", "--dry-run", str(src)])
    assert rc == 0
    out = capsys.readouterr().out
    # Every group label present in the human-readable report.
    for label in PLAN_LABELS:
        assert label in out, f"missing group label {label}"
    # Pinned fixture arithmetic.
    assert "exact_dupes          : 3" in out
    assert "near_dupes           : 2" in out
    assert "higher_res_originals : 1" in out
    assert "filename_collisions  : 2" in out
    assert "panels               : 1" in out
    assert "sidecars             : 1" in out
    assert "corrupt              : 1" in out
    assert "missing_date         : 1" in out


def test_consolidate_dry_run_is_default_mode(data_root, tmp_path, capsys):
    """No mode flag -> dry-run (the demo's default surface)."""
    src = _legacy_ssd(tmp_path)
    assert cli.main(["consolidate", str(src)]) == 0
    out = capsys.readouterr().out
    assert "(dry-run)" in out
    assert "exact_dupes          : 3" in out


def test_consolidate_dry_run_json_group_counts(data_root, tmp_path, capsys):
    src = _legacy_ssd(tmp_path)
    assert cli.main(["consolidate", "--dry-run", "--json", str(src)]) == 0
    doc = json.loads(capsys.readouterr().out)
    # to_json() serializes full group membership (mirrors IngestReport); derive
    # the pinned counts from the structure rather than re-deriving arithmetic.
    assert sum(len(g) for g in doc["exact_dupes"]) == 3
    assert sum(len(g) for g in doc["near_dupes"]) == 2
    assert len(doc["higher_res_originals"]) == 1
    assert sum(len(g) for g in doc["filename_collisions"]) == 2
    assert doc["panels"] == ["2024-03-01_panel.jpg"]
    assert doc["sidecars"] == ["2024-03-01_panel.xmp"]
    assert len(doc["corrupt"]) == 1 and doc["corrupt"][0]["path"] == "broken.jpg"
    assert doc["missing_date"] == ["nodate.jpg"]


def test_consolidate_execute_promotes_all_sources_untouched(data_root, tmp_path, capsys):
    src = _legacy_ssd(tmp_path)
    rc = cli.main(["consolidate", "--execute", str(src)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "promoted       : 11" in out
    assert f"unique library : {UNIQUE_LIBRARY_FILES}" in out
    # Library materialized and content-addressed (9 distinct blobs).
    lib = data_root / "library"
    assert lib.is_dir()
    n_blobs = sum(1 for p in lib.rglob("*") if p.is_file())
    assert n_blobs == UNIQUE_LIBRARY_FILES
    # Sources completely untouched: every fixture file still present at source.
    for rel in (
        "2024-01-01_exact.jpg",
        "2024-01-02_near_base.jpg",
        "2024-03-01_panel.jpg",
        "2024-03-01_panel.xmp",
        "a/2024-02-01_photo.jpg",
        "b/2024-02-01_photo.jpg",
        "broken.jpg",
        "nodate.jpg",
    ):
        assert (src / rel).exists(), f"source file removed: {rel}"


def test_consolidate_execute_json(data_root, tmp_path, capsys):
    src = _legacy_ssd(tmp_path)
    assert cli.main(["consolidate", "--execute", "--json", str(src)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["staged"] == CONSOLIDATED_FILES
    assert doc["verified"] == CONSOLIDATED_FILES
    assert doc["promoted"] == CONSOLIDATED_FILES
    assert doc["skipped"] == 0
    assert doc["unique_library_files"] == UNIQUE_LIBRARY_FILES
    assert doc["errors"] == []


def test_consolidate_execute_resume_is_idempotent(data_root, tmp_path, capsys):
    src = _legacy_ssd(tmp_path)
    assert cli.main(["consolidate", "--execute", str(src)]) == 0
    capsys.readouterr()  # drain

    # A fresh run with --resume skips every already-promoted file.
    assert cli.main(["consolidate", "--execute", "--resume", str(src)]) == 0
    out = capsys.readouterr().out
    assert "skipped        : 11" in out
    assert "promoted       : 0" in out
    # Library count unchanged (content-addressed convergence holds on resume).
    lib = data_root / "library"
    n_blobs = sum(1 for p in lib.rglob("*") if p.is_file())
    assert n_blobs == UNIQUE_LIBRARY_FILES


def test_consolidate_execute_archive_moves_source(data_root, tmp_path, capsys):
    src = _legacy_ssd(tmp_path)
    assert cli.main(["consolidate", "--execute", "--archive", str(src)]) == 0
    out = capsys.readouterr().out
    assert "promoted       : 11" in out
    assert "archived to" in out
    # The explicitly-approved source folder moved intact beneath <root>/archive/.
    archived = data_root / "archive" / "legacy-ssd"
    assert archived.is_dir()
    assert (archived / "2024-01-01_exact.jpg").exists()
    assert (archived / "2024-03-01_panel.xmp").exists()
    # Source folder no longer at its original location (it was relocated intact).
    assert not src.exists()


def test_consolidate_missing_dir_returns_error(data_root, tmp_path, capsys):
    rc = cli.main(["consolidate", "/no/such/folder"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()
