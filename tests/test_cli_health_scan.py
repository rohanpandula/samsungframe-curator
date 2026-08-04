"""Tests for the S04 headless CLI health surface.

Proves ``curator health --json`` returns a machine-parseable payload whose entry
count reflects the catalog after an ingest (and 0 for a fresh, migrated catalog)
and exits 0. Subcommands are invoked in-process via :func:`curator.cli.main`
under the shared ``data_root`` fixture, following the established capsys pattern.
"""

from __future__ import annotations

import json

from curator import cli, db
from fixture_library import INDEXED_FILES, build_fixture


def test_health_json_returns_entries_count(data_root, tmp_path, capsys):
    """Entry count in health output matches catalog_entries rows after an ingest."""
    folder = build_fixture(tmp_path / "fixture").root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()  # drain ingest report

    rc = cli.main(["health", "--json"])
    assert rc == 0

    doc = json.loads(capsys.readouterr().out)
    assert doc == {"status": "healthy", "catalog_entries": INDEXED_FILES}

    # Cross-check against the actual catalog_entries row count.
    conn = db.connect()
    try:
        n = conn.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()[0]
    finally:
        conn.close()
    assert doc["catalog_entries"] == n == INDEXED_FILES


def test_health_json_exit_0(data_root, capsys):
    """``health --json`` exits 0 even with an empty (freshly migrated) catalog."""
    assert cli.main(["catalog", "init"]) == 0
    capsys.readouterr()  # drain init output

    rc = cli.main(["health", "--json"])
    assert rc == 0

    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "healthy"
    assert doc["catalog_entries"] == 0


def test_health_auto_migrates_and_human_summary(data_root, capsys):
    """``health`` (no flag) migrates on demand and prints a human summary."""
    rc = cli.main(["health"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "healthy" in out
    assert "0 catalog entries" in out


# ---------------------------------------------------------------------------
# T3: `curator scan PATH [--json]` + ScanDiff (exit 0 changes / 3 no-change)
# ---------------------------------------------------------------------------


def _valid_png_bytes() -> bytes:
    """Return a small, valid, decodable PNG payload (fresh every call)."""
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (16, 16), (140, 70, 35)).save(buf, format="PNG")
    return buf.getvalue()


def test_scan_no_change_exit_3(data_root, tmp_path, capsys):
    """Scanning an unchanged folder after a completed ingest reports no diff (exit 3).

    Uses the full 50-file fixture: RAW and corrupt files (never cataloged by ingest)
    must not surface as "new" drift, so an unchanged re-scan is ``no_changes``.
    """
    folder = build_fixture(tmp_path / "fixture").root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()  # drain ingest report

    rc = cli.main(["scan", "--json", str(folder)])
    assert rc == 3

    doc = json.loads(capsys.readouterr().out)
    assert doc["no_changes"] is True
    assert doc["new"] == []
    assert doc["changed"] == []
    assert doc["missing"] == []


def test_scan_reports_new_changed_missing(data_root, tmp_path, capsys):
    """Scan classifies add/edit/delete drift as new / changed / missing and exits 0."""
    fixture = build_fixture(tmp_path / "fixture")
    folder = fixture.root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()  # drain ingest report

    # A brand-new, decodable image -> new.
    new_file = folder / "brand_new.png"
    new_file.write_bytes(_valid_png_bytes())
    # Overwrite a cataloged file with different (still-valid) bytes -> changed.
    changed_target = folder / fixture.singleton_files[0]
    changed_target.write_bytes(_valid_png_bytes())
    # Delete a cataloged file -> missing.
    missing_target = folder / fixture.singleton_files[1]
    missing_target.unlink()

    rc = cli.main(["scan", "--json", str(folder)])
    assert rc == 0

    doc = json.loads(capsys.readouterr().out)
    new_ids = {e["asset_id"] for e in doc["new"]}
    changed_ids = {e["asset_id"] for e in doc["changed"]}
    assert str(new_file.resolve()) in new_ids
    assert str(changed_target.resolve()) in changed_ids
    assert str(missing_target.resolve()) in doc["missing"]


def test_scan_json_parseable(data_root, tmp_path, capsys):
    """``scan --json`` emits a parseable diff with the documented keys."""
    fixture = build_fixture(tmp_path / "fixture")
    folder = fixture.root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()  # drain ingest report
    # Introduce exactly one drift so the diff is non-trivial.
    missing_target = folder / fixture.singleton_files[0]
    missing_target.unlink()

    rc = cli.main(["scan", "--json", str(folder)])
    assert rc == 0

    doc = json.loads(capsys.readouterr().out)
    assert set(doc) == {"connector_id", "new", "changed", "missing", "no_changes"}
    assert isinstance(doc["new"], list)
    assert isinstance(doc["changed"], list)
    assert isinstance(doc["missing"], list)
    assert doc["missing"] == [str(missing_target.resolve())]


def test_scan_rejects_non_directory(data_root, tmp_path, capsys):
    """Scanning a non-directory source is a fatal CuratorError, like ingest."""
    rc = cli.main(["scan", str(tmp_path / "no-such-folder")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not a directory" in err
