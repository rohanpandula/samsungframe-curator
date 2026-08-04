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
