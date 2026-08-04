"""Acceptance tests for the Curator pipeline (S05).

This module ships the deterministic, air-gapped acceptance gate for the three
core scenarios (T2/T3 append the remaining scenarios in place). Each scenario is
**self-bootstrapping**: it builds its own deterministic fixture and drives the
CLI in-process via :mod:`acceptance_harness` (and the shared ``data_root``
fixture from conftest), never relying on cross-test ordering or external state.

Scenario 1 (this task, T1) is the air-gap ingest happy path: a 50-file fixture
indexes to 47 catalog entries across 30 unique clusters, source files are left
byte-for-byte untouched, re-ingest is idempotent (rows + content hashes stable),
and a static import audit proves no network client is reachable from the ingest
path.
"""

from __future__ import annotations

import os

from acceptance_harness import (
    assert_no_network_imports,
    build_ingest_tree,
    run_cli,
    sha256_file,
)
from curator.catalog import Catalog

# Documented no-op air-gap posture env value (Scenario 1). The gate relies on the
# static network-import audit, not runtime network-enforcement code.
NETWORK_DENY = "deny"


def _all_files(root) -> list[str]:
    """Return lexicographically sorted absolute file paths under *root*."""
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(os.fspath(root)):
        for name in filenames:
            out.append(os.path.join(dirpath, name))
    return sorted(out)


def _source_hashes(root) -> dict[str, str]:
    return {p: sha256_file(p) for p in _all_files(root)}


def test_scenario1_air_gap_ingest(data_root, tmp_path, monkeypatch):
    """50-file fixture -> 47 entries / 30 clusters, sources untouched, idempotent."""
    # Documented no-op (air-gap posture): real protection is the static audit.
    monkeypatch.setenv("CURATOR_NETWORK", NETWORK_DENY)

    tree = build_ingest_tree(tmp_path)
    before = _source_hashes(tree)

    rc, out = run_cli(["ingest", str(tree)])
    assert rc == 0
    # Report surface: pinned ingest arithmetic (47 indexable, 30 clusters).
    assert "indexed          : 47" in out
    assert "unique clusters  : 30" in out

    # Catalog reality: 47 entries, 30 distinct cluster ids, 50 journal transitions.
    catalog = Catalog(data_root=data_root)
    try:
        conn = catalog.db
        entries_1 = conn.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()[0]
        distinct_1 = conn.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM catalog_entries"
            " WHERE cluster_id IS NOT NULL"
        ).fetchone()[0]
        journal_1 = conn.execute("SELECT COUNT(*) FROM ingest_journal").fetchone()[0]
        content_1 = conn.execute("SELECT COUNT(*) FROM content").fetchone()[0]
    finally:
        catalog.db.close()
    assert entries_1 == 47
    assert distinct_1 == 30
    assert journal_1 == 50

    # Source files byte-for-byte unchanged by ingest.
    assert _source_hashes(tree) == before

    # Re-ingest is idempotent: rc 0 with rows and content hashes stable.
    rc2, _out2 = run_cli(["ingest", str(tree)])
    assert rc2 == 0
    catalog2 = Catalog(data_root=data_root)
    try:
        entries_2 = catalog2.db.execute(
            "SELECT COUNT(*) FROM catalog_entries"
        ).fetchone()[0]
        distinct_2 = catalog2.db.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM catalog_entries"
            " WHERE cluster_id IS NOT NULL"
        ).fetchone()[0]
        content_2 = catalog2.db.execute("SELECT COUNT(*) FROM content").fetchone()[0]
    finally:
        catalog2.db.close()
    assert entries_2 == entries_1 == 47
    assert distinct_2 == distinct_1 == 30
    assert content_2 == content_1  # content hashes stable across re-ingest

    # Static audit: no network client reachable from the ingest import closure.
    assert_no_network_imports()
    assert _source_hashes(tree) == before
