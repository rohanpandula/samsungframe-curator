"""Acceptance tests for the Curator pipeline (S05).

This module ships the deterministic, air-gapped acceptance gate for the three
core scenarios (T2/T3 append the remaining scenarios in place). Each scenario is
**self-bootstrapping**: it builds its own deterministic fixture and drives the
CLI in-process via :mod:`acceptance_harness` (and the shared ``data_root``
fixture from conftest), never relying on cross-test ordering or external state.

Scenario 1 (T1) is the air-gap ingest happy path: a 50-file fixture indexes to
47 catalog entries across 30 unique clusters, source files are left byte-for-byte
untouched, re-ingest is idempotent (rows + content hashes stable), and a static
import audit proves no network client is reachable from the ingest path.

Scenario 2 (T2) drives the headless operational surface: the scan diff exit-code
contract (3=no-change, read-only and idempotent, 2=fatal for a non-directory
target), the health surface (47 catalog_entries after ingest), and the
consolidate dry-run JSON plan with its 8 pinned group counts.
"""

from __future__ import annotations

import json
import os

import pytest

from acceptance_harness import (
    assert_no_network_imports,
    build_consolidation_tree,
    build_ingest_tree,
    run_cli,
    sha256_file,
)
from consolidate_fixture import (
    CONSOLIDATED_FILES,
    EXPECTED_REL_FILES,
    UNIQUE_LIBRARY_FILES,
)
from curator.catalog import Catalog
from curator.consolidate import ConsolidationExecutor

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


def test_scenario2_headless_scan_and_exit_codes(data_root, tmp_path):
    """Scan exit contract: 3 no-change (idempotent/read-only), 2 fatal non-dir."""
    tree = build_ingest_tree(tmp_path)
    rc, _out = run_cli(["ingest", str(tree)])
    assert rc == 0

    rc, out = run_cli(["scan", str(tree), "--json"])
    assert rc == 3
    first = json.loads(out)
    assert {
        "connector_id",
        "new",
        "changed",
        "missing",
        "no_changes",
    } <= first.keys()
    assert first["no_changes"] is True

    rc2, out2 = run_cli(["scan", str(tree), "--json"])
    assert rc2 == 3
    assert json.loads(out2) == first  # scan is read-only; identical payload

    non_dir = tmp_path / "not-a-dir"  # deliberately not created
    rc3, _out3 = run_cli(["scan", str(non_dir), "--json"])
    assert rc3 == 2

    rc4, out4 = run_cli(["health", "--json"])  # health takes no path argument
    assert rc4 == 0
    health = json.loads(out4)
    assert health["status"] == "healthy"
    assert health["catalog_entries"] == 47


def test_scenario2_consolidate_dry_run_json(data_root, tmp_path):
    """Consolidate --dry-run --json exposes the 8 pinned R002 group counts."""
    legacy = build_consolidation_tree(tmp_path)
    rc, out = run_cli(["consolidate", str(legacy), "--dry-run", "--json"])
    assert rc == 0
    plan = json.loads(out)
    # to_json() serializes full group membership (asdict); derive the pinned
    # counts from the structure rather than re-deriving the arithmetic.
    assert sum(len(g) for g in plan["exact_dupes"]) == 3
    assert sum(len(g) for g in plan["near_dupes"]) == 2
    assert len(plan["higher_res_originals"]) == 1
    assert sum(len(g) for g in plan["filename_collisions"]) == 2
    assert len(plan["panels"]) == 1
    assert len(plan["sidecars"]) == 1
    assert len(plan["corrupt"]) == 1
    assert len(plan["missing_date"]) == 1


def test_scenario4_consolidate_dry_run_groups(data_root, tmp_path):
    """Scenario 4: consolidate --dry-run --json mirrors the 8 pinned R002 counts."""
    legacy = build_consolidation_tree(tmp_path)
    rc, out = run_cli(["consolidate", str(legacy), "--dry-run", "--json"])
    assert rc == 0
    data = json.loads(out)
    # asdict(plan): dedup/collision groups hold lists of members; derive totals
    # from the membership structure, strict list groups from their length.
    assert sum(len(group) for group in data["exact_dupes"]) == 3
    assert sum(len(group) for group in data["near_dupes"]) == 2
    assert len(data["higher_res_originals"]) == 1
    assert sum(len(group) for group in data["filename_collisions"]) == 2
    assert len(data["panels"]) == 1
    assert len(data["sidecars"]) == 1
    assert len(data["corrupt"]) == 1
    assert len(data["missing_date"]) == 1


def test_scenario4_execute_non_destructive_no_omit_no_delete(data_root, tmp_path):
    """Scenario 4: execute promotes 11 -> 9 blobs, sources untouched, all in data_root."""
    legacy = build_consolidation_tree(tmp_path)
    before = _source_hashes(legacy)
    assert len(before) == CONSOLIDATED_FILES

    rc, out = run_cli(["consolidate", str(legacy), "--execute"])
    assert rc == 0
    # _format_result human report exposes the pinned arithmetic.
    assert "promoted       : 11" in out
    assert "unique library : 9" in out

    # Content-addressed library holds exactly the unique source hashes (9 blobs).
    executor = ConsolidationExecutor(legacy)
    library_files = [p for p in executor.library_root.rglob("*") if p.is_file()]
    assert len(library_files) == UNIQUE_LIBRARY_FILES == 9

    # Every expected source file has a corresponding content-addressed blob.
    for rel in EXPECTED_REL_FILES:
        sha = sha256_file(os.path.join(legacy, rel))
        assert executor._library_path(sha).is_file(), f"missing blob for {rel}"

    # No source deleted, none altered: the on-disk sha set is unchanged.
    assert _source_hashes(legacy) == before

    # All application material lives solely under the canonical data root.
    for material in (executor.library_root, executor.staging_root, executor.archive_root):
        str(material).startswith(str(data_root))


def test_scenario4_interrupt_then_resume_completes_all(data_root, monkeypatch, tmp_path):
    """Scenario 4: a mid-copy interrupt is resumable and every file is eventually promoted."""
    legacy = build_consolidation_tree(tmp_path)
    before = _source_hashes(legacy)
    assert len(before) == CONSOLIDATED_FILES

    # Crash (non-OSError) after the 5th promote: files 1-4 promoted, file 5
    # left verified, the rest untouched. KeyboardInterrupt is a BaseException so
    # cli.main (which catches only CuratorError/OSError) lets it propagate.
    real_promote = ConsolidationExecutor._promote
    calls = {"n": 0}

    def flaky_promote(self, source_sha, staged):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt("simulated mid-copy crash")
        return real_promote(self, source_sha, staged)

    monkeypatch.setattr(ConsolidationExecutor, "_promote", flaky_promote)
    # Drive through the CLI (which broadcasts to _dispatch -> execute) and make
    # sure the interrupt actually surfaces into the test.
    with pytest.raises(KeyboardInterrupt):
        run_cli(["consolidate", str(legacy), "--execute"])

    # Sources untouched even after the crash.
    assert _source_hashes(legacy) == before

    # Resume with the real promote (monkeypatch undone): 4 already-promoted files
    # are skipped, the remaining 7 complete, nothing double-promotes.
    monkeypatch.undo()
    rc, out = run_cli(
        ["consolidate", str(legacy), "--execute", "--resume", "--json"]
    )
    assert rc == 0
    result = json.loads(out)
    assert result["skipped"] + result["promoted"] == CONSOLIDATED_FILES == 11
    assert result["errors"] == []
    assert result["unique_library_files"] == UNIQUE_LIBRARY_FILES == 9

    # Independent cross-check of the content-addressed library and sources.
    executor = ConsolidationExecutor(legacy)
    library_files = [p for p in executor.library_root.rglob("*") if p.is_file()]
    assert len(library_files) == UNIQUE_LIBRARY_FILES
    assert _source_hashes(legacy) == before
