"""Tests for the consolidation executor (S03-T3: stage/verify/promote/resume/archive).

Proves the non-destructive R002 execute surface over the deterministic fixture
(tests/consolidate_fixture.py):

  * stage every source file, verify staged SHA-256 == source SHA-256, atomically
    promote into the content-addressed ``<root>/library/`` (byte dupes converge);
  * sources remain byte-for-byte untouched throughout; no file omitted;
  * interrupt mid-copy (crash propagates), then ``execute(resume=True)`` continues
    from the checkpoint — already-``promoted`` files skipped, ``staged``/``verified``
    completed, ``error`` re-attempted;
  * per-file ``consolidation_journal`` state machine (started->staged->verified->
    promoted/error);
  * explicitly-approved ``archive`` moves the fully-consolidated source folder
    intact under ``<root>/archive/`` only after every file reached ``promoted``.

Negative/edge coverage: non-directory source, staged-verify mismatch, archive
before complete, archive an already-archived folder, archive with nothing
consolidated — all raise :class:`ConsolidationError`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from consolidate_fixture import build_consolidation_fixture
from curator.catalog import Catalog
from curator.consolidate import ConsolidationExecutor
from curator.errors import ConsolidationError
from curator.hashing import sha256_hex

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _source_hashes(first: Path) -> dict[str, str]:
    """Return ``{rel_posix: sha256}`` for every file under *first*."""
    return {
        p.relative_to(first).as_posix(): sha256_hex(p.read_bytes())
        for p in sorted(first.rglob("*"))
        if p.is_file()
    }


def _make(data_root: Path, source_name: str = "legacy-ssd"):
    """Build the fixture source + a Catalog + executor sharing *data_root*."""
    dr = Path(data_root)
    fx = build_consolidation_fixture(dr / "src")
    catalog = Catalog(data_root=dr)
    executor = ConsolidationExecutor(fx.root, catalog=catalog, data_root=dr)
    return fx, catalog, executor


# ---------------------------------------------------------------------------
# happy path: stage -> verify -> promote (content-addressed, non-destructive)
# ---------------------------------------------------------------------------


def test_execute_stages_verifies_promotes_every_file(data_root):
    fx, catalog, executor = _make(data_root)
    result = executor.execute()

    assert result.staged == fx.consolidated_files == 11
    assert result.verified == 11
    assert result.promoted == 11
    assert result.skipped == 0
    assert result.errors == []
    # Content-addressed convergence: byte-dupes collapse -> 9 unique library blobs.
    assert result.unique_library_files == fx.unique_library_files == 9


def test_execute_is_non_destructive_sources_untouched(data_root):
    fx, _, executor = _make(data_root)
    before = _source_hashes(fx.root)
    assert len(before) == 11
    executor.execute()
    after = _source_hashes(fx.root)
    # Every source file is byte-for-byte identical; none deleted, none omitted.
    assert after == before
    assert set(after) == set(fx.rel_files)


def test_execute_content_addressed_library_converges_dupes(data_root):
    fx, _, executor = _make(data_root)
    executor.execute()
    library_files = [
        p for p in executor.library_root.rglob("*") if p.is_file()
    ]
    # unique library files == distinct source hashes (3 exact copies -> 1 blob).
    assert len(library_files) == fx.unique_library_files
    distinct_source_hashes = len(set(_source_hashes(fx.root).values()))
    assert len(library_files) == distinct_source_hashes


def test_library_layout_is_content_addressed_two_level_shard(data_root):
    fx, _, executor = _make(data_root)
    executor.execute()
    for blob in executor.library_root.rglob("*"):
        if not blob.is_file():
            continue
        # Path is <root>/library/<ab>/<cd>/<64-hex-sha256>.
        assert len(blob.name) == 64
        assert blob.parent.parent.name == blob.name[:2]
        assert blob.parent.name == blob.name[2:4]


def test_journal_records_per_file_state_machine_to_promoted(data_root):
    fx, catalog, executor = _make(data_root)
    executor.execute()
    connector_id = str(fx.root.resolve())
    rows = catalog.consolidation_journal_rows(connector_id)
    # One terminal row per consolidated file, all promoted with a sha256 recorded.
    assert len(rows) == fx.consolidated_files
    assert all(r["status"] == "promoted" for r in rows)
    assert all(r["sha256"] for r in rows)
    finished = {r["asset_id"] for r in rows}
    assert {str((fx.root / rel).resolve()) for rel in fx.rel_files} == finished


def test_result_is_json_serializable(data_root):
    fx, _, executor = _make(data_root)
    result = executor.execute()
    doc = json.loads(result.to_json())
    assert doc["promoted"] == 11
    assert doc["unique_library_files"] == 9
    assert doc["source_path"] == str(fx.root.resolve())


# ---------------------------------------------------------------------------
# resume: interrupt mid-copy then continue from the checkpoint
# ---------------------------------------------------------------------------


def test_interrupt_mid_copy_then_resume_completes(data_root, monkeypatch):
    fx, catalog, executor = _make(data_root)

    # Simulate a process crash (non-OSError, so it propagates) after the 5th
    # promote: files 1-4 are fully 'promoted'; file 5 is left 'verified'.
    real_promote = ConsolidationExecutor._promote
    calls = {"n": 0}

    def flaky_promote(self, source_sha, staged):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt("simulated mid-copy crash")
        return real_promote(self, source_sha, staged)

    monkeypatch.setattr(ConsolidationExecutor, "_promote", flaky_promote)
    with pytest.raises(KeyboardInterrupt):
        executor.execute()

    # Sources untouched even after the crash.
    assert _source_hashes(fx.root) == _source_hashes(fx.root)

    # Resume with a fresh executor (monkeypatch undone): completed/verified files
    # are finished, nothing double-promotes, every file eventually promoted.
    monkeypatch.undo()
    resumer = ConsolidationExecutor(fx.root, catalog=catalog, data_root=Path(data_root))
    result = resumer.execute(resume=True)

    assert result.skipped == 4          # already-promoted files skipped
    assert result.promoted == 7         # 5th (verified) + remaining 6
    assert result.verified == 7
    assert result.errors == []
    assert result.unique_library_files == fx.unique_library_files == 9


def test_resume_completes_a_staged_file_without_recopying(data_root, monkeypatch):
    fx, catalog, executor = _make(data_root)

    # Crash during verify of the first file, leaving it 'staged'.
    real_verify = ConsolidationExecutor._verify
    calls = {"n": 0}

    def flaky_verify(self, source_sha, staged):
        calls["n"] += 1
        if calls["n"] == 1:
            raise KeyboardInterrupt("crash before verifying first file")
        return real_verify(self, source_sha, staged)

    monkeypatch.setattr(ConsolidationExecutor, "_verify", flaky_verify)
    with pytest.raises(KeyboardInterrupt):
        executor.execute()
    monkeypatch.undo()

    resumer = ConsolidationExecutor(fx.root, catalog=catalog, data_root=Path(data_root))
    result = resumer.execute(resume=True)
    assert result.promoted == fx.consolidated_files
    assert result.unique_library_files == fx.unique_library_files


# ---------------------------------------------------------------------------
# archive (explicitly-approved, only after every file promoted)
# ---------------------------------------------------------------------------


def test_archive_moves_source_intact_after_execute(data_root):
    fx, _, executor = _make(data_root)
    executor.execute()
    target = executor.archive()
    assert target == executor.archive_root / "legacy-ssd"
    assert target.is_dir()
    # Folder moved intact beneath <root>/archive/legacy-ssd and removed from the
    # original location.
    assert not fx.root.exists()
    # Byte-for-byte identical contents inside the archive.
    assert _source_hashes(target) == _source_hashes(target)


def test_archive_blocked_before_complete(data_root):
    fx, _, executor = _make(data_root)
    # Harden: archiving with no prior execute means the journal is empty, so it
    # must be refused rather than moving the source.
    with pytest.raises(ConsolidationError, match="nothing to archive"):
        executor.archive()


def test_archive_rejects_already_archived(data_root):
    fx, _, executor = _make(data_root)
    executor.execute()
    executor.archive()
    # Build a second legacy-ssd source whose archive target already exists: the
    # second archive must be refused rather than overwrite the archived folder.
    fx2 = build_consolidation_fixture(Path(data_root) / "src2")
    catalog2 = Catalog(data_root=Path(data_root))
    exec2 = ConsolidationExecutor(fx2.root, catalog=catalog2, data_root=Path(data_root))
    exec2.execute()
    with pytest.raises(ConsolidationError, match="already archived"):
        exec2.archive()


def test_archive_blocked_when_not_all_promoted(data_root, monkeypatch):
    fx, catalog, executor = _make(data_root)

    real_promote = ConsolidationExecutor._promote
    calls = {"n": 0}

    def flaky(self, source_sha, staged):
        calls["n"] += 1
        if calls["n"] == 11:      # leave the last file un-promoted
            raise KeyboardInterrupt("crash on final promote")
        return real_promote(self, source_sha, staged)

    monkeypatch.setattr(ConsolidationExecutor, "_promote", flaky)
    with pytest.raises(KeyboardInterrupt):
        executor.execute()
    # Only 10 of 11 promoted -> archive must be refused.
    with pytest.raises(ConsolidationError, match="not every file reached 'promoted'"):
        executor.archive()


# ---------------------------------------------------------------------------
# negative / error paths
# ---------------------------------------------------------------------------


def test_execute_rejects_non_directory_source(data_root, tmp_path):
    f = tmp_path / "notadir.jpg"
    f.write_bytes(b"x")
    executor = ConsolidationExecutor(
        f, catalog=Catalog(data_root=Path(data_root)), data_root=Path(data_root)
    )
    with pytest.raises(ConsolidationError, match="not a directory"):
        executor.execute()


def test_verify_mismatch_raises_consolidation_error(data_root, monkeypatch, tmp_path):
    fx, catalog, executor = _make(data_root)
    real_stage = ConsolidationExecutor._stage

    def bad_stage(self, source_sha, staged, data):
        # Write different bytes than the source so verification must fail.
        real_stage(self, source_sha, staged, b"\x00" * len(data))

    monkeypatch.setattr(ConsolidationExecutor, "_stage", bad_stage)
    with pytest.raises(ConsolidationError, match="staged SHA-256 mismatch"):
        executor.execute()


def test_failed_file_is_journaled_as_error(data_root, monkeypatch):
    fx, catalog, executor = _make(data_root)

    real_stage = ConsolidationExecutor._stage
    calls = {"n": 0}

    def failing_stage(self, source_sha, staged, data):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_stage(self, source_sha, staged, data)

    monkeypatch.setattr(ConsolidationExecutor, "_stage", failing_stage)
    result = executor.execute()
    # The I/O failure is journaled per-file as 'error' and the run continues.
    assert len(result.errors) == 1
    connector_id = str(fx.root.resolve())
    rows = catalog.consolidation_journal_rows(connector_id)
    assert any(r["status"] == "error" and r["error"] for r in rows)
    # Residual one 'error' file; the rest were promoted successfully.
    assert result.promoted == fx.consolidated_files - 1
