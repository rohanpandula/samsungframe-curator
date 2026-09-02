"""Tests for M005/S01 — destination adapters + transactional publish.

Covers the filesystem and simulator backends and the publish coordinator's
state machine (staged -> verified -> applied | error), its idempotent resume,
and its transactional rollback (a failed write leaves the prior artifact intact).
"""

from __future__ import annotations

import sqlite3

import pytest

from curator import db
from curator.dest import (
    DestJournalEntry,
    FilesystemDestinationAdapter,
    PublishCoordinator,
    PublishResult,
    SimulatorDestinationAdapter,
    publish,
)
from curator.dest.base import STATUS_APPLIED, STATUS_ERROR, DestinationError
from curator.hashing import sha256_hex


@pytest.fixture
def journal_db():
    """An in-memory catalog DB with all migrations (incl. v8 dest_journal) applied."""
    conn = sqlite3.connect(":memory:")
    db.migrate(conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Filesystem adapter
# ---------------------------------------------------------------------------


def test_filesystem_put_writes_exact_id_with_bytes_and_hash(tmp_path):
    adapter = FilesystemDestinationAdapter(tmp_path)
    data = b"hello frame"
    sha = adapter.put("art-1", data)

    path = tmp_path / "art-1"
    assert path.is_file()
    assert path.read_bytes() == data
    assert sha == sha256_hex(data)


def test_filesystem_replace_overwrites_same_id(tmp_path):
    adapter = FilesystemDestinationAdapter(tmp_path)
    adapter.put("art-1", b"first")

    first_path = tmp_path / "art-1"
    assert first_path.read_bytes() == b"first"

    sha = adapter.replace("art-1", b"second")
    assert (tmp_path / "art-1").is_file()
    assert (tmp_path / "art-1") == first_path
    assert (tmp_path / "art-1").read_bytes() == b"second"
    assert sha == sha256_hex(b"second")


def test_filesystem_remove_deletes(tmp_path):
    adapter = FilesystemDestinationAdapter(tmp_path)
    adapter.put("art-1", b"data")
    assert (tmp_path / "art-1").exists()

    adapter.remove("art-1")
    assert not (tmp_path / "art-1").exists()
    # removing a missing artifact is a no-op
    adapter.remove("art-1")


def test_filesystem_get_state_reflects_content(tmp_path):
    adapter = FilesystemDestinationAdapter(tmp_path)
    data = b"state bytes"
    adapter.put("art-1", data)

    state = adapter.get_state("art-1")
    assert state == {"sha256": sha256_hex(data), "size": len(data)}
    assert adapter.get_state("does-not-exist") is None


def test_filesystem_observe_lists_files(tmp_path):
    adapter = FilesystemDestinationAdapter(tmp_path)
    adapter.put("b-art", b"1")
    adapter.put("a-art", b"2")
    assert adapter.observe() == ["a-art", "b-art"]

    adapter.remove("a-art")
    assert adapter.observe() == ["b-art"]


# ---------------------------------------------------------------------------
# Simulator adapter
# ---------------------------------------------------------------------------


def test_simulator_put_records_ledger():
    adapter = SimulatorDestinationAdapter()
    data = b"sim-data"
    sha = adapter.put("art-1", data, meta={"tag": "x"})

    ledger = adapter.ledger()
    assert ledger["art-1"]["bytes"] == data
    assert ledger["art-1"]["sha"] == sha == sha256_hex(data)
    assert ledger["art-1"]["meta"] == {"tag": "x"}
    assert adapter.get_state("art-1") == {"sha256": sha, "size": len(data)}


def test_simulator_fail_next_write_leaves_prior_intact():
    adapter = SimulatorDestinationAdapter()
    adapter.put("art-1", b"good")

    adapter.fail_next_write()
    with pytest.raises(DestinationError):
        adapter.put("art-1", b"bad")

    # prior artifact is unchanged despite the failed put
    assert adapter.ledger()["art-1"]["bytes"] == b"good"
    assert adapter.get_state("art-1") == {
        "sha256": sha256_hex(b"good"),
        "size": len(b"good"),
    }
    # the failure flag was consumed by the failed write
    assert adapter.put("art-1", b"after") == sha256_hex(b"after")
    assert adapter.ledger()["art-1"]["bytes"] == b"after"


def test_simulator_replace_under_pending_failure_leaves_prior():
    adapter = SimulatorDestinationAdapter()
    adapter.put("art-1", b"prior")

    adapter.fail_next_write()
    with pytest.raises(DestinationError):
        adapter.replace("art-1", b"replacement")

    assert adapter.ledger()["art-1"]["bytes"] == b"prior"


def test_simulator_remove_and_observe():
    adapter = SimulatorDestinationAdapter()
    adapter.put("x", b"1")
    adapter.put("y", b"2")
    assert adapter.observe() == ["x", "y"]

    adapter.remove("x")
    assert adapter.observe() == ["y"]
    assert adapter.get_state("x") is None

    adapter.reset()
    assert adapter.observe() == []
    assert adapter.ledger() == {}


# ---------------------------------------------------------------------------
# Publish coordinator
# ---------------------------------------------------------------------------


def test_publish_success_journals_staged_verified_applied(journal_db):
    adapter = SimulatorDestinationAdapter()
    coordinator = PublishCoordinator(adapter, journal_db)
    data = b"publish me"

    result = coordinator.publish("art-1", data)
    assert result.status == STATUS_APPLIED
    assert result.skipped is False
    assert result.sha == sha256_hex(data)

    history = coordinator.history("art-1")
    assert len(history) == 1
    assert history[0].status == STATUS_APPLIED
    assert history[0].sha == sha256_hex(data)
    assert history[0].artifact_id == "art-1"
    assert history[0].adapter_id == "SimulatorDestinationAdapter"
    assert history[0].op == "put"

    # ledger reflects the applied artifact
    assert adapter.ledger()["art-1"]["bytes"] == data


def test_publish_republish_same_sha_is_idempotent(journal_db):
    adapter = SimulatorDestinationAdapter()
    coordinator = PublishCoordinator(adapter, journal_db)
    data = b"same bytes"

    first = coordinator.publish("art-1", data)
    assert first.status == STATUS_APPLIED
    assert first.skipped is False

    second = coordinator.publish("art-1", data)
    assert second.skipped is True
    assert second.status == STATUS_APPLIED

    # no duplicate 'applied' row for the same sha
    assert len(coordinator.rows_with_status(STATUS_APPLIED)) == 1
    assert len(coordinator.history("art-1")) == 1
    assert len(adapter.ops()) == 1  # adapter.put ran exactly once


def test_publish_failing_adapter_journals_error_and_keeps_prior(journal_db):
    adapter = SimulatorDestinationAdapter()
    coordinator = PublishCoordinator(adapter, journal_db)

    prior = coordinator.publish("art-1", b"prior-good")
    assert prior.status == STATUS_APPLIED

    adapter.fail_next_write()
    result = coordinator.publish("art-1", b"new-version")
    assert result.status == STATUS_ERROR
    assert result.error is not None

    # prior artifact remains intact on the destination
    assert adapter.ledger()["art-1"]["bytes"] == b"prior-good"

    # journal captured the error, and the latest row for art-1 is 'error'
    assert len(coordinator.rows_with_status(STATUS_ERROR)) == 1
    assert coordinator.history("art-1")[-1].status == STATUS_ERROR


def test_publish_resume_after_failure_reapplies_no_duplicate_error(journal_db):
    adapter = SimulatorDestinationAdapter()
    coordinator = PublishCoordinator(adapter, journal_db)
    data = b"retry me"

    adapter.fail_next_write()
    failed = coordinator.publish("art-1", data)
    assert failed.status == STATUS_ERROR
    assert len(coordinator.rows_with_status(STATUS_ERROR)) == 1
    assert len(coordinator.history("art-1")) == 1

    # clear the failure flag and resume: the same row becomes 'applied'
    resumed = coordinator.publish("art-1", data)
    assert resumed.status == STATUS_APPLIED
    assert resumed.skipped is False

    # still exactly one row total; no duplicate error row
    history = coordinator.history("art-1")
    assert len(history) == 1
    assert history[0].status == STATUS_APPLIED
    assert len(coordinator.rows_with_status(STATUS_ERROR)) == 0
    assert adapter.ledger()["art-1"]["bytes"] == data


def test_publish_replace_op_uses_adapter_replace(journal_db):
    adapter = SimulatorDestinationAdapter()
    coordinator = PublishCoordinator(adapter, journal_db)
    coordinator.publish("art-1", b"v1")
    result = coordinator.publish("art-1", b"v2", op="replace")

    assert result.status == STATUS_APPLIED
    assert adapter.ops()[-1]["op"] == "replace"
    assert adapter.ledger()["art-1"]["bytes"] == b"v2"


def test_publish_function_wrapper(journal_db):
    adapter = SimulatorDestinationAdapter()
    data = b"wrapped"
    result = publish(adapter, journal_db, "art-1", data)

    assert result.status == STATUS_APPLIED
    assert result.sha == sha256_hex(data)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        DestJournalEntry(
            id=3,
            adapter_id="SimulatorDestinationAdapter",
            artifact_id="art-1",
            op="put",
            sha="abc123",
            status="applied",
            error=None,
            created_at="2026-01-01T00:00:00.000Z",
        ),
        DestJournalEntry(
            id=None,
            adapter_id="x",
            artifact_id="a",
            op="replace",
            sha=None,
            status="error",
            error="boom",
            created_at=None,
        ),
    ],
)
def test_dest_journal_entry_json_round_trip(entry):
    assert DestJournalEntry.from_dict(entry.to_dict()) == entry


def test_publish_result_json_round_trip():
    result = PublishResult(
        artifact_id="art-1",
        adapter_id="FilesystemDestinationAdapter",
        op="put",
        status="applied",
        sha="deadbeef",
        skipped=False,
        row_id=7,
        error=None,
    )
    assert PublishResult.from_dict(result.to_dict()) == result
    assert result.to_json()


def test_dest_journal_entry_ignores_unknown_keys():
    entry = DestJournalEntry.from_dict(
        {"id": 1, "artifact_id": "a", "status": "applied", "bogus": 99}
    )
    assert entry.artifact_id == "a"
    assert not hasattr(entry, "bogus")


def test_journal_idempotency_is_scoped_per_adapter(tmp_path) -> None:
    """M011/S01: the same artifact id on two destinations applies on both.

    Before the fix, "applied on the simulator" made a later folder publish of the
    same artifact id report ``skipped`` — the journal lookup ignored adapter_id.
    """
    from curator.db import connect, migrate
    from curator.dest.filesystem import FilesystemDestinationAdapter
    from curator.dest.publish import PublishCoordinator
    from curator.dest.simulator import SimulatorDestinationAdapter

    db = connect(tmp_path / "root")
    migrate(db)
    data = b"same-bytes"
    first = PublishCoordinator(SimulatorDestinationAdapter(), db, adapter_id="sim").publish(
        "wall-001.png", data
    )
    second = PublishCoordinator(
        FilesystemDestinationAdapter(tmp_path / "usb"), db, adapter_id="folder"
    ).publish("wall-001.png", data)
    assert first.skipped is False and second.skipped is False
    assert (tmp_path / "usb" / "wall-001.png").read_bytes() == data
    # Re-publishing on the folder alone is still idempotent.
    third = PublishCoordinator(
        FilesystemDestinationAdapter(tmp_path / "usb"), db, adapter_id="folder"
    ).publish("wall-001.png", data)
    assert third.skipped is True
