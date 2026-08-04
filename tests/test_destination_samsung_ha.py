"""Tests for M005/S02 — Samsung Art Mode + HA coordination.

Covers the canary-then-exact-ID-replace publish flow, single-writer lease
exclusivity, rollback/restore on transport failure, idempotent re-publish, and
the air-gapped (simulator-only, no sockets) property. All transports are
simulators, so the suite is deterministic.
"""

from __future__ import annotations

import socket
import sqlite3

import pytest

from curator import db
from curator.dest.base import (
    OP_REPLACE,
    STATUS_APPLIED,
    STATUS_ERROR,
    DestinationError,
)
from curator.dest.samsung import (
    OP_CANARY,
    SamsungArtModeDestinationAdapter,
    SimulatorSamsungTransport,
)
from curator.dest.simulator import SimulatorDestinationAdapter
from curator.ha import HomeAssistantCoordinationAdapter, SimulatorLeaseManager
from curator.hashing import sha256_hex


@pytest.fixture
def journal_db():
    """An in-memory catalog DB with all migrations (incl. v8 dest_journal) applied."""
    conn = sqlite3.connect(":memory:")
    db.migrate(conn)
    yield conn
    conn.close()


def _make_adapter(
    journal_db,
    sim: SimulatorDestinationAdapter | None = None,
    ha: HomeAssistantCoordinationAdapter | None = None,
    prior_state: dict | None = None,
):
    sim = sim or SimulatorDestinationAdapter()
    transport = SimulatorSamsungTransport(sim)
    if ha is None:
        ha = HomeAssistantCoordinationAdapter(
            SimulatorLeaseManager(), prior_automation_state=prior_state
        )
    adapter = SamsungArtModeDestinationAdapter(transport, ha, db=journal_db)
    return sim, transport, ha, adapter


# ---------------------------------------------------------------------------
# Canary then exact-ID replace
# ---------------------------------------------------------------------------


def test_canary_then_exact_id_replace(journal_db):
    sim, transport, ha, adapter = _make_adapter(journal_db)
    data = b"frame art bytes"
    sha = adapter.put("art-1", data)

    # transport received a canary op first, then an exact-ID replace
    ops = sim.ops()
    assert ops[0]["op"] == "put"
    assert ops[0]["artifact_id"].startswith("canary:")
    assert ops[1]["op"] == "replace"
    assert ops[1]["artifact_id"] == "art-1"

    # final state has the artifact
    assert sha == sha256_hex(data)
    assert adapter.get_state("art-1") == {
        "sha256": sha256_hex(data),
        "size": len(data),
    }

    # dest_journal has applied rows for both phases
    applied = journal_db.execute(
        "SELECT op, status FROM dest_journal WHERE status = ? ORDER BY id",
        (STATUS_APPLIED,),
    ).fetchall()
    assert [row[0] for row in applied] == [OP_CANARY, OP_REPLACE]
    assert all(row[1] == STATUS_APPLIED for row in applied)


def test_lease_released_after_successful_publish(journal_db):
    lease = SimulatorLeaseManager()
    ha = HomeAssistantCoordinationAdapter(lease)
    sim, transport, ha, adapter = _make_adapter(journal_db, ha=ha)
    adapter.put("art-1", b"data")
    assert not lease.is_held()


# ---------------------------------------------------------------------------
# Rollback on failure
# ---------------------------------------------------------------------------


def test_rollback_on_failure_restores_prior_and_journals_error(journal_db):
    sim = SimulatorDestinationAdapter()
    transport = SimulatorSamsungTransport(sim)
    lease = SimulatorLeaseManager()
    prior_state = {"mode": "off", "brightness": 50}
    ha = HomeAssistantCoordinationAdapter(lease, prior_automation_state=prior_state)
    adapter = SamsungArtModeDestinationAdapter(transport, ha, db=journal_db)

    adapter.put("art-1", b"prior-good")
    assert ha.restore_calls == 0

    transport.fail_next_write()
    with pytest.raises(DestinationError):
        adapter.put("art-1", b"new-version")

    # prior artifact intact
    assert adapter.get_state("art-1") == {
        "sha256": sha256_hex(b"prior-good"),
        "size": len(b"prior-good"),
    }
    # HA prior state restored via the coordination adapter
    assert ha.restore_calls == 1
    assert ha.current_state == prior_state
    # lease released even on failure
    assert not lease.is_held()
    # journal has exactly one error row for the failed replace
    error_rows = journal_db.execute(
        "SELECT op, status, error FROM dest_journal WHERE status = ?",
        (STATUS_ERROR,),
    ).fetchall()
    assert len(error_rows) == 1
    assert error_rows[0][0] == OP_REPLACE
    assert "simulated Samsung write failure" in error_rows[0][2]


def test_publish_under_held_lease_raises_before_any_write(journal_db):
    lease = SimulatorLeaseManager()
    lease.acquire("frame-a")
    ha = HomeAssistantCoordinationAdapter(lease)
    sim, transport, ha, adapter = _make_adapter(journal_db, ha=ha)

    with pytest.raises(DestinationError):
        adapter.put("art-1", b"blocked")

    # no canary/replace write reached the transport
    assert sim.ops() == []
    # the other holder still owns the lease; no premature restore
    assert lease.holder == "frame-a"
    assert ha.restore_calls == 0


# ---------------------------------------------------------------------------
# Lease exclusivity
# ---------------------------------------------------------------------------


def test_lease_exclusivity():
    lease = SimulatorLeaseManager()
    assert lease.acquire("a") is True
    assert lease.is_held()
    assert lease.holder == "a"
    # a different holder cannot acquire
    assert lease.acquire("b") is False
    # same holder is re-entrant
    assert lease.acquire("a") is True
    lease.release("a")
    assert not lease.is_held()
    assert lease.holder is None
    # now free for the other holder
    assert lease.acquire("b") is True
    assert lease.holder == "b"
    # a non-holder release is a no-op
    lease.release("c")
    assert lease.holder == "b"


def test_ha_acquire_conflict_raises_destination_error():
    lease = SimulatorLeaseManager()
    first = HomeAssistantCoordinationAdapter(lease, holder="frame-a")
    second = HomeAssistantCoordinationAdapter(lease, holder="frame-b")

    assert first.acquire_lease() is True
    with pytest.raises(DestinationError):
        second.acquire_lease()
    first.release_lease()
    assert second.acquire_lease() is True
    second.release_lease()
    assert not lease.is_held()


# ---------------------------------------------------------------------------
# HA restore
# ---------------------------------------------------------------------------


def test_ha_restore_prior_state_restores_recorded_state():
    prior = {"mode": "off"}
    ha = HomeAssistantCoordinationAdapter(
        SimulatorLeaseManager(), prior_automation_state=prior
    )
    ha.current_state = {"mode": "on", "via": "publish"}
    restored = ha.restore_prior_state()
    assert ha.restore_calls == 1
    assert restored == prior
    assert ha.current_state == prior


# ---------------------------------------------------------------------------
# Idempotent / replace
# ---------------------------------------------------------------------------


def test_republish_same_id_same_bytes_is_replace_not_error(journal_db):
    sim, transport, ha, adapter = _make_adapter(journal_db)
    data = b"stable art"
    first = adapter.put("art-1", data)
    second = adapter.put("art-1", data)

    assert first == second == sha256_hex(data)
    # exact-ID replace ran again; no duplicate error
    art_ops = [o for o in sim.ops() if o["artifact_id"] == "art-1"]
    assert len(art_ops) == 2
    assert all(o["op"] == "replace" for o in art_ops)
    assert journal_db.execute(
        "SELECT COUNT(*) FROM dest_journal WHERE status = ?",
        (STATUS_ERROR,),
    ).fetchone()[0] == 0
    # both attempts applied
    assert journal_db.execute(
        "SELECT COUNT(*) FROM dest_journal WHERE op = ? AND status = ?",
        (OP_REPLACE, STATUS_APPLIED),
    ).fetchone()[0] == 2
    # final state intact
    assert adapter.get_state("art-1")["sha256"] == sha256_hex(data)


# ---------------------------------------------------------------------------
# Air-gapped
# ---------------------------------------------------------------------------


def test_air_gapped_no_network_sockets(journal_db, monkeypatch):
    def _no_socket(*args, **kwargs):
        raise AssertionError("air-gapped publish must not open network sockets")

    monkeypatch.setattr(socket, "socket", _no_socket)
    sim, transport, ha, adapter = _make_adapter(journal_db)
    adapter.put("art-1", b"offline art")
    assert adapter.get_state("art-1") == {
        "sha256": sha256_hex(b"offline art"),
        "size": len(b"offline art"),
    }
