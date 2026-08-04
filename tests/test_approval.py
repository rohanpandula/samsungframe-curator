"""Tests for src/curator/approve/approval.py (M003/S03, R010).

Proves the append-only approval/history contract: approve/reject persist and
``current`` reflects the latest resolved decision while ``history`` retains both;
undo/redo toggle the active decision by appending TRANSITIONS without ever
deleting a row (history length grows and every prior decision stays present);
batch_approve marks a set; ``current`` is ``None`` before any decision; and
:class:`ApprovalEvent` round-trips through JSON.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from curator.approve import ApprovalError, ApprovalEvent, ApprovalService, Decision
from curator.catalog import Catalog


@pytest.fixture
def catalog(data_root):
    """A Catalog backed by a fresh migrated DB + ContentStore under the data root."""
    return Catalog(data_root=data_root)


@pytest.fixture
def service(catalog):
    """An ApprovalService sharing the Catalog's migrated DB."""
    return ApprovalService(catalog)


def _add(catalog, asset_id: str) -> int:
    """Add one source entry and return its catalog entry id."""
    catalog.add_source("conn-local", asset_id, b"approval-photo-bytes-" + asset_id.encode())
    entry = catalog.get_by_source("conn-local", asset_id)
    assert entry is not None
    return int(entry["id"])


def test_current_none_before_any_decision(catalog, service):
    entry_id = _add(catalog, "asset-0")
    assert service.current(entry_id) is None
    assert service.history(entry_id) == []


def test_approve_persists_and_current_reflects_latest(catalog, service):
    entry_id = _add(catalog, "asset-1")
    ev = service.approve(entry_id, rationale="great light")

    assert ev.decision is Decision.APPROVED
    assert ev.rationale == "great light"
    assert ev.seq == 1
    assert ev.created_at

    cur = service.current(entry_id)
    assert cur is not None
    assert cur.decision is Decision.APPROVED
    assert cur.rationale == "great light"

    # Rationale and decision are genuinely persisted in SQLite.
    row = catalog.db.execute(
        "SELECT decision, rationale FROM approvals WHERE catalog_entry_id = ?",
        (entry_id,),
    ).fetchone()
    assert row == ("APPROVED", "great light")

    # A later reject supersedes: current reflects the latest, history keeps both.
    ev2 = service.reject(entry_id, rationale="too grainy")
    cur2 = service.current(entry_id)
    assert cur2 is not None
    assert cur2.decision is Decision.REJECTED
    assert cur2.rationale == "too grainy"
    assert ev2.seq == 2
    decisions = [e.decision for e in service.history(entry_id)]
    assert decisions == [Decision.APPROVED, Decision.REJECTED]


def test_validates_catalog_entry_exists(catalog, service):
    with pytest.raises(ApprovalError):
        service.approve(999_999)
    with pytest.raises(ApprovalError):
        service.reject(999_999)


def test_undo_redo_append_without_erasure(catalog, service):
    entry_id = _add(catalog, "asset-2")
    service.approve(entry_id, rationale="initial approve")
    service.reject(entry_id, rationale="reject it")

    history_before = service.history(entry_id)
    assert len(history_before) == 2

    # undo reverts the latest (REJECTED -> APPROVED) by APPENDING a row.
    undone = service.undo(entry_id)
    assert undone.decision is Decision.APPROVED
    assert service.current(entry_id).decision is Decision.APPROVED

    # redo re-applies the previously active decision (APPROVED) by appending again.
    redone = service.redo(entry_id)
    assert redone.decision is Decision.REJECTED
    assert service.current(entry_id).decision is Decision.REJECTED

    # No history row was ever deleted: length grew and every prior decision remains.
    history = service.history(entry_id)
    assert len(history) == 4
    assert len(history_before) < len(history)
    assert [e.decision for e in history] == [
        Decision.APPROVED,
        Decision.REJECTED,
        Decision.APPROVED,
        Decision.REJECTED,
    ]
    # Every prior event is still present verbatim.
    for prior in history_before:
        assert prior in history

    # sequp monotonically per entry.
    assert [e.seq for e in history] == [1, 2, 3, 4]


def test_undo_redo_with_no_history_raises(catalog, service):
    entry_id = _add(catalog, "asset-3")
    with pytest.raises(ApprovalError):
        service.undo(entry_id)
    with pytest.raises(ApprovalError):
        service.redo(entry_id)


def test_batch_approve_marks_all(catalog, service):
    ids = [_add(catalog, f"batch-{i}") for i in range(3)]
    events = service.batch_approve(ids, rationale="approved in batch")

    assert [e.decision for e in events] == [Decision.APPROVED] * 3
    assert all(e.rationale == "approved in batch" for e in events)
    for entry_id in ids:
        cur = service.current(entry_id)
        assert cur is not None
        assert cur.decision is Decision.APPROVED


def test_approval_event_json_round_trip(service):
    event = ApprovalEvent(
        catalog_entry_id=42,
        decision=Decision.REJECTED,
        rationale="too dark to pair",
        created_at="2026-01-01T00:00:00.000Z",
        seq=3,
    )
    encoded = json.dumps(event.to_dict())
    restored = ApprovalEvent.from_dict(json.loads(encoded))
    assert restored == event
    assert restored.decision is Decision.REJECTED
    assert event.to_dict() == {
        "catalog_entry_id": 42,
        "decision": "REJECTED",
        "rationale": "too dark to pair",
        "created_at": "2026-01-01T00:00:00.000Z",
        "seq": 3,
    }


def test_service_accepts_raw_connection(catalog):
    svc = ApprovalService(catalog.db)
    assert isinstance(svc.db, sqlite3.Connection)
