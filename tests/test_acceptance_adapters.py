"""Acceptance gate for the M005 adapter / watcher / collection subsystem (S06).

This module ships the deterministic, air-gapped acceptance gate for the four
interchangeable subsystem surfaces added across M005/S01-S05. Each scenario is
**self-bootstrapping**: it mints a fresh :class:`~curator.catalog.Catalog` over
the isolated ``data_root`` (from conftest) and drives the subsystem objects
directly — never relying on cross-test ordering, a live server, or the network.

* S1 — destination adapters + transactional publish: filesystem exact-ID
  replace, simulator fault-injection rollback with idempotent resume, and the
  Home Assistant exclusive lease (acquired / second-acquire fails / released).
* S2 — durable watcher: a burst enqueues each path once, a restart over the
  same DB does no duplicate ingest, unprocessed rows drain, and reconciliation
  never re-enqueues already-done or already-cataloged content.
* S3 — deterministic rotation: a fixed-seed engine replays identically with
  explainable reasons, shuffled rotation never immediately repeats, the
  interval gate blocks/permits, show-now is honored, and state round-trips
  through the persisted store.
* S4 — Immich connector + feedback: browse + verified on-demand download,
  idempotent checkpointed sync, tombstoned (never deleted) references, and a
  disabled-by-default feedback sink that only writes confirmed favorites.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from curator.catalog import Catalog
from curator.collections import Playlist, RotationEngine, RotationState, RotationStore
from curator.connectors import (
    ImmichConnector,
    ImmichFeedbackSink,
    SyntheticImmichTransport,
)
from curator.dest import (
    FilesystemDestinationAdapter,
    PublishCoordinator,
    SimulatorDestinationAdapter,
)
from curator.dest.base import STATUS_APPLIED, STATUS_ERROR, DestinationError
from curator.ha import HomeAssistantCoordinationAdapter, SimulatorLeaseManager
from curator.hashing import sha256_hex
from curator.watch import Watcher


def _write_png(path: Path, color: str = "red") -> Path:
    """Encode a small synthetic PNG at *path* (deterministic, decodable)."""
    Image.new("RGB", (6, 6), color).save(path, "PNG")
    return path


# ---------------------------------------------------------------------------
# S1 — destination adapters + transactional publish + HA lease
# ---------------------------------------------------------------------------


def test_acceptance_destination_exact_id_replace_and_rollback(data_root):
    """FS exact-ID replace; simulator rollback + idempotent resume; HA lease."""
    # -- filesystem adapter: exact-ID replace keeps a single path -------------
    fs = FilesystemDestinationAdapter(data_root / "fs-dest")
    v1 = b"frame-v1"
    fs.put("frame.png", v1)
    assert fs.get_state("frame.png") == {
        "sha256": sha256_hex(v1),
        "size": len(v1),
    }
    assert fs.observe() == ["frame.png"]

    v2 = b"frame-v2-replacement"
    assert fs.replace("frame.png", v2) == sha256_hex(v2)
    assert fs.get_state("frame.png") == {
        "sha256": sha256_hex(v2),
        "size": len(v2),
    }
    assert fs.observe() == ["frame.png"]  # exact-ID replace, still one path

    # -- simulator: fail_next_write rollback + coordinator resume -------------
    db = Catalog(data_root=data_root).db
    sim = SimulatorDestinationAdapter()
    coord = PublishCoordinator(sim, db, adapter_id="sim")
    a1 = b"artifact-one"
    ok = coord.publish("art", a1)
    assert ok.status == STATUS_APPLIED
    assert sim.get_state("art")["sha256"] == sha256_hex(a1)

    # Exclusive lease: acquired during publish, rival refused, released after.
    lease = SimulatorLeaseManager()
    frame_ha = HomeAssistantCoordinationAdapter(lease)
    assert frame_ha.acquire_lease() is True
    assert lease.is_held()
    rival = HomeAssistantCoordinationAdapter(lease, holder="other-frame")
    with pytest.raises(DestinationError):
        rival.acquire_lease()  # exclusivity: second holder refused
    sim.fail_next_write()
    failed = coord.publish("art", b"artifact-two")
    frame_ha.release_lease()  # released once the publish is done
    assert not lease.is_held()

    # Failing write journals an error and leaves the prior artifact intact.
    assert failed.status == STATUS_ERROR
    assert sim.get_state("art")["sha256"] == sha256_hex(a1)  # keep last-known-good
    assert len(coord.rows_with_status(STATUS_ERROR)) == 1

    # Fresh coordinator + cleared fail flag: resumes on the SAME error row.
    coord2 = PublishCoordinator(sim, db, adapter_id="sim")
    resumed = coord2.publish("art", b"artifact-two")
    assert resumed.status == STATUS_APPLIED
    assert sim.get_state("art")["sha256"] == sha256_hex(b"artifact-two")
    assert len(coord2.rows_with_status(STATUS_ERROR)) == 0  # no dup error rows

    # Re-publishing the same sha is an idempotent no-op (no new journal row).
    rows_before = len(coord2.history("art"))
    idem = coord2.publish("art", b"artifact-two")
    assert idem.skipped and idem.status == STATUS_APPLIED
    assert len(coord2.history("art")) == rows_before


# ---------------------------------------------------------------------------
# S2 — durable watcher: process once, restart-safe, reconcile-once
# ---------------------------------------------------------------------------


def test_acceptance_watcher_process_once_and_reconcile(data_root, tmp_path):
    """A burst enqueues once; restart/reconcile never duplicate work."""
    src = tmp_path / "watch-src"
    src.mkdir()
    p1 = _write_png(src / "one.png", "red")
    p2 = _write_png(src / "two.png", "blue")

    catalog = Catalog(data_root=data_root)

    w1 = Watcher(
        catalog, src, decode_confirmation=True, settle_calls=1, settle_interval=0.0
    )
    # A burst pins each stable path to a single enqueue; re-poll adds nothing.
    assert w1.poll_once() == 2
    assert len(w1.queue_rows()) == 2
    assert w1.poll_once() == 0

    # Drain + mark_done processes both rows to a terminal state.
    drained = list(w1.drain())
    assert {r["path"] for r in drained} == {
        str(p1.resolve()),
        str(p2.resolve()),
    }
    for row in drained:
        w1.mark_done(row["path"])
    assert all(r["state"] == "done" for r in w1.queue_rows())

    # Restart: a NEW watcher over the same DB does no duplicate ingest.
    w2 = Watcher(
        catalog, src, decode_confirmation=True, settle_calls=1, settle_interval=0.0
    )
    assert w2.poll_once() == 0

    # A newly written file is picked up, and its unprocessed row drains.
    p3 = _write_png(src / "three.png", "green")
    assert w2.poll_once() == 1
    remaining = list(w2.drain())
    assert {r["path"] for r in remaining} == {str(p3.resolve())}
    w2.mark_done(str(p3.resolve()))
    assert w2.reconcile_once() == 0  # all done -> skip

    # Content already in the catalog is never re-enqueued by reconcile.
    p4 = _write_png(src / "four.png", "yellow")
    catalog.add_source("conn-local", "four", p4.read_bytes())
    assert w2.reconcile_once() == 0
    assert str(p4.resolve()) not in [r["path"] for r in w2.queue_rows()]


# ---------------------------------------------------------------------------
# S3 — deterministic, explainable rotation + persisted round-trip
# ---------------------------------------------------------------------------


def test_acceptance_rotation_deterministic_and_explainable(data_root):
    """Fixed-seed rotation replays identically; explainable, interval+show-now."""
    catalog = Catalog(data_root=data_root)
    ids: list[int] = []
    for i in range(5):
        catalog.add_source("conn-rot", f"r-{i}", f"rot-{i}".encode())
        entry = catalog.get_by_source("conn-rot", f"r-{i}")
        assert entry is not None
        ids.append(int(entry["id"]))

    engine = RotationEngine()
    now = datetime(2026, 1, 1, 12, 0, 0)
    seed = 7

    # No rotation interval on the deterministic playlist so the replay unrolls.
    det = Playlist(id=1, name="gallery", members=ids, shuffle=True)
    gated = Playlist(
        id=2, name="gated", members=ids, rotation_interval_seconds=60
    )

    def replay():
        seq: list[int] = []
        reasons: list[str] = []
        state = RotationState(playlist_id=1, seed=seed)
        t = now
        for _ in range(5):
            state, step = engine.advance(det, t, state, seed=seed)
            assert step.entry_id is not None
            seq.append(step.entry_id)
            reasons.extend(step.reason)
            t = t.replace(second=t.second + 1)
        return seq, reasons, state

    seq_a, reasons_a, state = replay()
    seq_b, _reasons_b, _state_b = replay()

    # Determinism: same seed + same times -> identical sequence.
    assert seq_a == seq_b
    assert reasons_a  # explainable, non-empty

    # Seeded shuffle: an entry never immediately follows itself.
    assert all(a != b for a, b in zip(seq_a, seq_a[1:]))

    # Interval gate: blocked before it elapses, permitted after.
    gated_state = RotationState(
        playlist_id=2, seed=seed, last_entry_id=ids[0], last_played_at=now
    )
    within = engine.advance(
        gated, now + timedelta(seconds=30), gated_state, seed=seed
    )[1]
    assert within.waiting
    assert "interval" in within.reason[0]
    after = engine.advance(
        gated, now + timedelta(seconds=61), gated_state, seed=seed
    )[1]
    assert not after.waiting
    assert after.entry_id != ids[0]

    # Show-now override is honored (bypasses interval/schedule).
    show = RotationState(
        playlist_id=1,
        seed=seed,
        last_entry_id=ids[0],
        last_played_at=now,
        override_entry_id=ids[2],
        override_active=True,
    )
    show_state, show_step = engine.advance(det, now, show, seed=seed)
    assert show_step.entry_id == ids[2]
    assert "show-now override" in show_step.reason

    # RotationStore persists playlist + state and round-trips via the DB.
    store = RotationStore(catalog)
    store.save_playlist(det)
    assert store.load_playlist(1) == det
    assert store.members(1) == ids
    store.save_state(state)
    assert store.load_state(1) == state
    assert show_state.seed == seed


# ---------------------------------------------------------------------------
# S4 — Immich sync: verified download, idempotent sync, tombstone, feedback
# ---------------------------------------------------------------------------


def test_acceptance_immich_sync_tombstone_feedback(data_root):
    """Browse/verified download, idempotent sync, tombstone, safe feedback."""
    transport = SyntheticImmichTransport(
        [
            {"asset_id": "a-1", "data": b"alpha-bytes", "favorite": False},
            {"asset_id": "b-2", "data": b"bravo-bytes"},
        ],
        page_size=1,  # exercise cursor pagination
    )
    catalog = Catalog(data_root=data_root)
    conn = ImmichConnector("immich-main", transport, catalog=catalog)

    # Browse + on-demand download whose SHA-256 equals the recorded checksum.
    metas = list(conn.enumerate())
    assert [m.asset_id for m in metas] == ["a-1", "b-2"]
    for meta in metas:
        assert (
            sha256_hex(conn.read_original(meta.asset_id))
            == transport.get_checksum(meta.asset_id)
        )

    # sync() persists state + cursor; a re-run does no duplicate work.
    first = conn.sync()
    assert first.enumerated == 2 and first.written == 2
    second = conn.sync()
    assert second.written == 0 and second.cursor == first.cursor

    # mark_unavailable: not enumerated, but the reference is preserved (never deleted).
    transport.mark_unavailable("b-2")
    assert "b-2" not in [m.asset_id for m in conn.enumerate()]
    tombstone = conn.sync()
    assert tombstone.tombstoned == 1
    row = catalog.db.execute(
        "SELECT available FROM immich_asset_state"
        " WHERE connector_id = ? AND asset_id = ?",
        ("immich-main", "b-2"),
    ).fetchone()
    assert row is not None and int(row[0]) == 0  # available=False, still present
    assert conn.sync().written == 0

    # Feedback sink is disabled by default -> a no-op, never a write.
    sink_off = ImmichFeedbackSink(transport, enabled=False)
    excluded = sink_off.write_favorite("a-1")
    assert excluded.ok is False and excluded.status == "excluded"
    assert transport.is_favorite("a-1") is False

    # Enabled + capability present -> writes a confirmed favorite, never deletes.
    sink_on = ImmichFeedbackSink(transport, enabled=True)
    written = sink_on.write_favorite("a-1", album_id="album-1")
    assert written.ok is True and written.status == "written"
    assert transport.album_members("album-1") == {"a-1"}
    assert "a-1" in [m.asset_id for m in conn.enumerate()]  # nothing deleted
