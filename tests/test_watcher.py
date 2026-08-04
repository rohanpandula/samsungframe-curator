"""Tests for the durable watcher (M005/S03): src/curator/watch/.

Proves stabilization (quiescence + decode confirmation), burst coalescing in
``poll_once``, idempotent enqueue, durable drain/restart, and SHA-vs-catalog
reconciliation. All tests avoid real sleeps by injecting a deterministic
``sleep`` and/or using ``settle_interval=0.0`` with small timeouts.
"""

from __future__ import annotations

import json

from PIL import Image

from curator.catalog import Catalog
from curator.hashing import sha256_hex
from curator.watch import Watcher, WatcherRunReport, WatcherService, stabilize


def _catalog(data_root):
    return Catalog(data_root=data_root)


def _write_png(path, size: int = 4, color: str = "red") -> None:
    Image.new("RGB", (size, size), color).save(path, "PNG")


# -- stabilization ---------------------------------------------------------


def test_stabilize_true_once_content_quiets(tmp_path):
    p = tmp_path / "img.png"
    _write_png(p)
    assert stabilize(p, settle_calls=2, settle_interval=0.0, decode_confirmation=True)


def test_stabilize_requires_quiescence(tmp_path):
    p = tmp_path / "changing.png"
    p.write_bytes(b"A" * 64)
    calls = {"n": 0}

    def keep_changing(_interval: float) -> None:
        calls["n"] += 1
        p.write_bytes(b"X" * (64 + calls["n"]))

    # Content changes every poll, so (size, mtime) never matches twice in a row:
    # stabilization keeps retrying until timeout and reports not-stable.
    assert (
        stabilize(
            p,
            settle_calls=2,
            settle_interval=0.0,
            timeout=0.05,
            decode_confirmation=False,
            sleep=keep_changing,
        )
        is False
    )
    # Once writes stop, the same file is reported stable.
    assert (
        stabilize(p, settle_calls=2, settle_interval=0.0, decode_confirmation=False)
        is True
    )


def test_stabilize_decode_confirmation_rejects_garbage(tmp_path):
    p = tmp_path / "garbage.png"
    p.write_bytes(b"definitely not a real image")
    # Quiescent, but undecodable => not stable with decode confirmation on.
    assert (
        stabilize(p, settle_calls=2, settle_interval=0.0, timeout=0.2,
                  decode_confirmation=True)
        is False
    )
    # Without decode confirmation the quiescent file is stable.
    assert (
        stabilize(p, settle_calls=2, settle_interval=0.0, decode_confirmation=False)
        is True
    )


# -- Watcher construction / burst coalescing -------------------------------


def test_poll_once_coalesces_repeated_sightings(data_root, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    watcher = Watcher(_catalog(data_root), src, decode_confirmation=False)
    f = src / "a.png"
    f.write_bytes(b"burst bytes")
    assert watcher.poll_once() == 1
    # Repeated sighting of the same path coalesces into the single prior enqueue.
    assert watcher.poll_once() == 0
    rows = watcher.queue_rows()
    assert len(rows) == 1
    assert rows[0]["path"] == str(f.resolve())
    assert rows[0]["state"] == "queued"


# -- idempotent enqueue -----------------------------------------------------


def test_enqueue_idempotent_path_and_done_sha(data_root, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    watcher = Watcher(_catalog(data_root), src, decode_confirmation=False)
    f = src / "b.png"
    f.write_bytes(b"idempotent")
    path = str(f.resolve())
    sha = sha256_hex(f.read_bytes())

    assert watcher.enqueue(path, sha) is True
    assert watcher.enqueue(path, sha) is False  # same path already queued
    assert len(watcher.queue_rows()) == 1

    watcher.mark_done(path)
    # After done, re-enqueuing the same sha (under a different path) is skipped.
    path2 = str((src / "c.png").resolve())
    assert watcher.enqueue(path2, sha) is False
    # A genuinely new sha on a fresh path is allowed.
    assert watcher.enqueue(path2, "other-sha") is True


# -- durable drain / restart -------------------------------------------------


def test_durable_restart_reclaims_inflight(data_root, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    watcher1 = Watcher(_catalog(data_root), src, decode_confirmation=False)
    paths = []
    for name in ("a.png", "b.png", "c.png"):
        f = src / name
        f.write_bytes(f"bytes-{name}".encode())
        paths.append(str(f.resolve()))
        watcher1.enqueue(paths[-1], sha256_hex(f.read_bytes()))

    assert len(list(watcher1.drain())) == 3
    watcher1.mark_done(paths[0])
    watcher1.mark_done(paths[1])
    # paths[2] was left 'processing' (simulated crash between enqueue and done).

    # "Restart": a fresh Watcher on the same catalog re-opens the same DB.
    watcher2 = Watcher(_catalog(data_root), src, decode_confirmation=False)
    drained = list(watcher2.drain())
    assert [r["path"] for r in drained] == [paths[2]]  # only the orphan reclaimed
    assert len(watcher2.queue_rows()) == 3  # nothing duplicated across drains
    states = {r["path"]: r["state"] for r in watcher2.queue_rows()}
    assert states == {
        paths[0]: "done",
        paths[1]: "done",
        paths[2]: "processing",
    }


def test_mark_error_keeps_row_and_does_not_duplicate(data_root, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    watcher = Watcher(_catalog(data_root), src, decode_confirmation=False)
    f = src / "e.png"
    f.write_bytes(b"will error")
    path = str(f.resolve())
    watcher.enqueue(path, sha256_hex(f.read_bytes()))
    watcher.mark_error(path, "decode failed")
    assert watcher.queue_rows()[0]["state"] == "error"
    assert watcher.enqueue(path, None) is True  # error rows are retryable


# -- reconciliation ----------------------------------------------------------


def test_reconcile_once_dedups_by_queue_and_done(data_root, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    watcher = Watcher(_catalog(data_root), src, decode_confirmation=False)
    f = src / "img.png"
    f.write_bytes(b"reconcile-me")
    sha = sha256_hex(f.read_bytes())

    assert watcher.reconcile_once() == 1
    # Still not cataloged, but already in the queue (any state) => no re-enqueue.
    assert watcher.reconcile_once() == 0
    assert len(watcher.queue_rows()) == 1

    watcher.mark_done(str(f.resolve()))
    # After done, reconciliation does NOT re-enqueue the same content.
    assert watcher.reconcile_once() == 0
    assert len(watcher.queue_rows()) == 1
    assert watcher.queue_rows()[0]["sha"] == sha


def test_reconcile_skips_content_already_in_catalog(data_root, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    catalog = _catalog(data_root)
    watcher = Watcher(catalog, src, decode_confirmation=False)
    f = src / "cat.png"
    data = b"already-cataloged"
    f.write_bytes(data)
    catalog.add_source("conn-local", "asset-1", data)

    # Content sha is already in the catalog => reconcile enqueues nothing.
    assert watcher.reconcile_once() == 0
    assert watcher.queue_rows() == []


# -- service / serialization ------------------------------------------------


def test_watcher_service_run_once_and_report_roundtrip(data_root, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    watcher = Watcher(_catalog(data_root), src, decode_confirmation=False)
    f = src / "run.png"
    f.write_bytes(b"run-once-bytes")

    service = WatcherService(watcher)
    report = service.run_once()
    assert report.enqueued == 1
    assert len(report.drained) == 1
    assert report.drained[0]["path"] == str(f.resolve())

    # to_dict / from_dict round-trip.
    assert WatcherRunReport.from_dict(report.to_dict()).to_dict() == report.to_dict()
    assert json.loads(report.to_json())["enqueued"] == 1

    # poll() surface: another poll finds nothing new (already enqueued/processed).
    assert service.poll() == 0


def test_poll_and_reconcile_exposed_on_service(data_root, tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    watcher = Watcher(_catalog(data_root), src, decode_confirmation=False)
    f = src / "p.png"
    f.write_bytes(b"poll")
    service = WatcherService(watcher)
    assert service.poll() == 1
    assert service.reconcile() == 0  # content already enqueued
