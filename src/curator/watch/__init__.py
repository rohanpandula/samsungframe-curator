"""Durable source watcher — stabilization, idempotent enqueue, reconciliation (S03).

This package makes watching a source folder durable and deterministic:

- :func:`stabilize` decides when a file has stopped changing (repeated identical
  size/mtime) so a half-written file is never enqueued.
- :class:`Watcher` scans a source root, coalesces event bursts per path, and
  idempotently enqueues stable paths into the ``watcher_queue`` table.
- :func:`Watcher.drain` yields unprocessed rows and marks them ``processing``,
  so nothing is lost if the process dies between enqueue and done.
- :func:`Watcher.reconcile_once` re-scans the source and enqueues any file whose
  SHA-256 is not yet in the catalog, so each stable source is processed once.
- :class:`WatcherService` ties poll + drain together in one synchronous step.

Everything is synchronous and sleep-free by default (``settle_interval=0.0``)
and takes an injectable ``time.sleep`` so tests stay deterministic.
"""

from __future__ import annotations

from curator.watch.watcher import (
    Watcher,
    WatcherRunReport,
    WatcherService,
    stabilize,
)

__all__ = [
    "Watcher",
    "WatcherRunReport",
    "WatcherService",
    "stabilize",
]
