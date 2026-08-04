"""Durable watcher internals — stabilization, enqueue, drain, reconciliation.

See :mod:`curator.watch` for the subsystem overview. This module holds the
concrete implementation. All timestamps use SQLite defaults; all size/mtime
snapshots use nanosecond precision for deterministic stabilization.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from curator.catalog import Catalog
from curator.hashing import sha256_hex
from curator.ingest.decode import decode_image

# Decodable image suffixes the watcher tracks (matches LocalConnector's
# supported set; RAW/unsupported files are intentionally not watched).
_SUPPORTED = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".tiff", ".tif")

# watcher_queue state machine (mirrors journal posture):
#   queued -> processing -> done | error
STATE_QUEUED = "queued"
STATE_PROCESSING = "processing"
STATE_DONE = "done"
STATE_ERROR = "error"

_TIMESTAMP = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

_WATCHER_COLUMNS = [
    "id",
    "path",
    "sha",
    "state",
    "size",
    "mtime",
    "enqueued_at",
    "processed_at",
]

# Sentinel so a missing file's ``None`` snapshot is never mistaken for a repeat.
_MISSING = object()


def _snapshot(path: Path) -> tuple[int, int] | None:
    """Return ``(size, mtime_ns)`` for *path*, or ``None`` when it cannot be stat'ed."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (st.st_size, st.st_mtime_ns)


def stabilize(
    path: Path | str,
    settle_calls: int = 2,
    settle_interval: float = 0.0,
    timeout: float = 1.0,
    decode_confirmation: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Return True once *path* has been quiescent for ``settle_calls`` polls.

    A file is considered stable when ``settle_calls`` consecutive reads yield an
    identical ``(size, mtime_ns)`` snapshot, spaced ``settle_interval`` seconds
    apart. When *decode_confirmation* is set (default), the file must also decode
    through the M001 path (:func:`~curator.ingest.decode.decode_image`); a
    corrupt/undecodable file is treated as not-yet-stable. Returns False if the
    file never stabilizes within ``timeout`` seconds.

    *sleep* is injectable so tests can avoid real sleeps while keeping behavior
    identical.
    """
    deadline = time.monotonic() + timeout
    last: Any = _MISSING
    runs = 0
    while True:
        token = _snapshot(Path(path))
        if token is None:
            last = _MISSING
            runs = 0
        elif token == last:
            runs += 1
            if runs >= settle_calls:
                if decode_confirmation:
                    try:
                        decode_image(Path(path).read_bytes())
                    except Exception:
                        return False
                return True
        else:
            last = token
            runs = 1
        if time.monotonic() >= deadline:
            return False
        sleep(settle_interval)


class Watcher:
    """Scans a source *root* and durably enqueues stable image paths.

    Owns a :class:`~curator.catalog.Catalog` (the system of record); the catalog's
    ``db`` connection backs ``watcher_queue`` and its content lookups back
    reconciliation dedup by SHA-256.
    """

    def __init__(
        self,
        catalog: Catalog,
        root: Path | str,
        *,
        settle_calls: int = 2,
        settle_interval: float = 0.0,
        timeout: float = 1.0,
        decode_confirmation: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.catalog = catalog
        self.db = catalog.db
        self.root = Path(root).resolve()
        self.settle_calls = settle_calls
        self.settle_interval = settle_interval
        self.timeout = timeout
        self.decode_confirmation = decode_confirmation
        self._sleep = sleep
        # Paths enqueued during this session (coalesces ongoing bursts).
        self._enqueued: set[str] = set()
        # Reclaim in-flight rows left 'processing' by a previous crashed run so
        # nothing is lost between enqueue and done on restart.
        self._reclaim_incomplete()

    # -- polling -----------------------------------------------------------

    def poll_once(self) -> int:
        """Scan the source once; enqueue each stable, newly-seen image path.

        Repeated sightings of a path already enqueued this session are coalesced
        into a single enqueue. Returns the number of paths enqueued this poll.
        """
        count = 0
        for path in self._enumerate():
            key = str(path.resolve())
            if key in self._enqueued:
                continue  # burst already coalesced this session
            if not self._stabilize(key):
                continue
            try:
                data = Path(key).read_bytes()
            except OSError:
                continue
            sha = sha256_hex(data)
            if self.enqueue(key, sha):
                self._enqueued.add(key)
                count += 1
        return count

    def _stabilize(self, path: str) -> bool:
        return stabilize(
            path,
            settle_calls=self.settle_calls,
            settle_interval=self.settle_interval,
            timeout=self.timeout,
            decode_confirmation=self.decode_confirmation,
            sleep=self._sleep,
        )

    # -- enqueue / drain / status -------------------------------------------

    def enqueue(self, path: str, sha: str | None) -> bool:
        """Idempotently append a ``watcher_queue`` row for *path*.

        Skips (returns False) when *path* already has a live row
        (``queued``/``processing``/``done``) or when any ``done`` row already
        carries the same *sha*, so the same source or content is never enqueued
        twice. Returns True when a new ``queued`` row was inserted.
        """
        live = self.db.execute(
            "SELECT id FROM watcher_queue WHERE path = ? AND state != 'error' LIMIT 1",
            (path,),
        ).fetchone()
        if live is not None:
            return False
        if sha:
            done = self.db.execute(
                "SELECT id FROM watcher_queue WHERE sha = ? AND state = 'done' LIMIT 1",
                (sha,),
            ).fetchone()
            if done is not None:
                return False
        size, mtime = _stat(Path(path))
        try:
            cur = self.db.execute(
                "INSERT INTO watcher_queue(path, sha, state, size, mtime)"
                " VALUES (?, ?, ?, ?, ?)",
                (path, sha, STATE_QUEUED, size, mtime),
            )
            self.db.commit()
        except sqlite3.Error:
            self.db.rollback()
            raise
        return cur.lastrowid is not None

    def drain(self):
        """Yield each unprocessed (``queued``) row, marking it ``processing``.

        ``processing`` rows left behind by a previous crashed run are reclaimed
        to ``queued`` first, so a crash between enqueue and done is never lost.
        Each yielded row is a dict: ``id``, ``path``, ``sha``, ``size``, ``mtime``.
        """
        self._reclaim_incomplete()
        rows = self.db.execute(
            "SELECT id, path, sha, size, mtime FROM watcher_queue"
            " WHERE state = 'queued' ORDER BY id"
        ).fetchall()
        for row in rows:
            self.db.execute(
                "UPDATE watcher_queue SET state = 'processing' WHERE id = ?", (row[0],)
            )
            self.db.commit()
            yield {
                "id": row[0],
                "path": row[1],
                "sha": row[2],
                "size": row[3],
                "mtime": row[4],
            }

    def mark_done(self, path: str) -> None:
        """Transition the row(s) for *path* to ``done`` and stamp ``processed_at``."""
        self.db.execute(
            "UPDATE watcher_queue SET state = 'done',"
            f" processed_at = {_TIMESTAMP} WHERE path = ?",
            (path,),
        )
        self.db.commit()

    def mark_error(self, path: str, err: Exception | str) -> None:
        """Transition the row(s) for *path* to ``error`` (retryable later)."""
        self.db.execute(
            "UPDATE watcher_queue SET state = 'error',"
            f" processed_at = {_TIMESTAMP} WHERE path = ?",
            (path,),
        )
        self.db.commit()

    def queue_rows(self) -> list[dict[str, Any]]:
        """Return every ``watcher_queue`` row, oldest first (for inspection/tests)."""
        cur = self.db.execute("SELECT * FROM watcher_queue ORDER BY id")
        return [dict(zip(_WATCHER_COLUMNS, row)) for row in cur.fetchall()]

    def _reclaim_incomplete(self) -> None:
        """Reclaim ``processing`` rows (left by a crash) back to ``queued``."""
        self.db.execute(
            "UPDATE watcher_queue SET state = 'queued' WHERE state = 'processing'"
        )
        self.db.commit()

    # -- reconciliation -----------------------------------------------------

    def reconcile_once(self) -> int:
        """Rescan the source; enqueue any file whose SHA-256 is not yet handled.

        A file is skipped when its SHA-256 is already in the catalog (dedup by
        content hash) or already present in ``watcher_queue`` in any state, so
        each completed source is processed exactly once. Returns the number of
        paths enqueued.
        """
        count = 0
        for path in self._enumerate():
            key = str(path.resolve())
            try:
                data = Path(key).read_bytes()
            except OSError:
                continue
            sha = sha256_hex(data)
            if self.catalog.get_by_hash(sha):
                continue  # already cataloged
            if self._has_sha(sha):
                continue  # already queued/processing/done/error
            if self.enqueue(key, sha):
                count += 1
        return count

    def _has_sha(self, sha: str) -> bool:
        row = self.db.execute(
            "SELECT id FROM watcher_queue WHERE sha = ? LIMIT 1", (sha,)
        ).fetchone()
        return row is not None

    # -- helpers -------------------------------------------------------------

    def _enumerate(self):
        """Yield supported image files under the root, sorted deterministically."""
        if not self.root.is_dir():
            return
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.suffix.lower() in _SUPPORTED:
                yield path


def _stat(path: Path) -> tuple[int | None, int | None]:
    try:
        st = path.stat()
    except OSError:
        return None, None
    return st.st_size, st.st_mtime_ns


@dataclass
class WatcherRunReport:
    """JSON-serializable result of one :meth:`WatcherService.run_once` step."""

    enqueued: int = 0
    drained: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize this report as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WatcherRunReport:
        """Reconstruct a report from a previously serialized dict."""
        return cls(
            enqueued=int(data.get("enqueued", 0)),
            drained=list(data.get("drained", [])),
        )


class WatcherService:
    """Synchronous orchestration tying poll + drain together into one step."""

    def __init__(self, watcher: Watcher) -> None:
        self.watcher = watcher

    def poll(self) -> int:
        """Scan once and enqueue any newly stable paths; returns enqueued count."""
        return self.watcher.poll_once()

    def reconcile(self) -> int:
        """Reconcile the source against the catalog; returns enqueued count."""
        return self.watcher.reconcile_once()

    def run_once(self) -> WatcherRunReport:
        """Poll for new paths, then drain the queue into ``processing``.

        Returns a :class:`WatcherRunReport` capturing what was enqueued and the
        drained (now ``processing``) rows — a deterministic, testable unit of
        work with no long-lived thread.
        """
        enqueued = self.watcher.poll_once()
        drained = list(self.watcher.drain())
        return WatcherRunReport(enqueued=enqueued, drained=drained)
