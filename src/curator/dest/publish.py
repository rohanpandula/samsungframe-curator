"""Publish coordinator — transactional artifact publishing (M005/S01).

:class:`PublishCoordinator` drives one artifact through an idempotent state
machine, journaling each transition into the append-only ``dest_journal`` table
(schema v8):

    staged -> verified -> applied        (success)
    staged -> verified -> error          (adapter write failed)

**Transactional safety:** if the adapter's ``put``/``replace`` raises
:class:`DestinationError`, the coordinator records an ``error`` row and returns —
it performs **no rollback to clear the prior artifact**, because the adapter
contract guarantees a failing write leaves the prior artifact intact (keep
last-known-good).

**Idempotent resume:** re-publishing an artifact whose latest journal row is
``error`` reuses that row and transitions it ``error -> applied`` on the same id,
so a retry never duplicates the error row. Re-publishing an artifact whose latest
row is ``applied`` with the same sha is a no-op (skip).

Every row mirrors the ingest/consolidation journal posture: one row per attempt,
advanced in place via UPDATE.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from curator.dest.base import (
    OP_PUT,
    STATUS_APPLIED,
    STATUS_ERROR,
    STATUS_STAGED,
    STATUS_VERIFIED,
    DestinationAdapter,
    DestinationError,
)
from curator.hashing import sha256_hex

#: Journal columns selected back for resume / idempotency checks.
_JOURNAL_COLUMNS = "id, adapter_id, artifact_id, op, sha, status, error, created_at"


@dataclass(frozen=True)
class DestJournalEntry:
    """One row of the append-only ``dest_journal`` (JSON round-trippable)."""

    id: int | None = None
    adapter_id: str = ""
    artifact_id: str = ""
    op: str = ""
    sha: str | None = None
    status: str = ""
    error: str | None = None
    created_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return this entry serialized as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DestJournalEntry:
        """Build an entry from a dict, ignoring unknown keys."""
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


@dataclass(frozen=True)
class PublishResult:
    """JSON-serializable outcome of one :meth:`PublishCoordinator.publish`."""

    artifact_id: str
    adapter_id: str
    op: str
    status: str
    sha: str
    skipped: bool = False
    row_id: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return this result serialized as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PublishResult:
        """Build a result from a dict, ignoring unknown keys."""
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)


class PublishCoordinator:
    """Journals and executes the per-artifact publish state machine."""

    def __init__(
        self,
        adapter: DestinationAdapter,
        db: sqlite3.Connection,
        adapter_id: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.db = db
        self.adapter_id = adapter_id if adapter_id is not None else type(adapter).__name__

    # -- journal helpers -----------------------------------------------------

    def _insert(
        self,
        artifact_id: str,
        op: str,
        sha: str | None,
        status: str,
        error: str | None = None,
    ) -> int:
        cur = self.db.execute(
            "INSERT INTO dest_journal(adapter_id, artifact_id, op, sha, status, error)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (self.adapter_id, artifact_id, op, sha, status, error),
        )
        self.db.commit()
        row_id = cur.lastrowid
        if row_id is None:
            raise DestinationError("failed to obtain dest_journal row id")
        return int(row_id)

    def _update(
        self,
        row_id: int,
        status: str,
        sha: str | None = None,
        error: str | None = None,
    ) -> None:
        self.db.execute(
            "UPDATE dest_journal SET status = ?,"
            " sha = COALESCE(?, sha), error = ? WHERE id = ?",
            (status, sha, error, row_id),
        )
        self.db.commit()

    def _last_row(self, artifact_id: str) -> dict[str, Any] | None:
        """The newest journal row for *artifact_id* **on this adapter**.

        Scoped by ``adapter_id`` (M011/S01): an artifact hangs on a destination,
        so "already applied on the simulator" must never make the same artifact
        id skip on a folder — before this, the first wall published anywhere made
        every other destination report ``skipped`` for the same file names.
        """
        row = self.db.execute(
            f"SELECT {_JOURNAL_COLUMNS} FROM dest_journal"
            " WHERE artifact_id = ? AND adapter_id = ? ORDER BY id DESC LIMIT 1",
            (artifact_id, self.adapter_id),
        ).fetchone()
        if row is None:
            return None
        return dict(zip(_JOURNAL_COLUMNS.split(", "), row))

    # -- queries -------------------------------------------------------------

    def already_applied(self, artifact_id: str, sha: str) -> bool:
        """True when the latest journal row is ``applied`` with *sha*."""
        row = self._last_row(artifact_id)
        return bool(row and row["status"] == STATUS_APPLIED and row["sha"] == sha)

    def history(self, artifact_id: str) -> list[DestJournalEntry]:
        """Return all journal rows for *artifact_id*, oldest first."""
        rows = self.db.execute(
            f"SELECT {_JOURNAL_COLUMNS} FROM dest_journal"
            " WHERE artifact_id = ? ORDER BY id",
            (artifact_id,),
        ).fetchall()
        return [
            DestJournalEntry(**dict(zip(_JOURNAL_COLUMNS.split(", "), row)))
            for row in rows
        ]

    def rows_with_status(self, status: str) -> list[DestJournalEntry]:
        """Return journal rows matching *status* (across all artifacts)."""
        rows = self.db.execute(
            f"SELECT {_JOURNAL_COLUMNS} FROM dest_journal WHERE status = ? ORDER BY id",
            (status,),
        ).fetchall()
        return [
            DestJournalEntry(**dict(zip(_JOURNAL_COLUMNS.split(", "), row)))
            for row in rows
        ]

    # -- publish -------------------------------------------------------------

    def publish(
        self,
        artifact_id: str,
        data: bytes,
        meta: dict[str, Any] | None = None,
        op: str = OP_PUT,
    ) -> PublishResult:
        """Publish *data* to *artifact_id*, journaled and idempotently.

        If the artifact is already ``applied`` with the same sha, returns a
        ``skipped`` result with no new journal row. Otherwise stage -> verify ->
        apply; a failing adapter write records ``error`` (prior artifact intact).
        """
        sha = sha256_hex(data)
        latest = self._last_row(artifact_id)
        if latest and latest["status"] == STATUS_APPLIED and latest["sha"] == sha:
            return PublishResult(
                artifact_id=artifact_id,
                adapter_id=self.adapter_id,
                op=op,
                status=STATUS_APPLIED,
                sha=sha,
                skipped=True,
                row_id=latest["id"],
            )

        # Resume seam: reuse an unfinished/errored row instead of duplicating it.
        if latest and latest["status"] in (STATUS_STAGED, STATUS_VERIFIED, STATUS_ERROR):
            row_id = latest["id"]
        else:
            row_id = self._insert(artifact_id, op, sha, STATUS_STAGED)
        self._update(row_id, STATUS_VERIFIED, sha=sha)

        try:
            actual = (
                self.adapter.replace(artifact_id, data, meta)
                if op == "replace"
                else self.adapter.put(artifact_id, data, meta)
            )
        except DestinationError as exc:
            self._update(row_id, STATUS_ERROR, error=str(exc))
            return PublishResult(
                artifact_id=artifact_id,
                adapter_id=self.adapter_id,
                op=op,
                status=STATUS_ERROR,
                sha=sha,
                row_id=row_id,
                error=str(exc),
            )

        if actual != sha:
            message = f"published sha {actual!r} != expected {sha!r} for {artifact_id}"
            self._update(row_id, STATUS_ERROR, error=message)
            return PublishResult(
                artifact_id=artifact_id,
                adapter_id=self.adapter_id,
                op=op,
                status=STATUS_ERROR,
                sha=sha,
                row_id=row_id,
                error=message,
            )

        self._update(row_id, STATUS_APPLIED, sha=actual)
        return PublishResult(
            artifact_id=artifact_id,
            adapter_id=self.adapter_id,
            op=op,
            status=STATUS_APPLIED,
            sha=actual,
            row_id=row_id,
        )


def publish(
    adapter: DestinationAdapter,
    db: sqlite3.Connection,
    artifact_id: str,
    data: bytes,
    meta: dict[str, Any] | None = None,
    op: str = OP_PUT,
    adapter_id: str | None = None,
) -> PublishResult:
    """One-shot publish convenience wrapper around :class:`PublishCoordinator`."""
    coordinator = PublishCoordinator(adapter, db, adapter_id=adapter_id)
    return coordinator.publish(artifact_id, data, meta=meta, op=op)
