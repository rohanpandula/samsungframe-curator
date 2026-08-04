"""Approval & history service (M003/S03, R010).

Explicit per-catalog-entry decisions (approve/reject) are persisted to the
append-only ``approvals`` table (schema v6). Every transition — including undo
and redo — is a new row; history is never erased or rewritten. "Current" for an
entry is defined as the latest resolved decision (the newest row), so undo/redo
flip the active decision by appending a transition rather than deleting one.

:class:`ApprovalEvent` is the JSON-serializable record of a single decision
(entry id, decision, rationale, creation time, and a per-entry sequence number).
"""

from __future__ import annotations

import enum
import sqlite3
from dataclasses import dataclass
from typing import Any

from curator.catalog import Catalog
from curator.errors import ApprovalError


class Decision(enum.StrEnum):
    """The two possible per-entry decisions."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


def _flip(decision: Decision) -> Decision:
    """Return the opposite decision (undo/redo toggles the active state)."""
    if decision is Decision.APPROVED:
        return Decision.REJECTED
    return Decision.APPROVED


@dataclass(frozen=True)
class ApprovalEvent:
    """One immutable, persisted decision transition for a catalog entry."""

    catalog_entry_id: int
    decision: Decision
    rationale: str = ""
    created_at: str = ""
    seq: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog_entry_id": self.catalog_entry_id,
            "decision": self.decision.value,
            "rationale": self.rationale,
            "created_at": self.created_at,
            "seq": self.seq,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ApprovalEvent:
        return cls(
            catalog_entry_id=data["catalog_entry_id"],
            decision=Decision(data["decision"]),
            rationale=data.get("rationale", ""),
            created_at=data.get("created_at", ""),
            seq=data.get("seq", 0),
        )


class ApprovalService:
    """Persist and query per-entry approval decisions (append-only).

    Accepts either a :class:`~curator.catalog.Catalog` or a raw SQLite
    connection; when a Catalog is passed its shared ``.db`` is used. The schema
    is expected to already be migrated (a Catalog migrates it on construction).
    """

    def __init__(self, db: sqlite3.Connection | Catalog) -> None:
        self.db = db.db if isinstance(db, Catalog) else db

    # -- public API ---------------------------------------------------------

    def approve(self, catalog_entry_id: int, rationale: str = "") -> ApprovalEvent:
        """Record an APPROVED decision for the entry (append-only)."""
        return self._append(catalog_entry_id, Decision.APPROVED, rationale)

    def reject(self, catalog_entry_id: int, rationale: str = "") -> ApprovalEvent:
        """Record a REJECTED decision for the entry (append-only)."""
        return self._append(catalog_entry_id, Decision.REJECTED, rationale)

    def undo(self, catalog_entry_id: int) -> ApprovalEvent:
        """Append a transition reverting the latest resolved decision.

        The active decision flips to its opposite; the prior history is kept.
        Raises :class:`ApprovalError` when there is no decision to undo.
        """
        current = self.current(catalog_entry_id)
        if current is None:
            raise ApprovalError(
                f"cannot undo: no decision for catalog entry {catalog_entry_id}"
            )
        return self._append(catalog_entry_id, _flip(current.decision))

    def redo(self, catalog_entry_id: int) -> ApprovalEvent:
        """Append a transition re-applying the previously active decision.

        Uses the decision immediately preceding the latest one (i.e. the state
        that was undone). Raises :class:`ApprovalError` when nothing can be
        redone (no history or only a single decision exists).
        """
        events = self.history(catalog_entry_id)
        if len(events) < 2:
            raise ApprovalError(
                f"cannot redo: no prior decision to re-apply for"
                f" catalog entry {catalog_entry_id}"
            )
        return self._append(catalog_entry_id, events[-2].decision)

    def batch_approve(
        self, catalog_entry_ids: list[int], rationale: str = ""
    ) -> list[ApprovalEvent]:
        """Approve every entry id in order, returning one event per entry."""
        return [self.approve(eid, rationale) for eid in catalog_entry_ids]

    def history(self, catalog_entry_id: int) -> list[ApprovalEvent]:
        """Return all decisions for the entry, oldest first (append-only)."""
        rows = self.db.execute(
            "SELECT catalog_entry_id, decision, rationale, created_at"
            " FROM approvals WHERE catalog_entry_id = ? ORDER BY id",
            (catalog_entry_id,),
        ).fetchall()
        return [
            ApprovalEvent(
                catalog_entry_id=row[0],
                decision=Decision(row[1]),
                rationale=row[2] or "",
                created_at=row[3],
                seq=seq,
            )
            for seq, row in enumerate(rows, start=1)
        ]

    def current(self, catalog_entry_id: int) -> ApprovalEvent | None:
        """Return the latest resolved decision, or ``None`` if none exists."""
        events = self.history(catalog_entry_id)
        return events[-1] if events else None

    # -- impl ---------------------------------------------------------------

    def _append(
        self, catalog_entry_id: int, decision: Decision, rationale: str = ""
    ) -> ApprovalEvent:
        self._validate_entry(catalog_entry_id)
        cur = self.db.execute(
            "INSERT INTO approvals(catalog_entry_id, decision, rationale)"
            " VALUES (?, ?, ?)",
            (catalog_entry_id, decision.value, rationale),
        )
        self.db.commit()
        created = self.db.execute(
            "SELECT created_at FROM approvals WHERE id = ?", (cur.lastrowid,)
        ).fetchone()[0]
        count = self.db.execute(
            "SELECT COUNT(*) FROM approvals WHERE catalog_entry_id = ?",
            (catalog_entry_id,),
        ).fetchone()[0]
        return ApprovalEvent(
            catalog_entry_id=catalog_entry_id,
            decision=decision,
            rationale=rationale,
            created_at=created,
            seq=int(count),
        )

    def _validate_entry(self, catalog_entry_id: int) -> None:
        row = self.db.execute(
            "SELECT 1 FROM catalog_entries WHERE id = ?", (catalog_entry_id,)
        ).fetchone()
        if row is None:
            raise ApprovalError(
                f"no catalog entry with id {catalog_entry_id}"
            )
