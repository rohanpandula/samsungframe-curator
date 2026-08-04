"""Approval & history subsystem (M003/S03).

Exposes the append-only :class:`~curator.approve.approval.ApprovalService` that
persists per-catalog-entry approve/reject/undo/redo decisions (R010) and the
JSON-serializable :class:`~curator.approve.approval.ApprovalEvent`.
"""

from curator.approve.approval import (
    ApprovalError,
    ApprovalEvent,
    ApprovalService,
    Decision,
)

__all__ = [
    "ApprovalError",
    "ApprovalEvent",
    "ApprovalService",
    "Decision",
]
