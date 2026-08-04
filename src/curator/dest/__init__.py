"""Destination publication subsystem (M005/S01) — adapters + transactional publish.

Provides the abstract :class:`~curator.dest.base.DestinationAdapter` write
surface, a filesystem backend
(:class:`~curator.dest.filesystem.FilesystemDestinationAdapter`), an in-memory
fault-injectable simulator
(:class:`~curator.dest.simulator.SimulatorDestinationAdapter`), and the
idempotent, rollback-safe publish coordinator
(:class:`~curator.dest.publish.PublishCoordinator`) backed by the append-only
``dest_journal`` table (schema v8).
"""

from __future__ import annotations

from curator.dest.base import (
    OP_PUT,
    OP_REMOVE,
    OP_REPLACE,
    STATUS_APPLIED,
    STATUS_ERROR,
    STATUS_STAGED,
    STATUS_VERIFIED,
    DestinationAdapter,
    DestinationError,
)
from curator.dest.filesystem import FilesystemDestinationAdapter
from curator.dest.publish import (
    DestJournalEntry,
    PublishCoordinator,
    PublishResult,
    publish,
)
from curator.dest.simulator import SimulatorDestinationAdapter

__all__ = [
    "OP_PUT",
    "OP_REMOVE",
    "OP_REPLACE",
    "STATUS_STAGED",
    "STATUS_VERIFIED",
    "STATUS_APPLIED",
    "STATUS_ERROR",
    "DestinationAdapter",
    "DestinationError",
    "FilesystemDestinationAdapter",
    "SimulatorDestinationAdapter",
    "DestJournalEntry",
    "PublishCoordinator",
    "PublishResult",
    "publish",
]
