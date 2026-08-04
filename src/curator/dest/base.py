"""Destination adapters — the abstract write surface for M005/S01.

A :class:`DestinationAdapter` exposes a uniform contract for publishing rendered
artifacts somewhere (a filesystem root, a simulator, later a Samsung Frame
device). Adapters are **exact-ID** stores: an ``artifact_id`` maps deterministically
to one unit (a file path, a ledger key), so ``put`` writes at that ID and
``replace`` overwrites it.

The contract is the narrow seam the :mod:`curator.dest.publish` state machine
drives: stage -> verify -> apply -> journal, with a failing write leaving the
prior artifact intact.
"""

from __future__ import annotations

import abc
from typing import Any

from curator.errors import CuratorError

#: Journal ``op`` vocabulary for ``dest_journal.op``.
OP_PUT = "put"
OP_REPLACE = "replace"
OP_REMOVE = "remove"

#: Journal ``status`` vocabulary for ``dest_journal.status`` (per-artifact state
#: machine: ``staged -> verified -> applied | error``).
STATUS_STAGED = "staged"
STATUS_VERIFIED = "verified"
STATUS_APPLIED = "applied"
STATUS_ERROR = "error"


class DestinationError(CuratorError):
    """Raised when a destination write / replace / remove / read fails."""


class DestinationAdapter(abc.ABC):
    """Abstract write destination for curated artifacts.

    Adapters are held to a transactional contract: a failing ``put`` / ``replace``
    must leave the previously-published artifact (if any) byte-identical and in
    place, so the publish coordinator's rollback is a no-op (keep last-known-good).
    """

    @abc.abstractmethod
    def capabilities(self) -> dict[str, bool]:
        """Return feature flags: ``supports_put`` / ``supports_replace`` /
        ``supports_remove`` / ``exact_ids``."""

    def probe(self) -> dict[str, Any]:
        """Return a liveness/health dict, e.g. ``{"ok": True}``."""
        return {"ok": True}

    @abc.abstractmethod
    def put(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        """Write *data* to *artifact_id* and return its SHA-256 hex digest.

        Raises :class:`DestinationError` before mutating state on failure.
        """

    def replace(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        """Overwrite *artifact_id* with *data* and return its SHA-256 hex digest.

        Defaults to :meth:`put` (exact-ID overwrite); an adapter may override to
        require replace capabilities.
        """
        return self.put(artifact_id, data, meta)

    def remove(self, artifact_id: str) -> None:
        """Delete *artifact_id* if present; a missing artifact is a no-op."""
        return None

    def get_state(self, artifact_id: str) -> dict[str, Any] | None:
        """Return ``{"sha256", "size"}`` for *artifact_id*, or ``None`` if absent."""
        return None

    def observe(self) -> list[str]:
        """Return the current artifact ids present on the destination."""
        return []
