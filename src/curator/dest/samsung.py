"""Samsung Art Mode destination adapter (M005/S02).

Publishing to a Samsung Frame in Art Mode is coordinated with Home Assistant:
:class:`SamsungArtModeDestinationAdapter` acquires an exclusive lease, probes
the transport, uploads a small **canary** probe, then lands the artifact via an
**exact-ID replace** (Samsung devices have no separate create path). Any
:class:`DestinationError` restores the pre-mutation Home Assistant automation
state, journals an ``error`` row to ``dest_journal`` (schema v8), and re-raises
— the transactional posture from M005/S01 (a failing write leaves the prior
artifact intact).
"""

from __future__ import annotations

import abc
import sqlite3
from typing import Any

from curator.dest.base import (
    OP_REPLACE,
    STATUS_APPLIED,
    STATUS_ERROR,
    STATUS_STAGED,
    STATUS_VERIFIED,
    DestinationAdapter,
    DestinationError,
)
from curator.dest.simulator import SimulatorDestinationAdapter
from curator.ha import HomeAssistantCoordinationAdapter, SimulatorLeaseManager
from curator.hashing import sha256_hex

#: Journal ``op`` vocabulary for the canary probe phase.
OP_CANARY = "canary"

#: Small probe payload written to the transport before the exact-ID replace.
_CANARY_BYTES = b"frame-canary-probe"


class SamsungTransport(abc.ABC):
    """Abstract write surface for a Samsung Frame device."""

    @abc.abstractmethod
    def probe(self) -> dict[str, Any]:
        """Return a liveness/health dict, e.g. ``{"ok": True}``."""

    @abc.abstractmethod
    def canary_upload(self, probe_id: str, data: bytes) -> str:
        """Write a small probe to *probe_id*; return its SHA-256 hex digest."""

    @abc.abstractmethod
    def put(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        """Write *data* at *artifact_id* and return its SHA-256 hex digest."""

    @abc.abstractmethod
    def replace(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        """Overwrite *artifact_id* with *data* and return its SHA-256 hex digest."""

    @abc.abstractmethod
    def remove(self, artifact_id: str) -> None:
        """Delete *artifact_id* if present; a missing artifact is a no-op."""

    @abc.abstractmethod
    def get_state(self, artifact_id: str) -> dict[str, Any] | None:
        """Return ``{"sha256", "size"}`` for *artifact_id*, or ``None`` if absent."""

    @abc.abstractmethod
    def observe(self) -> list[str]:
        """Return the artifact ids currently present on the destination."""


class SimulatorSamsungTransport(SamsungTransport):
    """Samsung transport backed by the S01 in-memory simulator.

    The canonical Art Mode write flow is **canary upload first, exact-ID replace
    second**: the small canary probe must succeed before the real artifact
    lands. :meth:`fail_next_write` arms a single failure for the next exact-ID
    write (after the canary succeeds) so rollback is exercised deterministically.
    """

    def __init__(self, simulator: SimulatorDestinationAdapter | None = None) -> None:
        self._sim = simulator if simulator is not None else SimulatorDestinationAdapter()
        self._fail_next = False

    def fail_next_write(self) -> None:
        """Arm a single-write failure for the NEXT exact-ID put/replace."""
        self._fail_next = True

    # -- transport surface ---------------------------------------------------

    def probe(self) -> dict[str, Any]:
        return self._sim.probe()

    def _canary_key(self, probe_id: str) -> str:
        return f"canary:{probe_id}"

    def canary_upload(self, probe_id: str, data: bytes) -> str:
        return self._sim.put(self._canary_key(probe_id), data)

    def _exact_id_write(
        self,
        artifact_id: str,
        data: bytes,
        meta: dict[str, Any] | None,
    ) -> str:
        if self._fail_next:
            self._fail_next = False
            raise DestinationError(f"simulated Samsung write failure for {artifact_id}")
        return self._sim.replace(artifact_id, data, meta)

    def put(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        return self._exact_id_write(artifact_id, data, meta)

    def replace(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        return self._exact_id_write(artifact_id, data, meta)

    def remove(self, artifact_id: str) -> None:
        self._sim.remove(artifact_id)

    def get_state(self, artifact_id: str) -> dict[str, Any] | None:
        return self._sim.get_state(artifact_id)

    def observe(self) -> list[str]:
        return self._sim.observe()


class SamsungArtModeDestinationAdapter(DestinationAdapter):
    """Art Mode destination: lease -> probe -> canary -> exact-ID replace.

    Holds a :class:`SamsungTransport` and a
    :class:`~curator.ha.HomeAssistantCoordinationAdapter`. Each ``put`` /
    ``replace`` acquires the HA lease, probes the transport, uploads a canary
    probe, then performs the exact-ID replace — journaling each phase into
    ``dest_journal`` (schema v8; ``op`` in ``canary|replace``). Any
    :class:`DestinationError` restores the prior HA automation state, journals
    an ``error`` row, and re-raises.
    """

    def __init__(
        self,
        transport: SamsungTransport,
        ha_adapter: HomeAssistantCoordinationAdapter | None = None,
        db: sqlite3.Connection | None = None,
        adapter_id: str | None = None,
    ) -> None:
        self._transport = transport
        if ha_adapter is None:
            ha_adapter = HomeAssistantCoordinationAdapter(SimulatorLeaseManager())
        self._ha = ha_adapter
        self._db = db
        self._adapter_id = adapter_id if adapter_id is not None else type(self).__name__

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_put": True,
            "supports_replace": True,
            "supports_remove": True,
            "exact_ids": True,
        }

    def probe(self) -> dict[str, Any]:
        return self._transport.probe()

    def put(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        return self._publish(artifact_id, data, meta)

    def replace(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        return self._publish(artifact_id, data, meta)

    def remove(self, artifact_id: str) -> None:
        self._transport.remove(artifact_id)

    def get_state(self, artifact_id: str) -> dict[str, Any] | None:
        return self._transport.get_state(artifact_id)

    def observe(self) -> list[str]:
        return self._transport.observe()

    # -- journal helpers (PublishCoordinator-style, dest_journal v8) ----------

    def _journal_insert(
        self,
        artifact_id: str,
        op: str,
        sha: str | None,
        status: str,
        error: str | None = None,
    ) -> int | None:
        if self._db is None:
            return None
        cur = self._db.execute(
            "INSERT INTO dest_journal(adapter_id, artifact_id, op, sha, status, error)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (self._adapter_id, artifact_id, op, sha, status, error),
        )
        self._db.commit()
        return cur.lastrowid

    def _journal_update(
        self,
        row_id: int | None,
        status: str,
        sha: str | None = None,
        error: str | None = None,
    ) -> None:
        if self._db is None or row_id is None:
            return
        self._db.execute(
            "UPDATE dest_journal SET status = ?,"
            " sha = COALESCE(?, sha), error = ? WHERE id = ?",
            (status, sha, error, row_id),
        )
        self._db.commit()

    def _probe_id(self, artifact_id: str) -> str:
        return f"probe:{artifact_id}"

    def _publish(
        self,
        artifact_id: str,
        data: bytes,
        meta: dict[str, Any] | None,
    ) -> str:
        sha = sha256_hex(data)
        self._ha.acquire_lease()
        canary_row: int | None = None
        replace_row: int | None = None
        try:
            probe = self._transport.probe()
            if not probe.get("ok", False):
                raise DestinationError(f"transport probe failed: {probe}")

            canary_row = self._journal_insert(artifact_id, OP_CANARY, sha, STATUS_STAGED)
            self._journal_update(canary_row, STATUS_VERIFIED)
            try:
                canary_sha = self._transport.canary_upload(
                    self._probe_id(artifact_id), _CANARY_BYTES
                )
            except DestinationError as exc:
                self._journal_update(canary_row, STATUS_ERROR, error=str(exc))
                raise
            self._journal_update(canary_row, STATUS_APPLIED, sha=canary_sha)

            replace_row = self._journal_insert(artifact_id, OP_REPLACE, sha, STATUS_STAGED)
            self._journal_update(replace_row, STATUS_VERIFIED)
            try:
                actual = self._transport.replace(artifact_id, data, meta)
            except DestinationError as exc:
                self._journal_update(replace_row, STATUS_ERROR, error=str(exc))
                raise
            if actual != sha:
                message = f"replace sha {actual!r} != expected {sha!r} for {artifact_id}"
                self._journal_update(replace_row, STATUS_ERROR, error=message)
                raise DestinationError(message)
            self._journal_update(replace_row, STATUS_APPLIED, sha=actual)
            return actual
        except DestinationError:
            self._ha.restore_prior_state()
            raise
        finally:
            self._ha.release_lease()
