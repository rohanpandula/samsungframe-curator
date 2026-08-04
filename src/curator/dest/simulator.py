"""In-memory simulator destination adapter (M005/S01).

Keeps a JSON-friendly ledger ``{artifact_id: {"bytes", "sha", "meta"}}`` and
records every operation. :meth:`fail_next_write` arms a flag: the **next** ``put``
/ ``replace`` raises :class:`DestinationError` *before* touching the ledger,
simulating a mid-write failure that leaves the prior artifact intact — the
failure model the publish coordinator's rollback (do-nothing / keep
last-known-good) relies on.
"""

from __future__ import annotations

from typing import Any

from curator.dest.base import OP_PUT, OP_REPLACE, DestinationAdapter, DestinationError
from curator.hashing import sha256_hex


class SimulatorDestinationAdapter(DestinationAdapter):
    """Fault-injectable, in-memory destination for tests and dry-runs."""

    def __init__(self) -> None:
        self._ledger: dict[str, dict[str, Any]] = {}
        self._ops: list[dict[str, Any]] = []
        self._fail_next: bool = False

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_put": True,
            "supports_replace": True,
            "supports_remove": True,
            "exact_ids": True,
        }

    def probe(self) -> dict[str, Any]:
        return {"ok": True}

    # -- fault injection -----------------------------------------------------

    def fail_next_write(self) -> None:
        """Arm a single-write failure for the NEXT put/replace."""
        self._fail_next = True

    # -- adapter surface -----------------------------------------------------

    def put(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        return self._write(OP_PUT, artifact_id, data, meta)

    def replace(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        return self._write(OP_REPLACE, artifact_id, data, meta)

    def _write(
        self,
        op: str,
        artifact_id: str,
        data: bytes,
        meta: dict[str, Any] | None,
    ) -> str:
        sha = sha256_hex(data)
        self._ops.append({"op": op, "artifact_id": artifact_id, "sha": sha})
        if self._fail_next:
            self._fail_next = False
            raise DestinationError(
                f"simulated write failure for {artifact_id} ({op})"
            )
        self._ledger[artifact_id] = {"bytes": data, "sha": sha, "meta": meta}
        return sha

    def remove(self, artifact_id: str) -> None:
        self._ledger.pop(artifact_id, None)

    def get_state(self, artifact_id: str) -> dict[str, Any] | None:
        entry = self._ledger.get(artifact_id)
        if entry is None:
            return None
        return {"sha256": entry["sha"], "size": len(entry["bytes"])}

    def observe(self) -> list[str]:
        return sorted(self._ledger.keys())

    # -- inspection / reset --------------------------------------------------

    def ledger(self) -> dict[str, dict[str, Any]]:
        """Return a copy of the internal ledger."""
        return dict(self._ledger)

    def ops(self) -> list[dict[str, Any]]:
        """Return the recorded operation log."""
        return list(self._ops)

    def reset(self) -> None:
        """Clear the ledger, op log, and any armed failure flag."""
        self._ledger.clear()
        self._ops.clear()
        self._fail_next = False
