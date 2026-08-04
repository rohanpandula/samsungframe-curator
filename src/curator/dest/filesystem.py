"""Filesystem-backed destination adapter (M005/S01).

Writes artifacts as flat files directly beneath a root directory, keyed by
``artifact_id`` (the filename). Every write is **atomic**: bytes go to a unique
temp sibling first, are fsynced, then ``os.replace``d over the final path — so a
crash or a failing write never leaves a torn file, and ``replace`` is an exact-ID
overwrite of the same path.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

from curator.dest.base import DestinationAdapter, DestinationError
from curator.hashing import sha256_hex


class FilesystemDestinationAdapter(DestinationAdapter):
    """Exact-ID file backend rooted at *root_dir*."""

    def __init__(self, root_dir: Path | str) -> None:
        self.root = Path(root_dir)

    def capabilities(self) -> dict[str, bool]:
        return {
            "supports_put": True,
            "supports_replace": True,
            "supports_remove": True,
            "exact_ids": True,
        }

    def probe(self) -> dict[str, Any]:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    # -- path resolution -----------------------------------------------------

    def _path(self, artifact_id: str) -> Path:
        """Return the on-disk path for *artifact_id*.

        The id is treated as a bare filename (the basename is taken) so no id can
        escape the root via path traversal.
        """
        name = Path(artifact_id).name
        if not name:
            raise DestinationError(f"invalid artifact id: {artifact_id!r}")
        return self.root / name

    # -- adapter surface -----------------------------------------------------

    def put(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        return self._write(artifact_id, data)

    def replace(self, artifact_id: str, data: bytes, meta: dict[str, Any] | None = None) -> str:
        return self._write(artifact_id, data)

    def _write(self, artifact_id: str, data: bytes) -> str:
        """Atomically write *data* at *artifact_id*'s exact path."""
        path = self._path(artifact_id)
        tmp = path.parent / f"{path.name}.tmp-{secrets.token_hex(8)}"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise DestinationError(f"write failed for {artifact_id}: {exc}") from exc
        return sha256_hex(data)

    def remove(self, artifact_id: str) -> None:
        path = self._path(artifact_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DestinationError(f"remove failed for {artifact_id}: {exc}") from exc

    def get_state(self, artifact_id: str) -> dict[str, Any] | None:
        path = self._path(artifact_id)
        if not path.is_file():
            return None
        data = path.read_bytes()
        return {"sha256": sha256_hex(data), "size": len(data)}

    def observe(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_file())
