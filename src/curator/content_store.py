"""Content-addressed artifact store.

The ContentStore is the single deduplication point of the catalog: any blob is
stored once, keyed by the SHA-256 of its bytes, and retrieved by that hash.
Convergence is trivial — identical bytes always produce an identical hash, so
re-adding the same content is a no-op and surfaces the same identity everywhere.

Writing is atomic: bytes are hashed, written to a random temp path under
``<root>/content/tmp/``, fsynced, then ``os.replace``d into a two-level hex
shard path ``<root>/content/ab/cd/<64-hex-sha256>``. A reader therefore never
observes a partially-written blob, and an interrupted ``put`` leaves only an
inert temp file behind (blobs are never deleted in S01).
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from curator.config import CuratorConfig
from curator.errors import StorageError
from curator.hashing import sha256_hex


class ContentStore:
    """Content-addressed blob store rooted under a CURATOR_DATA_ROOT-derived path.

    The default root is resolved from the six-axis config (``CURATOR_DATA_ROOT``,
    default ``~/.curator``) so the store always lives alongside the catalog DB.
    """

    def __init__(self, root: Path | None = None) -> None:
        if root is None:
            root = CuratorConfig().data_root
        self.root = Path(root)
        self.content_root = self.root / "content"
        self.tmp_root = self.content_root / "tmp"

    # -- internal layout helpers -------------------------------------------------

    def _blob_path(self, sha256: str) -> Path:
        """Return the final shard path for *sha256* (two-level hex sharding)."""
        if len(sha256) != 64:
            raise StorageError(f"invalid content hash {sha256!r}: expected 64 hex chars")
        return self.content_root / sha256[:2] / sha256[2:4] / sha256

    def _tmp_path(self) -> Path:
        """Return a fresh random temp path under the tmp root."""
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        return self.tmp_root / secrets.token_hex(16)

    # -- public API -------------------------------------------------------------

    def put(self, data: bytes) -> str:
        """Store *data* content-addressed and return its SHA-256 hex digest.

        If a blob with the same hash already exists, the write is a no-op and the
        existing blob is left untouched (idempotent). Otherwise the bytes are
        written to a temp file, fsynced, and atomically moved into the two-level
        shard path.
        """
        digest = sha256_hex(data)
        final_path = self._blob_path(digest)
        if final_path.exists():
            # Idempotent: matching blob already present; verify size matches to
            # guard against a truncated/corrupt leftover.
            if final_path.stat().st_size == len(data):
                return digest
            # Mismatched existing blob — overwrite atomically below.
        self._write_atomic(final_path, data)
        return digest

    def get(self, sha256: str) -> bytes:
        """Return the stored bytes for *sha256*.

        Raises :class:`StorageError` when no blob exists for the hash.
        """
        path = self._blob_path(sha256)
        if not path.exists():
            raise StorageError(f"content not found for hash {sha256!r}")
        try:
            return path.read_bytes()
        except OSError as exc:  # pragma: no cover - defensive read guard
            raise StorageError(f"failed to read content blob {path}: {exc}") from exc

    def exists(self, sha256: str) -> bool:
        """Return True when a blob for *sha256* is present in the store."""
        return self._blob_path(sha256).exists()

    # -- impl -------------------------------------------------------------------

    def _write_atomic(self, final_path: Path, data: bytes) -> None:
        """Write *data* to a temp file, fsync, then atomically move into place."""
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._tmp_path()
        with open(tmp_path, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Atomic move into the final shard path; replaces any stale/mismatched blob.
        os.replace(tmp_path, final_path)
