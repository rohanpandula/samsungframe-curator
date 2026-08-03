"""LocalConnector — enumerate a folder on disk as a source.

Uses a **normalized absolute path** as the opaque ``asset_id`` and a stat-based
revision token (``mtime_ns:size``) so that a content change or rename surfaces
as a new revision. ``read_original`` streams the file bytes directly. The
connector is read-only over the folder — it never deletes or moves files.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from curator.connectors.base import (
    AssetMetadata,
    ConnectorCapabilities,
    ConnectorHealth,
    RevisionObservation,
    SourceConnector,
)
from curator.errors import ConnectorError

# Media types this connector will surface. Everything else in the folder is
# ignored by enumeration (e.g. dotfiles, directories).
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic")


def _revision_token(st) -> str:
    """Stat-derived revision token: ``mtime_ns:size``."""
    return f"{st.st_mtime_ns}:{st.st_size}"


class LocalConnector(SourceConnector):
    """Serves files under *folder* as source assets.

    The opaque ``asset_id`` is ``str(path.resolve())`` — a normalized absolute
    path, so the same folder yields identical ids regardless of how it was
    referenced (relative path, symlink, trailing slash).
    """

    def __init__(self, folder: Path, connector_id: str | None = None) -> None:
        self.folder = Path(folder).resolve()
        self.connector_id = connector_id or f"local:{self.folder}"
        self.capabilities = ConnectorCapabilities(
            supported_media_types=SUPPORTED_SUFFIXES,
            cursor_pagination=True,
            preview_stream=False,
            original_stream=True,
            revision_support=True,
        )

    # -- capability / health --------------------------------------------------

    def health(self) -> ConnectorHealth:
        if self.folder.is_dir():
            media = sum(1 for _ in self._media_files())
            return ConnectorHealth(
                healthy=True, detail=f"{self.folder} readable ({media} media files)"
            )
        return ConnectorHealth(
            healthy=False,
            detail=f"{self.folder} is not a directory",
        )

    # -- enumeration ----------------------------------------------------------

    def enumerate(self, cursor: str | None = None) -> Iterator[AssetMetadata]:
        """Yield metadata for every supported media file, sorted by path.

        Only files with a supported suffix are surfaced; directories and other
        files are ignored. Ordering is deterministic (sorted by absolute path).
        """
        for path in self._media_files():
            if cursor is not None and str(path) <= cursor:
                continue
            yield self._metadata(path)

    def _media_files(self) -> Iterator[Path]:
        if not self.folder.is_dir():
            return
        for path in sorted(self.folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
                yield path

    def _metadata(self, path: Path) -> AssetMetadata:
        st = path.stat()
        return AssetMetadata(
            asset_id=self._asset_id(path),
            connector_id=self.connector_id,
            revision=_revision_token(st),
            media_type=path.suffix.lower(),
            size_bytes=st.st_size,
            available=path.is_file(),
            extra={"path": str(path)},
        )

    def _asset_id(self, path: Path) -> str:
        """Normalized absolute path = opaque asset id."""
        return str(path.resolve())

    # -- streams --------------------------------------------------------------

    def read_original(self, asset_id: str) -> bytes:
        path = Path(asset_id)
        if not path.is_file():
            raise ConnectorError(f"local asset not readable: {asset_id!r}")
        try:
            return path.read_bytes()
        except OSError as exc:
            raise ConnectorError(f"failed to read local asset {asset_id!r}: {exc}") from exc

    # -- revisions / availability ----------------------------------------------

    def revisions(self, asset_id: str) -> Iterator[RevisionObservation]:
        """Yield the current state observation for *asset_id*.

        If the file is still on disk it yields one available observation; if the
        normalized path no longer resolves to a file, it yields a tombstone
        (``available=False``) without deleting history.
        """
        path = Path(asset_id)
        if path.is_file():
            st = path.stat()
            yield RevisionObservation(
                asset_id=self._asset_id(path),
                revision=_revision_token(st),
                changed=True,
                available=True,
            )
        else:
            yield RevisionObservation(
                asset_id=asset_id,
                revision="deleted",
                changed=True,
                available=False,
            )
