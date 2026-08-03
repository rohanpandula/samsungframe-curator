"""SyntheticRemoteConnector — deterministic, in-memory remote-source fixture.

Fully **air-gapped**: no network, no I/O, purely in-memory. Used only by the
contract tests (T07) and the isolated remote tests to exercise the connector
contract (opaque UUID-style identity, revision history, availability
tombstones) without any external service.

The fixture models the semantics every real remote connector must honour:

- Opaque immutable ``asset_id`` (in tests these are UUIDs).
- Append-only revision history via :meth:`revisions` — changing an asset appends
  an observation with a new revision token; removing an asset appends a
  **tombstone** (``available=False``) and the reference is preserved, never
  deleted from history.
- Deterministic enumeration (sorted by asset_id).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from curator.connectors.base import (
    AssetMetadata,
    ConnectorCapabilities,
    ConnectorHealth,
    RevisionObservation,
    SourceConnector,
)
from curator.errors import ConnectorError


@dataclass
class _RemoteAsset:
    """Internal live state for one synthetic remote asset."""

    asset_id: str
    data: bytes
    media_type: str = ".jpg"
    dimensions: tuple[int, int] | None = None
    available: bool = True


class SyntheticRemoteConnector(SourceConnector):
    """In-memory remote source for deterministic connector contract tests."""

    def __init__(
        self,
        connector_id: str,
        assets: dict[str, bytes] | None = None,
        media_types: dict[str, str] | None = None,
    ) -> None:
        self.connector_id = connector_id
        self.capabilities = ConnectorCapabilities(
            supported_media_types=(".jpg", ".png", ".webp"),
            cursor_pagination=True,
            preview_stream=True,
            original_stream=True,
            revision_support=True,
        )
        self._assets: dict[str, _RemoteAsset] = {}
        self._history: dict[str, list[RevisionObservation]] = {}
        self._counters: dict[str, int] = {}
        for asset_id, data in (assets or {}).items():
            self.upsert(
                asset_id,
                data,
                media_type=(media_types or {}).get(asset_id, ".jpg"),
            )

    # -- mutation helpers (test fixtures only) --------------------------------

    def upsert(
        self,
        asset_id: str,
        data: bytes,
        media_type: str = ".jpg",
        dimensions: tuple[int, int] | None = None,
    ) -> None:
        """Add or update an asset, appending a revision observation."""
        self._assets[asset_id] = _RemoteAsset(
            asset_id=asset_id,
            data=data,
            media_type=media_type,
            dimensions=dimensions,
            available=True,
        )
        self._observe(asset_id)

    def remove(self, asset_id: str) -> None:
        """Tombstone *asset_id*: mark unavailable, preserve history, never delete."""
        existing = self._assets.get(asset_id)
        if existing is None or not existing.available:
            raise ConnectorError(f"remote asset not present: {asset_id!r}")
        existing.available = False
        self._observe(asset_id)

    def _observe(self, asset_id: str) -> None:
        asset = self._assets[asset_id]
        self._counters[asset_id] = self._counters.get(asset_id, 0) + 1
        observation = RevisionObservation(
            asset_id=asset_id,
            revision=f"r{self._counters[asset_id]}{'-tomb' if not asset.available else ''}",
            changed=True,
            available=asset.available,
        )
        self._history.setdefault(asset_id, []).append(observation)

    # -- capability / health --------------------------------------------------

    def health(self) -> ConnectorHealth:
        live = sum(1 for a in self._assets.values() if a.available)
        detail = (
            f"ok ({live} live assets, {len(self._assets)} known)"
            if live == len(self._assets)
            else f"degraded ({live}/{len(self._assets)} live)"
        )
        return ConnectorHealth(healthy=len(self._assets) > 0, detail=detail)

    # -- enumeration ----------------------------------------------------------

    def enumerate(self, cursor: str | None = None) -> Iterator[AssetMetadata]:
        """Yield metadata for available assets, sorted by asset_id (deterministic)."""
        for asset_id in sorted(self._assets):
            asset = self._assets[asset_id]
            if not asset.available:
                continue
            if cursor is not None and asset_id <= cursor:
                continue
            yield AssetMetadata(
                asset_id=asset.asset_id,
                connector_id=self.connector_id,
                revision=self._last_revision(asset_id),
                dimensions=asset.dimensions,
                media_type=asset.media_type,
                size_bytes=len(asset.data),
                available=True,
            )

    def _last_revision(self, asset_id: str) -> str:
        hist = self._history.get(asset_id)
        return hist[-1].revision if hist else "r0"

    # -- streams --------------------------------------------------------------

    def _require_available(self, asset_id: str) -> _RemoteAsset:
        asset = self._assets.get(asset_id)
        if asset is None or not asset.available:
            raise ConnectorError(f"remote asset not available: {asset_id!r}")
        return asset

    def read_original(self, asset_id: str) -> bytes:
        return self._require_available(asset_id).data

    def read_preview(self, asset_id: str) -> bytes:
        """Return a deterministic lightweight preview.

        For the synthetic fixture the preview is the leading half of the bytes
        (a stand-in for a downscaled thumbnail); available only when the asset
        is live.
        """
        asset = self._require_available(asset_id)
        return asset.data[: len(asset.data) // 2]

    # -- revisions / availability ----------------------------------------------

    def revisions(self, asset_id: str) -> Iterator[RevisionObservation]:
        """Yield the append-only history for *asset_id* (never rewritten)."""
        if asset_id not in self._history:
            raise ConnectorError(f"no observations for remote asset: {asset_id!r}")
        yield from self._history[asset_id]
