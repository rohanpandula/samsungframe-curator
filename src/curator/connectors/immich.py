"""Immich source connector + feedback sink (M005/S05).

Implements the SourceConnector boundary (base.py) over an Immich-style remote
server, plus a checkpointed ``sync()`` and a safely-gated feedback sink. Fully
**air-gapped**: the shipped :class:`SyntheticImmichTransport` is a deterministic,
in-memory fixture runtime — no network, no I/O — so the whole surface is
exercisable offline.

- **Transport.** :class:`ImmichTransport` describes a paginated, filterable
  remote with optional smart-search and feedback support. The synthetic
  implementation provides cursor pagination, filters (favorite / star / tags /
  date range), smart-search, on-demand original download whose SHA-256 matches
  each asset's recorded checksum, and ``mark_unavailable`` to simulate a
  removed/remote asset.
- **Connector.** :class:`ImmichConnector` mirrors the SourceConnector shape:
  capability/health probe, deterministic cursor enumeration, original (and
  deterministic preview) streams with SHA-256 verification, and append-only
  revision observations. ``sync()`` is checkpointed against the
  ``immich_sync_state``/``immich_asset_state`` tables (schema v11): the persisted
  query cursor makes a completed sync resume with no duplicate work, per-asset
  writes are idempotent, and state is keyed per connector instance.
- **Availability tombstones.** A removed remote asset is no longer enumerated,
  but its ``immich_asset_state`` row is flipped to ``available=0`` — never
  deleted (mirrors M001).
- **Feedback sink.** :class:`ImmichFeedbackSink` is **disabled by default** and
  only ever writes a **user-confirmed** favorite / album membership to the
  synthetic backend under a ``user:write`` scope. It never deletes and never
  performs admin operations; when not enabled or unsupported it returns a clear
  no-op ``excluded``/``unsupported`` result.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

from curator.connectors.base import (
    AssetMetadata,
    ConnectorCapabilities,
    ConnectorHealth,
    RevisionObservation,
    SourceConnector,
)
from curator.errors import ConnectorError
from curator.hashing import sha256_hex


@dataclass(frozen=True)
class ImmichAsset:
    """One normalized asset descriptor returned by the transport."""

    asset_id: str
    revision: str
    checksum: str
    favorite: bool = False
    star: float = 0.0
    tags: tuple[str, ...] = ()
    date: str | None = None
    media_type: str = ".jpg"
    dimensions: tuple[int, int] | None = None
    size_bytes: int = 0


@dataclass(frozen=True)
class ImmichQuery:
    """Filter predicate for browse/search.

    ``None`` leaves a dimension unfiltered; ``tags`` is a required-subset match.
    """

    favorite: bool | None = None
    star_min: float | None = None
    tags: tuple[str, ...] = ()
    date_from: str | None = None
    date_to: str | None = None
    text: str | None = None


@dataclass(frozen=True)
class SyncReport:
    """Result of one :meth:`ImmichConnector.sync` run."""

    connector_id: str
    enumerated: int
    written: int
    tombstoned: int
    cursor: str | None


@dataclass(frozen=True)
class FeedbackCapabilities:
    """What the feedback sink is allowed/able to do."""

    enabled: bool
    supported: bool
    permission_scope: str
    detail: str = ""


@dataclass(frozen=True)
class FeedbackResult:
    """Outcome of one feedback write; a no-op never raises."""

    ok: bool
    status: str
    detail: str = ""


class ImmichTransport:
    """Abstract remote-server surface the connector drives.

    ``browse`` is cursor-paginated and *query*-filtered; ``search`` is the
    optional smart-search path (raise :class:`ConnectorError` when unavailable).
    ``download_original`` must return bytes whose SHA-256 equals the asset's
    recorded ``checksum``. ``mark_unavailable``/feedback mutations are the
    synthetic fixture's write primitives.
    """

    def probe(self) -> dict[str, Any]:
        """Return version + capability info (filters, smart_search, feedback)."""
        raise NotImplementedError

    def browse(
        self,
        query: ImmichQuery,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[ImmichAsset], str | None]:
        """Return a page of matching assets plus the resume cursor (None at end)."""
        raise NotImplementedError

    def search(self, query: ImmichQuery) -> list[ImmichAsset]:
        """Smart-search matching assets; raises when unsupported."""
        raise NotImplementedError

    def download_original(self, asset_id: str) -> bytes:
        """Return the full original bytes for *asset_id*."""
        raise NotImplementedError

    def get_availability(self, asset_id: str) -> bool:
        """Whether *asset_id* is currently available server-side."""
        raise NotImplementedError

    def get_checksum(self, asset_id: str) -> str:
        """Return the recorded checksum for *asset_id*."""
        raise NotImplementedError

    def revisions(self, asset_id: str) -> Iterator[RevisionObservation]:
        """Yield the append-only revision observations for *asset_id*."""
        raise NotImplementedError

    def mark_favorite(self, asset_id: str) -> None:
        """Confirm a user-chosen favorite on *asset_id* (no delete/admin)."""
        raise NotImplementedError

    def add_to_album(self, album_id: str, asset_id: str) -> None:
        """Add *asset_id* membership to album *album_id* (no delete/admin)."""
        raise NotImplementedError


@dataclass
class _LiveAsset:
    asset_id: str
    favorite: bool
    star: float
    tags: tuple[str, ...]
    date: str | None
    revision: str
    checksum: str
    media_type: str
    dimensions: tuple[int, int] | None
    data: bytes
    available: bool = True


class SyntheticImmichTransport(ImmichTransport):
    """Deterministic, in-memory Immich fixture runtime (air-gapped).

    Assets are sorted by asset_id for stable cursor pagination. ``download_original``
    returns the recorded bytes, whose SHA-256 always equals the asset's checksum.
    """

    def __init__(
        self,
        assets: Iterable[Mapping[str, Any]] | None = None,
        smart_search: bool = True,
        page_size: int = 100,
    ) -> None:
        self.smart_search = smart_search
        self.page_size = page_size
        self._assets: dict[str, _LiveAsset] = {}
        self._history: dict[str, list[RevisionObservation]] = {}
        self._counters: dict[str, int] = {}
        self._albums: dict[str, set[str]] = {}
        for spec in (assets or []):
            d = dict(spec)
            data = d.pop("data", b"")
            if isinstance(data, str):
                data = data.encode()
            self.add(asset_id=str(d.pop("asset_id")), data=data, **d)

    # -- fixture mutation helpers ---------------------------------------------

    def add(
        self,
        asset_id: str,
        data: bytes,
        favorite: bool = False,
        star: float = 0.0,
        tags: Iterable[str] = (),
        date: str | None = None,
        media_type: str = ".jpg",
        dimensions: tuple[int, int] | None = None,
    ) -> None:
        """Add or update an asset, appending a revision observation."""
        checksum = sha256_hex(data)
        self._assets[asset_id] = _LiveAsset(
            asset_id=asset_id,
            favorite=favorite,
            star=star,
            tags=tuple(tags),
            date=date,
            revision="r1",
            checksum=checksum,
            media_type=media_type,
            dimensions=dimensions,
            data=data,
            available=True,
        )
        self._observe(asset_id)

    def mark_unavailable(self, asset_id: str) -> None:
        """Tombstone *asset_id*: mark unavailable, preserve history, never delete."""
        existing = self._assets.get(asset_id)
        if existing is None or not existing.available:
            raise ConnectorError(f"immich asset not present: {asset_id!r}")
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

    # -- transport contract ----------------------------------------------------

    def probe(self) -> dict[str, Any]:
        return {
            "version": "1.0.0-synthetic",
            "capabilities": {
                "filters": ["favorite", "star", "tags", "date"],
                "smart_search": self.smart_search,
                "feedback": True,
                "feedback_scope": "user:write",
            },
        }

    def browse(
        self,
        query: ImmichQuery,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> tuple[list[ImmichAsset], str | None]:
        matches = [
            a
            for a in sorted(self._assets.values(), key=lambda a: a.asset_id)
            if (cursor is None or a.asset_id > cursor)
            and a.available
            and self._matches(query, a)
        ]
        size = limit or self.page_size
        page = matches[:size]
        more = len(matches) > size
        next_cursor = page[-1].asset_id if (page and more) else None
        return [self._meta(a) for a in page], next_cursor

    def search(self, query: ImmichQuery) -> list[ImmichAsset]:
        if not self.smart_search:
            raise ConnectorError("smart search unsupported by this server")
        text = (query.text or "").lower()
        matches = [
            a
            for a in sorted(self._assets.values(), key=lambda a: a.asset_id)
            if a.available
            and self._matches(query, a)
            and (not text or text in a.asset_id.lower() or any(text in t.lower() for t in a.tags))
        ]
        return [self._meta(a) for a in matches]

    def download_original(self, asset_id: str) -> bytes:
        asset = self._assets.get(asset_id)
        if asset is None or not asset.available:
            raise ConnectorError(f"immich asset not available: {asset_id!r}")
        return asset.data

    def get_availability(self, asset_id: str) -> bool:
        asset = self._assets.get(asset_id)
        return bool(asset and asset.available)

    def get_checksum(self, asset_id: str) -> str:
        asset = self._assets.get(asset_id)
        if asset is None:
            raise ConnectorError(f"immich asset unknown: {asset_id!r}")
        return asset.checksum

    def revisions(self, asset_id: str) -> Iterator[RevisionObservation]:
        if asset_id not in self._history:
            raise ConnectorError(f"no observations for immich asset: {asset_id!r}")
        yield from self._history[asset_id]

    # -- feedback mutations (synthetic fixture write primitives) ----------------

    def mark_favorite(self, asset_id: str) -> None:
        asset = self._assets.get(asset_id)
        if asset is None or not asset.available:
            raise ConnectorError(f"immich asset not available: {asset_id!r}")
        asset.favorite = True

    def add_to_album(self, album_id: str, asset_id: str) -> None:
        if asset_id not in self._assets:
            raise ConnectorError(f"immich asset unknown: {asset_id!r}")
        self._albums.setdefault(album_id, set()).add(asset_id)

    def is_favorite(self, asset_id: str) -> bool:
        asset = self._assets.get(asset_id)
        return bool(asset and asset.favorite)

    def album_members(self, album_id: str) -> set[str]:
        return set(self._albums.get(album_id, set()))

    # -- helpers ----------------------------------------------------------------

    def _matches(self, query: ImmichQuery, asset: _LiveAsset) -> bool:
        if query.favorite is not None and asset.favorite != query.favorite:
            return False
        if query.star_min is not None and asset.star < query.star_min:
            return False
        if query.tags and not set(query.tags).issubset(set(asset.tags)):
            return False
        if query.date_from and asset.date and asset.date < query.date_from:
            return False
        if query.date_to and asset.date and asset.date > query.date_to:
            return False
        return True

    def _meta(self, asset: _LiveAsset) -> ImmichAsset:
        return ImmichAsset(
            asset_id=asset.asset_id,
            revision=asset.revision,
            checksum=asset.checksum,
            favorite=asset.favorite,
            star=asset.star,
            tags=asset.tags,
            date=asset.date,
            media_type=asset.media_type,
            dimensions=asset.dimensions,
            size_bytes=len(asset.data),
        )


@dataclass(frozen=True)
class _StateRecord:
    revision: str | None
    checksum: str | None
    available: bool


# Column list shared by both immich_asset_state upserts (schema v11).
_ASSET_STATE_COLS = "connector_id, asset_id, revision, checksum, available"
_ASSET_STATE_UPDATE = (
    " revision = excluded.revision, checksum = excluded.checksum,"
    " available = excluded.available, updated_at = datetime('now')"
)


class ImmichConnector(SourceConnector):
    """SourceConnector over an Immich-style remote (transport-backed).

    ``enumerate``/``read_original``/``revisions`` expose the SourceConnector
    shape; ``sync`` persists the query cursor + per-asset state checkpointed and
    isolated per connector instance.
    """

    def __init__(
        self,
        connector_id: str,
        transport: ImmichTransport,
        query: ImmichQuery | None = None,
        catalog: Any | None = None,
    ) -> None:
        self.connector_id = connector_id
        self.transport = transport
        self.catalog = catalog
        self.query = query or ImmichQuery()
        caps = transport.probe().get("capabilities", {})
        self.filters: tuple[str, ...] = tuple(caps.get("filters", ()))
        self.smart_search: bool = bool(caps.get("smart_search", False))
        self.feedback_supported: bool = bool(caps.get("feedback", False))
        self.feedback_scope: str = str(caps.get("feedback_scope", "read-only"))
        self.capabilities = ConnectorCapabilities(
            supported_media_types=(".jpg", ".png", ".webp", ".heic"),
            cursor_pagination=True,
            preview_stream=True,
            original_stream=True,
            revision_support=True,
        )

    # -- capability / health --------------------------------------------------

    def health(self) -> ConnectorHealth:
        try:
            probe = self.transport.probe()
        except Exception as exc:  # noqa: BLE001 - never raises per contract
            return ConnectorHealth(
                healthy=False,
                detail="probe failed",
                last_error=str(exc),
            )
        return ConnectorHealth(
            healthy=True,
            detail=f"ok (v{probe.get('version', '?')})",
        )

    # -- enumeration ----------------------------------------------------------

    def enumerate(self, cursor: str | None = None) -> Iterator[AssetMetadata]:
        """Yield metadata for available assets matching the query, in sorted order.

        ``cursor`` is the last seen asset_id; assets with id <= cursor are skipped
        (deterministic, non-overlapping resume).
        """
        for asset in self._browse_all(cursor):
            yield AssetMetadata(
                asset_id=asset.asset_id,
                connector_id=self.connector_id,
                revision=asset.revision,
                dimensions=asset.dimensions,
                media_type=asset.media_type,
                size_bytes=asset.size_bytes,
                available=True,
                extra={
                    "favorite": asset.favorite,
                    "star": asset.star,
                    "tags": list(asset.tags),
                    "date": asset.date,
                },
            )

    def _browse_all(self, cursor: str | None = None) -> Iterator[ImmichAsset]:
        while True:
            batch, next_cursor = self.transport.browse(self.query, cursor=cursor)
            yield from batch
            if next_cursor is None:
                break
            cursor = next_cursor

    # -- streams --------------------------------------------------------------

    def _download_verified(self, asset_id: str) -> bytes:
        data = self.transport.download_original(asset_id)
        recorded = self.transport.get_checksum(asset_id)
        if sha256_hex(data) != recorded:
            raise ConnectorError(f"checksum mismatch for immich asset: {asset_id!r}")
        return data

    def read_original(self, asset_id: str) -> bytes:
        return self._download_verified(asset_id)

    def read_preview(self, asset_id: str) -> bytes:
        data = self._download_verified(asset_id)
        return data[: len(data) // 2]

    # -- revisions / availability ----------------------------------------------

    def revisions(self, asset_id: str) -> Iterator[RevisionObservation]:
        yield from self.transport.revisions(asset_id)

    # -- sync (checkpointed + isolated + idempotent) ---------------------------

    def sync(self, catalog: Any | None = None) -> SyncReport:
        """Walk the remote and persist the query cursor + per-asset state.

        Checkpointed: the persisted cursor resumes the next run exactly where the
        last completed run stopped, so a second, complete sync does no duplicate
        enumeration or writes. Writes are isolated per connector instance (state
        keyed by connector_id) and idempotent (unchanged revisions are skipped).
        An asset that disappears is recorded with ``available=0`` and **never
        deleted**.
        """
        catalog = catalog or self.catalog
        if catalog is None:
            raise ConnectorError("sync requires a Catalog")
        db = catalog.db
        cursor = self._load_cursor(db)
        state = self._load_state(db)
        seen: set[str] = set()
        written = 0
        next_cursor = cursor
        last_id = cursor if cursor is not None else None
        while True:
            batch, nxt = self.transport.browse(self.query, cursor=next_cursor)
            for asset in batch:
                seen.add(asset.asset_id)
                prev = state.get(asset.asset_id)
                if self._write_asset_state(db, asset, prev):
                    written += 1
                if last_id is None or asset.asset_id > last_id:
                    last_id = asset.asset_id
            if nxt is None:
                break
            next_cursor = nxt
        self._save_cursor(db, last_id)
        tombstoned = self._write_tombstones(db, seen, state)
        db.commit()
        return SyncReport(
            connector_id=self.connector_id,
            enumerated=len(seen),
            written=written,
            tombstoned=tombstoned,
            cursor=last_id,
        )

    # -- sync persistence -------------------------------------------------------

    def _load_cursor(self, db: Any) -> str | None:
        row = db.execute(
            "SELECT cursor FROM immich_sync_state WHERE connector_id = ?",
            (self.connector_id,),
        ).fetchone()
        return str(row[0]) if row and row[0] is not None else None

    def _save_cursor(self, db: Any, cursor: str | None) -> None:
        db.execute(
            "INSERT INTO immich_sync_state(connector_id, cursor) VALUES (?, ?)"
            " ON CONFLICT(connector_id) DO UPDATE SET"
            "   cursor = excluded.cursor,"
            "   updated_at = datetime('now')",
            (self.connector_id, cursor),
        )

    def _load_state(self, db: Any) -> dict[str, _StateRecord]:
        rows = db.execute(
            "SELECT asset_id, revision, checksum, available FROM immich_asset_state"
            " WHERE connector_id = ?",
            (self.connector_id,),
        ).fetchall()
        return {
            str(r[0]): _StateRecord(
                revision=str(r[1]) if r[1] is not None else None,
                checksum=str(r[2]) if r[2] is not None else None,
                available=bool(r[3]),
            )
            for r in rows
        }

    def _write_asset_state(
        self, db: Any, asset: ImmichAsset, prev: _StateRecord | None
    ) -> bool:
        if prev is not None and prev.revision == asset.revision:
            return False
        db.execute(
            f"INSERT INTO immich_asset_state({_ASSET_STATE_COLS})"
            " VALUES (?, ?, ?, ?, 1)"
            " ON CONFLICT(connector_id, asset_id) DO UPDATE SET"
            f"{_ASSET_STATE_UPDATE}",
            (self.connector_id, asset.asset_id, asset.revision, asset.checksum),
        )
        return True

    def _write_tombstones(
        self,
        db: Any,
        seen: set[str],
        state: dict[str, _StateRecord],
    ) -> int:
        tombstoned = 0
        for asset_id, prev in state.items():
            if asset_id in seen:
                continue
            if self.transport.get_availability(asset_id):
                continue
            db.execute(
                f"INSERT INTO immich_asset_state({_ASSET_STATE_COLS})"
                " VALUES (?, ?, ?, ?, 0)"
                " ON CONFLICT(connector_id, asset_id) DO UPDATE SET"
                f"{_ASSET_STATE_UPDATE}",
                (self.connector_id, asset_id, prev.revision, prev.checksum),
            )
            tombstoned += 1
        return tombstoned


class ImmichFeedbackSink:
    """Writes only **user-confirmed** favorite / album-membership to the backend.

    Disabled by default. Never deletes and never performs admin operations; when
    disabled or unsupported, a write returns a clear ``excluded``/``unsupported``
    no-op result rather than raising.
    """

    def __init__(self, transport: ImmichTransport, enabled: bool = False) -> None:
        self.transport = transport
        self.enabled = enabled

    def capabilities(self) -> FeedbackCapabilities:
        caps = self.transport.probe().get("capabilities", {})
        supported = bool(caps.get("feedback", False))
        scope = str(caps.get("feedback_scope", "read-only"))
        detail = (
            "feedback enabled"
            if (self.enabled and supported)
            else ("feedback disabled" if not self.enabled else "feedback unsupported by server")
        )
        return FeedbackCapabilities(
            enabled=self.enabled,
            supported=supported,
            permission_scope=scope,
            detail=detail,
        )

    def write_favorite(self, asset_id: str, album_id: str | None = None) -> FeedbackResult:
        """Write a confirmed favorite, or album membership when *album_id* is set."""
        caps = self.capabilities()
        if not self.enabled or not caps.supported:
            status = "excluded" if not self.enabled else "unsupported"
            return FeedbackResult(
                ok=False,
                status=status,
                detail=f"{status} for {asset_id}; read-only no-op",
            )
        if album_id is not None:
            self.transport.add_to_album(album_id, asset_id)
            return FeedbackResult(
                ok=True,
                status="written",
                detail=f"added {asset_id} to album {album_id}",
            )
        self.transport.mark_favorite(asset_id)
        return FeedbackResult(
            ok=True,
            status="written",
            detail=f"confirmed favorite on {asset_id}",
        )
