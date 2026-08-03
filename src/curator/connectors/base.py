"""SourceConnector ABC — the connector-neutral ingest boundary (R028).

Every source of incoming art/photos (local folder in S01, the future Immich
connector in M005) implements :class:`SourceConnector`. The ABC fixes the
identity/availability/revision semantics that all downstream ingest (S02) and
the fixture contract tests rely on:

- **Connector-scoped opaque identity.** Each asset has an opaque `asset_id`
  that is only meaningful *within* a single connector instance. LocalConnector
  uses a normalized absolute path; remote connectors may use immutable UUIDs.
  The catalog keeps source identity distinct from content identity (D002), so
  the same opaque `asset_id` across two connector instances must never collide
  (enforced by the schema's ``UNIQUE(connector_id, asset_id)``).
- **Cursor-based enumeration.** ``enumerate`` walks assets deterministically and
  accepts a cursor for resumable/paginated traversal.
- **Revision observations + availability tombstones.** History is append-only:
  a revision that changes is observed with ``changed=True``; a reference that
  becomes unavailable is recorded with ``available=False`` (a **tombstone**) and
  is *never deleted* from the observation history.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass, field

from curator.errors import ConnectorError


@dataclass(frozen=True)
class ConnectorCapabilities:
    """Static advertisement of what a connector can do.

    Used by callers to decide whether a connector supports cursor pagination,
    preview vs. original streams, and revision history — without branching on
    connector type (R022 no-bespoke-axis-code-paths spirit).
    """

    supported_media_types: tuple[str, ...] = ()
    cursor_pagination: bool = True
    preview_stream: bool = False
    original_stream: bool = True
    revision_support: bool = True


@dataclass(frozen=True)
class ConnectorHealth:
    """Result of a connector health probe."""

    healthy: bool
    detail: str = ""
    last_error: str | None = None


@dataclass(frozen=True)
class AssetMetadata:
    """Normalized metadata for one enumerated source asset.

    ``asset_id`` is the opaque, connector-scoped identity. ``revision`` is a
    connector-produced version token (a stat tuple for LocalConnector, an
    incrementing token for the synthetic remote) used to detect change.
    """

    asset_id: str
    connector_id: str
    revision: str
    dimensions: tuple[int, int] | None = None  # (width, height)
    orientation: str | None = None
    color_profile: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    available: bool = True
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RevisionObservation:
    """One append-only observation of an asset's revision/availability state.

    ``available=False`` records a **tombstone**: the reference became unavailable
    but is preserved in history, never deleted. ``changed`` marks whether the
    revision token changed versus the previous observation.
    """

    asset_id: str
    revision: str
    changed: bool = True
    available: bool = True


class SourceConnector(abc.ABC):
    """Abstract base class for all source connectors."""

    connector_id: str
    capabilities: ConnectorCapabilities

    # -- capability / health --------------------------------------------------

    @abc.abstractmethod
    def health(self) -> ConnectorHealth:
        """Probe connector health; never raises."""

    # -- enumeration ----------------------------------------------------------

    @abc.abstractmethod
    def enumerate(self, cursor: str | None = None) -> Iterator[AssetMetadata]:
        """Yield normalized :class:`AssetMetadata` for the connector's assets.

        Ordering is deterministic. ``cursor`` resumes from a previous position
        when supported by :attr:`capabilities.cursor_pagination`.
        """

    # -- streams --------------------------------------------------------------

    @abc.abstractmethod
    def read_original(self, asset_id: str) -> bytes:
        """Return the full original bytes for *asset_id*.

        Raises :class:`ConnectorError` when the asset is unknown or unreadable.
        """

    def read_preview(self, asset_id: str) -> bytes:
        """Return a lightweight preview for *asset_id*.

        Only supported when :attr:`capabilities.preview_stream` is True. The
        default raises :class:`ConnectorError`.
        """
        raise ConnectorError(
            f"{type(self).__name__} does not support preview streams"
        )

    # -- revisions / availability ----------------------------------------------

    @abc.abstractmethod
    def revisions(self, asset_id: str) -> Iterator[RevisionObservation]:
        """Yield the append-only revision/availability observations for *asset_id*.

        History is never rewritten: an unavailable reference appears as an
        observation with ``available=False`` (tombstone), not as a deletion.
        """
