"""Source connectors — the connector-neutral ingest boundary.

Exposes the :class:`SourceConnector` ABC and its shared metadata/capability
types (from :mod:`curator.connectors.base`) plus the S01 concrete fixtures:
:class:`curator.connectors.local.LocalConnector`,
:class:`curator.connectors.remote_fixture.SyntheticRemoteConnector`, and the
M005/S05 Immich connector + synthetic transport + feedback sink from
:mod:`curator.connectors.immich`.
"""

from __future__ import annotations

from curator.connectors.base import (
    AssetMetadata,
    ConnectorCapabilities,
    ConnectorHealth,
    RevisionObservation,
    SourceConnector,
)
from curator.connectors.immich import (
    FeedbackCapabilities,
    FeedbackResult,
    ImmichAsset,
    ImmichConnector,
    ImmichFeedbackSink,
    ImmichQuery,
    ImmichTransport,
    SyncReport,
    SyntheticImmichTransport,
)
from curator.connectors.local import LocalConnector
from curator.connectors.remote_fixture import SyntheticRemoteConnector

__all__ = [
    "AssetMetadata",
    "ConnectorCapabilities",
    "ConnectorHealth",
    "FeedbackCapabilities",
    "FeedbackResult",
    "ImmichAsset",
    "ImmichConnector",
    "ImmichFeedbackSink",
    "ImmichQuery",
    "ImmichTransport",
    "LocalConnector",
    "RevisionObservation",
    "SourceConnector",
    "SyncReport",
    "SyntheticImmichTransport",
    "SyntheticRemoteConnector",
]
