"""Source connectors — the connector-neutral ingest boundary.

Exposes the :class:`SourceConnector` ABC and its shared metadata/capability
types (from :mod:`curator.connectors.base`) plus the S01 concrete fixtures:
:class:`curator.connectors.local.LocalConnector` and
:class:`curator.connectors.remote_fixture.SyntheticRemoteConnector`.
"""

from __future__ import annotations

from curator.connectors.base import (
    AssetMetadata,
    ConnectorCapabilities,
    ConnectorHealth,
    RevisionObservation,
    SourceConnector,
)
from curator.connectors.local import LocalConnector
from curator.connectors.remote_fixture import SyntheticRemoteConnector

__all__ = [
    "AssetMetadata",
    "ConnectorCapabilities",
    "ConnectorHealth",
    "LocalConnector",
    "RevisionObservation",
    "SourceConnector",
    "SyntheticRemoteConnector",
]
