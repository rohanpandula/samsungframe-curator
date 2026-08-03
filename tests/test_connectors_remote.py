"""Isolated tests for SyntheticRemoteConnector (T05).

Verifies the remote connector contract in isolation (fully in-memory /
air-gapped): deterministic cursor enumeration, opaque UUID-style identity,
capability/health reporting, original + preview streams, append-only revision
history, and availability tombstones.
"""

from __future__ import annotations

import uuid

import pytest

from curator.connectors import SyntheticRemoteConnector
from curator.errors import ConnectorError

UUID_A = str(uuid.uuid4())
UUID_B = str(uuid.uuid4())


def _remote() -> SyntheticRemoteConnector:
    return SyntheticRemoteConnector(
        connector_id="svc:immich-1",
        assets={
            UUID_A: b"alpha-remote-data",
            UUID_B: b"bravo-remote-data!",
        },
        media_types={UUID_A: ".jpg", UUID_B: ".png"},
    )


def test_health_healthy_with_assets():
    connector = _remote()
    health = connector.health()
    assert health.healthy is True
    assert "2 live assets" in health.detail


def test_health_not_healthy_when_empty():
    connector = SyntheticRemoteConnector("svc:empty")
    assert connector.health().healthy is False


def test_capabilities_advertised():
    connector = _remote()
    caps = connector.capabilities
    assert caps.original_stream is True
    assert caps.preview_stream is True
    assert caps.cursor_pagination is True
    assert caps.revision_support is True
    assert ".jpg" in caps.supported_media_types


def test_enumerate_opaque_asset_ids_and_connector_id():
    connector = _remote()
    metas = {meta.asset_id: meta for meta in connector.enumerate()}
    assert set(metas) == {UUID_A, UUID_B}
    for meta in metas.values():
        assert meta.connector_id == "svc:immich-1"
        assert isinstance(meta.asset_id, str)
        assert meta.available is True


def test_enumerate_deterministic_sorted_order():
    connector = _remote()
    ids = [meta.asset_id for meta in connector.enumerate()]
    assert ids == sorted(ids)


def test_enumerate_cursor_resumes():
    connector = _remote()
    all_ids = [meta.asset_id for meta in connector.enumerate()]
    resumed = [meta.asset_id for meta in connector.enumerate(all_ids[0])]
    assert resumed == all_ids[1:]


def test_read_original_round_trip():
    connector = _remote()
    assert connector.read_original(UUID_A) == b"alpha-remote-data"
    assert connector.read_original(UUID_B) == b"bravo-remote-data!"


def test_read_original_unknown_raises_connector_error():
    connector = _remote()
    with pytest.raises(ConnectorError):
        connector.read_original("unknown-uuid")


def test_preview_stream_returns_deterministic_subset():
    connector = _remote()
    assert connector.read_preview(UUID_B) == b"bravo-rem"
    with pytest.raises(ConnectorError):
        connector.read_preview("unknown-uuid")


def test_revision_history_is_append_only():
    connector = _remote()
    connector.upsert(UUID_A, b"alpha-v2")
    connector.upsert(UUID_A, b"alpha-v3")
    observations = list(connector.revisions(UUID_A))
    assert [o.revision for o in observations] == ["r1", "r2", "r3"]
    assert all(o.changed for o in observations)
    assert all(o.available for o in observations)


def test_revision_tombstone_marks_unavailable_never_deletes():
    connector = _remote()
    connector.upsert(UUID_A, b"alpha-v2")
    connector.remove(UUID_A)
    # No longer enumerated (unavailable), but history is preserved.
    assert UUID_A not in {meta.asset_id for meta in connector.enumerate()}
    observations = list(connector.revisions(UUID_A))
    assert [o.revision for o in observations] == ["r1", "r2", "r3-tomb"]
    assert observations[-1].available is False  # tombstone preserved, not deleted
    assert observations[0].available is True  # prior history untouched


def test_remove_unknown_raises_connector_error():
    connector = _remote()
    with pytest.raises(ConnectorError):
        connector.remove("never-existed")


def test_read_original_tombstoned_raises_connector_error():
    connector = _remote()
    connector.remove(UUID_B)
    with pytest.raises(ConnectorError):
        connector.read_original(UUID_B)
