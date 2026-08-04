"""Connector-contract acceptance suite (S05).

T1 shipped a single minimal placeholder contract test (connector-scoped
identity). T4 replaces it with the full six-part suite: cursor resume / scoped
identity / content convergence / append-only tombstone / pipeline failure
isolation / changed-revision preservation. Runs fully in-process against the
``data_root`` fixture — no source code under ``src/curator`` is modified.
"""

from __future__ import annotations

import uuid
from io import BytesIO

from PIL import Image

from curator.catalog import Catalog
from curator.connectors import SyntheticRemoteConnector
from curator.errors import ConnectorError
from curator.ingest.pipeline import IngestPipeline

ASSETS = {
    str(uuid.uuid4()): b"asset-a",
    str(uuid.uuid4()): b"asset-b",
    str(uuid.uuid4()): b"asset-c",
}


def _remote() -> SyntheticRemoteConnector:
    return SyntheticRemoteConnector(
        connector_id="svc:synthetic",
        assets=ASSETS,
        media_types={aid: ".jpg" for aid in ASSETS},
    )


def _png_bytes(size=(32, 32), color=(10, 120, 200)) -> bytes:
    """Return valid in-memory PNG bytes a connector can hand the pipeline."""
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def test_cursor_resume_yields_remaining_non_overlapping(data_root):
    """Enumerating with a cursor yields exactly the remaining asset_ids."""
    connector = _remote()
    all_ids = [meta.asset_id for meta in connector.enumerate()]
    resumed = [meta.asset_id for meta in connector.enumerate(all_ids[0])]
    assert resumed == all_ids[1:]


def test_connector_scoped_identity_distinct(data_root):
    """Same opaque asset_id via two connectors lands in two distinct rows."""
    catalog = Catalog(data_root=data_root)
    try:
        catalog.add_source("conn-a", "shared-asset", b"bytes-a")
        catalog.add_source("conn-b", "shared-asset", b"bytes-b")
        rows = catalog.db.execute(
            "SELECT connector_id, asset_id FROM source_assets ORDER BY connector_id"
        ).fetchall()
    finally:
        catalog.db.close()
    assert rows == [("conn-a", "shared-asset"), ("conn-b", "shared-asset")]


def test_duplicate_content_converges_on_one_content_row(data_root):
    """Identical bytes via two connectors converge on ONE content row."""
    catalog = Catalog(data_root=data_root)
    try:
        digest = catalog.add_source("conn-a", "asset-1", b"the-same-image")
        catalog.add_source("conn-b", "asset-2", b"the-same-image")
        assert catalog.db.execute("SELECT COUNT(*) FROM content").fetchone()[0] == 1
        assert len(catalog.get_by_hash(digest)) == 2
    finally:
        catalog.db.close()


def test_append_only_history_and_availability_tombstone(data_root):
    """remove() excludes from enumerate, keeps a tombstone, never deletes."""
    id_a = next(iter(ASSETS))
    connector = _remote()
    connector.upsert(id_a, b"asset-a-v2")
    connector.remove(id_a)

    assert id_a not in {meta.asset_id for meta in connector.enumerate()}
    observations = list(connector.revisions(id_a))
    assert observations[-1].available is False  # tombstone preserved, not deleted
    assert observations[0].available is True  # prior history untouched
    assert all(o.available for o in observations[:-1])  # only the tombstone flips


class _FailingRemote(SyntheticRemoteConnector):
    """SyntheticRemoteConnector whose read_original fails for one asset."""

    def __init__(
        self, connector_id: str, assets: dict[str, bytes], fail_on: set[str] | None = None
    ) -> None:
        super().__init__(connector_id, assets=assets)
        self._fail_on = fail_on or set()

    def read_original(self, asset_id: str) -> bytes:
        if asset_id in self._fail_on:
            raise ConnectorError(f"read failed for {asset_id}")
        return super().read_original(asset_id)


def test_pipeline_failure_isolation(data_root):
    """One asset read-failure isolates: others indexed, error recorded, no abort."""
    ok_id, bad_id = "ok", "bad-bytes"
    ok_bytes = _png_bytes()
    connector = _FailingRemote(
        "svc:fail",
        assets={ok_id: ok_bytes, bad_id: b"\x00\x01garbage"},
        fail_on={bad_id},
    )
    catalog = Catalog(data_root=data_root)
    report = IngestPipeline(connector, catalog=catalog).run()

    assert report.total_enumerated == 2
    assert report.indexed_count == 1
    assert report.error_count == 1
    err = next(f for f in report.failures if f.status == "error")
    assert err.asset_id == bad_id
    assert "read failed for bad-bytes" in (err.error or "")
    # The surviving asset made it into the catalog despite the sibling failure.
    assert catalog.get_by_source("svc:fail", ok_id) is not None
    journal = catalog.db.execute(
        "SELECT asset_id, status FROM ingest_journal ORDER BY id"
    ).fetchall()
    assert ("ok", "indexed") in journal
    assert ("bad-bytes", "error") in journal


def test_changed_revision_preserves_history_no_erasure(data_root):
    """Re-adding different bytes appends a revision, never erases the prior row."""
    catalog = Catalog(data_root=data_root)
    try:
        first = catalog.add_source("conn-local", "asset-1", b"original-bytes")
        second = catalog.add_source("conn-local", "asset-1", b"changed-bytes")
        assert first != second  # revision defaults to the (different) content sha

        rows = catalog.db.execute(
            "SELECT revision, sha256 FROM catalog_entries"
            " WHERE connector_id='conn-local' AND asset_id='asset-1' ORDER BY id"
        ).fetchall()
        assert len(rows) == 2  # both revisions retained, nothing erased
        assert {r[1] for r in rows} == {first, second}
        # get_by_source resolves the most recent revision.
        assert catalog.get_by_source("conn-local", "asset-1")["sha256"] == second
    finally:
        catalog.db.close()
