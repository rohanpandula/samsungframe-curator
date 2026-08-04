"""Connector-contract acceptance suite (S05).

T1 ships a single minimal placeholder contract test (connector-scoped identity)
so the acceptance gate is green from the start. T4 replaces this placeholder with
the full six-part suite (cursor resume / scoped identity / content convergence /
append-only tombstone / pipeline failure isolation / changed-revision
preservation). No source-code under ``src/curator`` is modified.
"""

from __future__ import annotations

from curator.catalog import Catalog


def test_connector_scoped_identity_placeholder(data_root):
    """Same opaque asset_id via two connectors lands in two distinct source rows."""
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
