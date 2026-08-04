"""Tests for :func:`curator.analysis.cli_utils.resolve_catalog_entry` (M002/S04).

Proves the shared source->catalog-entry bridge: it returns the latest
``catalog_entries.id`` for a ``(connector_id, asset_id)`` pair using the same
scoping the ingest pipeline writes, and raises :class:`CatalogEntryNotFound`
for a source that has not been cataloged.
"""

from __future__ import annotations

import pytest

from curator.analysis.cli_utils import resolve_catalog_entry
from curator.analysis.errors import CatalogEntryNotFound
from curator.catalog import Catalog
from curator.cli import main as cli_main
from curator.connectors import LocalConnector
from fixture_library import build_fixture


def test_resolve_returns_latest_catalog_id(data_root, tmp_path):
    """resolve_catalog_entry returns the row id queried for the same source."""
    folder = build_fixture(tmp_path / "fixture").root
    assert cli_main(["ingest", str(folder)]) == 0
    conn = LocalConnector(folder).connector_id
    asset_id = str((folder / "single_00.jpg").resolve())

    catalog = Catalog()
    try:
        entry_id = resolve_catalog_entry(catalog, conn, asset_id)
        assert isinstance(entry_id, int)

        row = catalog.db.execute(
            "SELECT id FROM catalog_entries"
            " WHERE connector_id=? AND asset_id=? ORDER BY id DESC LIMIT 1",
            (conn, asset_id),
        ).fetchone()
        assert row is not None
        assert entry_id == int(row[0])
    finally:
        catalog.db.close()


def test_resolve_missing_asset_raises(data_root, tmp_path):
    """resolve_catalog_entry raises CatalogEntryNotFound for an uncataloged source."""
    folder = build_fixture(tmp_path / "fixture").root
    assert cli_main(["ingest", str(folder)]) == 0
    conn = LocalConnector(folder).connector_id

    catalog = Catalog()
    try:
        with pytest.raises(CatalogEntryNotFound):
            resolve_catalog_entry(
                catalog, conn, str((folder / "never_cataloged.png").resolve())
            )
    finally:
        catalog.db.close()
