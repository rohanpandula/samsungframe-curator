"""CLI helpers for mapping sources to catalog entries (M002/S04).

:func:`resolve_catalog_entry` is the shared bridge used by the ``analyze`` CLI
surface: given a source asset identity (LocalConnector's normalized absolute
path) it returns the owning row id in ``catalog_entries``, or raises
:class:`~curator.analysis.errors.CatalogEntryNotFound` when the asset has not
been cataloged yet.
"""

from __future__ import annotations

from curator.analysis.errors import CatalogEntryNotFound
from curator.catalog import Catalog


def resolve_catalog_entry(
    catalog: Catalog, connector_id: str, asset_id: str
) -> int:
    """Return the ``catalog_entries.id`` for ``(connector_id, asset_id)``.

    Uses the same (connector_id, asset_id) scoping the ingest pipeline writes,
    ordering by ``id`` desc and taking the most recent entry for that source.
    Raises :class:`CatalogEntryNotFound` when no catalog entry exists for the
    source (so the caller can treat the asset as not-yet-cataloged).
    """
    row = catalog.db.execute(
        "SELECT id FROM catalog_entries"
        " WHERE connector_id = ? AND asset_id = ? ORDER BY id DESC LIMIT 1",
        (connector_id, asset_id),
    ).fetchone()
    if row is None:
        raise CatalogEntryNotFound(
            f"no catalog entry for {connector_id}/{asset_id}"
        )
    return int(row[0])
