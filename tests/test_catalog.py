"""Tests for src/curator/catalog.py.

Proves the system-of-record Catalog API (R001): add_source writes the correct
SHA-256 and connector-scoped source identity; get_by_source / get_by_hash lookup
entries; re-add of the same file+asset is idempotent (one row); two connector
instances holding the same opaque asset_id create two distinct source_assets rows
(scoping) while identical bytes converge on one content row (dedup); and quality
flags round-trip through update_quality_flags. Includes negative paths (missing
lift / update on a non-existent entry).
"""

from __future__ import annotations

import pytest

from curator.catalog import Catalog
from curator.errors import CatalogError
from curator.hashing import sha256_hex


@pytest.fixture
def catalog(data_root):
    """A Catalog backed by a fresh migrated DB + ContentStore under the data root."""
    return Catalog(data_root=data_root)


def test_add_source_writes_sha256_and_identity(catalog):
    data = b"an-art-photo-bytes"
    digest = catalog.add_source("conn-local", "asset-1", data)

    assert digest == sha256_hex(data)
    # Blob is persisted in the content store.
    assert catalog.content.exists(digest)
    # Connector row + scoped asset row exist.
    assert catalog.db.execute(
        "SELECT 1 FROM source_connectors WHERE connector_id='conn-local'"
    ).fetchone()
    assert catalog.db.execute(
        "SELECT 1 FROM source_assets WHERE connector_id='conn-local' AND asset_id='asset-1'"
    ).fetchone()


def test_content_convergence_identical_bytes(catalog):
    """Identical bytes from two connectors converge on ONE content row."""
    digest = catalog.add_source("conn-a", "asset-1", b"the-same-image")
    catalog.add_source("conn-b", "asset-2", b"the-same-image")

    assert catalog.db.execute("SELECT COUNT(*) FROM content").fetchone()[0] == 1
    # And both entries resolve via get_by_hash -> two rows for the same digest.
    entries = catalog.get_by_hash(digest)
    assert len(entries) == 2


def test_connector_scoped_identity_distinct(catalog):
    """Same opaque asset_id across two connector instances = two source_assets rows."""
    catalog.add_source("conn-a", "shared-asset", b"bytes-a")
    catalog.add_source("conn-b", "shared-asset", b"bytes-b")

    rows = catalog.db.execute(
        "SELECT connector_id, asset_id FROM source_assets ORDER BY connector_id"
    ).fetchall()
    assert rows == [("conn-a", "shared-asset"), ("conn-b", "shared-asset")]


def test_get_by_source_returns_entry(catalog):
    digest = catalog.add_source("conn-local", "asset-1", b"findable")
    entry = catalog.get_by_source("conn-local", "asset-1")

    assert entry is not None
    assert entry["sha256"] == digest
    assert entry["connector_id"] == "conn-local"
    assert entry["asset_id"] == "asset-1"
    # The source identity is distinct from content identity.
    assert entry["sha256"] != "asset-1"


def test_get_by_source_missing_returns_none(catalog):
    assert catalog.get_by_source("no-such", "asset-9") is None


def test_readd_same_file_is_idempotent(catalog):
    """Re-adding the same bytes + same asset upserts a single catalog_entries row."""
    data = b"same-content-re-add"
    d1 = catalog.add_source("conn-local", "asset-1", data)
    d2 = catalog.add_source("conn-local", "asset-1", data)

    assert d1 == d2
    assert catalog.db.execute(
        "SELECT COUNT(*) FROM catalog_entries"
        " WHERE connector_id='conn-local' AND asset_id='asset-1'"
    ).fetchone()[0] == 1


def test_readd_with_explicit_revision_preserves_revisions(catalog):
    """A supplied revision keys separate rows; identical bytes+revision coalesce."""
    data = b"revisioned-content"
    catalog.add_source("conn-local", "asset-1", data, metadata={"revision": "v1"})
    catalog.add_source("conn-local", "asset-1", data, metadata={"revision": "v1"})
    catalog.add_source("conn-local", "asset-1", b"changed-bytes", metadata={"revision": "v2"})

    # Same (asset, revision) coalesces; distinct revisions are separate history rows.
    assert catalog.db.execute(
        "SELECT COUNT(*) FROM catalog_entries"
        " WHERE connector_id='conn-local' AND asset_id='asset-1'"
    ).fetchone()[0] == 2


def test_get_by_hash_returns_matching_entries(catalog):
    digest = catalog.add_source("conn-local", "asset-1", b"hash-lookup")
    catalog.add_source("conn-remote", "asset-2", b"hash-lookup")

    entries = catalog.get_by_hash(digest)
    assert {e["asset_id"] for e in entries} == {"asset-1", "asset-2"}


def test_get_by_hash_missing_returns_empty(catalog):
    assert catalog.get_by_hash("f" * 64) == []


def test_update_quality_flags(catalog):
    digest = catalog.add_source(
        "conn-local",
        "asset-1",
        b"quality-check",
        metadata={"quality_score": 0.5, "quality_reason": "initial"},
    )
    entry = catalog.get_by_source("conn-local", "asset-1")
    assert entry["quality_score"] == 0.5
    assert entry["quality_reason"] == "initial"

    catalog.update_quality_flags("conn-local", "asset-1", quality_score=0.9, quality_reason="sharp")
    entry = catalog.get_by_source("conn-local", "asset-1")
    assert entry["quality_score"] == 0.9
    assert entry["quality_reason"] == "sharp"
    assert entry["sha256"] == digest


def test_update_quality_flags_by_revision(catalog):
    catalog.add_source("conn-local", "asset-1", b"rev-update", metadata={"revision": "v1"})
    catalog.update_quality_flags(
        "conn-local", "asset-1", quality_score=0.75, quality_reason="ok", revision="v1"
    )
    entry = catalog.get_by_source("conn-local", "asset-1")
    assert entry["quality_score"] == 0.75


def test_update_quality_flags_missing_raises(catalog):
    with pytest.raises(CatalogError):
        catalog.update_quality_flags("conn-local", "no-such-asset", quality_score=0.5)


def test_connector_type_from_metadata(catalog):
    catalog.add_source(
        "conn-remote",
        "asset-1",
        b"type-check",
        metadata={"connector_type": "synthetic-remote"},
    )
    t = catalog.db.execute(
        "SELECT connector_type FROM source_connectors WHERE connector_id='conn-remote'"
    ).fetchone()[0]
    assert t == "synthetic-remote"


def test_catalog_auto_migrates(data_root):
    """A Catalog created without an explicit conn migrates its own tables."""
    cat = Catalog(data_root=data_root)
    try:
        cat.add_source("c", "a", b"auto-migrate")
        assert cat.get_by_source("c", "a") is not None
    finally:
        cat.db.close()
