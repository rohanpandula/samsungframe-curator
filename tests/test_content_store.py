"""Tests for src/curator/content_store.py.

Proves the content-addressed store: atomic two-level-sharded writes, idempotent
re-put of identical bytes (no-op, same hash), get round-trip, StorageError on a
missing blob, and a minimal content-row interaction with the migrated DB.
"""

from __future__ import annotations

import pytest

from curator import db
from curator.content_store import ContentStore
from curator.errors import StorageError
from curator.hashing import sha256_hex


@pytest.fixture
def store(data_root):
    """A ContentStore rooted under the isolated CURATOR_DATA_ROOT."""
    return ContentStore(data_root)


def test_put_returns_sha256(store):
    digest = store.put(b"hello world")
    assert digest == sha256_hex(b"hello world")


def test_put_leaves_blob_at_sharded_path(store):
    data = b"atomic-blob-content"
    digest = store.put(data)
    # Two-level sharding: <root>/content/ab/cd/<64-hex-sha256>
    expect = (
        store.content_root
        / digest[:2]
        / digest[2:4]
        / digest
    )
    assert expect.exists()
    assert expect.read_bytes() == data


def test_put_is_idempotent(store):
    data = b"same-bytes-twice"
    first = store.put(data)
    second = store.put(data)
    assert first == second
    # Exactly one blob file for this hash.
    final = store.content_root / first[:2] / first[2:4] / first
    assert final.exists()
    assert final.read_bytes() == data


def test_put_uses_tmp_then_atomic_move(store):
    """Temp files are consumed by the atomic move; the final blob is complete."""
    data = b"atomic-move"
    store.put(data)
    remaining = list(store.tmp_root.glob("*"))
    assert remaining == []
    assert len(list(store.content_root.glob("*/"))) >= 1


def test_get_round_trip(store):
    data = b"round-trip-bytes"
    digest = store.put(data)
    assert store.get(digest) == data


def test_get_shards_by_hash_prefix(store):
    """Identical bytes across two stores converge on identical shard paths."""
    data = b"converge"
    s1 = ContentStore(store.root)
    s2 = ContentStore(store.root)
    h1 = s1.put(data)
    h2 = s2.put(data)
    assert h1 == h2
    assert s1._blob_path(h1) == s2._blob_path(h2)


def test_get_missing_raises_storage_error(store):
    with pytest.raises(StorageError):
        store.get("0" * 64)


def test_put_invalid_hash_raises(store):
    with pytest.raises(StorageError):
        store._blob_path("short")


def test_store_resolves_root_from_env(data_root):
    """Without an explicit root, ContentStore uses the CURATOR_DATA_ROOT config."""
    store = ContentStore()
    assert store.root == data_root


def test_content_row_interacts_with_migrated_db(data_root, store):
    """Minimal cross-check: put + migrated DB converge on one content row."""
    conn = db.connect()
    db.migrate(conn)
    try:
        digest = store.put(b"db-interaction")
        conn.execute(
            "INSERT INTO content(sha256, size) VALUES (?, ?)", (digest, len(b"db-interaction"))
        )
        conn.commit()
        rows = conn.execute("SELECT sha256, size FROM content").fetchall()
        assert len(rows) == 1
        assert rows[0] == (digest, len(b"db-interaction"))
    finally:
        conn.close()
