"""Isolated tests for LocalConnector (T05).

Verifies the connector contract in isolation: capability/health reporting,
deterministic cursor enumeration, normalized absolute-path opacity, stat-based
revision tokens, original stream reads, and availability tombstones on the
local connector.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from curator.connectors import LocalConnector
from curator.errors import ConnectorError


def _make_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "media"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"alpha-bytes")
    (folder / "b.png").write_bytes(b"bravo-bytes!")
    (folder / "nested").mkdir()
    (folder / "nested" / "c.webp").write_bytes(b"charlie")
    (folder / "ignore.txt").write_bytes(b"not-media")
    return folder


def test_health_healthy_on_real_folder(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    health = connector.health()
    assert health.healthy is True
    assert "3 media files" in health.detail


def test_health_not_healthy_on_missing_folder(tmp_path):
    connector = LocalConnector(tmp_path / "missing")
    health = connector.health()
    assert health.healthy is False


def test_capabilities_advertised(tmp_path):
    connector = LocalConnector(_make_folder(tmp_path))
    caps = connector.capabilities
    assert caps.original_stream is True
    assert caps.preview_stream is False
    assert caps.cursor_pagination is True
    assert caps.revision_support is True
    assert ".jpg" in caps.supported_media_types


def test_enumerate_normalized_absolute_asset_ids(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    assets = list(connector.enumerate())
    assert len(assets) == 3  # a.jpg, b.png, nested/c.webp (not ignore.txt)
    for meta in assets:
        # Normalized absolute path is the opaque id; must equal resolved path.
        assert meta.asset_id == str(Path(meta.asset_id).resolve())
        assert meta.asset_id.startswith(str(folder.resolve()))
        assert meta.connector_id == connector.connector_id
        assert meta.available is True


def test_enumerate_deterministic_order(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    ids = [meta.asset_id for meta in connector.enumerate()]
    assert ids == sorted(ids)


def test_enumerate_cursor_resumes(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    all_ids = [meta.asset_id for meta in connector.enumerate()]
    resumed = [meta.asset_id for meta in connector.enumerate(all_ids[0])]
    assert resumed == all_ids[1:]


def test_revision_token_changes_when_mtime_changes(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    before = {meta.asset_id: meta.revision for meta in connector.enumerate()}
    target = folder / "a.jpg"
    aid = str(target.resolve())
    # Bump mtime deterministically forward by a distinct, non-overlapping ns offset.
    os.utime(target, ns=(target.stat().st_atime_ns + 300, target.stat().st_mtime_ns + 300))
    after = {meta.asset_id: meta.revision for meta in connector.enumerate()}
    # Revision token is stat-based "mtime_ns:size".
    assert ":" in before[aid]
    # A real mtime change must produce a fresh revision token.
    assert after[aid] != before[aid]


def test_read_original_round_trip(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    meta = next(m for m in connector.enumerate() if m.media_type == ".png")
    assert connector.read_original(meta.asset_id) == b"bravo-bytes!"


def test_read_original_missing_raises_connector_error(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    with pytest.raises(ConnectorError):
        connector.read_original(str(folder / "does-not-exist.jpg"))


def test_revision_tombstone_when_file_removed(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    meta = next(m for m in connector.enumerate() if m.media_type == ".jpg")
    target = Path(meta.asset_id)
    target.unlink()
    observations = list(connector.revisions(meta.asset_id))
    assert len(observations) == 1
    assert observations[0].available is False  # tombstone, not deletion
    assert observations[0].asset_id == meta.asset_id


def test_connector_id_defaults_to_folder(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder)
    assert connector.connector_id == f"local:{folder.resolve()}"


def test_connector_id_custom_override(tmp_path):
    folder = _make_folder(tmp_path)
    connector = LocalConnector(folder, connector_id="my-local")
    assert connector.connector_id == "my-local"


# ---------------------------------------------------------------------------
# T02: TIFF + RAW surfacing (explicit-unsupported surface)
# ---------------------------------------------------------------------------


def test_tiff_surfaced_as_supported(tmp_path):
    """TIFF is enumerable and advertised as a decodable, supported format."""
    folder = tmp_path / "media"
    folder.mkdir()
    (folder / "a.tiff").write_bytes(b"tiff-bytes")
    (folder / "b.tif").write_bytes(b"tif-bytes")
    connector = LocalConnector(folder)
    assets = list(connector.enumerate())
    assert {meta.media_type for meta in assets} == {".tiff", ".tif"}
    # TIFF is never flagged unsupported — it is decodable (R003).
    for meta in assets:
        assert meta.extra.get("unsupported") is not True
    assert ".tiff" in connector.capabilities.supported_media_types
    assert ".tif" in connector.capabilities.supported_media_types


def test_raw_surfaced_but_flagged_unsupported(tmp_path):
    """RAW files are enumerated (never silently dropped) but flagged unsupported."""
    folder = tmp_path / "media"
    folder.mkdir()
    (folder / "photo.cr2").write_bytes(b"cr2-raw")
    (folder / "shot.nef").write_bytes(b"nef-raw")
    (folder / "pic.arw").write_bytes(b"arw-raw")
    (folder / "frame.dng").write_bytes(b"dng-raw")
    connector = LocalConnector(folder)
    assets = list(connector.enumerate())
    assert len(assets) == 4
    for meta in assets:
        assert meta.extra.get("unsupported") is True  # explicit unsupported (R003)
        assert meta.available is True
        assert meta.asset_id.startswith(str(folder.resolve()))
    # RAW is surfaced but never advertised as decodable/supported.
    raw_types = {meta.media_type for meta in assets}
    assert raw_types & set(connector.capabilities.supported_media_types) == set()


def test_mixed_folder_surfaces_supported_and_raw_ignores_others(tmp_path):
    """Enumerates supported + RAW, and still ignores unrelated/dot files."""
    folder = tmp_path / "media"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"a")
    (folder / "b.tiff").write_bytes(b"b")
    (folder / "c.cr2").write_bytes(b"c")
    (folder / "d.txt").write_bytes(b"d")
    (folder / ".hidden").write_bytes(b"h")
    connector = LocalConnector(folder)
    by_type = {meta.media_type: meta for meta in connector.enumerate()}
    assert set(by_type) == {".jpg", ".tiff", ".cr2"}
    assert by_type[".jpg"].extra.get("unsupported") is not True
    assert by_type[".tiff"].extra.get("unsupported") is not True
    assert by_type[".cr2"].extra.get("unsupported") is True


def test_health_counts_raw_files(tmp_path):
    """Health media count includes surfaced RAW files."""
    folder = tmp_path / "media"
    folder.mkdir()
    (folder / "a.jpg").write_bytes(b"a")
    (folder / "b.cr2").write_bytes(b"b")
    connector = LocalConnector(folder)
    assert "2 media files" in connector.health().detail


def test_raw_and_supported_suffix_classifiers():
    """Classifier helpers are case-insensitive and partition the two sets."""
    from curator.connectors.local import (
        RAW_SUFFIXES,
        SUPPORTED_SUFFIXES,
        is_raw_suffix,
        is_supported_suffix,
    )

    assert ".tiff" in SUPPORTED_SUFFIXES
    assert ".tif" in SUPPORTED_SUFFIXES
    assert ".cr2" in RAW_SUFFIXES
    assert ".dng" in RAW_SUFFIXES
    # Case-insensitive.
    assert is_raw_suffix(".NEF") is True
    assert is_supported_suffix(".WebP") is True
    # A supported suffix is never RAW; a RAW suffix is never supported.
    assert is_raw_suffix(".jpg") is False
    assert is_supported_suffix(".cr2") is False
    assert is_supported_suffix(".tiff") is True


def test_raw_read_original_round_trip(tmp_path):
    """Surface a RAW file as an ordinary readable asset (bytes stream intact)."""
    folder = tmp_path / "media"
    folder.mkdir()
    payload = b"raw-payload"
    (folder / "photo.cr2").write_bytes(payload)
    connector = LocalConnector(folder)
    meta = next(iter(connector.enumerate()))
    assert connector.read_original(meta.asset_id) == payload
