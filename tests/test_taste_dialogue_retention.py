"""Tests for src/curator/taste/dialogue/retention.py (M008/S01/T3).

Proves ephemeral third-party retention: a decodable image is kept as a small
deterministic thumbnail + content hash with no full-resolution file written; the
same bytes retain identically; explicit save-to-catalog promotes full-resolution
bytes into the catalog while keeping the thumbnail; non-image bytes raise a
clear error; and the retention policy documents thumb+hash-only with explicit
save.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw

from curator.catalog import Catalog
from curator.errors import CuratorError
from curator.hashing import sha256_hex
from curator.taste.dialogue.retention import (
    MAX_THUMB_DIM,
    retain_ephemeral,
    retention_policy,
    save_to_catalog,
)


def _image_bytes(width: int = 1024, height: int = 768) -> bytes:
    """Return a decodable PNG larger than the thumbnail bound (banded gradient)."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for band in range(256):
        color = (band, (band * 7) % 256, (band * 13) % 256)
        top = band * height // 256
        bottom = (band + 1) * height // 256
        draw.rectangle([0, top, width, bottom], fill=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def test_retain_ephemeral_writes_thumb_only(data_root):
    image_bytes = _image_bytes()
    ref = retain_ephemeral(image_bytes, data_root)

    assert ref.ephemeral is True
    assert ref.catalog_saved is False
    assert ref.sha256 == sha256_hex(image_bytes)
    assert ref.thumb_path.is_absolute()
    assert ref.thumb_path.exists()

    with Image.open(ref.thumb_path) as thumb:
        assert thumb.width <= MAX_THUMB_DIM
        assert thumb.height <= MAX_THUMB_DIM

    files = [p for p in data_root.rglob("*") if p.is_file()]
    assert ref.thumb_path in files
    # Only the thumbnail is written: nothing under the data root exceeds it.
    thumb_size = ref.thumb_path.stat().st_size
    assert all(p.stat().st_size <= thumb_size for p in files)
    assert all(
        str(p.relative_to(data_root)).startswith("taste_ephemeral") for p in files
    )


def test_retain_ephemeral_deterministic(data_root):
    image_bytes = _image_bytes()
    first = retain_ephemeral(image_bytes, data_root)
    second = retain_ephemeral(image_bytes, data_root)

    assert first.sha256 == second.sha256
    assert first.thumb_path == second.thumb_path
    assert first.thumb_path.read_bytes() == second.thumb_path.read_bytes()


def test_save_to_catalog_promotes_and_keeps_thumb(data_root):
    image_bytes = _image_bytes()
    ref = retain_ephemeral(image_bytes, data_root)
    catalog = Catalog(data_root=data_root)

    saved = save_to_catalog(ref, image_bytes, catalog)

    assert saved.catalog_saved is True
    assert saved.sha256 == ref.sha256
    # A catalog entry exists for the promoted full-resolution bytes.
    assert catalog.get_by_source("taste-ephemeral", ref.sha256) is not None
    assert catalog.content.exists(ref.sha256)
    # Evidence retained: the ephemeral thumbnail still exists.
    assert ref.thumb_path.exists()


def test_retain_ephemeral_rejects_non_image(data_root):
    with pytest.raises((ValueError, CuratorError)):
        retain_ephemeral(b"definitely not an image", data_root)


def test_retention_policy_documents_thumb_hash_and_explicit_save():
    policy = retention_policy()

    assert policy
    assert "thumbnail" in policy
    assert "SHA-256" in policy
    assert "full-resolution" in policy
    assert "save-to-catalog" in policy
