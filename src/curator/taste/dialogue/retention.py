"""Ephemeral retention for taste-dialogue third-party images (M008/S01/T3).

Third-party images surfaced during a taste dialogue are retained as evidence
only: a small deterministic thumbnail plus the content SHA-256 of the original
bytes, under ``<data_root>/taste_ephemeral/``. The full-resolution image is never
written to disk by default; promoting it into the catalog is an explicit,
separate action (``save_to_catalog``).
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

from curator.catalog import Catalog
from curator.hashing import sha256_hex
from curator.ingest.decode import decode_image
from curator.taste.dialogue.observation import ImageRef

#: Longest edge of the retained thumbnail, in pixels.
MAX_THUMB_DIM = 512

#: JPEG quality for the deterministic thumbnail bytes.
_THUMB_QUALITY = 85


def retain_ephemeral(image_bytes: bytes, data_root: Path) -> ImageRef:
    """Retain *image_bytes* as a thumbnail + content hash and return an ImageRef.

    The bytes are first decoded through the canonical ingest decode path — a
    non-decodable blob raises :class:`~curator.errors.CuratorError`. A
    deterministic thumbnail (longest edge ≤ :data:`MAX_THUMB_DIM`) is then
    written to
    ``<data_root>/taste_ephemeral/<sha[:2]>/<sha[2:4]>/<sha>.thumb.jpg``
    (mirroring ContentStore sharding). The full-resolution bytes are never
    written.
    """
    decode_image(image_bytes)
    sha = sha256_hex(image_bytes)
    thumb_path = _thumbnail_path(Path(data_root), sha)
    thumb_path.parent.mkdir(parents=True, exist_ok=True)
    thumb_path.write_bytes(_thumbnail_bytes(image_bytes))
    return ImageRef(
        sha256=sha,
        thumb_path=thumb_path.absolute(),
        ephemeral=True,
        catalog_saved=False,
    )


def save_to_catalog(
    ref: ImageRef,
    image_bytes: bytes,
    catalog: Catalog,
    *,
    connector_id: str = "taste-ephemeral",
    asset_id: str | None = None,
) -> ImageRef:
    """Explicitly promote *ref*'s full-resolution bytes into the *catalog*.

    Routes through :meth:`Catalog.add_source` so the bytes land in the
    content-addressed store and a catalog entry links them to
    ``(connector_id, asset_id or ref.sha256)``. Returns a new :class:`ImageRef`
    with ``catalog_saved=True``; the ephemeral thumbnail is left in place as
    evidence.
    """
    catalog.add_source(
        connector_id,
        asset_id or ref.sha256,
        image_bytes,
        metadata={"source": "taste-dialogue-ephemeral"},
    )
    return ImageRef(
        sha256=ref.sha256,
        thumb_path=ref.thumb_path,
        ephemeral=ref.ephemeral,
        catalog_saved=True,
    )


def retention_policy() -> str:
    """Human-readable statement of the ephemeral-retention policy."""
    return (
        "Third-party ephemeral images are retained as thumbnails (longest edge "
        f"≤ {MAX_THUMB_DIM}px) plus their content SHA-256 for evidence only; the "
        "full-resolution image is never stored by default. An explicit "
        "save-to-catalog action promotes the full-resolution image into the "
        "catalog, leaving the thumbnail in place as evidence."
    )


def _thumbnail_bytes(image_bytes: bytes) -> bytes:
    """Return deterministic JPEG thumbnail bytes for *image_bytes*."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        oriented = ImageOps.exif_transpose(img)
        oriented.thumbnail((MAX_THUMB_DIM, MAX_THUMB_DIM))
        if oriented.mode not in ("RGB", "L"):
            oriented = oriented.convert("RGB")
        buffer = io.BytesIO()
        oriented.save(buffer, format="JPEG", quality=_THUMB_QUALITY, exif=b"")
        return buffer.getvalue()


def _thumbnail_path(data_root: Path, sha: str) -> Path:
    """Return the sharded thumb path for *sha* (mirrors ContentStore sharding)."""
    return data_root / "taste_ephemeral" / sha[:2] / sha[2:4] / f"{sha}.thumb.jpg"
