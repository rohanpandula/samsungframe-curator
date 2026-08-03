"""Image decode step for the ingest pipeline (R003).

Turns raw source bytes into a content-addressed image signature: the SHA-256
byte identity (R004), decoded pixel dimensions, and a 64-bit perceptual hash
(phash) used for near-dupe clustering.

HEIC/HEIF support (R003) is provided by registering pillow-heif with Pillow once
at import time, so the single :func:`decode_image` path covers the full supported
format set (JPEG, PNG, GIF, WebP, TIFF, HEIC/HEIF) with no per-format branching.

A blob that cannot be decoded as an image raises :class:`DecodeError` carrying the
underlying reason — it is never silently dropped. The caller (IngestPipeline)
classifies that as a corrupt file and preserves the error text in the report and
journal so the failure is observable rather than invisible.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import imagehash
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from curator.errors import IngestError
from curator.hashing import sha256_hex

# Register HEIC/HEIF support with Pillow once at import time. Subsequent
# ``Image.open`` calls transparently handle .heic/.heif (R003).
register_heif_opener()


class DecodeError(IngestError):
    """Raised when source bytes cannot be decoded into an image signature."""


@dataclass(frozen=True)
class DecodedImage:
    """The derived signature of one successfully decoded image.

    ``phash`` is the lowercase 16-hex-char (64-bit) perceptual hash. All fields
    are set post-decode; a decoded image always has dimensions and a phash.
    """

    sha256: str
    width: int
    height: int
    phash: str


def decode_image(data: bytes) -> DecodedImage:
    """Decode *data* into a content-addressed image signature.

    The returned :class:`DecodedImage` carries the exact content hash, the
    orientation-corrected pixel dimensions, and the perceptual hash. On any
    decode failure (unknown format, truncated/corrupt bytes, unsupported codec)
    a :class:`DecodeError` is raised with the underlying reason preserved.
    """
    sha = sha256_hex(data)
    try:
        with Image.open(io.BytesIO(data)) as img:
            # Respect EXIF orientation so dimensions/phash reflect the displayed
            # image, keeping the pipeline's aspect-ratio and resolution heuristics
            # consistent with what a viewer sees.
            oriented = ImageOps.exif_transpose(img)
            width, height = oriented.size
            perceptual = imagehash.phash(oriented)
    except DecodeError:
        raise
    except Exception as exc:  # PIL raises varied types for malformed input
        raise DecodeError(f"failed to decode image ({type(exc).__name__}): {exc}") from exc
    return DecodedImage(
        sha256=sha,
        width=width,
        height=height,
        phash=str(perceptual).lower(),
    )
