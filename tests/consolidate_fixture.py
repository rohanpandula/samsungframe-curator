"""Deterministic legacy-SSD consolidation fixture (S03-T3, S04/S05 reuse).

This module builds a reproducible, in-process (Pillow-only, no network / no
external assets) legacy-ssd source folder whose 8-group consolidation arithmetic
is pinned, so the executor + CLI + acceptance can assert exact outcomes. It
reuses the shared band-limited ``scene_image`` (phash-stable) and a generated
Samsung Frame panel.

Fixture arithmetic (all 11 files are consolidated; 9 distinct content SHA-256):

  * exact_dupes          : 3 byte-identical copies    (2024-01-01_exact*.jpg) -> 1 hash
  * near_dupes           : base + 2x resized variant  (2024-01-02_near_*.jpg) -> 2 hashes
  * filename_collisions  : a/ + b/ same basename      (2024-02-01_photo.jpg)  -> 2 hashes
  * panels               : 1 x 1920x1080 panel        (2024-03-01_panel.jpg)  -> 1 hash
  * sidecars             : 1 x .xmp companion         (2024-03-01_panel.xmp)  -> 1 hash
  * corrupt              : broken.jpg (garbage bytes)                          -> 1 hash
  * missing_date         : undated scene              (nodate.jpg)             -> 1 hash

Totals: consolidated files = 11, unique library blobs = 9 (byte-dupes converge).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from fixture_library import scene_image

# Pinned fixture arithmetic (public so S04/S05 can assert without re-deriving).
CONSOLIDATED_FILES = 11
UNIQUE_LIBRARY_FILES = 9

# The exhaustive set of relative POSIX paths that must be consolidated.
EXPECTED_REL_FILES: tuple[str, ...] = (
    "2024-01-01_exact.jpg",
    "2024-01-01_exact_copy1.jpg",
    "2024-01-01_exact_copy2.jpg",
    "2024-01-02_near_base.jpg",
    "2024-01-02_near_resized.jpg",
    "2024-03-01_panel.jpg",
    "2024-03-01_panel.xmp",
    "a/2024-02-01_photo.jpg",
    "b/2024-02-01_photo.jpg",
    "broken.jpg",
    "nodate.jpg",
)


@dataclass(frozen=True)
class ConsolidationFixture:
    """Metadata describing a generated legacy-ssd fixture folder."""

    root: Path          # the legacy-ssd source folder
    consolidated_files: int
    unique_library_files: int
    rel_files: tuple[str, ...]


def _panel_image() -> Image.Image:
    """A smooth, band-limited 1920x1080 Samsung Frame panel."""
    img = Image.new("RGB", (1920, 1080), (100, 100, 100))
    ImageDraw.Draw(img).ellipse([300, 220, 1620, 860], fill=(150, 150, 150))
    return img.filter(ImageFilter.GaussianBlur(40))


def _write(path: Path, img: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), format="JPEG")


def build_consolidation_fixture(root: Path) -> ConsolidationFixture:
    """Create the deterministic legacy-ssd folder under *root*; return metadata.

    The builder is idempotent: an existing folder is rebuilt from scratch so
    repeated calls are reproducible.
    """
    src = root / "legacy-ssd"
    if src.exists():
        import shutil

        shutil.rmtree(src)
    src.mkdir(parents=True, exist_ok=True)

    # exact_dupes: 3 byte-identical copies -> 1 content hash.
    exact = scene_image(1)
    _write(src / "2024-01-01_exact.jpg", exact)
    data = (src / "2024-01-01_exact.jpg").read_bytes()
    (src / "2024-01-01_exact_copy1.jpg").write_bytes(data)
    (src / "2024-01-01_exact_copy2.jpg").write_bytes(data)

    # near_dupes: base + 2x resized variant (resize-stable phash) -> 2 hashes.
    base = scene_image(2)
    _write(src / "2024-01-02_near_base.jpg", base)
    _write(src / "2024-01-02_near_resized.jpg", base.resize((384, 384), Image.LANCZOS))

    # filename_collisions: same basename, different content, two subdirs -> 2 hashes.
    _write(src / "a" / "2024-02-01_photo.jpg", scene_image(3))
    _write(src / "b" / "2024-02-01_photo.jpg", scene_image(4))

    # panels + sidecar companion.
    _write(src / "2024-03-01_panel.jpg", _panel_image())
    (src / "2024-03-01_panel.xmp").write_bytes(b"<x:xmpmeta/>")

    # corrupt: supported suffix, undecodable bytes.
    (src / "broken.jpg").write_bytes(b"this is not an image at all")

    # missing_date: undated scene.
    _write(src / "nodate.jpg", scene_image(5))

    return ConsolidationFixture(
        root=src,
        consolidated_files=CONSOLIDATED_FILES,
        unique_library_files=UNIQUE_LIBRARY_FILES,
        rel_files=EXPECTED_REL_FILES,
    )
