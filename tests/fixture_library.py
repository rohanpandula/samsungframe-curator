"""Deterministic 50-file / 30-cluster mixed-format ingest fixture (T05, S05 reuse).

This module is the **shared S05 fixture generator**: it builds a reproducible
folder of 50 media files entirely in-process with Pillow (no network, no
external assets) whose dedup structure is pinned so that the IngestPipeline /
clusterer produce exactly 30 unique clusters. ``build_fixture`` is idempotent
and deterministic (fixed PRNG seeds), so re-running it yields byte-identical
files and the same cluster arithmetic every time.

Fixture structure (documented arithmetic — the acceptance
``SELECT COUNT(DISTINCT cluster_id)`` equals 30 relies on it):

  * 5 **exact-dupe families** (sizes ``[3,3,3,3,2]`` = 14 files, 9 byte-identical
    duplicate copies) -> 5 clusters.
  * 5 **resize families** (base + 2x LANCZOS upscale = 10 files, 5 resized
    variants) -> 5 clusters; the resized member is the highest-resolution
    member, so it is selected as ``best_original``.
  * 3 **near-dupe families** (base + slightly-blurred edit = 6 files, 3 near
    variants) -> 3 clusters.
  * 17 **singleton scenes** (17 unique images) -> 17 clusters.
  * 2 **RAW** files (``.cr2``, ``.dng``) — surfaced as explicit-unsupported
    (R003), not silently dropped.
  * 1 **corrupt** file (``.jpg`` with garbage bytes) — classified corrupt with
    the decode error preserved.

Arithmetic:

    Indexable files = 14 + 10 + 6 + 17          = 47
    Total files     = 47 + 2 RAW + 1 corrupt     = 50
    Unique clusters = 5 + 5 + 3 + 17             = 30
    Redundant copies (files beyond the first per cluster) = 47 - 30 = 17

Each cluster derives from a distinct *smooth, band-limited* scene (soft radial
bumps on a mid-gray field, heavily Gaussian-blurred). Band-limited content is
resize-stable in phash, and the 30 scene seeds were selected (greedily, margin
> 14 bits, measured against ``imagehash.phash``) to be mutually far apart while
each family's variants (resize / slight-blur) fall well inside the near
threshold (:data:`curator.ingest.clustering.PHASH_NEAR_THRESHOLD`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# ---------------------------------------------------------------------------
# Documented fixture arithmetic (public constants for tests / S05).
# ---------------------------------------------------------------------------
TOTAL_FILES = 50
TOTAL_CLUSTERS = 30
INDEXED_FILES = 47
RAW_FILES = 2
CORRUPT_FILES = 1

EXACT_FAMILIES = 5
EXACT_FAMILY_SIZES = (3, 3, 3, 3, 2)
EXACT_DUPE_COPIES = sum(s - 1 for s in EXACT_FAMILY_SIZES)  # 9
EXACT_FILES = sum(EXACT_FAMILY_SIZES)  # 14

RESIZE_FAMILIES = 5
RESIZED_VARIANTS = RESIZE_FAMILIES  # 5
RESIZE_FILES = RESIZE_FAMILIES * 2  # 10

NEAR_FAMILIES = 3
NEAR_VARIANTS = NEAR_FAMILIES  # 3
NEAR_FILES = NEAR_FAMILIES * 2  # 6

SINGLETONS = 17
SINGLETON_FILES = SINGLETONS  # 17

# The 30 mutually-far (>14 bits) distinct scene seeds used as cluster bases.
# Derived by greedy selection over ``imagehash.phash``; seed 23 is skipped as
# too close to a neighbor. Kept as an explicit tuple so the fixture is stable
# and self-documenting.
SCENE_SEEDS: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
    15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 29, 30,
)

# Scene seeds grouped by family role (the first 5 = exact, next 5 = resize,
# next 3 = near, remaining 17 = singletons).
_EXACT_SEEDS = SCENE_SEEDS[0:EXACT_FAMILIES]
_RESIZE_SEEDS = SCENE_SEEDS[EXACT_FAMILIES : EXACT_FAMILIES + RESIZE_FAMILIES]
_NEAR_SEEDS = SCENE_SEEDS[
    EXACT_FAMILIES + RESIZE_FAMILIES : EXACT_FAMILIES + RESIZE_FAMILIES + NEAR_FAMILIES
]
_SINGLETON_SEEDS = SCENE_SEEDS[EXACT_FAMILIES + RESIZE_FAMILIES + NEAR_FAMILIES :]

# Base / variant pixel dimensions (matters for best-original-by-resolution).
BASE_SIZE = 192
RESIZE_SCALE = 2  # resized variant = BASE_SIZE * 2
RESIZED_SIZE = BASE_SIZE * RESIZE_SCALE
NEAR_BLUR_RADIUS = 3

# File extensions per family role (mixed-format coverage: jpg/jpeg/png/webp/gif/
# tiff/heic — includes TIFF from T02 and HEIC covered by R003 decode).
_EXACT_EXTS = (".jpg", ".png", ".jpeg", ".png", ".webp")
_RESIZE_EXTS = (".jpg", ".png", ".jpeg", ".tiff", ".webp")
# NOTE: HEIC intentionally is *not* a resize-family format — pillow-heif's
# default HEIC encode is lossy and shifts the perceptual hash enough to exceed
# the near threshold for a base+resize pair. HEIC still appears in the fixture
# (single-file coverage via _SINGLETON_EXTS, R003 decode), so `curator ingest`
# exercises the HEIC decode path.
_NEAR_EXTS = (".jpg", ".png", ".gif")
_SINGLETON_EXTS = (".jpg", ".png", ".webp", ".gif", ".tiff", ".heic", ".jpeg")

_FMT_BY_EXT = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".gif": "GIF",
    ".tiff": "TIFF",
}

# RAW / corrupt files (surfaced unsupported / classified corrupt).
RAW_FILENAMES = ("photo_01.cr2", "photo_02.dng")
CORRUPT_FILENAME = "broken.jpg"


@dataclass(frozen=True)
class FixtureBuild:
    """Metadata describing a generated fixture folder.

    Exposes the deterministic structure so tests / S05 can assert membership
    without re-deriving the arithmetic (e.g. which files are resized
    best-originals, which are RAW, which is corrupt).
    """

    root: Path
    total_files: int
    clusters: int
    exact_files: list[str]
    resize_files: list[str]
    near_files: list[str]
    singleton_files: list[str]
    raw_files: list[str]
    corrupt_file: str
    # names of the highest-resolution (best-original) members of resize families
    resize_best_originals: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# scene generation (smooth, band-limited -> phash-stable under resize/edits)
# ---------------------------------------------------------------------------


def _rgb(scalar: float) -> tuple[int, int, int]:
    g = int(scalar * 255)
    return (g, g, g)


def scene_image(seed: int, size: int = BASE_SIZE) -> Image.Image:
    """Build the deterministic smooth scene for *seed* at *size*\u00b2 pixels.

    The scene is a mid-gray field overlaid with soft radial bumps that are then
    heavily Gaussian-blurred, producing band-limited (no hard edges) content
    whose perceptual hash is stable under resizing (resize family) and under a
    mild additional blur (near-dupe family).
    """
    rng = random.Random(500 + seed)
    base_lum = rng.uniform(0.25, 0.75)
    img = Image.new("RGB", (size, size), _rgb(base_lum))
    for _ in range(rng.randrange(2, 4)):
        cx = rng.randint(0, size)
        cy = rng.randint(0, size)
        rad = rng.randint(size // 4, size // 2)
        lum = rng.uniform(0.05, 0.6)
        blob = Image.new("L", (size, size), 0)
        ImageDraw.Draw(blob).ellipse(
            [cx - rad, cy - rad, cx + rad, cy + rad], fill=int(255 * lum)
        )
        blob = blob.filter(ImageFilter.GaussianBlur(rad // 2))
        img = Image.composite(Image.new("RGB", (size, size), _rgb(lum)), img, blob)
    return img.filter(ImageFilter.GaussianBlur(8))


def _save(img: Image.Image, path: Path) -> None:
    """Save *img* to *path*, choosing the right codec from the file extension."""
    ext = path.suffix.lower()
    if ext == ".heic":
        from pillow_heif import from_pillow

        heif = from_pillow(img)
        heif.save(str(path))
        return
    fmt = _FMT_BY_EXT[ext]
    img.save(str(path), format=fmt)


# ---------------------------------------------------------------------------
# public builder
# ---------------------------------------------------------------------------


def build_fixture(dest: Path) -> FixtureBuild:
    """Generate the deterministic 50-file / 30-cluster fixture under *dest*.

    The folder is created (idempotently) and populated; an empty or stale
    *dest* is replaced so repeated calls are reproducible. Returns a
    :class:`FixtureBuild` describing the generated structure.
    """
    dest = Path(dest)
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    exact_files: list[str] = []
    resize_files: list[str] = []
    near_files: list[str] = []
    singleton_files: list[str] = []
    resize_best: list[str] = []

    # -- exact-dupe families ---------------------------------------------------
    for i, (seed, k, ext) in enumerate(
        zip(_EXACT_SEEDS, EXACT_FAMILY_SIZES, _EXACT_EXTS)
    ):
        base_file = dest / f"exact_f{i}_base{ext}"
        _save(scene_image(seed), base_file)
        data = base_file.read_bytes()  # byte-identical copies -> exact sha256
        exact_files.append(base_file.name)
        for j in range(1, k):
            copy = dest / f"exact_f{i}_copy{j}{ext}"
            copy.write_bytes(data)
            exact_files.append(copy.name)

    # -- resize families (base + 2x LANCZOS upscale) ---------------------------
    for i, (seed, ext) in enumerate(zip(_RESIZE_SEEDS, _RESIZE_EXTS)):
        base = scene_image(seed, BASE_SIZE)
        base_file = dest / f"resize_f{i}_base{ext}"
        _save(base, base_file)
        resize_files.append(base_file.name)
        resized_file = dest / f"resize_f{i}_resized{ext}"
        _save(base.resize((RESIZED_SIZE, RESIZED_SIZE), Image.LANCZOS), resized_file)
        resize_files.append(resized_file.name)
        resize_best.append(resized_file.name)  # highest resolution -> best-original

    # -- near-dupe families (base + slightly-blurred edit) ----------------------
    for i, (seed, ext) in enumerate(zip(_NEAR_SEEDS, _NEAR_EXTS)):
        base = scene_image(seed, BASE_SIZE)
        base_file = dest / f"near_f{i}_base{ext}"
        _save(base, base_file)
        near_files.append(base_file.name)
        edit_file = dest / f"near_f{i}_edit{ext}"
        _save(base.filter(ImageFilter.GaussianBlur(NEAR_BLUR_RADIUS)), edit_file)
        near_files.append(edit_file.name)

    # -- singleton scenes -------------------------------------------------------
    for idx, seed in enumerate(_SINGLETON_SEEDS):
        ext = _SINGLETON_EXTS[idx % len(_SINGLETON_EXTS)]
        name = f"single_{idx:02d}{ext}"
        _save(scene_image(seed, BASE_SIZE), dest / name)
        singleton_files.append(name)

    # -- RAW (explicit unsupported, R003) --------------------------------------
    for name in RAW_FILENAMES:
        (dest / name).write_bytes(b"raw-payload-not-decodable-by-curator")

    # -- corrupt (decode error preserved) ---------------------------------------
    (dest / CORRUPT_FILENAME).write_bytes(b"this is not an image at all")

    return FixtureBuild(
        root=dest,
        total_files=TOTAL_FILES,
        clusters=TOTAL_CLUSTERS,
        exact_files=exact_files,
        resize_files=resize_files,
        near_files=near_files,
        singleton_files=singleton_files,
        raw_files=list(RAW_FILENAMES),
        corrupt_file=CORRUPT_FILENAME,
        resize_best_originals=resize_best,
    )
