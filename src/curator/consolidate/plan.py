"""Directory-inventory consolidation planner (S03-T2 / R002 dry-run).

:class:`ConsolidationPlan` is the S03 dry-run deliverable: a JSON-serializable
summary of a **direct directory scan** of a legacy-ssd source folder, grouping
every file into 8 observable categories:

- ``exact_dupes``          — families of byte-identical files (shared content
                             SHA-256 with >1 member).
- ``near_dupes``           — near-dupe clusters (>1 distinct SHA-256 merged by
                             perceptual hash, crop-tolerant). Members are the
                             resized / edited variants of one scene.
- ``higher_res_originals`` — the highest-resolution (best-original) member of
                             each near-dupe cluster: the canonical that should
                             survive when lower-res members are substituted.
- ``filename_collisions``  — basename collision groups (two+ files share a
                             basename but differ in content), which the canonical
                             root's basename layout must disambiguate.
- ``panels``               — decodable images whose decoded dimensions match a
                             Samsung Frame panel (1920x1080 / 3840x2160).
- ``sidecars``             — non-media companion files (e.g. ``.xmp``) that pair
                             with media and must move with it.
- ``corrupt``              — supported-suffix files that do not decode as an
                             image (error text preserved, like ReportIssue).
- ``missing_date``         — decodable images with no detectable capture date
                             (no EXIF ``DateTimeOriginal`` and no filename date).

The plan is built from the directory rather than the catalog/IngestReport because
panel dimensions, sidecar pairing, filename collisions, and missing-date are
directory-inventory concepts S02 does not capture. It **reuses** S02 ingest
primitives (:func:`~curator.ingest.decode.decode_image`,
:class:`~curator.ingest.clustering.ImageItem`,
:func:`~curator.ingest.clustering.cluster_images`) so there is no CV/logic
duplication — the clusterer is pure/stateless and designed for this reuse.

No new runtime dependency is introduced for date detection: Pillow's ``getexif``
reads EXIF without ``piexif``; a filename date pattern is the zero-dep fallback.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from curator.connectors.local import SUPPORTED_SUFFIXES
from curator.errors import CuratorError
from curator.ingest.clustering import ImageItem, cluster_images
from curator.ingest.decode import DecodeError, decode_image

# Samsung Frame panel dimensions (the frames this pipeline targets). Any file
# whose decoded pixel dimensions match exactly is inventoried as a generated
# panel rather than ordinary photo content.
PANEL_DIMENSIONS: frozenset[tuple[int, int]] = frozenset(
    {(1920, 1080), (3840, 2160)}
)

# Non-media companion extensions that "move with" a paired media file. These are
# never attempted as images; they are inventoried as sidecars.
SIDECAR_SUFFIXES: frozenset[str] = frozenset(
    {".xmp", ".json", ".txt", ".md"}
)

# EXIF tag 0x9003 == DateTimeOriginal (the capture-time stamp we prefer).
_EXIF_DATETIME_ORIGINAL = 0x9003

# Filename date patterns: ``IMG_YYYYMMDD_...``, ``YYYY-MM-DD_...``, ``YYYY_MM_DD``,
# ``YYYYMMDD``. Falls back to classifying a file as missing_date when absent.
_FILENAME_DATE_RE = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")


@dataclass
class ConsolidationPlan:
    """JSON-serializable result of a directory-inventory consolidation dry-run.

    ``source_path`` is the resolved absolute path of the scanned legacy folder.
    Group members are stored as **relative POSIX paths** beneath ``source_path``
    so the plan is human-readable and stable regardless of the temp root.

    Dedup groups (``exact_dupes`` / ``near_dupes``) are disjoint by construction
    (every decodable image in a multi-member cluster lands in exactly one).
    ``panels`` / ``missing_date`` are independent flag groups and may overlap
    with dedup groups; ``filename_collisions`` captures basename clashes.
    """

    source_path: str
    exact_dupes: list[list[str]] = field(default_factory=list)
    near_dupes: list[list[str]] = field(default_factory=list)
    higher_res_originals: list[str] = field(default_factory=list)
    filename_collisions: list[list[str]] = field(default_factory=list)
    panels: list[str] = field(default_factory=list)
    sidecars: list[str] = field(default_factory=list)
    corrupt: list[dict[str, Any]] = field(default_factory=list)
    missing_date: list[str] = field(default_factory=list)

    # -- JSON surface (mirrors IngestReport) -----------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict (nested dataclasses expanded)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return this plan serialized as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    # -- reporting -------------------------------------------------------------

    def group_counts(self) -> dict[str, int]:
        """Return the number of files inventoried in each of the 8 groups.

        ``exact_dupes`` / ``near_dupes`` / ``filename_collisions`` are the total
        member-file counts across all their groups; the strict list groups are
        their length.
        """
        return {
            "exact_dupes": sum(len(group) for group in self.exact_dupes),
            "near_dupes": sum(len(group) for group in self.near_dupes),
            "higher_res_originals": len(self.higher_res_originals),
            "filename_collisions": sum(
                len(group) for group in self.filename_collisions
            ),
            "panels": len(self.panels),
            "sidecars": len(self.sidecars),
            "corrupt": len(self.corrupt),
            "missing_date": len(self.missing_date),
        }


def build_plan(source: Path) -> ConsolidationPlan:
    """Inventory *source* (a directory) into a :class:`ConsolidationPlan`.

    Walks *source* recursively, separating sidecars from supported-suffix media,
    decoding each media file into its content signature (
    :func:`~curator.ingest.decode.decode_image`), classifying decode failures as
    corrupt, and grouping the remaining images with the pure clusterer
    (:func:`~curator.ingest.clustering.cluster_images`) into exact/near dupes and
    higher-res originals. Raises :class:`CuratorError` when *source* is not a
    directory.
    """
    source = Path(source)
    if not source.is_dir():
        raise CuratorError(
            f"consolidate source is not a directory: {source}"
        )

    scan = _scan_directory(source)
    plan = ConsolidationPlan(source_path=str(source.resolve()))
    plan.sidecars = scan.sidecars
    plan.corrupt = scan.corrupt
    plan.filename_collisions = scan.filename_collisions

    if not scan.media:
        return plan

    # Panel + missing-date flags (independent of dedup grouping).
    for rel, data, sig in scan.media:
        if (sig.width, sig.height) in PANEL_DIMENSIONS:
            plan.panels.append(rel)
        if _detect_date(rel, data) is None:
            plan.missing_date.append(rel)

    # Dedup grouping via the pure clusterer (exact + near, crop-aware). Keys are
    # relative paths so groups are human-readable.
    items = [
        ImageItem(
            key=rel,
            sha256=sig.sha256,
            phash=sig.phash,
            width=sig.width,
            height=sig.height,
        )
        for rel, _, sig in scan.media
    ]
    for cluster in cluster_images(items):
        distinct_sha = {m.sha256 for m in cluster.members}
        member_keys = sorted(m.key for m in cluster.members)
        if len(cluster.members) < 2:
            continue  # singleton scene — not a dupe, not a near-dupe
        if len(distinct_sha) == 1:
            plan.exact_dupes.append(member_keys)
        else:
            plan.near_dupes.append(member_keys)
            plan.higher_res_originals.append(cluster.best_key)
    return plan


# ---------------------------------------------------------------------------
# internal scan helpers
# ---------------------------------------------------------------------------


@dataclass
class _ScanResult:
    media: list[tuple[str, bytes, Any]] = field(default_factory=list)  # (rel, data, sig)
    sidecars: list[str] = field(default_factory=list)
    corrupt: list[dict[str, Any]] = field(default_factory=list)
    filename_collisions: list[list[str]] = field(default_factory=list)


def _scan_directory(source: Path) -> _ScanResult:
    """Walk *source* once, separating media / sidecars / corrupt / collisions."""
    result = _ScanResult()
    by_name: dict[str, list[str]] = {}

    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        rel = _relposix(path, source)
        suffix = path.suffix.lower()
        if suffix in SIDECAR_SUFFIXES:
            result.sidecars.append(rel)
            by_name.setdefault(path.name, []).append(rel)
            continue
        if suffix not in SUPPORTED_SUFFIXES:
            continue  # unknown/RAW — outside the 8-group inventory surface
        by_name.setdefault(path.name, []).append(rel)
        data = path.read_bytes()
        try:
            sig = decode_image(data)
        except DecodeError as exc:
            result.corrupt.append({"path": rel, "error": str(exc)})
            continue
        result.media.append((rel, data, sig))

    # Basename collision groups: two+ files share a basename. (Content differing
    # is implied — same basename + same bytes would be exact dupes instead.)
    result.filename_collisions = [
        paths for paths in by_name.values() if len(paths) > 1
    ]
    return result


def _relposix(path: Path, source: Path) -> str:
    """Return *path* relative to *source* using ``/`` separators."""
    return path.relative_to(source).as_posix()


def _detect_date(rel: str, data: bytes) -> str | None:
    """Return a capture-date string for a media file, or ``None``.

    Prefers EXIF ``DateTimeOriginal`` (Pillow ``getexif``, no ``piexif`` dep),
    then a filename date pattern. ``None`` => the file is inventoried as
    ``missing_date``.
    """
    exif = _exif_datetime(data)
    if exif:
        return exif
    return _filename_date(rel)


def _exif_datetime(data: bytes) -> str | None:
    """Read EXIF ``DateTimeOriginal`` from image bytes, or ``None``."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            tag = img.getexif().get(_EXIF_DATETIME_ORIGINAL)
    except Exception:
        return None
    if not tag:
        return None
    value = str(tag).strip()
    return value or None


def _filename_date(rel: str) -> str | None:
    """Match a date pattern in *rel*'s basename, or ``None``.

    Recognizes ``YYYY-MM-DD``, ``YYYY_MM_DD``, or ``YYYYMMDD`` runs (optionally
    prefixed, e.g. ``IMG_20240101_``) and normalizes to ``YYYY-MM-DD``.
    """
    match = _FILENAME_DATE_RE.search(Path(rel).name)
    if not match:
        return None
    year, month, day = match.group(1), match.group(2), match.group(3)
    return f"{year}-{month}-{day}"
