"""Pure phash clustering (R004: exact / near / resized / crop-aware dedup).

This module is deliberately **pure and stateless**: it turns a list of
:class:`ImageItem` descriptors into a list of :class:`Cluster` groups with a
single best-original per cluster. It has no storage, no filesystem, and no I/O,
so S03's consolidation grouping can reuse it without coupling to the catalog.

Clustering does three passes over the input, each deterministic:

1. **Exact** — items sharing the same ``sha256`` (identical bytes) collapse into
   one group. These are exact dupes with a single canonical entry.
2. **Near** — within-phash-distance grouping over each exact group's
   representative using a Hamming distance threshold
   (:data:`PHASH_NEAR_THRESHOLD`). Transitive connectivity is resolved with a
   union-find so an A~B, B~C chain forms one cluster.
3. **Crop-aware aspect-ratio heuristic** — a near pair is only merged when its
   aspect ratios are compatible within :data:`CROP_AR_TOLERANCE`. Equal-ratio
   pairs are resizes; a modest ratio divergence (a moderate crop) is still
   merged; an extreme aspect-ratio divergence is not, which rejects coincidental
   near-phash merges of visually unrelated layouts.

Best-original selection is by resolution: the member with the largest pixel
``area`` (width * height), tie-broken by larger width then lexicographic key, so
the result is stable for identical inputs regardless of input ordering.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

# Hamming-distance upper bound (of a 64-bit phash) for two images to be
# considered near-dupes. Resizes of the same scene are ~0; near-dupe edits
# typically land <= 10 (research notes). Exact dupes are handled by sha256 and
# always have distance 0.
PHASH_NEAR_THRESHOLD: int = 10

# Max factor by which two aspect ratios may differ and still be considered a
# possible crop of one another. Equal ratio == 1.0 (resize); a 4:3 -> 1:1 crop
# is ~1.33; extreme divergences are rejected to avoid false near-merges.
CROP_AR_TOLERANCE: float = 2.0

# Prefix for the synthetic, content-stable cluster id.
_CLUSTER_ID_PREFIX = "cl-"


@dataclass(frozen=True)
class ImageItem:
    """One decodable image descriptor fed to the clusterer.

    ``key`` is the caller's opaque identity (a catalog asset id in the pipeline).
    ``sha256`` is the exact content hash (byte identity, R004); ``phash`` is the
    lowercase hex perceptual hash (may be ``None`` when decode produced no hash).
    ``width``/``height`` are decoded pixel dimensions used for the crop-aware
    aspect-ratio heuristic and best-original-by-resolution selection.
    ``metadata`` is a passthrough for caller-owned attributes (never inspected).
    """

    key: str
    sha256: str
    phash: str | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def area(self) -> int:
        """Pixel area (0 when dimensions are unknown)."""
        if self.width is not None and self.height is not None:
            return self.width * self.height
        return 0

    @property
    def aspect_ratio(self) -> float:
        """Width/height ratio (1.0 when unknown or degenerate)."""
        if self.width and self.height:
            return self.width / self.height
        return 1.0


def hamming_distance(phash_a: str, phash_b: str) -> int:
    """Hamming distance between two lowercase hex phash strings.

    Counts differing bits of the two 64-bit perceptual hashes. Returns 0 for
    identical hashes. Treats an empty/unknown hash as maximally different by
    raising :class:`ValueError` so callers fail loudly rather than miscount.
    """
    if not phash_a or not phash_b:
        raise ValueError("hamming_distance requires two non-empty phash strings")
    bits_a = int(phash_a, 16)
    bits_b = int(phash_b, 16)
    return (bits_a ^ bits_b).bit_count()


def best_original(items: Iterable[ImageItem]) -> ImageItem:
    """Return the highest-resolution item (best-original) from *items*.

    Selection is by largest pixel area, tie-broken by larger width then
    lexicographically smaller key, so it is deterministic for identical input
    regardless of source ordering.
    """
    return min(items, key=lambda it: (-it.area, -(it.width or 0), it.key))


@dataclass(frozen=True)
class Cluster:
    """A dedup cluster: one or more image items collapsed to a single entry.

    ``best_key`` names the highest-resolution member. Members are ordered for
    determinism (sorted by key). ``cluster_id`` is content-derived from the
    best member's sha256 so it is stable for identical content.
    """

    cluster_id: str
    members: list[ImageItem]
    best_key: str

    @property
    def best(self) -> ImageItem:
        """The best-original member of this cluster."""
        for member in self.members:
            if member.key == self.best_key:
                return member
        raise ValueError(f"best_key {self.best_key!r} not a member of cluster")

    @property
    def size(self) -> int:
        return len(self.members)


def _aspect_ratio_compatible(a: ImageItem, b: ImageItem) -> bool:
    """True when *a* and *b* could be resize/crop variants of one scene.

    Unknown or degenerate dimensions are treated as compatible (no reason to
    reject). Otherwise the wider/narrower aspect-ratio ratio must be within
    :data:`CROP_AR_TOLERANCE`.
    """
    if a.width is None or a.height is None or b.width is None or b.height is None:
        return True
    if a.width <= 0 or a.height <= 0 or b.width <= 0 or b.height <= 0:
        return True
    ar_a = a.width / a.height
    ar_b = b.width / b.height
    if ar_a == 0.0 or ar_b == 0.0:
        return True
    lo, hi = sorted((ar_a, ar_b))
    return (hi / lo) <= CROP_AR_TOLERANCE


def _near(a: ImageItem, b: ImageItem, threshold: int) -> bool:
    """True when *a* and *b* should merge as near-dupes.

    Both must have phash values within *threshold* Hamming distance, and their
    aspect ratios must be crop-compatible. Absence of a phash (``None``) means
    the pair can never merge on the near axis (exact-sha grouping still applies).
    """
    if a.phash is None or b.phash is None:
        return False
    if hamming_distance(a.phash, b.phash) > threshold:
        return False
    return _aspect_ratio_compatible(a, b)


class _UnionFind:
    """Minimal integer union-find for deriving transitive near-clusters."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1


def cluster_images(
    items: Iterable[ImageItem],
    phash_near_threshold: int = PHASH_NEAR_THRESHOLD,
) -> list[Cluster]:
    """Group *items* into dedup clusters.

    Exact-sha groups, then transitive near-phash (crop-tolerant) groups, then
    best-original selection. Returns clusters sorted deterministically by
    content-derived cluster id.
    """
    items = list(items)
    if not items:
        return []

    # -- Pass 1: exact grouping by content sha256 --------------------------------
    exact: dict[str, list[ImageItem]] = {}
    for item in items:
        exact.setdefault(item.sha256, []).append(item)
    groups = list(exact.values())  # insertion-ordered by first occurrence
    reps = [best_original(group) for group in groups]

    # -- Pass 2: near-phash transitive connectivity over representatives ---------
    uf = _UnionFind(len(groups))
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if _near(reps[i], reps[j], phash_near_threshold):
                uf.union(i, j)

    # -- Pass 3: compose clusters + best-original ---------------------------------
    by_root: dict[int, list[ImageItem]] = {}
    for gi, root in enumerate(uf.find(i) for i in range(len(groups))):
        by_root.setdefault(root, []).extend(groups[gi])

    clusters: list[Cluster] = []
    for members in by_root.values():
        members = sorted(members, key=lambda it: (it.key, it.sha256))
        best = best_original(members)
        clusters.append(
            Cluster(
                cluster_id=f"{_CLUSTER_ID_PREFIX}{best.sha256[:16]}",
                members=members,
                best_key=best.key,
            )
        )
    clusters.sort(key=lambda c: c.cluster_id)
    return clusters
