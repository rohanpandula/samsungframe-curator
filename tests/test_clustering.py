"""Tests for the pure phash clusterer (src/curator/ingest/clustering.py).

The clusterer is deliberately stateless, so these tests drive it with
hand-crafted ``sha256``/``phash`` hex strings and pixel dimensions rather than
decoding real images — exact, near, resized, crop-aware, and best-original
behavior are each pinned deterministically.
"""

from __future__ import annotations

import pytest

from curator.ingest.clustering import (
    CROP_AR_TOLERANCE,
    PHASH_NEAR_THRESHOLD,
    ImageItem,
    best_original,
    cluster_images,
    hamming_distance,
)


def _phash(value: int) -> str:
    """Render a 64-bit int as a lowercase 16-hex-char phash string."""
    return format(value, "016x")


_BASE = 0xAAAAAAAAAAAAAAAA


def _item(
    key: str,
    sha256: str = "s",
    phash: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> ImageItem:
    return ImageItem(key=key, sha256=sha256, phash=phash, width=width, height=height)


# ---------------------------------------------------------------------------
# hamming_distance
# ---------------------------------------------------------------------------


class TestHammingDistance:
    def test_identical_is_zero(self):
        p = _phash(_BASE)
        assert hamming_distance(p, p) == 0

    def test_single_bit_flip(self):
        assert hamming_distance(_phash(_BASE), _phash(_BASE ^ 1)) == 1

    def test_multi_bit(self):
        flipped = _BASE ^ 0b101011
        assert hamming_distance(_phash(_BASE), _phash(flipped)) == 4

    def test_complement_is_max_distance(self):
        # A and 5 interleave all four bits of each nibble -> 64 bits differ.
        assert hamming_distance(_phash(_BASE), _phash(0x5555555555555555)) == 64

    def test_uppercase_is_equivalent(self):
        assert hamming_distance("AAAAAAAAAAAAAAAA", _phash(_BASE)) == 0

    def test_empty_raises_value_error(self):
        with pytest.raises(ValueError):
            hamming_distance("", _phash(_BASE))
        with pytest.raises(ValueError):
            hamming_distance(_phash(_BASE), "")


# ---------------------------------------------------------------------------
# best_original
# ---------------------------------------------------------------------------


class TestBestOriginal:
    def test_largest_area_wins(self):
        items = [
            _item("small", width=100, height=100),
            _item("large", width=300, height=200),
        ]
        assert best_original(items).key == "large"

    def test_equal_area_prefers_larger_width(self):
        # 200x100 == 100x200 area, but 200-wide wins the tie-break.
        items = [
            _item("portrait", width=100, height=200),
            _item("landscape", width=200, height=100),
        ]
        assert best_original(items).key == "landscape"

    def test_tie_prefers_lexicographically_first_key(self):
        items = [
            _item("b", width=100, height=100),
            _item("a", width=100, height=100),
        ]
        assert best_original(items).key == "a"

    def test_unknown_dimensions_rank_lowest(self):
        items = [
            _item("known", width=50, height=50),
            _item("unknown"),
        ]
        assert best_original(items).key == "known"


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------


class TestExactDupes:
    def test_identical_bytes_collapse_to_single_cluster(self):
        p = _phash(_BASE)
        items = [
            _item("orig", sha256="sha1", phash=p, width=200, height=200),
            _item("copy", sha256="sha1", phash=p, width=200, height=200),
        ]
        clusters = cluster_images(items)
        assert len(clusters) == 1
        assert {m.key for m in clusters[0].members} == {"orig", "copy"}
        assert clusters[0].size == 2

    def test_exact_cluster_has_single_canonical_best(self):
        p = _phash(_BASE)
        clusters = cluster_images(
            [
                _item("a", sha256="sha1", phash=p, width=300, height=300),
                _item("b", sha256="sha1", phash=p, width=300, height=300),
                _item("c", sha256="sha1", phash=p, width=300, height=300),
            ]
        )
        assert len(clusters) == 1
        # Only the best member is flagged; other exact members are dupes.
        assert clusters[0].best.key == "a"
        assert len(clusters[0].members) == 3


class TestNearDupes:
    def test_near_phash_merges_into_one_cluster(self):
        near_b = _BASE ^ 0b11  # 2 bits differ -> near dup
        items = [
            _item("a", sha256="sha_a", phash=_phash(_BASE), width=200, height=200),
            _item("b", sha256="sha_b", phash=_phash(near_b), width=200, height=200),
        ]
        clusters = cluster_images(items)
        assert len(clusters) == 1
        assert {m.key for m in clusters[0].members} == {"a", "b"}

    def test_far_phash_stays_separate(self):
        # A and 5 differ in all 64 bits -> far beyond threshold.
        items = [
            _item("a", sha256="sha_a", phash=_phash(_BASE), width=200, height=200),
            _item("b", sha256="sha_b", phash=_phash(0x5555555555555555), width=200, height=200),
        ]
        clusters = cluster_images(items)
        assert len(clusters) == 2

    def test_threshold_boundary(self):
        # Exactly at the threshold -> merged; one over -> not merged.
        p1 = _phash(_BASE)
        p2 = _phash(_BASE ^ ((1 << PHASH_NEAR_THRESHOLD) - 1))  # distance == threshold
        p3 = _phash(_BASE ^ ((1 << (PHASH_NEAR_THRESHOLD + 1)) - 1))
        # p3 differs in (PHASH_NEAR_THRESHOLD + 1) bits -> just over threshold.
        assert hamming_distance(p1, p2) == PHASH_NEAR_THRESHOLD
        assert hamming_distance(p1, p3) == PHASH_NEAR_THRESHOLD + 1
        at_threshold = cluster_images(
            [_item("a", sha256="s1", phash=p1), _item("b", sha256="s2", phash=p2)]
        )
        over_threshold = cluster_images(
            [_item("a", sha256="s1", phash=p1), _item("b", sha256="s2", phash=p3)]
        )
        assert len(at_threshold) == 1
        assert len(over_threshold) == 2

    def test_transitive_chain_forms_one_cluster(self):
        p_a = _phash(_BASE)
        p_b = _phash(_BASE ^ 0b1)
        p_c = _phash(_BASE ^ 0b10)
        clusters = cluster_images(
            [
                _item("a", sha256="s1", phash=p_a),
                _item("b", sha256="s2", phash=p_b),
                _item("c", sha256="s3", phash=p_c),
            ]
        )
        assert len(clusters) == 1
        assert {m.key for m in clusters[0].members} == {"a", "b", "c"}


class TestResized:
    def test_resized_same_scene_clusters(self):
        # Same phash, same aspect ratio (1.0), different resolutions.
        clusters = cluster_images(
            [
                _item("thumb", sha256="s1", phash=_phash(_BASE), width=50, height=50),
                _item("full", sha256="s2", phash=_phash(_BASE), width=800, height=800),
            ]
        )
        assert len(clusters) == 1
        assert clusters[0].best.key == "full"

    def test_best_original_is_highest_resolution(self):
        p = _phash(_BASE)
        clusters = cluster_images(
            [
                _item("small", sha256="s1", phash=p, width=100, height=100),
                _item("medium", sha256="s2", phash=p, width=300, height=200),
                _item("large", sha256="s3", phash=p, width=600, height=400),
            ]
        )
        assert clusters[0].best.key == "large"
        assert clusters[0].best.width == 600 and clusters[0].best.height == 400


class TestCropAware:
    def test_moderate_crop_differs_ar_but_merges(self):
        p = _phash(_BASE)
        # 100x100 (AR 1.0) and 160x100 (AR 1.6): ratio 1.6 within tolerance.
        clusters = cluster_images(
            [
                _item("square", sha256="s1", phash=p, width=100, height=100),
                _item("crop", sha256="s2", phash=p, width=160, height=100),
            ]
        )
        assert len(clusters) == 1

    def test_extreme_aspect_gap_not_merged_even_when_near(self):
        p = _phash(_BASE)
        # 100x100 (AR 1.0) vs 100x10 (AR ~0.1): ratio ~10 > tolerance -> rejected.
        assert (100 / 10) / (100 / 100) > CROP_AR_TOLERANCE
        clusters = cluster_images(
            [
                _item("square", sha256="s1", phash=p, width=100, height=100),
                _item("strip", sha256="s2", phash=p, width=100, height=10),
            ]
        )
        assert len(clusters) == 2

    def test_unknown_dims_are_compatible(self):
        p = _phash(_BASE)
        clusters = cluster_images(
            [_item("a", sha256="s1", phash=p), _item("b", sha256="s2", phash=p)]
        )
        assert len(clusters) == 1


class TestNoPhash:
    def test_no_phash_still_exact_groups_by_sha(self):
        # Same sha256, no phash -> exact dupe cluster.
        clusters = cluster_images(
            [
                _item("a", sha256="sha1"),
                _item("b", sha256="sha1"),
            ]
        )
        assert len(clusters) == 1
        assert clusters[0].size == 2

    def test_no_phash_does_not_near_merge(self):
        # Distinct sha256 and no phash -> never near-merged.
        clusters = cluster_images(
            [
                _item("a", sha256="sha1", phash=_phash(_BASE)),
                _item("b", sha256="sha2"),
            ]
        )
        assert len(clusters) == 2


class TestDeterminismAndIds:
    def test_cluster_ids_content_derived(self):
        p = _phash(_BASE)
        clusters = cluster_images(
            [_item("only", sha256="deadbeef", phash=p, width=100, height=100)]
        )
        assert clusters[0].cluster_id.startswith("cl-")
        assert clusters[0].cluster_id.endswith("deadbeef")

    def test_output_sorted_and_stable(self):
        p1 = _phash(_BASE)
        p2 = _phash(0x5555555555555555)
        items = [
            _item("z", sha256="z", phash=p1, width=10, height=10),
            _item("a", sha256="a", phash=p2, width=10, height=10),
        ]
        first = cluster_images(items)
        second = cluster_images(items)
        assert [c.cluster_id for c in first] == [c.cluster_id for c in second]

    def test_empty_input_returns_empty(self):
        assert cluster_images([]) == []

    def test_metadata_passthrough_preserved(self):
        item = ImageItem(
            key="k", sha256="s", width=1, height=1, metadata={"path": "/x/y.jpg"}
        )
        clusters = cluster_images([item])
        assert clusters[0].members[0].metadata["path"] == "/x/y.jpg"


class TestMixed:
    def test_full_mix_counts(self):
        """Exact dupes + one near + one unrelated = 2 clusters.

        The near item one bit from the exact scene merges into the exact cluster
        (one cluster), the unrelated image is its own second cluster.
        """
        p = _phash(_BASE)
        near = _phash(_BASE ^ 0b1)
        far = _phash(0x5555555555555555)
        clusters = cluster_images(
            [
                _item("e1", sha256="s_exact", phash=p, width=100, height=100),
                _item("e2", sha256="s_exact", phash=p, width=100, height=100),
                _item("n", sha256="s_near", phash=near, width=100, height=100),
                _item("u", sha256="s_uniq", phash=far, width=100, height=100),
            ]
        )
        assert len(clusters) == 2
        by_key = {c.best.key: c for c in clusters}
        assert set(by_key) == {"e1", "u"}
        # e1 is the canonical best of the exact+near cluster.
        assert {m.key for m in by_key["e1"].members} == {"e1", "e2", "n"}
        assert {m.key for m in by_key["u"].members} == {"u"}
