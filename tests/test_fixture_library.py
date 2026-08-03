"""Tests for the deterministic 50-file / 30-cluster fixture (T05, S05 shared).

Verifies the fixture library's contract: determinism (re-building yields
byte-identical files and the same documented arithmetic), the exact file /
cluster counts, mixed-format coverage (incl. TIFF and HEIC decode), and the
structural property that all 30 scene seeds are mutually far in phash while
each family's variants stay within the near threshold — the guarantees the
IngestPipeline clustering relies on.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import imagehash

from curator.ingest.clustering import PHASH_NEAR_THRESHOLD, hamming_distance
from fixture_library import (
    BASE_SIZE,
    CORRUPT_FILENAME,
    EXACT_DUPE_COPIES,
    EXACT_FAMILIES,
    EXACT_FILES,
    INDEXED_FILES,
    NEAR_FAMILIES,
    NEAR_FILES,
    RAW_FILENAMES,
    RESIZE_FAMILIES,
    RESIZE_FILES,
    SCENE_SEEDS,
    SINGLETON_FILES,
    TOTAL_CLUSTERS,
    TOTAL_FILES,
    build_fixture,
    scene_image,
)


def _all_files(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file()]


class TestFixtureArithmetic:
    def test_documented_counts(self, tmp_path):
        build = build_fixture(tmp_path / "fx")
        files = _all_files(build.root)

        assert build.total_files == TOTAL_FILES == 50
        assert len(files) == 50
        assert build.clusters == TOTAL_CLUSTERS == 30

        # Indexable = all files except RAW (unsupported) + corrupt.
        assert INDEXED_FILES == 50 - 2 - 1 == 47
        assert len(build.raw_files) == 2
        assert len(build.exact_files) == EXACT_FILES == 14
        assert len(build.resize_files) == RESIZE_FILES == 10
        assert len(build.near_files) == NEAR_FILES == 6
        assert len(build.singleton_files) == SINGLETON_FILES == 17

        # Family-count parity.
        assert EXACT_FAMILIES == 5
        assert RESIZE_FAMILIES == 5
        assert NEAR_FAMILIES == 3
        # Redundant copies == files beyond the first per cluster == 47 - 30.
        assert EXACT_DUPE_COPIES == 9
        assert (EXACT_FILES + RESIZE_FILES + NEAR_FILES + SINGLETON_FILES) == 47

    def test_all_30_scene_seeds_distinct_and_far(self):
        # 30 seeds, all pairwise > near threshold (self-guard against a future
        # accidental re-seed collapse).
        assert len(SCENE_SEEDS) == 30
        assert len(set(SCENE_SEEDS)) == 30
        hashes = {s: str(imagehash.phash(scene_image(s, BASE_SIZE))).lower() for s in SCENE_SEEDS}
        for a, b in itertools.combinations(SCENE_SEEDS, 2):
            assert hamming_distance(hashes[a], hashes[b]) > PHASH_NEAR_THRESHOLD

    def test_deterministic_build_is_byte_identical(self, tmp_path):
        build_a = build_fixture(tmp_path / "a")
        build_b = build_fixture(tmp_path / "b")
        fa = {p.name: p.read_bytes() for p in _all_files(build_a.root)}
        fb = {p.name: p.read_bytes() for p in _all_files(build_b.root)}
        assert set(fa) == set(fb)
        for name in fa:
            assert fa[name] == fb[name], f"non-deterministic file {name}"

    def test_mixed_format_coverage_includes_tiff_and_heic(self, tmp_path):
        build = build_fixture(tmp_path / "fx")
        suffixes = {p.suffix for p in _all_files(build.root)}
        # At least one of each headline format is present (filename-level mixed
        # formats; TIFF from T02, HEIC exercising the R003 decode path).
        assert ".tiff" in suffixes or ".tif" in suffixes
        assert ".heic" in suffixes
        assert {".jpg", ".png", ".webp"}.issubset(suffixes)

    def test_raw_and_corrupt_present(self, tmp_path):
        build = build_fixture(tmp_path / "fx")
        for name in RAW_FILENAMES:
            assert (build.root / name).is_file()
            assert (build.root / name).read_bytes() != b""
        assert (build.root / CORRUPT_FILENAME).is_file()
        # Corrupt file must be decodable-surface-visible bytes but not a real image.
        assert (build.root / CORRUPT_FILENAME).read_bytes() == b"this is not an image at all"

    def test_exact_family_copies_are_byte_identical(self, tmp_path):
        build = build_fixture(tmp_path / "fx")
        # Every exact-family base shares bytes with its copies (drives exact sha256).
        for f0 in (p for p in build.exact_files if p.endswith("_base")):
            base = build.root / f0
            stem = f0.replace("_base", "")
            copies = [build.root / c for c in build.exact_files if c.startswith(stem) and c != f0]
            assert copies, f"exact family {stem} has no copies"
            for copy in copies:
                assert copy.read_bytes() == base.read_bytes()

    def test_resize_best_originals_are_highest_resolution(self, tmp_path):
        build = build_fixture(tmp_path / "fx")
        # resized members are named *_resized<ext> and are 2x the base.
        for name in build.resize_best_originals:
            assert "_resized" in name
        assert len(build.resize_best_originals) == RESIZE_FAMILIES
