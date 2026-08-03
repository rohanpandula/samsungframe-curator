"""Tests for the IngestPipeline orchestration (T04).

Covers the decode step (incl. HEIC per R003), the JSON-serializable report, the
unsupported/corrupt/error classification surface, idempotent re-ingest,
resumable journal checkpointing, and cluster/catalog write-through.

Images are generated in-process with Pillow into ``tmp_path`` fixtures (no
.gitignored paths, no network). Solid-color images are degenerate for phash
(distance 0 across colors), so 'far' content uses a checkerboard overlay whose
phash is ~31 bits from a solid field — reliably beyond the near threshold.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from curator.catalog import Catalog
from curator.connectors import LocalConnector
from curator.connectors.base import (
    AssetMetadata,
    ConnectorCapabilities,
    ConnectorHealth,
    RevisionObservation,
    SourceConnector,
)
from curator.errors import ConnectorError
from curator.hashing import sha256_hex
from curator.ingest.decode import DecodeError, decode_image
from curator.ingest.pipeline import IngestPipeline
from curator.ingest.report import IngestReport, ReportEntry, ReportIssue
from fixture_library import (
    CORRUPT_FILENAME,
    RAW_FILENAMES,
    TOTAL_CLUSTERS,
    TOTAL_FILES,
    build_fixture,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _solid(path: Path, size=(64, 64), color=(200, 50, 50), fmt="JPEG") -> None:
    """Write a solid-color image to *path*."""
    Image.new("RGB", size, color).save(str(path), format=fmt)


def _stripe(path: Path, color=(200, 50, 50), fmt="JPEG") -> None:
    """Solid image plus a small bright stripe (a near-edit of the base)."""
    img = Image.new("RGB", (64, 64), color)
    ImageDraw.Draw(img).rectangle([0, 0, 8, 64], fill=(255, 255, 255))
    img.save(str(path), format=fmt)


def _checker(path: Path, fmt="PNG") -> None:
    """Checkerboard on a red field — visually far from solid colors."""
    img = Image.new("RGB", (64, 64), (200, 50, 50))
    d = ImageDraw.Draw(img)
    c, bs = 8, 8
    for x in range(c):
        for y in range(c):
            if (x + y) % 2 == 0:
                d.rectangle([x * bs, y * bs, (x + 1) * bs, (y + 1) * bs], fill=(0, 0, 200))
    img.save(str(path), format=fmt)


def _build_folder(tmp_path: Path) -> Path:
    """Deterministic mixed fixture: exact dupes, resize, near-edit, far, RAW, corrupt."""
    folder = tmp_path / "media"
    folder.mkdir()
    _solid(folder / "red.jpg")           # base 64x64
    (folder / "copy1.jpg").write_bytes((folder / "red.jpg").read_bytes())
    (folder / "copy2.jpg").write_bytes((folder / "red.jpg").read_bytes())
    _solid(folder / "resized.jpg", size=(128, 128))  # same scene, bigger
    _stripe(folder / "edited.jpg")       # near-edit of base
    _checker(folder / "checker.png")     # far -> own cluster
    (folder / "photo.cr2").write_bytes(b"raw-payload-not-decodable")  # RAW
    (folder / "broken.jpg").write_bytes(b"this is not an image")       # corrupt
    return folder


class _StubConnector(SourceConnector):
    """In-memory connector that can force a read failure on one asset."""

    connector_id = "stub"
    capabilities = ConnectorCapabilities(
        supported_media_types=(".jpg",),
        original_stream=True,
    )

    def __init__(self, assets: dict[str, bytes], fail_on: set[str] | None = None) -> None:
        self._assets = assets
        self._fail_on = fail_on or set()

    def health(self) -> ConnectorHealth:
        return ConnectorHealth(healthy=True)

    def enumerate(self, cursor: str | None = None):
        for aid in sorted(self._assets):
            if cursor is not None and aid <= cursor:
                continue
            yield AssetMetadata(
                asset_id=aid, connector_id=self.connector_id, revision="r",
                media_type=".jpg", size_bytes=len(self._assets[aid]),
            )

    def read_original(self, asset_id: str) -> bytes:
        if asset_id in self._fail_on:
            raise ConnectorError(f"read failed for {asset_id}")
        return self._assets[asset_id]

    def revisions(self, asset_id: str):
        yield RevisionObservation(asset_id=asset_id, revision="r")


# ---------------------------------------------------------------------------
# decode step (R003)
# ---------------------------------------------------------------------------


class TestDecode:
    def test_decode_png_signature(self, tmp_path):
        p = tmp_path / "img.png"
        _solid(p, size=(40, 30))
        data = p.read_bytes()
        sig = decode_image(data)
        assert sig.sha256 == sha256_hex(data)
        assert (sig.width, sig.height) == (40, 30)
        assert len(sig.phash) == 16
        assert sig.phash == sig.phash.lower()

    def test_decode_heic(self, tmp_path):
        """HEIC decodes via pillow-heif (R003)."""
        from pillow_heif import from_pillow as heif_from_pillow

        img = Image.new("RGB", (48, 32), (30, 90, 160))
        heif = heif_from_pillow(img)
        buf = __import__("io").BytesIO()
        heif.save(buf, format="HEIF")
        sig = decode_image(buf.getvalue())
        assert (sig.width, sig.height) == (48, 32)
        assert len(sig.phash) == 16

    def test_decode_corrupt_raises_with_reason(self):
        with pytest.raises(DecodeError) as ei:
            decode_image(b"this is not an image at all")
        assert "failed to decode" in str(ei.value)

    def test_decode_empty_raises(self):
        with pytest.raises(DecodeError):
            decode_image(b"")


# ---------------------------------------------------------------------------
# IngestReport JSON serialization
# ---------------------------------------------------------------------------


class TestReportJson:
    def test_report_json_round_trip(self):
        rep = IngestReport(
            connector_id="c",
            total_enumerated=2,
            indexed_count=1,
            unique_clusters=1,
            exact_clusters=1,
            entries=[ReportEntry(asset_id="a", connector_id="c", sha256="s",
                                 cluster_id="cl-x", best_original=True)],
            failures=[ReportIssue(status="corrupt", asset_id="b", connector_id="c",
                                  media_type=".jpg", error="boom")],
        )
        payload = json.loads(rep.to_json())
        assert payload["unique_clusters"] == 1
        assert payload["entries"][0]["best_original"] is True
        assert payload["failures"][0]["error"] == "boom"
        assert payload["entries"][0]["cluster_id"] == "cl-x"


# ---------------------------------------------------------------------------
# IngestPipeline integration
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    def _run(self, folder: Path, data_root: Path) -> IngestReport:
        catalog = Catalog(data_root=data_root)
        pipeline = IngestPipeline(LocalConnector(folder), catalog=catalog)
        return pipeline.run()

    def test_full_fixture_classification(self, tmp_path, data_root):
        folder = _build_folder(tmp_path)
        report = self._run(folder, data_root)

        assert report.connector_id.startswith("local:")
        assert report.total_enumerated == 8
        # red+copy1+copy2+resized+edited+checker = 6 indexable; raw+broken excluded.
        assert report.indexed_count == 6
        assert report.unsupported_count == 1
        assert report.corrupt_count == 1
        assert report.error_count == 0

        # 6 indexable -> 2 clusters: base family (near) + checker (single).
        assert report.unique_clusters == 2
        assert report.near_clusters == 1  # red family merges resize+edit
        assert report.exact_clusters == 0  # no pure-dupe-only cluster here

        assert len(report.entries) == 6
        assert len(report.failures) == 2

        # RAW is explicit-unsupported, never silently dropped (R003).
        raw = next(f for f in report.failures if f.status == "unsupported")
        assert raw.media_type == ".cr2"
        assert raw.status == "unsupported"

        # Corrupt preserves the decode error text.
        corrupt = next(f for f in report.failures if f.status == "corrupt")
        assert corrupt.error and "failed to decode" in corrupt.error

    def test_cluster_members_best_original_and_distance(self, tmp_path, data_root):
        folder = _build_folder(tmp_path)
        report = self._run(folder, data_root)

        family = next(
            c for c in _by_cluster(report.entries).values() if len(c) > 1
        )
        best = next(e for e in family if e.best_original)
        # Highest resolution member (resized 128x128) is the best-original.
        assert best.asset_id.endswith("resized.jpg")
        assert (best.width, best.height) == (128, 128)
        assert best.phash_distance == 0
        # Every dupe points at the best, and none is itself flagged best.
        for entry in family:
            if not entry.best_original:
                assert entry.dupe_of == best.asset_id
        # The far checker is its own single-member cluster.
        checker_entries = [
            e for e in report.entries if e.asset_id.endswith("checker.png")
        ]
        assert len(checker_entries) == 1
        assert checker_entries[0].best_original is True

    def test_idempotent_reingest(self, tmp_path, data_root):
        folder = _build_folder(tmp_path)
        catalog = Catalog(data_root=data_root)

        IngestPipeline(LocalConnector(folder), catalog=catalog).run()
        entries_after_first = _count(catalog, "catalog_entries")
        hashes_first = _content_hashes(catalog)

        IngestPipeline(LocalConnector(folder), catalog=catalog).run()
        entries_after_second = _count(catalog, "catalog_entries")
        hashes_second = _content_hashes(catalog)

        assert entries_after_first == 6
        assert entries_after_second == entries_after_first  # idempotent
        assert hashes_first == hashes_second  # byte-hashes unchanged

    def test_journal_status_transitions(self, tmp_path, data_root):
        folder = _build_folder(tmp_path)
        catalog = Catalog(data_root=data_root)
        IngestPipeline(LocalConnector(folder), catalog=catalog).run()

        rows = catalog.db.execute(
            "SELECT status, error FROM ingest_journal ORDER BY id"
        ).fetchall()
        statuses = [r[0] for r in rows]
        # 8 assets, one row each; each transitions in place started -> terminal.
        assert len(rows) == 8
        assert statuses.count("indexed") == 6
        assert statuses.count("unsupported") == 1
        assert statuses.count("corrupt") == 1
        assert statuses.count("error") == 0
        assert statuses.count("started") == 0  # all finished this run
        # Corrupt journal row preserves the decode error.
        corrupt_row = next(r for r in rows if r[0] == "corrupt")
        assert corrupt_row[1] and "failed to decode" in corrupt_row[1]

    def test_resume_skips_reedecode(self, tmp_path, data_root, monkeypatch):
        folder = _build_folder(tmp_path)
        catalog = Catalog(data_root=data_root)

        calls = {"n": 0}

        def counting_decode(data):
            calls["n"] += 1
            from curator.ingest.decode import decode_image as real

            return real(data)

        monkeypatch.setattr("curator.ingest.pipeline.decode_image", counting_decode)
        pipe = IngestPipeline(LocalConnector(folder), catalog=catalog)
        first = pipe.run()
        # 6 valid decodes + 1 corrupt attempt (broken.jpg) = 7 calls.
        assert calls["n"] == 7

        calls["n"] = 0
        second = pipe.run(resume=True)
        # 6 previously-indexed assets reused, not re-decoded; only the corrupt
        # file (never indexed) is retried -> 1 call.
        assert calls["n"] == 1
        assert second.unique_clusters == first.unique_clusters

    def test_error_path_read_failure(self, tmp_path, data_root):
        img_path = tmp_path / "ok.jpg"
        _solid(img_path)
        stub = _StubConnector(
            assets={"ok": img_path.read_bytes(), "bad": b"\x00\x01garbage"},
            fail_on={"bad"},
        )
        catalog = Catalog(data_root=data_root)
        report = IngestPipeline(stub, catalog=catalog).run()

        assert report.total_enumerated == 2
        assert report.indexed_count == 1
        assert report.error_count == 1
        err = next(f for f in report.failures if f.status == "error")
        assert "read failed for bad" in (err.error or "")


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------


def _by_cluster(entries: list[ReportEntry]) -> dict[str, list[ReportEntry]]:
    out: dict[str, list[ReportEntry]] = {}
    for e in entries:
        if e.cluster_id is not None:
            out.setdefault(e.cluster_id, []).append(e)
    return out


def _count(catalog: Catalog, table: str) -> int:
    row = catalog.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    return int(row[0])


def _content_hashes(catalog: Catalog) -> set[str]:
    rows = catalog.db.execute("SELECT sha256 FROM content ORDER BY sha256").fetchall()
    return {r[0] for r in rows}


# ---------------------------------------------------------------------------
# T05: full 50-file / 30-cluster fixture verification (S05 shared fixture)
# ---------------------------------------------------------------------------


class TestFiftyFileFixture:
    """End-to-end verification over the deterministic 50-file / 30-cluster fixture.

    This is the slice acceptance surface: the shared S05 fixture ingests to
    exactly 30 unique clusters with documented exact/near/RAW/corrupt counts and
    the ``SELECT COUNT(DISTINCT cluster_id) = 30`` contract.
    """

    def _report(self, tmp_path, data_root, resume: bool = False):
        from curator.connectors import LocalConnector

        build = build_fixture(tmp_path / "fixture")
        catalog = Catalog(data_root=data_root)
        report = IngestPipeline(LocalConnector(build.root), catalog=catalog).run(
            resume=resume
        )
        return report, build

    def test_ingests_50_files_into_30_clusters(self, tmp_path, data_root):
        report, build = self._report(tmp_path, data_root)

        # Documented arithmetic (fixture spec).
        assert build.total_files == TOTAL_FILES == 50
        assert report.total_enumerated == 50
        assert report.indexed_count == 47
        assert report.unique_clusters == TOTAL_CLUSTERS == 30
        assert report.exact_clusters == 5
        assert report.near_clusters == 8  # 5 resize + 3 near families
        assert report.unsupported_count == 2
        assert report.corrupt_count == 1
        assert report.error_count == 0
        assert len(report.entries) == 47
        assert len(report.failures) == 3

    def test_acceptance_distinct_cluster_count_equals_30(self, tmp_path, data_root):
        report, build = self._report(tmp_path, data_root)
        # The acceptance: unique entries are distinct dedup clusters.
        from curator.catalog import Catalog

        catalog = Catalog(data_root=data_root)
        distinct = catalog.count_unique_clusters()
        assert distinct == 30
        row = catalog.db.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM catalog_entries"
            " WHERE cluster_id IS NOT NULL"
        ).fetchone()
        assert int(row[0]) == 30

    def test_cluster_sizes_match_fixture_families(self, tmp_path, data_root):
        report, _ = self._report(tmp_path, data_root)
        grouped = _by_cluster(report.entries)
        sizes = sorted(len(v) for v in grouped.values())
        # 4 exact triples (3,3,3,3) + 1 exact pair (2) + 5 resize pairs + 3 near
        # pairs + 17 singles.
        assert sizes == ([1] * 17 + [2] * 9 + [3] * 4)

    def test_raw_unsupported_and_corrupt_preserved(self, tmp_path, data_root):
        report, build = self._report(tmp_path, data_root)

        unsupported = [f for f in report.failures if f.status == "unsupported"]
        assert len(unsupported) == 2
        assert {Path(f.asset_id).name for f in unsupported} == set(RAW_FILENAMES)
        assert all(f.media_type in (".cr2", ".dng") for f in unsupported)

        corrupt = [f for f in report.failures if f.status == "corrupt"]
        assert len(corrupt) == 1
        assert Path(corrupt[0].asset_id).name == CORRUPT_FILENAME
        assert corrupt[0].error and "failed to decode" in corrupt[0].error

    def test_resize_best_original_is_highest_resolution(self, tmp_path, data_root):
        report, build = self._report(tmp_path, data_root)
        by_name = {Path(e.asset_id).name: e for e in report.entries}

        for resized in build.resize_best_originals:
            entry = by_name[resized]
            # The 2x resized member is the highest-resolution -> best_original.
            assert entry.best_original is True
            assert entry.phash_distance == 0
            assert (entry.width, entry.height) == (384, 384)
            # Its base sits in the same cluster but is NOT best-original.
            base = resized.replace("_resized", "_base")
            base_entry = by_name[base]
            assert base_entry.cluster_id == entry.cluster_id
            assert base_entry.best_original is False
            assert base_entry.dupe_of == entry.asset_id

    def test_every_cluster_has_exactly_one_best_original(self, tmp_path, data_root):
        report, _ = self._report(tmp_path, data_root)
        for cid, members in _by_cluster(report.entries).items():
            bests = [m for m in members if m.best_original]
            assert len(bests) == 1
            assert bests[0].dupe_of is None

    def test_reingest_idempotent_and_hashes_unchanged(self, tmp_path, data_root):
        from curator.connectors import LocalConnector

        build = build_fixture(tmp_path / "fixture")
        catalog = Catalog(data_root=data_root)
        connector = LocalConnector(build.root)

        IngestPipeline(connector, catalog=catalog).run()
        entries_first = _count(catalog, "catalog_entries")
        hashes_first = _content_hashes(catalog)

        IngestPipeline(connector, catalog=catalog).run()
        entries_second = _count(catalog, "catalog_entries")
        hashes_second = _content_hashes(catalog)

        assert entries_first == 47
        assert entries_second == entries_first  # idempotent re-ingest
        assert hashes_first == hashes_second  # content byte-hashes unchanged

    def test_journal_50_rows_terminal_statuses(self, tmp_path, data_root):
        report, _ = self._report(tmp_path, data_root)
        from curator.catalog import Catalog

        catalog = Catalog(data_root=data_root)
        rows = catalog.db.execute(
            "SELECT status, error FROM ingest_journal ORDER BY id"
        ).fetchall()
        assert len(rows) == 50
        statuses = [r[0] for r in rows]
        assert statuses.count("indexed") == 47
        assert statuses.count("unsupported") == 2
        assert statuses.count("corrupt") == 1
        assert statuses.count("started") == 0  # all terminal this run
        corrupt_row = next(r for r in rows if r[0] == "corrupt")
        assert corrupt_row[1] and "failed to decode" in corrupt_row[1]

    def test_report_is_json_serializable_over_full_fixture(self, tmp_path, data_root):
        report, _ = self._report(tmp_path, data_root)
        payload = json.loads(report.to_json())
        assert payload["unique_clusters"] == 30
        assert len(payload["entries"]) == 47
        assert len(payload["failures"]) == 3
