"""Tests for the analysis pipeline + ``analysis_results`` persistence (M002/S02).

Covers: analyzing a fully-ingested fixture where every entry gets an ``ok`` row;
history-preserving append on re-run with byte-identical ``analysis_json`` per
entry; corrupt/non-image files recorded as ``corrupt`` rows with a reason; an
unexpected provider exception counted as ``error`` (run never aborts); and the
``AnalysisRunReport`` JSON round-trip.

The full 50-file/30-cluster fixture covers decodable determinism + history;
the corrupt/round-trip cases use a small in-test fixture (2 JPEGs + 1 garbage
file) because the fixture library's corrupt file is never cataloged by ingest
(so it has no entry id to analyze against).
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from curator.analysis.pipeline import (
    AnalysisAsset,
    AnalysisPipeline,
    AnalysisRunReport,
)
from curator.analysis.schema import AnalysisResult
from curator.catalog import Catalog
from curator.connectors import LocalConnector
from curator.ingest.pipeline import IngestPipeline
from fixture_library import INDEXED_FILES, build_fixture


def _solid(path: Path, size=(32, 32), color=(60, 120, 200), fmt="JPEG") -> None:
    """Write a solid-color image to *path*."""
    Image.new("RGB", size, color).save(str(path), format=fmt)


def _entry_id(catalog: Catalog, connector_id: str, asset_id: str) -> int:
    """Return the catalog entry id for ``(connector_id, asset_id)``."""
    row = catalog.db.execute(
        "SELECT id FROM catalog_entries"
        " WHERE connector_id=? AND asset_id=?"
        " ORDER BY id DESC LIMIT 1",
        (connector_id, asset_id),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _ordered_ok_rows(catalog: Catalog) -> list[tuple[int, str]]:
    """Return ``[(catalog_entry_id, analysis_json), ...]`` for ok rows, by id."""
    rows = catalog.db.execute(
        "SELECT catalog_entry_id, analysis_json FROM analysis_results"
        " WHERE status='ok' ORDER BY id"
    ).fetchall()
    return [(int(r[0]), str(r[1])) for r in rows]


def _normalized_analysis(json_text: str) -> dict:
    """Return the analysis JSON with the wall-clock ``timing_ms`` removed."""
    data = json.loads(json_text)
    data.get("metadata", {}).pop("timing_ms", None)
    return data


# ---------------------------------------------------------------------------
# full 50-file / 30-cluster fixture: decodable determinism + history
# ---------------------------------------------------------------------------


class TestFullLibrary:
    def _ingest(self, tmp_path, data_root):
        build = build_fixture(tmp_path / "fixture")
        catalog = Catalog(data_root=data_root)
        report = IngestPipeline(LocalConnector(build.root), catalog=catalog).run()
        return build, catalog, report

    def _assets(self, catalog, connector_id) -> list[AnalysisAsset]:
        rows = catalog.db.execute(
            "SELECT id, asset_id FROM catalog_entries WHERE connector_id=? ORDER BY id",
            (connector_id,),
        ).fetchall()
        return [AnalysisAsset(entry_id=int(r[0]), source=str(r[1])) for r in rows]

    def test_every_decodable_asset_analyzed_ok(self, tmp_path, data_root):
        build, catalog, report = self._ingest(tmp_path, data_root)
        assets = self._assets(catalog, report.connector_id)

        result = AnalysisPipeline(catalog).run(assets)

        assert build.total_files == 50
        assert result.total_assets == INDEXED_FILES == 47
        assert result.analyzed_count == 47
        assert result.corrupt_count == 0
        assert result.error_count == 0
        ok = _ordered_ok_rows(catalog)
        assert len(ok) == 47
        assert {rid for rid, _ in ok} == {a.entry_id for a in assets}

    def test_rerun_appends_history_and_json_is_deterministic(self, tmp_path, data_root):
        build, catalog, report = self._ingest(tmp_path, data_root)
        assets = self._assets(catalog, report.connector_id)

        AnalysisPipeline(catalog).run(assets)
        first_ok = _ordered_ok_rows(catalog)
        assert len(first_ok) == 47

        AnalysisPipeline(catalog).run(assets)
        second_ok = _ordered_ok_rows(catalog)

        # History preserved: the second run appends, never replaces (94 rows).
        assert len(second_ok) == 94
        assert [rid for rid, _ in second_ok] == [rid for rid, _ in first_ok] * 2
        # Determinism: every analytical signal is byte-identical across runs
        # (only wall-clock `metadata.timing_ms` varies, so it is normalized away).
        first_norm = [_normalized_analysis(js) for _, js in first_ok]
        second_norm = [_normalized_analysis(js) for _, js in second_ok]
        assert second_norm[:47] == first_norm
        assert second_norm[47:] == first_norm


# ---------------------------------------------------------------------------
# small in-test fixture: corrupt visibility + report round-trip
# ---------------------------------------------------------------------------


class TestSmallFixture:
    def _build(self, tmp_path, data_root):
        folder = tmp_path / "small"
        folder.mkdir()
        g1 = folder / "good1.jpg"
        g2 = folder / "good2.png"
        garbage = folder / "garbage.bin"
        _solid(g1)
        _solid(g2, color=(20, 180, 90), fmt="PNG")
        garbage.write_bytes(b"this is not an image at all")

        catalog = Catalog(data_root=data_root)
        conn = "smallconn"
        for path in (g1, g2, garbage):
            catalog.add_source(conn, str(path.resolve()), path.read_bytes())
        assets = [
            AnalysisAsset(_entry_id(catalog, conn, str(p.resolve())), str(p.resolve()))
            for p in (g1, g2, garbage)
        ]
        return catalog, assets

    def test_corrupt_non_image_recorded_with_reason(self, tmp_path, data_root):
        catalog, assets = self._build(tmp_path, data_root)

        report = AnalysisPipeline(catalog).run(assets)

        assert report.total_assets == 3
        assert report.analyzed_count == 2
        assert report.corrupt_count == 1
        assert report.error_count == 0

        corrupt = catalog.db.execute(
            "SELECT status, corrupt_reason FROM analysis_results"
            " WHERE status='corrupt'"
        ).fetchone()
        assert corrupt is not None
        assert corrupt[0] == "corrupt"
        assert corrupt[1] and "failed to decode" in corrupt[1]

        ok = catalog.db.execute(
            "SELECT COUNT(*) FROM analysis_results WHERE status='ok'"
        ).fetchone()
        assert int(ok[0]) == 2

    def test_report_json_round_trip(self, tmp_path, data_root):
        catalog, assets = self._build(tmp_path, data_root)

        report = AnalysisPipeline(catalog).run(assets)
        payload = report.to_dict()
        rebuilt = AnalysisRunReport.from_dict(payload)

        assert rebuilt.to_dict() == payload
        assert rebuilt.profile == "balanced"
        assert rebuilt.total_assets == 3
        assert rebuilt.analyzed_count == 2
        assert rebuilt.corrupt_count == 1
        assert len(payload["entries"]) == 3
        statuses = {e["status"] for e in payload["entries"]}
        assert statuses == {"ok", "corrupt"}


class _ErrorProvider:
    """Provider that raises an unexpected (non-AnalysisError) exception."""

    engine_version = "error-1.0.0"

    def analyze(self, source, profile, asset_id=None) -> AnalysisResult:
        del source, profile, asset_id
        raise RuntimeError("unexpected provider failure")


def test_unexpected_error_counts_and_never_aborts(tmp_path, data_root):
    """A non-AnalysisError exception is counted as an error and the run continues."""
    catalog = Catalog(data_root=data_root)
    g = tmp_path / "g.jpg"
    _solid(g)
    catalog.add_source("c", str(g.resolve()), g.read_bytes())
    entry_id = _entry_id(catalog, "c", str(g.resolve()))

    report = AnalysisPipeline(catalog, provider=_ErrorProvider()).run(
        [AnalysisAsset(entry_id=entry_id, source=str(g.resolve()))]
    )

    assert report.total_assets == 1
    assert report.analyzed_count == 0
    assert report.corrupt_count == 0
    assert report.error_count == 1
    assert report.entries[0].status == "error"
    assert "unexpected provider failure" in (report.entries[0].reason or "")
    # Errors leave no analysis_results row (only ok/corrupt are persisted).
    count = catalog.db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()
    assert int(count[0]) == 0
