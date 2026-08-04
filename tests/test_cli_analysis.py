"""Tests for ``curator analyze`` CLI surface (M002/S04).

Proves the ``analyze`` command maps a folder's cataloged assets into the local
analysis engine and reports a machine-parseable :class:`AnalysisRunReport`
(``--json``) or a human-readable summary, honoring the exit-code contract:
0 when nothing was corrupt/errored, 1 on partial (corrupt/error), 2 for a
non-directory source. RAW/corrupt files the ingest pipeline never catalogs are
excluded from a plain run; force-cataloging them surfaces them as corrupt.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from analysis_factory import make_result
from curator import cli
from curator.artdirection.manifest import ArtDirectionManifest
from curator.catalog import Catalog
from curator.connectors import LocalConnector
from fixture_library import (
    CORRUPT_FILENAME,
    INDEXED_FILES,
    RAW_FILENAMES,
    TOTAL_FILES,
    build_fixture,
)


def test_analyze_ingest_tree_json(data_root, tmp_path, capsys):
    """Analyzing a fully-ingested folder reports cataloged assets, exit 0."""
    folder = build_fixture(tmp_path / "fixture").root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()  # drain ingest report

    rc = cli.main(["analyze", "--json", str(folder)])
    assert rc == 0

    doc = json.loads(capsys.readouterr().out)
    assert doc["profile"] == "balanced"
    assert doc["total_assets"] == INDEXED_FILES == 47
    assert doc["analyzed_count"] == 47
    assert doc["corrupt_count"] == 0
    assert doc["error_count"] == 0


def test_analyze_corrupt_and_raw_appear_corrupt(data_root, tmp_path, capsys):
    """Corrupt/RAW files that are cataloged surface as corrupt; exit 1 (partial)."""
    folder = build_fixture(tmp_path / "fixture").root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()  # drain ingest report

    # Ingest never catalogs RAW/corrupt files; force-catalog them under the same
    # connector id the analyze command resolves, so they join the run and are
    # recorded as corrupt (undecodable input).
    conn = LocalConnector(folder).connector_id
    catalog = Catalog()
    try:
        for name in RAW_FILENAMES + (CORRUPT_FILENAME,):
            path = folder / name
            catalog.add_source(conn, str(path.resolve()), path.read_bytes())
    finally:
        catalog.db.close()

    rc = cli.main(["analyze", "--json", str(folder)])
    assert rc == 1  # corrupt present -> EXIT_PARTIAL

    doc = json.loads(capsys.readouterr().out)
    assert doc["total_assets"] == TOTAL_FILES == 50
    assert doc["analyzed_count"] == 47
    assert doc["corrupt_count"] == 3
    assert doc["error_count"] == 0

    corrupt_ids = {
        entry["entry_id"]
        for entry in doc["entries"]
        if entry["status"] == "corrupt"
    }
    assert len(corrupt_ids) == 3


def test_analyze_fast_and_quality_profiles_accepted(data_root, tmp_path, capsys):
    """Both ``fast`` and ``quality`` profiles are accepted and echoed in the report."""
    folder = build_fixture(tmp_path / "fixture").root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()  # drain ingest report

    assert cli.main(["analyze", "--profile", "fast", "--json", str(folder)]) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == "fast"

    assert cli.main(["analyze", "--profile", "quality", "--json", str(folder)]) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == "quality"


def test_analyze_rejects_non_directory(data_root, tmp_path, capsys):
    """Analyzing a non-directory path is a fatal CuratorError (exit 2)."""
    rc = cli.main(["analyze", str(tmp_path / "no-such-folder")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# propose / manifest (M002/S04 T3)
# ---------------------------------------------------------------------------


def _seed_analysis(data_root, folder, asset, result):
    """Persist an ``ok`` analysis row for *asset* so propose/manifest reuse it."""
    conn = LocalConnector(folder).connector_id
    catalog = Catalog()
    try:
        row = catalog.db.execute(
            "SELECT id FROM catalog_entries WHERE connector_id = ? AND asset_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (conn, asset),
        ).fetchone()
        assert row is not None
        catalog.db.execute(
            "INSERT INTO analysis_results"
            " (catalog_entry_id, profile, engine_version, analysis_json, status)"
            " VALUES (?, 'max', 'local-1.0.0', ?, 'ok')",
            (row[0], json.dumps(result.to_dict())),
        )
        catalog.db.commit()
    finally:
        catalog.db.close()


def test_propose_returns_ranked_proposals(data_root, tmp_path, capsys):
    """Propose emits ranked proposals with rationale; a single source never diptych."""
    folder = build_fixture(tmp_path / "fixture").root
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    asset = str((folder / "single_00.jpg").resolve())

    rc = cli.main(["propose", asset, "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc
    scores = [p["score"] for p in doc]
    assert scores == sorted(scores, reverse=True)  # ranked
    assert all(p["rationale"] for p in doc)
    assert all(p["treatment"] != "diptych" for p in doc)


def test_propose_no_candidates_returns_no_change(data_root, tmp_path, capsys):
    """A low-signal, crop-safe, resolution-sufficient analysis yields no proposals."""
    folder = tmp_path / "flat"
    folder.mkdir()
    Image.new("RGB", (128, 128), (128, 128, 128)).save(folder / "flat.png")
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    asset = str((folder / "flat.png").resolve())

    # Seed an ok analysis that the policy finds nothing for: crop-safe and
    # resolution-sufficient (no contain-matte) but sub-threshold aesthetics and
    # a non-square, non-panoramic aspect (no full-bleed / square / panoramic).
    _seed_analysis(
        data_root,
        folder,
        asset,
        make_result(
            "flat",
            aesthetic_quality=0.4,
            resolution_sufficient=True,
            map_size=(1600, 1200),
        ),
    )

    rc = cli.main(["propose", asset, "--json"])
    assert rc == 3


def test_propose_fullbleed_via_persisted_analysis(data_root, tmp_path, capsys):
    """Propose reuses a persisted crop-safe analysis and yields a full-bleed."""
    folder = tmp_path / "propose"
    folder.mkdir()
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(folder / "img.png")
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    asset = str((folder / "img.png").resolve())

    _seed_analysis(data_root, folder, asset, make_result("crop_safe"))

    rc = cli.main(["propose", asset, "--json"])
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    treatments = {p["treatment"] for p in doc}
    assert "single_fullbleed" in treatments
    fullbleed = [p for p in doc if p["treatment"] == "single_fullbleed"][0]
    assert fullbleed["rationale"]
    assert "diptych" not in treatments


def test_propose_nonexistent_asset_is_fatal(tmp_path, capsys):
    """Proposing a file that is not in the catalog is a fatal error (exit 2)."""
    rc = cli.main(["propose", str(tmp_path / "missing.png"), "--json"])
    assert rc == 2


def test_manifest_roundtrip_and_override(data_root, tmp_path, capsys):
    """Manifest round-trips through ArtDirectionManifest and honors --target."""
    folder = tmp_path / "manifest"
    folder.mkdir()
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(folder / "img.png")
    assert cli.main(["ingest", str(folder)]) == 0
    capsys.readouterr()
    asset = str((folder / "img.png").resolve())

    _seed_analysis(data_root, folder, asset, make_result("crop_safe"))

    rc = cli.main(["manifest", asset, "--json"])
    assert rc == 0
    base = json.loads(capsys.readouterr().out)
    assert base["manifest_version"] == "1"
    assert base["sources"] == [asset]
    assert base["layout_treatment"] == "single_fullbleed"
    assert base["rationale"]
    assert base["processing_intent"]["upscale_warning"] is False
    # round-trips losslessly and validates
    manifest = ArtDirectionManifest.from_dict(base)
    assert manifest.validate() is None
    assert manifest.to_dict() == base

    rc = cli.main(["manifest", asset, "--target", "4k", "--json"])
    assert rc == 0
    four_k = json.loads(capsys.readouterr().out)
    assert four_k["layout_treatment"] == base["layout_treatment"]
    assert four_k["processing_intent"]["upscale_warning"] is True
    assert (
        four_k["processing_intent"]["color_profile"]
        == base["processing_intent"]["color_profile"]
    )


def test_manifest_nonexistent_asset_is_fatal(tmp_path, capsys):
    """Manifesting an uncataloged / missing asset is a fatal error (exit 2)."""
    rc = cli.main(["manifest", str(tmp_path / "missing.png"), "--json"])
    assert rc == 2
