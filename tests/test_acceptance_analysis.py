"""Acceptance tests for the M002/S04 analysis / propose / manifest surface (T3+T4).

Each scenario is **self-bootstrapping**: it builds its own deterministic fixture
and drives the CLI in-process via :mod:`acceptance_harness` and the shared
``data_root`` fixture, never relying on cross-test ordering or external state.

This gate covers the analyze -> propose -> manifest lifecycle that lands in
T3/T4 of M002/S04:

* Scenario A1 — ``analyze`` maps the 50-file tree to the local engine, records an
  ``analysis_results`` row per decodable asset, and surfaces RAW/corrupt files as
  ``corrupt`` with an actionable reason.
* Scenario A2 — ``analyze`` is deterministic: two runs store identical analysis
  JSON for a given entry (ignoring timing) .
* Scenario A3 — profiles nest monotonically: ``fast < balanced < quality < max``
  in the per-asset pipeline stages observable from the stored analysis.
* Scenario A4 — ``propose`` returns a ranked, rationalised full-bleed for a
  crop-safe composition and a contain-matte for a crop-risky one; a single source
  never yields a diptych.
* Scenario A5 — ``manifest`` produces a versioned payload and honors the
  ``--target`` override (``resolved_for`` precedence).
* Scenario A6 — a corrupt file surfaces its reason in ``analyze --json``.

The real CPU engine caps aesthetic quality well below the policy's full-bleed
threshold, so the full-bleed path is exercised through the CLI's documented
"reuse persisted analysis" path (a deterministic, air-gapped synthetic row) while
the crop-risky path runs over a real constructed image via fresh analysis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from acceptance_harness import build_ingest_tree, run_cli
from analysis_factory import make_result
from curator.catalog import Catalog
from curator.connectors import LocalConnector
from fixture_library import CORRUPT_FILENAME, RAW_FILENAMES, TOTAL_FILES


def _force_catalog(tree: Path, names) -> str:
    """Catalog *names* (RAW + corrupt) under the tree's connector id.

    Ingest never catalogs RAW/corrupt files; force-cataloguing them under the
    same connector id the ``analyze`` command resolves lets them join a run and
    surface as ``corrupt`` (undecodable input).
    """
    conn = LocalConnector(tree).connector_id
    catalog = Catalog()
    try:
        for name in names:
            path = (tree / name).resolve()
            catalog.add_source(conn, str(path), path.read_bytes())
    finally:
        catalog.db.close()
    return conn


def _entry_id(catalog: Catalog, asset_id: str) -> int:
    row = catalog.db.execute(
        "SELECT id FROM catalog_entries WHERE asset_id = ? ORDER BY id DESC LIMIT 1",
        (asset_id,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def _populated_signal_count(analysis) -> int:
    """Count the policy-relevant signal groups populated in *analysis*."""
    count = 0
    w, h = analysis["saliency"]["map_size"]
    if w > 0 and h > 0:
        count += 1
    crop = analysis["crop_safety"]
    if any(crop[k] > 0 for k in ("margin_north", "margin_south", "margin_east", "margin_west")):
        count += 1
    color = analysis["color_story"]
    if color["colorfulness"] > 0 or color["dominant_colors"]:
        count += 1
    if analysis["pairing"]["affinity"] > 0:
        count += 1
    return count


# ---------------------------------------------------------------------------
# A1 — analyze over the full tree, corrupt visibility, per-asset rows
# ---------------------------------------------------------------------------


def test_scenario_a1_analyze_full_tree_and_corrupt(data_root, tmp_path):
    """analyze records every decodable asset and surfaces RAW/corrupt as corrupt."""
    tree = build_ingest_tree(tmp_path)
    rc, _out = run_cli(["ingest", str(tree)])
    assert rc == 0

    _force_catalog(tree, RAW_FILENAMES + (CORRUPT_FILENAME,))

    rc, out = run_cli(["analyze", str(tree), "--profile", "balanced", "--json"])
    assert rc == 1  # corrupt present -> EXIT_PARTIAL

    doc = json.loads(out)
    assert doc["profile"] == "balanced"
    assert doc["total_assets"] == TOTAL_FILES == 50
    assert doc["analyzed_count"] == 47
    assert doc["corrupt_count"] == 3
    assert doc["error_count"] == 0

    corrupt = [e for e in doc["entries"] if e["status"] == "corrupt"]
    assert len(corrupt) == 3
    assert all(e["reason"] for e in corrupt)
    assert any("decode" in e["reason"] for e in corrupt)

    # Every decodable asset has an analysis_results row.
    catalog = Catalog(data_root=data_root)
    try:
        rows = catalog.db.execute("SELECT COUNT(*) FROM analysis_results").fetchone()[0]
        assert rows == 50  # 47 ok + 3 corrupt
        # spot-check a decodable singleton has an ok row
        asset = str((tree / "single_00.jpg").resolve())
        eid = _entry_id(catalog, asset)
        ok = catalog.db.execute(
            "SELECT status FROM analysis_results WHERE catalog_entry_id = ?"
            " AND status='ok' LIMIT 1",
            (eid,),
        ).fetchone()
        assert ok is not None
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# A2 — determinism
# ---------------------------------------------------------------------------


def test_scenario_a2_analysis_determinism(data_root, tmp_path):
    """Two analyze runs store byte-identical analysis JSON (ignoring timing)."""
    tree = build_ingest_tree(tmp_path)
    assert run_cli(["ingest", str(tree)])[0] == 0

    rc, _ = run_cli(["analyze", str(tree), "--json"])
    assert rc == 0
    rc, _ = run_cli(["analyze", str(tree), "--json"])
    assert rc == 0

    catalog = Catalog(data_root=data_root)
    try:
        asset = str((tree / "single_00.jpg").resolve())
        eid = _entry_id(catalog, asset)
        rows = catalog.db.execute(
            "SELECT analysis_json FROM analysis_results WHERE catalog_entry_id = ?"
            " ORDER BY id",
            (eid,),
        ).fetchall()
        assert len(rows) == 2
        parsed = []
        for (text,) in rows:
            doc = json.loads(text)
            doc["metadata"]["timing_ms"] = 0.0
            parsed.append(doc)
        assert parsed[0] == parsed[1]
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# A3 — profile monotonicity
# ---------------------------------------------------------------------------


def test_scenario_a3_profile_monotonicity(data_root, tmp_path):
    """fast < balanced < quality < max in per-asset populated pipeline stages."""
    tree = build_ingest_tree(tmp_path)
    assert run_cli(["ingest", str(tree)])[0] == 0

    for profile in ("fast", "balanced", "quality", "max"):
        rc, _ = run_cli(["analyze", str(tree), "--profile", profile, "--json"])
        assert rc == 0

    catalog = Catalog(data_root=data_root)
    try:
        asset = str((tree / "single_00.jpg").resolve())
        eid = _entry_id(catalog, asset)
        counts: dict[str, int] = {}
        for profile in ("fast", "balanced", "quality", "max"):
            row = catalog.db.execute(
                "SELECT analysis_json FROM analysis_results WHERE catalog_entry_id = ?"
                " AND profile = ? AND status = 'ok' ORDER BY id DESC LIMIT 1",
                (eid, profile),
            ).fetchone()
            assert row is not None
            counts[profile] = _populated_signal_count(json.loads(row[0]))
        assert (
            counts["fast"]
            < counts["balanced"]
            < counts["quality"]
            < counts["max"]
        )
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# A4 — propose: contain-matte (crop-risky, fresh) + full-bleed (reuse path)
# ---------------------------------------------------------------------------


def _write_risky_folder(tmp_path) -> Path:
    """Build and ingest a small folder holding one crop-risky image."""
    src = tmp_path / "propose"
    src.mkdir()
    # Content hugs the top-left corner: south/east crop margins collapse.
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[0:190, 0:120] = 200
    Image.fromarray(arr).save(src / "risky.png")
    assert run_cli(["ingest", str(src)])[0] == 0
    return src


def test_scenario_a4_propose_contain_matte_single_source(data_root, tmp_path):
    """A crop-risky real image proposes contain-matte; no full-bleed or diptych."""
    src = _write_risky_folder(tmp_path)
    asset = str((src / "risky.png").resolve())

    rc, out = run_cli(["propose", asset, "--json"])
    assert rc == 0
    proposals = json.loads(out)
    assert proposals  # non-empty
    treatments = {p["treatment"] for p in proposals}
    assert "contain_matte" in treatments
    assert "single_fullbleed" not in treatments
    assert "diptych" not in treatments
    # ranked by score descending
    scores = [p["score"] for p in proposals]
    assert scores == sorted(scores, reverse=True)

    catalog = Catalog(data_root=data_root)
    try:
        eid = _entry_id(catalog, asset)
        rows = catalog.db.execute(
            "SELECT COUNT(*) FROM proposals WHERE catalog_entry_id = ?", (eid,)
        ).fetchone()[0]
        assert rows == len(proposals)  # append-only persistence
    finally:
        catalog.db.close()


def test_scenario_a4_propose_fullbleed_reuse(data_root, tmp_path):
    """Propose returns a rationalised full-bleed via the persisted-analysis path."""
    src = _write_risky_folder(tmp_path)
    asset = str((src / "risky.png").resolve())

    # Seed a crop-safe, high-aesthetic analysis row; propose reuses it.
    catalog = Catalog(data_root=data_root)
    try:
        eid = _entry_id(catalog, asset)
        safe = make_result("crop_safe")  # aesthetic 0.85, all margins 0.15
        catalog.db.execute(
            "INSERT INTO analysis_results"
            " (catalog_entry_id, profile, engine_version, analysis_json, status)"
            " VALUES (?, 'max', 'local-1.0.0', ?, 'ok')",
            (eid, json.dumps(safe.to_dict())),
        )
        catalog.db.commit()
    finally:
        catalog.db.close()

    rc, out = run_cli(["propose", asset, "--json"])
    assert rc == 0
    proposals = json.loads(out)
    treatments = {p["treatment"] for p in proposals}
    assert "single_fullbleed" in treatments
    fullbleed = [p for p in proposals if p["treatment"] == "single_fullbleed"][0]
    assert fullbleed["rationale"]
    assert "diptych" not in treatments  # single source never yields a diptych


# ---------------------------------------------------------------------------
# A5 — manifest: versioned payload + target override precedence
# ---------------------------------------------------------------------------


def test_scenario_a5_manifest_versioned_and_target_override(data_root, tmp_path):
    """manifest emits a versioned payload and honors --target via resolved_for."""
    src = _write_risky_folder(tmp_path)
    asset = str((src / "risky.png").resolve())

    catalog = Catalog(data_root=data_root)
    try:
        eid = _entry_id(catalog, asset)
        safe = make_result("crop_safe")
        catalog.db.execute(
            "INSERT INTO analysis_results"
            " (catalog_entry_id, profile, engine_version, analysis_json, status)"
            " VALUES (?, 'max', 'local-1.0.0', ?, 'ok')",
            (eid, json.dumps(safe.to_dict())),
        )
        catalog.db.commit()
        # A prior propose run persists the full-bleed proposal we manifest from.
    finally:
        catalog.db.close()
    rc, _ = run_cli(["propose", asset, "--json"])
    assert rc == 0

    rc, out = run_cli(["manifest", asset, "--json"])
    assert rc == 0
    base = json.loads(out)
    assert base["manifest_version"] == "1"
    assert base["sources"] == [asset]
    assert base["layout_treatment"] == "single_fullbleed"
    assert base["rationale"]
    assert base["processing_intent"]["upscale_warning"] is False

    rc, out = run_cli(["manifest", asset, "--target", "4k", "--json"])
    assert rc == 0
    four_k = json.loads(out)
    # Override precedence: the 4k override flips the flag; base is unchanged.
    assert four_k["processing_intent"]["upscale_warning"] is True
    assert (
        four_k["processing_intent"]["color_profile"]
        == base["processing_intent"]["color_profile"]
    )
    assert four_k["layout_treatment"] == base["layout_treatment"]

    catalog = Catalog(data_root=data_root)
    try:
        rows = catalog.db.execute(
            "SELECT COUNT(*) FROM art_direction_manifests WHERE catalog_entry_id = ?",
            (eid,),
        ).fetchone()[0]
        assert rows >= 1  # append-only; one row per manifest invocation
        version = catalog.db.execute(
            "SELECT manifest_version FROM art_direction_manifests"
            " WHERE catalog_entry_id = ?",
            (eid,),
        ).fetchone()[0]
        assert version == "1"
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# A6 — corrupt-signal visibility
# ---------------------------------------------------------------------------


def test_scenario_a6_corrupt_reason_visible(data_root, tmp_path):
    """A corrupt file surfaces its decode reason in analyze --json output."""
    tree = build_ingest_tree(tmp_path)
    assert run_cli(["ingest", str(tree)])[0] == 0
    _force_catalog(tree, (CORRUPT_FILENAME,))

    rc, out = run_cli(["analyze", str(tree), "--json"])
    assert rc == 1
    doc = json.loads(out)
    corrupt = [e for e in doc["entries"] if e["status"] == "corrupt"]
    assert len(corrupt) == 1
    assert corrupt[0]["reason"] and "decode" in corrupt[0]["reason"]
