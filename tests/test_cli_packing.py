"""Tests for the multi-asset ``propose``/``manifest`` CLI surface (M010/S02).

R047's automated proof: N-up layouts must be reachable from a non-API surface.
Each test drives the real :func:`curator.cli.main` entry point end to end —
``propose`` ranking a named template, ``manifest`` materializing its cells, and
``render`` rasterizing that manifest — because a per-layer unit test would stay
green while the pipeline as a whole remained a no-op. Diptych gets the identical
three-step treatment: until this slice it was proposable only through a raw API
call, which is exactly the reachability gap R047 exists to close.

Analysis is real (never a seeded fixture row): the sources are deliberate
near-kin frames, so ``LocalAnalysisProvider`` derives a genuine cross-image
affinity above the N-up threshold.

M010/S03 extends the same three-step treatment to ``packed`` at N=5 — the count
with no named template at all — and to ``propose --weights``, whose round trip is
the load-bearing case: ``manifest`` re-materializes from a proposal reloaded out
of SQLite, so a weight that did not survive persistence would silently produce
equal cells on the second command.
"""

from __future__ import annotations

import io
import json

from PIL import Image, ImageDraw

from curator import cli
from curator.artdirection.manifest import MAX_LAYOUT_SOURCES, ArtDirectionManifest

TARGET_1080P = (1920, 1080)
TARGET_4K = (3840, 2160)


def _kin_image(path, marker: int, width: int = 1600, height: int = 1200) -> None:
    """Write one frame of a near-kin set: same palette, distinct bytes.

    The shared background and subject give a real ``pairing.affinity`` above
    :data:`~curator.artdirection.policy.NUP_AFFINITY`; the per-frame marker makes
    each file's SHA-256 (and therefore its catalog identity) distinct.
    """
    img = Image.new("RGB", (width, height), (60, 90, 170))
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 150, 900, 900], fill=(210, 180, 60))
    draw.rectangle([10, 10, 20 + marker, 20], fill=(0, 0, 0))
    img.save(path)


def _kin_folder(tmp_path, count: int, name: str = "kin") -> list[str]:
    """Ingest *count* near-kin frames and return their resolved asset paths."""
    folder = tmp_path / name
    folder.mkdir()
    paths = []
    for index in range(count):
        asset = folder / f"frame{index}.png"
        _kin_image(asset, index)
        paths.append(str(asset.resolve()))
    assert cli.main(["ingest", str(folder)]) == 0
    return paths


def _uningested_files(tmp_path, count: int) -> list[str]:
    """Write *count* real files that are deliberately never cataloged."""
    folder = tmp_path / "uncataloged"
    folder.mkdir()
    paths = []
    for index in range(count):
        asset = folder / f"loose{index}.png"
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (index * 10, 40, 40)).save(buf, format="PNG")
        asset.write_bytes(buf.getvalue())
        paths.append(str(asset.resolve()))
    return paths


def _crop_risky_first_of_three(tmp_path) -> list[str]:
    """Three cataloged assets whose first is genuinely crop-risky (CR-01, real analysis).

    A bright block touching two edges of the primary frame is what makes
    ``LocalAnalysisProvider`` propose CONTAIN_MATTE for it — the other two frames
    are unrelated content, since only the primary's own crop safety decides a
    single-cell proposal.
    """
    folder = tmp_path / "crop_risky_trio"
    folder.mkdir()
    risky = folder / "frame0.png"
    img = Image.new("RGB", (1600, 1200), (20, 20, 20))
    ImageDraw.Draw(img).rectangle([0, 0, 900, 800], fill=(200, 200, 200))
    img.save(risky)
    paths = [str(risky.resolve())]
    for index in (1, 2):
        asset = folder / f"frame{index}.png"
        Image.new("RGB", (1600, 1200), (30 * index, 60, 90)).save(asset)
        paths.append(str(asset.resolve()))
    assert cli.main(["ingest", str(folder)]) == 0
    return paths


def _json_out(capsys):
    """Return the JSON document the last CLI command printed."""
    return json.loads(capsys.readouterr().out)


def _write_manifest(tmp_path, document, name: str = "m.json") -> str:
    """Persist a printed manifest document to disk for ``curator render``."""
    path = tmp_path / name
    path.write_text(json.dumps(document), encoding="utf-8")
    return str(path)


def _treatments(proposals) -> set[str]:
    return {proposal["treatment"] for proposal in proposals}


# -- propose: N assets in, a named template out -------------------------------


def test_propose_three_assets_ranks_a_triptych(data_root, tmp_path, capsys):
    """Three cataloged assets rank a triptych — N-up from a non-API surface."""
    assets = _kin_folder(tmp_path, 3)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--json"]) == 0
    proposals = _json_out(capsys)
    assert "triptych" in _treatments(proposals)

    triptych = next(p for p in proposals if p["treatment"] == "triptych")
    assert triptych["evidence"]["sources"] == 3
    assert triptych["evidence"]["affinity_source"] == "pairing.affinity"
    assert len(triptych["evidence"]["cells"]) == 3
    assert triptych["rationale"]


def test_propose_four_assets_ranks_a_quad(data_root, tmp_path, capsys):
    """Four cataloged assets rank a quad rather than a truncated pair."""
    assets = _kin_folder(tmp_path, 4)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--json"]) == 0
    proposals = _json_out(capsys)
    assert "quad" in _treatments(proposals)
    assert "triptych" not in _treatments(proposals)


def test_propose_single_asset_is_unchanged(data_root, tmp_path, capsys):
    """One asset still proposes only single-source treatments (no regression)."""
    assets = _kin_folder(tmp_path, 1)
    capsys.readouterr()

    assert cli.main(["propose", assets[0], "--json"]) == 0
    treatments = _treatments(_json_out(capsys))
    assert treatments
    assert treatments.isdisjoint({"diptych", "triptych", "quad"})


# -- manifest + render: the full multi-asset pipeline --------------------------


def test_manifest_and_render_triptych_end_to_end(data_root, tmp_path, capsys):
    """propose -> manifest -> render for three assets, all through the CLI."""
    assets = _kin_folder(tmp_path, 3)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--json"]) == 0
    capsys.readouterr()

    assert cli.main(["manifest", *assets, "--treatment", "triptych", "--json"]) == 0
    document = _json_out(capsys)
    assert document["layout_treatment"] == "triptych"
    assert document["sources"] == assets
    assert document["pairing_order"] == assets

    regions = document["regions"]
    assert len(regions) == 3
    assert all(r["w"] > 0 and r["h"] > 0 for r in regions)
    assert max(r["x"] + r["w"] for r in regions) == TARGET_1080P[0]
    assert max(r["y"] + r["h"] for r in regions) == TARGET_1080P[1]
    # The printed document is a real manifest, not a display shape.
    assert ArtDirectionManifest.from_dict(document).validate() is None

    manifest_path = _write_manifest(tmp_path, document)
    assert cli.main(["render", manifest_path, "--target", "1080p", "--json"]) == 0
    rendered = _json_out(capsys)
    assert rendered["treatment"] == "triptych"
    assert rendered["sources"] == assets
    assert (rendered["target_width"], rendered["target_height"]) == TARGET_1080P


def test_triptych_manifest_renders_at_4k(data_root, tmp_path, capsys):
    """The same manifest renders at 4K — cells are repacked for the target."""
    assets = _kin_folder(tmp_path, 3)
    capsys.readouterr()

    assert cli.main(["manifest", *assets, "--treatment", "triptych", "--json"]) == 0
    manifest_path = _write_manifest(tmp_path, _json_out(capsys))

    assert cli.main(["render", manifest_path, "--target", "4k", "--json"]) == 0
    rendered = _json_out(capsys)
    assert (rendered["target_width"], rendered["target_height"]) == TARGET_4K
    assert len(rendered["sources"]) == 3


def test_diptych_reachable_end_to_end_from_the_cli(data_root, tmp_path, capsys):
    """The pre-existing gap, closed: a diptych without a raw API call."""
    assets = _kin_folder(tmp_path, 2)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--json"]) == 0
    assert "diptych" in _treatments(_json_out(capsys))

    assert cli.main(["manifest", *assets, "--treatment", "diptych", "--json"]) == 0
    document = _json_out(capsys)
    assert document["layout_treatment"] == "diptych"
    assert len(document["regions"]) == 2
    assert all(region["w"] > 0 for region in document["regions"])

    manifest_path = _write_manifest(tmp_path, document, name="diptych.json")
    assert cli.main(["render", manifest_path, "--target", "1080p", "--json"]) == 0
    rendered = _json_out(capsys)
    assert rendered["treatment"] == "diptych"
    assert len(rendered["sources"]) == 2


# -- reject, never truncate ----------------------------------------------------


def test_propose_over_cap_assets_exits_fatal(data_root, tmp_path, capsys):
    """Ten assets exit 2 naming the cap — and cost zero catalog work.

    The files are deliberately never ingested: if the cap were checked after
    resolution the failure would be ``no catalog entry``, so the cap message is
    itself the proof that nothing was resolved or analyzed.
    """
    assets = _uningested_files(tmp_path, MAX_LAYOUT_SOURCES + 1)

    assert cli.main(["propose", *assets, "--json"]) == 2
    err = capsys.readouterr().err
    assert f"at most {MAX_LAYOUT_SOURCES} assets" in err
    assert "got 10" in err
    assert "never truncated" in err


def test_manifest_over_cap_assets_exits_fatal(data_root, tmp_path, capsys):
    """The same cap guards ``manifest`` before any catalog work."""
    assets = _uningested_files(tmp_path, MAX_LAYOUT_SOURCES + 1)

    assert cli.main(["manifest", *assets, "--json"]) == 2
    assert "never truncated" in capsys.readouterr().err


def test_manifest_rejects_a_template_that_cannot_lay_out_the_assets(
    data_root, tmp_path, capsys
):
    """Two assets can never be a triptych: exit 2, never a truncated render."""
    assets = _kin_folder(tmp_path, 2)
    capsys.readouterr()

    assert cli.main(["manifest", *assets, "--treatment", "triptych", "--json"]) == 2
    assert "no proposal available" in capsys.readouterr().err


def test_manifest_rejects_a_single_cell_treatment_for_three_assets(
    data_root, tmp_path, capsys
):
    """CR-01: forcing a single-cell treatment onto three assets used to silently
    fabricate real, tiled 3-cell geometry that CONTAIN_MATTE's renderer branch
    then ignored — rendering only ``frame0.png`` while ``curator validate``
    reported the manifest as a genuine three-way composition. It must now be
    rejected at selection time (``cli._lays_out``), before the materializer.
    """
    assets = _crop_risky_first_of_three(tmp_path)
    capsys.readouterr()

    assert (
        cli.main(["manifest", *assets, "--treatment", "contain_matte", "--json"]) == 2
    )
    err = capsys.readouterr().err
    assert "no proposal available" in err


# -- M010/S03: packed — arbitrary N, weighted, end to end ---------------------


def _packed(proposals) -> dict:
    return next(p for p in proposals if p["treatment"] == "packed")


def test_propose_five_assets_ranks_a_packed_layout(data_root, tmp_path, capsys):
    """Five assets have no named template — arbitrary N is reachable anyway."""
    assets = _kin_folder(tmp_path, 5)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--json"]) == 0
    proposals = _json_out(capsys)
    assert "packed" in _treatments(proposals)
    assert _treatments(proposals).isdisjoint({"triptych", "quad"})

    packed = _packed(proposals)
    assert packed["evidence"]["sources"] == 5
    assert len(packed["evidence"]["cells"]) == 5
    assert len(packed["evidence"]["weights"]) == 5
    assert packed["evidence"]["weight_source"] == "quality.aesthetic_quality"
    assert packed["rationale"]


def test_propose_weights_widen_the_first_cell(data_root, tmp_path, capsys):
    """The override is real: the same three assets, a visibly bigger first cell."""
    assets = _kin_folder(tmp_path, 3)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--json"]) == 0
    baseline = _packed(_json_out(capsys))
    assert baseline["evidence"]["weight_source"] == "quality.aesthetic_quality"

    assert cli.main(["propose", *assets, "--weights", "0.9,0.4,0.4", "--json"]) == 0
    weighted = _packed(_json_out(capsys))
    assert weighted["evidence"]["weight_source"] == "caller_override"
    assert weighted["evidence"]["weights"] == [0.9, 0.4, 0.4]
    assert weighted["evidence"]["cells"][0]["w"] > baseline["evidence"]["cells"][0]["w"]
    assert weighted["evidence"]["cells"][0]["w"] > weighted["evidence"]["cells"][1]["w"]


def test_propose_weights_count_mismatch_exits_fatal(data_root, tmp_path, capsys):
    """One weight per asset, or exit 2 — never padded, never truncated."""
    assets = _kin_folder(tmp_path, 3)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--weights", "0.9,0.4", "--json"]) == 2
    err = capsys.readouterr().err
    assert "2 value(s) for 3 asset(s)" in err


def test_propose_weights_non_numeric_exits_fatal(data_root, tmp_path, capsys):
    assets = _kin_folder(tmp_path, 3)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--weights", "0.9,huge,0.4", "--json"]) == 2
    assert "not a number" in capsys.readouterr().err


def test_manifest_and_render_packed_end_to_end(data_root, tmp_path, capsys):
    """propose -> manifest -> render at N=5, with the weights surviving SQLite."""
    assets = _kin_folder(tmp_path, 5)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--json"]) == 0
    cells = _packed(_json_out(capsys))["evidence"]["cells"]

    assert cli.main(["manifest", *assets, "--treatment", "packed", "--json"]) == 0
    document = _json_out(capsys)
    assert document["layout_treatment"] == "packed"
    assert document["sources"] == assets
    assert document["pairing_order"] == assets

    regions = document["regions"]
    assert len(regions) == 5
    # Re-materialized from the *persisted* proposal: identical geometry, which is
    # only true because the weights travelled in evidence rather than in a local.
    assert [(r["source_sha256"], r["x"], r["y"], r["w"], r["h"]) for r in regions] == [
        (c["sha"], c["x"], c["y"], c["w"], c["h"]) for c in cells
    ]
    assert max(r["x"] + r["w"] for r in regions) == TARGET_1080P[0]
    assert max(r["y"] + r["h"] for r in regions) == TARGET_1080P[1]
    assert ArtDirectionManifest.from_dict(document).validate() is None

    manifest_path = _write_manifest(tmp_path, document, name="packed.json")
    assert cli.main(["render", manifest_path, "--target", "1080p", "--json"]) == 0
    rendered = _json_out(capsys)
    assert rendered["treatment"] == "packed"
    assert len(rendered["sources"]) == 5


def test_packed_manifest_renders_at_4k(data_root, tmp_path, capsys):
    """The 1080p-materialized packed manifest fills a 4K canvas, not a quadrant."""
    assets = _kin_folder(tmp_path, 5)
    capsys.readouterr()

    assert cli.main(["manifest", *assets, "--treatment", "packed", "--json"]) == 0
    manifest_path = _write_manifest(tmp_path, _json_out(capsys), name="packed4k.json")

    assert cli.main(["render", manifest_path, "--target", "4k", "--json"]) == 0
    rendered = _json_out(capsys)
    assert (rendered["target_width"], rendered["target_height"]) == TARGET_4K
    assert len(rendered["sources"]) == 5


def test_manifest_of_one_asset_ignores_a_packed_proposal(data_root, tmp_path, capsys):
    """A packed proposal from a five-asset propose is not an answer to one asset."""
    assets = _kin_folder(tmp_path, 5)
    capsys.readouterr()

    assert cli.main(["propose", *assets, "--json"]) == 0
    capsys.readouterr()

    assert cli.main(["manifest", assets[0], "--json"]) == 0
    document = _json_out(capsys)
    assert document["layout_treatment"] != "packed"
    assert len(document["regions"]) == 1


def test_manifest_packed_skips_a_stale_smaller_proposal(data_root, tmp_path, capsys):
    """A packed row persisted at N=2 is not an answer to a later N=5 manifest.

    Repeated ``propose`` runs against the same primary asset each persist a
    ``packed`` row ("the primary source owns the row"), and ``_load_proposals``
    breaks a score tie toward the oldest id — so before the M010 review
    follow-up fix, ``manifest --treatment packed`` at N=5 selected the stale
    2-weight row and was rejected by ``materialize_manifest`` with a confusing
    count-mismatch error. Selection must instead skip any variable-width row
    whose own recorded weights cannot cover the current request
    (10-VERIFICATION.md, Additional Finding).
    """
    assets = _kin_folder(tmp_path, 5)
    capsys.readouterr()

    # Older, smaller propose against the same primary: persists a 2-weight
    # packed row with a lower rowid than anything persisted after it.
    assert cli.main(["propose", assets[0], assets[1], "--json"]) == 0
    assert "packed" in _treatments(_json_out(capsys))

    # Fresh five-asset propose: persists the 5-weight packed row this
    # manifest request actually needs.
    assert cli.main(["propose", *assets, "--json"]) == 0
    cells = _packed(_json_out(capsys))["evidence"]["cells"]

    assert cli.main(["manifest", *assets, "--treatment", "packed", "--json"]) == 0
    document = _json_out(capsys)
    assert document["layout_treatment"] == "packed"
    regions = document["regions"]
    assert len(regions) == 5
    # The fresh row was selected, not merely *a* row: geometry matches the
    # five-asset propose's own recorded cells exactly.
    assert [(r["source_sha256"], r["x"], r["y"], r["w"], r["h"]) for r in regions] == [
        (c["sha"], c["x"], c["y"], c["w"], c["h"]) for c in cells
    ]
