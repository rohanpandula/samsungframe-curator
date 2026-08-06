"""Tests for the S04 headless CLI ``render`` and ``validate`` surfaces.

Proves ``curator render`` renders a media asset (or manifest) to a target
deterministically (identical sRGB PNG + SHA-256 across repeated runs, exact
dimensions for both ``1080p`` and ``4k``) and blocks an unapproved upscale as a
fatal ``RenderError`` (R008). Proves ``curator validate`` gates an artifact
against expected dimensions + SHA-256, exiting 0 when publishable, 1 when a
check (hash / dimensions) fails. Subcommands run in-process via
:func:`curator.cli.main` under the capsys pattern.
"""

from __future__ import annotations

import io
import json

from PIL import Image

from acceptance_harness import run_cli
from curator import cli
from curator.hashing import sha256_hex


def _save_png(path, width, height, color=(120, 60, 30)) -> bytes:
    """Write a solid RGB PNG of *width* x *height* to *path*; return its bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    data = buf.getvalue()
    path.write_bytes(data)
    return data


def _build_review_catalog(tmp_path):
    """Ingest a 2-image folder and return the two resolved asset paths."""
    folder = tmp_path / "review"
    folder.mkdir()
    a = folder / "a.png"
    b = folder / "b.png"
    _save_png(a, 800, 600, color=(10, 20, 30))
    _save_png(b, 800, 600, color=(40, 50, 60))
    assert run_cli(["ingest", str(folder)])[0] == 0
    return str(a.resolve()), str(b.resolve())


def _review_map(capsys) -> dict[str, str | None]:
    """Run ``review --json`` and return {asset_id: decision}."""
    assert cli.main(["review", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    return {r["asset_id"]: r["decision"] for r in doc}


def test_review_json_shows_pending_then_approved(tmp_path, capsys, data_root):
    """Before any decision entries are pending; approve flips one to approved."""
    asset_a, asset_b = _build_review_catalog(tmp_path)

    mapping = _review_map(capsys)
    assert mapping[asset_a] is None
    assert mapping[asset_b] is None

    assert cli.main(["review", "approve", asset_a]) == 0
    capsys.readouterr()

    mapping = _review_map(capsys)
    assert mapping[asset_a] == "approved"
    assert mapping[asset_b] is None


def test_review_undo_reverts_and_reject(tmp_path, capsys, data_root):
    """undo toggles the decision off (approved -> rejected); reject records rejected."""
    asset_a, _ = _build_review_catalog(tmp_path)

    assert cli.main(["review", "approve", asset_a]) == 0
    capsys.readouterr()
    assert _review_map(capsys)[asset_a] == "approved"

    assert cli.main(["review", "undo", asset_a]) == 0
    capsys.readouterr()
    assert _review_map(capsys)[asset_a] == "rejected"

    assert cli.main(["review", "reject", asset_a]) == 0
    capsys.readouterr()
    assert _review_map(capsys)[asset_a] == "rejected"


def test_review_undo_back_to_approved(tmp_path, capsys, data_root):
    """undo of a reject flips the decision back to approved."""
    asset_a, _ = _build_review_catalog(tmp_path)

    cli.main(["review", "reject", asset_a])
    capsys.readouterr()
    assert _review_map(capsys)[asset_a] == "rejected"

    cli.main(["review", "undo", asset_a])
    capsys.readouterr()
    assert _review_map(capsys)[asset_a] == "approved"


def test_review_status_filter_and_batch(tmp_path, capsys, data_root):
    """--status returns only matching entries; --batch approves all at once."""
    asset_a, asset_b = _build_review_catalog(tmp_path)

    cli.main(["review", "approve", asset_a])
    capsys.readouterr()
    cli.main(["review", "reject", asset_b])
    capsys.readouterr()

    assert cli.main(["review", "--status", "approved", "--json"]) == 0
    approved = {r["asset_id"] for r in json.loads(capsys.readouterr().out)}
    assert approved == {asset_a}

    assert cli.main(["review", "--status", "rejected", "--json"]) == 0
    rejected = {r["asset_id"] for r in json.loads(capsys.readouterr().out)}
    assert rejected == {asset_b}

    assert cli.main(["review", "--status", "pending", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []

    assert cli.main(["review", "approve", "--batch", f"{asset_a},{asset_b}"]) == 0
    capsys.readouterr()
    mapping = _review_map(capsys)
    assert mapping[asset_a] == "approved"
    assert mapping[asset_b] == "approved"


def test_review_approve_unknown_asset_exits_fatal(tmp_path, capsys, data_root):
    """Reviewing an asset with no catalog entry is a fatal error (exit 2)."""
    _build_review_catalog(tmp_path)
    missing = str(tmp_path / "nope.png")

    rc = cli.main(["review", "approve", missing])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no catalog entry" in err


def test_render_deterministic_json(tmp_path, capsys):
    """Rendering the same asset twice yields identical dims, sha256 and sRGB."""
    source = tmp_path / "art.png"
    _save_png(source, 2000, 1500)

    outs = []
    for _ in range(2):
        assert cli.main(["render", "--json", str(source)]) == 0
        outs.append(json.loads(capsys.readouterr().out))

    first, second = outs
    assert first == second
    assert first["target_width"] == 1920
    assert first["target_height"] == 1080
    assert first["color_profile"] == "sRGB"
    assert first["color_mode"] == "RGB"
    assert first["sha256"] == second["sha256"]
    assert first["size_bytes"] == second["size_bytes"]


def test_render_1080p_and_4k_dims(tmp_path, capsys):
    """``1080p`` and ``4k`` targets map to the documented pixel dimensions."""
    source = tmp_path / "big.png"
    _save_png(source, 4000, 3000)

    assert cli.main(["render", "--json", "--target", "1080p", str(source)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert (doc["target_width"], doc["target_height"]) == (1920, 1080)

    assert cli.main(["render", "--json", "--target", "4k", str(source)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert (doc["target_width"], doc["target_height"]) == (3840, 2160)


def test_render_custom_wh_dims(tmp_path, capsys):
    """A ``WxH`` target parses to the exact requested dimensions."""
    source = tmp_path / "wide.png"
    _save_png(source, 3000, 1500)

    assert cli.main(["render", "--json", "--target", "2560x1440", str(source)]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert (doc["target_width"], doc["target_height"]) == (2560, 1440)


def test_render_unapproved_upscale_is_fatal(tmp_path, capsys):
    """Upscaling a tiny source into a large target without approval is exit 2."""
    source = tmp_path / "tiny.png"
    _save_png(source, 100, 100)

    rc = cli.main(["render", "--json", "--target", "4k", str(source)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "upscale" in err
    assert "R008" in err


def test_validate_correct_artifact_exits_zero(tmp_path, capsys):
    """A matching artifact is publishable and exits 0."""
    artifact = tmp_path / "out.png"
    data = _save_png(artifact, 1920, 1080)
    expected = sha256_hex(data)

    rc = cli.main(
        ["validate", "--json", str(artifact), "--expected-sha", expected, "--target", "1080p"]
    )
    assert rc == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["publishable"] is True
    assert doc["valid"] is True


def test_validate_tampered_bytes_not_publishable(tmp_path, capsys):
    """Tampered bytes fail the hash check, exit 1, publishable false."""
    artifact = tmp_path / "out.png"
    data = _save_png(artifact, 1920, 1080)
    expected = sha256_hex(data)

    flipped = bytearray(data)
    flipped[0] ^= 0xFF
    artifact.write_bytes(bytes(flipped))

    rc = cli.main(
        ["validate", "--json", str(artifact), "--expected-sha", expected, "--target", "1080p"]
    )
    assert rc == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["publishable"] is False
    hash_check = next(c for c in doc["checks"] if c["name"] == "hash")
    assert hash_check["passed"] is False
    assert "sha256 mismatch" in hash_check["reason"]


def test_validate_wrong_dimensions(tmp_path, capsys):
    """An off-dimension artifact fails the dimensions check, exit 1."""
    artifact = tmp_path / "out.png"
    data = _save_png(artifact, 1000, 1000)
    expected = sha256_hex(data)

    rc = cli.main(
        ["validate", "--json", str(artifact), "--expected-sha", expected, "--target", "1080p"]
    )
    assert rc == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["publishable"] is False
    dims_check = next(c for c in doc["checks"] if c["name"] == "dimensions")
    assert dims_check["passed"] is False
    assert "width" in dims_check["reason"]


# -- M010/S01: curator validate --manifest (the production region reader) -----


def _seed_diptych(tmp_path, count: int = 2, regions: str = "packed"):
    """Store *count* sources and write a manifest JSON; return (path, manifest, sources)."""
    from curator.artdirection.manifest import ArtDirectionManifest, LayoutTreatment
    from curator.artdirection.manifest import SourceRegion as Region
    from curator.artdirection.packing import Cell, equal_cells, gutter_for_target
    from curator.content_store import ContentStore

    store = ContentStore()
    sources = {}
    for index in range(count):
        data = _save_png(
            tmp_path / f"src{index}.png", 1600, 1200, color=(10 + index * 30, 60, 90)
        )
        sources[store.put(data)] = data
    shas = list(sources)
    target = (1920, 1080)
    if regions == "packed":
        cells = equal_cells(shas, Cell(0, 0, *target), gap=gutter_for_target(target))
    elif regions == "unset":
        cells = [Region(source_sha256=sha) for sha in shas]
    else:  # "short" — one region for many sources, the invariant Task 1 added
        cells = [Region(source_sha256=shas[0])]
    manifest = ArtDirectionManifest(
        sources=shas,
        regions=cells,
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=shas,
    )
    path = tmp_path / "diptych.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path, manifest, sources


def test_validate_manifest_round_trip_checks_every_cell(tmp_path, capsys, data_root):
    """Render a 2-source manifest, then validate that artifact cell by cell."""
    from curator.render.renderer import DeterministicRenderer

    manifest_path, manifest, sources = _seed_diptych(tmp_path)

    assert cli.main(["render", "--json", str(manifest_path), "--target", "1080p"]) == 0
    rendered = json.loads(capsys.readouterr().out)
    assert rendered["treatment"] == "diptych"

    artifact = tmp_path / "out.png"
    payload = DeterministicRenderer().render_bytes(manifest, sources, (1920, 1080))
    artifact.write_bytes(payload)
    assert sha256_hex(payload) == rendered["sha256"]

    rc = cli.main(
        [
            "validate",
            "--json",
            str(artifact),
            "--expected-sha",
            rendered["sha256"],
            "--target",
            "1080p",
            "--manifest",
            str(manifest_path),
        ]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    names = {c["name"] for c in report["checks"]}
    assert "source_region[0]" in names
    assert "source_region[1]" in names
    assert "no_unintended_crop[0]" in names
    assert "source_regions_disjoint" in names
    assert report["publishable"] is True


def test_validate_manifest_recomputes_legacy_all_zero_regions(
    tmp_path, capsys, data_root
):
    """An all-zero (unset) manifest still yields real per-cell checks."""
    from curator.render.renderer import DeterministicRenderer

    manifest_path, manifest, sources = _seed_diptych(tmp_path, regions="unset")
    payload = DeterministicRenderer().render_bytes(manifest, sources, (1920, 1080))
    artifact = tmp_path / "legacy.png"
    artifact.write_bytes(payload)

    rc = cli.main(
        [
            "validate",
            "--json",
            str(artifact),
            "--expected-sha",
            sha256_hex(payload),
            "--target",
            "1080p",
            "--manifest",
            str(manifest_path),
        ]
    )
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert {"source_region[0]", "source_region[1]"} <= {
        c["name"] for c in report["checks"]
    }


def test_validate_malformed_manifest_is_fatal(tmp_path, capsys, data_root):
    """Unreadable manifest JSON is a clean exit 2, never a traceback."""
    artifact = tmp_path / "out.png"
    data = _save_png(artifact, 1920, 1080)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")

    rc = cli.main(
        [
            "validate",
            str(artifact),
            "--expected-sha",
            sha256_hex(data),
            "--target",
            "1080p",
            "--manifest",
            str(bad),
        ]
    )
    assert rc == 2
    assert "failed to load manifest" in capsys.readouterr().err


def test_render_rejects_region_count_mismatch(tmp_path, capsys, data_root):
    """A hand-edited manifest with 2 sources and 1 region exits 2 (M010/S01)."""
    manifest_path, _, _ = _seed_diptych(tmp_path, regions="short")
    rc = cli.main(["render", "--json", str(manifest_path), "--target", "1080p"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "2 source(s) but 1 region(s)" in err
    assert "one region per source" in err


def test_render_rejects_over_cap_source_count(tmp_path, capsys, data_root):
    """A ten-source manifest is rejected by the cap, never truncated."""
    manifest_path, _, _ = _seed_diptych(tmp_path, count=10, regions="unset")
    rc = cli.main(["render", "--json", str(manifest_path), "--target", "1080p"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "9-source layout cap" in err
    assert "never truncated" in err


def test_validate_human_summary_lists_failing_checks(tmp_path, capsys):
    """Human validate output shows the publishable flag and failing-check reason."""
    artifact = tmp_path / "out.png"
    data = _save_png(artifact, 640, 480)
    expected = sha256_hex(data)

    rc = cli.main(
        ["validate", str(artifact), "--expected-sha", expected, "--target", "1080p"]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "publishable : false" in out
    assert "dimensions" in out
    assert "width" in out
