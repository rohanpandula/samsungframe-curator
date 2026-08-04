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
