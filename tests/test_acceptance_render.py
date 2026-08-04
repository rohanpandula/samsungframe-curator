"""Acceptance tests for the M003/S04 T3 acceptance gate (render/validate/review).

Each scenario is **self-bootstrapping**: it builds its own deterministic synthetic
fixture (Pillow images + a manifest JSON) and drives the CLI in-process via
:mod:`acceptance_harness`, never relying on cross-test ordering or external state.

This gate covers the publish/review acceptance surface:

* R1 — ``render`` is byte-deterministic for a given manifest/sources/target: two
  runs emit identical ``sha256``, and the target dims (1920x1080 then 3840x2160)
  plus sRGB/RGB colour metadata are exact.
* R2 — ``validate`` gates a rendered artifact against expected provenance:
  matching sha -> publishable (exit 0), a single tampered byte -> not publishable
  (exit 1, hash check failed).
* R3 — an unapproved upscale is blocked (exit 2) with an actionable R008 error.
* R4 — ``review`` lists pending entries, ``approve`` marks one approved, and
  ``undo`` flips the decision back — with churn recorded as history, never erased.
"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from PIL import Image, ImageDraw

from acceptance_harness import run_cli, sha256_file
from curator.approve import ApprovalService
from curator.artdirection.manifest import (
    ArtDirectionManifest,
    LayoutTreatment,
    ProcessingIntent,
    SourceRegion,
)
from curator.catalog import Catalog
from curator.cli import main as cli_main
from curator.hashing import sha256_hex
from curator.render.renderer import DeterministicRenderer


def _synthetic_image(path: Path, size: tuple[int, int] = (1000, 800)) -> None:
    """Write a deterministic non-trivial RGB PNG to *path*."""
    img = Image.new("RGB", size, (20, 40, 60))
    draw = ImageDraw.Draw(img)
    for i in range(0, size[0], 32):
        draw.line([(i, 0), (0, size[1])], fill=(i % 255, 128, 200))
        draw.line([(0, i), (size[0], size[1] - i)], fill=(200, i % 255, 80))
    img.save(path, format="PNG")


def _write_manifest(path: Path, sha: str, approved: bool) -> None:
    """Write a single-full-bleed manifest JSON referencing *sha* to *path*."""
    manifest = ArtDirectionManifest(
        sources=[sha],
        regions=[SourceRegion(source_sha256=sha)],
        layout_treatment=LayoutTreatment.SINGLE_FULLBLEED,
        processing_intent=ProcessingIntent(upscale_warning=approved),
    )
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")


def _seed_content(image_path: Path) -> str:
    """Store *image_path* bytes in the content store and return their SHA-256."""
    data = image_path.read_bytes()
    run_cli(["catalog", "add", str(image_path)])
    return sha256_hex(data)


def _run_cli(argv) -> tuple[int, str, str]:
    """Run ``curator.cli.main`` capturing stdout and stderr separately."""
    out_buf, err_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        rc = cli_main(list(argv))
    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------------------------------------------------------------------------
# R1 — render determinism (1080p + 4k) + exact dimensions + colour metadata
# ---------------------------------------------------------------------------


def test_render_determinism_1080p_and_4k(data_root, tmp_path):
    """Render is byte-deterministic and targets 1920x1080 then 3840x2160, sRGB/RGB."""
    image = tmp_path / "source.png"
    _synthetic_image(image)
    sha = _seed_content(image)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, sha, approved=True)  # upscale approved

    rc1, out1, err1 = _run_cli(["render", str(manifest), "--target", "1080p", "--json"])
    rc2, out2, _err2 = _run_cli(["render", str(manifest), "--target", "1080p", "--json"])
    assert (rc1, rc2) == (0, 0) and err1 == ""
    doc1, doc2 = json.loads(out1), json.loads(out2)
    # Byte determinism: identical sha256 across two independent renders.
    assert doc1["sha256"] == doc2["sha256"]
    assert doc1["sha256"] and len(doc1["sha256"]) == 64
    # Exact 1080p dimensions and colour metadata.
    assert (doc1["target_width"], doc1["target_height"]) == (1920, 1080)
    assert doc1["color_profile"] == "sRGB"
    assert doc1["color_mode"] == "RGB"

    rc3, out3, _err3 = _run_cli(["render", str(manifest), "--target", "4k", "--json"])
    rc4, out4, _err4 = _run_cli(["render", str(manifest), "--target", "4k", "--json"])
    assert (rc3, rc4) == (0, 0)
    doc3, doc4 = json.loads(out3), json.loads(out4)
    assert doc3["sha256"] == doc4["sha256"]
    assert (doc3["target_width"], doc3["target_height"]) == (3840, 2160)
    assert doc3["color_profile"] == "sRGB"
    assert doc3["color_mode"] == "RGB"


# ---------------------------------------------------------------------------
# R2 — validate publishable; a single tampered byte flips publishable -> exit 1
# ---------------------------------------------------------------------------


def _render_to_file(manifest: Path, sha: str, image: Path, target, out_path: Path):
    """Render *manifest* in-process to a PNG file (CLI reports a print summary only)."""
    renderer = DeterministicRenderer()
    payload = renderer.render_bytes(
        ArtDirectionManifest.from_dict(
            json.loads(manifest.read_text(encoding="utf-8"))
        ).resolved_for("1080p"),
        {sha: image.read_bytes()},
        target,
    )
    out_path.write_bytes(payload)
    return out_path


def test_validate_publishable_and_tamper(data_root, tmp_path):
    """A matching sha validates publishable (exit 0); tampering fails the hash."""
    image = tmp_path / "source.png"
    _synthetic_image(image)
    sha = _seed_content(image)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, sha, approved=True)

    artifact = _render_to_file(manifest, sha, image, (1920, 1080), tmp_path / "art.png")
    expected = sha256_file(artifact)

    rc, out, err = _run_cli(
        ["validate", str(artifact), "--expected-sha", expected, "--target", "1080p", "--json"]
    )
    assert rc == 0 and err == ""
    report = json.loads(out)
    assert report["publishable"] is True
    assert report["valid"] is True
    assert {c["name"]: c["passed"] for c in report["checks"]}.get("hash") is True

    # Tamper a single byte in the rendered artifact.
    tampered = artifact.read_bytes()
    tampered = tampered[:10] + bytes([tampered[10] ^ 0xFF]) + tampered[11:]
    artifact.write_bytes(tampered)

    rc2, out2, _err2 = _run_cli(
        ["validate", str(artifact), "--expected-sha", expected, "--target", "1080p", "--json"]
    )
    assert rc2 == 1
    report2 = json.loads(out2)
    assert report2["publishable"] is False
    hash_check = next(c for c in report2["checks"] if c["name"] == "hash")
    assert hash_check["passed"] is False
    assert "sha256 mismatch" in hash_check["reason"]


# ---------------------------------------------------------------------------
# R3 — unapproved upscale is blocked (exit 2) with an actionable R008 error
# ---------------------------------------------------------------------------


def test_render_unapproved_upscale_blocks(data_root, tmp_path):
    """A tiny source rendered to 4K without approval fails fatally with R008."""
    image = tmp_path / "tiny.png"
    _synthetic_image(image, (64, 64))
    sha = _seed_content(image)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, sha, approved=False)  # upscale NOT approved

    rc, out, err = _run_cli(["render", str(manifest), "--target", "4k", "--json"])
    assert rc == 2
    assert out == ""
    assert "upscale" in err.lower()
    assert "R008" in err


# ---------------------------------------------------------------------------
# R4 — review: pending -> approve -> undo flips the decision; history kept
# ---------------------------------------------------------------------------


def test_review_approve_undo(data_root, tmp_path):
    """review lists pending, approve marks approved, undo flips back, history intact."""
    src = tmp_path / "review"
    src.mkdir()
    asset_a = src / "a.png"
    asset_b = src / "b.png"
    _synthetic_image(asset_a, (200, 150))
    _synthetic_image(asset_b, (150, 200))
    assert run_cli(["ingest", str(src)])[0] == 0

    rc, out, _err = _run_cli(["review", "--json"])
    assert rc == 0
    entries = json.loads(out)
    a = next(e for e in entries if e["asset_id"] == str(asset_a.resolve()))
    assert a["decision"] is None  # pending

    rc, _out, _err = _run_cli(["review", "approve", str(asset_a.resolve())])
    assert rc == 0
    rc, out, _err = _run_cli(["review", "--status", "approved", "--json"])
    assert rc == 0
    approved = json.loads(out)
    assert any(e["asset_id"] == str(asset_a.resolve()) for e in approved)

    rc, _out, _err = _run_cli(["review", "undo", str(asset_a.resolve())])
    assert rc == 0
    rc, out, _err = _run_cli(["review", "--json"])
    assert rc == 0
    after = json.loads(out)
    a_after = next(e for e in after if e["asset_id"] == str(asset_a.resolve()))
    # Undo flips the decision (approved -> rejected), it is not erased to pending.
    assert a_after["decision"] == "rejected"
    rc, out, _err = _run_cli(["review", "--status", "approved", "--json"])
    assert not any(e["asset_id"] == str(asset_a.resolve()) for e in json.loads(out))

    # Churn is recorded as append-only history, never erased.
    catalog = Catalog(data_root=data_root)
    try:
        eid = a["entry_id"]
        events = ApprovalService(catalog).history(eid)
        assert len(events) == 2  # approve + undo, in order
        assert [e.decision.value.lower() for e in events] == ["approved", "rejected"]
    finally:
        catalog.db.close()
