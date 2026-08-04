"""Tests for the artifact validator / publish gate (M003/S02)."""

from __future__ import annotations

import json
from io import BytesIO

from PIL import Image

from curator.artdirection.manifest import (
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
    SourceRegion,
)
from curator.hashing import sha256_hex
from curator.render.renderer import DeterministicRenderer
from curator.render.validate import ArtifactValidator, ValidationCheck, ValidationReport

renderer = DeterministicRenderer()
validator = ArtifactValidator()

TARGET = (1920, 1080)


def make_source(width: int, height: int, color=(40, 80, 160)) -> tuple[str, bytes]:
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    return sha256_hex(data), data


def render(width: int = 4000, height: int = 3000) -> tuple[bytes, str]:
    sha, data = make_source(width, height)
    manifest = ArtDirectionManifest(
        sources=[sha],
        layout_treatment=LayoutTreatment.SINGLE_FULLBLEED,
        background=BackgroundSpec(),
    )
    result = renderer.render(manifest, {sha: data}, TARGET)
    return renderer.render_bytes(manifest, {sha: data}, TARGET), result.sha256


def in_bounds_region() -> SourceRegion:
    return SourceRegion(source_sha256="s", x=100, y=100, w=500, h=500)


def by_name(report: ValidationReport, name: str) -> ValidationCheck:
    return next(c for c in report.checks if c.name == name)


# -- accept -------------------------------------------------------------------


def test_accept_correct_artifact() -> None:
    payload, sha = render()
    report = validator.validate(
        payload, sha, TARGET, source_region=in_bounds_region()
    )
    assert report.publishable is True
    assert report.valid is True
    assert len(report.checks) == 6
    for check in report.checks:
        assert check.passed is True, (check.name, check.reason)


def test_accept_without_source_region() -> None:
    payload, sha = render()
    report = validator.validate(payload, sha, TARGET)
    assert report.publishable is True
    names = {c.name for c in report.checks}
    assert names == {"dimensions", "color_mode", "color_profile", "hash"}


# -- reject: wrong dimensions -------------------------------------------------


def test_reject_wrong_dimensions() -> None:
    payload, sha = render()
    report = validator.validate(payload, sha, (1921, 1080))
    check = by_name(report, "dimensions")
    assert check.passed is False
    assert "width 1920 != expected 1921" in check.reason
    assert report.publishable is False


def test_reject_wrong_height() -> None:
    payload, sha = render()
    report = validator.validate(payload, sha, (1920, 1079))
    check = by_name(report, "dimensions")
    assert check.passed is False
    assert "height 1080 != expected 1079" in check.reason


# -- reject: wrong color mode / profile ---------------------------------------


def test_reject_wrong_color_mode() -> None:
    payload, sha = render()
    report = validator.validate(payload, sha, TARGET, color_mode="RGBA")
    check = by_name(report, "color_mode")
    assert check.passed is False
    assert "color mode 'RGB' != expected 'RGBA'" in check.reason


def test_reject_non_srgb_profile() -> None:
    from PIL import ImageCms

    lab_bytes = ImageCms.core.profile_tobytes(ImageCms.createProfile("LAB"))
    img = Image.new("RGB", TARGET, (40, 80, 160))
    img.info["icc_profile"] = lab_bytes
    buf = BytesIO()
    img.save(buf, format="PNG")
    payload = buf.getvalue()
    sha = sha256_hex(payload)
    report = validator.validate(payload, sha, TARGET)
    check = by_name(report, "color_profile")
    assert check.passed is False
    assert "color profile is not 'sRGB'" in check.reason
    # mode is still RGB, so only the profile check trips.
    assert by_name(report, "color_mode").passed is True
    assert report.publishable is False


# -- reject: hash mismatch ----------------------------------------------------


def test_reject_hash_mismatch_on_tamper() -> None:
    payload, sha = render()
    tampered = payload + b"\x00"
    assert sha256_hex(tampered) != sha
    report = validator.validate(tampered, sha, TARGET)
    check = by_name(report, "hash")
    assert check.passed is False
    assert "sha256 mismatch" in check.reason
    assert report.publishable is False


def test_reject_wrong_expected_sha() -> None:
    payload, sha = render()
    report = validator.validate(payload, "0" * 64, TARGET)
    check = by_name(report, "hash")
    assert check.passed is False
    assert sha in check.reason


# -- reject: source region out of bounds --------------------------------------


def test_reject_negative_width_out_of_bounds() -> None:
    payload, sha = render()
    region = SourceRegion(source_sha256="s", x=100, y=100, w=-10, h=100)
    report = validator.validate(payload, sha, TARGET, source_region=region)
    check = by_name(report, "source_region")
    assert check.passed is False
    assert "source region out of bounds" in check.reason
    assert report.publishable is False


def test_reject_region_start_outside_target() -> None:
    payload, sha = render()
    region = SourceRegion(source_sha256="s", x=2000, y=100, w=100, h=100)
    report = validator.validate(payload, sha, TARGET, source_region=region)
    check = by_name(report, "source_region")
    assert check.passed is False
    assert "source region out of bounds" in check.reason


# -- reject: unintended crop --------------------------------------------------


def test_reject_unintended_crop() -> None:
    payload, sha = render()
    region = SourceRegion(source_sha256="s", x=1900, y=100, w=100, h=100)
    report = validator.validate(payload, sha, TARGET, source_region=region)
    check = by_name(report, "no_unintended_crop")
    assert check.passed is False
    assert "would crop outside target" in check.reason
    # x=1900 is in-bounds (<=1920) so the region check still passes.
    assert by_name(report, "source_region").passed is True
    assert report.publishable is False


# -- JSON round-trip ----------------------------------------------------------


def test_json_roundtrip() -> None:
    payload, sha = render()
    report = validator.validate(
        payload, sha, TARGET, source_region=in_bounds_region()
    )
    text = json.dumps(report.to_dict())
    loaded = json.loads(text)
    rebuilt = ValidationReport.from_dict(loaded)
    assert rebuilt == report
    assert rebuilt.publishable is True


def test_publishable_only_when_all_checks_pass() -> None:
    payload, sha = render()
    passing = validator.validate(payload, sha, TARGET)
    assert passing.publishable is True
    failing = validator.validate(payload, sha, (1921, 1080))
    assert failing.publishable is False
    assert failing.valid is False
    assert any(not c.passed for c in failing.checks)


def test_validation_check_roundtrip() -> None:
    check = ValidationCheck("hash", False, "sha256 mismatch")
    assert ValidationCheck.from_dict(json.loads(json.dumps(check.to_dict()))) == check


# -- T2: renderer integration -------------------------------------------------


def test_render_to_validate_passes() -> None:
    payload, sha = render()
    report = validator.validate(
        payload, sha, TARGET, source_region=in_bounds_region()
    )
    assert report.publishable is True


def test_tampered_bytes_only_fail_hash() -> None:
    payload, sha = render()
    tampered = payload + b"\x01"
    report = validator.validate(tampered, sha, TARGET)
    for check in report.checks:
        if check.name == "hash":
            assert check.passed is False
        else:
            assert check.passed is True, (check.name, check.reason)


def test_validator_is_deterministic() -> None:
    payload, sha = render()
    a = validator.validate(payload, sha, TARGET, source_region=in_bounds_region())
    b = validator.validate(payload, sha, TARGET, source_region=in_bounds_region())
    assert a == b
    assert a.to_dict() == b.to_dict()
