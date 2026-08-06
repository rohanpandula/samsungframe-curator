"""Tests for the Art Direction Manifest (M002/S03)."""

from __future__ import annotations

import json

import pytest

from curator.artdirection.manifest import (
    MAX_LAYOUT_SOURCES,
    MULTI_CELL_TREATMENTS,
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
    ManifestError,
    ProcessingIntent,
    SourceRegion,
)


def full_manifest() -> ArtDirectionManifest:
    return ArtDirectionManifest(
        sources=["abc123", "def456"],
        # These x/y/w/h values assert JSON round-tripping only; they make no claim
        # about the coordinate space, which is output-canvas pixels (M010/S01, N1).
        regions=[
            SourceRegion(source_sha256="abc123", x=0.1, y=0.2, w=0.5, h=0.4, crop="center"),
            SourceRegion(source_sha256="def456"),
        ],
        layout_treatment=LayoutTreatment.DIPTYCH,
        background=BackgroundSpec(
            background_choice="matte", color="#eeeeee", width=24, style="solid"
        ),
        processing_intent=ProcessingIntent(
            color_profile="display-p3", upscale_warning=True, notes=["notch it gently"]
        ),
        pairing_order=["abc123", "def456"],
        rationale=["strong pairing", "dates adjacent"],
        target_overrides={
            "4k": {"layout_treatment": "single_fullbleed", "background": {"color": "#000000"}},
        },
    )


def test_roundtrip_identity() -> None:
    m = full_manifest()
    assert ArtDirectionManifest.from_dict(m.to_dict()) == m


def test_json_serializable() -> None:
    m = full_manifest()
    text = json.dumps(m.to_dict())
    loaded = json.loads(text)
    assert ArtDirectionManifest.from_dict(loaded) == m


def test_missing_required_field_raises() -> None:
    with pytest.raises(ManifestError):
        ArtDirectionManifest(sources=[]).validate()


def test_invalid_treatment_raises() -> None:
    with pytest.raises(ManifestError):
        ArtDirectionManifest.from_dict({"layout_treatment": "tiled"})


def test_invalid_schema_version_raises() -> None:
    with pytest.raises(ManifestError):
        ArtDirectionManifest.from_dict({"manifest_version": "999"})


def test_override_precedence_resolves_target() -> None:
    m = full_manifest()
    resolved = m.resolved_for("4k")
    assert resolved.layout_treatment is LayoutTreatment.SINGLE_FULLBLEED
    assert resolved.background.color == "#000000"
    assert m.layout_treatment is LayoutTreatment.DIPTYCH
    assert m.background.color == "#eeeeee"


def test_override_partial_merge_keeps_base_key() -> None:
    m = full_manifest()
    resolved = m.resolved_for("4k")
    assert resolved.background.width == 24
    assert resolved.background.style == "solid"


def test_missing_target_returns_unchanged() -> None:
    m = full_manifest()
    assert m.resolved_for("nonexistent") is m


def test_unknown_override_field_raises() -> None:
    bad = ArtDirectionManifest(
        target_overrides={"4k": {"not_a_field": "x"}}
    )
    with pytest.raises(ManifestError):
        bad.resolved_for("4k")


def test_deterministic_to_dict() -> None:
    assert full_manifest().to_dict() == full_manifest().to_dict()


def test_empty_optional_fields_roundtrip() -> None:
    m = ArtDirectionManifest()
    rt = ArtDirectionManifest.from_dict(m.to_dict())
    assert rt == m
    assert rt.sources == []
    assert rt.pairing_order == []
    assert rt.target_overrides == {}


# -- M010/S01: coordinate space, the unset sentinel, and the N invariants ------


def test_region_count_mismatch_raises() -> None:
    """regions populated but shorter than sources is rejected (was silent)."""
    m = ArtDirectionManifest(
        sources=["a", "b", "c"], regions=[SourceRegion(source_sha256="a")]
    )
    with pytest.raises(ManifestError) as exc:
        m.validate()
    assert "one region per source" in str(exc.value)
    assert "3 source(s) but 1 region(s)" in str(exc.value)


def test_region_referencing_unknown_source_raises() -> None:
    """A region naming a sha absent from sources is rejected (was silent)."""
    m = ArtDirectionManifest(
        sources=["a"], regions=[SourceRegion(source_sha256="deadbeef")]
    )
    with pytest.raises(ManifestError) as exc:
        m.validate()
    assert "not in manifest" in str(exc.value)
    assert "deadbeef" in str(exc.value)


def test_over_cap_source_count_raises() -> None:
    """More than MAX_LAYOUT_SOURCES sources is rejected, never truncated."""
    m = ArtDirectionManifest(sources=[str(i) for i in range(MAX_LAYOUT_SOURCES + 1)])
    with pytest.raises(ManifestError) as exc:
        m.validate()
    assert str(MAX_LAYOUT_SOURCES) in str(exc.value)
    assert "never truncated" in str(exc.value)


def test_source_count_at_cap_validates() -> None:
    """Exactly MAX_LAYOUT_SOURCES sources is allowed; the cap is inclusive."""
    shas = [str(i) for i in range(MAX_LAYOUT_SOURCES)]
    ArtDirectionManifest(sources=shas).validate()
    assert len(shas) == 9


def test_all_zero_regions_still_validate() -> None:
    """Legacy manifests (every region all-zero == unset) keep validating."""
    shas = ["a", "b"]
    m = ArtDirectionManifest(
        sources=shas, regions=[SourceRegion(source_sha256=s) for s in shas]
    )
    m.validate()
    assert all(r.is_unset for r in m.regions)


def test_is_unset_distinguishes_declared_geometry() -> None:
    """All-zero means 'no geometry declared'; any non-zero extent means set."""
    assert SourceRegion(source_sha256="a").is_unset is True
    assert SourceRegion(source_sha256="a", w=10).is_unset is False
    assert SourceRegion(source_sha256="a", x=0, y=0, w=1920, h=1080).is_unset is False


def test_is_unset_is_not_serialized() -> None:
    """is_unset is a property, so to_dict / from_dict are untouched by it."""
    region = SourceRegion(source_sha256="a", x=0, y=0, w=1920, h=1080)
    data = region.to_dict()
    assert "is_unset" not in data
    assert SourceRegion.from_dict(data) == region


def test_multi_cell_treatments_names_diptych_today() -> None:
    """The single place 'does this treatment need >1 cell' is answered."""
    assert LayoutTreatment.DIPTYCH in MULTI_CELL_TREATMENTS
    assert LayoutTreatment.SINGLE_FULLBLEED not in MULTI_CELL_TREATMENTS
