"""Tests for the Art Direction Manifest (M002/S03)."""

from __future__ import annotations

import json

import pytest

from curator.artdirection.manifest import (
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
