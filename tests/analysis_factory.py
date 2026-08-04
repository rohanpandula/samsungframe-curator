"""Shared factory for fully/partially-populated AnalysisResult fixtures (M002/S01)."""

from __future__ import annotations

from curator.analysis.schema import (
    AnalysisMetadata,
    AnalysisResult,
    BoundingBox,
    ColorStory,
    CropSafety,
    Pairing,
    PerceptualRepresentation,
    Point,
    QualitySignals,
    Saliency,
)


def full_result(asset_id: str = "asset_001") -> AnalysisResult:
    """Build a fully-populated AnalysisResult covering every signal field."""
    return AnalysisResult(
        asset_id=asset_id,
        quality=QualitySignals(
            technical_quality=0.92,
            aesthetic_quality=0.87,
            sharpness=0.81,
            exposure=0.74,
            contrast=0.66,
            resolution_sufficient=True,
        ),
        saliency=Saliency(
            map_size=(64, 64),
            subjects=[BoundingBox(x=0.1, y=0.2, w=0.3, h=0.4)],
            focal_point=Point(x=0.25, y=0.3),
        ),
        crop_safety=CropSafety(
            safe_north=True,
            safe_south=False,
            safe_east=True,
            safe_west=True,
            margin_north=0.1,
            margin_south=0.02,
            margin_east=0.15,
            margin_west=0.12,
        ),
        color_story=ColorStory(
            dominant_colors=[{"hex": "#112233", "ratio": 0.5}],
            colorfulness=0.7,
            harmony=0.8,
            background_candidates=["#112233", "#ffffff"],
            background_choice="#ffffff",
        ),
        pairing=Pairing(
            affinity=0.9,
            phash_distance=3,
            palette_distance=0.2,
            date_proximity=1.0,
            orientation_match=True,
        ),
        perceptual=PerceptualRepresentation(
            method="phash", dim=64, vector=[0.1, 0.2, 0.3]
        ),
        metadata=AnalysisMetadata(
            profile="max",
            compute_backend="cpu",
            model_spec={"name": "ref-model", "version": "1.0.0"},
            engine_version="golden-1",
            deterministic=True,
            timing_ms=12.5,
        ),
    )


def make_result(
    asset_id: str = "asset",
    *,
    aesthetic_quality: float = 0.85,
    technical_quality: float = 0.9,
    resolution_sufficient: bool = True,
    safe_north: bool = True,
    safe_south: bool = True,
    safe_east: bool = True,
    safe_west: bool = True,
    margin_north: float = 0.15,
    margin_south: float = 0.15,
    margin_east: float = 0.15,
    margin_west: float = 0.15,
    map_size: tuple[int, int] = (1600, 1200),
    affinity: float = 1.0,
    background_choice: str | None = "#eeeeee",
) -> AnalysisResult:
    """Build a targeted AnalysisResult for policy tests.

    ``map_size`` drives the aspect ratio (width / height) that the policy uses
    for PANORAMIC / SQUARE gating, so callers control aspect via this field.
    Defaults are a crop-safe, well-margined, high-aesthetic singleton that the
    policy ranks as SINGLE_FULLBLEED.
    """
    return AnalysisResult(
        asset_id=asset_id,
        quality=QualitySignals(
            technical_quality=technical_quality,
            aesthetic_quality=aesthetic_quality,
            resolution_sufficient=resolution_sufficient,
        ),
        saliency=Saliency(map_size=map_size),
        crop_safety=CropSafety(
            safe_north=safe_north,
            safe_south=safe_south,
            safe_east=safe_east,
            safe_west=safe_west,
            margin_north=margin_north,
            margin_south=margin_south,
            margin_east=margin_east,
            margin_west=margin_west,
        ),
        color_story=ColorStory(background_choice=background_choice),
        pairing=Pairing(affinity=affinity),
    )


def crop_safe_result(asset_id: str = "safe_bb") -> AnalysisResult:
    """Crop-safe, well-margined, high-aesthetic composition -> SINGLE_FULLBLEED."""
    return make_result(asset_id=asset_id)


def crop_risky_result(
    asset_id: str = "risky", unsafe: str = "south"
) -> AnalysisResult:
    """A composition with one unsafe/low-margin crop direction -> CONTAIN_MATTE."""
    unsafe_dirs = {"north", "south", "east", "west"}
    if unsafe not in unsafe_dirs:
        raise ValueError(f"unsafe must be one of {sorted(unsafe_dirs)}")
    safe_flags = {d: True for d in unsafe_dirs}
    margins = {f"margin_{d}": 0.15 for d in unsafe_dirs}
    safe_flags[unsafe] = False
    margins[f"margin_{unsafe}"] = 0.01
    return make_result(
        asset_id=asset_id,
        safe_north=safe_flags["north"],
        safe_south=safe_flags["south"],
        safe_east=safe_flags["east"],
        safe_west=safe_flags["west"],
        margin_north=margins["margin_north"],
        margin_south=margins["margin_south"],
        margin_east=margins["margin_east"],
        margin_west=margins["margin_west"],
    )


def wide_result(asset_id: str = "wide") -> AnalysisResult:
    """Wide aspect (4.0) with modest aesthetics -> PANORAMIC (not fullbleed)."""
    return make_result(
        asset_id=asset_id, aesthetic_quality=0.5, map_size=(4000, 1000)
    )


def square_result(asset_id: str = "square") -> AnalysisResult:
    """Near-square aspect (1.0) with moderate aesthetics -> SQUARE."""
    return make_result(
        asset_id=asset_id, aesthetic_quality=0.6, map_size=(1000, 1000)
    )


def paired_result(asset_id: str = "pair", affinity: float = 0.9) -> AnalysisResult:
    """A result carrying a pairing affinity, for DIPTYCH gating tests."""
    return make_result(asset_id=asset_id, affinity=affinity)
