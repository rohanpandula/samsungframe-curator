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
