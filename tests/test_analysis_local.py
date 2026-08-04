"""Tests for the local CPU signal engine provider (M002/S02).

Uses Pillow to synthesize deterministic images and verifies determinism, the
per-signal heuristics, error behavior, and full-result population of the
:class:`~curator.analysis.schema.AnalysisResult`.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from curator.analysis.errors import AnalysisError
from curator.analysis.local import LocalAnalysisProvider
from curator.analysis.profiles import AnalysisProfile

MAX = AnalysisProfile.MAX
CPU = LocalAnalysisProvider()


def _bytes(pil: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def _circle_image(width: int, height: int, radius: float, cx: float, cy: float) -> bytes:
    img = Image.new("RGB", (width, height), (20, 20, 20))
    d = ImageDraw.Draw(img)
    d.ellipse(
        [
            cx * width - radius,
            cy * height - radius,
            cx * width + radius,
            cy * height + radius,
        ],
        fill=(240, 240, 240),
    )
    return _bytes(img)


def test_deterministic_same_image() -> None:
    data = _circle_image(640, 360, 40, 0.5, 0.5)
    a = CPU.analyze(data, profile=MAX).to_dict()
    b = CPU.analyze(data, profile=MAX).to_dict()
    # Wall-clock timing_ms legitimately varies per run; every signal is identical.
    del a["metadata"]["timing_ms"], b["metadata"]["timing_ms"]
    assert a == b


def test_sharp_vs_blur() -> None:
    sharp = Image.fromarray(
        np.tile(
            np.array([[255, 0], [0, 255]], dtype=np.uint8),
            (200, 400),
        )
    ).convert("RGB")
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=6))
    sharp_res = CPU.analyze(_bytes(sharp), profile=MAX)
    blur_res = CPU.analyze(_bytes(blurred), profile=MAX)
    assert sharp_res.quality.sharpness > blur_res.quality.sharpness
    assert sharp_res.quality.technical_quality > blur_res.quality.technical_quality


def test_resolution_sufficiency() -> None:
    hd = Image.new("RGB", (1920, 1080), (120, 120, 120))
    tiny = Image.new("RGB", (100, 100), (120, 120, 120))
    hd_res = CPU.analyze(_bytes(hd), profile=MAX)
    tiny_res = CPU.analyze(_bytes(tiny), profile=MAX)
    assert hd_res.quality.resolution_sufficient is True
    assert tiny_res.quality.resolution_sufficient is False


def test_quality_values_in_range() -> None:
    res = CPU.analyze(_circle_image(800, 400, 60, 0.5, 0.5), profile=MAX)
    q = res.quality
    for value in (
        q.technical_quality,
        q.aesthetic_quality,
        q.sharpness,
        q.exposure,
        q.contrast,
        res.color_story.colorfulness,
    ):
        assert 0.0 <= value <= 1.0


def test_centered_subject_focal_near_center() -> None:
    res = CPU.analyze(_circle_image(640, 360, 45, 0.5, 0.5), profile=MAX)
    assert abs(res.saliency.focal_point.x - 0.5) < 0.1
    assert abs(res.saliency.focal_point.y - 0.5) < 0.1
    assert len(res.saliency.subjects) >= 1


def test_blank_image_low_saliency() -> None:
    blank = Image.new("RGB", (640, 360), (100, 100, 100))
    res = CPU.analyze(_bytes(blank), profile=MAX)
    assert len(res.saliency.subjects) == 0


def test_crop_safety_centered_safer_than_edge() -> None:
    centered = CPU.analyze(_circle_image(800, 400, 50, 0.5, 0.5), profile=MAX).crop_safety
    edge = CPU.analyze(_circle_image(800, 400, 50, 0.08, 0.5), profile=MAX).crop_safety
    centered_safe = sum(
        (centered.safe_north, centered.safe_south, centered.safe_east, centered.safe_west)
    )
    edge_safe = sum((edge.safe_north, edge.safe_south, edge.safe_east, edge.safe_west))
    assert centered_safe > edge_safe
    assert not edge.safe_west


def test_uniform_color_yields_single_dominant_and_neutral() -> None:
    uniform = Image.new("RGB", (320, 180), (128, 128, 128))
    res = CPU.analyze(_bytes(uniform), profile=MAX)
    dominant = [c for c in res.color_story.dominant_colors if c["ratio"] >= 0.05]
    assert len(dominant) == 1
    assert res.color_story.background_choice is not None


def test_corrupt_input_raises_analysis_error() -> None:
    with pytest.raises(AnalysisError):
        CPU.analyze(b"this is not an image at all", profile=MAX)


def test_full_result_populated() -> None:
    res = CPU.analyze(_circle_image(800, 400, 60, 0.5, 0.5), profile=MAX)
    assert res.metadata.engine_version
    assert res.metadata.deterministic is True
    assert res.perceptual.dim == 128 and res.perceptual.vector
    assert res.quality.sharpness > 0
    assert res.saliency.subjects
    assert res.crop_safety.safe_north
    assert res.color_story.dominant_colors
    assert res.pairing.affinity == 1.0
    assert res.metadata.model_spec["depth_available"] is False


def test_pairing_scores_between_analyzed_assets() -> None:
    img_a = _circle_image(640, 360, 45, 0.5, 0.5)
    img_b = _circle_image(640, 360, 60, 0.3, 0.4)
    res_a = CPU.analyze(img_a, profile=MAX, asset_id="pair-a")
    res_b = CPU.analyze(img_b, profile=MAX, asset_id="pair-b")
    pairing = CPU.pairing_scores_between(res_a, res_b)
    assert 0.0 <= pairing.affinity <= 1.0
    assert pairing.phash_distance is not None
    assert pairing.palette_distance is not None
    assert pairing.orientation_match is True
    assert pairing.affinity < 1.0
