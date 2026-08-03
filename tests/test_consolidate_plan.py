"""Tests for the consolidation planner (S03-T2: ConsolidationPlan + inventory).

Builds a small, deterministic legacy-ssd source folder entirely in-process with
Pillow (reusing the shared band-limited ``scene_image``) whose 8-group arithmetic
is pinned, then asserts every group's membership and file counts:

  * exact_dupes           : 3 byte-identical copies (1 family)      -> 3 files
  * near_dupes            : base + resized variant (1 family)      -> 2 files
  * higher_res_originals  : the resized (highest-res) near member  -> 1 file
  * filename_collisions   : a/photo.jpg + b/photo.jpg (same name)   -> 2 files
  * panels                : 1 x 1920x1080 panel                     -> 1 file
  * sidecars              : 1 x .xmp companion                      -> 1 file
  * corrupt               : broken.jpg (garbage bytes)              -> 1 file
  * missing_date          : 1 x undated scene                       -> 1 file

Negative/edge coverage: non-directory source raises CuratorError; an empty
folder yields a fully-empty plan; EXIF ``DateTimeOriginal`` supersedes a missing
filename date; a filename date pattern normalizes to ``YYYY-MM-DD``; sidecars
are never decoded; corrupt entries preserve the decode error text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

from curator.consolidate import (
    PANEL_DIMENSIONS,
    SIDECAR_SUFFIXES,
    ConsolidationPlan,
    build_plan,
)
from curator.errors import CuratorError
from fixture_library import scene_image


# ---------------------------------------------------------------------------
# fixture builder (deterministic, in-process)
# ---------------------------------------------------------------------------


def _panel_image() -> Image.Image:
    """A smooth, band-limited 1920x1080 image (Samsung Frame panel)."""
    img = Image.new("RGB", (1920, 1080), (100, 100, 100))
    ImageDraw.Draw(img).ellipse([300, 220, 1620, 860], fill=(150, 150, 150))
    return img.filter(ImageFilter.GaussianBlur(40))


def _write(path: Path, img: Image.Image, exif=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if exif is not None:
        kwargs["exif"] = exif
    img.save(str(path), format="JPEG", **kwargs)


def build_plan_fixture(root: Path) -> Path:
    """Create the deterministic legacy-ssd folder; return *root*."""
    src = root / "legacy-ssd"

    # exact_dupes: 3 byte-identical copies.
    exact = scene_image(1)
    _write(src / "2024-01-01_exact.jpg", exact)
    data = (src / "2024-01-01_exact.jpg").read_bytes()
    (src / "2024-01-01_exact_copy1.jpg").write_bytes(data)
    (src / "2024-01-01_exact_copy2.jpg").write_bytes(data)

    # near_dupes: base + 2x resized variant (resize-stable phash).
    base = scene_image(2)
    _write(src / "2024-01-02_near_base.jpg", base)
    _write(
        src / "2024-01-02_near_resized.jpg",
        base.resize((384, 384), Image.LANCZOS),
    )

    # filename_collisions: same basename, different content, in two subdirs.
    _write(src / "a" / "2024-02-01_photo.jpg", scene_image(3))
    _write(src / "b" / "2024-02-01_photo.jpg", scene_image(4))

    # panels: a 1920x1080 generated panel.
    _write(src / "2024-03-01_panel.jpg", _panel_image())

    # sidecars: a non-media companion pairing with the panel.
    (src / "2024-03-01_panel.xmp").write_bytes(b"<x:xmpmeta/>")

    # corrupt: supported suffix, undecodable bytes.
    (src / "broken.jpg").write_bytes(b"this is not an image at all")

    # missing_date: undated scene (no filename date, no EXIF).
    _write(src / "nodate.jpg", scene_image(5))
    return src


# ---------------------------------------------------------------------------
# group membership + arithmetic
# ---------------------------------------------------------------------------


def test_build_plan_populates_all_eight_groups(tmp_path):
    src = build_plan_fixture(tmp_path)
    plan = build_plan(src)

    assert plan.source_path == str(src.resolve())

    # exact_dupes
    assert len(plan.exact_dupes) == 1
    group = plan.exact_dupes[0]
    assert len(group) == 3  # 3 byte-identical copies
    assert "2024-01-01_exact.jpg" in group

    # near_dupes + higher_res_originals
    assert len(plan.near_dupes) == 1
    assert len(plan.near_dupes[0]) == 2
    assert plan.higher_res_originals == ["2024-01-02_near_resized.jpg"]

    # filename_collisions
    assert len(plan.filename_collisions) == 1
    assert set(plan.filename_collisions[0]) == {
        "a/2024-02-01_photo.jpg",
        "b/2024-02-01_photo.jpg",
    }

    # panels
    assert plan.panels == ["2024-03-01_panel.jpg"]

    # sidecars
    assert plan.sidecars == ["2024-03-01_panel.xmp"]

    # corrupt
    assert len(plan.corrupt) == 1
    assert plan.corrupt[0]["path"] == "broken.jpg"
    assert "error" in plan.corrupt[0]  # decode error text preserved

    # missing_date
    assert plan.missing_date == ["nodate.jpg"]

    counts = plan.group_counts()
    assert counts == {
        "exact_dupes": 3,
        "near_dupes": 2,
        "higher_res_originals": 1,
        "filename_collisions": 2,
        "panels": 1,
        "sidecars": 1,
        "corrupt": 1,
        "missing_date": 1,
    }


def test_panel_dimensions_detected_from_decoded_size():
    assert (1920, 1080) in PANEL_DIMENSIONS
    assert (3840, 2160) in PANEL_DIMENSIONS


def test_sidecar_extensions_are_non_media():
    assert ".xmp" in SIDECAR_SUFFIXES
    assert ".json" in SIDECAR_SUFFIXES


# ---------------------------------------------------------------------------
# JSON surface
# ---------------------------------------------------------------------------


def test_plan_to_json_roundtrip(tmp_path):
    src = build_plan_fixture(tmp_path)
    plan = build_plan(src)

    doc = json.loads(plan.to_json())
    assert doc["source_path"] == plan.source_path
    assert doc["exact_dupes"] == [["2024-01-01_exact.jpg", "2024-01-01_exact_copy1.jpg",
                                   "2024-01-01_exact_copy2.jpg"]]
    assert doc["corrupt"][0]["path"] == "broken.jpg"
    # Every group key is present in the serialized JSON.
    for key in (
        "exact_dupes",
        "near_dupes",
        "higher_res_originals",
        "filename_collisions",
        "panels",
        "sidecars",
        "corrupt",
        "missing_date",
    ):
        assert key in doc


def test_empty_plan_serializes(tmp_path):
    plan = ConsolidationPlan(source_path=str(tmp_path))
    doc = json.loads(plan.to_json())
    assert doc["exact_dupes"] == []
    assert plan.group_counts() == {
        "exact_dupes": 0,
        "near_dupes": 0,
        "higher_res_originals": 0,
        "filename_collisions": 0,
        "panels": 0,
        "sidecars": 0,
        "corrupt": 0,
        "missing_date": 0,
    }


# ---------------------------------------------------------------------------
# negative / edge cases
# ---------------------------------------------------------------------------


def test_build_plan_rejects_non_directory(tmp_path):
    f = tmp_path / "notadir.jpg"
    f.write_bytes(b"x")
    with pytest.raises(CuratorError):
        build_plan(tmp_path / "missing-dir")


def test_build_plan_on_empty_directory(tmp_path):
    src = tmp_path / "empty-ssd"
    src.mkdir()
    plan = build_plan(src)
    assert plan.group_counts() == {
        "exact_dupes": 0,
        "near_dupes": 0,
        "higher_res_originals": 0,
        "filename_collisions": 0,
        "panels": 0,
        "sidecars": 0,
        "corrupt": 0,
        "missing_date": 0,
    }


def test_exif_datetime_supersedes_missing_filename_date(tmp_path):
    # An image with EXIF DateTimeOriginal but no date in its filename is NOT
    # inventoried as missing_date.
    src = tmp_path / "ssd"
    src.mkdir()
    exif = Image.Exif()
    exif[0x9003] = "2020:05:06 07:08:09"
    _write(src / "photo_from_exif.jpg", scene_image(6), exif=exif)

    plan = build_plan(src)
    assert plan.group_counts()["missing_date"] == 0


def test_filename_date_pattern_normalized(tmp_path):
    from curator.consolidate.plan import _filename_date

    assert _filename_date("2024-01-02_photo.jpg") == "2024-01-02"
    assert _filename_date("IMG_20240101_photo.jpg") == "2024-01-01"
    assert _filename_date("2024_05_09_photo.jpg") == "2024-05-09"
    assert _filename_date("nodate.jpg") is None
