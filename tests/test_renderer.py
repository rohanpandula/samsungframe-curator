"""Tests for the deterministic frame renderer (M003/S01, T1+T2)."""

from __future__ import annotations

import json
from io import BytesIO

import pytest
from PIL import Image

from curator.artdirection.manifest import (
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
    ProcessingIntent,
)
from curator.hashing import sha256_hex
from curator.render.renderer import (
    RENDERER_VERSION,
    DeterministicRenderer,
    RenderError,
    RenderResult,
)

renderer = DeterministicRenderer()

SRC_BLUE = (40, 80, 160)
MATTE_WHITE = "#f2f2f2"


def make_source(width: int, height: int, color=SRC_BLUE) -> tuple[str, bytes]:
    img = Image.new("RGB", (width, height), color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    return sha256_hex(data), data


def manifest(
    width: int,
    height: int,
    treatment: LayoutTreatment = LayoutTreatment.SINGLE_FULLBLEED,
    background: BackgroundSpec | None = None,
) -> tuple[ArtDirectionManifest, dict[str, bytes]]:
    sha, data = make_source(width, height)
    m = ArtDirectionManifest(
        sources=[sha],
        layout_treatment=treatment,
        background=background or BackgroundSpec(),
    )
    return m, {sha: data}


def open_rgb(png: bytes) -> Image.Image:
    with Image.open(BytesIO(png)) as img:
        return img.convert("RGB")


# -- determinism -------------------------------------------------------------


def test_byte_determinism() -> None:
    m, src = manifest(4000, 3000)
    a = renderer.render(m, src, (1920, 1080))
    b = renderer.render(m, src, (1920, 1080))
    assert a.sha256 == b.sha256
    assert renderer.render_bytes(m, src, (1920, 1080)) == renderer.render_bytes(
        m, src, (1920, 1080)
    )
    assert a.to_dict() == b.to_dict()


# -- dimensions / color ------------------------------------------------------


@pytest.mark.parametrize(
    "target,width,height",
    [((1920, 1080), 4000, 3000), ((3840, 2160), 6000, 4000)],
)
def test_exact_dims(target, width, height) -> None:
    m, src = manifest(width, height)
    r = renderer.render(m, src, target)
    assert (r.target_width, r.target_height) == target
    with Image.open(BytesIO(renderer.render_bytes(m, src, target))) as out:
        assert out.size == target


def test_image_size_matches_target() -> None:
    m, src = manifest(4000, 3000)
    r = renderer.render(m, src, (1920, 1080))
    assert (r.target_width, r.target_height) == (1920, 1080)
    with Image.open(BytesIO(renderer.render_bytes(m, src, (1920, 1080)))) as out:
        assert out.size == (1920, 1080)


def test_srgb_fields() -> None:
    m, src = manifest(4000, 3000)
    r = renderer.render(m, src, (1920, 1080))
    assert r.color_mode == "RGB"
    assert r.color_profile == "sRGB"
    assert r.renderer_version == RENDERER_VERSION


# -- SINGLE_FULLBLEED --------------------------------------------------------


def test_fullbleed_fills_target_no_letterbox() -> None:
    # A much taller-than-target source is center-cropped, not letterboxed.
    m, src = manifest(2000, 4000)
    r = renderer.render(m, src, (1920, 1080))
    img = open_rgb(renderer.render_bytes(m, src, (1920, 1080)))
    assert img.size == (1920, 1080)
    # Corners must be the (center-cropped) source fill color — no background edge.
    for xy in [(0, 0), (1919, 0), (0, 1079), (1919, 1079)]:
        assert img.getpixel(xy) == SRC_BLUE
    assert r.treatment == LayoutTreatment.SINGLE_FULLBLEED.value


# -- CONTAIN_MATTE -----------------------------------------------------------


def test_contain_matte_letterboxes_nonmatching_aspect() -> None:
    bg = BackgroundSpec(background_choice="matte", color=MATTE_WHITE, width=0)
    m, src = manifest(4000, 1000, treatment=LayoutTreatment.CONTAIN_MATTE, background=bg)
    r = renderer.render(m, src, (1920, 1080))
    img = open_rgb(renderer.render_bytes(m, src, (1920, 1080)))
    assert img.size == (1920, 1080)
    # Top/bottom edges are matte background (source is wide, letterboxed).
    matte = (242, 242, 242)
    assert img.getpixel((960, 0)) == matte
    assert img.getpixel((960, 1079)) == matte
    # The horizontal centerline is source fill color.
    assert img.getpixel((960, 540)) == SRC_BLUE
    assert r.treatment == LayoutTreatment.CONTAIN_MATTE.value


def test_contain_matte_default_background_black() -> None:
    m, src = manifest(4000, 1000, treatment=LayoutTreatment.CONTAIN_MATTE)
    img = open_rgb(renderer.render_bytes(m, src, (1920, 1080)))
    assert img.getpixel((960, 0)) == (0, 0, 0)


def test_contain_matte_is_deterministic() -> None:
    m, src = manifest(4000, 1000, treatment=LayoutTreatment.CONTAIN_MATTE)
    a = renderer.render(m, src, (1920, 1080))
    b = renderer.render(m, src, (1920, 1080))
    assert a.sha256 == b.sha256


# -- PANORAMIC / SQUARE ------------------------------------------------------


def test_panoramic_dims_and_determinism() -> None:
    m, src = manifest(4000, 1000, treatment=LayoutTreatment.PANORAMIC)
    r = renderer.render(m, src, (1920, 1080))
    assert (r.target_width, r.target_height) == (1920, 1080)
    a = renderer.render(m, src, (1920, 1080))
    b = renderer.render(m, src, (1920, 1080))
    assert a.sha256 == b.sha256
    assert r.treatment == LayoutTreatment.PANORAMIC.value


def test_square_layout_dims_and_determinism() -> None:
    # 1000px source into a 1080px square canvas upscales, so approve it.
    sha, data = make_source(1000, 1000)
    m = ArtDirectionManifest(
        sources=[sha],
        layout_treatment=LayoutTreatment.SQUARE,
        processing_intent=ProcessingIntent(upscale_warning=True),
    )
    r = renderer.render(m, {sha: data}, (1920, 1080))
    assert (r.target_width, r.target_height) == (1920, 1080)
    img = open_rgb(renderer.render_bytes(m, {sha: data}, (1920, 1080)))
    # Centered square canvas: black matte bars on the left/right.
    assert img.getpixel((0, 540)) == (0, 0, 0)
    assert img.getpixel((1919, 540)) == (0, 0, 0)
    a = renderer.render(m, {sha: data}, (1920, 1080))
    b = renderer.render(m, {sha: data}, (1920, 1080))
    assert a.sha256 == b.sha256
    assert r.treatment == LayoutTreatment.SQUARE.value


# -- JSON round-trip ---------------------------------------------------------


def test_json_roundtrip() -> None:
    m, src = manifest(4000, 3000)
    r = renderer.render(m, src, (1920, 1080))
    text = json.dumps(r.to_dict())
    loaded = json.loads(text)
    assert RenderResult.from_dict(loaded) == r


def test_from_dict_accepts_existing_instance() -> None:
    m, src = manifest(4000, 3000)
    r = renderer.render(m, src, (1920, 1080))
    assert RenderResult.from_dict(r) == r


# -- upscaled_warning --------------------------------------------------------


def test_upscaled_warning_defaults_false_on_downscale() -> None:
    m, src = manifest(6000, 4000)
    r = renderer.render(m, src, (1920, 1080))
    assert r.upscaled_warning is False


def test_upscaled_warning_true_on_approved_upscale() -> None:
    sha, data = make_source(800, 600)
    m = ArtDirectionManifest(
        sources=[sha],
        processing_intent=ProcessingIntent(upscale_warning=True),
    )
    r = renderer.render(m, {sha: data}, (1920, 1080))
    assert r.upscaled_warning is True


# -- T3/T4: arbitrary custom dims --------------------------------------------


@pytest.mark.parametrize(
    "target,width,height",
    [((1440, 900), 3000, 2000), ((2560, 1440), 3000, 2000)],
)
def test_custom_exact_dims(target, width, height) -> None:
    m, src = manifest(width, height)
    r = renderer.render(m, src, target)
    assert (r.target_width, r.target_height) == target
    with Image.open(BytesIO(renderer.render_bytes(m, src, target))) as out:
        assert out.size == target
    assert renderer.render(m, src, target).sha256 == r.sha256


# -- T3/T4: upscale invariant ------------------------------------------------


def test_upscale_without_approval_raises() -> None:
    m, src = manifest(800, 600)
    with pytest.raises(RenderError):
        renderer.render(m, src, (1920, 1080))


def test_upscale_with_approval_renders_exact() -> None:
    sha, data = make_source(800, 600)
    m = ArtDirectionManifest(
        sources=[sha],
        processing_intent=ProcessingIntent(upscale_warning=True),
    )
    r = renderer.render(m, {sha: data}, (1920, 1080))
    assert r.upscaled_warning is True
    img = open_rgb(renderer.render_bytes(m, {sha: data}, (1920, 1080)))
    assert img.size == (1920, 1080)


def test_large_source_to_small_target_no_upscale() -> None:
    m, src = manifest(6000, 4000)
    r = renderer.render(m, src, (1920, 1080))
    assert r.upscaled_warning is False


def test_diptych_upscale_without_approval_raises() -> None:
    sha_a, dat_a = make_source(800, 600)
    sha_b, dat_b = make_source(600, 800)
    m = ArtDirectionManifest(
        sources=[sha_a, sha_b],
        layout_treatment=LayoutTreatment.DIPTYCH,
    )
    with pytest.raises(RenderError):
        renderer.render(m, {sha_a: dat_a, sha_b: dat_b}, (1920, 1080))


def test_diptych_upscale_no_warning_when_downscaling() -> None:
    sha_a, dat_a = make_source(3000, 2000)
    sha_b, dat_b = make_source(2000, 3000)
    m = ArtDirectionManifest(
        sources=[sha_a, sha_b],
        layout_treatment=LayoutTreatment.DIPTYCH,
    )
    r = renderer.render(m, {sha_a: dat_a, sha_b: dat_b}, (1920, 1080))
    assert r.upscaled_warning is False


# -- T3/T4: diptych -----------------------------------------------------------


def _diptych_sources():
    sha_a, dat_a = make_source(1600, 1000, color=(160, 40, 40))
    sha_b, dat_b = make_source(1000, 1600, color=(40, 160, 40))
    return (sha_a, dat_a), (sha_b, dat_b)


def test_diptych_deterministic_exact_dims_and_order() -> None:
    (sha_a, dat_a), (sha_b, dat_b) = _diptych_sources()
    m = ArtDirectionManifest(
        sources=[sha_a, sha_b],
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=[sha_b, sha_a],
    )
    src = {sha_a: dat_a, sha_b: dat_b}
    r1 = renderer.render(m, src, (1920, 1080))
    r2 = renderer.render(m, src, (1920, 1080))
    assert r1.sha256 == r2.sha256
    assert r1.to_dict() == r2.to_dict()
    assert (r1.target_width, r1.target_height) == (1920, 1080)
    assert r1.sources == [sha_b, sha_a]
    assert r1.treatment == LayoutTreatment.DIPTYCH.value
    img = open_rgb(renderer.render_bytes(m, src, (1920, 1080)))
    assert img.size == (1920, 1080)


def test_diptych_side_by_side_two_panels() -> None:
    # Landscape target -> side-by-side: A (red) left, B (green) right.
    (sha_a, dat_a), (sha_b, dat_b) = _diptych_sources()
    m = ArtDirectionManifest(
        sources=[sha_a, sha_b],
        layout_treatment=LayoutTreatment.DIPTYCH,
    )
    src = {sha_a: dat_a, sha_b: dat_b}
    img = open_rgb(renderer.render_bytes(m, src, (1920, 1080)))
    assert img.getpixel((480, 540)) == (160, 40, 40)
    assert img.getpixel((1440, 540)) == (40, 160, 40)


def test_diptych_stacked_two_panels() -> None:
    # Portrait target -> stacked: A (red) top, B (green) bottom.
    (sha_a, dat_a), (sha_b, dat_b) = _diptych_sources()
    m = ArtDirectionManifest(
        sources=[sha_a, sha_b],
        layout_treatment=LayoutTreatment.DIPTYCH,
    )
    src = {sha_a: dat_a, sha_b: dat_b}
    img = open_rgb(renderer.render_bytes(m, src, (900, 1440)))
    assert img.size == (900, 1440)
    assert img.getpixel((450, 350)) == (160, 40, 40)
    assert img.getpixel((450, 1075)) == (40, 160, 40)


def test_diptych_falls_back_to_source_order() -> None:
    (sha_a, dat_a), (sha_b, dat_b) = _diptych_sources()
    m = ArtDirectionManifest(
        sources=[sha_a, sha_b],
        layout_treatment=LayoutTreatment.DIPTYCH,
    )
    src = {sha_a: dat_a, sha_b: dat_b}
    r = renderer.render(m, src, (1920, 1080))
    assert r.sources == [sha_a, sha_b]


def test_diptych_requires_two_sources() -> None:
    sha, data = make_source(1600, 1000)
    m = ArtDirectionManifest(
        sources=[sha],
        layout_treatment=LayoutTreatment.DIPTYCH,
    )
    with pytest.raises(RenderError):
        renderer.render(m, {sha: data}, (1920, 1080))


# -- T3/T4: provenance --------------------------------------------------------


def test_diptych_provenance_fields_and_roundtrip() -> None:
    (sha_a, dat_a), (sha_b, dat_b) = _diptych_sources()
    m = ArtDirectionManifest(
        sources=[sha_a, sha_b],
        layout_treatment=LayoutTreatment.DIPTYCH,
    )
    r = renderer.render(m, {sha_a: dat_a, sha_b: dat_b}, (1920, 1080))
    assert r.renderer_version == RENDERER_VERSION
    assert r.color_mode == "RGB"
    assert r.color_profile == "sRGB"
    assert r.size_bytes == len(renderer.render_bytes(m, {sha_a: dat_a, sha_b: dat_b}, (1920, 1080)))
    assert RenderResult.from_dict(json.loads(json.dumps(r.to_dict()))) == r


def test_fullbleed_provenance_fields_present() -> None:
    m, src = manifest(4000, 3000)
    r = renderer.render(m, src, (1920, 1080))
    assert r.renderer_version == RENDERER_VERSION
    assert r.color_mode == "RGB"
    assert r.color_profile == "sRGB"
    assert r.sources == list(m.sources)
    assert isinstance(r.sha256, str) and len(r.sha256) == 64
    assert r.size_bytes > 0
