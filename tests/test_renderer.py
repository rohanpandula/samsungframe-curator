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
    ManifestError,
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


# -- M010/S02: one region-iterating branch for every multi-cell treatment ------


NUP_COLORS = [(200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40)]


def _nup_sources(count: int, width: int = 1600, height: int = 1200):
    """Return ({sha: bytes}, [sha...]) for *count* distinctly-colored sources."""
    data: dict[str, bytes] = {}
    order: list[str] = []
    for index in range(count):
        sha, payload = make_source(width, height, color=NUP_COLORS[index])
        data[sha] = payload
        order.append(sha)
    return data, order


def _nup_manifest(
    treatment: LayoutTreatment,
    order: list[str],
    *,
    upscale_warning: bool = False,
) -> ArtDirectionManifest:
    """A regionless N-up manifest — the renderer packs its cells at render time."""
    return ArtDirectionManifest(
        sources=list(order),
        layout_treatment=treatment,
        pairing_order=list(order),
        processing_intent=ProcessingIntent(upscale_warning=upscale_warning),
    )


@pytest.mark.parametrize("target", [(1920, 1080), (3840, 2160)])
@pytest.mark.parametrize(
    "treatment,count",
    [(LayoutTreatment.TRIPTYCH, 3), (LayoutTreatment.QUAD, 4)],
)
def test_nup_renders_exact_target_dims(treatment, count, target) -> None:
    """Triptych and quad render at exact 1080p and 4K dimensions."""
    src, order = _nup_sources(count)
    m = _nup_manifest(treatment, order)
    r = renderer.render(m, src, target)
    assert (r.target_width, r.target_height) == target
    assert r.treatment == treatment.value
    assert open_rgb(renderer.render_bytes(m, src, target)).size == target


def test_triptych_paints_every_cell() -> None:
    """Each source lands in its own cell — three panels, not a truncated two."""
    from curator.artdirection.packing import Cell, equal_cells, gutter_for_target

    src, order = _nup_sources(3)
    m = _nup_manifest(LayoutTreatment.TRIPTYCH, order)
    img = open_rgb(renderer.render_bytes(m, src, (1920, 1080)))
    cells = equal_cells(
        order, Cell(0, 0, 1920, 1080), gap=gutter_for_target((1920, 1080))
    )
    for index, cell in enumerate(cells):
        center = (int(cell.x + cell.w // 2), int(cell.y + cell.h // 2))
        assert img.getpixel(center) == NUP_COLORS[index]


def test_nup_result_lists_every_source() -> None:
    """RenderResult.sources reports all three sources, never the first two."""
    src, order = _nup_sources(3)
    r = renderer.render(_nup_manifest(LayoutTreatment.TRIPTYCH, order), src, (1920, 1080))
    assert r.sources == order
    assert len(r.sources) == 3


def test_triptych_byte_determinism() -> None:
    """An N-up render is byte-deterministic, like every other treatment."""
    src, order = _nup_sources(3)
    m = _nup_manifest(LayoutTreatment.TRIPTYCH, order)
    first = renderer.render(m, src, (1920, 1080))
    second = renderer.render(m, src, (1920, 1080))
    assert first.sha256 == second.sha256
    assert renderer.render_bytes(m, src, (1920, 1080)) == renderer.render_bytes(
        m, src, (1920, 1080)
    )


def test_diptych_with_three_sources_is_rejected_not_truncated() -> None:
    """The verified silent third-source drop is now a loud RenderError."""
    src, order = _nup_sources(3)
    m = _nup_manifest(LayoutTreatment.DIPTYCH, order)
    with pytest.raises(RenderError) as excinfo:
        renderer.render(m, src, (1920, 1080))
    message = str(excinfo.value)
    assert "exactly 2" in message
    assert "got 3" in message
    assert "never truncated" in message


def test_triptych_with_two_sources_is_rejected() -> None:
    """An under-count N-up keeps the pre-M010 message shape."""
    src, order = _nup_sources(2)
    m = _nup_manifest(LayoutTreatment.TRIPTYCH, order)
    with pytest.raises(RenderError) as excinfo:
        renderer.render(m, src, (1920, 1080))
    assert "triptych requires at least three sources" in str(excinfo.value)


def test_nup_missing_source_bytes_raises() -> None:
    """A sha with no bytes is named, before any cell is packed."""
    src, order = _nup_sources(3)
    del src[order[2]]
    with pytest.raises(RenderError) as excinfo:
        renderer.render(_nup_manifest(LayoutTreatment.TRIPTYCH, order), src, (1920, 1080))
    assert order[2] in str(excinfo.value)


def test_nup_upscale_without_approval_raises_then_renders_with_it() -> None:
    """A cell that would upscale its source is blocked by R008 until approved."""
    src, order = _nup_sources(3, width=200, height=150)
    blocked = _nup_manifest(LayoutTreatment.TRIPTYCH, order)
    with pytest.raises(RenderError) as excinfo:
        renderer.render(blocked, src, (1920, 1080))
    assert "R008" in str(excinfo.value)

    approved = _nup_manifest(LayoutTreatment.TRIPTYCH, order, upscale_warning=True)
    r = renderer.render(approved, src, (1920, 1080))
    assert r.upscaled_warning is True
    assert (r.target_width, r.target_height) == (1920, 1080)


def test_diptych_bytes_identical_to_pre_m010_box_math() -> None:
    """The rewrite changes no rendered byte: parity against the old two-box math.

    Reproduces ``_diptych``'s exact pre-M010 arithmetic (renderer.py:306-326 as
    of M003/S01) and asserts the PNG bytes match what the region loop now
    produces, at landscape, portrait and 4K targets.
    """
    from PIL import Image as PILImage

    from curator.render.renderer import (
        _background_color,
        _encode_png,
        _open_rgb,
        _paste_panel,
    )

    (sha_a, dat_a), (sha_b, dat_b) = _diptych_sources()
    m = ArtDirectionManifest(
        sources=[sha_a, sha_b],
        layout_treatment=LayoutTreatment.DIPTYCH,
        # Approved only so the 4K leg clears the R008 gate; the flag changes no
        # pixel, so both sides of the parity comparison stay comparable.
        processing_intent=ProcessingIntent(upscale_warning=True),
    )
    src = {sha_a: dat_a, sha_b: dat_b}

    for target in [(1920, 1080), (900, 1440), (3840, 2160)]:
        tw, th = target
        canvas = PILImage.new("RGB", target, _background_color(m.background))
        gap = max(1, min(tw, th) // 32)
        if tw >= th:
            panel_w = max(1, (tw - gap) // 2)
            box_a = (0, 0, panel_w, th)
            box_b = (tw - panel_w, 0, panel_w, th)
        else:
            panel_h = max(1, (th - gap) // 2)
            box_a = (0, 0, tw, panel_h)
            box_b = (0, th - panel_h, tw, panel_h)
        _paste_panel(canvas, _open_rgb(dat_a)[0], box_a)
        _paste_panel(canvas, _open_rgb(dat_b)[0], box_b)
        assert _encode_png(canvas) == renderer.render_bytes(m, src, target)


# -- M010/S03: PACKED, the variable-count treatment ---------------------------

PACKED_COLORS = [
    (200, 40, 40),
    (40, 200, 40),
    (40, 40, 200),
    (200, 200, 40),
    (40, 200, 200),
    (200, 40, 200),
    (120, 120, 120),
    (240, 160, 40),
    (80, 40, 160),
]


def _packed_sources(count: int, width: int = 1600, height: int = 1200):
    """Return ({sha: bytes}, [sha...]) for up to nine distinctly-colored sources."""
    data: dict[str, bytes] = {}
    order: list[str] = []
    for index in range(count):
        sha, payload = make_source(width, height, color=PACKED_COLORS[index])
        data[sha] = payload
        order.append(sha)
    return data, order


@pytest.mark.parametrize("count", [2, 5, 9])
@pytest.mark.parametrize("target", [(1920, 1080), (3840, 2160)])
def test_packed_renders_exact_target_dims_for_any_n(count: int, target) -> None:
    """PACKED needed no render branch — only a MULTI_CELL_TREATMENTS member."""
    src, order = _packed_sources(count)
    m = _nup_manifest(LayoutTreatment.PACKED, order, upscale_warning=True)
    r = renderer.render(m, src, target)
    assert (r.target_width, r.target_height) == target
    assert r.treatment == "packed"
    assert r.sources == order
    assert open_rgb(renderer.render_bytes(m, src, target)).size == target


def test_packed_paints_every_cell_at_five_sources() -> None:
    from curator.artdirection.packing import Cell, gutter_for_target, resolve_regions

    src, order = _packed_sources(5)
    m = _nup_manifest(LayoutTreatment.PACKED, order)
    img = open_rgb(renderer.render_bytes(m, src, (1920, 1080)))
    assert Cell(0, 0, 1920, 1080).w == 1920  # the packer's own box, for clarity
    assert gutter_for_target((1920, 1080)) == 33
    for index, cell in enumerate(resolve_regions(m, (1920, 1080))):
        center = (int(cell.x + cell.w // 2), int(cell.y + cell.h // 2))
        assert img.getpixel(center) == PACKED_COLORS[index]


def test_packed_with_one_source_is_rejected() -> None:
    """A countless treatment is still bounded on both sides, never truncated."""
    src, order = _packed_sources(1)
    with pytest.raises(RenderError) as excinfo:
        renderer.render(_nup_manifest(LayoutTreatment.PACKED, order), src, (1920, 1080))
    message = str(excinfo.value)
    assert "lays out 2 to 9 sources" in message
    assert "got 1" in message


def test_packed_over_the_cap_is_rejected_before_any_decode() -> None:
    """The cap fires in validate(), _render's first statement — zero Pillow work.

    Deliberately a ``ManifestError`` and not the ``_multi_cell`` bound: the
    earliest gate is the one that must catch an over-cap N, which is why the
    tenth sha here has no bytes in *src* at all (a missing-source ``RenderError``
    would prove the check ran too late).
    """
    src, order = _packed_sources(9)
    order = order + ["deadbeef"]
    m = ArtDirectionManifest(
        sources=order,
        layout_treatment=LayoutTreatment.PACKED,
        pairing_order=order,
    )
    with pytest.raises(ManifestError) as excinfo:
        renderer.render(m, src, (1920, 1080))
    message = str(excinfo.value)
    assert "10 source(s)" in message
    assert "never truncated" in message
