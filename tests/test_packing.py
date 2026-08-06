"""Tests for the deterministic cell packer (M010/S01).

Proves the geometry layer every later M010 slice builds on: one cell per source,
every cell inside its box and at least a pixel on both axes, cells pairwise
disjoint, output order equal to input order, byte-identical repeat calls, and —
the parity property that lets S02 replace the diptych special case without
changing a rendered byte — cells identical to the current ``_diptych`` box math
at N=2. Also covers ``resolve_regions``' all-or-nothing stored-vs-recomputed
decision.
"""

from __future__ import annotations

import pytest

from curator.artdirection.manifest import (
    ArtDirectionManifest,
    LayoutTreatment,
    SourceRegion,
)
from curator.artdirection.packing import (
    Cell,
    PackingError,
    equal_cells,
    gutter_for_target,
    resolve_regions,
)

LANDSCAPE = (1920, 1080)
PORTRAIT = (1080, 1920)
UHD = (3840, 2160)


def shas(count: int) -> list[str]:
    return [f"sha{i}" for i in range(count)]


def boxes(regions: list[SourceRegion]) -> list[tuple[float, float, float, float]]:
    return [(r.x, r.y, r.w, r.h) for r in regions]


def diptych_boxes(target: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """The current ``renderer.py:314-322`` diptych arithmetic, reproduced verbatim.

    Kept as literal duplicated math on purpose: it is the fixed reference
    ``equal_cells`` must match at N=2, so it must not import from the packer.
    """
    tw, th = target
    gap = max(1, min(tw, th) // 32)
    if tw >= th:
        panel_w = max(1, (tw - gap) // 2)
        return [(0, 0, panel_w, th), (tw - panel_w, 0, panel_w, th)]
    panel_h = max(1, (th - gap) // 2)
    return [(0, 0, tw, panel_h), (0, th - panel_h, tw, panel_h)]


def overlaps(a: SourceRegion, b: SourceRegion) -> bool:
    return (
        a.x < b.x + b.w
        and b.x < a.x + a.w
        and a.y < b.y + b.h
        and b.y < a.y + a.h
    )


# -- gutter -------------------------------------------------------------------


def test_gutter_matches_the_renderer_formula() -> None:
    assert gutter_for_target(LANDSCAPE) == 33
    assert gutter_for_target(UHD) == 67
    assert gutter_for_target(PORTRAIT) == 33


def test_gutter_never_degenerates_to_zero() -> None:
    """A tiny target still gets a visible gutter (the max(1, ...) floor)."""
    assert gutter_for_target((16, 16)) == 1


# -- equal_cells: shape -------------------------------------------------------


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_one_region_per_sha(count: int) -> None:
    regions = equal_cells(shas(count), Cell(0, 0, 1920, 1080), gap=33)
    assert len(regions) == count


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5])
def test_output_order_equals_input_order(count: int) -> None:
    """`identical input => identical output *and* order` — pairing_order is real."""
    order = shas(count)
    regions = equal_cells(order, Cell(0, 0, 1920, 1080), gap=33)
    assert [r.source_sha256 for r in regions] == order


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 9])
@pytest.mark.parametrize("target", [LANDSCAPE, PORTRAIT, UHD])
def test_cells_in_bounds_and_at_least_one_pixel(
    count: int, target: tuple[int, int]
) -> None:
    tw, th = target
    box = Cell(0, 0, tw, th)
    regions = equal_cells(shas(count), box, gap=gutter_for_target(target))
    for region in regions:
        assert region.w >= 1 and region.h >= 1
        assert 0 <= region.x and 0 <= region.y
        assert region.x + region.w <= tw
        assert region.y + region.h <= th


@pytest.mark.parametrize("count", [2, 3, 4, 5, 9])
@pytest.mark.parametrize("target", [LANDSCAPE, PORTRAIT, UHD])
def test_cells_are_pairwise_disjoint(count: int, target: tuple[int, int]) -> None:
    regions = equal_cells(
        shas(count), Cell(0, 0, *target), gap=gutter_for_target(target)
    )
    for i, a in enumerate(regions):
        for b in regions[i + 1 :]:
            assert not overlaps(a, b), (boxes([a]), boxes([b]))


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_gutter_between_neighbours_is_at_least_gap(count: int) -> None:
    """Flooring both extents parks the slack in the gutter, never below *gap*."""
    gap = gutter_for_target(LANDSCAPE)
    regions = equal_cells(shas(count), Cell(0, 0, 1920, 1080), gap=gap)
    for i, a in enumerate(regions):
        for b in regions[i + 1 :]:
            horizontal = min(a.x + a.w, b.x + b.w) <= max(a.x, b.x)
            vertical = min(a.y + a.h, b.y + b.h) <= max(a.y, b.y)
            separation = max(
                max(a.x, b.x) - min(a.x + a.w, b.x + b.w) if horizontal else 0,
                max(a.y, b.y) - min(a.y + a.h, b.y + b.h) if vertical else 0,
            )
            assert separation >= gap


def test_single_sha_covers_the_box_exactly() -> None:
    regions = equal_cells(["only"], Cell(10, 20, 300, 400), gap=33)
    assert boxes(regions) == [(10, 20, 300, 400)]


# -- equal_cells: diptych parity (the S02 no-byte-change guarantee) ------------


def test_n2_reproduces_diptych_boxes_landscape() -> None:
    regions = equal_cells(
        shas(2), Cell(0, 0, *LANDSCAPE), gap=gutter_for_target(LANDSCAPE)
    )
    assert boxes(regions) == diptych_boxes(LANDSCAPE)
    assert boxes(regions) == [(0, 0, 943, 1080), (977, 0, 943, 1080)]


def test_n2_reproduces_diptych_boxes_portrait() -> None:
    regions = equal_cells(
        shas(2), Cell(0, 0, *PORTRAIT), gap=gutter_for_target(PORTRAIT)
    )
    assert boxes(regions) == diptych_boxes(PORTRAIT)
    assert boxes(regions) == [(0, 0, 1080, 943), (0, 977, 1080, 943)]


def test_n2_reproduces_diptych_boxes_at_4k() -> None:
    regions = equal_cells(shas(2), Cell(0, 0, *UHD), gap=gutter_for_target(UHD))
    assert boxes(regions) == diptych_boxes(UHD)


# -- equal_cells: determinism and failure -------------------------------------


def test_repeat_calls_are_identical() -> None:
    first = equal_cells(shas(5), Cell(0, 0, 1920, 1080), gap=33)
    second = equal_cells(shas(5), Cell(0, 0, 1920, 1080), gap=33)
    assert first == second
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]


def test_unsplittable_box_raises_rather_than_emitting_a_zero_cell() -> None:
    with pytest.raises(PackingError) as exc:
        equal_cells(shas(2), Cell(0, 0, 30, 30), gap=33)
    assert "cannot split" in str(exc.value)


def test_zero_sources_raises() -> None:
    with pytest.raises(PackingError):
        equal_cells([], Cell(0, 0, 1920, 1080), gap=33)


def test_cell_roundtrips() -> None:
    cell = Cell(1, 2, 3, 4)
    assert Cell.from_dict(cell.to_dict()) == cell


# -- resolve_regions ----------------------------------------------------------


def packed_manifest(target: tuple[int, int], count: int = 2) -> ArtDirectionManifest:
    order = shas(count)
    return ArtDirectionManifest(
        sources=order,
        regions=equal_cells(order, Cell(0, 0, *target), gap=gutter_for_target(target)),
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=order,
    )


def test_resolve_uses_stored_regions_when_they_tile_the_target() -> None:
    manifest = packed_manifest(LANDSCAPE)
    assert resolve_regions(manifest, LANDSCAPE) == manifest.regions


def test_resolve_recomputes_when_stored_regions_do_not_tile_the_target() -> None:
    """A 1080p-materialized manifest must fill 4K, not its top-left quadrant."""
    manifest = packed_manifest(LANDSCAPE)
    regions = resolve_regions(manifest, UHD)
    assert regions != manifest.regions
    assert max(r.x + r.w for r in regions) == 3840
    assert max(r.y + r.h for r in regions) == 2160


def test_resolve_recomputes_when_any_region_is_unset() -> None:
    """All-or-nothing: one unset cell means every cell is recomputed."""
    manifest = packed_manifest(LANDSCAPE)
    mixed = ArtDirectionManifest(
        sources=manifest.sources,
        regions=[manifest.regions[0], SourceRegion(source_sha256=manifest.sources[1])],
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=manifest.pairing_order,
    )
    assert resolve_regions(mixed, LANDSCAPE) == manifest.regions


def test_resolve_recomputes_for_a_legacy_all_zero_manifest() -> None:
    order = shas(2)
    legacy = ArtDirectionManifest(
        sources=order,
        regions=[SourceRegion(source_sha256=sha) for sha in order],
        layout_treatment=LayoutTreatment.DIPTYCH,
    )
    regions = resolve_regions(legacy, LANDSCAPE)
    assert boxes(regions) == diptych_boxes(LANDSCAPE)


def test_resolve_follows_pairing_order_not_source_order() -> None:
    order = ["b", "a"]
    manifest = ArtDirectionManifest(
        sources=["a", "b"],
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=order,
    )
    assert [r.source_sha256 for r in resolve_regions(manifest, LANDSCAPE)] == order


def test_resolve_is_deterministic() -> None:
    manifest = packed_manifest(LANDSCAPE, count=3)
    assert resolve_regions(manifest, UHD) == resolve_regions(manifest, UHD)
