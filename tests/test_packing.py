"""Tests for the deterministic cell packer (M010/S01).

Proves the geometry layer every later M010 slice builds on: one cell per source,
every cell inside its box and at least a pixel on both axes, cells pairwise
disjoint, output order equal to input order, byte-identical repeat calls, and —
the parity property that lets S02 replace the diptych special case without
changing a rendered byte — cells identical to the current ``_diptych`` box math
at N=2. Also covers ``resolve_regions``' all-or-nothing stored-vs-recomputed
decision.

M010/S03 adds the weighted half: ``slice_cells`` is now the one arithmetic path
and ``equal_cells`` its uniform-weight delegate, so **every assertion above this
line is untouched on purpose** — their passing against the delegated
implementation is the proof that S03 generalized the packer instead of replacing
it. The new tests below cover the weight-monotone property, the largest-index
tie-break (``ceil(N/2)`` on uniform weights, for N = 2..9), the packer's
invariants at every N from 2 to 9 at two targets, and the weight-hygiene
fallbacks.
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
    WeightedSource,
    _bisect_by_weight,
    equal_cells,
    gutter_for_target,
    resolve_regions,
    slice_cells,
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


# -- slice_cells: the weighted generalization (M010/S03) ----------------------


def weighted(*weights: float) -> list[WeightedSource]:
    return [WeightedSource(f"sha{i}", w) for i, w in enumerate(weights)]


def uniform(count: int) -> list[WeightedSource]:
    return [WeightedSource(sha) for sha in shas(count)]


def area(region: SourceRegion) -> float:
    return region.w * region.h


def test_weighted_source_roundtrips() -> None:
    item = WeightedSource("abc", 0.75)
    assert WeightedSource.from_dict(item.to_dict()) == item
    assert WeightedSource("abc").weight == 1.0


def test_heavier_weight_gets_a_strictly_larger_cell_at_n2() -> None:
    first, second = slice_cells(weighted(0.9, 0.3), Cell(0, 0, 1920, 1080), gap=33)
    assert first.w > second.w
    assert area(first) > area(second)


def test_heavier_weight_gets_a_strictly_larger_cell_at_n3() -> None:
    regions = slice_cells(weighted(3.0, 1.0, 1.0), Cell(0, 0, 1920, 1080), gap=33)
    assert area(regions[0]) > area(regions[1])
    assert area(regions[0]) > area(regions[2])


def test_equal_weights_give_the_equal_cells_geometry() -> None:
    """The generalization claim, stated directly rather than only via delegation."""
    for count in range(2, 10):
        box = Cell(0, 0, 1920, 1080)
        assert slice_cells(uniform(count), box, gap=33) == equal_cells(
            shas(count), box, gap=33
        )


# -- the tie-break rule -------------------------------------------------------


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 9])
def test_uniform_weights_put_ceil_half_on_the_left(count: int) -> None:
    """Largest tied index == ceil(N/2) == M010/S01's stated equal-cells rule."""
    assert _bisect_by_weight(uniform(count)) == -(-count // 2)


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 9])
def test_the_first_ceil_half_of_cells_sit_on_the_origin_side(count: int) -> None:
    split = -(-count // 2)
    regions = slice_cells(uniform(count), Cell(0, 0, 1920, 1080), gap=33)
    cut = max(r.x + r.w for r in regions[:split])
    assert min(r.x for r in regions[split:]) >= cut


def test_the_tie_break_is_the_largest_index_not_the_smallest() -> None:
    """Falsifies "smallest wins": both k are minimal, and 2 must be chosen."""
    assert _bisect_by_weight(weighted(1.0, 1.0, 1.0)) == 2


def test_the_split_follows_the_weights_not_the_count() -> None:
    """One heavy item balances two light ones, so the cut lands after it."""
    assert _bisect_by_weight(weighted(2.0, 1.0, 1.0)) == 1


def test_bisect_rejects_a_list_it_cannot_split() -> None:
    with pytest.raises(PackingError):
        _bisect_by_weight(uniform(1))


# -- slice_cells: invariants across every N and both targets ------------------

WEIGHT_VECTORS = [
    None,  # uniform
    "descending",  # 1.0, 0.9, 0.8, ...
    "spiky",  # one dominant source
]


def weights_for(kind: str | None, count: int) -> list[WeightedSource]:
    if kind is None:
        return uniform(count)
    if kind == "descending":
        return [WeightedSource(f"sha{i}", 1.0 - 0.05 * i) for i in range(count)]
    return [WeightedSource(f"sha{i}", 3.0 if i == 0 else 1.0) for i in range(count)]


@pytest.mark.parametrize("kind", WEIGHT_VECTORS)
@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("target", [LANDSCAPE, UHD])
def test_weighted_cells_in_bounds_and_at_least_one_pixel(
    kind: str | None, count: int, target: tuple[int, int]
) -> None:
    tw, th = target
    regions = slice_cells(
        weights_for(kind, count), Cell(0, 0, tw, th), gap=gutter_for_target(target)
    )
    assert len(regions) == count
    for region in regions:
        assert region.w >= 1 and region.h >= 1
        assert region.x >= 0 and region.y >= 0
        assert region.x + region.w <= tw
        assert region.y + region.h <= th


@pytest.mark.parametrize("kind", WEIGHT_VECTORS)
@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("target", [LANDSCAPE, UHD])
def test_weighted_cells_are_pairwise_disjoint(
    kind: str | None, count: int, target: tuple[int, int]
) -> None:
    regions = slice_cells(
        weights_for(kind, count), Cell(0, 0, *target), gap=gutter_for_target(target)
    )
    for i, a in enumerate(regions):
        for b in regions[i + 1 :]:
            assert not overlaps(a, b), (boxes([a]), boxes([b]))


@pytest.mark.parametrize("kind", WEIGHT_VECTORS)
@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 9])
@pytest.mark.parametrize("target", [LANDSCAPE, UHD])
def test_weighted_gutter_is_never_below_gap_at_any_depth(
    kind: str | None, count: int, target: tuple[int, int]
) -> None:
    """Any two cells meet at some ancestor cut, whose gutter is >= gap."""
    gap = gutter_for_target(target)
    regions = slice_cells(weights_for(kind, count), Cell(0, 0, *target), gap=gap)
    for i, a in enumerate(regions):
        for b in regions[i + 1 :]:
            horizontal = min(a.x + a.w, b.x + b.w) <= max(a.x, b.x)
            vertical = min(a.y + a.h, b.y + b.h) <= max(a.y, b.y)
            separation = max(
                max(a.x, b.x) - min(a.x + a.w, b.x + b.w) if horizontal else 0,
                max(a.y, b.y) - min(a.y + a.h, b.y + b.h) if vertical else 0,
            )
            assert separation >= gap, (boxes([a]), boxes([b]))


@pytest.mark.parametrize("kind", WEIGHT_VECTORS)
@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 9])
def test_weighted_output_order_equals_input_order(
    kind: str | None, count: int
) -> None:
    items = weights_for(kind, count)
    regions = slice_cells(items, Cell(0, 0, 1920, 1080), gap=33)
    assert [r.source_sha256 for r in regions] == [item.sha for item in items]


@pytest.mark.parametrize("count", [2, 5, 9])
def test_weighted_repeat_calls_are_identical(count: int) -> None:
    items = weights_for("descending", count)
    first = slice_cells(items, Cell(0, 0, 1920, 1080), gap=33)
    second = slice_cells(items, Cell(0, 0, 1920, 1080), gap=33)
    assert first == second
    assert [r.to_dict() for r in first] == [r.to_dict() for r in second]


# -- weight hygiene (T-10-12: the weight vector is a trust boundary) ----------


@pytest.mark.parametrize("weights", [(0.0, 0.0, 0.0), (-1.0, -2.0, -3.0)])
def test_degenerate_weight_vectors_fall_back_to_uniform(
    weights: tuple[float, ...]
) -> None:
    box = Cell(0, 0, 1920, 1080)
    assert slice_cells(weighted(*weights), box, gap=33) == equal_cells(
        shas(len(weights)), box, gap=33
    )


def test_non_finite_weights_are_clamped_rather_than_crashing() -> None:
    """NaN / inf name no share of a canvas; they clamp to 0.0, never to a crash."""
    box = Cell(0, 0, 1920, 1080)
    assert slice_cells(
        weighted(float("nan"), float("nan")), box, gap=33
    ) == equal_cells(shas(2), box, gap=33)
    regions = slice_cells(weighted(float("inf"), 1.0), box, gap=33)
    assert [r.source_sha256 for r in regions] == ["sha0", "sha1"]
    assert regions[0].w >= 1 and regions[1].w >= 1


def test_a_partly_zero_weight_vector_still_tiles_in_bounds() -> None:
    """A starved side keeps its 1px floor, and the gutter still holds at gap."""
    regions = slice_cells(weighted(-1.0, 2.0), Cell(0, 0, 1920, 1080), gap=33)
    assert regions[0].w >= 1
    assert regions[1].x + regions[1].w <= 1920
    assert regions[1].x - (regions[0].x + regions[0].w) >= 33


def test_zero_items_raises() -> None:
    with pytest.raises(PackingError):
        slice_cells([], Cell(0, 0, 1920, 1080), gap=33)


def test_unsplittable_box_raises_for_weighted_items_too() -> None:
    with pytest.raises(PackingError) as exc:
        slice_cells(weighted(1.0, 1.0), Cell(0, 0, 30, 30), gap=33)
    assert "cannot split" in str(exc.value)


# -- R046: determinism at both targets, not just one --------------------------


@pytest.mark.parametrize("target", [LANDSCAPE, UHD])
@pytest.mark.parametrize("count", [2, 3, 5, 9])
def test_weighted_geometry_is_identical_on_repeat_at_both_targets(
    count: int, target: tuple[int, int]
) -> None:
    """Identical sources and weights produce identical geometry, at 1080p *and* 4K.

    A determinism check at one resolution cannot catch per-cell rounding drift
    that only shows at the other (01-RESEARCH.md Pitfall 8), so both are asserted
    here rather than one standing in for the pair.
    """
    items = weights_for("descending", count)
    box = Cell(0, 0, *target)
    gap = gutter_for_target(target)
    first = slice_cells(items, box, gap=gap)
    second = slice_cells(items, box, gap=gap)
    assert first == second
    assert boxes(first) == boxes(second)
