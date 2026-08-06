"""Deterministic equal-area cell packing for N-source layouts (M010/S01).

Answers one question — *which rectangle of the output canvas does each source
occupy?* — and answers it as real :class:`SourceRegion` geometry in
output-canvas pixels. This is the geometry layer the policy engine (writer), the
artifact validator (reader) and, from M010/S02, the renderer all agree on. It is
the depth-1 slicing tree that ``renderer.py``'s ``_diptych`` has always been,
lifted out and carried to arbitrary depth.

Purity contract, matching :mod:`curator.artdirection.policy`: no image library,
no database, no RNG, no wall clock, and no import from the taste or rendering
packages. The only imports are the typed error base and the manifest schema, so
identical input always yields identical output *and* identical order — the split
rule is a stated count rule, never a tie-break left to chance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from curator.artdirection.manifest import ArtDirectionManifest, SourceRegion
from curator.errors import CuratorError


class PackingError(CuratorError):
    """Raised when a box cannot hold the requested cells.

    Loud by construction: a box too small to split raises rather than emitting a
    degenerate zero-extent cell that would silently render as an invisible
    panel.
    """


@dataclass(frozen=True)
class Cell:
    """A rectangular area of the output canvas, in whole pixels.

    ``x`` / ``y`` are the top-left origin and ``w`` / ``h`` the extents, in the
    same output-canvas pixel space :class:`SourceRegion` uses. Integers, because
    a cell is ultimately a paste box.
    """

    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> dict[str, Any]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cell:
        if isinstance(data, cls):
            return data
        return cls(
            x=int(data["x"]),
            y=int(data["y"]),
            w=int(data["w"]),
            h=int(data["h"]),
        )


def gutter_for_target(target: tuple[int, int]) -> int:
    """Return the gutter (inter-cell gap) in pixels for a *target* canvas.

    ``max(1, min(tw, th) // 32)`` — the exact formula the diptych render path
    has always used (``renderer.py:314``), lifted here so there is one source of
    truth for it.

    The gutter is computed **once from the target dims and threaded down** every
    recursion level, never recomputed from a shrinking box: recomputing per
    level produces visibly thinner gutters at depth >= 2, which is the opposite
    of what a growing N needs.
    """
    tw, th = target
    return max(1, min(tw, th) // 32)


def equal_cells(shas: Sequence[str], box: Cell, *, gap: int) -> list[SourceRegion]:
    """Split *box* into one equal-area cell per sha, returned in input order.

    Recursive equal-area bisection. The rules are stated exhaustively so that
    M010/S03's weighted packer generalizes this function rather than replacing
    it:

    * One sha: it covers *box* exactly.
    * Otherwise the list is split at ``k = ceil(N / 2)``; ``shas[:k]`` take the
      left/top half and ``shas[k:]`` the right/bottom half. This is the explicit
      RNG-free tie-break — the direct generalization of ``_TREATMENT_RANK``'s
      role as a *stated* rather than implicit ordering rule.
    * Cut axis: vertical (side by side) when ``box.w >= box.h``, horizontal
      (stacked) otherwise — the same rule as ``renderer.py:315``.
    * With ``extent`` the box size on the cut axis and ``avail = extent - gap``,
      the two halves get ``max(1, (avail * k) // N)`` and
      ``max(1, (avail * (N - k)) // N)``, both integer floor division. The first
      half is anchored at the box origin, the second at the box's *far edge*.
      Anchoring the second half rather than deriving it by an independent
      rounding is what keeps cells from drifting a pixel off the canvas edge at
      depth, and flooring both extents parks the leftover slack inside the
      gutter, so the gutter is always at least *gap* wide.

    Raises :class:`PackingError` when ``avail < 2`` — a box that cannot be split
    at all is an error, never a zero-width cell.

    At ``N == 2`` this reproduces the diptych render path's boxes exactly (for a
    1920x1080 target with ``gap=33``: ``(0, 0, 943, 1080)`` and
    ``(977, 0, 943, 1080)``), which is what lets M010/S02 replace that special
    case with a region loop without changing a single rendered byte.
    """
    count = len(shas)
    if count == 0:
        raise PackingError("cannot pack zero sources into a cell")
    if count == 1:
        return [
            SourceRegion(
                source_sha256=shas[0], x=box.x, y=box.y, w=box.w, h=box.h
            )
        ]

    vertical = box.w >= box.h
    extent = box.w if vertical else box.h
    avail = extent - gap
    if avail < 2:
        raise PackingError(
            f"cannot split a {box.w}x{box.h} box with a {gap}px gutter for "
            f"{count} sources: {avail}px of usable extent on the cut axis"
        )

    split = -(-count // 2)  # ceil(N / 2)
    first = max(1, (avail * split) // count)
    second = max(1, (avail * (count - split)) // count)
    if vertical:
        first_box = Cell(box.x, box.y, first, box.h)
        second_box = Cell(box.x + box.w - second, box.y, second, box.h)
    else:
        first_box = Cell(box.x, box.y, box.w, first)
        second_box = Cell(box.x, box.y + box.h - second, box.w, second)

    return equal_cells(shas[:split], first_box, gap=gap) + equal_cells(
        shas[split:], second_box, gap=gap
    )


def resolve_regions(
    manifest: ArtDirectionManifest, target: tuple[int, int]
) -> list[SourceRegion]:
    """Return the cells *manifest* occupies at *target*, in layout order.

    Layout order is ``pairing_order`` when it is set, else ``sources``. The
    manifest's own regions are used verbatim **only if** every sha in that order
    has a set region (:attr:`SourceRegion.is_unset` is False) *and* those
    regions exactly tile the requested target — ``max(x + w) == target width``
    and ``max(y + h) == target height``. Otherwise every cell is recomputed with
    :func:`equal_cells`.

    The exact-tile condition is what makes a manifest materialized at 1080p
    render correctly at 4K: its stored 1920x1080 cells do not tile a 3840x2160
    canvas, so they are recomputed rather than packed into the top-left quadrant.

    All-or-nothing by design — stored and computed cells are never mixed, so a
    partially-populated manifest resolves to one internally consistent tiling
    instead of a collage of two coordinate systems.
    """
    tw, th = target
    order = list(manifest.pairing_order) if manifest.pairing_order else list(manifest.sources)
    stored = {region.source_sha256: region for region in manifest.regions if not region.is_unset}
    if order and all(sha in stored for sha in order):
        cells = [stored[sha] for sha in order]
        tiles = (
            max(cell.x + cell.w for cell in cells) == tw
            and max(cell.y + cell.h for cell in cells) == th
        )
        if tiles:
            return cells
    return equal_cells(order, Cell(0, 0, tw, th), gap=gutter_for_target(target))
