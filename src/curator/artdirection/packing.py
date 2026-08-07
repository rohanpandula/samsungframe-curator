"""Deterministic weighted cell packing for N-source layouts (M010/S01, S03).

Answers one question — *which rectangle of the output canvas does each source
occupy?* — and answers it as real :class:`SourceRegion` geometry in
output-canvas pixels. This is the geometry layer the policy engine (writer), the
artifact validator (reader) and, from M010/S02, the renderer all agree on. It is
the depth-1 slicing tree that ``renderer.py``'s ``_diptych`` has always been,
lifted out and carried to arbitrary depth.

M010/S03 makes the split point weight-driven: :func:`slice_cells` is the one
arithmetic path, and :func:`equal_cells` is its uniform-weight delegate. There is
**no search loop** anywhere here — no iteration count, no temperature schedule,
no candidate-layout list. The geometry rule has no genuine ties left to
randomize, so the packer needs no RNG at all: every tie is resolved by a stated
rule (see :func:`_bisect_by_weight`).

Purity contract, matching :mod:`curator.artdirection.policy`: no image library,
no database, no RNG, no wall clock, and no import from the taste or rendering
packages. The only imports are ``math``, the typed error base and the manifest
schema, so identical input always yields identical output *and* identical order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class WeightedSource:
    """One source and the share of its box it should get (M010/S03).

    ``weight`` is an *importance* share, not a size: the packer converts a
    subtree's summed weight into an extent along the cut axis, so only the ratios
    between weights matter. The default ``1.0`` is what makes
    :func:`equal_cells` a one-line delegate.

    There is deliberately **no** ``aspect`` field. Per-source aspect reasoning
    lands in M010/S05 inside the policy engine, where it informs the per-cell
    *fit* decision (letterbox vs crop-to-fill) and never the cell geometry — the
    packer needs a weight and a box, nothing else.
    """

    sha: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {"sha": self.sha, "weight": self.weight}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightedSource:
        if isinstance(data, cls):
            return data
        return cls(sha=str(data["sha"]), weight=float(data.get("weight", 1.0)))


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


def _usable_weights(items: Sequence[WeightedSource]) -> list[float]:
    """Return *items*' weights with the hygiene rules applied (M010/S03).

    Two rules, both stated here because a weight is a caller-supplied value that
    reaches the geometry engine from ``curator propose --weights`` and from a
    persisted proposal's ``evidence``:

    * A weight that does not name a share of a canvas — negative, NaN or
      infinite — is clamped to ``0.0``. (``max(0.0, w)`` alone would let NaN
      through on the other argument order, so the test is explicit.)
    * When the resulting total is not positive, *every* weight falls back to
      ``1.0``: an all-zero (or all-negative) vector means "no opinion", which is
      exactly the uniform case, never a degenerate zero-extent layout.
    """
    cleaned = []
    for item in items:
        weight = float(item.weight)
        cleaned.append(weight if math.isfinite(weight) and weight > 0.0 else 0.0)
    if sum(cleaned) <= 0.0:
        return [1.0] * len(cleaned)
    return cleaned


def _bisect_by_weight(items: Sequence[WeightedSource]) -> int:
    """Return the split index ``k`` in ``1..N-1`` that best balances *items*.

    ``k`` minimizes ``abs(2 * cumulative_weight(k) - total_weight)`` — the
    imbalance between the two halves. **Among ties, the largest ``k`` wins**, and
    that choice is load-bearing rather than arbitrary: on all-equal weights the
    imbalance is symmetric around ``N / 2``, so the largest minimizing index is
    exactly ``ceil(N / 2)`` — precisely the stated equal-cells rule M010/S01
    documented. Verified by hand for N = 2..9 (N=3 ties at k in {1,2} -> 2; N=5
    ties at {2,3} -> 3; N=7 ties at {3,4} -> 4; N=9 ties at {4,5} -> 5; the even
    N have a unique minimum at N/2). Choosing the *smallest* index instead would
    put ``floor(N / 2)`` on the left and silently contradict S01 at every odd N,
    which is why :func:`slice_cells` is a strict generalization of
    :func:`equal_cells` and not a replacement for it.

    No RNG: the rule resolves every tie by itself, so there is nothing left to
    seed.
    """
    if len(items) < 2:
        raise PackingError(
            f"cannot bisect {len(items)} item(s): at least two are required"
        )
    weights = _usable_weights(items)
    total = sum(weights)
    best_k = 1
    best_imbalance = -1.0
    running = 0.0
    for k in range(1, len(weights)):
        running += weights[k - 1]
        imbalance = abs(2.0 * running - total)
        # `<=` is what makes the largest tied index win, since k ascends.
        if best_imbalance < 0.0 or imbalance <= best_imbalance:
            best_imbalance = imbalance
            best_k = k
    return best_k


def slice_cells(
    items: Sequence[WeightedSource], box: Cell, *, gap: int
) -> list[SourceRegion]:
    """Split *box* into one weight-proportional cell per item, in input order.

    The single arithmetic path for every M010 layout — :func:`equal_cells` is a
    uniform-weight delegate over this function. Recursive weighted bisection:

    * One item: it covers *box* exactly.
    * Otherwise the list is split at ``k = _bisect_by_weight(items)``;
      ``items[:k]`` take the left/top half and ``items[k:]`` the right/bottom
      half. That is the explicit RNG-free tie-break, and on uniform weights it is
      ``ceil(N / 2)``, i.e. M010/S01's stated count rule.
    * Cut axis: vertical (side by side) when ``box.w >= box.h``, horizontal
      (stacked) otherwise — the same rule as ``renderer.py:315``.
    * With ``extent`` the box size on the cut axis and ``avail = extent - gap``,
      the halves get ``max(1, floor(avail * sum_left / sum_all))`` and
      ``max(1, floor(avail * sum_right / sum_all))``. The first half is anchored
      at the box origin, the second at the box's *far edge*. Anchoring the second
      half rather than deriving it by an independent rounding is what keeps cells
      from drifting a pixel off the canvas edge at depth, and flooring both
      extents parks the leftover slack inside the gutter, so the gutter is always
      at least *gap* wide.

    Weight hygiene (negative/NaN/infinite clamped to ``0.0``; a non-positive
    total falling back to uniform) is documented on :func:`_usable_weights`.

    Raises :class:`PackingError` when ``avail < 2`` — a box that cannot be split
    at all is an error, never a zero-width cell. A weight vector extreme enough
    to starve a subtree (``0,0,5``) surfaces as that same loud error rather than
    as an invisible panel.

    At ``N == 2`` with uniform weights this reproduces the diptych render path's
    boxes exactly (for a 1920x1080 target with ``gap=33``: ``(0, 0, 943, 1080)``
    and ``(977, 0, 943, 1080)``), which is what lets M010/S02 replace that
    special case with a region loop without changing a single rendered byte.
    """
    count = len(items)
    if count == 0:
        raise PackingError("cannot pack zero sources into a cell")
    if count == 1:
        return [
            SourceRegion(
                source_sha256=items[0].sha, x=box.x, y=box.y, w=box.w, h=box.h
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

    weights = _usable_weights(items)
    split = _bisect_by_weight(items)
    total = sum(weights)
    left = sum(weights[:split])
    right = sum(weights[split:])
    # The multiplication MUST come before the division. `avail * left / total`
    # groups as `(avail * left) / total` and is exact for the uniform case, while
    # the algebraically identical `avail * (left / total)` is not: 22 * (15/22)
    # floors to 14 because the float 15/22 rounds just below the exact ratio,
    # where (22 * 15) / 22 floors to 15. The two forms look interchangeable; only
    # this one reproduces equal_cells' integer arithmetic pixel for pixel.
    first = max(1, math.floor(avail * left / total))
    second = max(1, math.floor(avail * right / total))
    if first + second > avail:
        # Both extents are floored, so they can only overshoot when the 1px floor
        # bumped a starved side up from 0 — at most by one pixel, and never for
        # uniform weights. Trim the dominant side so the gutter stays >= gap.
        if first >= second:
            first = avail - second
        else:
            second = avail - first
    if vertical:
        first_box = Cell(box.x, box.y, first, box.h)
        second_box = Cell(box.x + box.w - second, box.y, second, box.h)
    else:
        first_box = Cell(box.x, box.y, box.w, first)
        second_box = Cell(box.x, box.y + box.h - second, box.w, second)

    return slice_cells(items[:split], first_box, gap=gap) + slice_cells(
        items[split:], second_box, gap=gap
    )


def equal_cells(shas: Sequence[str], box: Cell, *, gap: int) -> list[SourceRegion]:
    """Split *box* into one equal-area cell per sha, returned in input order.

    The uniform-weight delegate over :func:`slice_cells` (M010/S03) — one
    arithmetic path, so a change to the split rule cannot make the weighted and
    the equal-area layouts disagree. Every geometry rule, every error and the
    N=2 diptych parity property are documented there; M010/S01's tests for this
    function pass **unedited** against the delegated implementation, which is the
    proof that S03 generalized the packer rather than replacing it.
    """
    return slice_cells([WeightedSource(sha) for sha in shas], box, gap=gap)


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

    Recomputing replaces the *geometry* only: each source's declared
    :attr:`SourceRegion.crop` is carried onto its new cell (M010/S05). A crop is
    a **fit** directive, not a coordinate — dropping it here would make the same
    manifest fill at 1080p and letterbox at 4K, silently discarding an intent the
    caller stated, which is precisely the failure this milestone exists to
    remove. The crop map is built from *every* region, set or unset, so a legacy
    all-zero region that nonetheless names a fit keeps it.
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
    declared = {region.source_sha256: region.crop for region in manifest.regions}
    return [
        replace(cell, crop=declared.get(cell.source_sha256))
        for cell in equal_cells(order, Cell(0, 0, tw, th), gap=gutter_for_target(target))
    ]
