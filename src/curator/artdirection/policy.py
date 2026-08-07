"""Deterministic art-direction policy engine (M002/S03 T2+T3, M010/S02, S03).

Given one or more :class:`~curator.analysis.schema.AnalysisResult` fixtures and
an :class:`ArtDirectionRequest`, :func:`propose_treatments` ranks which
:class:`~curator.artdirection.manifest.LayoutTreatment` choices are applicable
and returns them as ``TreatmentProposal`` objects ordered by score descending.
Single-source treatments read the primary result; the multi-source ones —
DIPTYCH at two sources, TRIPTYCH at three and QUAD at four (M010/S02), plus
PACKED at any count up to
:data:`~curator.artdirection.manifest.MAX_LAYOUT_SOURCES` (M010/S03) — are gated
on a cross-image affinity and carry their cell geometry as evidence.

PACKED's cells are sized by an importance **weight** per source, defaulting to
``quality.aesthetic_quality`` and overridable through ``request.context["weights"]``
(``curator propose --weights``). The weights are recorded in ``evidence`` — not
kept in a local — because that is the only way they survive the proposals-table
round trip into :func:`materialize_manifest`.

The policy is a pure, deterministic rules engine: identical analysis + request
always yield an identical, identically-ordered list of proposals. It consumes
only the signals the analysis schema exposes — ``crop_safety``, ``quality``,
``saliency.map_size`` (doubling as image aspect), ``color_story``, and
``pairing`` — with no image I/O of its own.

:func:`materialize_manifest` closes the analyze -> propose -> manifest lifecycle,
building a :class:`~curator.artdirection.manifest.ArtDirectionManifest` from a
chosen proposal.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from curator.analysis.schema import AnalysisResult
from curator.artdirection.manifest import (
    _TREATMENT_SOURCE_COUNT,
    CROP_FILL,
    MANIFEST_VERSION,
    MAX_LAYOUT_SOURCES,
    MULTI_CELL_TREATMENTS,
    VALID_CROP_MODES,
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
    ManifestError,
    SourceRegion,
)
from curator.artdirection.packing import (
    Cell,
    WeightedSource,
    _bisect_by_weight,
    equal_cells,
    gutter_for_target,
    slice_cells,
)

if TYPE_CHECKING:
    from curator.analysis.local import LocalAnalysisProvider

#: Minimum aesthetic quality a composition needs to be proposed full-bleed.
MIN_FULLBLEED_AESTHETIC = 0.7

#: Minimum normalized crop margin across all directions for a full-bleed fit.
MIN_FULLBLEED_MARGIN = 0.05

#: Margin below which a crop direction is considered risky (favors matte).
CROP_RISK_MARGIN = 0.05

#: Aspect ratio (w / h) at or above which a composition is panoramic.
PANORAMIC_MIN_ASPECT = 2.0

#: Inclusive aspect-ratio band treated as square-friendly.
SQUARE_ASPECT_MIN = 0.8
SQUARE_ASPECT_MAX = 1.25

#: Minimum pair affinity to propose a diptych.
DIPTYCH_AFFINITY = 0.75

#: Minimum mean group affinity to propose a named N-up template (M010/S02).
#:
#: A **stated, revisable engineering default, not a researched number.** It sits
#: below :data:`DIPTYCH_AFFINITY` (0.75) deliberately: an N-up reads as a wall
#: and tolerates more variety than two adjacent panels, where any mismatch is
#: read as a direct comparison between exactly two images.
NUP_AFFINITY = 0.6

#: How far a cell's aspect must depart from its source's before a crop pays
#: for itself (M010/S05).
#:
#: A **stated, revisable engineering default, not a researched number.** Inside
#: 10% the letterbox bars are a thin sliver, so cropping would throw away real
#: pixels to recover almost no canvas: the cell letterboxes even when it would be
#: perfectly safe to fill. Measured relative to the *source's* own aspect,
#: because the question is how far the cell departs from the shape the image
#: already has.
CELL_CROP_ASPECT_TOLERANCE = 0.1

#: Source count -> the named template that lays that many cells out (M010/S02).
_NUP_TREATMENTS: dict[int, LayoutTreatment] = {
    3: LayoutTreatment.TRIPTYCH,
    4: LayoutTreatment.QUAD,
}

#: Fixed ordering used as a deterministic tie-break for equal scores.
#:
#: Multi-cell treatments rank ahead of the single-source ones, which shifted by
#: +3 in M010/S02 but kept their **relative** order — the dict is only ever used
#: as a relative sort key (see the sort in :func:`propose_treatments`), so no
#: ordering assertion depends on the absolute numbers.
_TREATMENT_RANK = {
    LayoutTreatment.DIPTYCH: 0,
    LayoutTreatment.TRIPTYCH: 1,
    LayoutTreatment.QUAD: 2,
    # PACKED scores exactly what the named template at the same N scores, so this
    # rank is what puts TRIPTYCH/QUAD first at N=3/N=4 (a template wins at its
    # own N) while PACKED covers N=5..MAX_LAYOUT_SOURCES on its own (M010/S03).
    LayoutTreatment.PACKED: 3,
    LayoutTreatment.SINGLE_FULLBLEED: 4,
    LayoutTreatment.CONTAIN_MATTE: 5,
    LayoutTreatment.PANORAMIC: 6,
    LayoutTreatment.SQUARE: 7,
}

# Completeness guard (M010/S02): a LayoutTreatment with no _TREATMENT_RANK entry
# is a bare KeyError at sort time, arbitrarily far from the line that added it.
# An explicit raise (never `assert`, which `python -O` strips) turns that into a
# clear import-time failure naming exactly what is missing.
_UNRANKED_TREATMENTS = frozenset(LayoutTreatment) - frozenset(_TREATMENT_RANK)
if _UNRANKED_TREATMENTS:
    raise RuntimeError(
        "every LayoutTreatment needs a _TREATMENT_RANK entry (propose_treatments "
        "sorts on it); missing: "
        + ", ".join(sorted(member.name for member in _UNRANKED_TREATMENTS))
    )


@dataclass(frozen=True)
class ArtDirectionRequest:
    """What the caller wants rendered and from which sources."""

    target: str
    target_width: int
    target_height: int
    sources: list[str] = field(default_factory=list)
    allow_diptych: bool = True
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "sources": list(self.sources),
            "allow_diptych": self.allow_diptych,
            "context": {str(k): v for k, v in self.context.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtDirectionRequest:
        if isinstance(data, cls):
            return data
        return cls(
            target=data["target"],
            target_width=int(data["target_width"]),
            target_height=int(data["target_height"]),
            sources=list(data.get("sources", [])),
            allow_diptych=bool(data.get("allow_diptych", True)),
            context=dict(data.get("context", {})),
        )


@dataclass(frozen=True)
class TreatmentProposal:
    """A proposed layout treatment with human-readable rationale and scores."""

    treatment: LayoutTreatment
    rationale: list[str] = field(default_factory=list)
    score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "treatment": self.treatment.value,
            "rationale": list(self.rationale),
            "score": self.score,
            "evidence": {str(k): v for k, v in self.evidence.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TreatmentProposal:
        if isinstance(data, cls):
            return data
        return cls(
            treatment=LayoutTreatment(data["treatment"]),
            rationale=list(data.get("rationale", [])),
            score=float(data["score"]),
            evidence=dict(data.get("evidence", {})),
        )


def propose_treatments(
    analysis: AnalysisResult | list[AnalysisResult],
    request: ArtDirectionRequest,
    provider: Any | None = None,
) -> list[TreatmentProposal]:
    """Rank applicable treatments for *analysis* under *request* (deterministic).

    *analysis* is a single :class:`AnalysisResult` (one source) or a list of them
    (one per source). ``provider`` is optional and, when supplied for a two-source
    request, is used to derive a real cross-image pairing affinity; otherwise the
    caller-supplied ``pairing.affinity`` of the first result is used.

    Multi-source eligibility (M010/S02) is driven by ``n``, the number of sources
    that have both an analysis result *and* a request entry: ``n == 2`` may yield
    a DIPTYCH, ``n == 3`` a TRIPTYCH and ``n == 4`` a QUAD. Extra results beyond
    the request's sources are never silently laid out.

    M010/S03 adds PACKED for **any** ``n`` in ``2..MAX_LAYOUT_SOURCES`` over the
    same ``NUP_AFFINITY`` gate, with the same score as the named template at that
    ``n`` — so ``_TREATMENT_RANK`` keeps a template ahead of PACKED at its own
    ``n``, and PACKED is the only multi-cell proposal from ``n == 5`` up.
    """
    results = [analysis] if isinstance(analysis, AnalysisResult) else list(analysis)
    if not results:
        return []

    n = min(len(results), len(request.sources))
    primary = results[0]
    aspect = _aspect(primary)
    proposals: list[TreatmentProposal] = []

    crop = primary.crop_safety
    safe_all, min_margin = _crop_safety_gate(primary)
    directions = {
        "north": crop.safe_north,
        "south": crop.safe_south,
        "east": crop.safe_east,
        "west": crop.safe_west,
    }
    unsafe = [d for d, ok in directions.items() if not ok]
    aesthetic = primary.quality.aesthetic_quality
    margins = {
        "north": crop.margin_north,
        "south": crop.margin_south,
        "east": crop.margin_east,
        "west": crop.margin_west,
    }

    if safe_all and min_margin >= MIN_FULLBLEED_MARGIN and aesthetic >= MIN_FULLBLEED_AESTHETIC:
        proposals.append(
            TreatmentProposal(
                treatment=LayoutTreatment.SINGLE_FULLBLEED,
                rationale=[
                    f"crop-safe in all directions (min margin {min_margin:.2f})",
                    f"aesthetic quality {aesthetic:.2f} >= {MIN_FULLBLEED_AESTHETIC}",
                ],
                score=round(0.6 * aesthetic + 0.4 * min_margin, 4),
                evidence={
                    "min_margin": min_margin,
                    "margins": margins,
                    "aesthetic_quality": aesthetic,
                },
            )
        )

    crop_risky = (not safe_all) or min_margin < CROP_RISK_MARGIN
    if crop_risky or not primary.quality.resolution_sufficient:
        reasons = []
        if unsafe:
            reasons.append(f"unsafe crop direction(s): {', '.join(sorted(unsafe))}")
        if min_margin < CROP_RISK_MARGIN:
            reasons.append(f"low crop margin ({min_margin:.2f})")
        if not primary.quality.resolution_sufficient:
            reasons.append("resolution insufficient for full-bleed")
        reasons.append("contain/matte is the safer fit")
        proposals.append(
            TreatmentProposal(
                treatment=LayoutTreatment.CONTAIN_MATTE,
                rationale=reasons,
                score=0.5,
                evidence={
                    "crop_risky": crop_risky,
                    "resolution_sufficient": primary.quality.resolution_sufficient,
                    "min_margin": min_margin,
                    "unsafe_directions": sorted(unsafe),
                    "background_choice": primary.color_story.background_choice,
                },
            )
        )

    if aspect is not None:
        if aspect >= PANORAMIC_MIN_ASPECT:
            proposals.append(
                TreatmentProposal(
                    treatment=LayoutTreatment.PANORAMIC,
                    rationale=[
                        f"wide aspect {aspect:.2f} >= {PANORAMIC_MIN_ASPECT}"
                    ],
                    score=round(min(1.0, aspect / 4.0), 4),
                    evidence={"aspect": aspect},
                )
            )
        if SQUARE_ASPECT_MIN <= aspect <= SQUARE_ASPECT_MAX:
            proposals.append(
                TreatmentProposal(
                    treatment=LayoutTreatment.SQUARE,
                    rationale=[
                        f"near-square aspect {aspect:.2f} in "
                        f"[{SQUARE_ASPECT_MIN}, {SQUARE_ASPECT_MAX}]"
                    ],
                    score=round(
                        max(0.1, 0.5 * (aesthetic + 1.0 - abs(aspect - 1.0))), 4
                    ),
                    evidence={"aspect": aspect, "aesthetic_quality": aesthetic},
                )
            )

    # ``allow_diptych`` stays DIPTYCH-specific (M010/S02): it is an existing
    # public field of ArtDirectionRequest whose name states exactly one
    # treatment, so the N-up block below deliberately does *not* read it rather
    # than silently repurposing it as a general "multi-source allowed" flag.
    if (
        request.allow_diptych
        and len(results) >= 2
        and len(request.sources) >= 2
    ):
        affinity = _pair_affinity(results, provider)
        if affinity >= DIPTYCH_AFFINITY:
            # A diptych is a two-cell layout like any other, so M010/S05 gives it
            # the same per-cell fit record: without `cells` in evidence its crop
            # decision could not survive the proposals-table round trip that
            # `cli._manifest` performs, and both cells would silently letterbox.
            pair = results[:2]
            pair_shas = list(request.sources[:2])
            pair_target = (request.target_width, request.target_height)
            pair_cells = equal_cells(
                pair_shas,
                Cell(0, 0, pair_target[0], pair_target[1]),
                gap=gutter_for_target(pair_target),
            )
            pair_fits = _cell_fits(pair, pair_cells)
            proposals.append(
                TreatmentProposal(
                    treatment=LayoutTreatment.DIPTYCH,
                    rationale=[
                        f"pair affinity {affinity:.2f} >= {DIPTYCH_AFFINITY} with "
                        f"two sources",
                        _fit_rationale(pair_fits),
                    ],
                    score=round(affinity, 4),
                    evidence={
                        "affinity": affinity,
                        "phash_distance": results[1].pairing.phash_distance,
                        "palette_distance": results[1].pairing.palette_distance,
                        "orientation_match": results[1].pairing.orientation_match,
                        "cells": [
                            {
                                "sha": cell.source_sha256,
                                "x": cell.x,
                                "y": cell.y,
                                "w": cell.w,
                                "h": cell.h,
                                **fit,
                            }
                            for cell, fit in zip(pair_cells, pair_fits, strict=True)
                        ],
                    },
                )
            )

    # One affinity computation serves both the named template (when there is one
    # at this N) and PACKED, which covers every N in 2..MAX_LAYOUT_SOURCES
    # (M010/S03) — including the N=3/N=4 where a template also applies, so
    # `--treatment packed` is selectable at any N.
    if 2 <= n <= MAX_LAYOUT_SOURCES:
        group = results[:n]
        group_affinity = _group_affinity(group, provider)
        if group_affinity >= NUP_AFFINITY:
            nup_treatment = _NUP_TREATMENTS.get(n)
            if nup_treatment is not None:
                proposals.append(
                    _nup_proposal(
                        nup_treatment, group, request, provider, group_affinity
                    )
                )
            proposals.append(
                _packed_proposal(group, request, provider, group_affinity)
            )

    proposals.sort(key=lambda p: (-p.score, _TREATMENT_RANK[p.treatment]))
    return proposals


def propose(
    source_shas: list[str],
    request: ArtDirectionRequest,
    provider: LocalAnalysisProvider | None = None,
    *,
    analysis: AnalysisResult | list[AnalysisResult] | None = None,
) -> list[TreatmentProposal]:
    """Convenience wrapper: accept precomputed *analysis* or analyze on the fly.

    When *analysis* is supplied, it is passed straight through. Otherwise
    *provider* (a :class:`~curator.analysis.local.LocalAnalysisProvider`) analyzes
    each of *source_shas* (treated as image paths) before proposing. With neither
    given, an empty proposal list is returned.
    """
    if analysis is not None:
        return propose_treatments(analysis, request, provider=provider)
    if provider is None:
        return []
    results: list[AnalysisResult] = [provider.analyze(sha) for sha in source_shas]
    return propose_treatments(results, request, provider=provider)


def materialize_manifest(
    proposal: TreatmentProposal,
    request: ArtDirectionRequest,
    sources_sha: list[str],
    rationale: list[str] | None = None,
) -> ArtDirectionManifest:
    """Build an :class:`ArtDirectionManifest` from a chosen *proposal*.

    Sources are recorded verbatim with one
    :class:`~curator.artdirection.manifest.SourceRegion` per source, carrying
    real output-canvas-pixel geometry from
    :func:`~curator.artdirection.packing.equal_cells` (M010/S01). This is the
    first production writer of region geometry — before M010 every materialized
    region was four zeros, i.e. unset. A diptych sets ``pairing_order`` to the
    source order and is packed in that order. A contain-matte proposal carries
    its background choice through ``evidence`` and materializes it into the
    manifest's :class:`BackgroundSpec`.

    The pack target is ``request.target_width`` / ``request.target_height``, so
    the signature is deliberately unchanged, even now that M010/S03 packs by
    weight: the weights are read from ``proposal.evidence["weights"]``, not from
    a new parameter. That is load-bearing rather than stylistic — ``cli._manifest``
    re-materializes from a proposal **reloaded out of the ``proposals`` table**,
    so a weight passed as an argument by ``curator propose`` would silently
    become uniform on the following ``curator manifest``. A proposal with no
    stored weights (every single-source treatment, and every pre-S03 row) packs
    uniformly, which is exactly S01's behavior.

    **This function is the only writer of** ``crop="fill"`` **in the codebase**
    (M010/S05), and it only ever copies a value the crop-safety gate in
    :func:`_cell_fit` already approved for that source. That invariant is
    load-bearing rather than tidy: the renderer holds no
    :class:`~curator.analysis.schema.AnalysisResult` at render time and therefore
    *cannot* re-verify crop safety, so it trusts this gate. Each crop is read out
    of ``proposal.evidence["cells"]`` — keyed by sha, so a caller-reordered source
    list moves each verdict with its own source rather than onto a neighbour's
    cell — and validated against
    :data:`~curator.artdirection.manifest.VALID_CROP_MODES`.

    Raises :class:`~curator.artdirection.manifest.ManifestError` when more than
    :data:`~curator.artdirection.manifest.MAX_LAYOUT_SOURCES` sources are given,
    when a fixed-size named template is handed the wrong number of sources
    (M010/S02), or when the stored per-cell fit is out of vocabulary or does not
    cover this source count — rather than materializing a manifest
    :meth:`ArtDirectionManifest.validate` would reject, or one the renderer would
    quietly truncate.
    """
    if len(sources_sha) > MAX_LAYOUT_SOURCES:
        raise ManifestError(
            f"cannot materialize a manifest for {len(sources_sha)} source(s), over the "
            f"{MAX_LAYOUT_SOURCES}-source layout cap — an over-cap request is "
            f"rejected, never truncated"
        )
    reasons = list(rationale) if rationale is not None else list(proposal.rationale)
    treatment = proposal.treatment
    required = _TREATMENT_SOURCE_COUNT.get(treatment)
    if required is not None and len(sources_sha) != required:
        raise ManifestError(
            f"{treatment.value} requires exactly {required} source(s), got "
            f"{len(sources_sha)} — a count mismatch is rejected, never truncated"
        )
    pairing_order = list(sources_sha) if treatment in MULTI_CELL_TREATMENTS else []
    background = BackgroundSpec()
    if treatment is LayoutTreatment.CONTAIN_MATTE:
        choice = proposal.evidence.get("background_choice")
        background = BackgroundSpec(
            background_choice=str(choice) if choice is not None else "none"
        )
    target = (request.target_width, request.target_height)
    order = pairing_order or list(sources_sha)
    weights = _manifest_weights(proposal, sources_sha)
    regions = slice_cells(
        [
            WeightedSource(sha, weight)
            for sha, weight in zip(order, weights, strict=True)
        ],
        Cell(0, 0, request.target_width, request.target_height),
        gap=gutter_for_target(target),
    )
    crops = _manifest_crops(proposal, sources_sha)
    regions = [
        replace(region, crop=crops.get(region.source_sha256))
        for region in regions
    ]
    return ArtDirectionManifest(
        manifest_version=MANIFEST_VERSION,
        sources=list(sources_sha),
        regions=regions,
        layout_treatment=treatment,
        background=background,
        pairing_order=pairing_order,
        rationale=reasons,
    )


def _coerce_weights(value: Any, count: int, *, origin: str) -> list[float]:
    """Return *value* as exactly *count* floats, or raise :class:`ManifestError`.

    The one validator for both weight trust boundaries (M010/S03): a caller's
    ``request.context["weights"]`` and a persisted proposal's
    ``evidence["weights"]`` reloaded out of SQLite. *origin* names which one, so
    the message is actionable at the surface the caller actually used. Rejecting
    a length mismatch here is what stops a hand-edited or stale row from packing
    N sources against M weights.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ManifestError(
            f"{origin} must be a list of {count} number(s), got {type(value).__name__}"
        )
    if len(value) != count:
        raise ManifestError(
            f"{origin} has {len(value)} weight(s) for {count} source(s) — one "
            f"weight per source is required"
        )
    weights: list[float] = []
    for item in value:
        try:
            weights.append(float(item))
        except (TypeError, ValueError):
            raise ManifestError(
                f"{origin} value is not a number: {item!r}"
            ) from None
    return weights


def _manifest_weights(
    proposal: TreatmentProposal, sources_sha: list[str]
) -> list[float]:
    """Return the per-source weights *proposal* was packed with (M010/S03).

    Read back from ``evidence["weights"]`` so a manifest materialized from a
    reloaded proposal reproduces the proposal's own geometry exactly. A proposal
    that never recorded weights (every single-source treatment, every pre-S03
    row) packs uniformly.
    """
    stored = proposal.evidence.get("weights")
    if stored is None:
        return [1.0] * len(sources_sha)
    return _coerce_weights(stored, len(sources_sha), origin="proposal evidence")


def _manifest_crops(
    proposal: TreatmentProposal, sources_sha: list[str]
) -> dict[str, str | None]:
    """Return the per-source fit *proposal* recorded, keyed by sha (M010/S05).

    Read back out of ``evidence["cells"]`` for the same reason weights are
    (M010/S03): ``cli._manifest`` re-materializes from a proposal **reloaded out
    of the ``proposals`` table**, so a crop decision held in a local would
    silently become "letterbox everything" on the following command.

    A proposal with no recorded cells — every single-source treatment, and every
    row written before M010/S05 — yields no crops at all, i.e. every cell
    letterboxes, which is exactly the pre-slice behavior. A cell count that does
    not match the source count is a stale or hand-edited row and is **rejected**,
    never silently applied to the wrong cells. A value outside
    :data:`~curator.artdirection.manifest.VALID_CROP_MODES` is rejected here so a
    caller learns at manifest time, not at render time.
    """
    cells = proposal.evidence.get("cells")
    if cells is None:
        return {}
    if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence):
        raise ManifestError(
            f"proposal evidence cells must be a list, got {type(cells).__name__}"
        )
    if len(cells) != len(sources_sha):
        raise ManifestError(
            f"proposal evidence has {len(cells)} cell(s) for "
            f"{len(sources_sha)} source(s) — one cell per source is required"
        )
    crops: dict[str, str | None] = {}
    for cell, sha in zip(cells, sources_sha, strict=True):
        entry = cell if isinstance(cell, dict) else {}
        value = entry.get("crop")
        if value is None or value == "":
            crops[str(entry.get("sha", sha))] = None
            continue
        if value not in VALID_CROP_MODES:
            raise ManifestError(
                f"unknown cell crop mode {value!r} — accepted values are "
                f"{sorted(VALID_CROP_MODES)} or null (letterbox)"
            )
        crops[str(entry.get("sha", sha))] = str(value)
    return crops


def _packed_weights(
    results: list[AnalysisResult], request: ArtDirectionRequest
) -> tuple[list[float], str]:
    """Return the packing weights for *results* plus the literal naming their source.

    The locked default is each source's ``quality.aesthetic_quality`` — an
    explicit, inspectable signal rather than an inferred one, matching this
    repo's "providers propose, policy decides" posture. A caller may override it
    per request through ``context["weights"]`` (validated to one float per
    source). Either way the provenance is recorded, and a vector that would
    degenerate to uniform inside the packer reports ``"uniform_fallback"`` rather
    than claiming a provenance the geometry did not actually use — the predicate
    below is the same one ``packing._usable_weights`` applies.
    """
    override = request.context.get("weights")
    if override is None:
        weights = [float(r.quality.aesthetic_quality) for r in results]
        source = "quality.aesthetic_quality"
    else:
        weights = _coerce_weights(override, len(results), origin="weights override")
        source = "caller_override"
    if sum(w for w in weights if math.isfinite(w) and w > 0.0) <= 0.0:
        return [1.0] * len(results), "uniform_fallback"
    return weights, source


def _packed_proposal(
    results: list[AnalysisResult],
    request: ArtDirectionRequest,
    provider: Any | None,
    affinity: float,
) -> TreatmentProposal:
    """Build the PACKED proposal for *results* under *request* (M010/S03).

    ``evidence["cells"]`` comes from the *same*
    :func:`~curator.artdirection.packing.slice_cells` call
    :func:`materialize_manifest` will make from ``evidence["weights"]``, so the
    advertised geometry and the materialized regions agree by construction rather
    than by convention — and each cell carries the weight that produced it, plus
    its own :func:`_cell_fit` record (M010/S05), so a human can check the
    rationale against the boxes.
    """
    shas = list(request.sources[: len(results)])
    weights, weight_source = _packed_weights(results, request)
    target = (request.target_width, request.target_height)
    gap = gutter_for_target(target)
    items = [
        WeightedSource(sha, weight)
        for sha, weight in zip(shas, weights, strict=True)
    ]
    cells = slice_cells(items, Cell(0, 0, target[0], target[1]), gap=gap)
    fits = _cell_fits(results, cells)
    vertical = target[0] >= target[1]
    cut_axis = "vertical" if vertical else "horizontal"
    near, far = ("left", "right") if vertical else ("top", "bottom")
    split = _bisect_by_weight(items)
    pairwise = dict(
        zip(shas[1:], _pairwise_affinities(results, provider), strict=True)
    )
    return TreatmentProposal(
        treatment=LayoutTreatment.PACKED,
        rationale=[
            f"{len(shas)} cells packed by importance "
            f"(weights {'/'.join(f'{w:.2f}' for w in weights)})",
            f"root cut {cut_axis}, {split} cell(s) {near} / "
            f"{len(shas) - split} {far}, {gap}px gutter",
            f"mean group affinity {affinity:.2f} >= {NUP_AFFINITY}",
            _fit_rationale(fits),
        ],
        score=round(affinity, 4),
        evidence={
            "sources": len(shas),
            "group_affinity": affinity,
            "affinity_source": "pairing.affinity",
            "pairwise_affinity": pairwise,
            "weights": weights,
            "weight_source": weight_source,
            "gap": gap,
            # The *root* cut only; deeper cuts alternate by box aspect and are
            # readable from the cell geometry below.
            "cut_axis": cut_axis,
            "cells": [
                {
                    "sha": cell.source_sha256,
                    "x": cell.x,
                    "y": cell.y,
                    "w": cell.w,
                    "h": cell.h,
                    "weight": weight,
                    **fit,
                }
                for cell, weight, fit in zip(cells, weights, fits, strict=True)
            ],
        },
    )


def _aspect(result: AnalysisResult) -> float | None:
    """Return the composition aspect ratio (w / h) from ``saliency.map_size``.

    ``saliency.map_size`` doubles as the image aspect and is already computed and
    stored, so a cell never has to rediscover its source's shape at placement
    time. M010/S05 calls this **once per candidate** rather than only for
    ``results[0]``: every cell's fit is decided from its own source.
    """
    w, h = result.saliency.map_size
    if w <= 0 or h <= 0:
        return None
    return float(w) / float(h)


def _crop_safety_gate(result: AnalysisResult) -> tuple[bool, float]:
    """Return *result*'s ``(safe in all four directions, minimum margin)``.

    The single expression behind both crop approvals in this module: whether a
    whole frame may go SINGLE_FULLBLEED, and — from M010/S05 — whether one cell
    of an N-up may crop to fill. Both call sites compare ``min_margin`` against
    :data:`MIN_FULLBLEED_MARGIN` themselves; there is deliberately no second
    threshold, because a per-cell crop is the same act as a full-bleed crop
    performed on a smaller canvas.
    """
    safety = result.crop_safety
    safe_all = (
        safety.safe_north
        and safety.safe_south
        and safety.safe_east
        and safety.safe_west
    )
    min_margin = min(
        safety.margin_north,
        safety.margin_south,
        safety.margin_east,
        safety.margin_west,
    )
    return safe_all, min_margin


def _cell_fit(result: AnalysisResult, cell: SourceRegion) -> dict[str, Any]:
    """Decide how one cell fits its own source, with the inputs that decided it.

    A cell is marked :data:`~curator.artdirection.manifest.CROP_FILL` only when
    **all three** hold for *its own* source (M010/S05):

    1. ``crop_safety`` is safe in all four directions;
    2. the minimum of the four margins is at least
       :data:`MIN_FULLBLEED_MARGIN` — the same gate, and the same constant,
       ``SINGLE_FULLBLEED`` uses for a whole frame;
    3. the cell's aspect departs from the source's by more than
       :data:`CELL_CROP_ASPECT_TOLERANCE`, so the crop actually buys canvas.

    Otherwise the fit is ``None``: **letterbox is the default**, exactly as every
    multi-region treatment has always behaved. A source with no usable
    ``saliency.map_size`` has no aspect to compare and therefore letterboxes.

    The returned record carries the verdict *and* its four inputs
    (``source_aspect``, ``cell_aspect``, ``crop_safe``, ``min_margin``), so a
    reader can check the decision rather than trust it.
    """
    crop_safe, min_margin = _crop_safety_gate(result)
    source_aspect = _aspect(result)
    cell_aspect = float(cell.w) / float(cell.h) if cell.h else None
    crop: str | None = None
    if (
        crop_safe
        and min_margin >= MIN_FULLBLEED_MARGIN
        and source_aspect is not None
        and cell_aspect is not None
        and abs(cell_aspect - source_aspect) / source_aspect
        > CELL_CROP_ASPECT_TOLERANCE
    ):
        crop = CROP_FILL
    return {
        "crop": crop,
        "source_aspect": source_aspect,
        "cell_aspect": cell_aspect,
        "crop_safe": crop_safe,
        "min_margin": min_margin,
    }


def _cell_fits(
    results: Sequence[AnalysisResult], cells: Sequence[SourceRegion]
) -> list[dict[str, Any]]:
    """Zip *results* with *cells* into one :func:`_cell_fit` record each.

    ``strict=True``: a result count that does not match the cell count is a
    programming error in the caller, never a silently shortened layout.
    """
    return [
        _cell_fit(result, cell)
        for result, cell in zip(results, cells, strict=True)
    ]


def _fit_rationale(fits: Sequence[dict[str, Any]]) -> str:
    """One human-checkable line naming how many cells fill, how many letterbox."""
    filled = sum(1 for fit in fits if fit["crop"] == CROP_FILL)
    return (
        f"{filled} cell(s) crop-to-fill, {len(fits) - filled} letterbox — a cell "
        f"fills only when its own source is crop-safe in all four directions "
        f"with min margin >= {MIN_FULLBLEED_MARGIN} and its aspect differs from "
        f"the cell's by more than {CELL_CROP_ASPECT_TOLERANCE:.0%}"
    )


def _pair_affinity(
    results: list[AnalysisResult], provider: Any | None
) -> float:
    """Return the cross-image affinity for a two-source request."""
    if provider is not None:
        try:
            return float(provider.pairing_scores_between(results[0], results[1]).affinity)
        except Exception:
            return float(results[0].pairing.affinity)
    return float(results[0].pairing.affinity)


def _affinity_between(
    primary: AnalysisResult, other: AnalysisResult, provider: Any | None
) -> float:
    """Return the affinity between *primary* and *other* (M010/S02).

    Mirrors :func:`_pair_affinity`'s shape — a real cross-image score from
    *provider* when one is supplied, falling back to the stored
    ``pairing.affinity`` on any provider failure. The fallback is per comparison
    and reads *other*'s recorded affinity, so one unanalyzed pair cannot poison
    a whole group's score.
    """
    if provider is not None:
        try:
            return float(provider.pairing_scores_between(primary, other).affinity)
        except Exception:
            return float(other.pairing.affinity)
    return float(other.pairing.affinity)


def _pairwise_affinities(
    results: list[AnalysisResult], provider: Any | None
) -> list[float]:
    """Return the affinity of ``results[0]`` against each of ``results[1:]``.

    **N-1 comparisons, never N x N.** The primary is the fixed reference point,
    so the cost is linear in the group size and no full affinity matrix is ever
    materialized — the property a future whole-library grouping feature must
    keep rather than rediscover.
    """
    return [_affinity_between(results[0], other, provider) for other in results[1:]]


def _group_affinity(
    results: list[AnalysisResult], provider: Any | None
) -> float:
    """Return the mean of :func:`_pairwise_affinities` (M010/S02).

    The N-up eligibility signal: how well the group coheres around its primary.
    Returns ``0.0`` for a group of one, which no N-up template accepts anyway.
    Streaming by construction — see :func:`_pairwise_affinities` for why this is
    N-1 comparisons and not an N x N matrix.
    """
    scores = _pairwise_affinities(results, provider)
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def _nup_proposal(
    treatment: LayoutTreatment,
    results: list[AnalysisResult],
    request: ArtDirectionRequest,
    provider: Any | None,
    affinity: float,
) -> TreatmentProposal:
    """Build the TRIPTYCH/QUAD proposal for *results* under *request* (M010/S02).

    ``evidence["cells"]`` is computed with the *same*
    :func:`~curator.artdirection.packing.equal_cells` call
    :func:`materialize_manifest` makes, so the proposal's advertised geometry and
    the materialized manifest's regions agree by construction rather than by
    convention. ``evidence["affinity_source"]`` names *which* affinity produced
    the score — the analysis pipeline's ``pairing.affinity``, never an embedding
    cosine (M010/S04 emits a different literal for its own question). Each cell
    also carries its own :func:`_cell_fit` record (M010/S05).
    """
    shas = list(request.sources[: len(results)])
    target = (request.target_width, request.target_height)
    gap = gutter_for_target(target)
    cells = equal_cells(shas, Cell(0, 0, target[0], target[1]), gap=gap)
    fits = _cell_fits(results, cells)
    cut_axis = "vertical" if target[0] >= target[1] else "horizontal"
    pairwise = dict(
        zip(shas[1:], _pairwise_affinities(results, provider), strict=True)
    )
    return TreatmentProposal(
        treatment=treatment,
        rationale=[
            f"{len(shas)} sources, mean group affinity {affinity:.2f} >= {NUP_AFFINITY}",
            f"{len(shas)} equal cells on a {target[0]}x{target[1]} canvas, "
            f"{cut_axis} cut, {gap}px gutter",
            _fit_rationale(fits),
        ],
        score=round(affinity, 4),
        evidence={
            "sources": len(shas),
            "group_affinity": affinity,
            "affinity_source": "pairing.affinity",
            "pairwise_affinity": pairwise,
            "gap": gap,
            "cut_axis": cut_axis,
            "cells": [
                {
                    "sha": cell.source_sha256,
                    "x": cell.x,
                    "y": cell.y,
                    "w": cell.w,
                    "h": cell.h,
                    **fit,
                }
                for cell, fit in zip(cells, fits, strict=True)
            ],
        },
    )
