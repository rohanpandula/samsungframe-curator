"""Deterministic art-direction policy engine (M002/S03 T2+T3, M010/S02).

Given one or more :class:`~curator.analysis.schema.AnalysisResult` fixtures and
an :class:`ArtDirectionRequest`, :func:`propose_treatments` ranks which
:class:`~curator.artdirection.manifest.LayoutTreatment` choices are applicable
and returns them as ``TreatmentProposal`` objects ordered by score descending.
Single-source treatments read the primary result; the multi-source ones —
DIPTYCH at two sources, TRIPTYCH at three and QUAD at four (M010/S02) — are
gated on a cross-image affinity and carry their cell geometry as evidence.

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

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from curator.analysis.schema import AnalysisResult
from curator.artdirection.manifest import (
    _TREATMENT_SOURCE_COUNT,
    MANIFEST_VERSION,
    MAX_LAYOUT_SOURCES,
    MULTI_CELL_TREATMENTS,
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
    ManifestError,
)
from curator.artdirection.packing import Cell, equal_cells, gutter_for_target

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
    LayoutTreatment.SINGLE_FULLBLEED: 3,
    LayoutTreatment.CONTAIN_MATTE: 4,
    LayoutTreatment.PANORAMIC: 5,
    LayoutTreatment.SQUARE: 6,
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
    """
    results = [analysis] if isinstance(analysis, AnalysisResult) else list(analysis)
    if not results:
        return []

    n = min(len(results), len(request.sources))
    primary = results[0]
    aspect = _aspect(primary)
    proposals: list[TreatmentProposal] = []

    crop = primary.crop_safety
    safe_all = crop.safe_north and crop.safe_south and crop.safe_east and crop.safe_west
    directions = {
        "north": crop.safe_north,
        "south": crop.safe_south,
        "east": crop.safe_east,
        "west": crop.safe_west,
    }
    unsafe = [d for d, ok in directions.items() if not ok]
    min_margin = min(crop.margin_north, crop.margin_south, crop.margin_east, crop.margin_west)
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
            proposals.append(
                TreatmentProposal(
                    treatment=LayoutTreatment.DIPTYCH,
                    rationale=[
                        f"pair affinity {affinity:.2f} >= {DIPTYCH_AFFINITY} with "
                        f"two sources"
                    ],
                    score=round(affinity, 4),
                    evidence={
                        "affinity": affinity,
                        "phash_distance": results[1].pairing.phash_distance,
                        "palette_distance": results[1].pairing.palette_distance,
                        "orientation_match": results[1].pairing.orientation_match,
                    },
                )
            )

    nup_treatment = _NUP_TREATMENTS.get(n)
    if nup_treatment is not None:
        group = results[:n]
        group_affinity = _group_affinity(group, provider)
        if group_affinity >= NUP_AFFINITY:
            proposals.append(
                _nup_proposal(nup_treatment, group, request, provider, group_affinity)
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
    the signature is deliberately unchanged: 01-PATTERNS.md sketched extra
    ``target`` / ``weights`` parameters, which are unnecessary here because the
    request already carries the dims and S01 packs at equal weight. M010/S03,
    which adds a real weights path, is where that sketch becomes relevant.

    Raises :class:`~curator.artdirection.manifest.ManifestError` when more than
    :data:`~curator.artdirection.manifest.MAX_LAYOUT_SOURCES` sources are given,
    or when a fixed-size named template is handed the wrong number of sources
    (M010/S02) — rather than materializing a manifest
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
    regions = equal_cells(
        pairing_order or sources_sha,
        Cell(0, 0, request.target_width, request.target_height),
        gap=gutter_for_target(target),
    )
    return ArtDirectionManifest(
        manifest_version=MANIFEST_VERSION,
        sources=list(sources_sha),
        regions=regions,
        layout_treatment=treatment,
        background=background,
        pairing_order=pairing_order,
        rationale=reasons,
    )


def _aspect(result: AnalysisResult) -> float | None:
    """Return the composition aspect ratio (w / h) from ``saliency.map_size``."""
    w, h = result.saliency.map_size
    if w <= 0 or h <= 0:
        return None
    return float(w) / float(h)


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
    cosine (M010/S04 emits a different literal for its own question).
    """
    shas = list(request.sources[: len(results)])
    target = (request.target_width, request.target_height)
    gap = gutter_for_target(target)
    cells = equal_cells(shas, Cell(0, 0, target[0], target[1]), gap=gap)
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
                }
                for cell in cells
            ],
        },
    )
