"""Analysis profile contract + ordered pipeline-stage specs (M002/S01).

``AnalysisProfile`` selects how much of the analysis pipeline runs: a higher
profile is a strict superset (prefix) of a lower one, so results never regress
when moving down the scale. :func:`profile_specs` returns that ordered stage
list; :func:`custom_profile` pins an explicit, validated stage sequence for
inflexible policy requirements.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from curator.analysis.errors import AnalysisError

#: Known pipeline stages in the curated canonical order. S02's cascade and S03's
#: policy consume these stage names as opaque strings.
KNOWN_STAGES: frozenset[str] = frozenset(
    {
        "perceptual",
        "technical",
        "aesthetic",
        "saliency",
        "cropsafety",
        "colorstory",
        "pairing",
    }
)

# Canonical pipeline order (ascending rank). Used to build StageSpec ranks.
_CANONICAL_ORDER: tuple[str, ...] = (
    "perceptual",
    "technical",
    "aesthetic",
    "saliency",
    "cropsafety",
    "colorstory",
    "pairing",
)


class AnalysisProfile(enum.Enum):
    """An analysis workload profile — higher profiles run more pipeline stages."""

    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    MAX = "max"
    CUSTOM = "custom"


#: Monotonic nesting index: FAST ⊂ BALANCED ⊂ QUALITY ⊂ MAX (strict less-than).
PROFILE_ORDER: dict[AnalysisProfile, int] = {
    AnalysisProfile.FAST: 0,
    AnalysisProfile.BALANCED: 1,
    AnalysisProfile.QUALITY: 2,
    AnalysisProfile.MAX: 3,
}


@dataclass(frozen=True)
class StageSpec:
    """One ordered pipeline stage."""

    stage: str
    rank: int


def profile_order(a: AnalysisProfile, b: AnalysisProfile) -> int:
    """Strict less-than ordering over profiles.

    Returns ``-1``/``0``/``1`` for *a* < / == / > *b*. ``CUSTOM`` is unordered
    relative to the built-ins and returns ``0`` (equal) to keep comparisons total
    and deterministic.
    """
    if a is b:
        return 0
    ra = PROFILE_ORDER.get(a)
    rb = PROFILE_ORDER.get(b)
    if ra is None or rb is None:
        return 0  # CUSTOM is not on the built-in nesting ladder
    return -1 if ra < rb else 1


# Per-profile stage sequences, monotonically nested (each is a strict prefix of
# the next). Ranks follow the canonical order so StageSpec ordering is explicit.
_PROFILE_STAGES: dict[AnalysisProfile, tuple[str, ...]] = {
    AnalysisProfile.FAST: ("perceptual", "technical"),
    AnalysisProfile.BALANCED: ("perceptual", "technical", "aesthetic", "saliency"),
    AnalysisProfile.QUALITY: (
        "perceptual",
        "technical",
        "aesthetic",
        "saliency",
        "cropsafety",
        "colorstory",
    ),
    AnalysisProfile.MAX: (
        "perceptual",
        "technical",
        "aesthetic",
        "saliency",
        "cropsafety",
        "colorstory",
        "pairing",
    ),
}


def _ranks(stages: tuple[str, ...]) -> list[StageSpec]:
    """Turn a stage-name tuple into ordered :class:`StageSpec` values."""
    return [
        StageSpec(stage=name, rank=rank)
        for rank, name in enumerate(stages)
    ]


def profile_specs(profile: AnalysisProfile) -> list[StageSpec]:
    """Return the ordered pipeline-stage list for *profile*.

    Nested monotonicity guarantee: every stage of a lower profile appears, in
    order, as a prefix of every higher profile's list.
    """
    if profile is AnalysisProfile.CUSTOM:
        raise AnalysisError(
            "CUSTOM profile has no fixed stage list; use custom_profile(stages)"
        )
    return _ranks(_PROFILE_STAGES[profile])


def custom_profile(stages: list[str]) -> list[StageSpec]:
    """Build a pinned :class:`StageSpec` list from explicit stage names.

    Validates every stage against :data:`KNOWN_STAGES` and raises
    :class:`AnalysisError` on an unknown stage name. Ranks follow the caller's
    order, so a custom pipeline may be any subset in any order.
    """
    unknown = [name for name in stages if name not in KNOWN_STAGES]
    if unknown:
        raise AnalysisError(
            f"unknown pipeline stage(s): {sorted(unknown)} "
            f"(known: {sorted(KNOWN_STAGES)})"
        )
    return _ranks(tuple(stages))
