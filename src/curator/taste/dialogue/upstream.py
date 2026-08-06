"""The Taste Profile as the upstream taste signal (M008/S05, R036 + R038).

Existing taste features change their *source of truth*, not their UX: Taste Lens
(M007/S01) rerank explanations cite profile entries by the user's verbatim,
Taste Lens Discovery (M007/S04) ranks by profile fit with the Familiar↔Surprising
dial moving along profile dimensions, and M002 pairing rationale may cite the
profile. Nothing is gated on the new feature — every function here returns the
untouched baseline for an empty (or ``None``) profile, so an installation that
never opens the Reaction Room behaves exactly as it did before M008.

Two R038 anti-goals are structural rather than tested-after-the-fact:

* **No jargon laundering** — a citation's surface text is
  :attr:`ProfileCitation.quote`, the user's own words, byte-exact from the
  observation. Controlled-vocabulary attributes are metadata *about* the quote,
  never a replacement for it.
* **Swipes/approvals demoted** — claims seeded from approve/reject and pairwise
  history (``provenance="low"``, see
  :class:`~curator.taste.dialogue.profile.ColdStartSeeder`) are scaled by
  :data:`CORROBORATING_WEIGHT` so they corroborate the user's stated taste
  instead of driving it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from curator.analysis.schema import AnalysisResult
from curator.taste.dialogue.profile import TasteClaim, TasteProfile
from curator.taste.profiles import SIGNAL_NAMES
from curator.taste.profiles import TasteProfile as LensProfile
from curator.taste.rank import TasteRanker

#: Scale applied to low-provenance (approval/pairwise-derived) claims — they
#: corroborate the profile's stated taste, they never drive it (R036).
CORROBORATING_WEIGHT = 0.25

#: Default number of quotes a single explanation cites (never a lecture).
DEFAULT_CITATION_LIMIT = 2

#: Controlled-vocabulary attribute -> M002 signal leanings, each in ``-1..1``.
#: Hand-authored and deliberately coarse: this maps *what the user said* onto the
#: axes the existing rankers already understand, so profile fit is expressed in
#: M002 signal space without inventing new analysis. ``vibrancy`` is not listed —
#: it aliases ``colorfulness`` (see :class:`~curator.taste.rank.TasteRanker`) and
#: is mirrored in automatically by :func:`attribute_signal_leanings`.
_ATTRIBUTE_SIGNALS: dict[str, dict[str, float]] = {
    "negative-space": {"colorfulness": -0.3, "aesthetic_quality": 0.2},
    "muted-palette": {"colorfulness": -0.5},
    "lone-subject": {"aesthetic_quality": 0.3, "pairing_affinity": -0.2},
    "symmetry": {"harmony": 0.5, "aesthetic_quality": 0.2},
    "warm-tones": {"colorfulness": 0.4, "harmony": 0.3},
    "high-contrast": {"technical_quality": 0.3, "colorfulness": 0.2},
    "breathing-room": {"aesthetic_quality": 0.3, "colorfulness": -0.2},
    "texture": {"technical_quality": 0.4},
    "motion": {"aesthetic_quality": 0.2, "technical_quality": -0.2},
    "repetition": {"harmony": 0.4, "pairing_affinity": 0.3},
    "minimal": {"colorfulness": -0.4, "harmony": 0.3},
    "dense": {"colorfulness": 0.4, "harmony": -0.2},
    "geometric": {"harmony": 0.4, "technical_quality": 0.2},
    "organic": {"harmony": 0.2, "colorfulness": 0.2},
    "nostalgic": {"colorfulness": -0.2},
    "quiet": {"colorfulness": -0.4, "aesthetic_quality": 0.2},
    "heavy": {"colorfulness": 0.2, "aesthetic_quality": -0.1},
    "light": {"colorfulness": -0.1, "aesthetic_quality": 0.2},
}

#: Claim-id prefixes: dialogue patterns vs cold-start history claims.
_PATTERN_PREFIX = "pattern:"
_HISTORY_PREFIX = "history:"

#: Marker written by :class:`~curator.taste.dialogue.profile.ColdStartSeeder`
#: for the direction of a history claim. Coupled to that claim text by design —
#: both live in this milestone and are covered by the same tests.
_HIGH_MARKER = "lean toward high "


@dataclass(frozen=True)
class ProfileCitation:
    """One profile entry quoted by an upstream explanation.

    ``quote`` is the user's verbatim, byte-exact — the surface text of every
    citation. ``usage_count`` is how many pieces of evidence back the claim
    ("you've called this 'quiet' 3 times").
    """

    claim_id: str
    quote: str
    usage_count: int
    provenance: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "quote": self.quote,
            "usage_count": self.usage_count,
            "provenance": self.provenance,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProfileCitation:
        if isinstance(data, ProfileCitation):
            return data
        fields = dict(data)
        return cls(
            claim_id=str(fields["claim_id"]),
            quote=str(fields["quote"]),
            usage_count=int(fields.get("usage_count", 0)),
            provenance=str(fields.get("provenance", "high")),
            confidence=float(fields.get("confidence", 0.0)),
        )

    def render(self) -> str:
        """Render the citation the way an explanation says it out loud."""
        times = "time" if self.usage_count == 1 else "times"
        return f"you've called this {self.quote!r} {self.usage_count} {times}"


@dataclass(frozen=True)
class RankExplanation:
    """A rerank explanation: human ``rationale`` + machine ``evidence``.

    ``rationale`` follows the repo-wide proposal convention (human-readable text
    alongside machine evidence). With an empty profile it is exactly the M007
    baseline string and ``citations`` is empty — no consumer can tell M008 is
    installed.
    """

    rationale: str
    citations: list[ProfileCitation]
    evidence: list[dict[str, Any]]
    delta: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "citations", list(self.citations))
        object.__setattr__(self, "evidence", [dict(e) for e in self.evidence])

    def to_dict(self) -> dict[str, Any]:
        return {
            "rationale": self.rationale,
            "citations": [c.to_dict() for c in self.citations],
            "evidence": [dict(e) for e in self.evidence],
            "delta": self.delta,
        }


def is_empty(profile: TasteProfile | None) -> bool:
    """True when *profile* carries no claim any consumer could cite or rank on."""
    if profile is None:
        return True
    return not profile.patterns and not profile.tensions


def profile_dimensions(profile: TasteProfile | None) -> tuple[str, ...]:
    """Return the attributes/signals the profile actually names, ordered.

    These are the dimensions the Familiar↔Surprising dial moves along (R036):
    the profile's own vocabulary, not a fixed global axis. Empty profile -> ``()``.
    """
    if is_empty(profile):
        return ()
    assert profile is not None
    dims: list[str] = []
    for claim in profile.patterns:
        dim = _claim_dimension(claim.id)
        if dim and dim not in dims:
            dims.append(dim)
    return tuple(dims)


def citations_for(
    profile: TasteProfile | None,
    attribute: str | None = None,
    *,
    limit: int = DEFAULT_CITATION_LIMIT,
) -> list[ProfileCitation]:
    """Return the quotes backing *attribute* (or the whole profile).

    High-provenance (Reaction Room) claims come first, then the most-evidenced,
    then claim id — a total order, so citations are deterministic. Disputed
    claims are simply absent: :meth:`ProfileStore.dispute` removed them from the
    document, so a dispute silences the citation *and* the ranking together.
    """
    if is_empty(profile) or limit <= 0:
        return []
    assert profile is not None
    claims = [
        claim
        for claim in profile.patterns
        if attribute is None or _claim_dimension(claim.id) == attribute
    ]
    ordered = sorted(
        claims,
        key=lambda c: (0 if c.provenance == "high" else 1, -len(c.evidence), c.id),
    )
    citations: list[ProfileCitation] = []
    for claim in ordered[:limit]:
        if not claim.evidence:
            continue
        # The user's own words are the surface text (no jargon laundering).
        newest = max(claim.evidence, key=lambda ref: (ref.created_at, ref.image_sha))
        citations.append(
            ProfileCitation(
                claim_id=claim.id,
                quote=newest.verbatim,
                usage_count=len(claim.evidence),
                provenance=claim.provenance,
                confidence=newest.confidence,
            )
        )
    return citations


def attribute_signal_leanings(attribute: str) -> dict[str, float]:
    """Return *attribute*'s M002 signal leanings (``vibrancy`` mirrors colorfulness)."""
    leanings = dict(_ATTRIBUTE_SIGNALS.get(attribute, {}))
    if "colorfulness" in leanings:
        leanings["vibrancy"] = leanings["colorfulness"]
    return leanings


def claim_signal_leanings(claim: TasteClaim) -> dict[str, float]:
    """Return the signal leanings *claim* implies, keyed by M002 signal name.

    ``pattern:<attribute>`` claims go through :data:`_ATTRIBUTE_SIGNALS`;
    cold-start ``history:<source>:<signal>`` claims already name their signal and
    carry their direction in the claim text.
    """
    if claim.id.startswith(_HISTORY_PREFIX):
        signal = _claim_dimension(claim.id)
        if signal not in SIGNAL_NAMES:
            return {}
        return {signal: 1.0 if _HIGH_MARKER in claim.text else -1.0}
    return attribute_signal_leanings(_claim_dimension(claim.id))


def profile_fit(profile: TasteProfile | None, signals: dict[str, float]) -> float:
    """Return how well one work's *signals* fit the profile.

    ``sum(provenance_weight * leaning * value)`` over every claim and signal.
    An empty profile scores ``0.0`` for every work, so any consumer that adds
    this term leaves its baseline order untouched (no hard dependency, R038).
    """
    if is_empty(profile):
        return 0.0
    assert profile is not None
    total = 0.0
    for claim in profile.patterns:
        weight = 1.0 if claim.provenance == "high" else CORROBORATING_WEIGHT
        for signal, leaning in claim_signal_leanings(claim).items():
            total += weight * leaning * float(signals.get(signal, 0.0))
    return total


def familiar_surprising_dimensions(
    profile: TasteProfile | None, dial: float
) -> tuple[str, ...]:
    """Return the profile dimensions the Familiar↔Surprising *dial* moves along.

    Familiar (``dial <= 0``) walks the profile's dimensions strongest-first;
    Surprising (``dial > 0``) walks them weakest-first, so exploration moves
    along the user's own axes rather than a generic novelty score. An empty
    profile has no dimensions to move along and returns ``()``.
    """
    dims = profile_dimensions(profile)
    if not dims:
        return ()
    return dims if dial <= 0 else tuple(reversed(dims))


def explain_rank(
    analysis: AnalysisResult,
    lens_profile: LensProfile | None,
    profile: TasteProfile | None = None,
    *,
    limit: int = DEFAULT_CITATION_LIMIT,
) -> RankExplanation:
    """Explain one rerank, citing the taste profile when it has something to say.

    The M007 contributions are always the machine evidence and always drive
    ``delta`` — M008 adds words, not weights. With an empty profile the rationale
    is byte-identical to the uncited baseline string (R036 graceful degradation).

    Citing is independent of whether the M007 lens profile is enabled: an inert
    lens means nothing was reranked, but the taste profile still has something to
    say about the work, and R036 is about explanations quoting the profile.
    """
    ranker = TasteRanker()
    delta = 0.0
    contributions: list[dict[str, Any]] = []
    if lens_profile is not None and ranker.is_enabled(lens_profile):
        delta, contributions = ranker.personal_delta(analysis, lens_profile)
        rationale = _delta_rationale(delta, contributions)
    else:
        rationale = _BASELINE_RATIONALE
    citations = citations_for(profile, limit=limit)
    if citations:
        rationale = f"{rationale}; " + "; ".join(c.render() for c in citations)
    return RankExplanation(
        rationale=rationale,
        citations=citations,
        evidence=[dict(c) for c in contributions],
        delta=delta,
    )


def pairing_rationale(
    base_rationale: str,
    profile: TasteProfile | None = None,
    *,
    limit: int = 1,
) -> str:
    """Return *base_rationale*, optionally citing the profile (M002 pairing).

    Layout proposals *may* cite the profile; with an empty profile the string is
    returned unchanged so M002 rationale text is stable without M008.
    """
    citations = citations_for(profile, limit=limit)
    if not citations:
        return base_rationale
    return f"{base_rationale}; " + "; ".join(c.render() for c in citations)


# -- pure helpers ------------------------------------------------------------

#: The rationale emitted when no personalization applies — the exact M007
#: baseline wording, reused so an empty profile is indistinguishable.
_BASELINE_RATIONALE = "baseline order (no taste profile applied)"


def _claim_dimension(claim_id: str) -> str:
    """Return the attribute/signal a claim id names (``""`` when it names none)."""
    if claim_id.startswith(_PATTERN_PREFIX):
        return claim_id[len(_PATTERN_PREFIX):]
    if claim_id.startswith(_HISTORY_PREFIX):
        return claim_id.rsplit(":", 1)[-1]
    return ""


def _delta_rationale(delta: float, contributions: list[dict[str, Any]]) -> str:
    """Render the M007 half of the rationale: direction + the strongest signal."""
    direction = "ranked up" if delta > 0 else "ranked down" if delta < 0 else "unchanged"
    strongest = max(
        contributions,
        key=lambda c: (abs(float(c["contribution"])), str(c["signal"])),
        default=None,
    )
    if strongest is None or float(strongest["contribution"]) == 0.0:
        return f"{direction} ({delta:+.3f})"
    return (
        f"{direction} ({delta:+.3f}) — strongest signal "
        f"{str(strongest['signal']).replace('_', ' ')}"
    )
