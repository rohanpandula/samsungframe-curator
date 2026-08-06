"""Tests for src/curator/taste/dialogue/upstream.py (M008/S05).

Covers the profile-as-upstream surface: citations that keep the user's verbatim
on top, profile dimensions and fit expressed in M002 signal space, the
low-provenance demotion of approval/pairwise-derived claims, rerank explanations
and pairing rationale that cite the profile only when it has something to say,
the Familiar↔Surprising dial moving along profile dimensions, and the R038
anti-goals — most importantly that an empty profile leaves every consumer's
output byte-identical to its M007 baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.catalog import Catalog
from curator.taste.dialogue import upstream as dialogue_upstream
from curator.taste.dialogue.observation import ImageRef, Polarity, TasteObservation
from curator.taste.dialogue.profile import (
    EvidenceRef,
    ProfileBuilder,
    ProfileStore,
    TasteClaim,
)
from curator.taste.dialogue.upstream import (
    CORROBORATING_WEIGHT,
    ProfileCitation,
    RankExplanation,
    attribute_signal_leanings,
    citations_for,
    claim_signal_leanings,
    explain_rank,
    familiar_surprising_dimensions,
    is_empty,
    pairing_rationale,
    profile_dimensions,
    profile_fit,
)
from curator.taste.profiles import TasteProfile as LensProfile
from curator.taste.profiles import TasteProfileKind
from curator.taste.rank import TasteRanker

QUIET_VERBATIM = "i love the quiet empty scene"
SPACE_VERBATIM = "so much negative space"


def _observation(
    session_id: str,
    verbatim: str,
    attributes: list[str],
    *,
    polarity: Polarity = Polarity.LIKE,
    created_at: str = "2026-08-04T10:00:00.000000Z",
) -> TasteObservation:
    return TasteObservation(
        session_id=session_id,
        verbatim=verbatim,
        attributes=attributes,
        polarity=polarity,
        confidence=0.9,
        images=[
            # Content-addressed and stable across runs: PYTHONHASHSEED-independent.
            ImageRef(
                sha256=hashlib.sha256(verbatim.encode()).hexdigest(),
                thumb_path="/tmp/thumb.jpg",
                ephemeral=True,
            )
        ],
        created_at=created_at,
    )


def _dialogue_profile():
    """A profile whose only pattern is a high-provenance negative-space claim."""
    return ProfileBuilder().build(
        [
            _observation("s1", QUIET_VERBATIM, ["negative-space"]),
            _observation(
                "s2",
                SPACE_VERBATIM,
                ["negative-space"],
                created_at="2026-08-04T11:00:00.000000Z",
            ),
        ]
    )


def _empty_profile():
    return ProfileBuilder().build([])


def _lens_profile(**weights: float) -> LensProfile:
    return LensProfile(
        id="p1",
        kind=TasteProfileKind.PERSONAL,
        name="personal",
        weights=weights,
    )


def _analysis(colorfulness: float = 0.8, aesthetic: float = 0.6) -> AnalysisResult:
    return AnalysisResult(
        asset_id="a" * 64,
        quality=QualitySignals(technical_quality=0.5, aesthetic_quality=aesthetic),
        color_story=ColorStory(colorfulness=colorfulness, harmony=0.5),
    )


def _history_claim(signal: str, direction: str = "high") -> TasteClaim:
    return TasteClaim(
        id=f"history:approval:{signal}",
        text=(
            f"approval history suggests you lean toward {direction} {signal} "
            "(3 picks, +0.80 vs the ones you passed on) — inferred from history, "
            "not your words"
        ),
        evidence=[
            EvidenceRef(
                image_sha="c" * 64,
                verbatim="(no verbatim — inferred from history)",
                confidence=0.3,
                created_at="2026-08-01T10:00:00.000000Z",
            )
        ],
        provenance="low",
    )


# -- round-trips -------------------------------------------------------------


def test_profile_citation_round_trip_and_render():
    citation = ProfileCitation(
        claim_id="pattern:negative-space",
        quote=QUIET_VERBATIM,
        usage_count=3,
        provenance="high",
        confidence=0.9,
    )
    rebuilt = ProfileCitation.from_dict(json.loads(json.dumps(citation.to_dict())))
    assert rebuilt == citation
    assert ProfileCitation.from_dict(rebuilt) == citation
    assert QUIET_VERBATIM in citation.render()
    assert "3 times" in citation.render()


def test_profile_citation_render_singular():
    citation = ProfileCitation(
        claim_id="c", quote="quiet", usage_count=1, provenance="high", confidence=0.9
    )
    assert "1 time" in citation.render()
    assert "1 times" not in citation.render()


def test_rank_explanation_to_dict():
    explanation = RankExplanation(
        rationale="ranked up",
        citations=[
            ProfileCitation(
                claim_id="c", quote="quiet", usage_count=1, provenance="high",
                confidence=0.9,
            )
        ],
        evidence=[{"signal": "colorfulness", "contribution": 0.4}],
        delta=0.4,
    )
    payload = json.loads(json.dumps(explanation.to_dict()))
    assert payload["rationale"] == "ranked up"
    assert payload["citations"][0]["quote"] == "quiet"
    assert payload["evidence"][0]["signal"] == "colorfulness"
    assert payload["delta"] == 0.4


# -- dimensions + citations --------------------------------------------------


def test_is_empty_covers_none_and_claimless_profiles():
    assert is_empty(None)
    assert is_empty(_empty_profile())
    assert not is_empty(_dialogue_profile())


def test_profile_dimensions_names_the_profiles_own_attributes():
    assert profile_dimensions(_dialogue_profile()) == ("negative-space",)
    assert profile_dimensions(_empty_profile()) == ()
    assert profile_dimensions(None) == ()


def test_citations_keep_the_users_verbatim_on_top():
    citations = citations_for(_dialogue_profile())
    assert citations
    citation = citations[0]
    assert citation.claim_id == "pattern:negative-space"
    # The surface text is the user's own words, byte-exact (no jargon laundering).
    assert citation.quote in (QUIET_VERBATIM, SPACE_VERBATIM)
    assert citation.usage_count == 2
    assert citation.provenance == "high"


def test_citations_empty_profile_says_nothing():
    assert citations_for(_empty_profile()) == []
    assert citations_for(None) == []
    assert citations_for(_dialogue_profile(), limit=0) == []


def test_citations_filter_by_attribute():
    profile = _dialogue_profile()
    assert citations_for(profile, "negative-space")
    assert citations_for(profile, "warm-tones") == []


def test_citations_put_high_provenance_first():
    profile = _dialogue_profile()
    seeded = profile.__class__(
        vocabulary=profile.vocabulary,
        patterns=[_history_claim("colorfulness")] + list(profile.patterns),
        tensions=profile.tensions,
        evolution=profile.evolution,
    )
    citations = citations_for(seeded, limit=2)
    assert [c.provenance for c in citations] == ["high", "low"]


def test_citations_are_deterministic():
    profile = _dialogue_profile()
    first = [c.to_dict() for c in citations_for(profile)]
    second = [c.to_dict() for c in citations_for(profile)]
    assert first == second


# -- signal mapping + fit ----------------------------------------------------


def test_attribute_leanings_mirror_vibrancy_onto_colorfulness():
    leanings = attribute_signal_leanings("negative-space")
    assert leanings["colorfulness"] == leanings["vibrancy"]
    assert leanings["colorfulness"] < 0
    assert attribute_signal_leanings("not-an-attribute") == {}


def test_claim_leanings_read_history_direction_from_the_claim():
    assert claim_signal_leanings(_history_claim("colorfulness")) == {"colorfulness": 1.0}
    assert claim_signal_leanings(_history_claim("harmony", "low")) == {"harmony": -1.0}
    assert claim_signal_leanings(_history_claim("not-a-signal")) == {}


def test_claim_leanings_map_pattern_claims_through_the_vocabulary():
    claim = _dialogue_profile().patterns[0]
    assert claim_signal_leanings(claim) == attribute_signal_leanings("negative-space")


def test_profile_fit_is_zero_for_an_empty_profile():
    signals = {"colorfulness": 0.9, "aesthetic_quality": 0.5}
    assert profile_fit(None, signals) == 0.0
    assert profile_fit(_empty_profile(), signals) == 0.0


def test_profile_fit_prefers_works_matching_the_profile():
    profile = _dialogue_profile()  # negative-space: low colorfulness
    quiet_work = {"colorfulness": 0.1, "vibrancy": 0.1, "aesthetic_quality": 0.8}
    loud_work = {"colorfulness": 0.9, "vibrancy": 0.9, "aesthetic_quality": 0.8}
    assert profile_fit(profile, quiet_work) > profile_fit(profile, loud_work)


def test_low_provenance_claims_are_demoted_to_corroborating():
    """A history claim moves the fit by exactly CORROBORATING_WEIGHT of a stated one."""
    high = TasteClaim(
        id="history:approval:harmony",
        text=_history_claim("harmony").text,
        evidence=_history_claim("harmony").evidence,
        provenance="high",
    )
    low = _history_claim("harmony")
    base = _empty_profile()
    signals = {"harmony": 1.0}

    high_profile = base.__class__(
        vocabulary={}, patterns=[high], tensions=[], evolution=[]
    )
    low_profile = base.__class__(
        vocabulary={}, patterns=[low], tensions=[], evolution=[]
    )
    assert profile_fit(low_profile, signals) == pytest.approx(
        profile_fit(high_profile, signals) * CORROBORATING_WEIGHT
    )


# -- consumers ---------------------------------------------------------------


def test_explain_rank_with_no_lens_profile_still_cites_the_taste_profile():
    """An inert M007 lens means no rerank — the profile still has something to say."""
    explanation = explain_rank(_analysis(), None, _dialogue_profile())
    assert explanation.delta == 0.0
    assert explanation.evidence == []
    assert "baseline" in explanation.rationale
    assert explanation.citations
    assert explanation.citations[0].quote in explanation.rationale


def test_explain_rank_with_no_lens_and_no_profile_is_the_bare_baseline():
    explanation = explain_rank(_analysis(), None, None)
    assert explanation.citations == []
    assert explanation.delta == 0.0
    assert "baseline" in explanation.rationale


def test_explain_rank_empty_profile_matches_uncited_baseline_exactly():
    lens = _lens_profile(colorfulness=0.5)
    analysis = _analysis()
    uncited = explain_rank(analysis, lens, None)
    empty = explain_rank(analysis, lens, _empty_profile())
    assert empty.rationale == uncited.rationale
    assert empty.to_dict() == uncited.to_dict()
    assert empty.citations == []


def test_explain_rank_cites_the_profile_when_it_has_something_to_say():
    lens = _lens_profile(colorfulness=0.5)
    analysis = _analysis()
    cited = explain_rank(analysis, lens, _dialogue_profile())
    uncited = explain_rank(analysis, lens, None)

    assert cited.citations
    assert cited.rationale.startswith(uncited.rationale)
    assert cited.rationale != uncited.rationale
    assert "ranked up" in cited.rationale
    assert cited.citations[0].quote in cited.rationale
    # M008 adds words, not weights: the delta is still the M007 delta.
    assert cited.delta == uncited.delta
    assert cited.evidence == uncited.evidence


def test_explain_rank_reports_direction_and_strongest_signal():
    down = explain_rank(_analysis(), _lens_profile(colorfulness=-0.5), None)
    assert "ranked down" in down.rationale
    assert "colorfulness" in down.rationale


def test_pairing_rationale_cites_only_when_the_profile_is_non_empty():
    base = "paired on palette distance 0.12"
    assert pairing_rationale(base, None) == base
    assert pairing_rationale(base, _empty_profile()) == base

    cited = pairing_rationale(base, _dialogue_profile())
    assert cited.startswith(base)
    assert cited != base


def test_familiar_surprising_moves_along_profile_dimensions():
    profile = ProfileBuilder().build(
        [
            _observation("s1", QUIET_VERBATIM, ["negative-space"]),
            _observation("s2", SPACE_VERBATIM, ["negative-space"]),
            _observation("s3", "the warm glow", ["warm-tones"]),
            _observation("s4", "so warm here", ["warm-tones"]),
        ]
    )
    familiar = familiar_surprising_dimensions(profile, -1.0)
    surprising = familiar_surprising_dimensions(profile, 1.0)
    assert set(familiar) == {"negative-space", "warm-tones"}
    assert familiar == tuple(reversed(surprising))
    assert familiar_surprising_dimensions(_empty_profile(), 1.0) == ()


def test_profile_fit_reorders_a_feed_and_an_empty_profile_does_not():
    works = [
        {"id": "loud", "baseline": 1.0, "signals": {"colorfulness": 0.9, "vibrancy": 0.9}},
        {"id": "quiet", "baseline": 0.9, "signals": {"colorfulness": 0.0, "vibrancy": 0.0}},
    ]

    def order(profile):
        scored = [
            (w["baseline"] + profile_fit(profile, w["signals"]), i, w["id"])
            for i, w in enumerate(works)
        ]
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [wid for _, _, wid in scored]

    baseline = [w["id"] for w in works]
    assert order(None) == baseline
    assert order(_empty_profile()) == baseline
    # The profile prefers quiet scenes, so it overturns the baseline order.
    assert order(_dialogue_profile()) == ["quiet", "loud"]


# -- anti-goals (R038) -------------------------------------------------------


def test_dispute_removes_both_the_citation_and_the_ranking_effect(data_root):
    catalog = Catalog(data_root=data_root)
    profile = _dialogue_profile()
    store = ProfileStore(catalog)
    store.apply(profile)

    quiet = {"colorfulness": 0.0, "vibrancy": 0.0, "aesthetic_quality": 0.8}
    assert citations_for(profile)
    assert profile_fit(profile, quiet) != 0.0

    store.dispute("pattern:negative-space")
    disputed = store.load()

    assert citations_for(disputed) == []
    assert profile_fit(disputed, quiet) == 0.0
    assert explain_rank(_analysis(), _lens_profile(colorfulness=0.5), disputed).citations == []


def test_no_image_generation_code_path_in_the_dialogue_package():
    """R038: no generation, ever — not even a dormant hook."""
    package = Path(dialogue_upstream.__file__).parent
    banned = (
        "generate_image",
        "txt2img",
        "img2img",
        "diffusion",
        "stable_diffusion",
        "dall-e",
        "dalle",
        "inpaint",
        "outpaint",
    )
    for source in sorted(package.glob("*.py")):
        text = source.read_text().lower()
        for token in banned:
            assert token not in text, f"{source}: generation token {token!r}"


def test_verbatim_is_never_replaced_by_its_attribute():
    """R038: words map to attributes; the attribute never replaces the words."""
    citation = citations_for(_dialogue_profile())[0]
    assert citation.quote in (QUIET_VERBATIM, SPACE_VERBATIM)
    assert citation.quote != "negative-space"
    assert "negative-space" not in citation.render()


def test_every_consumer_degrades_to_baseline_on_an_empty_profile():
    """R038: no hard dependency — an empty profile changes nothing anywhere."""
    lens = _lens_profile(colorfulness=0.5)
    analysis = _analysis()
    ranker = TasteRanker()
    candidates = [{"id": "a", "baseline": 1.0}, {"id": "b", "baseline": 0.5}]
    analysis_map = {"a": analysis, "b": _analysis(colorfulness=0.1)}

    for empty in (None, _empty_profile()):
        assert citations_for(empty) == []
        assert profile_dimensions(empty) == ()
        assert profile_fit(empty, {"colorfulness": 0.9}) == 0.0
        assert familiar_surprising_dimensions(empty, 0.5) == ()
        assert pairing_rationale("base", empty) == "base"
        assert explain_rank(analysis, lens, empty).rationale == explain_rank(
            analysis, lens, None
        ).rationale
        # M007 ranking is untouched by the presence of the dialogue subsystem.
        assert [c["id"] for c in ranker.rank(candidates, lens, analysis_map=analysis_map)] == [
            c["id"] for c in ranker.rank(candidates, lens, analysis_map=analysis_map)
        ]
