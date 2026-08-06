"""Tests for src/curator/taste/pairwise (M007/S02 active learning + evidence).

Synthetic candidates with M002-style signals and a "prefers more colorful"
ground-truth user; all deterministic with no RNG hidden behind the seed.
"""

from __future__ import annotations

import json
import random

from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.taste.pairwise import (
    PairwiseEvidence,
    apply_preference,
    choose_pair,
    evaluate,
    promote_if_valid,
    uncertainty_score,
)
from curator.taste.profiles import (
    SIGNAL_NAMES,
    TasteProfile,
    TasteProfileKind,
    baseline_weights,
    default_profile,
)
from curator.taste.rank import TasteRanker


def _result(colorfulness: float = 0.5, aesthetic: float = 0.5) -> AnalysisResult:
    return AnalysisResult(
        asset_id="asset",
        quality=QualitySignals(aesthetic_quality=aesthetic),
        color_story=ColorStory(colorfulness=colorfulness),
    )


def _full(colorfulness: float, aesthetic: float, harmony: float) -> AnalysisResult:
    return AnalysisResult(
        asset_id="asset",
        quality=QualitySignals(aesthetic_quality=aesthetic),
        color_story=ColorStory(colorfulness=colorfulness, harmony=harmony),
    )


def _profile(col_weight: float = 0.0, harmony_weight: float = 0.0) -> TasteProfile:
    return TasteProfile(
        id="p-personal",
        kind=TasteProfileKind.PERSONAL,
        name="personal",
        weights={**baseline_weights(), "colorfulness": col_weight, "harmony": harmony_weight},
    )


def _signals(cands, amap):
    ranker = TasteRanker()
    return {c["id"]: ranker.signal_values(amap[c["id"]]) for c in cands}


# ---------------------------------------------------------------------------
# uncertainty
# ---------------------------------------------------------------------------


def test_uncertainty_near_tied_beats_far_apart():
    prof = _profile(col_weight=1.0)
    near_a = {name: 0.5 for name in SIGNAL_NAMES}
    near_b = {name: 0.5 for name in SIGNAL_NAMES}
    far_a = {name: 0.1 for name in SIGNAL_NAMES}
    far_b = {name: 0.9 for name in SIGNAL_NAMES}
    assert uncertainty_score(near_a, near_b, prof) == 1.0
    assert uncertainty_score(near_a, near_b, prof) > uncertainty_score(far_a, far_b, prof)
    assert uncertainty_score(far_a, far_b, prof) == uncertainty_score(far_a, far_b, prof)


# ---------------------------------------------------------------------------
# choose_pair
# ---------------------------------------------------------------------------


def test_choose_pair_picks_highest_uncertainty():
    # Two candidates are identical (near-tie, uncertainty 1.0); the third is far.
    cands = [
        {"id": "x", "baseline": 1.0},
        {"id": "y", "baseline": 1.0},
        {"id": "z", "baseline": 1.0},
    ]
    amap = {
        "x": _result(colorfulness=0.5, aesthetic=0.5),
        "y": _result(colorfulness=0.5, aesthetic=0.5),
        "z": _result(colorfulness=1.0, aesthetic=0.0),
    }
    prof = _profile(col_weight=1.0)
    a, b = choose_pair(cands, prof, analysis_map=amap, rng_seed=0)
    assert {a["id"], b["id"]} == {"x", "y"}


def test_choose_pair_deterministic_same_inputs():
    cands, amap = _three_candidates()
    prof = _profile(col_weight=1.0, harmony_weight=1.0)
    first = choose_pair(cands, prof, analysis_map=amap, rng_seed=0)
    second = choose_pair(cands, prof, analysis_map=amap, rng_seed=0)
    assert first == second


def test_choose_pair_exclude_skips_pair():
    cands, amap = _three_candidates()
    prof = _profile(col_weight=1.0, harmony_weight=1.0)
    a, b = choose_pair(cands, prof, analysis_map=amap, rng_seed=0, exclude=frozenset())
    a_id, b_id = a["id"], b["id"]
    again = choose_pair(
        cands,
        prof,
        analysis_map=amap,
        rng_seed=0,
        exclude={(a_id, b_id), (b_id, a_id)},
    )
    assert {again[0]["id"], again[1]["id"]} != {a_id, b_id}


def _three_candidates():
    cands = [
        {"id": "lo", "baseline": 1.0},
        {"id": "mid", "baseline": 1.0},
        {"id": "hi", "baseline": 1.0},
    ]
    amap = {
        "lo": _result(colorfulness=0.1, aesthetic=0.9),
        "mid": _result(colorfulness=0.5, aesthetic=0.5),
        "hi": _result(colorfulness=0.9, aesthetic=0.1),
    }
    return cands, amap


# ---------------------------------------------------------------------------
# apply_preference
# ---------------------------------------------------------------------------


def test_apply_preference_prefers_more_colorful():
    prof = _profile(col_weight=0.0)
    colorful = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 1.0}
    muted = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 0.0}
    updated = apply_preference(prof, colorful, muted, prefer_a=True)
    assert updated.weights["colorfulness"] > prof.weights["colorfulness"]
    assert updated.weights["harmony"] == prof.weights["harmony"]


def test_apply_preference_deterministic_and_new_profile():
    prof = _profile(col_weight=0.4)
    colorful = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 1.0}
    muted = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 0.0}
    a = apply_preference(prof, colorful, muted, prefer_a=True)
    b = apply_preference(prof, colorful, muted, prefer_a=True)
    assert a.weights == b.weights
    assert a is not prof
    assert a.version == prof.version + 1
    assert a.id == prof.id
    assert a.kind is prof.kind
    assert prof.weights["colorfulness"] == 0.4
    assert a.weights["colorfulness"] != prof.weights["colorfulness"]


def test_apply_preference_reverses_when_losing():
    colorful = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 1.0}
    muted = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 0.0}
    up = apply_preference(_profile(0.0), colorful, muted, prefer_a=True)
    down = apply_preference(_profile(0.0), colorful, muted, prefer_a=False)
    assert up.weights["colorfulness"] > down.weights["colorfulness"]


def test_apply_preference_kind_isolation():
    colorful = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 1.0}
    muted = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 0.0}
    personal = TasteProfile(
        id="p-personal", kind=TasteProfileKind.PERSONAL, name="p", weights=baseline_weights()
    )
    household = TasteProfile(
        id="p-household", kind=TasteProfileKind.HOUSEHOLD, name="h", weights=baseline_weights()
    )
    apply_preference(personal, colorful, muted, prefer_a=True)
    assert household.weights == baseline_weights()
    updated = apply_preference(personal, colorful, muted, prefer_a=True)
    assert updated.kind is TasteProfileKind.PERSONAL
    assert household.weights == baseline_weights()


# ---------------------------------------------------------------------------
# evaluate + evidence
# ---------------------------------------------------------------------------


def _colorful_candidates(n: int = 6):
    cands = [{"id": f"c{i}", "baseline": 1.0} for i in range(n)]
    amap = {f"c{i}": _result(colorfulness=i / (n - 1)) for i in range(n)}
    return cands, amap


def _all_pairs(ids):
    return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]


def _evaluate_like(prof, cands, amap, holdouts):
    ranker = TasteRanker()
    return evaluate(
        lambda a: ranker.personal_delta(a, prof)[0],
        lambda a: 0.0,
        cands,
        amap,
        holdouts,
        sample_efficiency_pairs=prof.version - 1,
    )


def test_evaluate_trained_profile_beats_baseline():
    cands, amap = _colorful_candidates(6)
    holdouts = [("c0", "c5", "c5")]  # extremes: correct answer is c5
    prof = _profile(0.0)
    sv = _signals(cands, amap)
    used: set[frozenset] = set()
    holdout_set = {frozenset(h) for h in holdouts}
    pool = [p for p in _all_pairs([c["id"] for c in cands]) if frozenset(p) not in holdout_set]
    for _ in range(len(pool)):
        a, b = choose_pair(cands, prof, analysis_map=amap, rng_seed=0,
                           exclude=used | holdout_set)
        aid, bid = a["id"], b["id"]
        prefer_a = amap[aid].color_story.colorfulness > amap[bid].color_story.colorfulness
        prof = apply_preference(prof, sv[aid], sv[bid], prefer_a=prefer_a)
        used.add(frozenset((aid, bid)))
    ev = _evaluate_like(prof, cands, amap, holdouts)
    assert ev.held_out_pairs == 1
    assert ev.held_out_accuracy == 1.0
    assert ev.ranking_lift_vs_baseline > 0.0
    assert ev.sample_efficiency_pairs >= 1
    # Baseline (identity) profile scores the held-out pair only by baseline = tie.
    base_ev = _evaluate_like(default_profile(), cands, amap, holdouts)
    assert ev.held_out_accuracy > base_ev.held_out_accuracy


# ---------------------------------------------------------------------------
# CR-01: a tie is an abstention, never a win — position-independent
# ---------------------------------------------------------------------------


def test_evaluate_tie_is_abstention_not_a_win_regardless_of_pair_position():
    """A zero-information scorer must score 0.0 accuracy no matter which id the
    caller happens to place in the ``aid`` slot.

    Reproduces CR-01 directly: a scorer with no opinion (always ties against an
    identical baseline) used to be silently scored "correct" whenever the
    *first* id of the triple happened to be the preferred one — which is
    exactly the convention ``api.py``/``cli.py`` always use in production
    (``(winner, loser, winner)``). Both pair-position conventions are checked
    here so the historically-tested "preferred == bid" fixture convention can
    never again mask a regression of this exact bug.
    """
    cands = [{"id": "c0", "baseline": 1.0}, {"id": "c1", "baseline": 1.0}]
    amap = {
        "c0": _result(colorfulness=0.5),
        "c1": _result(colorfulness=0.5),
    }

    def zero_scorer(_: AnalysisResult) -> float:
        return 0.0

    # PRODUCTION-STYLE convention: preferred id always in the `aid` (first) slot.
    production_style = evaluate(
        zero_scorer, zero_scorer, cands, amap, [("c0", "c1", "c0")],
        sample_efficiency_pairs=0,
    )
    # UNBIASED convention: preferred id always in the `bid` (second) slot —
    # the convention every pre-existing fixture in this repo happened to use.
    unbiased = evaluate(
        zero_scorer, zero_scorer, cands, amap, [("c1", "c0", "c0")],
        sample_efficiency_pairs=0,
    )
    assert production_style.held_out_accuracy == 0.0
    assert unbiased.held_out_accuracy == 0.0
    # Position-independent: identical result either way, never a coin-flip winner.
    assert production_style.held_out_accuracy == unbiased.held_out_accuracy


# ---------------------------------------------------------------------------
# PairwiseEvidence round-trip
# ---------------------------------------------------------------------------


def test_pairwise_evidence_round_trip():
    ev = PairwiseEvidence(
        held_out_pairs=12,
        held_out_accuracy=0.92,
        ranking_lift_vs_baseline=0.4,
        sample_efficiency_pairs=9,
    )
    rebuilt = PairwiseEvidence.from_dict(json.loads(json.dumps(ev.to_dict())))
    assert rebuilt == ev
    assert rebuilt.requires_validation is True


def test_pairwise_evidence_from_dict_defaults_validation():
    ev = PairwiseEvidence.from_dict(
        {
            "held_out_pairs": 3,
            "held_out_accuracy": 0.5,
            "ranking_lift_vs_baseline": 0.1,
            "sample_efficiency_pairs": 2,
        }
    )
    assert ev.requires_validation is True


# ---------------------------------------------------------------------------
# promotion gate
# ---------------------------------------------------------------------------


def test_promote_if_valid_only_when_validated_and_meets_threshold():
    good = PairwiseEvidence(10, 0.95, 0.3, 8)
    assert promote_if_valid(good) is True
    unvalidated = PairwiseEvidence(10, 0.95, 0.3, 8, requires_validation=False)
    assert promote_if_valid(unvalidated) is False
    low_acc = PairwiseEvidence(10, 0.4, 0.3, 8)
    assert promote_if_valid(low_acc) is False
    negative_lift = PairwiseEvidence(10, 0.95, -0.2, 8)
    assert promote_if_valid(negative_lift) is False


# ---------------------------------------------------------------------------
# sample efficiency
# ---------------------------------------------------------------------------


def _pair_candidates(n: int = 12):
    """Candidates varying on colorfulness (the ground-truth signal) plus a
    random distractor (aesthetic) so some delta pairs are near-tied/ambiguous."""
    rng = random.Random(42)
    cands = [{"id": f"c{i}", "baseline": 1.0} for i in range(n)]
    amap = {
        f"c{i}": _result(colorfulness=i / (n - 1), aesthetic=rng.random())
        for i in range(n)
    }
    ids = [c["id"] for c in cands]
    # Hold out the near-adjacent pairs (genuinely hard, close colorfulness).
    holdouts = []
    for i in range(len(ids) - 1):
        if abs(
            amap[ids[i]].color_story.colorfulness - amap[ids[i + 1]].color_story.colorfulness
        ) < 0.2:
            a, b = ids[i], ids[i + 1]
            pref = a if amap[a].color_story.colorfulness > amap[b].color_story.colorfulness else b
            holdouts.append((a, b, pref))
    return cands, amap, holdouts


def _train_until(cands, amap, holdouts, target, strategy, seed, max_iters=120):
    prof = _profile(0.0)
    sv = _signals(cands, amap)
    used: set[frozenset] = set()
    holdout_set = {frozenset(h) for h in holdouts}
    ids = [c["id"] for c in cands]
    pool = [p for p in _all_pairs(ids) if frozenset(p) not in holdout_set]
    for step in range(max_iters):
        if strategy == "informative":
            a, b = choose_pair(cands, prof, analysis_map=amap, rng_seed=seed,
                               exclude=used | holdout_set)
            aid, bid = a["id"], b["id"]
        else:
            remain = [p for p in pool if frozenset(p) not in used]
            if not remain:
                break
            aid, bid = random.Random(seed * 7 + step).choice(remain)
        prefer_a = amap[aid].color_story.colorfulness > amap[bid].color_story.colorfulness
        prof = apply_preference(prof, sv[aid], sv[bid], prefer_a=prefer_a)
        used.add(frozenset((aid, bid)))
        ev = _evaluate_like(prof, cands, amap, holdouts)
        if ev.held_out_pairs and ev.held_out_accuracy >= target:
            return step + 1
    return max_iters


def test_sample_efficiency_informative_beats_random():
    cands, amap, holdouts = _pair_candidates()
    target = 0.75
    informative = _train_until(cands, amap, holdouts, target, "informative", seed=4)
    random_sel = _train_until(cands, amap, holdouts, target, "random", seed=4)
    assert informative >= 1
    assert random_sel >= informative
    # Informative-pair selection reaches the target with fewer samples than random.
    assert informative < random_sel
