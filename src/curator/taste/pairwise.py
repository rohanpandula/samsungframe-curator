"""Active-learning pairwise engine + evidence (M007/S02).

Deterministic, bootstrap-free machinery on top of S01 profiles: an uncertainty
score that measures how close/near-tied a candidate pair is under a profile, a
most-informative pair chooser that is deterministic given its inputs, a frozen
weight update (:func:`apply_preference`) that never mutates its input, and a
gated promotion flow (:func:`evaluate` + :func:`promote_if_valid`) that only
admits a trained profile once held-out evidence clears a threshold.
"""

from __future__ import annotations

import random
from collections.abc import Sequence, Set
from dataclasses import dataclass
from typing import Any

from curator.analysis.schema import AnalysisResult
from curator.taste.profiles import SIGNAL_NAMES, TasteProfile
from curator.taste.rank import TasteRanker

#: Small per-preference weight adjustment (":func:`apply_preference`).
LEARNING_STEP = 0.1
#: Sanity bounds weights are clamped into.
MIN_WEIGHT = -10.0
MAX_WEIGHT = 10.0
#: Promotion gate: a profile is promoted only once validated and these clear.
ACCURACY_THRESHOLD = 0.8
LIFT_THRESHOLD = 0.0


def _delta(profile: TasteProfile, signal: dict[str, float]) -> float:
    """Personal delta (``sum(weight * value)``) for a signal dict."""
    return sum(
        profile.weights.get(name, 0.0) * signal.get(name, 0.0)
        for name in SIGNAL_NAMES
    )


def uncertainty_score(
    a_signal: dict[str, float], b_signal: dict[str, float], profile: TasteProfile
) -> float:
    """Score how near-tied a pair is: 1.0 when identical, dropping toward 0.

    ``diff = |delta(a) - delta(b)|`` is normalized by the maximum possible delta
    spread ``sum(|w| * |a-b|)`` so a tied pair scores 1.0 and a far-apart pair
    scores low. Purely a function of its inputs — no RNG.
    """
    diff = abs(_delta(profile, a_signal) - _delta(profile, b_signal))
    spread = 0.0
    for name in SIGNAL_NAMES:
        spread += abs(profile.weights.get(name, 0.0)) * abs(
            a_signal.get(name, 0.0) - b_signal.get(name, 0.0)
        )
    if spread == 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - diff / spread))


def choose_pair(
    candidates: list[dict[str, Any]],
    profile: TasteProfile,
    *,
    analysis_map: dict[Any, AnalysisResult],
    rng_seed: int = 0,
    exclude: Set[tuple[str, str]] = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick the most informative (highest-uncertainty) candidate pair.

    Ties among equally-uncertain pairs are resolved with a ``rng_seed``-seeded
    shuffle over the stable (sorted) pair enumeration, so the result is fully
    deterministic given its inputs. ``exclude`` holds ``(a, b)`` id pairs to
    skip. Returns the two candidate dicts in original order (the as-paired order
    is unspecified; callers use :func:`apply_preference` with an explicit winner).
    """
    ranker = TasteRanker()
    signal = {
        cand["id"]: ranker.signal_values(analysis_map[cand["id"]]) for cand in candidates
    }
    ids = sorted(cand["id"] for cand in candidates)
    by_id = {cand["id"]: cand for cand in candidates}
    excluded = {frozenset((a, b)) for a, b in exclude}
    scored: list[tuple[float, str, str]] = []
    best = -1.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            aid, bid = ids[i], ids[j]
            if frozenset((aid, bid)) in excluded:
                continue
            u = uncertainty_score(signal[aid], signal[bid], profile)
            scored.append((u, aid, bid))
            if u > best:
                best = u
    tied = sorted((aid, bid) for u, aid, bid in scored if u == best)
    if not tied:
        raise ValueError("no candidate pairs to choose from")
    rng = random.Random(rng_seed)
    rng.shuffle(tied)
    aid, bid = tied[0]
    return by_id[aid], by_id[bid]


def apply_preference(
    profile: TasteProfile,
    a_signal: dict[str, float],
    b_signal: dict[str, float],
    prefer_a: bool = True,
) -> TasteProfile:
    """Return a NEW profile with one preference folded into the weights.

    The winner's signal values are nudged up and the loser's down:
    ``delta = step * (value_winner - value_loser)`` per signal, then the weight
    is clamped to :data:`MIN_WEIGHT`/:data:`MAX_WEIGHT`. The input profile is
    never mutated (frozen); the returned profile carries the same id/kind/name,
    ``version + 1``, and only its own kind is touched.
    """
    winner = a_signal if prefer_a else b_signal
    loser = b_signal if prefer_a else a_signal
    weights = dict(profile.weights)
    for name in SIGNAL_NAMES:
        delta = LEARNING_STEP * (winner.get(name, 0.0) - loser.get(name, 0.0))
        weights[name] = min(MAX_WEIGHT, max(MIN_WEIGHT, weights.get(name, 0.0) + delta))
    return TasteProfile(
        id=profile.id,
        kind=profile.kind,
        name=profile.name,
        weights=weights,
        version=profile.version + 1,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


@dataclass(frozen=True)
class PairwiseEvidence:
    """Aggregated, JSON-serializable evidence about a trained profile."""

    held_out_pairs: int
    held_out_accuracy: float
    ranking_lift_vs_baseline: float
    sample_efficiency_pairs: int
    requires_validation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "held_out_pairs": self.held_out_pairs,
            "held_out_accuracy": self.held_out_accuracy,
            "ranking_lift_vs_baseline": self.ranking_lift_vs_baseline,
            "sample_efficiency_pairs": self.sample_efficiency_pairs,
            "requires_validation": self.requires_validation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PairwiseEvidence:
        if isinstance(data, cls):
            return data
        return cls(
            held_out_pairs=int(data["held_out_pairs"]),
            held_out_accuracy=float(data["held_out_accuracy"]),
            ranking_lift_vs_baseline=float(data["ranking_lift_vs_baseline"]),
            sample_efficiency_pairs=int(data["sample_efficiency_pairs"]),
            requires_validation=bool(data.get("requires_validation", True)),
        )


def _score(
    ranker: TasteRanker,
    analysis: AnalysisResult,
    profile: TasteProfile | None,
    baseline: float,
) -> float:
    delta, _ = ranker.personal_delta(analysis, profile) if profile is not None else (0.0, [])
    return baseline + delta


def _spearman(order_a: Sequence[str], order_b: Sequence[str]) -> float:
    """Spearman rank correlation over the shared items of two id orderings."""
    rank_b = {item: i for i, item in enumerate(order_b)}
    common = [item for item in order_a if item in rank_b]
    if len(common) < 2:
        return 0.0
    ra = [i for i, item in enumerate(order_a) if item in rank_b]
    rb = [rank_b[item] for item in common]
    n = len(common)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(ra, rb))
    va = sum((x - mean_a) ** 2 for x in ra)
    vb = sum((y - mean_b) ** 2 for y in rb)
    if va == 0 or vb == 0:
        return 0.0
    return cov / ((va * vb) ** 0.5)


def evaluate(
    profile_trained: TasteProfile,
    profile_baseline: TasteProfile,
    candidates: list[dict[str, Any]],
    analysis_map: dict[Any, AnalysisResult],
    held_out_pairs: Sequence[tuple[str, str, str]],
) -> PairwiseEvidence:
    """Measure a trained profile against ground-truth preferences.

    ``held_out_pairs`` is a list of ``(id_a, id_b, preferred_id)`` triples. The
    held-out pairwise accuracy is the fraction of triples where the trained
    profile's ``baseline + delta`` ranking matches the stated preference.
    ``ranking_lift_vs_baseline`` is the Spearman-correlation gap between the
    trained and baseline ranking against a preference-derived ground-truth
    ordering. ``sample_efficiency_pairs`` records how many preference updates
    (one per profile version past 1) produced the trained profile.
    """
    ranker = TasteRanker()
    baseline_map = {cand["id"]: float(cand["baseline"]) for cand in candidates}
    totals = len(held_out_pairs)
    correct = 0
    preferred_count: dict[str, int] = {}
    for aid, bid, pref in held_out_pairs:
        preferred_count[aid] = preferred_count.get(aid, 0)
        preferred_count[bid] = preferred_count.get(bid, 0)
        preferred_count[pref] = preferred_count.get(pref, 0) + 1
        score_a = _score(ranker, analysis_map[aid], profile_trained, baseline_map[aid])
        score_b = _score(ranker, analysis_map[bid], profile_trained, baseline_map[bid])
        trained_pref = aid if score_a >= score_b else bid
        if trained_pref == pref:
            correct += 1
    accuracy = correct / totals if totals else 0.0

    ground_truth = sorted(
        preferred_count,
        key=lambda cid: (-preferred_count[cid], cid),
    )
    trained_order = [
        c["id"]
        for c in ranker.rank(candidates, profile_trained, analysis_map=analysis_map)
    ]
    baseline_order = [
        c["id"]
        for c in ranker.rank(candidates, profile_baseline, analysis_map=analysis_map)
    ]
    lift = _spearman(trained_order, ground_truth) - _spearman(baseline_order, ground_truth)
    return PairwiseEvidence(
        held_out_pairs=totals,
        held_out_accuracy=accuracy,
        ranking_lift_vs_baseline=lift,
        sample_efficiency_pairs=profile_trained.version - 1,
    )


def promote_if_valid(evidence: PairwiseEvidence) -> bool:
    """Promotion gate: evidence-before-promotion.

    A profile is promotable only when it *requires* validation and the
    held-out accuracy is at/above :data:`ACCURACY_THRESHOLD` with a non-negative
    ranking lift (trained ranking no worse than baseline).
    """
    return (
        evidence.requires_validation
        and evidence.held_out_accuracy >= ACCURACY_THRESHOLD
        and evidence.ranking_lift_vs_baseline >= LIFT_THRESHOLD
    )
