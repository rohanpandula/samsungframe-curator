"""Tests for the uncertainty-aware head-to-head comparison (M009/S05, R043).

Every fixture in this file builds ``Scorer``/``embedding_scorer_factory``
callables directly — no ONNX inference, no ``EmbeddingStore``, no catalog —
mirroring ``tests/test_taste_embedding_head.py``'s/
``tests/test_taste_embedding_attribution.py``'s "synthetic data, real
machinery" pattern. The well-powered fixture below still exercises the real
:func:`~curator.taste.embedding.head.fit_embedding_head` (via a
``votes_subset -> Scorer`` factory shaped exactly like ``cli.py``'s/
``api.py``'s production one), so the learning-curve assertions test the real
nonparametric-head behavior, not a stub.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import curator.taste.embedding.compare as compare_module
from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.taste.embedding.compare import (
    MIN_DISCORDANT_PAIRS,
    HeadComparison,
    compare_heads,
    wilson_interval,
)
from curator.taste.embedding.head import VoteVectors, fit_embedding_head
from curator.taste.embedding.provider import EMBEDDING_DIM
from curator.taste.pairwise import Scorer


def _result(cid: str, colorfulness: float) -> AnalysisResult:
    """A synthetic candidate whose ``asset_id`` is its own candidate id.

    A test-only convenience — the real pipeline encodes a catalog entry id
    there instead (``f"entry-{id}"``); this file never touches the catalog,
    so it is free to control the id mapping directly.
    """
    return AnalysisResult(
        asset_id=cid,
        quality=QualitySignals(aesthetic_quality=0.5),
        color_story=ColorStory(colorfulness=colorfulness),
    )


def _zero_scorer(analysis: AnalysisResult) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# wilson_interval: pure math, independently testable from the head pipeline
# ---------------------------------------------------------------------------


def test_wilson_interval_zero_n_is_maximally_uninformative() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_large_n_near_certain_is_narrow_and_above_half() -> None:
    lo, hi = wilson_interval(190, 200)  # 95% observed, comfortably well-powered
    assert lo > 0.5
    assert hi - lo < 0.15  # narrow, unlike the small-n case below


def test_wilson_interval_small_n_straddles_half() -> None:
    """10 trials, 6 successes (60%) is small enough that the interval reaches
    both sides of 0.5 — the "tie" verdict is reachable in practice, not just a
    theoretical branch."""
    lo, hi = wilson_interval(6, 10)
    assert lo < 0.5 < hi


# ---------------------------------------------------------------------------
# insufficient_evidence: too few discordant pairs, never a coin-flip winner
# ---------------------------------------------------------------------------


def test_compare_heads_reports_insufficient_evidence_below_min_discordant_pairs() -> None:
    """Only 3 held-out pairs exist at all, so ``discordant_pairs`` can never
    reach :data:`MIN_DISCORDANT_PAIRS` (10) no matter how the two scorers
    disagree — the sample-size gate fires before any accuracy number is even
    computed."""
    n = 6
    candidates = [{"id": f"e{i}", "baseline": 0.0} for i in range(n)]
    analysis_map = {f"e{i}": _result(f"e{i}", colorfulness=i / (n - 1)) for i in range(n)}

    def lens_scorer(analysis: AnalysisResult) -> float:
        return analysis.color_story.colorfulness

    def embedding_scorer_factory(votes_subset: Sequence[VoteVectors]) -> Scorer:
        def scorer(analysis: AnalysisResult) -> float:
            return -analysis.color_story.colorfulness  # always disagrees with lens_scorer

        return scorer

    held_out_pairs = [("e0", "e1", "e1"), ("e2", "e3", "e3"), ("e4", "e5", "e5")]

    comparison = compare_heads(
        training_votes=[],
        held_out_pairs=held_out_pairs,
        candidates=candidates,
        analysis_map=analysis_map,
        lens_scorer=lens_scorer,
        embedding_scorer_factory=embedding_scorer_factory,
        baseline_scorer=_zero_scorer,
    )

    assert comparison.discordant_pairs == 3
    assert comparison.discordant_pairs < MIN_DISCORDANT_PAIRS
    assert comparison.verdict == "insufficient_evidence"
    assert comparison.head_to_head_accuracy is None
    assert comparison.head_to_head_ci is None


# ---------------------------------------------------------------------------
# CR-01: a zero-capacity embedding head must never be rewarded by ties
# ---------------------------------------------------------------------------


def test_compare_heads_zero_capacity_embedding_head_is_never_rewarded_by_ties() -> None:
    """Reproduces CR-01 against the real ``compare_heads`` end-to-end: a
    zero-capacity embedding head (``training_votes=[]`` — the real state of a
    fresh install before any ``--backfill``/embedding has run) ties against an
    identical baseline on every held-out pair. It must score 0.0 accuracy, never
    be promoted, and never register as "better" than a lens head that has a
    genuine opinion — using the exact production ``held_out_pairs`` shape
    ``(winner, loser, winner)`` that ``api.py``/``cli.py`` actually construct
    (the convention every other fixture in this file deliberately avoids, which
    is exactly why the original bug was invisible to this test suite).
    """
    n = 6
    candidates = [{"id": f"z{i}", "baseline": 0.0} for i in range(n)]
    analysis_map = {f"z{i}": _result(f"z{i}", colorfulness=i / (n - 1)) for i in range(n)}

    def lens_scorer(analysis: AnalysisResult) -> float:
        return analysis.color_story.colorfulness  # a genuine, deterministic opinion

    def zero_capacity_embedding_factory(votes_subset: Sequence[VoteVectors]) -> Scorer:
        head = fit_embedding_head(list(votes_subset), "v1")  # votes_subset is always []

        def scorer(analysis: AnalysisResult) -> float:
            return head.score(np.zeros(EMBEDDING_DIM, dtype=np.float32))

        return scorer

    # Production convention: the preferred (winning) id is always first — the
    # exact shape api.py/cli.py build from real VoteRecords.
    held_out_pairs = [(f"z{2 * k}", f"z{2 * k + 1}", f"z{2 * k}") for k in range(n // 2)]

    comparison = compare_heads(
        training_votes=[],
        held_out_pairs=held_out_pairs,
        candidates=candidates,
        analysis_map=analysis_map,
        lens_scorer=lens_scorer,
        embedding_scorer_factory=zero_capacity_embedding_factory,
        baseline_scorer=_zero_scorer,
    )

    assert comparison.embedding_evidence.sample_efficiency_pairs == 0
    # Must never be the spurious 1.0 the original bug reported for this exact
    # scenario — a head that has seen zero votes has no opinion, ever.
    assert comparison.embedding_evidence.held_out_accuracy == 0.0
    assert comparison.embedding_promoted is False
    # The zero-capacity head never disagrees with the lens head — it never has
    # an opinion to disagree with.
    assert comparison.discordant_pairs == 0
    assert comparison.verdict == "insufficient_evidence"


# ---------------------------------------------------------------------------
# a decisive verdict is reachable — not just always "insufficient_evidence"
# ---------------------------------------------------------------------------


def _axis_vector() -> np.ndarray:
    axis = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    axis[0] = 1.0
    return axis


def _embedding_scorer_factory_from_vectors(
    vector_by_id: dict[str, np.ndarray],
) -> Callable[[Sequence[VoteVectors]], Scorer]:
    """The same shape as ``cli.py``'s/``api.py``'s production factory: refit
    :func:`~curator.taste.embedding.head.fit_embedding_head` fresh per call,
    score by a pre-resolved ``asset_id -> vector`` map (a missing entry scores
    ``0.0`` rather than raising, mirroring the zero-evidence posture)."""

    def factory(votes_subset: Sequence[VoteVectors]) -> Scorer:
        head = fit_embedding_head(votes_subset, "v1")

        def scorer(analysis: AnalysisResult) -> float:
            vector = vector_by_id.get(analysis.asset_id)
            return head.score(vector) if vector is not None else 0.0

        return scorer

    return factory


def _well_powered_fixture() -> tuple[
    list[dict[str, Any]],
    dict[str, AnalysisResult],
    list[VoteVectors],
    list[tuple[str, str, str]],
    dict[str, np.ndarray],
]:
    """44 candidates whose *true* preference increases along a hidden axis the
    lens scorer never sees (every candidate's colorfulness is the same
    constant, so the lens scorer always ties and picks the first-named
    candidate); the embedding head is trained on votes along that same axis.

    Training votes are deliberately ordered 10 "noisy" (reversed — teach the
    WRONG direction) followed by 33 "correct" (teach the RIGHT direction): the
    first learning-curve checkpoint (25% of 43 votes = 10) sees only the noisy
    votes and should score at or near the *opposite* of the true order, while
    the final checkpoint (all 43) is dominated by the correct votes and should
    score every held-out pair correctly — a genuinely improving curve, not
    just a flat one.
    """
    n = 44
    candidates = [{"id": f"d{i}", "baseline": 0.0} for i in range(n)]
    analysis_map = {f"d{i}": _result(f"d{i}", colorfulness=0.5) for i in range(n)}
    axis = _axis_vector()
    vector_by_id = {f"d{i}": (i * axis).astype(np.float32) for i in range(n)}

    zero_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    training_votes: list[VoteVectors] = []
    for i in range(10):
        training_votes.append(
            VoteVectors(
                vote_group=f"noisy-{i}",
                winner_entry_id=1000 + i,
                loser_entry_id=2000 + i,
                winner_vector=zero_vec.copy(),
                loser_vector=axis.copy(),
            )
        )
    for i in range(33):
        training_votes.append(
            VoteVectors(
                vote_group=f"correct-{i}",
                winner_entry_id=3000 + i,
                loser_entry_id=4000 + i,
                winner_vector=axis.copy(),
                loser_vector=zero_vec.copy(),
            )
        )

    # 22 disjoint held-out pairs (d0,d1), (d2,d3), ... — the higher-index
    # (later) candidate of each pair is always the true preferred one.
    held_out_pairs = [(f"d{2 * k}", f"d{2 * k + 1}", f"d{2 * k + 1}") for k in range(22)]

    return candidates, analysis_map, training_votes, held_out_pairs, vector_by_id


def _lower_index_lens_scorer(analysis: AnalysisResult) -> float:
    """A lens with a genuine (never-tied), deterministically WRONG opinion.

    Reads the numeric suffix straight off ``asset_id`` (e.g. ``"d17"`` -> 17)
    and prefers lower indices — the opposite of every fixture built by
    :func:`_well_powered_fixture`/``_well_powered_comparison_fixture``, where
    the higher-indexed candidate of each held-out pair is always the true
    preferred one. Post-CR-01, a *tied* (zero-information) lens scorer
    correctly abstains rather than "winning" discordant pairs via the old
    aid tie-break, so a decisive-verdict fixture needs a lens with a real
    (if wrong) opinion — this is that opinion, never a coin-flip.
    """
    return -float(analysis.asset_id[1:])


def test_compare_heads_reaches_decisive_verdict_when_embedding_correlates() -> None:
    candidates, analysis_map, training_votes, held_out_pairs, vector_by_id = (
        _well_powered_fixture()
    )

    comparison = compare_heads(
        training_votes=training_votes,
        held_out_pairs=held_out_pairs,
        candidates=candidates,
        analysis_map=analysis_map,
        lens_scorer=_lower_index_lens_scorer,
        embedding_scorer_factory=_embedding_scorer_factory_from_vectors(vector_by_id),
        baseline_scorer=_zero_scorer,
    )

    assert comparison.discordant_pairs >= MIN_DISCORDANT_PAIRS
    assert comparison.verdict == "embedding_better"
    assert comparison.head_to_head_accuracy is not None
    assert comparison.head_to_head_accuracy > 0.5
    assert comparison.head_to_head_ci is not None
    assert comparison.head_to_head_ci[0] > 0.5


# ---------------------------------------------------------------------------
# learning curve: non-decreasing checkpoint votes; improves for the fixture above
# ---------------------------------------------------------------------------


def test_learning_curve_checkpoints_are_non_decreasing_and_improve_with_more_votes() -> None:
    candidates, analysis_map, training_votes, held_out_pairs, vector_by_id = (
        _well_powered_fixture()
    )

    comparison = compare_heads(
        training_votes=training_votes,
        held_out_pairs=held_out_pairs,
        candidates=candidates,
        analysis_map=analysis_map,
        lens_scorer=_zero_scorer,
        embedding_scorer_factory=_embedding_scorer_factory_from_vectors(vector_by_id),
        baseline_scorer=_zero_scorer,
    )

    votes_seen = [point["votes"] for point in comparison.learning_curve]
    assert votes_seen == sorted(votes_seen)
    assert len(comparison.learning_curve) >= 2

    first_accuracy = comparison.learning_curve[0]["embedding_held_out_accuracy"]
    final_accuracy = comparison.learning_curve[-1]["embedding_held_out_accuracy"]
    assert final_accuracy >= first_accuracy
    # The 10 "noisy" votes dominate the first (25%) checkpoint, inverting the
    # embedding head's ranking; by the final checkpoint the 33 correct votes
    # have won out. Not just non-decreasing — genuinely better.
    assert final_accuracy > first_accuracy


# ---------------------------------------------------------------------------
# promote_if_valid independence: individually "good enough" != head-to-head winner
# ---------------------------------------------------------------------------


def test_promote_if_valid_result_is_structurally_independent_of_verdict() -> None:
    """Both heads can be individually promotable (>= ``ACCURACY_THRESHOLD``,
    non-negative lift vs. baseline) while the head-to-head comparison is still
    ``insufficient_evidence`` — promotion and the head-to-head question are
    answered from entirely different evidence (each head's own held-out
    accuracy, vs. only the *discordant* pairs between the two heads)."""
    n = 20
    candidates = [{"id": f"c{i}", "baseline": 0.0} for i in range(n)]
    # Shuffle colorfulness relative to array index (index*7 mod n) so a
    # scorer that could only break ties by array-index position wouldn't
    # accidentally track the true preference order too.
    analysis_map = {
        f"c{i}": _result(f"c{i}", colorfulness=((i * 7) % n) / (n - 1)) for i in range(n)
    }

    def scorer(analysis: AnalysisResult) -> float:
        return analysis.color_story.colorfulness

    def embedding_scorer_factory(votes_subset: Sequence[VoteVectors]) -> Scorer:
        return scorer  # identical to the lens scorer -> zero disagreement

    ids = [c["id"] for c in candidates]
    held_out_pairs: list[tuple[str, str, str]] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            preferred = (
                a
                if analysis_map[a].color_story.colorfulness
                > analysis_map[b].color_story.colorfulness
                else b
            )
            held_out_pairs.append((a, b, preferred))

    comparison = compare_heads(
        training_votes=[],
        held_out_pairs=held_out_pairs,
        candidates=candidates,
        analysis_map=analysis_map,
        lens_scorer=scorer,
        embedding_scorer_factory=embedding_scorer_factory,
        baseline_scorer=_zero_scorer,
    )

    assert comparison.lens_promoted is True
    assert comparison.embedding_promoted is True
    # Identical scorers never disagree -> zero discordant pairs -> the
    # head-to-head question itself is unanswerable, independent of either
    # head's own promotion result.
    assert comparison.discordant_pairs == 0
    assert comparison.verdict == "insufficient_evidence"


# ---------------------------------------------------------------------------
# to_dict / from_dict round trip
# ---------------------------------------------------------------------------


def test_head_comparison_to_dict_from_dict_round_trip() -> None:
    candidates, analysis_map, training_votes, held_out_pairs, vector_by_id = (
        _well_powered_fixture()
    )
    comparison = compare_heads(
        training_votes=training_votes,
        held_out_pairs=held_out_pairs,
        candidates=candidates,
        analysis_map=analysis_map,
        lens_scorer=_zero_scorer,
        embedding_scorer_factory=_embedding_scorer_factory_from_vectors(vector_by_id),
        baseline_scorer=_zero_scorer,
    )

    rebuilt = HeadComparison.from_dict(json.loads(json.dumps(comparison.to_dict())))

    assert rebuilt == comparison


# ---------------------------------------------------------------------------
# no-deletion guard (T-09-10): structurally checked, not just asserted in prose
# ---------------------------------------------------------------------------


def test_head_comparison_has_no_deletion_field() -> None:
    names = [f.name.lower() for f in dataclasses.fields(HeadComparison)]
    assert not any("delete" in name or "retire" in name or "remove" in name for name in names)


def test_compare_module_never_mutates_a_persisted_profile() -> None:
    source = Path(compare_module.__file__).read_text()
    assert "save_profile" not in source
    assert "DELETE FROM" not in source
    assert "DROP TABLE" not in source
    assert "taste.controls" not in source
    assert "taste.profiles" not in source
