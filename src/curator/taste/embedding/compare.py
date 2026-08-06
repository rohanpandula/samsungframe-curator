"""Uncertainty-aware head-to-head comparison (M009/S05, R043).

``pairwise.evaluate``/``promote_if_valid`` (S02/M007, generalized in place by this
slice) answer "is this head good enough" against a fixed threshold — the right
question for gating one head's own promotion, and unchanged here. This module
answers a *different*, harder question honestly: "is the learned embedding head
better than the hand-crafted lens head", at the N (tens to low hundreds) a single
household actually produces, where one held-out accuracy number is statistically
underpowered to answer that on its own.

:func:`compare_heads` evaluates both heads independently (each against its own
gate), then restricts the head-to-head question to the **discordant pairs** —
the held-out pairs where the two heads' predictions actually disagree. Only
disagreement carries information about which head is *better* (a McNemar-shaped
test): if both heads predict the same winner, that pair is silent on which head
is more accurate. :func:`wilson_interval` reports a 95% confidence interval over
the embedding head's accuracy on just those pairs; below :data:`MIN_DISCORDANT_PAIRS`
the comparison honestly reports ``"insufficient_evidence"`` — a first-class
verdict, never a coin-flip winner dressed as a real one. A
:class:`HeadComparison` is a pure report: no field or code path expresses
"delete/retire the lens head" (T-09-10) — the incumbent is never at risk from a
single underpowered run.

This module deliberately does not import ``EmbeddingStore``/``fit_embedding_head``
itself, and never touches a catalog or database. Turning an ``AnalysisResult``
into an embedding-head score (fitting a head over a vote subset, resolving a
stored vector) is caller-side plumbing — the exact same "adapt with a one-line
closure at the call site" posture ``pairwise.evaluate``'s own generalization
established for ``TasteRanker.personal_delta``. Because the learning curve
(:func:`compare_heads` step 5) needs a *fresh* embedding scorer refit at several
different training-vote counts (not just once), the caller passes a **factory**
(``Sequence[VoteVectors] -> Scorer``) rather than a single already-fit
:data:`~curator.taste.pairwise.Scorer` — the smallest change that keeps this
module fitting-free while still letting it ask "how good is the head at N votes"
for several N using one uniform mechanism. See ``curator.cli``/``curator.api``'s
``compare`` command/route for the concrete factory (fits
:func:`~curator.taste.embedding.head.fit_embedding_head` fresh per call, scores
via a pre-resolved ``asset_id -> vector`` map so a missing vector scores ``0.0``
rather than raising mid-comparison).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from curator.analysis.schema import AnalysisResult
from curator.taste.embedding.head import VoteVectors
from curator.taste.pairwise import PairwiseEvidence, Scorer, evaluate, promote_if_valid

#: Below this many discordant (disagreement) pairs, no verdict is statistically
#: distinguishable from noise at conventional confidence. Worked example (the
#: research's own): roughly 100 held-out pairs at ~80% agreement leaves ~20
#: decidable (discordant) pairs, needing ~15/20 correct for p<0.05 under a
#: binomial test — 10 is a deliberately conservative floor beneath that, not a
#: fitted/tuned number.
MIN_DISCORDANT_PAIRS = 10

#: Fraction of the (non-retracted) vote history held out as the ground-truth
#: test set; callers take the most **recent** fraction (chronological, not
#: random) so the split is deterministic given unchanged vote history. CONTEXT
#: does not specify a split mechanism — this is Claude's discretion, documented
#: explicitly here rather than silently assumed.
HELD_OUT_FRACTION = 0.2


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the Wilson score confidence interval for *successes* out of *n*.

    Pure, no dependency (no scipy anywhere in this repo). ``n == 0`` returns the
    maximally uninformative interval ``(0.0, 1.0)`` rather than raising a
    division error — an explicit degenerate case, not an accidental crash.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    margin = (z / denom) * ((phat * (1.0 - phat) / n + (z * z) / (4 * n * n)) ** 0.5)
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class HeadComparison:
    """A head-to-head comparison report — pure data, never a mutation or an action.

    No field expresses "delete/retire the incumbent" — structurally guaranteed
    by this dataclass's own field list (checked by a source-scan unit test,
    T-09-10). ``head_to_head_accuracy``/``head_to_head_ci`` are ``None`` exactly
    when ``verdict == "insufficient_evidence"``; otherwise both are populated.
    """

    lens_evidence: PairwiseEvidence
    embedding_evidence: PairwiseEvidence
    lens_promoted: bool
    embedding_promoted: bool
    discordant_pairs: int
    discordant_correct_embedding: int
    head_to_head_accuracy: float | None
    head_to_head_ci: tuple[float, float] | None
    learning_curve: list[dict[str, Any]]
    verdict: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "learning_curve", [dict(c) for c in self.learning_curve])

    def to_dict(self) -> dict[str, Any]:
        return {
            "lens_evidence": self.lens_evidence.to_dict(),
            "embedding_evidence": self.embedding_evidence.to_dict(),
            "lens_promoted": self.lens_promoted,
            "embedding_promoted": self.embedding_promoted,
            "discordant_pairs": self.discordant_pairs,
            "discordant_correct_embedding": self.discordant_correct_embedding,
            "head_to_head_accuracy": self.head_to_head_accuracy,
            "head_to_head_ci": (
                list(self.head_to_head_ci) if self.head_to_head_ci is not None else None
            ),
            "learning_curve": [dict(c) for c in self.learning_curve],
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HeadComparison:
        if isinstance(data, cls):
            return data
        ci = data.get("head_to_head_ci")
        accuracy = data.get("head_to_head_accuracy")
        return cls(
            lens_evidence=PairwiseEvidence.from_dict(data["lens_evidence"]),
            embedding_evidence=PairwiseEvidence.from_dict(data["embedding_evidence"]),
            lens_promoted=bool(data["lens_promoted"]),
            embedding_promoted=bool(data["embedding_promoted"]),
            discordant_pairs=int(data["discordant_pairs"]),
            discordant_correct_embedding=int(data["discordant_correct_embedding"]),
            head_to_head_accuracy=(float(accuracy) if accuracy is not None else None),
            head_to_head_ci=((float(ci[0]), float(ci[1])) if ci is not None else None),
            learning_curve=[dict(c) for c in data.get("learning_curve", [])],
            verdict=str(data["verdict"]),
        )


def _predicted_winner(
    scorer: Scorer,
    aid: str,
    bid: str,
    analysis_map: dict[Any, AnalysisResult],
    baseline_map: dict[str, float],
) -> str:
    """Return *scorer*'s predicted winner of ``(aid, bid)`` — the ``_score``-equivalent
    ``baseline + scorer(analysis)`` comparison, ``>=`` breaking ties toward *aid*
    (the same tie-break :func:`~curator.taste.pairwise.evaluate` itself uses).
    """
    score_a = baseline_map[aid] + scorer(analysis_map[aid])
    score_b = baseline_map[bid] + scorer(analysis_map[bid])
    return aid if score_a >= score_b else bid


def compare_heads(
    training_votes: Sequence[VoteVectors],
    held_out_pairs: Sequence[tuple[str, str, str]],
    candidates: list[dict[str, Any]],
    analysis_map: dict[Any, AnalysisResult],
    lens_scorer: Scorer,
    embedding_scorer_factory: Callable[[Sequence[VoteVectors]], Scorer],
    baseline_scorer: Scorer,
) -> HeadComparison:
    """Compare the lens and embedding heads over the same held-out evidence.

    Pure — never mutates or persists anything, never takes a "retire the
    incumbent" branch (the return type structurally carries no such field).
    *training_votes* are resolved :class:`~curator.taste.embedding.head.VoteVectors`
    (already joined to their embedding vectors); *held_out_pairs* are
    ``(id_a, id_b, preferred_id)`` triples over the same id space as
    *candidates*/*analysis_map*, exactly what
    :func:`~curator.taste.pairwise.evaluate` itself expects.

    *embedding_scorer_factory* turns a votes subset into a fresh
    :data:`~curator.taste.pairwise.Scorer` (fits a head over that subset,
    scores by resolving each candidate's own embedding vector) — called once
    over the full *training_votes* for the main comparison, and once per
    learning-curve checkpoint over a prefix of *training_votes*, so "how good
    is the embedding head at N votes" is answered by the exact same mechanism
    at every N, never a special-cased first/last computation.

    1. Evaluate each scorer independently against *held_out_pairs*
       (``lens_evidence``/``embedding_evidence``), then gate each with the
       existing, unmodified :func:`~curator.taste.pairwise.promote_if_valid` —
       "is this head good enough", answered separately for both heads.
    2. Over *held_out_pairs*, find the **discordant** pairs — where the lens
       and embedding heads predict different winners. Only these carry
       information about which head is *better* (a McNemar-shaped test): a
       pair both heads call the same way is silent on that question.
    3. Fewer than :data:`MIN_DISCORDANT_PAIRS` discordant pairs ->
       ``verdict = "insufficient_evidence"``, ``head_to_head_accuracy``/
       ``head_to_head_ci`` both ``None`` — a sample-size gate, not a judgment.
       Otherwise the embedding head's accuracy on just the discordant pairs,
       plus its :func:`wilson_interval`, decide ``"embedding_better"``
       (CI lower bound ``> 0.5``), ``"lens_better"`` (CI upper bound ``< 0.5``),
       or ``"tie"`` (the interval straddles 0.5 — genuinely indeterminate even
       at adequate-looking N, distinct from ``"insufficient_evidence"``).
    4. A learning curve: at ``{n/4, n/2, 3n/4, n}`` training-vote checkpoints
       (``n = len(training_votes)``), refit the embedding head over just that
       prefix and re-evaluate against the same *held_out_pairs*, recording its
       held-out accuracy at each checkpoint — computed deterministically from
       the same vote history, no held-out re-sampling.
    """
    embedding_scorer = embedding_scorer_factory(training_votes)
    n_training = len(training_votes)

    lens_evidence = evaluate(
        lens_scorer,
        baseline_scorer,
        candidates,
        analysis_map,
        held_out_pairs,
        sample_efficiency_pairs=n_training,
    )
    embedding_evidence = evaluate(
        embedding_scorer,
        baseline_scorer,
        candidates,
        analysis_map,
        held_out_pairs,
        sample_efficiency_pairs=n_training,
    )
    lens_promoted = promote_if_valid(lens_evidence)
    embedding_promoted = promote_if_valid(embedding_evidence)

    baseline_map = {cand["id"]: float(cand["baseline"]) for cand in candidates}
    discordant_pairs = 0
    discordant_correct_embedding = 0
    for aid, bid, preferred in held_out_pairs:
        lens_pred = _predicted_winner(lens_scorer, aid, bid, analysis_map, baseline_map)
        embedding_pred = _predicted_winner(embedding_scorer, aid, bid, analysis_map, baseline_map)
        if lens_pred == embedding_pred:
            continue
        discordant_pairs += 1
        if embedding_pred == preferred:
            discordant_correct_embedding += 1

    head_to_head_accuracy: float | None
    head_to_head_ci: tuple[float, float] | None
    if discordant_pairs < MIN_DISCORDANT_PAIRS:
        head_to_head_accuracy = None
        head_to_head_ci = None
        verdict = "insufficient_evidence"
    else:
        head_to_head_accuracy = discordant_correct_embedding / discordant_pairs
        head_to_head_ci = wilson_interval(discordant_correct_embedding, discordant_pairs)
        if head_to_head_ci[0] > 0.5:
            verdict = "embedding_better"
        elif head_to_head_ci[1] < 0.5:
            verdict = "lens_better"
        else:
            verdict = "tie"

    checkpoints = sorted(
        {
            n
            for n in (
                max(1, n_training // 4),
                n_training // 2,
                (3 * n_training) // 4,
                n_training,
            )
            if n > 0
        }
    )
    learning_curve: list[dict[str, Any]] = []
    for n in checkpoints:
        checkpoint_scorer = embedding_scorer_factory(training_votes[:n])
        checkpoint_evidence = evaluate(
            checkpoint_scorer,
            baseline_scorer,
            candidates,
            analysis_map,
            held_out_pairs,
            sample_efficiency_pairs=n,
        )
        learning_curve.append(
            {"votes": n, "embedding_held_out_accuracy": checkpoint_evidence.held_out_accuracy}
        )

    return HeadComparison(
        lens_evidence=lens_evidence,
        embedding_evidence=embedding_evidence,
        lens_promoted=lens_promoted,
        embedding_promoted=embedding_promoted,
        discordant_pairs=discordant_pairs,
        discordant_correct_embedding=discordant_correct_embedding,
        head_to_head_accuracy=head_to_head_accuracy,
        head_to_head_ci=head_to_head_ci,
        learning_curve=learning_curve,
        verdict=verdict,
    )
