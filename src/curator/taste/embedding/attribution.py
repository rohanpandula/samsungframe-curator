"""Checkable attribution + exemplars for the embedding preference head (M009/S04, R042).

Every learned score this milestone produces must open to a *provable* per-vote
decomposition, not a plausible-looking approximation. S03's
:class:`~curator.taste.embedding.head.EmbeddingHead` was built specifically to make this
possible: it scores a query vector as ``sum_i alpha * dot(x, vote_terms_i)`` — one
retained delta term per fitted vote, not a single combined direction — so
:func:`attribute_score` does not *decompose* the score after the fact, it performs the
exact same per-term sum
:meth:`~curator.taste.embedding.head.EmbeddingHead.score` itself is defined as. This
mirrors :meth:`curator.taste.rank.TasteRanker.personal_delta`'s ``(delta,
contributions)`` evidence contract — the established analog in this codebase for "a
rerank score, opened up into the terms that sum to it exactly."

:func:`find_exemplars` never reaches outside the user's own liked (vote-winner) images —
a generic/external reference set would turn "why did the model prefer this" into a
plausible-sounding lie dressed as evidence. :func:`render_rationale` is a fixed,
deterministic string template over the same numbers :class:`AttributionResult`/
:class:`ExemplarResult` already carry — mirroring
:func:`curator.taste.dialogue.upstream._delta_rationale`'s direction-word-plus-
strongest-term idiom. No language model, extraction provider, or free-text generation is
imported anywhere in this file — a structural, source-checkable guarantee (verified both
by this module's own unit test and S06's acceptance reachability scan), not merely a
docstring claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from curator.taste.embedding.head import EmbeddingHead, VoteVectors
from curator.taste.embedding.store import EmbeddingStore


@dataclass(frozen=True)
class AttributionResult:
    """A head's score, opened into the exact per-vote terms that sum to it.

    ``contributions`` sums to ``score`` within float tolerance by construction — see
    :func:`attribute_score`. Each entry is ``{"vote_group", "winner_entry_id",
    "loser_entry_id", "contribution"}``.
    """

    score: float
    contributions: list[dict[str, Any]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "contributions", [dict(c) for c in self.contributions])

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "contributions": [dict(c) for c in self.contributions]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AttributionResult:
        """Round-trip inverse of :meth:`to_dict` (IN-02).

        Not on any persistence path in this milestone (this result is always
        freshly computed, never deserialized) — added for symmetry with every
        other frozen dataclass this phase touches (``EmbeddingHead``,
        ``HeadComparison``, ``PairwiseEvidence``), matching CLAUDE.md's stated
        frozen-dataclass ``to_dict``/``from_dict`` round-trip convention.
        """
        if isinstance(data, cls):
            return data
        return cls(
            score=float(data["score"]),
            contributions=[dict(c) for c in data.get("contributions", [])],
        )


def attribute_score(
    vector: np.ndarray, head: EmbeddingHead, votes: Sequence[VoteVectors]
) -> AttributionResult:
    """Decompose ``head.score(vector)`` into its exact per-vote contributions.

    *votes* must be the **same** :class:`~curator.taste.embedding.head.VoteVectors` list,
    in the same order, that :func:`~curator.taste.embedding.head.fit_embedding_head`
    produced *head* from — ``head.vote_terms[i]`` corresponds positionally to
    ``votes[i]`` by construction (S03's revised head stores one retained delta term per
    vote, not a single combined vector). ``zip(..., strict=True)`` turns a violation of
    that precondition into an immediate ``ValueError`` rather than a silently-truncated,
    wrongly-summing attribution.

    This function reads ``head.alpha`` directly — it does not import ``SHRINKAGE_FLOOR``
    or recompute shrinkage independently; ``head.alpha`` is the single source of truth
    for the per-vote coefficient, already computed once by
    :func:`~curator.taste.embedding.head.fit_embedding_head`.

    ``head.capacity == 0`` returns the empty-evidence case immediately, consistent with
    :meth:`~curator.taste.embedding.head.EmbeddingHead.score`'s own zero-vote guard.
    Otherwise each contribution is ``head.alpha * dot(vector, term)`` — the exact same
    per-term formula :meth:`~curator.taste.embedding.head.EmbeddingHead.score` itself
    sums over, so ``sum(contributions) == score`` is the same computation performed
    twice, not an algebraic identity verified after the fact.
    """
    if head.capacity == 0:
        return AttributionResult(score=0.0, contributions=[])
    contributions = [
        {
            "vote_group": v.vote_group,
            "winner_entry_id": v.winner_entry_id,
            "loser_entry_id": v.loser_entry_id,
            "contribution": head.alpha * float(np.dot(vector, term)),
        }
        for v, term in zip(votes, head.vote_terms, strict=True)
    ]
    return AttributionResult(score=head.score(vector), contributions=contributions)


@dataclass(frozen=True)
class ExemplarResult:
    """One nearest-neighbour exemplar drawn only from the user's own liked images."""

    sha256: str
    entry_id: int
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "entry_id": self.entry_id, "similarity": self.similarity}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExemplarResult:
        """Round-trip inverse of :meth:`to_dict` (IN-02) — see
        :meth:`AttributionResult.from_dict`'s docstring for why this exists
        despite never being persisted in this milestone."""
        if isinstance(data, cls):
            return data
        return cls(
            sha256=str(data["sha256"]),
            entry_id=int(data["entry_id"]),
            similarity=float(data["similarity"]),
        )


def find_exemplars(
    vector: np.ndarray,
    liked_shas: Sequence[str],
    sha_to_entry_id: dict[str, int],
    embedding_store: EmbeddingStore,
    model_version: str,
    *,
    top_k: int = 3,
) -> list[ExemplarResult]:
    """Return up to *top_k* nearest liked exemplars for *vector*, by cosine similarity.

    *liked_shas* is the set of content shas that were a vote **winner**
    (``preference=1``) in any non-retracted vote — resolved by the caller from
    :meth:`~curator.taste.store.TasteVoteStore.votes` plus a sha lookup; this function
    itself takes only the already-resolved sha list, keeping it a pure numpy operation
    with no catalog/db access. *sha_to_entry_id* is a small caller-supplied mapping (the
    same callers already build while resolving *liked_shas*) used only to attach an
    ``entry_id`` to each result — this function never queries the catalog itself, so it
    never grows a hidden dependency the "otherwise-pure" claim would then be false about.

    ``shas, matrix = embedding_store.get_matrix(model_version)``; filtered to rows whose
    sha is in *liked_shas*; brute-force cosine similarity
    (``matrix_filtered @ vector / (np.linalg.norm(matrix_filtered, axis=1) *
    np.linalg.norm(vector))``); sorted descending, top *top_k* taken.

    Never raises: returns ``[]`` when *liked_shas* is empty or none of them have a
    stored vector under *model_version* yet — an honest "no exemplars yet", not a crash.
    A zero-norm *vector*, or a stored row that is itself a zero vector (IN-01 —
    :meth:`~curator.taste.embedding.provider.OnnxEmbeddingProvider.embed` can
    return one rather than dividing by zero at the source), has no meaningfully
    defined cosine similarity — such rows are excluded before the division rather
    than left to produce ``NaN`` (which :func:`numpy.argsort` would otherwise sort
    as *greater than* any float, silently surfacing a ``NaN`` row as the *most*
    similar exemplar).
    """
    if not liked_shas:
        return []
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        return []
    liked_set = set(liked_shas)
    shas, matrix = embedding_store.get_matrix(model_version)
    keep = [i for i, sha in enumerate(shas) if sha in liked_set]
    if not keep:
        return []
    filtered_shas = [shas[i] for i in keep]
    filtered_matrix = matrix[keep]
    row_norms = np.linalg.norm(filtered_matrix, axis=1)
    nonzero = row_norms > 0
    if not np.any(nonzero):
        return []
    filtered_shas = [sha for sha, keep_row in zip(filtered_shas, nonzero) if keep_row]
    filtered_matrix = filtered_matrix[nonzero]
    row_norms = row_norms[nonzero]
    similarities = (filtered_matrix @ vector) / (row_norms * vector_norm)
    order = np.argsort(similarities)[::-1][:top_k]
    return [
        ExemplarResult(
            sha256=filtered_shas[i],
            entry_id=sha_to_entry_id[filtered_shas[i]],
            similarity=float(similarities[i]),
        )
        for i in order
    ]


def render_rationale(attribution: AttributionResult, exemplars: Sequence[ExemplarResult]) -> str:
    """Render a deterministic one-line rationale over already-computed numbers.

    A fixed string template, never free-text generation: direction word (``"ranked
    up"``/``"ranked down"``/``"unchanged"`` from the sign of ``attribution.score``,
    mirroring :func:`curator.taste.dialogue.upstream._delta_rationale`'s idiom), the
    strongest single vote's ``vote_group`` and its contribution magnitude (when there is
    one), and the top exemplar's short sha + similarity (when there is one). Calling this
    twice with identical inputs returns identical strings — deterministic by
    construction, no randomness or external call anywhere in this function.
    """
    score = attribution.score
    direction = "ranked up" if score > 0 else "ranked down" if score < 0 else "unchanged"
    text = f"{direction} ({score:+.3f})"
    if attribution.contributions:
        top_vote = max(
            attribution.contributions,
            key=lambda c: (abs(float(c["contribution"])), str(c["vote_group"])),
        )
        text += (
            f" — strongest vote {top_vote['vote_group']}"
            f" ({float(top_vote['contribution']):+.3f})"
        )
    if exemplars:
        top_exemplar = exemplars[0]
        text += (
            f"; most similar to your own {top_exemplar.sha256[:12]}"
            f" (similarity {top_exemplar.similarity:.2f})"
        )
    return text
