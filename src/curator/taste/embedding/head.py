"""Nonparametric preference head over frozen embedding vectors (M009/S03).

Fits a deliberately low-capacity scorer over the S02 embedding vectors, using
only the votes S01 actually recorded
(:class:`~curator.taste.store.VoteRecord`). Research is explicit that below
roughly 100 votes a linear probe beats anything more elaborate, and that
fitting a raw 512-dim head from scratch at household N is near-certain
overfitting. This module commits to the *nonparametric* branch
``01-CONTEXT.md`` allows, literally: the head's retained parameters ARE the
recorded votes. Each vote contributes one delta term
``winner_vector - loser_vector``; :class:`EmbeddingHead` stores that list of
terms (``vote_terms``, one per vote) plus a single shared shrinkage
coefficient ``alpha``, and scores a query vector as::

    score(x) = sum_i alpha * dot(x, winner_i - loser_i)

This is mathematically identical (by linearity of the dot product) to a
single combined ``direction = alpha * sum_i(winner_i - loser_i)`` vector —
but that combined, fixed-size form is deliberately never cached as the
head's state. The stored representation's size is ``len(vote_terms) ==
capacity``, by construction, for every N, which is what makes "fitted
parameter count proportionate to vote count" a structural fact this design
cannot violate, not a number a metadata field happens to echo (R041).

``alpha`` is a single coefficient shared across every term (not a per-vote
dual weight a full representer-theorem/ridge solution would produce) —
deliberately: solving for per-vote-specific coefficients would reintroduce
an O(N)-free-parameter fitting procedure, undermining the "zero free
hyperparameters, closed-form" property this head guarantees.
:func:`fit_embedding_head` takes no seed parameter: this is a per-vote
mean-shrinkage rule, not a stochastic optimization, so identical votes plus
identical vectors produce an identical result unconditionally.

Each vote's contribution to any future score is ``alpha * dot(x, winner_i -
loser_i)``, and by construction these contributions sum exactly to
``score(x)`` — not as an algebraic decomposition proven after the fact, but
because :meth:`EmbeddingHead.score` is *defined* as that sum in the first
place. This is what keeps S04's attribution exact: it is the same per-vote
sum ``score()`` already computes, not a decomposition of a combined vector.

Scope: this head does not participate in
:func:`~curator.taste.pairwise.choose_pair`/
:func:`~curator.taste.pairwise.uncertainty_score` in M009 — S01's existing
linear-head chooser keeps selecting which pair a person compares next; this
head only fits and scores against votes that chooser already collected
(``01-RESEARCH.md`` Open Question #5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from curator.taste.embedding.provider import EMBEDDING_DIM
from curator.taste.embedding.store import EmbeddingStore
from curator.taste.store import VoteRecord

#: The vote count at which shrinkage reaches its ceiling
#: (``min(1.0, num_votes / SHRINKAGE_FLOOR)``) — a simple, explainable
#: proportional-shrinkage rule tying head strength to evidence volume rather
#: than a tunable ridge hyperparameter.
SHRINKAGE_FLOOR = 50


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp (mirrors ``taste.dialogue.store._utc_now``).

    A small, deliberate, in-idiom duplication rather than a cross-subsystem
    import — ``taste.embedding`` and ``taste.dialogue`` are separate
    subsystems that neither imports business logic from the other today
    (the same posture ``TasteVoteStore._signals_by_entry`` documented in
    M009/S01). This timestamp is metadata only: it plays no part in
    :func:`fit_embedding_head`'s closed-form, deterministic computation.
    """
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class VoteVectors:
    """One resolved vote: winner/loser catalog entries joined to their S02 vectors.

    The join between S01's vote history
    (:class:`~curator.taste.store.VoteRecord`) and S02's stored embeddings
    (:class:`~curator.taste.embedding.store.EmbeddingStore`) — the input
    :func:`fit_embedding_head` (and S04's attribution) consumes. Not intended
    for ``==``/hashing: carries numpy arrays, mirroring
    :class:`~curator.taste.embedding.store.StoredEmbedding`'s documented
    posture.
    """

    vote_group: str
    winner_entry_id: int
    loser_entry_id: int
    winner_vector: np.ndarray
    loser_vector: np.ndarray


def resolve_vote_vectors(
    votes: Sequence[VoteRecord],
    embedding_store: EmbeddingStore,
    model_version: str,
) -> list[VoteVectors]:
    """Join *votes* to their S02 vectors under *model_version*.

    Skips (never raises on) any **retracted** vote, and any vote whose
    winner or loser catalog entry has no stored embedding for
    *model_version* yet — a vote cast before its images were embedded, or
    embedded under a different model version, simply does not participate
    in this head's fit yet. This is the expected steady state right after
    S02 lands and before a ``--backfill`` runs, not an error.

    Returns the fully-resolved subset in the same chronological order as
    *votes* (:meth:`~curator.taste.store.TasteVoteStore.votes` already
    returns oldest-first; this function only filters, never reorders) —
    load-bearing for S04, which zips *votes* positionally against
    ``head.vote_terms``.
    """
    resolved: list[VoteVectors] = []
    for vote in votes:
        if vote.retracted:
            continue
        rows = embedding_store.db.execute(
            "SELECT id, sha256 FROM catalog_entries WHERE id IN (?, ?)",
            (vote.winner_entry_id, vote.loser_entry_id),
        ).fetchall()
        sha_by_entry = {int(entry_id): str(sha) for entry_id, sha in rows}
        winner_sha = sha_by_entry.get(vote.winner_entry_id)
        loser_sha = sha_by_entry.get(vote.loser_entry_id)
        if winner_sha is None or loser_sha is None:
            continue
        winner_stored = embedding_store.get(winner_sha, model_version)
        loser_stored = embedding_store.get(loser_sha, model_version)
        if winner_stored is None or loser_stored is None:
            continue
        resolved.append(
            VoteVectors(
                vote_group=vote.vote_group,
                winner_entry_id=vote.winner_entry_id,
                loser_entry_id=vote.loser_entry_id,
                winner_vector=winner_stored.vector,
                loser_vector=loser_stored.vector,
            )
        )
    return resolved


@dataclass(frozen=True, eq=False)
class EmbeddingHead:
    """A nonparametric preference scorer: the retained parameters ARE the votes.

    There is no ``direction: np.ndarray`` field. ``vote_terms`` holds one
    ``(dim,)`` ``np.float32`` delta vector per fitted vote — the retained
    parameters — so ``len(vote_terms) == capacity`` always, by construction
    (:meth:`__post_init__` asserts it). ``alpha`` is the single coefficient
    shared across every term (``0.0`` when ``capacity == 0``). ``capacity``
    is kept as a convenience field for display/logging — derived from, never
    an independent source of truth against, ``vote_terms``.

    Not intended for ``==``/hashing (WR-04): ``vote_terms`` carries numpy
    arrays, mirroring :class:`VoteVectors`'s (this module) and
    :class:`~curator.taste.embedding.store.StoredEmbedding`'s documented
    posture — a default (auto-generated) dataclass ``__eq__`` would raise
    ``ValueError`` comparing arrays elementwise, and its ``__hash__`` would
    raise ``TypeError`` hashing an unhashable ndarray. ``eq=False`` makes
    that structural (falls back to identity-based ``==``/``hash()``, both of
    which are safe, never a confusing numpy error deep in dataclass
    internals) rather than only a docstring promise. Compare fields
    individually (``np.array_equal`` for ``vote_terms``), never dataclass
    equality.

    :meth:`to_dict`/:meth:`from_dict` exist for round-trip-testability and
    this codebase's frozen-dataclass convention — **no schema/table persists
    an** ``EmbeddingHead`` **in this milestone**; every caller (this slice's
    own CLI diagnostic, S04's explain, S05's compare) calls
    :func:`fit_embedding_head` fresh each time.
    """

    model_version: str
    dim: int
    vote_terms: tuple[np.ndarray, ...]
    alpha: float
    capacity: int
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vote_terms",
            tuple(np.asarray(term, dtype=np.float32) for term in self.vote_terms),
        )
        assert len(self.vote_terms) == self.capacity, (
            "capacity must equal the retained parameter count by construction"
        )

    def score(self, vector: np.ndarray) -> float:
        """Return this head's score for *vector*.

        Structural zero-vote guard first: an empty ``vote_terms`` tuple
        returns exactly ``0.0`` via an explicit early return, not as an
        arithmetic coincidence of ``sum(()) == 0``. Otherwise sums
        ``alpha * dot(vector, term)`` **per term**, in ``vote_terms`` order —
        the exact same sequence of operations S04's ``attribute_score``
        performs per vote, so their results agree by construction.
        """
        if not self.vote_terms:
            return 0.0
        return float(sum(self.alpha * float(np.dot(vector, term)) for term in self.vote_terms))

    def effective_direction(self) -> np.ndarray:
        """Collapse the retained per-vote terms into one vector.

        **Display/debugging convenience only, not the head's state.** This
        exists purely for a human-readable norm/summary (e.g. the CLI
        diagnostic in ``cli.py``). Nothing in ``head.py``, ``attribution.py``,
        or ``compare.py`` may read this to compute a score — only
        :meth:`score` does that, over ``vote_terms`` directly.
        """
        if not self.vote_terms:
            return np.zeros(self.dim, dtype=np.float32)
        return (self.alpha * np.sum(np.stack(self.vote_terms), axis=0)).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "dim": self.dim,
            "vote_terms": [term.tolist() for term in self.vote_terms],
            "alpha": self.alpha,
            "capacity": self.capacity,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingHead:
        if isinstance(data, cls):
            return data
        return cls(
            model_version=str(data["model_version"]),
            dim=int(data["dim"]),
            vote_terms=tuple(
                np.asarray(term, dtype=np.float32) for term in data.get("vote_terms", [])
            ),
            alpha=float(data.get("alpha", 0.0)),
            capacity=int(data.get("capacity", 0)),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


def fit_embedding_head(
    votes: Sequence[VoteVectors], model_version: str, dim: int = EMBEDDING_DIM
) -> EmbeddingHead:
    """Fit a nonparametric :class:`EmbeddingHead` from *votes*.

    Pure and closed-form — no seed parameter, deliberately: this is a
    per-vote mean-shrinkage rule, not a stochastic optimization, so
    identical *votes* (identical stored vectors) produce an identical
    result unconditionally, with no RNG or gradient descent anywhere in the
    fit.

    ``capacity = len(votes)``. At ``capacity == 0`` the head carries no
    terms and an ``alpha`` of ``0.0`` (:meth:`EmbeddingHead.score` then
    returns exactly ``0.0`` for anything). Otherwise ``shrinkage = min(1.0,
    capacity / SHRINKAGE_FLOOR)`` and ``alpha = shrinkage / capacity`` — a
    single coefficient shared across every term, so the estimator has zero
    free hyperparameters to overfit. ``vote_terms`` is built by iterating
    *votes* once, one ``(winner_vector - loser_vector)`` delta per vote, in
    the same order as *votes* — this order is load-bearing for S04, which
    zips *votes* against ``head.vote_terms`` positionally.
    """
    now = _utc_now()
    capacity = len(votes)
    if capacity == 0:
        return EmbeddingHead(
            model_version=model_version,
            dim=dim,
            vote_terms=(),
            alpha=0.0,
            capacity=capacity,
            created_at=now,
            updated_at=now,
        )
    shrinkage = min(1.0, capacity / SHRINKAGE_FLOOR)
    alpha = shrinkage / capacity
    vote_terms = tuple(
        (vote.winner_vector - vote.loser_vector).astype(np.float32) for vote in votes
    )
    return EmbeddingHead(
        model_version=model_version,
        dim=dim,
        vote_terms=vote_terms,
        alpha=alpha,
        capacity=capacity,
        created_at=now,
        updated_at=now,
    )
