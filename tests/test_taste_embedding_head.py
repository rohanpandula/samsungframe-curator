"""Tests for the nonparametric preference head (M009/S03).

Every fit in this file uses synthetic :class:`VoteVectors` directly — no ONNX
inference, no fixture model. Only :func:`resolve_vote_vectors`'s DB-join
behavior needs a real (isolated) catalog + ``EmbeddingStore``, seeded with the
same raw-SQL idiom ``tests/test_taste_vote_flow.py`` established.
"""

from __future__ import annotations

import json

import numpy as np

from curator.catalog import Catalog
from curator.taste.embedding.head import (
    EmbeddingHead,
    VoteVectors,
    fit_embedding_head,
    resolve_vote_vectors,
)
from curator.taste.embedding.provider import EMBEDDING_DIM
from curator.taste.embedding.store import EmbeddingStore
from curator.taste.store import TasteVoteStore


def _synthetic_votes(n: int, dim: int = EMBEDDING_DIM, seed: int = 0) -> list[VoteVectors]:
    """*n* votes with random (but fixed-seed) winner/loser vectors."""
    rng = np.random.RandomState(seed)
    votes = []
    for i in range(n):
        winner = rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)
        loser = rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)
        votes.append(
            VoteVectors(
                vote_group=f"synthetic-{i}",
                winner_entry_id=2 * i + 1,
                loser_entry_id=2 * i + 2,
                winner_vector=winner,
                loser_vector=loser,
            )
        )
    return votes


def _axis_votes(n: int, dim: int = EMBEDDING_DIM, axis_index: int = 0) -> list[VoteVectors]:
    """*n* votes whose winner is systematically further along a known axis.

    ``winner = base + 0.5*axis``, ``loser = base - 0.5*axis`` for a random
    (fixed-seed) per-vote ``base`` — the difference term is exactly ``axis``
    regardless of ``base``, since ``base`` cancels in ``winner - loser``.
    """
    rng = np.random.RandomState(11)
    axis = np.zeros(dim, dtype=np.float32)
    axis[axis_index] = 1.0
    votes = []
    for i in range(n):
        base = rng.uniform(-0.2, 0.2, size=(dim,)).astype(np.float32)
        winner = base + 0.5 * axis
        loser = base - 0.5 * axis
        votes.append(
            VoteVectors(
                vote_group=f"axis-{i}",
                winner_entry_id=2 * i + 1,
                loser_entry_id=2 * i + 2,
                winner_vector=winner.astype(np.float32),
                loser_vector=loser.astype(np.float32),
            )
        )
    return votes


def _fixed_delta_votes(n: int, dim: int = EMBEDDING_DIM, axis_index: int = 0) -> list[VoteVectors]:
    """*n* votes whose ``winner - loser`` term is exactly the same vector.

    Isolates the shrinkage-vs-N relationship: "same underlying per-vote
    difference magnitude", nothing else varying.
    """
    axis = np.zeros(dim, dtype=np.float32)
    axis[axis_index] = 1.0
    winner = 0.5 * axis
    loser = -0.5 * axis
    return [
        VoteVectors(
            vote_group=f"fixed-{i}",
            winner_entry_id=2 * i + 1,
            loser_entry_id=2 * i + 2,
            winner_vector=winner.copy(),
            loser_vector=loser.copy(),
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# zero-vote parity
# ---------------------------------------------------------------------------


def test_fit_embedding_head_zero_votes_exact_parity():
    head = fit_embedding_head([], "v1")
    assert head.capacity == 0
    assert head.vote_terms == ()

    rng = np.random.RandomState(3)
    for _ in range(5):
        vector = rng.uniform(-5.0, 5.0, size=(EMBEDDING_DIM,)).astype(np.float32)
        assert head.score(vector) == 0.0
    assert head.score(np.zeros(EMBEDDING_DIM, dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_fit_embedding_head_deterministic_no_seed():
    votes = _synthetic_votes(12)
    head1 = fit_embedding_head(votes, "v1")
    head2 = fit_embedding_head(votes, "v1")
    assert head1.alpha == head2.alpha
    assert head1.capacity == head2.capacity
    assert len(head1.vote_terms) == len(head2.vote_terms)
    assert all(np.array_equal(a, b) for a, b in zip(head1.vote_terms, head2.vote_terms))


# ---------------------------------------------------------------------------
# WR-04: == / hash() must not crash on a head holding real vote_terms
# ---------------------------------------------------------------------------


def test_embedding_head_eq_and_hash_do_not_raise():
    """Before the fix, ``head1 == head2`` raised ``ValueError`` (numpy array
    truth-value ambiguity) and ``hash(head1)`` raised ``TypeError`` (unhashable
    ndarray) the moment a head held any votes — ``eq=False`` makes both safe
    (identity-based), never a confusing numpy error deep in dataclass
    internals. Structural fields are still comparable individually."""
    votes = _synthetic_votes(3)
    head1 = fit_embedding_head(votes, "v1")
    head2 = fit_embedding_head(votes, "v1")

    assert (head1 == head2) is False  # identity-based, not structural
    assert (head1 == head1) is True
    assert isinstance(hash(head1), int)  # does not raise
    # Structural equivalence is still checkable field-by-field.
    assert head1.alpha == head2.alpha
    assert all(np.array_equal(a, b) for a, b in zip(head1.vote_terms, head2.vote_terms))


# ---------------------------------------------------------------------------
# directional sanity check
# ---------------------------------------------------------------------------


def test_head_scores_further_along_the_preferred_axis_higher():
    votes = _axis_votes(20)
    head = fit_embedding_head(votes, "v1")
    axis = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    axis[0] = 1.0
    further_along = 2.0 * axis
    further_against = -2.0 * axis
    assert head.score(further_along) > head.score(further_against)


# ---------------------------------------------------------------------------
# R041: fitted parameter count proportionate to vote count (non-tautological)
# ---------------------------------------------------------------------------


def test_fitted_parameter_count_proportionate_to_vote_count():
    """The guard 01-ROADMAP.md's S03 line names explicitly.

    N=0 is covered separately by
    ``test_fit_embedding_head_zero_votes_exact_parity`` — the schedule holds
    down to zero, not just up from one; this test walks {1, 5, 20, 60}.
    """
    votes = _synthetic_votes(60)
    for n in (1, 5, 20, 60):
        head = fit_embedding_head(votes[:n], "v1")
        # Independently — a head that echoed a fake capacity while still
        # retaining a fixed-size representation would fail the 2nd/3rd.
        assert head.capacity == n
        assert len(head.vote_terms) == n
        assert head.capacity == len(head.vote_terms)
        for term in head.vote_terms:
            assert term.shape == (EMBEDDING_DIM,)


# ---------------------------------------------------------------------------
# shrinkage
# ---------------------------------------------------------------------------


def test_shrinkage_grows_effective_direction_norm_with_more_votes():
    """All else equal (identical per-vote difference magnitude), more votes ->
    less shrinkage -> a larger effective_direction() norm."""
    small = fit_embedding_head(_fixed_delta_votes(5), "v1")
    large = fit_embedding_head(_fixed_delta_votes(60), "v1")
    small_norm = float(np.linalg.norm(small.effective_direction()))
    large_norm = float(np.linalg.norm(large.effective_direction()))
    assert small_norm < large_norm


# ---------------------------------------------------------------------------
# to_dict / from_dict round trip
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip_preserves_state_exactly():
    votes = _synthetic_votes(7)
    head = fit_embedding_head(votes, "v-round-trip")
    restored = EmbeddingHead.from_dict(json.loads(json.dumps(head.to_dict())))

    assert restored.model_version == head.model_version
    assert restored.alpha == head.alpha
    assert restored.capacity == head.capacity
    assert len(restored.vote_terms) == len(head.vote_terms)
    for original, rebuilt in zip(head.vote_terms, restored.vote_terms):
        assert np.allclose(original, rebuilt)


# ---------------------------------------------------------------------------
# resolve_vote_vectors
# ---------------------------------------------------------------------------


def _seed_catalog_entries(catalog: Catalog, n: int) -> list[int]:
    """Insert *n* ``(content, catalog_entries)`` row pairs; return their entry ids."""
    db = catalog.db
    db.execute(
        "INSERT INTO source_connectors(connector_id, connector_type, name)"
        " VALUES ('local', 'local', 'local')"
    )
    entry_ids = []
    for i in range(1, n + 1):
        sha = f"{i:064d}"
        db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (sha, 100))
        db.execute(
            "INSERT INTO catalog_entries(connector_id, asset_id, revision, sha256)"
            " VALUES ('local', ?, '1', ?)",
            (f"asset-{i}", sha),
        )
        entry_ids.append(i)
    db.commit()
    return entry_ids


def test_resolve_vote_vectors_skips_votes_without_a_stored_embedding(data_root):
    catalog = Catalog(data_root=data_root)
    try:
        entry_ids = _seed_catalog_entries(catalog, 4)
        vote_store = TasteVoteStore(catalog)
        embedded_vote = vote_store.record_vote(entry_ids[0], entry_ids[1])
        unembedded_vote = vote_store.record_vote(entry_ids[2], entry_ids[3])

        embed_store = EmbeddingStore(catalog)
        rng = np.random.RandomState(0)
        embed_store.set(
            f"{entry_ids[0]:064d}",
            "v1",
            rng.uniform(-1.0, 1.0, size=(EMBEDDING_DIM,)).astype(np.float32),
        )
        embed_store.set(
            f"{entry_ids[1]:064d}",
            "v1",
            rng.uniform(-1.0, 1.0, size=(EMBEDDING_DIM,)).astype(np.float32),
        )
        # entry_ids[2] / entry_ids[3] (unembedded_vote's pair) are deliberately
        # never embedded.

        all_votes = vote_store.votes()
        resolved = resolve_vote_vectors(all_votes, embed_store, "v1")
    finally:
        catalog.db.close()

    assert len(all_votes) == 2
    assert len(resolved) == 1
    assert resolved[0].vote_group == embedded_vote.vote_group
    assert resolved[0].vote_group != unembedded_vote.vote_group
    assert resolved[0].winner_vector.shape == (EMBEDDING_DIM,)
    assert resolved[0].loser_vector.shape == (EMBEDDING_DIM,)
