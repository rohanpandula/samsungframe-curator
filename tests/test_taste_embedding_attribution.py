"""Tests for checkable attribution + exemplars (M009/S04, R042).

Every fit in this file uses synthetic :class:`VoteVectors` directly, mirroring
``tests/test_taste_embedding_head.py``'s pattern — no ONNX inference, no fixture
model. Only :func:`find_exemplars` needs a real (isolated) ``EmbeddingStore``, seeded
with the same raw-SQL idiom already established in ``tests/test_taste_embedding_head.py``
/ ``tests/test_taste_embedding_provider.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import curator.taste.embedding.attribution as attribution_module
from curator.catalog import Catalog
from curator.taste.embedding.attribution import (
    AttributionResult,
    ExemplarResult,
    attribute_score,
    find_exemplars,
    render_rationale,
)
from curator.taste.embedding.head import VoteVectors, fit_embedding_head
from curator.taste.embedding.provider import EMBEDDING_DIM
from curator.taste.embedding.store import EmbeddingStore


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


@pytest.fixture
def catalog(data_root) -> Catalog:
    """A migrated Catalog under an isolated data root (schema v17 present)."""
    cat = Catalog(data_root=data_root)
    yield cat
    cat.db.close()


def _register_content(catalog: Catalog, sha: str) -> None:
    """Insert a minimal ``content`` row so FK-constrained embedding inserts succeed."""
    catalog.db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (sha, 100))
    catalog.db.commit()


# ---------------------------------------------------------------------------
# attribute_score: sums exactly to the score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 10])
def test_attribution_sums_to_score(n: int) -> None:
    votes = _synthetic_votes(n)
    head = fit_embedding_head(votes, "v1")
    rng = np.random.RandomState(99)
    vector = rng.uniform(-1.0, 1.0, size=(EMBEDDING_DIM,)).astype(np.float32)

    result = attribute_score(vector, head, votes)

    assert len(result.contributions) == n
    total = sum(c["contribution"] for c in result.contributions)
    assert abs(total - result.score) < 1e-6


def test_attribute_score_zero_vote_head_is_empty_evidence() -> None:
    zero_head = fit_embedding_head([], "v1")
    rng = np.random.RandomState(1)
    vector = rng.uniform(-1.0, 1.0, size=(EMBEDDING_DIM,)).astype(np.float32)

    result = attribute_score(vector, zero_head, [])

    assert result.score == 0.0
    assert result.contributions == []


def test_attribute_score_contribution_keys_match_the_contract() -> None:
    votes = _synthetic_votes(2)
    head = fit_embedding_head(votes, "v1")
    vector = np.ones(EMBEDDING_DIM, dtype=np.float32)

    result = attribute_score(vector, head, votes)

    for contribution, vote in zip(result.contributions, votes, strict=True):
        assert contribution["vote_group"] == vote.vote_group
        assert contribution["winner_entry_id"] == vote.winner_entry_id
        assert contribution["loser_entry_id"] == vote.loser_entry_id
        assert isinstance(contribution["contribution"], float)


# ---------------------------------------------------------------------------
# IN-02: to_dict / from_dict round trips
# ---------------------------------------------------------------------------


def test_attribution_result_to_dict_from_dict_round_trip() -> None:
    votes = _synthetic_votes(3)
    head = fit_embedding_head(votes, "v1")
    vector = np.ones(EMBEDDING_DIM, dtype=np.float32)
    result = attribute_score(vector, head, votes)

    rebuilt = AttributionResult.from_dict(json.loads(json.dumps(result.to_dict())))

    assert rebuilt.score == result.score
    assert rebuilt.contributions == result.contributions


def test_exemplar_result_to_dict_from_dict_round_trip() -> None:
    exemplar = ExemplarResult(sha256="a" * 64, entry_id=7, similarity=0.42)

    rebuilt = ExemplarResult.from_dict(json.loads(json.dumps(exemplar.to_dict())))

    assert rebuilt == exemplar


# ---------------------------------------------------------------------------
# find_exemplars: own-liked-set-only, never a crash
# ---------------------------------------------------------------------------


def test_find_exemplars_only_returns_liked_shas_even_when_a_non_liked_one_is_closer(
    catalog: Catalog,
) -> None:
    store = EmbeddingStore(catalog)
    liked_sha = "1" * 64
    non_liked_sha = "2" * 64
    _register_content(catalog, liked_sha)
    _register_content(catalog, non_liked_sha)

    query = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0

    # non_liked is numerically identical to the query (perfect cosine similarity);
    # liked is only partially aligned. liked must still be the only result because
    # non_liked is not in liked_shas.
    liked_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    liked_vector[0] = 0.5
    liked_vector[1] = 0.5
    non_liked_vector = query.copy()

    store.set(liked_sha, "v1", liked_vector)
    store.set(non_liked_sha, "v1", non_liked_vector)

    results = find_exemplars(
        query,
        [liked_sha],
        {liked_sha: 1, non_liked_sha: 2},
        store,
        "v1",
    )

    assert results
    assert all(r.sha256 == liked_sha for r in results)
    assert all(r.sha256 != non_liked_sha for r in results)
    assert results[0].entry_id == 1


def test_find_exemplars_empty_liked_shas_returns_empty_list_never_raises(
    catalog: Catalog,
) -> None:
    store = EmbeddingStore(catalog)
    query = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    assert find_exemplars(query, [], {}, store, "v1") == []


def test_find_exemplars_liked_sha_with_no_stored_vector_returns_empty_list(
    catalog: Catalog,
) -> None:
    store = EmbeddingStore(catalog)
    query = np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # "liked" but never embedded — an honest "no exemplars yet", not a crash.
    assert find_exemplars(query, ["3" * 64], {"3" * 64: 3}, store, "v1") == []


# ---------------------------------------------------------------------------
# IN-01: zero-vector guard — no NaN "false positive" exemplar
# ---------------------------------------------------------------------------


def test_find_exemplars_zero_query_vector_returns_empty_list_not_nan(catalog: Catalog) -> None:
    """A zero-norm *query* vector has no defined direction — must return ``[]``,
    never a ``NaN``-similarity result (which ``numpy.argsort`` would otherwise
    sort as the *most* similar, ahead of every real match)."""
    store = EmbeddingStore(catalog)
    liked_sha = "1" * 64
    _register_content(catalog, liked_sha)
    liked_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    liked_vector[0] = 1.0
    store.set(liked_sha, "v1", liked_vector)

    query = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    assert find_exemplars(query, [liked_sha], {liked_sha: 1}, store, "v1") == []


def test_find_exemplars_excludes_a_zero_stored_vector_not_nan(catalog: Catalog) -> None:
    """A stored zero vector (``OnnxEmbeddingProvider.embed`` can legitimately
    return one rather than dividing by zero at the source) must be excluded
    before the division, never surfaced as a spurious ``NaN``-similarity "most
    similar" result ahead of a real, non-zero match."""
    store = EmbeddingStore(catalog)
    zero_sha = "1" * 64
    real_sha = "2" * 64
    _register_content(catalog, zero_sha)
    _register_content(catalog, real_sha)
    store.set(zero_sha, "v1", np.zeros(EMBEDDING_DIM, dtype=np.float32))
    real_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    real_vector[0] = 1.0
    store.set(real_sha, "v1", real_vector)

    query = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0
    results = find_exemplars(
        query, [zero_sha, real_sha], {zero_sha: 1, real_sha: 2}, store, "v1"
    )
    assert results
    assert all(r.sha256 != zero_sha for r in results)
    assert all(not np.isnan(r.similarity) for r in results)


# ---------------------------------------------------------------------------
# render_rationale: deterministic template, never free text
# ---------------------------------------------------------------------------


def test_render_rationale_contains_direction_and_score_and_is_deterministic() -> None:
    attribution = AttributionResult(
        score=0.4321,
        contributions=[
            {"vote_group": "g1", "winner_entry_id": 1, "loser_entry_id": 2, "contribution": 0.3},
            {
                "vote_group": "g2",
                "winner_entry_id": 3,
                "loser_entry_id": 4,
                "contribution": 0.1321,
            },
        ],
    )
    exemplars = [ExemplarResult(sha256="a" * 64, entry_id=5, similarity=0.87)]

    text1 = render_rationale(attribution, exemplars)
    text2 = render_rationale(attribution, exemplars)

    assert text1 == text2
    assert "ranked up" in text1
    assert "+0.432" in text1
    assert "g1" in text1  # strongest contribution (0.3 > 0.1321)


def test_render_rationale_direction_words_and_empty_evidence() -> None:
    up = AttributionResult(score=0.5, contributions=[])
    down = AttributionResult(score=-0.5, contributions=[])
    unchanged = AttributionResult(score=0.0, contributions=[])

    assert "ranked up" in render_rationale(up, [])
    assert "ranked down" in render_rationale(down, [])
    assert "unchanged" in render_rationale(unchanged, [])


# ---------------------------------------------------------------------------
# structural no-LLM guarantee (fast unit-level tripwire; independently
# re-verified by S06's acceptance reachability scan)
# ---------------------------------------------------------------------------


def test_attribution_module_never_imports_llm_or_extraction_machinery() -> None:
    source = Path(attribution_module.__file__)
    text = source.read_text()
    assert "dialogue.extraction" not in text
    assert "CloudExtractionProvider" not in text
