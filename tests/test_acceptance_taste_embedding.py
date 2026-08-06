"""Acceptance gate for the M009 Embedding Taste Head milestone (R039-R043).

This module ships the 10th and final deterministic, air-gapped acceptance file for
this milestone. Each scenario is **self-bootstrapping**: it mints its own catalog,
votes, and synthetic vectors over the isolated ``data_root`` (from conftest) and
drives the subsystem objects directly (``TasteVoteStore``, ``OnnxEmbeddingProvider``,
``EmbeddingStore``, ``fit_embedding_head``, ``attribute_score``/``find_exemplars``,
``compare_heads``) — never a live server, never a subprocess CLI, never the network.

* Scenario A (R039) — vote flow: cast, list, and retract a pairwise vote through
  ``TasteVoteStore`` directly; retract never deletes a ``taste_preferences`` row, only
  stamps ``retracted_at``, and a fresh ``rebuild_profile()`` agrees exactly with what
  ``retract`` already computed — history survives a rebuild by construction.
* Scenario B (R040) — embedding provider: the committed tiny fixture ONNX model only,
  never a real checkpoint, never a network call anywhere in this file; ``embed()`` is
  bit-identical on repeat calls; a missing model reports ``probe().ok is False``
  cleanly, never an exception; a cross-``model_version`` comparison raises
  ``EmbeddingVersionError`` rather than returning a plausible-looking float.
* Scenario C (R041) — head: the explicit, exact-equality zero-vote parity assertion
  (mirrors ``tests/test_taste_dialogue_upstream.py``'s
  ``test_every_consumer_degrades_to_baseline_on_an_empty_profile``'s exact-``==``
  style, not ``pytest.approx``), determinism, and the non-tautological
  ``capacity == len(vote_terms)`` guard at 2+ vote counts in the same scenario.
* Scenario D (R042) — attribution + exemplars: contributions sum exactly to the score;
  ``find_exemplars`` never returns a non-liked vector even when it is numerically
  closer; ``attribution.py`` imports no LLM/extraction machinery (structural scan).
* Scenario E (R043) — comparison: an underpowered fixture reports
  ``insufficient_evidence`` (a property of the data, not a permanently-stuck code
  path); a well-powered fixture reaches a real decision; ``HeadComparison``/
  ``compare.py`` carry no field or code path that deletes the incumbent lens head.
* Scenario F (reachability) — the check this milestone exists to run: every public
  symbol M009 added is referenced by ``cli.py``/``api.py``, and every table it added
  or extended has a production ``INSERT`` writer in ``src/curator/`` — the automated
  restatement of the M007 audit's Finding 0.
"""

from __future__ import annotations

import ast
import dataclasses
import io
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import curator.taste.embedding.attribution as attribution_module
import curator.taste.embedding.compare as compare_module
from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.catalog import Catalog
from curator.taste.embedding.attribution import attribute_score, find_exemplars
from curator.taste.embedding.compare import MIN_DISCORDANT_PAIRS, HeadComparison, compare_heads
from curator.taste.embedding.errors import EmbeddingVersionError
from curator.taste.embedding.head import VoteVectors, fit_embedding_head
from curator.taste.embedding.provider import EMBEDDING_DIM, OnnxEmbeddingProvider
from curator.taste.embedding.store import EmbeddingStore, cosine_similarity
from curator.taste.pairwise import Scorer
from curator.taste.profiles import TasteProfileKind, default_profile
from curator.taste.store import TasteVoteStore, next_pair

FIXTURE_MODEL = Path(__file__).parent / "fixtures" / "tiny_embedding_model.onnx"


# ---------------------------------------------------------------------------
# fixtures — every scenario builds its own world
# ---------------------------------------------------------------------------


def _image_bytes(color: tuple[int, int, int] = (80, 140, 200), fmt: str = "PNG") -> bytes:
    img = Image.new("RGB", (256, 256), color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _catalog_analysis(sha: str, colorfulness: float) -> AnalysisResult:
    return AnalysisResult(
        asset_id=sha,
        quality=QualitySignals(aesthetic_quality=0.5, technical_quality=0.5),
        color_story=ColorStory(colorfulness=colorfulness, harmony=0.5),
    )


def _candidate_result(cid: str, colorfulness: float) -> AnalysisResult:
    """A synthetic candidate whose ``asset_id`` is its own candidate id (compare fixtures)."""
    return AnalysisResult(
        asset_id=cid,
        quality=QualitySignals(aesthetic_quality=0.5),
        color_story=ColorStory(colorfulness=colorfulness),
    )


def _seed_catalog(data_root, entries: int = 6, quality_score: float | None = 0.5) -> None:
    """Build (and close) a catalog with *entries* analyzed entries.

    Mirrors ``tests/test_taste_vote_flow.py``'s ``_seed_catalog`` / ``tests/
    test_acceptance_taste_dialogue.py``'s ``_seed_history`` raw-SQL fixture idiom.
    Colorfulness spreads across ``[1/entries, 1.0]`` so ``choose_pair``/
    ``apply_preference`` have a real signal to separate on.
    """
    catalog = Catalog(data_root=data_root)
    try:
        db = catalog.db
        db.execute(
            "INSERT INTO source_connectors(connector_id, connector_type, name)"
            " VALUES ('local', 'local', 'local')"
        )
        for i in range(1, entries + 1):
            sha = f"{i:064d}"
            db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (sha, 100))
            db.execute(
                "INSERT INTO catalog_entries"
                "(connector_id, asset_id, revision, sha256, quality_score)"
                " VALUES ('local', ?, '1', ?, ?)",
                (f"asset-{i}", sha, quality_score),
            )
            analysis = _catalog_analysis(sha, i / entries)
            db.execute(
                "INSERT INTO analysis_results(catalog_entry_id, profile, engine_version,"
                " analysis_json, status) VALUES (?, 'standard', 'v1', ?, 'ok')",
                (i, json.dumps(analysis.to_dict())),
            )
        db.commit()
    finally:
        catalog.db.close()


def _synthetic_votes(n: int, dim: int = EMBEDDING_DIM, seed: int = 0) -> list[VoteVectors]:
    """*n* votes with random (but fixed-seed) winner/loser vectors (mirrors S03/S04)."""
    rng = np.random.RandomState(seed)
    votes = []
    for i in range(n):
        winner = rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)
        loser = rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)
        votes.append(
            VoteVectors(
                vote_group=f"acceptance-synthetic-{i}",
                winner_entry_id=2 * i + 1,
                loser_entry_id=2 * i + 2,
                winner_vector=winner,
                loser_vector=loser,
            )
        )
    return votes


def _zero_scorer(analysis: AnalysisResult) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# Scenario A — vote flow (R039)
# ---------------------------------------------------------------------------


def test_acceptance_taste_embedding_vote_cast_retract_preserves_history(data_root):
    """Cast a vote via next_pair -> record_vote, retract it, prove nothing was deleted."""
    _seed_catalog(data_root, entries=6)
    catalog = Catalog(data_root=data_root)
    try:
        store = TasteVoteStore(catalog)
        fresh_install = store.load_profile()
        assert fresh_install.version == 1  # the fresh-install value
        # zero votes reproduce the exact baseline profile (explicit equality, not
        # just a version check) — the R039/R038 zero-vote-means-baseline contract.
        assert fresh_install == default_profile(TasteProfileKind.PERSONAL)

        pair = next_pair(catalog)
        assert pair is not None
        a, b = pair
        record = store.record_vote(int(a["id"]), int(b["id"]), note="acceptance-scenario-a")

        # the vote appears via votes()
        assert record.vote_group in {v.vote_group for v in store.votes()}

        # the profile moved from the fresh-install value
        after_vote = store.load_profile()
        assert after_vote.version == 2
        assert after_vote.version != fresh_install.version
        assert after_vote.weights != fresh_install.weights

        # before retract: both rows present, neither retracted
        rows_before = catalog.db.execute(
            "SELECT retracted_at FROM taste_preferences WHERE vote_group = ?",
            (record.vote_group,),
        ).fetchall()
        assert len(rows_before) == 2
        assert all(row[0] is None for row in rows_before)

        assert store.retract(record.vote_group) is True

        # load_profile() matches a fresh rebuild_profile() computed with that
        # vote_group already excluded (retract() already replayed the journal;
        # an independent rebuild_profile() call must agree exactly).
        after_retract = store.load_profile()
        rebuilt = store.rebuild_profile()
        assert after_retract.weights == rebuilt.weights
        assert after_retract.version == rebuilt.version

        # history survives a rebuild: row count unchanged, only retracted_at changed
        rows_after = catalog.db.execute(
            "SELECT retracted_at FROM taste_preferences WHERE vote_group = ?",
            (record.vote_group,),
        ).fetchall()
        assert len(rows_after) == len(rows_before) == 2
        assert all(row[0] is not None for row in rows_after)
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# Scenario B — embedding provider (R040)
# ---------------------------------------------------------------------------


def test_acceptance_taste_embedding_provider_fixture_only_and_version_guarded(
    data_root, tmp_path
):
    """The fixture-only provider never guesses and cross-version comparisons are refused."""
    provider = OnnxEmbeddingProvider(model_path=FIXTURE_MODEL, model_version="acceptance-v1")
    assert provider.probe().ok is True

    data = _image_bytes()
    v1 = provider.embed(data)
    v2 = provider.embed(data)
    assert v1.shape == (EMBEDDING_DIM,)
    assert np.array_equal(v1, v2)  # exact, not approximate

    missing = OnnxEmbeddingProvider(model_path=tmp_path / "nonexistent.onnx")
    probe = missing.probe()  # never raises, even though the path does not exist
    assert probe.ok is False
    assert probe.message
    assert "not available" in probe.message

    catalog = Catalog(data_root=data_root)
    try:
        sha = "e" * 64
        catalog.db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (sha, 100))
        catalog.db.commit()
        embed_store = EmbeddingStore(catalog)
        embed_store.set(sha, "acceptance-v1", v1)
        embed_store.set(sha, "acceptance-v2", v1)
        stored_v1 = embed_store.get(sha, "acceptance-v1")
        stored_v2 = embed_store.get(sha, "acceptance-v2")
        assert stored_v1 is not None
        assert stored_v2 is not None
        with pytest.raises(EmbeddingVersionError):
            cosine_similarity(stored_v1, stored_v2)
    finally:
        catalog.db.close()


def test_acceptance_taste_embedding_module_has_no_network_imports() -> None:
    """This file itself never touches the network.

    AST-parsed rather than a naive substring scan, so the banned-token set below
    cannot trip the very check it defines by merely appearing as a string literal.
    """
    banned = {"requests", "urllib", "huggingface_hub", "socket", "httpx", "aiohttp"}
    tree = ast.parse(Path(__file__).read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    assert not (found & banned)


# ---------------------------------------------------------------------------
# Scenario C — head (R041)
# ---------------------------------------------------------------------------


def test_acceptance_taste_embedding_head_zero_vote_parity_and_capacity_guard() -> None:
    """Exact zero-vote parity, determinism, and the non-tautological capacity guard."""
    zero_head = fit_embedding_head([], "acceptance-v1")
    # exact `==`, not pytest.approx — mirrors the exact-equality style established at
    # tests/test_taste_dialogue_upstream.py's
    # test_every_consumer_degrades_to_baseline_on_an_empty_profile
    assert zero_head.score(np.ones(EMBEDDING_DIM, dtype=np.float32)) == 0.0
    assert zero_head.vote_terms == ()

    votes = _synthetic_votes(30)
    head1 = fit_embedding_head(votes, "acceptance-v1")
    head2 = fit_embedding_head(votes, "acceptance-v1")
    assert head1.alpha == head2.alpha
    assert len(head1.vote_terms) == len(head2.vote_terms)
    assert all(np.array_equal(a, b) for a, b in zip(head1.vote_terms, head2.vote_terms))

    # Non-tautological: a head that echoed a fake `capacity` while retaining a
    # fixed-size representation would fail this at 2+ independent N values.
    for n in (3, 15):
        head = fit_embedding_head(votes[:n], "acceptance-v1")
        assert head.capacity == n
        assert len(head.vote_terms) == n
        assert head.capacity == len(head.vote_terms)


# ---------------------------------------------------------------------------
# Scenario D — attribution + exemplars (R042)
# ---------------------------------------------------------------------------


def test_acceptance_taste_embedding_attribution_sums_to_score_and_exemplars_own_set_only(
    data_root,
):
    """Attribution sums exactly to the score; exemplars never leak a closer non-liked vector."""
    votes = _synthetic_votes(5, seed=7)
    head = fit_embedding_head(votes, "acceptance-v1")
    rng = np.random.RandomState(42)
    query = rng.uniform(-1.0, 1.0, size=(EMBEDDING_DIM,)).astype(np.float32)

    result = attribute_score(query, head, votes)
    total = sum(c["contribution"] for c in result.contributions)
    assert abs(total - result.score) < 1e-6

    catalog = Catalog(data_root=data_root)
    try:
        liked_sha = "1" * 64
        non_liked_sha = "2" * 64
        catalog.db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (liked_sha, 100))
        catalog.db.execute(
            "INSERT INTO content(sha256, size) VALUES (?, ?)", (non_liked_sha, 100)
        )
        catalog.db.commit()
        embed_store = EmbeddingStore(catalog)

        probe_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        probe_vector[0] = 1.0
        # non_liked is numerically identical to the probe (perfect cosine similarity);
        # liked is only partially aligned. liked must still be the only result.
        liked_vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        liked_vector[0] = 0.5
        liked_vector[1] = 0.5
        non_liked_vector = probe_vector.copy()
        embed_store.set(liked_sha, "acceptance-v1", liked_vector)
        embed_store.set(non_liked_sha, "acceptance-v1", non_liked_vector)

        exemplars = find_exemplars(
            probe_vector,
            [liked_sha],  # deliberately excludes non_liked_sha, though it is closer
            {liked_sha: 1, non_liked_sha: 2},
            embed_store,
            "acceptance-v1",
        )
        assert exemplars
        assert all(e.sha256 != non_liked_sha for e in exemplars)
    finally:
        catalog.db.close()

    # no-LLM-rationale constraint, checked structurally
    source = Path(attribution_module.__file__).read_text()
    assert "dialogue.extraction" not in source
    assert "CloudExtractionProvider" not in source


# ---------------------------------------------------------------------------
# Scenario E — comparison (R043)
# ---------------------------------------------------------------------------


def _well_powered_comparison_fixture() -> tuple[
    list[dict[str, Any]],
    dict[str, AnalysisResult],
    list[VoteVectors],
    list[tuple[str, str, str]],
    dict[str, np.ndarray],
]:
    """30 candidates whose true preference increases along a hidden axis the lens
    scorer never sees (every candidate has identical colorfulness); the embedding
    head is trained on 20 correctly-labelled votes along that same axis.
    """
    n = 30
    axis = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    axis[0] = 1.0
    candidates = [{"id": f"w{i}", "baseline": 0.0} for i in range(n)]
    analysis_map = {f"w{i}": _candidate_result(f"w{i}", colorfulness=0.5) for i in range(n)}
    vector_by_id = {f"w{i}": (i * axis).astype(np.float32) for i in range(n)}
    zero_vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    training_votes = [
        VoteVectors(
            vote_group=f"acceptance-well-powered-{i}",
            winner_entry_id=1000 + i,
            loser_entry_id=2000 + i,
            winner_vector=axis.copy(),
            loser_vector=zero_vec.copy(),
        )
        for i in range(20)
    ]
    held_out_pairs = [(f"w{2 * k}", f"w{2 * k + 1}", f"w{2 * k + 1}") for k in range(n // 2)]
    return candidates, analysis_map, training_votes, held_out_pairs, vector_by_id


def _embedding_scorer_factory_from_vectors(
    vector_by_id: dict[str, np.ndarray],
) -> Callable[[Sequence[VoteVectors]], Scorer]:
    """Same shape as ``cli.py``'s/``api.py``'s production factory (S05 precedent):
    refit fresh per call, score via a pre-resolved ``asset_id -> vector`` map.
    """

    def factory(votes_subset: Sequence[VoteVectors]) -> Scorer:
        head = fit_embedding_head(list(votes_subset), "acceptance-v1")

        def scorer(analysis: AnalysisResult) -> float:
            vector = vector_by_id.get(analysis.asset_id)
            return head.score(vector) if vector is not None else 0.0

        return scorer

    return factory


def test_acceptance_taste_embedding_comparison_insufficient_then_decisive() -> None:
    """Too few held-out pairs -> insufficient_evidence; a well-powered fixture reaches a
    real decision; HeadComparison/compare.py carry no field or branch that removes the
    lens head."""
    # -- underpowered: only 3 held-out pairs, can never reach MIN_DISCORDANT_PAIRS (10) --
    n = 6
    small_candidates = [{"id": f"u{i}", "baseline": 0.0} for i in range(n)]
    small_analysis = {
        f"u{i}": _candidate_result(f"u{i}", colorfulness=i / (n - 1)) for i in range(n)
    }

    def lens_scorer(analysis: AnalysisResult) -> float:
        return analysis.color_story.colorfulness

    def disagreeing_embedding_factory(votes_subset: Sequence[VoteVectors]) -> Scorer:
        def scorer(analysis: AnalysisResult) -> float:
            return -analysis.color_story.colorfulness  # always disagrees with lens_scorer

        return scorer

    underpowered_pairs = [("u0", "u1", "u1"), ("u2", "u3", "u3"), ("u4", "u5", "u5")]
    underpowered = compare_heads(
        training_votes=[],
        held_out_pairs=underpowered_pairs,
        candidates=small_candidates,
        analysis_map=small_analysis,
        lens_scorer=lens_scorer,
        embedding_scorer_factory=disagreeing_embedding_factory,
        baseline_scorer=_zero_scorer,
    )
    assert underpowered.discordant_pairs < MIN_DISCORDANT_PAIRS
    assert underpowered.verdict == "insufficient_evidence"
    assert underpowered.head_to_head_accuracy is None
    assert underpowered.head_to_head_ci is None

    # -- well-powered: a real decision is reachable, not a permanently stuck code path --
    candidates, analysis_map, training_votes, held_out_pairs, vector_by_id = (
        _well_powered_comparison_fixture()
    )
    decisive = compare_heads(
        training_votes=training_votes,
        held_out_pairs=held_out_pairs,
        candidates=candidates,
        analysis_map=analysis_map,
        lens_scorer=_zero_scorer,
        embedding_scorer_factory=_embedding_scorer_factory_from_vectors(vector_by_id),
        baseline_scorer=_zero_scorer,
    )
    assert decisive.discordant_pairs >= MIN_DISCORDANT_PAIRS
    assert decisive.verdict in {"embedding_better", "lens_better"}

    # -- HeadComparison carries no field, and compare.py no branch, that removes the
    # -- lens head (T-09-10), mirroring S05's own unit-level guard.
    field_names = [f.name.lower() for f in dataclasses.fields(HeadComparison)]
    assert not any("delete" in name or "retire" in name for name in field_names)
    source = Path(compare_module.__file__).read_text()
    assert "DELETE FROM" not in source
    assert "DROP TABLE" not in source
    assert "save_profile" not in source


# ---------------------------------------------------------------------------
# Scenario F — reachability: the check this milestone exists to run
# ---------------------------------------------------------------------------


def test_acceptance_taste_embedding_reachability_symbols_and_table_writers() -> None:
    """Every public symbol M009 added is referenced by cli.py/api.py, and every table
    it added or extended has a production INSERT writer in src/curator/ — the
    automated restatement of the M007 audit's Finding 0.
    """
    symbols = [
        "TasteVoteStore",
        "next_pair",
        "OnnxEmbeddingProvider",
        "EmbeddingStore",
        "fit_embedding_head",
        "attribute_score",
        "find_exemplars",
        "compare_heads",
    ]
    repo_root = Path(__file__).resolve().parents[1]
    cli_source = (repo_root / "src" / "curator" / "cli.py").read_text()
    api_source = (repo_root / "src" / "curator" / "api.py").read_text()
    for symbol in symbols:
        assert symbol in cli_source or symbol in api_source, (
            f"{symbol} is unreachable from both cli.py and api.py"
        )

    src_root = repo_root / "src" / "curator"
    combined = "\n".join(
        path.read_text() for path in sorted(src_root.rglob("*.py")) if "tests" not in path.parts
    )
    for statement in (
        "INSERT INTO taste_profiles",
        "INSERT INTO taste_preferences",
        "INSERT INTO content_embeddings",
    ):
        assert statement in combined, f"no production writer found for: {statement}"
