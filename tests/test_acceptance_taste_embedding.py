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

Scenarios C (R041 — head), D (R042 — attribution/exemplars), E (R043 — comparison),
and F (structural reachability) land in the same file as this milestone's plan
progresses; this module grows incrementally, each scenario independently runnable.
"""

from __future__ import annotations

import ast
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.catalog import Catalog
from curator.taste.embedding.errors import EmbeddingVersionError
from curator.taste.embedding.provider import EMBEDDING_DIM, OnnxEmbeddingProvider
from curator.taste.embedding.store import EmbeddingStore, cosine_similarity
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
