"""Tests for ``POST /api/taste/embedding-explain`` and ``GET /api/taste/compare``.

CR-01's review found these were the two M009 routes with **zero** ``TestClient``
coverage — exactly why the tie-break exploit (a zero-capacity embedding head
scoring 100% "accurate" and beating the lens head) went unnoticed: the pure-function
unit tests never touched the real ``held_out_pairs`` shape ``api.py``/``cli.py``
actually build from live ``VoteRecord``s. This file drives both routes through
``fastapi.testclient.TestClient`` against a real (isolated) catalog, using only the
committed tiny synthetic ONNX fixture (never a real checkpoint, never a network
call) — mirroring ``tests/test_taste_vote_flow.py``'s API-testing pattern.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from acceptance_harness import run_cli
from curator.analysis.pipeline import AnalysisAsset, AnalysisPipeline
from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.api import create_app
from curator.catalog import Catalog
from curator.cli import EXIT_OK
from curator.connectors.local import LocalConnector
from curator.taste.embedding.provider import EMBEDDING_MODEL_VERSION, OnnxEmbeddingProvider
from curator.taste.embedding.store import EmbeddingStore

FIXTURE_MODEL = Path(__file__).parent / "fixtures" / "tiny_embedding_model.onnx"


def _use_fixture_model(monkeypatch) -> None:
    """Point the (internally-constructed, no-injection-seam) ``OnnxEmbeddingProvider``
    every ``api.py`` route builds at the committed tiny fixture model, never a real
    checkpoint — the same env-var override :func:`resolve_model_path` documents.
    """
    monkeypatch.setenv("CURATOR_TASTE_EMBEDDING_MODEL_PATH", str(FIXTURE_MODEL))


def _seed_catalog(data_root, entries: int = 2, quality_score: float | None = None) -> None:
    """Build (and close) a catalog with *entries* analyzed entries, raw-SQL.

    ``quality_score=None`` (the default) gives every candidate an identical
    ``baseline`` of ``0.0`` — deliberately, so a zero-capacity embedding scorer
    (which always contributes a ``0.0`` delta) ties against the baseline on
    every held-out pair, reproducing CR-01's exact exploit precondition. Mirrors
    ``tests/test_acceptance_taste_embedding.py``'s ``_seed_catalog`` raw-SQL idiom.
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
            analysis = AnalysisResult(
                asset_id=sha,
                quality=QualitySignals(aesthetic_quality=0.5, technical_quality=0.5),
                color_story=ColorStory(colorfulness=i / entries, harmony=0.5),
            )
            db.execute(
                "INSERT INTO analysis_results(catalog_entry_id, profile, engine_version,"
                " analysis_json, status) VALUES (?, 'standard', 'v1', ?, 'ok')",
                (i, json.dumps(analysis.to_dict())),
            )
        db.commit()
    finally:
        catalog.db.close()


def _cataloged_and_analyzed(catalog: Catalog, tmp_path, name: str, color) -> tuple[int, str, bytes]:
    """Register + analyze one real PNG; return ``(entry_id, sha256, data)``.

    Registers real ContentStore bytes (via :meth:`Catalog.add_source`) and runs
    the real :class:`AnalysisPipeline`, so ``POST /api/taste/embedding-explain``
    (which needs actual image bytes to embed) can resolve it — mirrors
    ``tests/test_taste_vote_flow.py``'s ``_cataloged_and_analyzed`` helper,
    additionally returning the raw bytes this file's fixture-embedding setup needs.
    """
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color).save(buf, format="PNG")
    data = buf.getvalue()
    folder = tmp_path / "fixture"
    folder.mkdir(exist_ok=True)
    asset = folder / name
    asset.write_bytes(data)
    connector = LocalConnector(folder)
    sha = catalog.add_source(connector.connector_id, str(asset.resolve()), data)
    entry_id = catalog.get_by_hash(sha)[0]["id"]
    AnalysisPipeline(catalog).run([AnalysisAsset(entry_id=entry_id, source=data)])
    return entry_id, sha, data


# ---------------------------------------------------------------------------
# GET /api/taste/compare — shape coverage
# ---------------------------------------------------------------------------


def test_api_compare_returns_full_report_shape(data_root, monkeypatch):
    _use_fixture_model(monkeypatch)
    _seed_catalog(data_root, entries=2)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))

    resp = client.get("/api/taste/compare")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "lens_evidence",
        "embedding_evidence",
        "lens_promoted",
        "embedding_promoted",
        "discordant_pairs",
        "discordant_correct_embedding",
        "head_to_head_accuracy",
        "head_to_head_ci",
        "learning_curve",
        "verdict",
    }
    assert set(body["lens_evidence"]) == {
        "held_out_pairs",
        "held_out_accuracy",
        "ranking_lift_vs_baseline",
        "sample_efficiency_pairs",
        "requires_validation",
    }


def test_api_compare_400_with_fewer_than_two_analyzed_candidates(data_root, monkeypatch):
    _use_fixture_model(monkeypatch)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    resp = client.get("/api/taste/compare")
    assert resp.status_code == 400


def test_api_compare_503_when_embedding_provider_unavailable(data_root, monkeypatch):
    """No ``CURATOR_TASTE_EMBEDDING_MODEL_PATH`` override -> the default (isolated,
    empty) data root has no model placed -> probe fails cleanly, 503, not a crash."""
    _seed_catalog(data_root, entries=2)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    resp = client.get("/api/taste/compare")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# CR-01 regression: a zero-capacity embedding head must never win on ties
# ---------------------------------------------------------------------------


def test_api_compare_zero_capacity_embedding_head_is_never_rewarded_by_ties(
    data_root, monkeypatch
):
    """Live, through ``TestClient(create_app(...))``, with a seeded catalog and one
    real cast vote — reproducing CR-01's third repro exactly. No ``--backfill`` has
    ever run and the cast vote's images were never explicitly embedded, so the
    embedding head trained over the (empty-resolvable) vote history has zero
    capacity: it must never be reported as 100% accurate or "embedding_better"
    purely because it ties against an identical (``quality_score=None``) baseline
    on every held-out pair.
    """
    _use_fixture_model(monkeypatch)
    _seed_catalog(data_root, entries=2)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))

    pair = client.get("/api/taste/pair").json()
    assert pair["available"] is True
    vote_resp = client.post(
        "/api/taste/vote",
        json={
            "prefer": "a",
            "note": "",
            "a_entry_id": pair["a"]["entry_id"],
            "b_entry_id": pair["b"]["entry_id"],
        },
    )
    assert vote_resp.status_code == 200

    resp = client.get("/api/taste/compare")
    assert resp.status_code == 200
    body = resp.json()

    # The embedding head trained on this vote history has resolved zero votes
    # (never embedded) — must never be reported as having any evidence at all.
    assert body["embedding_evidence"]["sample_efficiency_pairs"] == 0
    # This is the exact bug: the old tie-break reported 1.0/"embedding_better"/
    # promoted=True here. A head with zero information must score at or below
    # chance, never be promoted, and never win the head-to-head comparison.
    assert body["embedding_evidence"]["held_out_accuracy"] <= 0.5
    assert body["embedding_promoted"] is False
    assert body["verdict"] != "embedding_better"


# ---------------------------------------------------------------------------
# POST /api/taste/embedding-explain — shape coverage
# ---------------------------------------------------------------------------


def test_api_embedding_explain_returns_score_rationale_contributions_exemplars(
    data_root, tmp_path, monkeypatch
):
    _use_fixture_model(monkeypatch)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    entry_a, sha_a, data_a = _cataloged_and_analyzed(catalog, tmp_path, "a.png", (200, 30, 30))
    entry_b, sha_b, data_b = _cataloged_and_analyzed(catalog, tmp_path, "b.png", (30, 30, 200))

    vote_resp = client.post(
        "/api/taste/vote",
        json={"prefer": "a", "a_entry_id": entry_a, "b_entry_id": entry_b},
    )
    assert vote_resp.status_code == 200

    # Pre-embed BOTH sides of the cast vote (the route itself only embeds its
    # own target on demand) so the vote is actually resolvable and the
    # contributions/exemplars branches — not just the empty-evidence path —
    # get real coverage here.
    provider = OnnxEmbeddingProvider(
        model_path=FIXTURE_MODEL, model_version=EMBEDDING_MODEL_VERSION
    )
    store = EmbeddingStore(catalog)
    store.set(sha_a, EMBEDDING_MODEL_VERSION, provider.embed(data_a))
    store.set(sha_b, EMBEDDING_MODEL_VERSION, provider.embed(data_b))

    resp = client.post("/api/taste/embedding-explain", json={"asset": sha_a})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"entry_id", "sha256", "score", "rationale", "contributions", "exemplars"}
    assert body["entry_id"] == entry_a  # WR-03
    assert body["sha256"] == sha_a
    assert isinstance(body["score"], float)
    assert isinstance(body["rationale"], str) and body["rationale"]
    assert len(body["contributions"]) == 1  # the one vote, now fully resolvable
    assert body["contributions"][0]["contribution"] != 0.0 or body["score"] == 0.0


def test_api_embedding_explain_503_when_embedding_provider_unavailable(data_root, tmp_path):
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    _entry_a, sha_a, _data_a = _cataloged_and_analyzed(catalog, tmp_path, "a.png", (200, 30, 30))
    resp = client.post("/api/taste/embedding-explain", json={"asset": sha_a})
    assert resp.status_code == 503


def test_api_embedding_explain_uncataloged_bytes_503s_not_a_bad_entry_id(data_root, monkeypatch):
    """A ``bytes``-only request for genuinely new (never-cataloged) content 503s —
    ``content_embeddings`` has a real foreign key to ``content(sha256)``, and the
    ``bytes`` path never inserts a ``content`` row, so storing a freshly-computed
    vector fails before ``entry_id`` resolution is ever reached. WR-03's ``entry_id``
    is therefore ``None``-safe by construction (see ``_entry_id_for_sha``) rather
    than reachable as a ``200`` here; this test documents the actual (pre-existing,
    unrelated-to-WR-03) boundary rather than asserting a path that cannot occur."""
    _use_fixture_model(monkeypatch)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 200, 10)).save(buf, format="PNG")
    inline = base64.b64encode(buf.getvalue()).decode("ascii")

    resp = client.post("/api/taste/embedding-explain", json={"bytes": inline})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# WR-03: CLI/API embedding-explain payload parity — the two surfaces the
# route's own docstring claims produce "the identical JSON shape"
# ---------------------------------------------------------------------------


def test_embedding_explain_cli_and_api_payloads_match_key_for_key(data_root, tmp_path, monkeypatch):
    _use_fixture_model(monkeypatch)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    entry_a, sha_a, data_a = _cataloged_and_analyzed(catalog, tmp_path, "a.png", (200, 30, 30))
    entry_b, sha_b, data_b = _cataloged_and_analyzed(catalog, tmp_path, "b.png", (30, 30, 200))

    vote_resp = client.post(
        "/api/taste/vote",
        json={"prefer": "a", "a_entry_id": entry_a, "b_entry_id": entry_b},
    )
    assert vote_resp.status_code == 200

    # Pre-embed both sides so the vote is fully resolvable on both surfaces —
    # neither call below then mutates content_embeddings, keeping the two
    # computations over byte-identical inputs.
    provider = OnnxEmbeddingProvider(
        model_path=FIXTURE_MODEL, model_version=EMBEDDING_MODEL_VERSION
    )
    store = EmbeddingStore(catalog)
    store.set(sha_a, EMBEDDING_MODEL_VERSION, provider.embed(data_a))
    store.set(sha_b, EMBEDDING_MODEL_VERSION, provider.embed(data_b))

    rc, out = run_cli(["taste", "embedding-explain", sha_a, "--json"])
    assert rc == EXIT_OK
    cli_result = json.loads(out)

    api_result = client.post("/api/taste/embedding-explain", json={"asset": sha_a}).json()

    assert set(cli_result) == set(api_result)
    assert cli_result == api_result
