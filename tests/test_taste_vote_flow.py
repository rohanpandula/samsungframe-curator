"""Tests for the M009/S01 vote-capture loop (src/curator/taste/store.py + wiring).

Proves the CLI -> API -> profile round trip R039 requires: a vote cast through
:class:`~curator.taste.store.TasteVoteStore` directly, through the CLI
(``curator taste vote|votes|retract``), and through the API (``POST
/api/taste/vote`` etc., the same surface the webui's Taste Deck panel calls) is
visible through all three read paths and moves the persisted
``taste_profiles`` row. Retract reverses a vote without deleting any row, and
rebuilding the profile from the ``taste_preferences`` journal reproduces the
same state deterministically. A fresh (zero-vote) profile is byte-identical to
``default_profile(TasteProfileKind.PERSONAL)``.
"""

from __future__ import annotations

import io
import json

from fastapi.testclient import TestClient
from PIL import Image

from acceptance_harness import run_cli
from curator.analysis.pipeline import AnalysisAsset, AnalysisPipeline
from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.api import create_app
from curator.catalog import Catalog
from curator.cli import EXIT_NO_CHANGE, EXIT_OK
from curator.connectors.local import LocalConnector
from curator.taste.profiles import TasteProfileKind, default_profile
from curator.taste.store import TasteVoteStore, next_pair, resolve_vote_candidates


def _seed_catalog(data_root, entries: int = 4, quality_score: float | None = 0.5) -> None:
    """Build (and close) a catalog with *entries* analyzed entries.

    Colorfulness spreads across ``[1/entries, 1.0]`` so ``choose_pair``/
    ``apply_preference`` have a real signal to separate on — mirrors
    ``tests/test_taste_dialogue_profile.py``'s ``_seed_history`` fixture, the
    established pattern for a raw-SQL analyzed-entries fixture in this suite.
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


def _cataloged_and_analyzed(catalog: Catalog, tmp_path, name: str, color) -> tuple[int, str]:
    """Register + analyze one real PNG; return ``(entry_id, sha256)``.

    Unlike :func:`_seed_catalog`, this registers real ContentStore bytes (via
    :meth:`Catalog.add_source`) and runs the real :class:`AnalysisPipeline`, so
    ``POST /api/taste/explain`` (which re-analyzes the image fresh) can resolve
    it — mirrors ``tests/test_acceptance_webui.py``'s ``_cataloged`` helper.
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
    return entry_id, sha


# ---------------------------------------------------------------------------
# resolve_vote_candidates / next_pair
# ---------------------------------------------------------------------------


def test_resolve_vote_candidates_string_ids_and_baseline_fallback(data_root):
    _seed_catalog(data_root, entries=2, quality_score=None)
    catalog = Catalog(data_root=data_root)
    try:
        candidates, analysis_map = resolve_vote_candidates(catalog)
    finally:
        catalog.db.close()
    assert {c["id"] for c in candidates} == {"1", "2"}
    assert all(isinstance(c["id"], str) for c in candidates)
    assert all(c["baseline"] == 0.0 for c in candidates)  # NULL quality_score -> 0.0
    assert set(analysis_map) == {"1", "2"}


def test_next_pair_deterministic_given_unchanged_state(data_root):
    _seed_catalog(data_root, entries=5)
    catalog = Catalog(data_root=data_root)
    try:
        first = next_pair(catalog)
        second = next_pair(catalog)
    finally:
        catalog.db.close()
    assert first is not None
    assert first == second


def test_next_pair_none_when_fewer_than_two_candidates(data_root):
    catalog = Catalog(data_root=data_root)
    try:
        assert next_pair(catalog) is None  # zero analyzed entries
    finally:
        catalog.db.close()

    _seed_catalog(data_root, entries=1)
    catalog = Catalog(data_root=data_root)
    try:
        assert next_pair(catalog) is None  # exactly one analyzed entry
    finally:
        catalog.db.close()


def test_next_pair_none_when_every_pair_already_voted(data_root):
    _seed_catalog(data_root, entries=2)
    catalog = Catalog(data_root=data_root)
    try:
        a, b = next_pair(catalog)
        TasteVoteStore(catalog).record_vote(int(a["id"]), int(b["id"]))
        assert next_pair(catalog) is None
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# TasteVoteStore — zero-vote parity, record_vote, votes(), retract
# ---------------------------------------------------------------------------


def test_zero_votes_is_byte_identical_to_default_profile(data_root):
    _seed_catalog(data_root, entries=3)
    catalog = Catalog(data_root=data_root)
    try:
        loaded = TasteVoteStore(catalog).load_profile()
    finally:
        catalog.db.close()
    assert loaded == default_profile(TasteProfileKind.PERSONAL)


def test_record_vote_persists_grouped_row_pair(data_root):
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    try:
        record = TasteVoteStore(catalog).record_vote(1, 2, note="prefer 1")
        rows = catalog.db.execute(
            "SELECT catalog_entry_id, preference, vote_group, note, retracted_at"
            " FROM taste_preferences WHERE vote_group = ? ORDER BY id",
            (record.vote_group,),
        ).fetchall()
    finally:
        catalog.db.close()
    assert rows == [
        (1, 1, record.vote_group, "prefer 1", None),
        (2, -1, record.vote_group, "prefer 1", None),
    ]


def test_record_vote_moves_and_persists_the_lens_profile(data_root):
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    try:
        store = TasteVoteStore(catalog)
        before = store.load_profile()
        store.record_vote(1, 4, note="")  # colorfulness 0.25 vs 1.0: real separation
        after = store.load_profile()
    finally:
        catalog.db.close()
    assert before == default_profile(TasteProfileKind.PERSONAL)
    assert after.version == 2
    assert after.weights != before.weights

    # Persisted, not just in-memory: a fresh Catalog/store reads the same row back.
    catalog2 = Catalog(data_root=data_root)
    try:
        reloaded = TasteVoteStore(catalog2).load_profile()
    finally:
        catalog2.db.close()
    assert reloaded.weights == after.weights
    assert reloaded.version == after.version


def test_votes_lists_grouped_oldest_first_including_retracted(data_root):
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    try:
        store = TasteVoteStore(catalog)
        first = store.record_vote(1, 2)
        second = store.record_vote(3, 4)
        listed = store.votes()
        assert [v.vote_group for v in listed] == [first.vote_group, second.vote_group]
        assert all(v.retracted is False for v in listed)

        assert store.retract(first.vote_group) is True
        listed = store.votes()
    finally:
        catalog.db.close()
    # History is never hidden: both groups are still listed, one flagged retracted.
    by_group = {v.vote_group: v for v in listed}
    assert len(listed) == 2
    assert by_group[first.vote_group].retracted is True
    assert by_group[second.vote_group].retracted is False


def test_retract_unknown_vote_group_is_noop(data_root):
    _seed_catalog(data_root, entries=2)
    catalog = Catalog(data_root=data_root)
    try:
        assert TasteVoteStore(catalog).retract("does-not-exist") is False
    finally:
        catalog.db.close()


def test_retract_twice_is_noop_the_second_time(data_root):
    _seed_catalog(data_root, entries=2)
    catalog = Catalog(data_root=data_root)
    try:
        store = TasteVoteStore(catalog)
        record = store.record_vote(1, 2)
        assert store.retract(record.vote_group) is True
        assert store.retract(record.vote_group) is False
    finally:
        catalog.db.close()


def test_retract_reverses_profile_without_deleting_rows(data_root):
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    try:
        store = TasteVoteStore(catalog)
        record = store.record_vote(1, 4)
        assert store.load_profile().version == 2

        assert store.retract(record.vote_group) is True
        after_retract = store.load_profile()
        assert after_retract.weights == default_profile(TasteProfileKind.PERSONAL).weights

        row_count = catalog.db.execute(
            "SELECT COUNT(*) FROM taste_preferences WHERE vote_group = ?",
            (record.vote_group,),
        ).fetchone()[0]
        retracted_ats = catalog.db.execute(
            "SELECT retracted_at FROM taste_preferences WHERE vote_group = ?",
            (record.vote_group,),
        ).fetchall()
    finally:
        catalog.db.close()
    assert row_count == 2  # retract never deletes
    assert all(r[0] is not None for r in retracted_ats)


def test_rebuild_profile_reproduces_state_deterministically(data_root):
    _seed_catalog(data_root, entries=6)
    catalog = Catalog(data_root=data_root)
    try:
        store = TasteVoteStore(catalog)
        store.record_vote(1, 6)
        store.record_vote(2, 5)
        store.record_vote(3, 4)
        live = store.load_profile()
        rebuilt_once = store.rebuild_profile()
        rebuilt_twice = store.rebuild_profile()
    finally:
        catalog.db.close()
    assert live.weights == rebuilt_once.weights == rebuilt_twice.weights
    assert live.version == rebuilt_once.version == rebuilt_twice.version


def test_rebuild_survives_a_retraction_mid_history(data_root):
    """A retracted vote is excluded from replay; the surviving votes still fold in."""
    _seed_catalog(data_root, entries=6)
    catalog = Catalog(data_root=data_root)
    try:
        store = TasteVoteStore(catalog)
        first = store.record_vote(1, 6)
        store.record_vote(2, 5)
        store.retract(first.vote_group)
        after_retract = store.load_profile()

        # Independent rebuild from the (partially retracted) journal agrees exactly.
        rebuilt = store.rebuild_profile()
    finally:
        catalog.db.close()
    assert rebuilt.weights == after_retract.weights
    assert rebuilt.version == after_retract.version
    # Not simply back to baseline: the surviving (non-retracted) vote still counts.
    assert rebuilt.weights != default_profile(TasteProfileKind.PERSONAL).weights


# ---------------------------------------------------------------------------
# CLI: curator taste vote | votes | retract
# ---------------------------------------------------------------------------


def test_cli_vote_preview_is_deterministic_and_records_nothing(data_root):
    _seed_catalog(data_root, entries=4)
    rc1, out1 = run_cli(["taste", "vote"])
    rc2, out2 = run_cli(["taste", "vote"])
    assert rc1 == rc2 == EXIT_NO_CHANGE
    assert out1 == out2
    assert "current pair" in out1

    catalog = Catalog(data_root=data_root)
    try:
        assert TasteVoteStore(catalog).votes() == []
    finally:
        catalog.db.close()


def test_cli_vote_answer_records_and_reports_new_version(data_root):
    _seed_catalog(data_root, entries=4)
    rc, out = run_cli(["taste", "vote", "--prefer", "a", "--note", "cli vote"])
    assert rc == EXIT_OK
    assert "recorded" in out
    assert "profile now version 2" in out

    catalog = Catalog(data_root=data_root)
    try:
        votes = TasteVoteStore(catalog).votes()
    finally:
        catalog.db.close()
    assert len(votes) == 1
    assert votes[0].note == "cli vote"


def test_cli_votes_lists_recorded_votes_and_json(data_root):
    _seed_catalog(data_root, entries=4)
    run_cli(["taste", "vote", "--prefer", "b", "--note", "listed"])

    rc, out = run_cli(["taste", "votes"])
    assert rc == EXIT_OK
    assert "listed" in out

    rc, out = run_cli(["taste", "votes", "--json"])
    assert rc == EXIT_OK
    payload = json.loads(out)
    assert len(payload) == 1
    assert payload[0]["note"] == "listed"
    assert payload[0]["retracted"] is False


def test_cli_votes_empty_still_exits_ok(data_root):
    _seed_catalog(data_root, entries=2)
    rc, out = run_cli(["taste", "votes"])
    assert rc == EXIT_OK
    assert "none yet" in out


def test_cli_retract_unknown_exits_no_change(data_root):
    _seed_catalog(data_root, entries=2)
    rc, out = run_cli(["taste", "retract", "nonexistent"])
    assert rc == EXIT_NO_CHANGE
    assert "no vote with id" in out


def test_cli_retract_reverses_and_reports_version(data_root):
    _seed_catalog(data_root, entries=4)
    run_cli(["taste", "vote", "--prefer", "a"])
    catalog = Catalog(data_root=data_root)
    try:
        vote_group = TasteVoteStore(catalog).votes()[0].vote_group
    finally:
        catalog.db.close()

    rc, out = run_cli(["taste", "retract", vote_group])
    assert rc == EXIT_OK
    assert "profile now version 1" in out

    rc, out = run_cli(["taste", "retract", vote_group])
    assert rc == EXIT_NO_CHANGE  # already retracted


# ---------------------------------------------------------------------------
# API: GET /api/taste/pair, POST /api/taste/vote, GET /api/taste/votes,
#      POST /api/taste/retract, POST /api/taste/explain
# ---------------------------------------------------------------------------


def test_api_pair_unavailable_with_fewer_than_two_candidates(data_root):
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    body = client.get("/api/taste/pair").json()
    assert body == {
        "available": False,
        "reason": "fewer than two analyzed candidates remaining",
    }


def test_api_vote_matches_webui_request_shape_and_round_trips(data_root):
    """Mirrors webui/app.js's submitVote() body shape exactly."""
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))

    pair = client.get("/api/taste/pair").json()
    assert pair["available"] is True

    resp = client.post(
        "/api/taste/vote",
        json={
            "prefer": "a",
            "note": "",
            "a_entry_id": pair["a"]["entry_id"],
            "b_entry_id": pair["b"]["entry_id"],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["winner_entry_id"] == pair["a"]["entry_id"]
    assert body["loser_entry_id"] == pair["b"]["entry_id"]
    assert body["profile_version"] == 2

    votes = client.get("/api/taste/votes").json()
    assert votes["count"] == 1
    assert votes["votes"][0]["vote_group"] == body["vote_group"]


def test_api_vote_400_on_invalid_prefer(data_root):
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    pair = client.get("/api/taste/pair").json()
    resp = client.post(
        "/api/taste/vote",
        json={
            "prefer": "c",
            "a_entry_id": pair["a"]["entry_id"],
            "b_entry_id": pair["b"]["entry_id"],
        },
    )
    assert resp.status_code == 400


def test_api_vote_409_on_stale_pair_and_records_nothing(data_root):
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))

    resp = client.post(
        "/api/taste/vote", json={"prefer": "a", "a_entry_id": 999, "b_entry_id": 998}
    )
    assert resp.status_code == 409
    assert client.get("/api/taste/votes").json()["count"] == 0


def test_api_retract_404_on_unknown_vote_group(data_root):
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    resp = client.post("/api/taste/retract", json={"vote_group": "nope"})
    assert resp.status_code == 404


def test_api_retract_reverses_and_is_visible_via_votes(data_root):
    _seed_catalog(data_root, entries=4)
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    pair = client.get("/api/taste/pair").json()
    vote_group = client.post(
        "/api/taste/vote",
        json={
            "prefer": "a",
            "a_entry_id": pair["a"]["entry_id"],
            "b_entry_id": pair["b"]["entry_id"],
        },
    ).json()["vote_group"]

    resp = client.post("/api/taste/retract", json={"vote_group": vote_group})
    assert resp.status_code == 200
    assert resp.json()["profile_version"] == 1

    votes = client.get("/api/taste/votes").json()["votes"]
    assert votes[0]["retracted"] is True  # visible, not hidden


def test_api_explain_cites_persisted_profile_and_is_baseline_at_zero_votes(
    data_root, tmp_path
):
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    entry_a, sha_a = _cataloged_and_analyzed(catalog, tmp_path, "a.png", (200, 30, 30))
    _entry_b, sha_b = _cataloged_and_analyzed(catalog, tmp_path, "b.png", (30, 30, 200))

    baseline = client.post("/api/taste/explain", json={"asset": sha_a}).json()
    assert baseline["delta"] == 0.0
    assert baseline["rationale"] == "baseline order (no taste profile applied)"

    vote_group = client.post(
        "/api/taste/vote",
        json={"prefer": "a", "a_entry_id": entry_a, "b_entry_id": _entry_b},
    ).json()["vote_group"]

    moved = client.post("/api/taste/explain", json={"asset": sha_a}).json()
    assert moved["delta"] != 0.0

    client.post("/api/taste/retract", json={"vote_group": vote_group})
    reverted = client.post("/api/taste/explain", json={"asset": sha_a}).json()
    assert reverted == baseline  # byte-identical to the pre-vote baseline
    assert sha_b  # both fixture images exercised end-to-end


# ---------------------------------------------------------------------------
# Cross-surface: the same vote is visible through TasteVoteStore, the CLI,
# and the API — R039's central contract.
# ---------------------------------------------------------------------------


def test_a_vote_cast_on_any_surface_is_visible_on_all_three_read_paths(data_root):
    _seed_catalog(data_root, entries=8)

    # 1. Cast directly through TasteVoteStore.
    catalog = Catalog(data_root=data_root)
    try:
        direct = TasteVoteStore(catalog).record_vote(1, 2, note="direct")
    finally:
        catalog.db.close()

    # 2. Cast through the CLI.
    rc, _out = run_cli(["taste", "vote", "--prefer", "a", "--note", "cli"])
    assert rc == EXIT_OK
    catalog = Catalog(data_root=data_root)
    try:
        cli_group = next(v.vote_group for v in TasteVoteStore(catalog).votes() if v.note == "cli")
    finally:
        catalog.db.close()

    # 3. Cast through the API (the same surface the webui Taste Deck calls).
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    pair = client.get("/api/taste/pair").json()
    assert pair["available"] is True
    api_group = client.post(
        "/api/taste/vote",
        json={
            "prefer": "a",
            "note": "api",
            "a_entry_id": pair["a"]["entry_id"],
            "b_entry_id": pair["b"]["entry_id"],
        },
    ).json()["vote_group"]

    all_groups = {direct.vote_group, cli_group, api_group}

    # -- read path 1: TasteVoteStore.votes() ---------------------------------
    catalog = Catalog(data_root=data_root)
    try:
        via_store = {v.vote_group for v in TasteVoteStore(catalog).votes()}
    finally:
        catalog.db.close()
    assert all_groups <= via_store

    # -- read path 2: `curator taste votes` ----------------------------------
    rc, out = run_cli(["taste", "votes", "--json"])
    assert rc == EXIT_OK
    via_cli = {v["vote_group"] for v in json.loads(out)}
    assert all_groups <= via_cli

    # -- read path 3: GET /api/taste/votes -----------------------------------
    catalog2 = Catalog(data_root=data_root)
    client2 = TestClient(create_app(catalog=catalog2))
    via_api = {v["vote_group"] for v in client2.get("/api/taste/votes").json()["votes"]}
    assert all_groups <= via_api

    # And the profile moved by all three votes together (version 1 + 3 votes).
    catalog3 = Catalog(data_root=data_root)
    try:
        assert TasteVoteStore(catalog3).load_profile().version == 4
    finally:
        catalog3.db.close()
