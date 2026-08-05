"""Tests for src/curator/taste/dialogue/store + session (M008/S01/T2).

Covers the append-only observation journal (add/by_session/all/count) with
exact field preservation and no dedup/update, per-session grouping, the v14
schema migration (tables + index + SCHEMA_VERSION), TasteSession
round-trips, and SessionStore new_session/close.
"""

from __future__ import annotations

import json
from dataclasses import replace

from curator import db as _db
from curator.catalog import Catalog
from curator.schema import EXPECTED_TABLES, SCHEMA_VERSION
from curator.taste.dialogue import (
    ImageRef,
    ObservationStore,
    Polarity,
    SessionStore,
    TasteObservation,
    TasteSession,
    create_observation,
)


def _image_refs() -> list[ImageRef]:
    return [
        ImageRef(sha256="a" * 64),
        ImageRef(sha256="b" * 64, thumb_path="/tmp/thumbs/b.jpg", ephemeral=True),
    ]


def _observation(
    session_id: str = "sess-1",
    verbatim: str = "the fog makes it feel private\n  (quiet)",
    attributes: list[str] | None = None,
    polarity: Polarity = Polarity.LIKE,
    confidence: float = 0.87,
    images: list[ImageRef] | None = None,
    created_at: str = "2026-08-04T10:00:00Z",
) -> TasteObservation:
    return TasteObservation(
        session_id=session_id,
        verbatim=verbatim,
        attributes=attributes if attributes is not None else ["negative-space", "muted-palette"],
        polarity=polarity,
        confidence=confidence,
        images=images if images is not None else _image_refs(),
        created_at=created_at,
    )


def _store(data_root):
    return ObservationStore(Catalog(data_root=data_root))


def test_add_and_by_session_preserve_observation_exactly(data_root):
    store = _store(data_root)
    original = _observation()
    row_id = store.add(original)

    [rebuilt] = store.by_session("sess-1")
    assert rebuilt.id == row_id
    assert replace(rebuilt, id=None) == original
    assert rebuilt.verbatim == original.verbatim
    assert rebuilt.verbatim.encode("utf-8") == original.verbatim.encode("utf-8")
    assert rebuilt.attributes == original.attributes
    assert rebuilt.polarity is Polarity.LIKE
    assert rebuilt.confidence == original.confidence
    assert rebuilt.images == original.images
    assert rebuilt.created_at == "2026-08-04T10:00:00Z"


def test_add_without_created_at_stamps_db_default(data_root):
    store = _store(data_root)
    original = _observation(created_at="")
    store.add(original)

    [rebuilt] = store.by_session("sess-1")
    assert rebuilt.created_at
    assert replace(rebuilt, id=None, created_at="") == original


def test_append_only_same_observation_twice_creates_two_rows(data_root):
    store = _store(data_root)
    obs = _observation()
    first = store.add(obs)
    second = store.add(obs)

    assert first != second
    rows = store.by_session("sess-1")
    assert len(rows) == 2
    assert [r.id for r in rows] == [first, second]
    assert store.count() == 2
    assert len(store.all()) == 2
    assert all(replace(r, id=None) == obs for r in rows)


def test_by_session_groups_across_sessions(data_root):
    store = _store(data_root)
    s1 = _observation(session_id="sess-1", verbatim="first session")
    s2 = _observation(
        session_id="sess-2",
        verbatim="second session",
        polarity=Polarity.DISLIKE,
        confidence=0.2,
    )
    s1b = _observation(session_id="sess-1", verbatim="first session again")
    for obs in (s1, s2, s1b):
        store.add(obs)

    first = store.by_session("sess-1")
    second = store.by_session("sess-2")
    assert [replace(r, id=None) for r in first] == [s1, s1b]
    assert [replace(r, id=None) for r in second] == [s2]
    assert store.count() == 3
    assert len(store.all()) == 3


def test_all_returns_chronological_order(data_root):
    store = _store(data_root)
    a = _observation(session_id="s1", verbatim="one")
    b = _observation(session_id="s2", verbatim="two")
    c = _observation(session_id="s1", verbatim="three")
    for obs in (a, b, c):
        store.add(obs)

    verbs = [r.verbatim for r in store.all()]
    assert verbs == ["one", "two", "three"]


def test_v14_migration_tables_and_index(data_root):
    conn = _db.connect(data_root)
    _db.migrate(conn)
    try:
        assert SCHEMA_VERSION >= 14
        tables = _db.table_names(conn)
        assert "taste_observations" in tables
        assert "taste_sessions" in tables
        assert "taste_observations" in EXPECTED_TABLES
        assert "taste_sessions" in EXPECTED_TABLES
        indexes = {
            row[1] for row in conn.execute("PRAGMA index_list(taste_observations)")
        }
        assert "idx_taste_observations_session" in indexes
        obs_cols = {row[1] for row in conn.execute("PRAGMA table_info(taste_observations)")}
        assert {"id", "session_id", "verbatim", "attributes_json", "polarity",
                "confidence", "images_json", "created_at"} <= obs_cols
        sess_cols = {row[1] for row in conn.execute("PRAGMA table_info(taste_sessions)")}
        assert {"id", "kind", "images_json", "created_at", "closed_at"} <= sess_cols
    finally:
        conn.close()


def test_store_uses_json_columns_round_trip_via_from_dict(data_root):
    store = _store(data_root)
    original = _observation()
    row_id = store.add(original)

    [rebuilt] = store.by_session("sess-1")
    text = json.dumps(rebuilt.to_dict())
    assert isinstance(text, str)
    assert TasteObservation.from_dict(json.loads(text)) == rebuilt
    assert rebuilt.id == row_id
    assert rebuilt.images[1].ephemeral is True
    assert rebuilt.images[1].thumb_path == "/tmp/thumbs/b.jpg"


def test_taste_session_round_trip():
    session = TasteSession(
        id="sess-abc",
        kind="reaction-room",
        images=_image_refs(),
        started_at="2026-08-04T10:00:00Z",
        closed_at=None,
    )
    rebuilt = TasteSession.from_dict(json.loads(json.dumps(session.to_dict())))
    assert rebuilt == session
    assert rebuilt.id == "sess-abc"
    assert rebuilt.kind == "reaction-room"
    assert rebuilt.images == session.images
    assert rebuilt.closed_at is None
    assert TasteSession.from_dict(rebuilt) == session


def test_session_store_new_session_and_close(data_root):
    store = SessionStore(Catalog(data_root=data_root))
    session = store.new_session("cli")
    assert isinstance(session.id, str) and session.id
    assert session.kind == "cli"
    assert session.images == []
    assert session.started_at
    assert session.closed_at is None

    closed = store.close(session)
    assert closed.id == session.id
    assert closed.kind == session.kind
    assert closed.started_at == session.started_at
    assert closed.closed_at is not None

    fetched = store.get(session.id)
    assert fetched == closed
    assert fetched.closed_at is not None
    assert store.get("does-not-exist") is None


def test_session_store_round_trip_observations_together(data_root):
    catalog = Catalog(data_root=data_root)
    sessions = SessionStore(catalog)
    observations = ObservationStore(catalog)
    session = sessions.new_session("reaction-room")

    obs = create_observation(
        session_id=session.id,
        verbatim="i like the soft light",
        attributes=["soft-light", "quiet"],
        polarity=Polarity.LIKE,
        confidence=0.9,
        images=[ImageRef(sha256="c" * 64)],
    )
    observations.add(obs)

    [rebuilt] = observations.by_session(session.id)
    assert replace(rebuilt, id=None, created_at="") == obs
    assert rebuilt.session_id == session.id
