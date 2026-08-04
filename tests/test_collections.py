"""Tests for src/curator/collections (M005/S04 playlists + rotation).

Covers the deterministic rotation contract: seeded determinism (same inputs ->
same step), shuffled no-immediate-repeat, the rotation interval gate (waiting/no-op
before it elapses), favorites-first ordering, the schedule window (open vs closed),
show-now override resumption, and JSON + SQLite persistence round-trips.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

from curator import db as _db
from curator.catalog import Catalog
from curator.collections import (
    Playlist,
    RotationEngine,
    RotationState,
    RotationStep,
    RotationStore,
    ScheduleWindow,
)
from curator.schema import EXPECTED_TABLES, SCHEMA_VERSION


@pytest.fixture
def catalog(data_root):
    """A Catalog backed by a freshly migrated DB + ContentStore."""
    return Catalog(data_root=data_root)


@pytest.fixture
def entry_ids(catalog):
    """Add a handful of catalog entries and return their ids."""
    ids = []
    for i in range(6):
        catalog.add_source(
            "conn-local", f"asset-{i}", f"rot-photo-{i}".encode()
        )
        entry = catalog.get_by_source("conn-local", f"asset-{i}")
        assert entry is not None
        ids.append(int(entry["id"]))
    return ids


def advance_many(engine, playlist, states, n=3):
    """Advance *states* in place n times, returning the played entry sequence."""
    result = []
    now = states["now"]
    for _ in range(n):
        state, step = engine.advance(playlist, now, states["state"])
        if not step.waiting:
            result.append(step.entry_id)
        states["state"] = state
        now = now.replace(second=now.second + 1)
    return result


def test_seeded_determinism(entry_ids):
    engine = RotationEngine()
    playlist = Playlist(
        id=1, name="det", members=list(entry_ids), shuffle=True
    )
    now = datetime(2026, 1, 1, 10, 0, 0)
    seed = 42

    state_a, step_a = engine.advance(playlist, now, None, seed=seed)
    state_b, step_b = engine.advance(playlist, now, None, seed=seed)

    assert step_a.entry_id == step_b.entry_id
    assert step_a == step_b
    assert state_a == state_b
    assert state_a.seed == seed
    # The seed is persisted so later calls replay the same shuffle.
    _, step_c = engine.advance(playlist, now, state_a)
    _, step_d = engine.advance(playlist, now, state_b)
    assert step_c.entry_id == step_d.entry_id

    # Explainable, non-empty reason.
    assert step_a.entry_id is not None
    assert step_a.reason
    assert any("shuffle (seeded" in r for r in step_a.reason)


def test_shuffle_no_immediate_repeat(entry_ids):
    engine = RotationEngine()
    playlist = Playlist(
        id=2, name="shuf", members=list(entry_ids), shuffle=True
    )
    state: RotationState | None = None
    now = datetime(2026, 2, 1, 9, 0, 0)
    previous: int | None = None
    seen = 0
    for _ in range(40):
        state, step = engine.advance(playlist, now, state, seed=7)
        if step.waiting:
            now = now.replace(minute=now.minute + 1)
            continue
        if previous is not None:
            assert step.entry_id != previous, "entry followed itself"
        previous = step.entry_id
        seen += 1
        now = now.replace(second=now.second + 1)
    assert seen >= 6


def test_interval_waiting_then_advance(entry_ids):
    engine = RotationEngine()
    playlist = Playlist(
        id=3,
        name="int",
        members=list(entry_ids),
        rotation_interval_seconds=60,
    )
    now = datetime(2026, 3, 1, 8, 0, 0)
    first = engine.next(playlist, now)
    assert first.waiting is False
    assert "first play" in first.reason
    state = RotationState(
        playlist_id=3,
        last_entry_id=first.entry_id,
        last_played_at=now,
        history=[first.entry_id] if first.entry_id is not None else [],
    )

    # Before the interval elapses -> waiting no-op with an explainable reason.
    waiting = engine.next(playlist, now.replace(minute=0, second=10), state)
    assert waiting.waiting is True
    assert waiting.entry_id is None
    assert any("not yet elapsed" in r for r in waiting.reason)

    # After the interval -> advances again.
    advanced = engine.next(
        playlist, now.replace(minute=1, second=10), state
    )
    assert advanced.waiting is False
    assert advanced.entry_id is not None
    assert "rotation interval elapsed" in advanced.reason


def test_favorites_first(entry_ids):
    engine = RotationEngine()
    favs = set(entry_ids[2:4])
    playlist = Playlist(
        id=4,
        name="fav",
        members=list(entry_ids),
        shuffle=False,
        favorites_first=True,
    )
    state: RotationState | None = None
    now = datetime(2026, 4, 1, 12, 0, 0)
    order = []
    for _ in range(len(entry_ids)):
        state, step = engine.advance(playlist, now, state, favorites=favs)
        order.append(step.entry_id)
        now = now.replace(second=now.second + 1)

    assert order[0] in favs and order[1] in favs
    # The two favorites appear before the first non-favorite.
    fav_positions = [i for i, e in enumerate(order) if e in favs]
    nonfav_positions = [i for i, e in enumerate(order) if e not in favs]
    assert max(fav_positions) < min(nonfav_positions)


def test_schedule_window(entry_ids):
    engine = RotationEngine()
    playlist = Playlist(
        id=5,
        name="sched",
        members=list(entry_ids),
        schedule=ScheduleWindow(start_hour=9.0, end_hour=17.0),
    )
    open_t = datetime(2026, 5, 1, 11, 0, 0)
    closed_t = datetime(2026, 5, 1, 20, 0, 0)

    inside = engine.next(playlist, open_t)
    assert inside.waiting is False
    assert inside.entry_id == entry_ids[0]

    outside = engine.next(playlist, closed_t)
    assert outside.waiting is True
    assert outside.entry_id is None
    assert any("schedule window closed" in r for r in outside.reason)


def test_show_now_resumes(entry_ids):
    engine = RotationEngine()
    playlist = Playlist(
        id=6, name="show", members=list(entry_ids), shuffle=False
    )
    now = datetime(2026, 6, 1, 10, 0, 0)
    state: RotationState | None = None

    state, _ = engine.advance(playlist, now)
    state, _ = engine.advance(playlist, now.replace(second=1), state)
    target = entry_ids[4]
    state, override = engine.show_now(playlist, target, now.replace(second=2), state)
    assert override.entry_id == target
    assert "show-now override" in override.reason
    assert override.position == entry_ids.index(target)

    # Next normal tick resumes the deterministic sequence without repeating.
    resumed = engine.next(playlist, now.replace(second=3), state)
    assert resumed.waiting is False
    assert resumed.entry_id is not None
    assert resumed.entry_id != target, "override repeated immediately"


def test_schema_v10_tables_and_version(catalog):
    tables = _db.table_names(catalog.db)
    for table in ("playlists", "playlist_members", "rotation_state"):
        assert table in tables
    assert SCHEMA_VERSION == 11
    for table in ("playlists", "playlist_members", "rotation_state"):
        assert table in EXPECTED_TABLES
    for table in ("immich_sync_state", "immich_asset_state"):
        assert table in EXPECTED_TABLES
        assert table in tables


def test_dataclass_round_trip(entry_ids):
    playlist = Playlist(
        id=7,
        name="rt",
        members=list(entry_ids),
        rotation_interval_seconds=120,
        shuffle=True,
        favorites_first=True,
        schedule=ScheduleWindow(
            days=frozenset({0, 1, 2}),
            start_hour=9.0,
            end_hour=18.0,
            start_month=3,
            end_month=5,
        ),
        created_at="2026-01-01T00:00:00.000Z",
    )
    rebuilt = Playlist.from_dict(json.loads(json.dumps(playlist.to_dict())))
    assert rebuilt == playlist

    state = RotationState(
        playlist_id=7,
        last_entry_id=entry_ids[1],
        last_played_at=datetime(2026, 7, 1, 8, 30, 15),
        override_entry_id=entry_ids[2],
        override_active=True,
        seed=9,
        history=list(entry_ids[:3]),
    )
    rebuilt_state = RotationState.from_dict(
        json.loads(json.dumps(state.to_dict()))
    )
    assert rebuilt_state == state

    step = RotationStep(
        playlist_id=7,
        entry_id=entry_ids[0],
        reason=["rotation interval elapsed", "favorites first"],
        played_at=datetime(2026, 7, 1, 8, 30, 15),
        position=0,
    )
    assert RotationStep.from_dict(json.loads(json.dumps(step.to_dict()))) == step


def test_state_round_trips_through_db(catalog, entry_ids):
    store = RotationStore(catalog)
    playlist = Playlist(id=8, name="db", members=list(entry_ids), shuffle=True)
    store.save_playlist(playlist)

    loaded = store.load_playlist(8)
    assert loaded is not None
    assert loaded == playlist
    assert store.members(8) == list(entry_ids)

    state = RotationState(
        playlist_id=8,
        last_entry_id=entry_ids[0],
        last_played_at=datetime(2026, 8, 1, 9, 15, 0),
        seed=3,
        history=[entry_ids[0]],
    )
    store.save_state(state)
    assert store.load_state(8) == state

    mark = store.db.execute(
        "SELECT playlist_id FROM rotation_state WHERE playlist_id = 8"
    ).fetchone()
    assert mark == (8,)
