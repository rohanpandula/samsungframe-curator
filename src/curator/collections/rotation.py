"""Collections: playlists + deterministic rotation engine (M005/S04).

A :class:`Playlist` is an ordered list of catalog entry ids with rotation
preferences: an optional rotation interval, optional seeded shuffle, an optional
"favorites first" ordering, and an optional :class:`ScheduleWindow` that limits
when rotation is allowed to proceed.

:class:`RotationEngine` decides which entry to show next for a playlist. It is a
pure, stateless function of ``(playlist, now, prev_state, seed)`` — the same
inputs always produce the same :class:`RotationStep`, with no RNG surprises. All
shuffling is derived from a fixed seed (an explicit ``seed`` argument, the
persisted ``RotationState.seed``, or a constant default) plus the playlist's
member order, so re-running with the same state and seed replays identically.

Each decision returns an explainable :class:`RotationStep` carrying a
``reason`` list — ``"interval elapsed"``, ``"shuffle (seeded)"``, ``"favorites
first"``, ``"no immediate repeat: previous shown X"``, ``"schedule window
open"``, ``"show-now override"``, or a no-op explanation when rotation is
blocked by interval/schedule.

State is carried in a JSON-serializable :class:`RotationState` that round-trips
through :class:`RotationStore` (schema v10 ``rotation_state`` table).
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from curator.catalog import Catalog

#: Default shuffle seed when none is supplied and none is persisted.
_DEFAULT_SEED = 0


def _fmt_dt(value: datetime | None) -> str | None:
    """Serialize a datetime as an ISO-8601 string (or ``None``)."""
    return value.isoformat() if value is not None else None


def _parse_dt(value: Any) -> datetime | None:
    """Parse an ISO-8601 string back into a datetime (or ``None``)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


@dataclass(frozen=True)
class ScheduleWindow:
    """Permitted day/time/season window for rotation.

    All fields are optional and AND-combined: when a field is ``None`` it imposes
    no constraint. ``days`` is a set of weekdays (``0``=Monday .. ``6``=Sunday);
    ``start_hour``/``end_hour`` bound a half-open hour window (``start_hour <=
    hour < end_hour``); ``start_month``/``end_month`` bound an inclusive season
    month range. Times are interpreted in the same wall-clock basis as the
    ``now`` datetime passed to the engine (naive/UTC by convention).
    """

    days: frozenset[int] | None = None
    start_hour: float | None = None
    end_hour: float | None = None
    start_month: int | None = None
    end_month: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "days": sorted(self.days) if self.days is not None else None,
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
            "start_month": self.start_month,
            "end_month": self.end_month,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScheduleWindow:
        if isinstance(data, cls):
            return data
        days = data.get("days")
        return cls(
            days=frozenset(int(d) for d in days) if days else None,
            start_hour=_opt_float(data.get("start_hour")),
            end_hour=_opt_float(data.get("end_hour")),
            start_month=_opt_int(data.get("start_month")),
            end_month=_opt_int(data.get("end_month")),
        )

    def is_open(self, now: datetime) -> bool:
        """True when *now* falls within the permitted window."""
        if self.days is not None and now.weekday() not in self.days:
            return False
        if self.start_hour is not None and self.end_hour is not None:
            hour = now.hour + now.minute / 60.0 + now.second / 3600.0
            if not (self.start_hour <= hour < self.end_hour):
                return False
        if self.start_month is not None and self.end_month is not None:
            if not (self.start_month <= now.month <= self.end_month):
                return False
        return True


def _opt_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _opt_float(value: Any) -> float | None:
    return float(value) if value is not None else None


@dataclass(frozen=True)
class Playlist:
    """An ordered collection of catalog entry ids with rotation preferences."""

    id: int
    name: str
    members: list[int] = field(default_factory=list)
    rotation_interval_seconds: int | None = None
    shuffle: bool = False
    favorites_first: bool = False
    schedule: ScheduleWindow | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "members": list(self.members),
            "rotation_interval_seconds": self.rotation_interval_seconds,
            "shuffle": self.shuffle,
            "favorites_first": self.favorites_first,
            "schedule": self.schedule.to_dict() if self.schedule is not None else None,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Playlist:
        if isinstance(data, cls):
            return data
        schedule = data.get("schedule")
        return cls(
            id=int(data["id"]),
            name=str(data["name"]),
            members=[int(m) for m in data.get("members", [])],
            rotation_interval_seconds=_opt_int(
                data.get("rotation_interval_seconds")
            ),
            shuffle=bool(data.get("shuffle", False)),
            favorites_first=bool(data.get("favorites_first", False)),
            schedule=ScheduleWindow.from_dict(schedule) if schedule else None,
            created_at=str(data.get("created_at", "")),
        )


@dataclass(frozen=True)
class RotationState:
    """Persistent, JSON-serializable rotation position for one playlist.

    Tracks the last shown entry (``last_entry_id`` + ``last_played_at``), a fixed
    ``seed`` used for deterministic shuffling, an optional pending show-now
    override, and a rolling ``history`` of played entry ids for explainability.
    """

    playlist_id: int
    last_entry_id: int | None = None
    last_played_at: datetime | None = None
    override_entry_id: int | None = None
    override_active: bool = False
    seed: int | None = None
    history: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "playlist_id": self.playlist_id,
            "last_entry_id": self.last_entry_id,
            "last_played_at": _fmt_dt(self.last_played_at),
            "override_entry_id": self.override_entry_id,
            "override_active": self.override_active,
            "seed": self.seed,
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotationState:
        if isinstance(data, cls):
            return data
        return cls(
            playlist_id=int(data["playlist_id"]),
            last_entry_id=_opt_int(data.get("last_entry_id")),
            last_played_at=_parse_dt(data.get("last_played_at")),
            override_entry_id=_opt_int(data.get("override_entry_id")),
            override_active=bool(data.get("override_active", False)),
            seed=_opt_int(data.get("seed")),
            history=[int(h) for h in data.get("history", [])],
        )


@dataclass(frozen=True)
class RotationStep:
    """One explainable rotation decision for a playlist.

    An advancing step sets ``entry_id``/``played_at``/``position``; a no-op
    (waiting on interval, or schedule window closed) leaves those ``None`` and
    explains why in ``reason``.
    """

    playlist_id: int
    entry_id: int | None
    reason: list[str] = field(default_factory=list)
    played_at: datetime | None = None
    position: int | None = None

    @property
    def waiting(self) -> bool:
        """True when this step advanced nothing (interval/schedule no-op)."""
        return self.entry_id is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "playlist_id": self.playlist_id,
            "entry_id": self.entry_id,
            "reason": list(self.reason),
            "played_at": _fmt_dt(self.played_at),
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RotationStep:
        if isinstance(data, cls):
            return data
        return cls(
            playlist_id=int(data["playlist_id"]),
            entry_id=_opt_int(data.get("entry_id")),
            reason=[str(r) for r in data.get("reason", [])],
            played_at=_parse_dt(data.get("played_at")),
            position=_opt_int(data.get("position")),
        )


class RotationEngine:
    """Deterministic rotation decision-maker (stateless).

    Each call is a pure function of ``(playlist, now, prev_state, seed)``:
    identical inputs yield an identical :class:`RotationStep`, and shuffle order
    is a pure function of the fixed seed, so there are no RNG surprises.
    """

    def next(
        self,
        playlist: Playlist,
        now: datetime,
        prev_state: RotationState | None = None,
        seed: int | None = None,
        favorites: set[int] | None = None,
    ) -> RotationStep:
        """Return the next :class:`RotationStep`, ignoring the new state."""
        return self.advance(playlist, now, prev_state, seed=seed, favorites=favorites)[1]

    def advance(
        self,
        playlist: Playlist,
        now: datetime,
        prev_state: RotationState | None = None,
        seed: int | None = None,
        favorites: set[int] | None = None,
    ) -> tuple[RotationState, RotationStep]:
        """Advance *playlist* at *now*; return ``(new_state, step)``.

        A show-now override pending on *prev_state* is honored first (bypassing
        interval/schedule). Otherwise a closed schedule window or an un-elapsed
        interval yields a no-op step; else the next entry is selected from the
        deterministic rotation order and recorded.
        """
        prev = prev_state or RotationState(playlist_id=playlist.id)
        eff_seed = self._resolve_seed(playlist, prev, seed)

        if prev.override_active and prev.override_entry_id is not None:
            return self._show(playlist, prev.override_entry_id, now, prev)

        if playlist.schedule is not None and not playlist.schedule.is_open(now):
            step = RotationStep(
                playlist_id=playlist.id,
                entry_id=None,
                reason=[
                    "schedule window closed",
                    self._schedule_desc(playlist.schedule),
                ],
            )
            return prev, step

        if (
            playlist.rotation_interval_seconds is not None
            and prev.last_played_at is not None
            and not self._interval_elapsed(playlist, prev, now)
        ):
            elapsed = (now - prev.last_played_at).total_seconds()
            step = RotationStep(
                playlist_id=playlist.id,
                entry_id=None,
                reason=[
                    "rotation interval not yet elapsed: "
                    f"{elapsed:.0f}s < {playlist.rotation_interval_seconds}s"
                ],
            )
            return prev, step

        order = self._order(playlist, eff_seed, favorites)
        if not order:
            step = RotationStep(
                playlist_id=playlist.id,
                entry_id=None,
                reason=["playlist has no members"],
            )
            return prev, step
        entry = self._pick_next(order, prev.last_entry_id)
        return self._record(playlist, entry, now, prev, eff_seed)

    def show_now(
        self,
        playlist: Playlist,
        entry_id: int,
        now: datetime,
        prev_state: RotationState | None = None,
    ) -> tuple[RotationState, RotationStep]:
        """Force *entry_id* to play now; return ``(new_state, step)``.

        Records the override so the next normal tick resumes the deterministic
        sequence from this entry (never repeating it immediately).
        """
        if entry_id not in playlist.members:
            raise ValueError(
                f"entry {entry_id} is not a member of playlist {playlist.id!r}"
            )
        prev = prev_state or RotationState(playlist_id=playlist.id)
        return self._show(playlist, entry_id, now, prev)

    # -- impl ---------------------------------------------------------------

    def _show(
        self,
        playlist: Playlist,
        entry_id: int,
        now: datetime,
        prev: RotationState,
    ) -> tuple[RotationState, RotationStep]:
        """Record *entry_id* as played now, clearing any pending override."""
        state = RotationState(
            playlist_id=playlist.id,
            last_entry_id=entry_id,
            last_played_at=now,
            override_entry_id=None,
            override_active=False,
            seed=prev.seed,
            history=[*prev.history, entry_id],
        )
        step = RotationStep(
            playlist_id=playlist.id,
            entry_id=entry_id,
            reason=["show-now override"],
            played_at=now,
            position=playlist.members.index(entry_id),
        )
        return state, step

    def _record(
        self,
        playlist: Playlist,
        entry: int,
        now: datetime,
        prev: RotationState,
        seed: int | None,
    ) -> tuple[RotationState, RotationStep]:
        """Record a normal advance to *entry* and build an explainable step."""
        reasons = []
        if prev.last_played_at is not None:
            reasons.append("rotation interval elapsed")
        else:
            reasons.append("first play")
        if playlist.shuffle:
            reasons.append(f"shuffle (seeded: {seed})")
        if playlist.schedule is not None:
            reasons.append("schedule window open")
        if playlist.favorites_first and (
            playlist.shuffle or playlist.members
        ):
            reasons.append("favorites first")
        state = RotationState(
            playlist_id=playlist.id,
            last_entry_id=entry,
            last_played_at=now,
            override_entry_id=None,
            override_active=False,
            seed=seed,
            history=[*prev.history, entry],
        )
        reasons.append(
            f"no immediate repeat: previous shown {prev.last_entry_id}"
            if prev.last_entry_id is not None and entry != prev.last_entry_id
            else "positioned in rotation order"
        )
        step = RotationStep(
            playlist_id=playlist.id,
            entry_id=entry,
            reason=reasons,
            played_at=now,
            position=playlist.members.index(entry),
        )
        return state, step

    def _resolve_seed(
        self, playlist: Playlist, prev: RotationState, seed: int | None
    ) -> int | None:
        if seed is not None:
            return seed
        if prev.seed is not None:
            return prev.seed
        if playlist.shuffle:
            return _DEFAULT_SEED
        return None

    def _order(
        self,
        playlist: Playlist,
        seed: int | None,
        favorites: set[int] | None,
    ) -> list[int]:
        """Return the deterministic rotation order for *playlist*.

        Favorites-first (when enabled) places favorited members ahead, and a
        seeded shuffle randomizes within each group / the whole list, so the
        result is a fixed function of ``(seed, members)``.
        """
        members = list(playlist.members)
        if not playlist.shuffle:
            if playlist.favorites_first and favorites:
                favs = [m for m in members if m in favorites]
                rest = [m for m in members if m not in favorites]
                return favs + rest
            return members
        rng = random.Random(seed)
        if playlist.favorites_first and favorites:
            favs = [m for m in members if m in favorites]
            rest = [m for m in members if m not in favorites]
            rng.shuffle(favs)
            rng.shuffle(rest)
            return favs + rest
        shuffled = list(members)
        rng.shuffle(shuffled)
        return shuffled

    def _pick_next(self, order: list[int], last_entry_id: int | None) -> int:
        """Return the next entry, stepping cyclically after *last_entry_id*.

        For ``n >= 2`` adjacent entries in a permutation are distinct, so an
        entry never immediately follows itself.
        """
        if not order:
            raise ValueError("cannot pick from an empty rotation order")
        if last_entry_id is None:
            return order[0]
        if last_entry_id in order and len(order) > 1:
            index = order.index(last_entry_id)
            return order[(index + 1) % len(order)]
        return order[0]

    @staticmethod
    def _interval_elapsed(
        playlist: Playlist, prev: RotationState, now: datetime
    ) -> bool:
        if playlist.rotation_interval_seconds is None or prev.last_played_at is None:
            return True
        return (
            now - prev.last_played_at
        ).total_seconds() >= playlist.rotation_interval_seconds

    @staticmethod
    def _schedule_desc(schedule: ScheduleWindow) -> str:
        parts = []
        if schedule.days is not None:
            parts.append(f"days={sorted(schedule.days)}")
        if schedule.start_hour is not None and schedule.end_hour is not None:
            parts.append(
                f"hours={schedule.start_hour:g}-{schedule.end_hour:g}"
            )
        if schedule.start_month is not None and schedule.end_month is not None:
            parts.append(
                f"months={schedule.start_month}-{schedule.end_month}"
            )
        return "window(" + ", ".join(parts) + ")" if parts else "window()"


class RotationStore:
    """Persist/load playlists and rotation state (schema v10).

    Accepts either a :class:`~curator.catalog.Catalog` or a raw SQLite
    connection (sharing its migrated DB), mirroring :class:`ApprovalService`.
    Playlists store their full JSON in ``config_json`` (members included) while
    also materializing ``playlist_members`` rows for relational/integrity use;
    rotation state is kept as one JSON row per playlist.
    """

    def __init__(self, db: sqlite3.Connection | Catalog) -> None:
        self.db = db.db if isinstance(db, Catalog) else db

    def save_playlist(self, playlist: Playlist) -> None:
        """Upsert *playlist* and rewrite its membership rows."""
        self.db.execute(
            "INSERT INTO playlists(id, name, config_json)"
            " VALUES (?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            "   name = excluded.name,"
            "   config_json = excluded.config_json",
            (playlist.id, playlist.name, json.dumps(playlist.to_dict())),
        )
        self.db.execute(
            "DELETE FROM playlist_members WHERE playlist_id = ?", (playlist.id,)
        )
        for position, entry_id in enumerate(playlist.members):
            self.db.execute(
                "INSERT INTO playlist_members(playlist_id, catalog_entry_id, position)"
                " VALUES (?, ?, ?)",
                (playlist.id, entry_id, position),
            )
        self.db.commit()

    def load_playlist(self, playlist_id: int) -> Playlist | None:
        """Return the persisted *playlist_id*, or ``None`` when absent."""
        row = self.db.execute(
            "SELECT config_json FROM playlists WHERE id = ?", (playlist_id,)
        ).fetchone()
        if row is None:
            return None
        return Playlist.from_dict(json.loads(row[0]))

    def save_state(self, state: RotationState) -> None:
        """Upsert the single rotation-state row for the playlist."""
        self.db.execute(
            "INSERT INTO rotation_state(playlist_id, state_json, updated_at)"
            " VALUES (?, ?, datetime('now'))"
            " ON CONFLICT(playlist_id) DO UPDATE SET"
            "   state_json = excluded.state_json,"
            "   updated_at = datetime('now')",
            (state.playlist_id, json.dumps(state.to_dict())),
        )
        self.db.commit()

    def load_state(self, playlist_id: int) -> RotationState | None:
        """Return the persisted state for *playlist_id*, or ``None``."""
        row = self.db.execute(
            "SELECT state_json FROM rotation_state WHERE playlist_id = ?",
            (playlist_id,),
        ).fetchone()
        if row is None:
            return None
        return RotationState.from_dict(json.loads(row[0]))

    def members(self, playlist_id: int) -> list[int]:
        """Return the persisted membership ids in order for *playlist_id*."""
        rows = self.db.execute(
            "SELECT catalog_entry_id FROM playlist_members"
            " WHERE playlist_id = ? ORDER BY position",
            (playlist_id,),
        ).fetchall()
        return [int(r[0]) for r in rows]
