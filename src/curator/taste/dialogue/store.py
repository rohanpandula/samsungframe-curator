"""Observation + session persistence for the taste dialogue subsystem (M008/S01/T2).

:class:`ObservationStore` is the append-only journal for
:class:`~curator.taste.dialogue.observation.TasteObservation` rows
(``taste_observations``, schema v14): every :meth:`add` inserts a new row and
nothing is ever updated or deleted, so history is preserved and "current" is
simply the chronological replay. :class:`SessionStore` manages the enclosing
:class:`~curator.taste.dialogue.session.TasteSession` rows (``taste_sessions``),
creating sessions and stamping ``closed_at`` on close.

Both accept either a :class:`~curator.catalog.Catalog` (reusing its shared
``.db`` connection) or a raw ``sqlite3.Connection``, mirroring the
approval/jobs store posture. The schema is expected to already be migrated (a
Catalog migrates it on construction).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from typing import Any

from curator.catalog import Catalog
from curator.taste.dialogue.observation import ImageRef, TasteObservation
from curator.taste.dialogue.session import TasteSession

# Column order of ``SELECT`` on taste_observations (schema v14 DDL).
_OBSERVATION_COLUMNS = [
    "session_id",
    "verbatim",
    "attributes_json",
    "polarity",
    "confidence",
    "images_json",
    "id",
    "created_at",
]


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp (matches the schema *_at convention)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%fZ")


def _images_to_json(images: list[ImageRef]) -> str:
    return json.dumps([img.to_dict() for img in images])


class ObservationStore:
    """Append-only persistence for :class:`TasteObservation` rows.

    Takes either a :class:`~curator.catalog.Catalog` or a raw
    ``sqlite3.Connection``; when a Catalog is passed its shared ``.db`` is used.
    """

    def __init__(self, db: sqlite3.Connection | Catalog) -> None:
        self.db = db.db if isinstance(db, Catalog) else db

    def add(self, observation: TasteObservation) -> int:
        """INSERT one observation row (append-only) and return its new row id.

        ``verbatim``/``attributes``/``polarity``/``confidence``/``images`` are
        stored verbatim; ``created_at`` is preserved when the observation
        carries one, otherwise the DB's wall-clock default fills it. The same
        observation added twice produces two rows — never a dedup or update.
        """
        if observation.created_at:
            cur = self.db.execute(
                "INSERT INTO taste_observations"
                " (session_id, verbatim, attributes_json, polarity, confidence,"
                "  images_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    observation.session_id,
                    observation.verbatim,
                    json.dumps(observation.attributes),
                    observation.polarity.value,
                    observation.confidence,
                    _images_to_json(observation.images),
                    observation.created_at,
                ),
            )
        else:
            cur = self.db.execute(
                "INSERT INTO taste_observations"
                " (session_id, verbatim, attributes_json, polarity, confidence,"
                "  images_json)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    observation.session_id,
                    observation.verbatim,
                    json.dumps(observation.attributes),
                    observation.polarity.value,
                    observation.confidence,
                    _images_to_json(observation.images),
                ),
            )
        self.db.commit()
        row_id = cur.lastrowid
        if row_id is None:
            raise RuntimeError("failed to obtain taste_observations row id")
        return int(row_id)

    def by_session(self, session_id: str) -> list[TasteObservation]:
        """Return every observation for *session_id*, oldest first."""
        rows = self.db.execute(
            "SELECT " + ", ".join(_OBSERVATION_COLUMNS)
            + " FROM taste_observations WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def all(self) -> list[TasteObservation]:
        """Return every observation across all sessions, oldest first."""
        rows = self.db.execute(
            "SELECT " + ", ".join(_OBSERVATION_COLUMNS)
            + " FROM taste_observations ORDER BY id"
        ).fetchall()
        return [self._row_to_observation(row) for row in rows]

    def count(self) -> int:
        """Return the total number of persisted observations."""
        row = self.db.execute("SELECT COUNT(*) FROM taste_observations").fetchone()
        return int(row[0])

    def _row_to_observation(self, row: tuple[Any, ...]) -> TasteObservation:
        return TasteObservation.from_dict(
            {
                "session_id": row[0],
                "verbatim": row[1],
                "attributes": json.loads(row[2]),
                "polarity": row[3],
                "confidence": row[4],
                "images": json.loads(row[5]),
                "id": row[6],
                "created_at": row[7],
            }
        )


class SessionStore:
    """Create and close :class:`TasteSession` rows (``taste_sessions``)."""

    def __init__(self, db: sqlite3.Connection | Catalog) -> None:
        self.db = db.db if isinstance(db, Catalog) else db

    def new_session(self, kind: str) -> TasteSession:
        """Create and persist a fresh open session of *kind*."""
        session = TasteSession(
            id=uuid.uuid4().hex,
            kind=kind,
            images=[],
            started_at=_utc_now(),
        )
        self.db.execute(
            "INSERT INTO taste_sessions(id, kind, images_json, created_at, closed_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                session.id,
                session.kind,
                _images_to_json(session.images),
                session.started_at,
                None,
            ),
        )
        self.db.commit()
        return session

    def close(self, session: TasteSession) -> TasteSession:
        """Stamp *session*'s ``closed_at`` and return the closed session."""
        closed_at = _utc_now()
        self.db.execute(
            "UPDATE taste_sessions SET closed_at = ? WHERE id = ?",
            (closed_at, session.id),
        )
        self.db.commit()
        return TasteSession(
            id=session.id,
            kind=session.kind,
            images=session.images,
            started_at=session.started_at,
            closed_at=closed_at,
        )

    def get(self, session_id: str) -> TasteSession | None:
        """Return the session with *session_id*, or ``None`` when absent."""
        row = self.db.execute(
            "SELECT id, kind, images_json, created_at, closed_at"
            " FROM taste_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return TasteSession(
            id=row[0],
            kind=row[1],
            images=[ImageRef.from_dict(img) for img in json.loads(row[2] or "[]")],
            started_at=row[3],
            closed_at=row[4],
        )
