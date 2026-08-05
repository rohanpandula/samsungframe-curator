"""Reaction Room flow for the taste dialogue subsystem (M008/S03).

A :class:`ReactionRoom` is the short conversational surface where a user drops
images and reacts to them in plain language. Dropped third-party images are
retained as ephemeral evidence (thumbnail + content hash) under the data root;
cataloged images are mapped by their content SHA-256 and never re-retained. Each
reaction is extracted into a
:class:`~curator.taste.dialogue.observation.TasteObservation` and followed by at
most one short :class:`ProbeQuestion`, hard-capped at two per session — a gym rep,
never an interview. When no extraction provider is enabled the room is unavailable
(it never silently degrades to keyword matching).

:class:`ProbeGenerator` turns the extracted attributes of one reaction into at
most two deterministic, plain probing questions targeting attribute gaps; with no
attributes it yields a single generic opening question (used by the CLI's
no-``--note`` preview).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from curator.catalog import Catalog
from curator.errors import CuratorError
from curator.hashing import sha256_hex
from curator.taste.dialogue.extraction import (
    ExtractionProvider,
    extract_or_unavailable,
)
from curator.taste.dialogue.observation import (
    ImageRef,
    TasteObservation,
    create_observation,
)
from curator.taste.dialogue.retention import retain_ephemeral
from curator.taste.dialogue.session import TasteSession
from curator.taste.dialogue.store import ObservationStore, SessionStore

#: Hard cap on follow-up questions per session — the room never asks more.
MAX_FOLLOWUPS = 2


@dataclass(frozen=True)
class ProbeQuestion:
    """One short probing question asked during a reaction-room turn.

    ``target_attribute`` names the controlled-vocabulary attribute the question
    is trying to disambiguate (``None`` for a generic opening question).
    """

    text: str
    target_attribute: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "target_attribute": self.target_attribute}

    @classmethod
    def from_dict(cls, data: Any) -> ProbeQuestion:
        if isinstance(data, ProbeQuestion):
            return data
        fields = dict(data)
        return cls(
            text=str(fields["text"]),
            target_attribute=fields.get("target_attribute"),
        )


class ReactionRoomUnavailableError(CuratorError):
    """Raised when the Reaction Room cannot run (no enabled extraction provider)."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class RoomTurn:
    """The outcome of one reaction: the recorded observation plus a follow-up."""

    observation: TasteObservation
    question: ProbeQuestion | None
    followups_asked: int


#: Ordered ``(cue, gap, text)`` candidate probes; iteration order fixes the
#: deterministic question ordering. A candidate fires only when the cue attribute
#: is present and the gap attribute is absent (an informative gap).
_PROBES: tuple[tuple[str, str, str], ...] = (
    ("negative-space", "symmetry", "Is it the emptiness on the left, or the symmetry?"),
    ("negative-space", "lone-subject", "Is it the empty space, or the lone subject within it?"),
    ("quiet", "texture", "Is it the quiet stillness, or the texture?"),
    ("muted-palette", "warm-tones", "Is it the muted palette, or the warm tones?"),
    ("lone-subject", "breathing-room", "Is it the lone subject, or the breathing room around it?"),
    ("symmetry", "geometric", "Is it the symmetry, or the geometry?"),
    ("minimal", "dense", "Is it the minimalism, or the busyness?"),
    ("warm-tones", "high-contrast", "Is it the warm tones, or the contrast?"),
    ("texture", "organic", "Is it the texture, or the organic feel?"),
    ("breathing-room", "negative-space", "Is it the breathing room, or the emptiness?"),
    ("geometric", "repetition", "Is it the geometry, or the repetition?"),
    ("motion", "repetition", "Is it the motion, or the repetition?"),
    ("heavy", "light", "Is it the weight, or the lightness?"),
    ("nostalgic", "muted-palette", "Is it the nostalgia, or the muted palette?"),
    ("dense", "motion", "Is it the busyness, or the motion?"),
)

_OPENING_PROBE = ProbeQuestion(text="What draws you to this image?", target_attribute=None)


class ProbeGenerator:
    """Deterministic generator of short probing questions for a reaction.

    Given the extracted attributes of one observation, yields at most
    ``max_questions`` plain gap-targeted questions in a fixed order. Attributes
    that already cover a gap suppress that candidate, so a rich reaction produces
    fewer questions; an observation with no attributes yields a single generic
    opening question. Never lectures and never compliments.
    """

    def __init__(self) -> None:
        self.opening_question = _OPENING_PROBE

    def questions_for(
        self, observation: TasteObservation, max_questions: int = 2
    ) -> list[ProbeQuestion]:
        """Return deterministic probing questions for *observation* (≤ *max_questions*)."""
        attributes = set(observation.attributes)
        if not attributes:
            return [self.opening_question]
        questions: list[ProbeQuestion] = []
        seen_gaps: set[str] = set()
        for cue, gap, text in _PROBES:
            if cue not in attributes or gap in attributes or gap in seen_gaps:
                continue
            seen_gaps.add(gap)
            questions.append(ProbeQuestion(text=text, target_attribute=gap))
            if len(questions) >= max_questions:
                break
        return questions


class ReactionRoom:
    """The reaction-room surface: start a session, react, finish.

    Takes an :class:`ObservationStore` (or a :class:`Catalog`, reusing its shared
    ``.db``) or a raw ``sqlite3.Connection``, an optional
    :class:`ExtractionProvider`, and a data root for ephemeral retention. The room
    is deterministic: identical drops and reactions produce identical
    observations and questions.
    """

    def __init__(
        self,
        store: ObservationStore | Catalog | sqlite3.Connection,
        provider: ExtractionProvider | None,
        data_root: str | Path,
    ) -> None:
        if isinstance(store, ObservationStore):
            self.db = store.db
            self.observation_store = store
        else:
            self.db = store.db if isinstance(store, Catalog) else store
            self.observation_store = ObservationStore(self.db)
        self.session_store = SessionStore(self.db)
        self.provider = provider
        self.data_root = Path(data_root)
        self.generator = ProbeGenerator()
        self._session_images: dict[str, list[ImageRef]] = {}
        self._followups_asked: dict[str, int] = {}

    def start(
        self, images: list[str | Path], kind: str = "reaction-room"
    ) -> TasteSession:
        """Retain/map *images*, create a session of *kind*, and return it.

        Raises :class:`ReactionRoomUnavailableError` when no extraction provider is
        enabled — the room never degrades to keyword matching.
        """
        self._check_available()
        refs = [self._resolve_image(item) for item in images]
        session = self.session_store.new_session(kind)
        self._session_images[session.id] = refs
        return session

    def react(self, session: TasteSession, verbatim: str) -> RoomTurn:
        """Extract and record one reaction, returning the turn (observation + probe).

        The verbatim is recorded byte-exact along with the extracted attributes,
        polarity, confidence, and the session's images. A follow-up question is
        appended only when fewer than :data:`MAX_FOLLOWUPS` have been asked this
        session. Raises :class:`ReactionRoomUnavailableError` when extraction is
        unavailable — keyword-derived attributes are never produced.
        """
        self._check_available()
        images = self._session_images.get(session.id, [])
        probe = create_observation(session_id=session.id, verbatim=verbatim, images=images)
        result = extract_or_unavailable(self.provider, probe)
        if result is None:
            raise ReactionRoomUnavailableError(
                "attribute extraction unavailable (disabled or down)"
            )
        observation = create_observation(
            session_id=session.id,
            verbatim=result.verbatim,
            attributes=result.attributes,
            polarity=result.polarity,
            confidence=result.confidence,
            images=images,
        )
        self.observation_store.add(observation)
        asked = self._followups_asked.get(session.id, 0)
        question: ProbeQuestion | None = None
        if asked < MAX_FOLLOWUPS:
            candidates = self.generator.questions_for(observation, max_questions=2)
            if candidates:
                question = candidates[0]
                asked += 1
                self._followups_asked[session.id] = asked
        return RoomTurn(
            observation=observation, question=question, followups_asked=asked
        )

    def session_images(self, session: TasteSession) -> list[ImageRef]:
        """Return the :class:`ImageRef`s *session* was opened over.

        Callers need these to act on a drop after the fact — notably to promote
        an ephemeral image into the catalog with an explicit save.
        """
        return list(self._session_images.get(session.id, []))

    def finish(self, session: TasteSession) -> int:
        """Close *session* and return its observation count."""
        self.session_store.close(session)
        return len(self.observation_store.by_session(session.id))

    def _check_available(self) -> None:
        """Raise :class:`ReactionRoomUnavailableError` when extraction is off."""
        if self.provider is None or not self.provider.capabilities().enabled:
            raise ReactionRoomUnavailableError(
                "no extraction provider enabled — Reaction Room requires one"
            )

    def _resolve_image(self, item: str | Path) -> ImageRef:
        """Map *item* to an :class:`ImageRef`: ephemeral file or cataloged sha.

        An existing file whose bytes are already cataloged (and a bare catalog
        sha) maps to a non-ephemeral catalog reference; any other existing file is
        retained as an ephemeral thumbnail + content hash.
        """
        path = Path(item)
        if path.is_file():
            data = path.read_bytes()
            sha = sha256_hex(data)
            if self._catalog_has(sha):
                return ImageRef(sha256=sha, ephemeral=False, catalog_saved=True)
            return retain_ephemeral(data, self.data_root)
        sha = str(item)
        if not self._catalog_has(sha):
            raise CuratorError(
                f"image reference is neither a file nor a cataloged sha: {item!r}"
            )
        return ImageRef(sha256=sha, ephemeral=False, catalog_saved=True)

    def _catalog_has(self, sha: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM catalog_entries WHERE sha256 = ? LIMIT 1", (sha,)
        ).fetchone()
        return row is not None
