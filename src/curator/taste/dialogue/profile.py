"""Taste Profile model, builder, and store for the dialogue subsystem (M008/S04/T1+T2).

:class:`ProfileBuilder` turns a set of
:class:`~curator.taste.dialogue.observation.TasteObservation`s into an immutable
:class:`TasteProfile`: an aggregated word :attr:`TasteProfile.vocabulary` mapped
onto the S02 :data:`~curator.taste.dialogue.extraction.CONTROLLED_VOCABULARY`,
recurring ``patterns`` (one :class:`TasteClaim` per shared attribute, each with
traceable :class:`EvidenceRef` evidence), surfaced ``tensions`` (contradictions
are surfaced, never smoothed), and an ``evolution`` log summarizing shifts
against the previous profile. :class:`ProfileStore` is the append-only timeline
for claim pin/edit/dispute actions plus the persisted profile document.
:class:`WhatILearned.delta_after` reports the no-silent-learning delta for a
session.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, replace
from typing import Any

from curator.analysis.schema import AnalysisResult
from curator.catalog import Catalog
from curator.taste.dialogue.observation import Polarity, TasteObservation
from curator.taste.dialogue.store import ObservationStore, _utc_now
from curator.taste.profiles import SIGNAL_NAMES
from curator.taste.rank import TasteRanker

#: Minimum observations a verbatim word must appear in to enter the vocabulary.
_MIN_WORD_USES = 1

#: Minimum observations an attribute must appear in to become a pattern claim
#: ("recurring" attributes only).
_MIN_ATTRIBUTE_USES = 2

#: Confidence carried by history-derived evidence — deliberately low: history is
#: inferred, never spoken.
_COLD_START_CONFIDENCE = 0.3

#: Minimum liked history entries (with analysis) before a lean becomes a claim.
_MIN_HISTORY_SAMPLES = 2

#: Minimum separation between the liked mean and its contrast set for a claim.
_MIN_SIGNAL_LEAN = 0.1

#: Neutral midpoint used as the contrast baseline when nothing was disliked.
_NEUTRAL_SIGNAL = 0.5

#: Stand-in for evidence with no user words behind it (history, not dialogue).
_NO_VERBATIM = "(no verbatim — inferred from history)"

#: History sources, in the order their claims are emitted.
_HISTORY_SOURCES: tuple[str, ...] = ("approval", "pairwise")

#: Signals that are aliases of another signal and so would produce a duplicate
#: claim. ``vibrancy`` is defined as the color-story colorfulness value (see
#: :meth:`~curator.taste.rank.TasteRanker.signal_values`), so a cold start that
#: scored both emitted two word-for-word identical claims per source.
_ALIASED_SIGNALS: frozenset[str] = frozenset({"vibrancy"})

_HISTORY_LABELS: dict[str, str] = {
    "approval": "approval history",
    "pairwise": "pairwise voting history",
}

#: Semantically opposed attribute pairs; when both are liked a tension is surfaced.
_CONTRAST_PAIRS: tuple[tuple[str, str], ...] = (
    ("minimal", "dense"),
    ("light", "heavy"),
    ("negative-space", "dense"),
)

#: Ordered word -> attribute-tag map used to aggregate verbatim words onto the
#: S02 controlled vocabulary. The first tag is the word's canonical attribute.
_WORD_ATTRIBUTES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("quiet", ("negative-space", "muted-palette")),
    ("empty", ("negative-space", "breathing-room")),
    ("breathing", ("breathing-room",)),
    ("busy", ("dense",)),
    ("crowded", ("dense",)),
    ("chaotic", ("dense",)),
    ("warm", ("warm-tones",)),
    ("minimal", ("minimal",)),
    ("minimalist", ("minimal",)),
    ("symmetry", ("symmetry",)),
    ("symmetric", ("symmetry",)),
    ("lone", ("lone-subject",)),
    ("contrast", ("high-contrast",)),
    ("texture", ("texture",)),
    ("textured", ("texture",)),
    ("motion", ("motion",)),
    ("movement", ("motion",)),
    ("repetition", ("repetition",)),
    ("repeating", ("repetition",)),
    ("geometric", ("geometric",)),
    ("organic", ("organic",)),
    ("nostalgic", ("nostalgic",)),
    ("heavy", ("heavy",)),
    ("light", ("light",)),
)

_WORD_TO_ATTRIBUTES: dict[str, tuple[str, ...]] = dict(_WORD_ATTRIBUTES)


@dataclass(frozen=True)
class EvidenceRef:
    """One traceable piece of evidence for a :class:`TasteClaim`.

    Pairs a referenced image (``image_sha``) with the exact verbatim that drew
    it in, plus the observation's confidence and timestamp — every claim can be
    opened back to its images and words.
    """

    image_sha: str
    verbatim: str
    confidence: float
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_sha": self.image_sha,
            "verbatim": self.verbatim,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> EvidenceRef:
        if isinstance(data, EvidenceRef):
            return data
        fields = dict(data)
        return cls(
            image_sha=str(fields["image_sha"]),
            verbatim=str(fields["verbatim"]),
            confidence=float(fields.get("confidence", 0.5)),
            created_at=str(fields.get("created_at", "")),
        )


@dataclass(frozen=True)
class TasteClaim:
    """One statement about the user's taste, backed by :class:`EvidenceRef`s.

    ``status`` is one of ``"active"|"pinned"|"edited"|"disputed"``; ``provenance``
    is ``"low"|"high"`` (Reaction Room observations are high-provenance).
    """

    id: str
    text: str
    evidence: list[EvidenceRef]
    status: str = "active"
    provenance: str = "high"
    created_at: str = ""

    def __post_init__(self) -> None:
        if self.status not in ("active", "pinned", "edited", "disputed"):
            raise ValueError(f"invalid claim status {self.status!r}")
        if self.provenance not in ("low", "high"):
            raise ValueError(f"invalid claim provenance {self.provenance!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "evidence": [ref.to_dict() for ref in self.evidence],
            "status": self.status,
            "provenance": self.provenance,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TasteClaim:
        if isinstance(data, TasteClaim):
            return data
        fields = dict(data)
        return cls(
            id=str(fields["id"]),
            text=str(fields["text"]),
            evidence=[EvidenceRef.from_dict(ref) for ref in fields.get("evidence", ())],
            status=str(fields.get("status", "active")),
            provenance=str(fields.get("provenance", "high")),
            created_at=str(fields.get("created_at", "")),
        )


@dataclass(frozen=True)
class ProfileEvent:
    """One append-only entry on the profile timeline (pin/edit/dispute)."""

    claim_id: str
    kind: str
    detail: str
    created_at: str

    def __post_init__(self) -> None:
        if self.kind not in ("pin", "edit", "dispute"):
            raise ValueError(f"invalid event kind {self.kind!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind,
            "detail": self.detail,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ProfileEvent:
        if isinstance(data, ProfileEvent):
            return data
        fields = dict(data)
        return cls(
            claim_id=str(fields["claim_id"]),
            kind=str(fields["kind"]),
            detail=str(fields.get("detail", "")),
            created_at=str(fields.get("created_at", "")),
        )


@dataclass(frozen=True)
class TasteProfile:
    """An immutable summary of a user's taste derived from dialogue observations.

    ``vocabulary`` maps each recurring verbatim word to
    ``{"attribute": str, "usage_count": int}``; ``patterns`` and ``tensions`` are
    the claim lists (contradictions surfaced, never smoothed); ``evolution``
    logs ``{"at": str, "summary": str}`` shifts against the previous profile.
    """

    vocabulary: dict[str, dict[str, Any]]
    patterns: list[TasteClaim]
    tensions: list[TasteClaim]
    evolution: list[dict[str, str]]
    version: int = 1
    created_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "vocabulary", {
            str(word): {"attribute": str(v["attribute"]), "usage_count": int(v["usage_count"])}
            for word, v in self.vocabulary.items()
        })
        object.__setattr__(self, "patterns", list(self.patterns))
        object.__setattr__(self, "tensions", list(self.tensions))
        object.__setattr__(self, "evolution", [dict(e) for e in self.evolution])

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocabulary": {
                word: {"attribute": v["attribute"], "usage_count": v["usage_count"]}
                for word, v in self.vocabulary.items()
            },
            "patterns": [claim.to_dict() for claim in self.patterns],
            "tensions": [claim.to_dict() for claim in self.tensions],
            "evolution": [dict(e) for e in self.evolution],
            "version": self.version,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Any) -> TasteProfile:
        if isinstance(data, TasteProfile):
            return data
        fields = dict(data)
        return cls(
            vocabulary={
                str(word): {
                    "attribute": str(v.get("attribute", "")),
                    "usage_count": int(v.get("usage_count", 0)),
                }
                for word, v in fields.get("vocabulary", {}).items()
            },
            patterns=[TasteClaim.from_dict(c) for c in fields.get("patterns", ())],
            tensions=[TasteClaim.from_dict(c) for c in fields.get("tensions", ())],
            evolution=[
                {"at": str(e.get("at", "")), "summary": str(e.get("summary", ""))}
                for e in fields.get("evolution", ())
            ],
            version=int(fields.get("version", 1)),
            created_at=str(fields.get("created_at", "")),
        )


class ProfileBuilder:
    """Deterministic builder: observations -> :class:`TasteProfile`.

    Identical observations (with identical ``created_at`` stamps) produce
    identical profiles: all aggregations are sorted and no wall-clock or
    unseeded RNG enters the build.
    """

    def build(
        self,
        observations: list[TasteObservation],
        *,
        previous: TasteProfile | None = None,
    ) -> TasteProfile:
        obs = list(observations)
        vocabulary = self._build_vocabulary(obs)
        patterns = self._build_patterns(obs)
        tensions = self._build_tensions(obs)
        created_at = _profile_timestamp(obs)
        version = previous.version + 1 if previous is not None else 1
        evolution = self._build_evolution(
            vocabulary, patterns, tensions, previous, created_at
        )
        return TasteProfile(
            vocabulary=vocabulary,
            patterns=patterns,
            tensions=tensions,
            evolution=evolution,
            version=version,
            created_at=created_at,
        )

    @staticmethod
    def _build_vocabulary(obs: list[TasteObservation]) -> dict[str, dict[str, Any]]:
        counts: dict[str, int] = {}
        for observation in obs:
            for token in _tokens(observation.verbatim):
                if token in _WORD_TO_ATTRIBUTES:
                    counts[token] = counts.get(token, 0) + 1
        vocabulary: dict[str, dict[str, Any]] = {}
        for word in sorted(counts):
            if counts[word] < _MIN_WORD_USES:
                continue
            vocabulary[word] = {
                "attribute": _WORD_TO_ATTRIBUTES[word][0],
                "usage_count": counts[word],
            }
        return vocabulary

    @staticmethod
    def _build_patterns(obs: list[TasteObservation]) -> list[TasteClaim]:
        by_attr: dict[str, list[TasteObservation]] = {}
        for observation in obs:
            if not observation.images:
                continue
            for attr in observation.attributes:
                by_attr.setdefault(attr, []).append(observation)
        claims: list[TasteClaim] = []
        for attr in sorted(by_attr):
            group = by_attr[attr]
            if len(group) < _MIN_ATTRIBUTE_USES:
                continue
            words = _supporting_words(group, attr)
            text = (
                f"you favor {attr} — {', '.join(words)} ({len(group)} uses)"
                if words
                else f"you favor {attr} ({len(group)} uses)"
            )
            claims.append(
                TasteClaim(
                    id=f"pattern:{attr}",
                    text=text,
                    evidence=_evidence_for(group),
                    status="active",
                    provenance="high",
                    created_at=_profile_timestamp(group),
                )
            )
        return claims

    @staticmethod
    def _build_tensions(obs: list[TasteObservation]) -> list[TasteClaim]:
        by_attr: dict[str, dict[Polarity, list[TasteObservation]]] = {}
        for observation in obs:
            if not observation.images:
                continue
            for attr in observation.attributes:
                by_attr.setdefault(attr, {}).setdefault(
                    observation.polarity, []
                ).append(observation)
        tensions: list[TasteClaim] = []
        for attr in sorted(by_attr):
            likes = by_attr[attr].get(Polarity.LIKE, [])
            dislikes = by_attr[attr].get(Polarity.DISLIKE, [])
            if not likes or not dislikes:
                continue
            group = likes + dislikes
            evidence = _evidence_for(group)
            if not evidence:
                continue
            text = (
                f"you describe {attr} as both liked ({len(likes)} uses) and "
                f"disliked ({len(dislikes)} uses) — surfaced, not smoothed."
            )
            tensions.append(
                TasteClaim(
                    id=f"tension:{attr}:polarity",
                    text=text,
                    evidence=evidence,
                    status="active",
                    provenance="high",
                    created_at=_profile_timestamp(group),
                )
            )
        for a, b in _CONTRAST_PAIRS:
            likes_a = by_attr.get(a, {}).get(Polarity.LIKE, [])
            likes_b = by_attr.get(b, {}).get(Polarity.LIKE, [])
            if not likes_a or not likes_b:
                continue
            group = likes_a + likes_b
            evidence = _evidence_for(group)
            if not evidence:
                continue
            text = (
                f"you describe taste as {a} but also favor {b} scenes "
                f"({len(likes_a)}/{len(likes_b)} uses) — surfaced, not smoothed."
            )
            tensions.append(
                TasteClaim(
                    id=f"tension:{a}:{b}",
                    text=text,
                    evidence=evidence,
                    status="active",
                    provenance="high",
                    created_at=_profile_timestamp(group),
                )
            )
        return tensions

    @staticmethod
    def _build_evolution(
        vocabulary: dict[str, dict[str, Any]],
        patterns: list[TasteClaim],
        tensions: list[TasteClaim],
        previous: TasteProfile | None,
        created_at: str,
    ) -> list[dict[str, str]]:
        evolution: list[dict[str, str]] = []
        if previous is None:
            evolution.append(
                {
                    "at": created_at,
                    "summary": (
                        f"initial profile: {len(patterns)} patterns, "
                        f"{len(tensions)} tensions, {len(vocabulary)} vocabulary words"
                    ),
                }
            )
            return evolution
        for word in sorted(set(vocabulary) - set(previous.vocabulary)):
            evolution.append({"at": created_at, "summary": f"added '{word}' to vocabulary"})
        for word in sorted(set(previous.vocabulary) - set(vocabulary)):
            evolution.append({"at": created_at, "summary": f"dropped '{word}' from vocabulary"})
        prev_pattern_ids = {c.id for c in previous.patterns}
        pattern_ids = {c.id for c in patterns}
        for cid in sorted(pattern_ids - prev_pattern_ids):
            evolution.append(
                {"at": created_at, "summary": f"new pattern: {cid.removeprefix('pattern:')}"}
            )
        for cid in sorted(prev_pattern_ids - pattern_ids):
            evolution.append(
                {"at": created_at, "summary": f"retired pattern: {cid.removeprefix('pattern:')}"}
            )
        prev_tension_ids = {c.id for c in previous.tensions}
        tension_ids = {c.id for c in tensions}
        for cid in sorted(tension_ids - prev_tension_ids):
            evolution.append({"at": created_at, "summary": f"new tension surfaced: {cid}"})
        for cid in sorted(prev_tension_ids - tension_ids):
            evolution.append({"at": created_at, "summary": f"tension resolved: {cid}"})
        if not evolution:
            evolution.append({"at": created_at, "summary": "no material change"})
        return evolution


class ProfileStore:
    """Append-only persistence for the profile timeline and document.

    Takes a :class:`~curator.catalog.Catalog` (reusing its shared ``.db``) or a
    raw ``sqlite3.Connection``, mirroring the observation/store posture. The
    backing tables (``taste_profile_doc`` and ``taste_profile_events``) come from
    schema migration v15; a Catalog migrates them on construction. Events are
    appended and never erased; pin/edit/dispute also mutate the persisted
    profile document.
    """

    def __init__(self, db: sqlite3.Connection | Catalog) -> None:
        self.db = db.db if isinstance(db, Catalog) else db

    @staticmethod
    def _utc_now() -> str:
        return _utc_now()

    def pin(self, claim_id: str) -> ProfileEvent:
        """Record a pin for *claim_id* and mark the claim ``pinned``."""
        event = self._record_event(claim_id, "pin", claim_id)
        self._mutate(lambda p: _map_claim(p, claim_id, lambda c: replace(c, status="pinned")))
        return event

    def edit(self, claim_id: str, new_text: str) -> ProfileEvent:
        """Record an edit for *claim_id* and update the claim's text (status ``edited``)."""
        event = self._record_event(claim_id, "edit", new_text)
        self._mutate(
            lambda p: _map_claim(p, claim_id, lambda c: replace(c, text=new_text, status="edited"))
        )
        return event

    def dispute(self, claim_id: str) -> ProfileEvent:
        """Record a dispute and remove *claim_id* from active patterns/tensions.

        The claim's evidence is marked for re-interpretation in the event detail;
        the claim is removed from the profile (not erased — the timeline keeps it).
        """
        profile = self.load()
        shas = _claim_evidence_shas(profile, claim_id)
        detail = (
            f"evidence for re-interpretation: {', '.join(shas)}"
            if shas
            else claim_id
        )
        event = self._record_event(claim_id, "dispute", detail)
        self._mutate(lambda p: _dispute_claim(p, claim_id))
        return event

    def events(self) -> list[ProfileEvent]:
        """Return every recorded event, oldest first (append-only timeline)."""
        rows = self.db.execute(
            "SELECT claim_id, kind, detail, created_at"
            " FROM taste_profile_events ORDER BY id"
        ).fetchall()
        return [
            ProfileEvent(
                claim_id=row[0], kind=row[1], detail=row[2], created_at=row[3]
            )
            for row in rows
        ]

    def apply(self, profile: TasteProfile) -> None:
        """Persist *profile* as the current profile document."""
        self._save(profile)

    def load(self) -> TasteProfile | None:
        """Return the current profile document, or ``None`` when none applied."""
        row = self.db.execute(
            "SELECT profile_json FROM taste_profile_doc WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return TasteProfile.from_dict(json.loads(row[0]))

    def _record_event(self, claim_id: str, kind: str, detail: str) -> ProfileEvent:
        event = ProfileEvent(
            claim_id=claim_id, kind=kind, detail=detail, created_at=self._utc_now()
        )
        self.db.execute(
            "INSERT INTO taste_profile_events"
            " (claim_id, kind, detail, created_at) VALUES (?, ?, ?, ?)",
            (event.claim_id, event.kind, event.detail, event.created_at),
        )
        self.db.commit()
        return event

    def _mutate(self, mutate) -> None:
        profile = self.load()
        if profile is None:
            return
        updated = mutate(profile)
        if updated != profile:
            self._save(updated)

    def _save(self, profile: TasteProfile) -> None:
        self.db.execute(
            "INSERT INTO taste_profile_doc"
            " (id, profile_json, version, created_at, updated_at)"
            " VALUES (1, ?, ?, ?, ?)"
            " ON CONFLICT(id) DO UPDATE SET"
            " profile_json = excluded.profile_json,"
            " version = excluded.version,"
            " created_at = excluded.created_at,"
            " updated_at = excluded.updated_at",
            (
                json.dumps(profile.to_dict()),
                profile.version,
                profile.created_at,
                profile.created_at,
            ),
        )
        self.db.commit()


@dataclass(frozen=True)
class WhatILearned:
    """The no-silent-learning delta for a session: what entered the profile.

    ``summary`` is a short human summary, ``added`` lists the concrete additions,
    and ``version`` is the profile version the delta moves to (``0`` for a
    no-op). When a session added observations the delta is never empty: nothing
    enters the profile without appearing in ``added``.
    """

    summary: str
    added: list[str]
    version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "added", list(self.added))

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "added": list(self.added), "version": self.version}

    @classmethod
    def from_dict(cls, data: Any) -> WhatILearned:
        if isinstance(data, WhatILearned):
            return data
        fields = dict(data)
        return cls(
            summary=str(fields["summary"]),
            added=[str(a) for a in fields.get("added", ())],
            version=int(fields.get("version", 0)),
        )

    @classmethod
    def delta_after(
        cls,
        session_id: str,
        store: ObservationStore | Catalog | sqlite3.Connection,
    ) -> WhatILearned:
        """Return the learning delta produced by *session_id*'s observations.

        The "before" profile is built from every observation outside the session,
        the "after" from all observations, so the delta lists exactly what the
        session introduced. A session with no observations yields a no-op delta.
        """
        observation_store = (
            store
            if isinstance(store, ObservationStore)
            else ObservationStore(store.db if isinstance(store, Catalog) else store)
        )
        session_obs = observation_store.by_session(session_id)
        if not session_obs:
            return cls(
                summary=(
                    "No new observations in this session — nothing to learn.\n"
                    "The profile is unchanged."
                ),
                added=[],
                version=0,
            )
        all_obs = observation_store.all()
        before_obs = [o for o in all_obs if o.session_id != session_id]
        builder = ProfileBuilder()
        before = builder.build(before_obs)
        after = builder.build(all_obs)
        added = _delta_items(before, after)
        if not added:
            added = [
                f"reconfirmed existing taste signals ({len(session_obs)} reactions this session)"
            ]
        summary = (
            f"Learned from {len(session_obs)} new reactions, moving the profile to "
            f"version {after.version}.\n"
            f"{'; '.join(added)}.\n"
            "Nothing enters the profile without appearing here."
        )
        return cls(summary=summary, added=added, version=after.version)


@dataclass(frozen=True)
class HistoryDecision:
    """One prior like/dislike recovered from pre-dialogue history.

    ``source`` is ``"approval"`` (M003 approve/reject) or ``"pairwise"``
    (M007 preference rows). ``note`` is whatever rationale was recorded — it is
    *not* a Reaction Room verbatim, which is exactly why the claims it feeds are
    low-provenance.
    """

    source: str
    catalog_entry_id: int
    liked: bool
    note: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "catalog_entry_id": self.catalog_entry_id,
            "liked": self.liked,
            "note": self.note,
            "created_at": self.created_at,
        }


class ColdStartSeeder:
    """Seed a profile from M003 approvals + M007 pairwise history (R037).

    History is useful on day one but it is *inferred*, never spoken: every claim
    it produces is labeled ``provenance="low"`` and carries the
    :data:`_NO_VERBATIM` marker unless a rationale was recorded. Reaction Room
    observations stay ``provenance="high"`` and always sort first, so a seeded
    profile never lets a guess outrank the user's own words.

    A claim is emitted when the liked entries' mean value for an M002 signal
    separates from its contrast set (the disliked entries, else the neutral
    0.5 midpoint) by at least :data:`_MIN_SIGNAL_LEAN`. Signals that merely
    alias another (:data:`_ALIASED_SIGNALS`) are skipped so the profile never
    states the same thing twice. Reads only; nothing is written until
    :meth:`seed`'s result is applied by the caller.
    """

    def __init__(self, db: sqlite3.Connection | Catalog) -> None:
        self.db = db.db if isinstance(db, Catalog) else db

    # -- public API ---------------------------------------------------------

    def decisions(self) -> list[HistoryDecision]:
        """Return every recovered history decision, approvals first."""
        return self._approval_decisions() + self._pairwise_decisions()

    def claims(self) -> list[TasteClaim]:
        """Return the low-provenance claims implied by history (deterministic)."""
        decisions = self.decisions()
        if not decisions:
            return []
        signals = self._signals_by_entry()
        shas = self._sha_by_entry()
        claims: list[TasteClaim] = []
        for source in _HISTORY_SOURCES:
            claims.extend(
                self._claims_for_source(
                    source,
                    [d for d in decisions if d.source == source],
                    signals,
                    shas,
                )
            )
        return claims

    def seed(self, profile: TasteProfile | None = None) -> TasteProfile:
        """Return *profile* with history claims appended after its own claims.

        High-provenance patterns keep their order and position; low-provenance
        claims are appended and never replace a claim with the same id. An
        evolution line records the seeding so it is never silent.
        """
        base = profile if profile is not None else ProfileBuilder().build([])
        seeded = [c for c in self.claims() if c.id not in {p.id for p in base.patterns}]
        if not seeded:
            return base
        at = _profile_stamp_of(seeded) or base.created_at
        evolution = list(base.evolution) + [
            {
                "at": at,
                "summary": (
                    f"seeded {len(seeded)} low-provenance claims from approval/"
                    "pairwise history (not your words)"
                ),
            }
        ]
        return replace(
            base, patterns=list(base.patterns) + seeded, evolution=evolution
        )

    # -- impl ---------------------------------------------------------------

    def _approval_decisions(self) -> list[HistoryDecision]:
        """Return the latest approve/reject per catalog entry (M003 append-only)."""
        rows = self.db.execute(
            "SELECT a.catalog_entry_id, a.decision, a.rationale, a.created_at"
            " FROM approvals a"
            " JOIN (SELECT catalog_entry_id, MAX(id) AS id FROM approvals"
            "       GROUP BY catalog_entry_id) latest"
            "   ON a.id = latest.id"
            " ORDER BY a.catalog_entry_id"
        ).fetchall()
        return [
            HistoryDecision(
                source="approval",
                catalog_entry_id=int(row[0]),
                liked=str(row[1]) == "APPROVED",
                note=str(row[2] or ""),
                created_at=str(row[3] or ""),
            )
            for row in rows
        ]

    def _pairwise_decisions(self) -> list[HistoryDecision]:
        """Return the M007 pairwise preference rows (``preference`` 0 abstains)."""
        rows = self.db.execute(
            "SELECT catalog_entry_id, preference, note, created_at"
            " FROM taste_preferences WHERE preference != 0"
            " ORDER BY catalog_entry_id, id"
        ).fetchall()
        return [
            HistoryDecision(
                source="pairwise",
                catalog_entry_id=int(row[0]),
                liked=int(row[1]) > 0,
                note=str(row[2] or ""),
                created_at=str(row[3] or ""),
            )
            for row in rows
        ]

    def _signals_by_entry(self) -> dict[int, dict[str, float]]:
        """Return the newest ``ok`` analysis signal values per catalog entry."""
        rows = self.db.execute(
            "SELECT a.catalog_entry_id, a.analysis_json"
            " FROM analysis_results a"
            " JOIN (SELECT catalog_entry_id, MAX(id) AS id FROM analysis_results"
            "       WHERE status = 'ok' GROUP BY catalog_entry_id) latest"
            "   ON a.id = latest.id"
            " ORDER BY a.catalog_entry_id"
        ).fetchall()
        ranker = TasteRanker()
        signals: dict[int, dict[str, float]] = {}
        for entry_id, analysis_json in rows:
            try:
                analysis = AnalysisResult.from_dict(json.loads(analysis_json))
            except (ValueError, KeyError, TypeError):
                continue
            signals[int(entry_id)] = ranker.signal_values(analysis)
        return signals

    def _sha_by_entry(self) -> dict[int, str]:
        rows = self.db.execute(
            "SELECT id, sha256 FROM catalog_entries ORDER BY id"
        ).fetchall()
        return {int(row[0]): str(row[1]) for row in rows}

    @staticmethod
    def _claims_for_source(
        source: str,
        decisions: list[HistoryDecision],
        signals: dict[int, dict[str, float]],
        shas: dict[int, str],
    ) -> list[TasteClaim]:
        liked = [d for d in decisions if d.liked and d.catalog_entry_id in signals]
        contrast = [
            d for d in decisions if not d.liked and d.catalog_entry_id in signals
        ]
        if len(liked) < _MIN_HISTORY_SAMPLES:
            return []
        evidence = _history_evidence(liked, shas)
        if not evidence:
            return []
        claims: list[TasteClaim] = []
        for signal in SIGNAL_NAMES:
            if signal in _ALIASED_SIGNALS:
                continue
            liked_mean = _mean([signals[d.catalog_entry_id][signal] for d in liked])
            baseline = (
                _mean([signals[d.catalog_entry_id][signal] for d in contrast])
                if contrast
                else _NEUTRAL_SIGNAL
            )
            lean = liked_mean - baseline
            if abs(lean) < _MIN_SIGNAL_LEAN:
                continue
            direction = "high" if lean > 0 else "low"
            against = "the ones you passed on" if contrast else "neutral"
            claims.append(
                TasteClaim(
                    id=f"history:{source}:{signal}",
                    text=(
                        f"{_HISTORY_LABELS[source]} suggests you lean toward "
                        f"{direction} {signal.replace('_', ' ')} "
                        f"({len(liked)} picks, {lean:+.2f} vs {against}) — "
                        "inferred from history, not your words"
                    ),
                    evidence=evidence,
                    status="active",
                    provenance="low",
                    created_at=_history_stamp(liked),
                )
            )
        return claims


# -- pure helpers ------------------------------------------------------------


def _tokens(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens of *text*."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


def _profile_timestamp(obs: list[TasteObservation]) -> str:
    """Deterministic profile timestamp: the newest observation stamp (or "")."""
    stamps = [o.created_at for o in obs if o.created_at]
    return max(stamps) if stamps else ""


def _evidence_for(group: list[TasteObservation]) -> list[EvidenceRef]:
    """Build traceable evidence for *group*: every (image, verbatim, confidence)."""
    refs: list[EvidenceRef] = []
    ordered = sorted(group, key=lambda o: (o.created_at, o.id or 0, o.verbatim))
    for observation in ordered:
        for image in observation.images:
            refs.append(
                EvidenceRef(
                    image_sha=image.sha256,
                    verbatim=observation.verbatim,
                    confidence=observation.confidence,
                    created_at=observation.created_at,
                )
            )
    return refs


def _supporting_words(group: list[TasteObservation], attr: str) -> list[str]:
    """Return the sorted verbatim words in *group* that map onto *attr*."""
    words: set[str] = set()
    for observation in group:
        for token in _tokens(observation.verbatim):
            if attr in _WORD_TO_ATTRIBUTES.get(token, ()):
                words.add(token)
    return sorted(words)


def _map_claim(profile: TasteProfile, claim_id: str, fn) -> TasteProfile:
    """Return *profile* with *fn* applied to the matching claim, if present."""
    patterns = [fn(c) if c.id == claim_id else c for c in profile.patterns]
    tensions = [fn(c) if c.id == claim_id else c for c in profile.tensions]
    return replace(profile, patterns=patterns, tensions=tensions)


def _dispute_claim(profile: TasteProfile, claim_id: str) -> TasteProfile:
    """Return *profile* with *claim_id* removed from patterns and tensions."""
    patterns = [c for c in profile.patterns if c.id != claim_id]
    tensions = [c for c in profile.tensions if c.id != claim_id]
    return replace(profile, patterns=patterns, tensions=tensions)


def _claim_evidence_shas(profile: TasteProfile | None, claim_id: str) -> list[str]:
    """Return the evidence image shas of *claim_id* in *profile*, if present."""
    if profile is None:
        return []
    for claim in list(profile.patterns) + list(profile.tensions):
        if claim.id == claim_id:
            return [ref.image_sha for ref in claim.evidence]
    return []


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean of *values* (``0.0`` when empty)."""
    return sum(values) / len(values) if values else 0.0


def _history_evidence(
    decisions: list[HistoryDecision], shas: dict[int, str]
) -> list[EvidenceRef]:
    """Build low-confidence evidence for *decisions* that have a known sha."""
    refs: list[EvidenceRef] = []
    for decision in sorted(
        decisions, key=lambda d: (d.created_at, d.catalog_entry_id)
    ):
        sha = shas.get(decision.catalog_entry_id)
        if not sha:
            continue
        refs.append(
            EvidenceRef(
                image_sha=sha,
                verbatim=decision.note or _NO_VERBATIM,
                confidence=_COLD_START_CONFIDENCE,
                created_at=decision.created_at,
            )
        )
    return refs


def _history_stamp(decisions: list[HistoryDecision]) -> str:
    """Deterministic claim timestamp: the newest decision stamp (or "")."""
    stamps = [d.created_at for d in decisions if d.created_at]
    return max(stamps) if stamps else ""


def _profile_stamp_of(claims: list[TasteClaim]) -> str:
    """Deterministic stamp for a claim set: the newest claim stamp (or "")."""
    stamps = [c.created_at for c in claims if c.created_at]
    return max(stamps) if stamps else ""


def _delta_items(before: TasteProfile, after: TasteProfile) -> list[str]:
    """Return the ordered list of additions introduced by *after* vs *before*."""
    items: list[str] = []
    for word in sorted(set(after.vocabulary) - set(before.vocabulary)):
        items.append(f"added '{word}' to vocabulary")
    before_patterns = {c.id: c for c in before.patterns}
    after_patterns = {c.id: c for c in after.patterns}
    for cid in sorted(set(after_patterns) - set(before_patterns)):
        items.append(f"new pattern: {cid.removeprefix('pattern:')}")
    for cid in sorted(set(after_patterns) & set(before_patterns)):
        if len(after_patterns[cid].evidence) > len(before_patterns[cid].evidence):
            items.append(
                f"{cid.removeprefix('pattern:')} reinforced "
                f"({len(after_patterns[cid].evidence)} evidence refs)"
            )
    return items
