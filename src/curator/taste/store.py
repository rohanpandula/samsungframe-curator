"""Vote persistence: the M007 Lens profile's first writer (M009/S01).

``taste_profiles`` and ``taste_preferences`` (schema v13, M007) shipped with the
pairwise engine, the profile model, and the reranker — but nothing in ``src/``
ever wrote to them and no surface reached them, so no preference vote had ever
been recorded (R039). :class:`TasteVoteStore` is that writer: it persists one
:class:`~curator.taste.profiles.TasteProfile` (the single "personal" Lens
profile this slice wires) and the append-only ``taste_preferences`` journal of
grouped, retractable pairwise votes.

No new intelligence lives here — :func:`~curator.taste.pairwise.choose_pair` and
:func:`~curator.taste.pairwise.apply_preference` are reused exactly as M007 built
them. This module only makes that loop reachable, persistent, and reversible:
:meth:`TasteVoteStore.record_vote` writes a vote, :meth:`TasteVoteStore.retract`
reverses one without deleting history, and :meth:`TasteVoteStore.rebuild_profile`
is the replay engine that derives the current profile from the append-only
journal — the same mechanism that makes "history survives a rebuild" true by
construction.

:func:`resolve_vote_candidates` and :func:`next_pair` are the shared
candidate-resolution + pair-selection functions the CLI and the API both call, so
"the pair" a user sees is defined once, not reimplemented per surface.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from typing import Any

from curator.analysis.schema import AnalysisResult
from curator.catalog import Catalog
from curator.errors import CuratorError
from curator.taste.pairwise import apply_preference, choose_pair
from curator.taste.profiles import TasteProfile, TasteProfileKind, default_profile
from curator.taste.rank import TasteRanker

# ISO-8601 UTC timestamp used for the columns this module writes explicitly
# (mirrors ``curator.catalog``'s ``_TIMESTAMP`` upsert idiom).
_TIMESTAMP = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

#: The stable ``taste_profiles.uid`` for the single household Lens profile this
#: slice persists. No household/room/season profiles are wired yet — out of
#: scope for S01.
PERSONAL_PROFILE_UID = "personal"


class TasteVoteError(CuratorError):
    """Raised when a vote/profile persistence operation fails (mirrors CatalogError)."""


@dataclass(frozen=True)
class VoteRecord:
    """One recorded pairwise vote: the winner/loser pair of one ``vote_group``."""

    vote_group: str
    winner_entry_id: int
    loser_entry_id: int
    note: str
    created_at: str
    retracted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "vote_group": self.vote_group,
            "winner_entry_id": self.winner_entry_id,
            "loser_entry_id": self.loser_entry_id,
            "note": self.note,
            "created_at": self.created_at,
            "retracted": self.retracted,
        }


class TasteVoteStore:
    """Persist + replay the personal Lens profile and its pairwise vote journal.

    Takes a :class:`~curator.catalog.Catalog` (reusing its shared ``.db``) or a
    raw ``sqlite3.Connection``, mirroring
    :class:`~curator.taste.dialogue.profile.ProfileStore`/
    :class:`~curator.taste.dialogue.profile.ColdStartSeeder`'s coercion idiom —
    this project keeps domain-specific persistence in sibling store classes, not
    bolted onto :class:`~curator.catalog.Catalog`.
    """

    def __init__(self, db: sqlite3.Connection | Catalog) -> None:
        self.db = db.db if isinstance(db, Catalog) else db

    # -- profile persistence --------------------------------------------------

    def load_profile(self, uid: str = PERSONAL_PROFILE_UID) -> TasteProfile:
        """Return the persisted profile for *uid*, or the exact baseline default.

        A never-voted install has no ``taste_profiles`` row for *uid*, so this
        reads back byte-identical to ``default_profile(TasteProfileKind.PERSONAL)``
        — the R038 zero-votes-means-baseline contract.
        """
        row = self.db.execute(
            "SELECT weights_json, version, created_at, updated_at"
            " FROM taste_profiles WHERE uid = ?",
            (uid,),
        ).fetchone()
        if row is None:
            return default_profile(TasteProfileKind.PERSONAL)
        weights_json, version, created_at, updated_at = row
        base = default_profile(TasteProfileKind.PERSONAL)
        return TasteProfile(
            id=base.id,
            kind=base.kind,
            name=base.name,
            weights=json.loads(weights_json),
            version=int(version),
            created_at=str(created_at or ""),
            updated_at=str(updated_at or ""),
        )

    def save_profile(self, profile: TasteProfile, uid: str = PERSONAL_PROFILE_UID) -> None:
        """Upsert *profile* into ``taste_profiles`` keyed on *uid*.

        Follows :meth:`~curator.catalog.Catalog.set_image_signature`'s upsert
        idiom: ``ON CONFLICT(uid) DO UPDATE`` refreshes ``kind``/``name``/
        ``version``/``weights_json``/``updated_at``; ``created_at`` keeps its
        table default on first insert only, mirroring
        :meth:`~curator.catalog.Catalog.add_source`.
        """
        try:
            self.db.execute(
                "INSERT INTO taste_profiles(uid, kind, name, version, weights_json)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT(uid) DO UPDATE SET"
                "   kind = excluded.kind,"
                "   name = excluded.name,"
                "   version = excluded.version,"
                "   weights_json = excluded.weights_json,"
                f"   updated_at = {_TIMESTAMP}",
                (
                    uid,
                    profile.kind.value,
                    profile.name,
                    profile.version,
                    json.dumps(profile.weights),
                ),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise TasteVoteError(f"failed to save taste profile uid={uid!r}: {exc}") from exc

    def _ensure_profile_row(self, uid: str = PERSONAL_PROFILE_UID) -> int:
        """Return the ``taste_profiles.id`` for *uid*, creating it on first use."""
        row = self.db.execute(
            "SELECT id FROM taste_profiles WHERE uid = ?", (uid,)
        ).fetchone()
        if row is None:
            self.save_profile(default_profile(TasteProfileKind.PERSONAL), uid=uid)
            row = self.db.execute(
                "SELECT id FROM taste_profiles WHERE uid = ?", (uid,)
            ).fetchone()
        if row is None:
            raise TasteVoteError(f"failed to create taste_profiles row for uid={uid!r}")
        return int(row[0])

    def _signals_by_entry(self) -> dict[int, dict[str, float]]:
        """Return the newest ``ok`` analysis signal values per catalog entry.

        Same JOIN-latest-``ok``-``analysis_results`` query + ``signal_values``
        pattern as
        :meth:`~curator.taste.dialogue.profile.ColdStartSeeder._signals_by_entry`
        — a small, deliberate, in-idiom duplication: this module and
        ``dialogue/profile.py`` are separate subsystems and neither imports
        business logic from the other today.
        """
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

    # -- vote journal -----------------------------------------------------------

    def record_vote(
        self, winner_entry_id: int, loser_entry_id: int, note: str = ""
    ) -> VoteRecord:
        """Record one pairwise vote and immediately fold it into the profile.

        Inserts the winner row (``preference=1``) and the loser row
        (``preference=-1``) under a shared, freshly-generated ``vote_group`` in
        one transaction — an opaque row identifier, not a ranking decision, same
        category of non-determinism already accepted for ``taste_sessions.id``.
        Rolls back and raises :class:`TasteVoteError` on any ``sqlite3.Error``.
        """
        profile_id = self._ensure_profile_row()
        vote_group = uuid.uuid4().hex
        try:
            self.db.execute(
                "INSERT INTO taste_preferences"
                " (profile_id, catalog_entry_id, preference, note, vote_group)"
                " VALUES (?, ?, 1, ?, ?)",
                (profile_id, winner_entry_id, note, vote_group),
            )
            self.db.execute(
                "INSERT INTO taste_preferences"
                " (profile_id, catalog_entry_id, preference, note, vote_group)"
                " VALUES (?, ?, -1, ?, ?)",
                (profile_id, loser_entry_id, note, vote_group),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise TasteVoteError(f"failed to record vote {vote_group}: {exc}") from exc
        self.save_profile(self.rebuild_profile())
        row = self.db.execute(
            "SELECT created_at FROM taste_preferences"
            " WHERE vote_group = ? ORDER BY id LIMIT 1",
            (vote_group,),
        ).fetchone()
        return VoteRecord(
            vote_group=vote_group,
            winner_entry_id=winner_entry_id,
            loser_entry_id=loser_entry_id,
            note=note,
            created_at=str(row[0]) if row else "",
        )

    def votes(self) -> list[VoteRecord]:
        """Return one :class:`VoteRecord` per ``vote_group``, oldest first.

        Includes retracted votes (``retracted=True``) so history stays visible,
        not hidden — retraction never deletes a row.
        """
        rows = self.db.execute(
            "SELECT vote_group, catalog_entry_id, preference, note, created_at,"
            "       retracted_at"
            " FROM taste_preferences"
            " WHERE vote_group IS NOT NULL"
            " ORDER BY id"
        ).fetchall()
        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for vote_group, entry_id, preference, note, created_at, retracted_at in rows:
            if vote_group not in groups:
                groups[vote_group] = {
                    "winner": None,
                    "loser": None,
                    "note": note or "",
                    "created_at": created_at,
                    "retracted": False,
                }
                order.append(vote_group)
            group = groups[vote_group]
            if preference > 0:
                group["winner"] = int(entry_id)
            elif preference < 0:
                group["loser"] = int(entry_id)
            if retracted_at is not None:
                group["retracted"] = True
        records: list[VoteRecord] = []
        for vote_group in order:
            group = groups[vote_group]
            if group["winner"] is None or group["loser"] is None:
                continue  # defensive: partial/corrupt group, mirrors rebuild_profile
            records.append(
                VoteRecord(
                    vote_group=vote_group,
                    winner_entry_id=group["winner"],
                    loser_entry_id=group["loser"],
                    note=group["note"],
                    created_at=str(group["created_at"] or ""),
                    retracted=group["retracted"],
                )
            )
        return records

    def retract(self, vote_group: str) -> bool:
        """Mark every row of *vote_group* retracted; recompute + save the profile.

        Never deletes a row — ``retracted_at`` is stamped instead. Returns
        ``False`` (no-op) when *vote_group* is unknown or already retracted
        (``cur.rowcount == 0``); the caller maps this to ``EXIT_NO_CHANGE``/404.
        """
        try:
            cur = self.db.execute(
                f"UPDATE taste_preferences SET retracted_at = {_TIMESTAMP}"
                " WHERE vote_group = ? AND retracted_at IS NULL",
                (vote_group,),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise TasteVoteError(f"failed to retract vote {vote_group}: {exc}") from exc
        if cur.rowcount == 0:
            return False
        self.save_profile(self.rebuild_profile())
        return True

    def rebuild_profile(self, uid: str = PERSONAL_PROFILE_UID) -> TasteProfile:
        """Replay the active vote journal into a fresh profile (the replay engine).

        Starts from ``default_profile(TasteProfileKind.PERSONAL)`` and folds in
        :func:`~curator.taste.pairwise.apply_preference` once per complete,
        non-retracted ``vote_group`` in chronological (first-row-id) order — a
        plain ``dict`` preserves insertion order, so ``vote_group`` strings (random
        UUIDs) are never sorted, which would scramble chronology. A vote_group
        missing a winner or loser row (partial/corrupt), or whose entries carry no
        ``ok`` analysis, is skipped rather than crashing the rebuild. This is what
        makes retract "just work" (a retracted vote_group is excluded by the
        ``WHERE`` clause) and makes "history survives a rebuild" true by
        construction: rows are never deleted, only replayed.
        """
        rows = self.db.execute(
            "SELECT vote_group, catalog_entry_id, preference FROM taste_preferences"
            " WHERE retracted_at IS NULL AND vote_group IS NOT NULL"
            " ORDER BY id"
        ).fetchall()
        groups: dict[str, dict[str, int]] = {}
        for vote_group, entry_id, preference in rows:
            group = groups.setdefault(vote_group, {})
            if preference > 0:
                group["winner"] = int(entry_id)
            elif preference < 0:
                group["loser"] = int(entry_id)
        signals = self._signals_by_entry()
        profile = default_profile(TasteProfileKind.PERSONAL)
        for group in groups.values():
            winner = group.get("winner")
            loser = group.get("loser")
            if winner is None or loser is None:
                continue
            if winner not in signals or loser not in signals:
                continue
            profile = apply_preference(profile, signals[winner], signals[loser], prefer_a=True)
        return profile


# -- shared candidate resolution + pair selection (CLI + API) -----------------


def resolve_vote_candidates(
    catalog: Catalog,
) -> tuple[list[dict[str, Any]], dict[str, AnalysisResult]]:
    """Return ``(candidates, analysis_map)`` for every analyzed catalog entry.

    Same JOIN-latest-``ok``-``analysis_results`` query as
    :meth:`TasteVoteStore._signals_by_entry`, but keeping the full
    :class:`~curator.analysis.schema.AnalysisResult` (``choose_pair`` needs the
    full object, not just signal values). Candidate ids are strings
    (``str(catalog_entry_id)``), matching
    :func:`~curator.taste.pairwise.choose_pair`'s declared ``exclude`` type
    exactly; convert back to ``int`` only when writing
    ``taste_preferences.catalog_entry_id``. ``baseline`` is the entry's existing
    ``catalog_entries.quality_score``, or ``0.0`` when NULL.
    """
    rows = catalog.db.execute(
        "SELECT a.catalog_entry_id, e.quality_score, a.analysis_json"
        " FROM analysis_results a"
        " JOIN (SELECT catalog_entry_id, MAX(id) AS id FROM analysis_results"
        "       WHERE status = 'ok' GROUP BY catalog_entry_id) latest"
        "   ON a.id = latest.id"
        " JOIN catalog_entries e ON e.id = a.catalog_entry_id"
        " ORDER BY a.catalog_entry_id"
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    analysis_map: dict[str, AnalysisResult] = {}
    for entry_id, quality_score, analysis_json in rows:
        try:
            analysis = AnalysisResult.from_dict(json.loads(analysis_json))
        except (ValueError, KeyError, TypeError):
            continue
        cid = str(entry_id)
        candidates.append({"id": cid, "baseline": float(quality_score or 0.0)})
        analysis_map[cid] = analysis
    return candidates, analysis_map


def next_pair(
    catalog: Catalog, *, rng_seed: int = 0
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Return the current A/B taste comparison, or ``None`` when there isn't one.

    Resolves candidates, excludes every already-**non-retracted** voted pair
    (both orders), loads the current profile, and delegates pair choice to
    :func:`~curator.taste.pairwise.choose_pair`. Deterministic given unchanged
    catalog/vote state — this is the mechanism that lets a preview call
    (``curator taste vote``) and an answering call (``curator taste vote
    --prefer a``) be two separate, stateless invocations that agree on "the
    pair" without a pending-vote table. Never raises: fewer than two eligible
    candidates, or ``choose_pair``'s ``ValueError("no candidate pairs to choose
    from")``, both translate to ``None`` — the "nothing to vote on yet" signal.
    """
    candidates, analysis_map = resolve_vote_candidates(catalog)
    if len(candidates) < 2:
        return None
    store = TasteVoteStore(catalog)
    exclude: set[tuple[str, str]] = set()
    for vote in store.votes():
        if vote.retracted:
            continue
        a_id, b_id = str(vote.winner_entry_id), str(vote.loser_entry_id)
        exclude.add((a_id, b_id))
        exclude.add((b_id, a_id))
    profile = store.load_profile()
    try:
        return choose_pair(
            candidates, profile, analysis_map=analysis_map, rng_seed=rng_seed, exclude=exclude
        )
    except ValueError:
        return None
