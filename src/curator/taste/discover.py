"""Taste Lens Discover — synthetic federated creator catalog + Taste Deck (M007/S04).

Deterministic, offline, bootstrap-free machinery layered on S01 profiles and the
S02 pairwise :func:`~curator.taste.pairwise.apply_preference`: an immutable
:class:`Creator`/:class:`Work` model where every work carries its own
:data:`~curator.taste.profiles.SIGNAL_NAMES`-keyed signal map, a
:class:`CreatorCatalog` that simulates a federated creator network with outage
isolation (a down creator's works are simply skipped, never crashing a deck), a
:class:`TasteDeck` that deterministically composes a feed (familiar↔surprising
bias, artist spotlight, and enforced minimum diversity/exploration), a
:class:`PairwiseCalibration` that folds a stated deck preference into a new
versioned profile, and a :class:`PublicationDenied` policy that keeps a denied
creator's works out of every feed.

Signal isolation: each ``Work.signals`` map is defensively copied on construction
and a work's alignment uses *only* that work's signals — no cross-work, creator,
or context contamination. Every operation is a pure function of its inputs; any
shuffle/tie-break is seeded via ``rng_seed`` so composition is deterministic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from curator.taste.pairwise import apply_preference
from curator.taste.profiles import SIGNAL_NAMES, TasteProfile

#: Extra score a work gains when its creator is the current artist spotlight.
SPOTLIGHT_BONUS = 5.0
#: Default number of works a composed feed returns.
DEFAULT_FEED_SIZE = 12
#: Default cap on consecutive works from one creator.
DEFAULT_MAX_CONSECUTIVE = 3
#: Default minimum distinct creators to pull into the front of a feed.
DEFAULT_DIVERSITY_MIN = 2
#: Default number of leading slots checked for distinct-creator coverage.
DEFAULT_DIVERSITY_WINDOW = 6
#: Number of leading works kept stable when a ``wildcard`` feed is shuffled.
WILDCARD_KEEP = 2
#: Valid feed kinds accepted by :meth:`TasteDeck.compose`.
FEED_KINDS: tuple[str, ...] = ("likely", "adjacent", "wildcard")


def alignment(profile: TasteProfile, signals: dict[str, float]) -> float:
    """Personal delta (``sum(weight * value)``) of a single work's *signals*.

    Purely a function of the profile and one work's own signals (signal
    isolation): changing any other work's signals cannot affect this value.
    """
    return sum(profile.weights.get(n, 0.0) * signals.get(n, 0.0) for n in SIGNAL_NAMES)


@dataclass(frozen=True)
class Creator:
    """An immutable, JSON-serializable synthetic creator in the federated catalog."""

    id: str
    name: str
    genre: str
    style: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "genre": self.genre,
            "style": self.style,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Creator:
        if isinstance(data, cls):
            return data
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            genre=str(data["genre"]),
            style=str(data.get("style", "")),
        )


@dataclass(frozen=True)
class Work:
    """An immutable work carrying its own signal map plus a baseline score.

    ``signals`` maps every :data:`SIGNAL_NAMES` key to a float and is defensively
    copied on construction so later mutation of the caller's dict never reaches
    the (immutable) work (signal isolation).
    """

    id: str
    creator_id: str
    title: str
    baseline: float
    signals: dict[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "signals", dict(self.signals))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "creator_id": self.creator_id,
            "title": self.title,
            "baseline": self.baseline,
            "signals": dict(self.signals),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Work:
        if isinstance(data, cls):
            return data
        return cls(
            id=str(data["id"]),
            creator_id=str(data["creator_id"]),
            title=str(data["title"]),
            baseline=float(data["baseline"]),
            signals={str(k): float(v) for k, v in data.get("signals", {}).items()},
        )


class CreatorCatalog:
    """A deterministic synthetic federated creator catalog with outage isolation.

    ``creators`` and ``works`` are indexed on construction; lookups are
    deterministic (sorted by id). A creator marked down via :meth:`set_down` is
    treated as unavailable: :meth:`creator_available` returns ``False`` and
    :meth:`works` returns an empty list — so a deck never errors on an outage.
    """

    def __init__(self, creators: list[Creator], works: list[Work]) -> None:
        self._creators: dict[str, Creator] = {c.id: c for c in creators}
        self._works: dict[str, Work] = {w.id: w for w in works}
        self._by_creator: dict[str, list[str]] = {}
        for work in works:
            self._by_creator.setdefault(work.creator_id, []).append(work.id)
        for ids in self._by_creator.values():
            ids.sort()
        self._down: set[str] = set()

    def creators(self) -> list[Creator]:
        """Return all known creators in deterministic (id) order."""
        return [self._creators[cid] for cid in sorted(self._creators)]

    def creator_available(self, creator_id: str) -> bool:
        """True when the creator is known and not currently down."""
        return creator_id in self._creators and creator_id not in self._down

    def works(self, creator_id: str) -> list[Work]:
        """Return the creator's works, or ``[]`` when the creator is down.

        An *unknown* creator id raises :class:`KeyError` (a genuine error, not an
        outage); a down creator simply yields no works — never an exception.
        """
        if creator_id not in self._creators:
            raise KeyError(f"unknown creator: {creator_id!r}")
        if creator_id in self._down:
            return []
        return [self._works[wid] for wid in self._by_creator.get(creator_id, [])]

    def work(self, work_id: str) -> Work:
        """Return the work with *work_id*, raising :class:`KeyError` if unknown."""
        if work_id not in self._works:
            raise KeyError(f"unknown work: {work_id!r}")
        return self._works[work_id]

    def set_down(self, creator_id: str) -> None:
        """Mark *creator_id* down: its works are skipped by every caller."""
        self._down.add(creator_id)

    def bring_up(self, creator_id: str) -> None:
        """Clear an outage flag so the creator's works are served again."""
        self._down.discard(creator_id)


class PublicationDenied:
    """Policy gate that keeps denied creators' works out of every feed.

    ``can_publish`` is False for any creator added via :meth:`deny`. The deck
    filters denied creators *before* composition, so a denied work is never
    ranked, surfaced, returned by an action, or calibrated against.
    """

    def __init__(self, denied: set[str] | None = None) -> None:
        self._denied: set[str] = set(denied or ())

    def deny(self, creator_id: str) -> None:
        self._denied.add(creator_id)

    def can_publish(self, creator_id: str) -> bool:
        return creator_id not in self._denied


class TasteDeck:
    """Deterministic feed composer over a federated creator catalog.

    ``compose`` scores every eligible (not-down, publishable) work as
    ``baseline + (1 - 2 * familiar_surprising) * alignment`` (plus a spotlight
    bonus for the spotlight creator) and deterministically orders the result.
    ``familiar_surprising`` in ``(-1..1)`` biases toward known/recommended
    (negative = high alignment up top) or novel/exploration (positive = low
    alignment up top). A ``wildcard`` feed shuffles the tail with a seeded RNG
    to inject diversity. A greedy builder enforces no more than
    ``max_consecutive`` consecutive works from one creator and at least
    ``diversity_min`` distinct creators within the front ``diversity_window``.
    """

    def __init__(
        self,
        catalog: CreatorCatalog,
        *,
        profile: TasteProfile | None = None,
        publication_policy: PublicationDenied | None = None,
    ) -> None:
        self.catalog = catalog
        self.profile = profile
        self.publication_policy = publication_policy
        self._feed: list[Work] = []
        self._cursor = 0

    def _eligible_works(self) -> list[Work]:
        """All works whose creator is available AND permitted to publish."""
        out: list[Work] = []
        for creator in self.catalog.creators():
            if not self.catalog.creator_available(creator.id):
                continue
            if self.publication_policy is not None and not self.publication_policy.can_publish(
                creator.id
            ):
                continue
            out.extend(self.catalog.works(creator.id))
        return out

    @staticmethod
    def _build_feed(
        ordered: list[Work],
        *,
        max_consecutive: int,
        diversity_min: int,
        diversity_window: int,
    ) -> list[Work]:
        """Greedily reorder *ordered* to cap creator runs and force coverage."""
        if diversity_min < 1:
            diversity_min = 1
        if max_consecutive < 1:
            max_consecutive = 1
        result: list[Work] = []
        last: str | None = None
        consec = 0
        pool = list(ordered)
        while pool:
            chosen = 0
            for i, w in enumerate(pool):
                violates_consec = w.creator_id == last and consec + 1 > max_consecutive
                push_for_coverage = False
                if len(result) < diversity_window:
                    front = {x.creator_id for x in result}
                    missing = max(0, diversity_min - len(front))
                    if missing > 0:
                        unseen = w.creator_id not in front
                        another_unseen_ok = any(
                            x.creator_id not in front
                            and not (
                                x.creator_id == last and consec + 1 > max_consecutive
                            )
                            for x in pool
                            if x is not w
                        )
                        push_for_coverage = (not unseen) and another_unseen_ok
                if violates_consec or push_for_coverage:
                    continue
                chosen = i
                break
            w = pool.pop(chosen)
            if w.creator_id == last:
                consec += 1
            else:
                last = w.creator_id
                consec = 1
            result.append(w)
        return result

    def compose(
        self,
        feed_kind: str,
        profile: TasteProfile | None = None,
        *,
        familiar_surprising: float = 0.0,
        artist_spotlight: str | None = None,
        rng_seed: int = 0,
        diversity_min: int = DEFAULT_DIVERSITY_MIN,
        size: int = DEFAULT_FEED_SIZE,
        max_consecutive: int = DEFAULT_MAX_CONSECUTIVE,
        diversity_window: int = DEFAULT_DIVERSITY_WINDOW,
    ) -> list[Work]:
        """Compose and return the deterministic feed of :class:`Work` objects."""
        if feed_kind not in FEED_KINDS:
            raise ValueError(f"unknown feed_kind: {feed_kind!r}")
        if not -1.0 <= familiar_surprising <= 1.0:
            raise ValueError("familiar_surprising must be within [-1, 1]")
        prof = profile if profile is not None else self.profile
        if prof is None:
            raise ValueError("a profile is required to compose a feed")

        scored: list[tuple[float, str, Work]] = []
        for work in self._eligible_works():
            delta = alignment(prof, work.signals)
            bonus = SPOTLIGHT_BONUS if work.creator_id == artist_spotlight else 0.0
            score = work.baseline + (1.0 - 2.0 * familiar_surprising) * delta + bonus
            scored.append((score, work.id, work))
        scored.sort(key=lambda row: (-row[0], row[1]))
        ordered = [work for _, _, work in scored]

        if feed_kind == "wildcard":
            rng = random.Random(rng_seed)
            keep = ordered[:WILDCARD_KEEP]
            tail = ordered[WILDCARD_KEEP:]
            rng.shuffle(tail)
            ordered = keep + tail

        feed = self._build_feed(
            ordered,
            max_consecutive=max_consecutive,
            diversity_min=diversity_min,
            diversity_window=diversity_window,
        )
        feed = feed[:size]
        self._feed = list(feed)
        self._cursor = 0
        return list(feed)

    def feed_actions(self) -> list[str]:
        """List of actions the deck can execute."""
        return ["next", "previous", "spotlight", "calibrate_prefer_a", "calibrate_prefer_b"]

    def _work_by_id(self, work_id: str) -> Work:
        work = self.catalog.work(work_id)
        if not self.catalog.creator_available(work.creator_id):
            raise ValueError(f"creator of {work_id!r} is unavailable")
        if self.publication_policy is not None and not self.publication_policy.can_publish(
            work.creator_id
        ):
            raise ValueError(f"{work_id!r} belongs to a denied creator")
        return work

    def deck_action(self, action: str, *, channel: str = "button", **params: Any) -> Any:
        """Execute *action* deterministically; ``channel`` never changes the result.

        Parity: ``channel`` is a cosmetic label (``'button'`` / ``'keyboard'`` /
        ``'gesture'``) that is discarded before any computation, so the same
        action with the same inputs returns identical output regardless of how it
        was invoked — a single shared, deterministic code path.
        """
        del channel
        if action == "next":
            if self._cursor >= len(self._feed):
                raise IndexError("feed exhausted")
            work = self._feed[self._cursor]
            self._cursor += 1
            return work
        if action == "previous":
            if self._cursor <= 0:
                raise IndexError("at feed start")
            self._cursor -= 1
            return self._feed[self._cursor]
        if action == "spotlight":
            return self.compose(
                str(params.get("feed_kind", "likely")),
                params.get("profile", self.profile),
                familiar_surprising=float(params.get("familiar_surprising", 0.0)),
                artist_spotlight=params.get("artist_spotlight"),
                rng_seed=int(params.get("rng_seed", 0)),
            )
        if action in ("calibrate_prefer_a", "calibrate_prefer_b"):
            cal = PairwiseCalibration()
            prefer_a = action == "calibrate_prefer_a"
            prof = params.get("profile", self.profile)
            if prof is None:
                raise ValueError("a profile is required to calibrate")
            raw_a = params.get("work_a_id")
            a_id: str | None = str(raw_a) if raw_a is not None else (
                self._feed[0].id if self._feed else None
            )
            raw_b = params.get("work_b_id")
            b_id: str | None = str(raw_b) if raw_b is not None else (
                self._feed[1].id if len(self._feed) > 1 else None
            )
            if a_id is None or b_id is None:
                raise ValueError("need two deck works to calibrate")
            result = cal.calibrate(prof, self._work_by_id(a_id), self._work_by_id(b_id), prefer_a)
            self.profile = result.profile
            return result
        raise ValueError(f"unknown action: {action!r}")


@dataclass(frozen=True)
class CalibrationResult:
    """JSON-serializable result of one deck calibration preference."""

    profile: TasteProfile
    work_a_id: str
    work_b_id: str
    preferred_work_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "work_a_id": self.work_a_id,
            "work_b_id": self.work_b_id,
            "preferred_work_id": self.preferred_work_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationResult:
        if isinstance(data, cls):
            return data
        if isinstance(data.get("profile"), TasteProfile):
            prof = data["profile"]
        else:
            prof = TasteProfile.from_dict(data["profile"])
        return cls(
            profile=prof,
            work_a_id=str(data["work_a_id"]),
            work_b_id=str(data["work_b_id"]),
            preferred_work_id=str(data["preferred_work_id"]),
        )


class PairwiseCalibration:
    """Deterministic calibration that folds a deck preference into a profile.

    Uses S02 :func:`~curator.taste.pairwise.apply_preference` on the two works'
    own signal maps, returning a new :class:`CalibrationResult` carrying a
    version-bumped :class:`TasteProfile` whose weights shift toward the preferred
    work's signals. The input profile is never mutated.
    """

    def calibrate(
        self,
        profile: TasteProfile,
        work_a: Work,
        work_b: Work,
        prefer_a: bool = True,
    ) -> CalibrationResult:
        updated = apply_preference(profile, work_a.signals, work_b.signals, prefer_a=prefer_a)
        preferred = work_a.id if prefer_a else work_b.id
        return CalibrationResult(
            profile=updated,
            work_a_id=work_a.id,
            work_b_id=work_b.id,
            preferred_work_id=preferred,
        )
