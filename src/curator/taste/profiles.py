"""Taste profiles: immutable, JSON-serializable rerank weights (M007/S01).

A :class:`TasteProfile` records how strongly each M002 analysis signal is weighted
when re-ranking candidates that already carry a baseline score (see
:class:`~curator.taste.rank.TasteRanker`). :func:`baseline_weights` is the identity
(all weights zero), so a default profile is behaviorally inert: its rerank exactly
reproduces the baseline order. Profiles are frozen dataclasses — ranking with one
profile can never mutate another (kind isolation).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

#: Canonical M002 analysis signals a taste profile may weight. Most map 1:1 onto
#: :class:`~curator.analysis.schema.AnalysisResult`; ``vibrancy`` is a proxy backed
#: by the color-story colorfulness signal.
SIGNAL_NAMES: tuple[str, ...] = (
    "aesthetic_quality",
    "technical_quality",
    "colorfulness",
    "harmony",
    "pairing_affinity",
    "vibrancy",
)


class TasteProfileKind(enum.Enum):
    """The scope/role a taste profile is trained or deployed for."""

    PERSONAL = "personal"
    HOUSEHOLD = "household"
    ROOM = "room"
    SEASON = "season"
    GUEST_SAFE = "guest_safe"
    EXPERIMENTAL = "experimental"


def baseline_weights() -> dict[str, float]:
    """Return the deterministic identity weights (all zero).

    A profile carrying these weights produces a zero personal delta for every
    candidate, so ranking reduces to the baseline order — enabling default
    profile behavior without special-casing.
    """
    return {name: 0.0 for name in SIGNAL_NAMES}


@dataclass(frozen=True)
class TasteProfile:
    """A frozen, JSON-serializable set of per-signal rerank weights.

    ``weights`` maps a signal name (see :data:`SIGNAL_NAMES`) to a weight; the
    supplied mapping is defensively copied on construction so later mutation of
    the caller's dict never affects the (immutable) profile. ``kind`` is
    coerced from its string form by :meth:`from_dict`.
    """

    id: str
    kind: TasteProfileKind
    name: str
    weights: dict[str, float]
    version: int = 1
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", dict(self.weights))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "name": self.name,
            "weights": dict(self.weights),
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TasteProfile:
        if isinstance(data, cls):
            return data
        return cls(
            id=str(data["id"]),
            kind=TasteProfileKind(str(data["kind"])),
            name=str(data["name"]),
            weights={str(k): float(v) for k, v in data.get("weights", {}).items()},
            version=int(data.get("version", 1)),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


def default_profile(
    kind: TasteProfileKind = TasteProfileKind.PERSONAL,
    name: str | None = None,
) -> TasteProfile:
    """Return a profile carrying the identity :func:`baseline_weights`.

    Used as the inert/default profile: its ranking equals the baseline ranking.
    """
    return TasteProfile(
        id=f"default-{kind.value}",
        kind=kind,
        name=name if name is not None else f"{kind.value} default",
        weights=baseline_weights(),
    )
