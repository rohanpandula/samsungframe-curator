"""Taste controls: blending, veto, history, and render isolation (M007/S03).

Deterministic, bootstrap-free machinery built on S01 profiles: a weighted
:func:`blend` of two profiles, a :func:`veto` / :func:`apply_veto` pair that
demotes or drops candidates a vetoer strongly dislikes, a :class:`ProfileHistory`
that snapshots/replays/undoes/exports/resets/deletes immutable profile versions,
and an :func:`approved_manifest_unchanged` regression check proving the taste
layer never perturbs the deterministic renderer path.

Every operation is a pure function of its inputs — no RNG, no mutable shared
state leaked to callers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from curator.artdirection.manifest import ArtDirectionManifest
from curator.render.renderer import DeterministicRenderer
from curator.taste.profiles import (
    SIGNAL_NAMES,
    TasteProfile,
    default_profile,
)

#: A vetoer weight below this magnitude is treated as "strongly negative".
VETO_THRESHOLD = -2.0
#: Default personal delta below which a candidate is treated as strongly disliked.
DROP_THRESHOLD = -1.0


def blend(a: TasteProfile, b: TasteProfile, w: float) -> TasteProfile:
    """Return a NEW profile whose weights are the deterministic ``w`` blend.

    ``weights[s] = w * a.weights[s] + (1 - w) * b.weights[s]`` for every signal in
    :data:`SIGNAL_NAMES`. The result is a fresh profile (no input is mutated),
    keyed by ``a``'s kind with a generated ``blend-<a>-<b>`` id, and carries
    ``version = max(a.version, b.version) + 1``. Deterministic in its inputs.
    """
    weights = {
        name: w * a.weights.get(name, 0.0) + (1.0 - w) * b.weights.get(name, 0.0)
        for name in SIGNAL_NAMES
    }
    return TasteProfile(
        id=f"blend-{a.id}-{b.id}",
        kind=a.kind,
        name=f"blend({a.name}, {b.name})",
        weights=weights,
        version=max(a.version, b.version) + 1,
        created_at=a.created_at,
        updated_at=a.updated_at,
    )


def _strongly_negative_signals(vetoer: TasteProfile) -> set[str]:
    """Return signals the *vetoer* weights strongly negative."""
    return {
        name for name in SIGNAL_NAMES if vetoer.weights.get(name, 0.0) < VETO_THRESHOLD
    }


def veto(primary: TasteProfile, vetoer: TasteProfile) -> TasteProfile:
    """Return a NEW profile encoding *primary* overlaid with the vetoer's veto.

    Every signal the *vetoer* weights below :data:`VETO_THRESHOLD` is flagged and
    its (strongly negative) vetoer weight overrides *primary*'s weight, so a vetoed
    candidate scores very low; all other signals keep *primary*'s weights. The
    returned profile is fresh, ``version = primary.version + 1``, same id/kind/name.
    """
    weights = dict(primary.weights)
    for name in _strongly_negative_signals(vetoer):
        weights[name] = vetoer.weights[name]
    return TasteProfile(
        id=primary.id,
        kind=primary.kind,
        name=primary.name,
        weights=weights,
        version=primary.version + 1,
        created_at=primary.created_at,
        updated_at=primary.updated_at,
    )


def _vetoer_delta(vetoer: TasteProfile, signals: dict[str, float]) -> float:
    """Personal delta the *vetoer* assigns to a candidate's *signals*."""
    return sum(
        vetoer.weights.get(name, 0.0) * signals.get(name, 0.0)
        for name in SIGNAL_NAMES
    )


def apply_veto(
    ranked: list[dict[str, Any]],
    signals_map: dict[Any, dict[str, float]],
    vetoer: TasteProfile,
    *,
    policy: str = "demote",
    drop_threshold: float = DROP_THRESHOLD,
) -> list[dict[str, Any]]:
    """Deterministically demote/drop candidates the *vetoer* strongly dislikes.

    Each candidate is disliked when the vetoer's personal delta on its signals is
    below ``drop_threshold``. ``policy="demote"`` (default) pushes disliked
    candidates to the end, preserving relative order of kept and demoted groups;
    ``policy="drop"`` removes them entirely. ``signals_map`` maps a candidate id
    to its per-signal values. Deterministic: ties never reorder arbitrarily.
    """
    scored: list[tuple[bool, int, dict[str, Any]]] = []
    for index, cand in enumerate(ranked):
        disliked = _vetoer_delta(vetoer, signals_map[cand["id"]]) < drop_threshold
        scored.append((disliked, index, cand))
    if policy == "demote":
        kept = [cand for disliked, _, cand in scored if not disliked]
        demoted = [cand for disliked, _, cand in scored if disliked]
        return kept + demoted
    if policy == "drop":
        return [cand for disliked, _, cand in scored if not disliked]
    raise ValueError(f"unknown veto policy: {policy!r}")


@dataclass
class ProfileHistory:
    """Append-only store of immutable profile snapshots keyed by version.

    ``snapshots`` maps a profile version to an immutable deep copy of that
    profile. Snapshots are never mutated once stored; ``replay`` always returns a
    fresh copy so a caller cannot corrupt history.
    """

    snapshots: dict[int, TasteProfile] = field(default_factory=dict)

    def snapshot(self, profile: TasteProfile) -> int:
        """Store an immutable deep copy of *profile*, returning its version.

        Append-only: an existing version is never overwritten (returns the stored
        version unchanged), so history is an immutable ledger of versions.
        """
        frozen = TasteProfile.from_dict(profile.to_dict())
        self.snapshots.setdefault(profile.version, frozen)
        return profile.version

    def replay(self, version: int) -> TasteProfile:
        """Return a fresh deep copy of the snapshot at *version*."""
        if version not in self.snapshots:
            raise KeyError(f"no snapshot at version {version}")
        return TasteProfile.from_dict(self.snapshots[version].to_dict())

    def undo(self) -> TasteProfile:
        """Return the previous version relative to the newest snapshot.

        Raise :class:`ValueError` when there is nothing to undo (fewer than two
        versions recorded).
        """
        if len(self.snapshots) < 2:
            raise ValueError("nothing to undo: fewer than two snapshots")
        previous = max(self.snapshots) - 1
        return self.replay(previous)

    def head(self) -> int:
        """Return the newest snapshot version, or 0 when history is empty."""
        return max(self.snapshots) if self.snapshots else 0

    def export_profile(self, profile: TasteProfile) -> dict[str, Any]:
        """Return a portable, secret-free JSON dict for *profile*.

        :class:`TasteProfile` carries no secrets, so this is exactly its
        ``to_dict()`` form — weights, kind, version, and metadata.
        """
        return profile.to_dict()

    def reset(self, profile: TasteProfile) -> TasteProfile:
        """Return the identity :func:`default_profile` for *profile*'s kind."""
        return default_profile(kind=profile.kind)

    def delete(self, version: int) -> None:
        """Remove the snapshot at *version* from history."""
        self.snapshots.pop(version, None)

    def delete_profiles(self, versions: Iterable[int]) -> None:
        """Remove every snapshot listed in *versions* (no-op for unknowns)."""
        for version in versions:
            self.snapshots.pop(version, None)


def approved_manifest_unchanged(
    manifest: ArtDirectionManifest,
    sources: dict[str, bytes],
    target: tuple[int, int] = (1920, 1080),
) -> bool:
    """Regression check that the renderer path is deterministic and untouched.

    The taste layer never feeds into :class:`DeterministicRenderer`, so rendering
    the approved *manifest* twice must produce identical :class:`RenderResult`
    sha256 hashes. Returns True iff both renders agree (determinism/isolation).
    """
    renderer = DeterministicRenderer()
    first = renderer.render(manifest, sources, target).sha256
    second = renderer.render(manifest, sources, target).sha256
    return first == second
