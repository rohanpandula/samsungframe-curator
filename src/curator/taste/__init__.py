"""Taste subsystem (M007/S01): profile-driven deterministic reranking.

Exposes the immutable, JSON-serializable :class:`~curator.taste.profiles.TasteProfile`
(plus :class:`~curator.taste.profiles.TasteProfileKind` and the identity
:func:`~curator.taste.profiles.baseline_weights` / inert
:func:`~curator.taste.profiles.default_profile`) and the pure, deterministic
:class:`~curator.taste.rank.TasteRanker` that adds a profile's personal delta to a
candidate's baseline score (or reproduces the baseline order exactly when the
profile is disabled).
"""

from __future__ import annotations

from curator.taste.profiles import (
    SIGNAL_NAMES,
    TasteProfile,
    TasteProfileKind,
    baseline_weights,
    default_profile,
)
from curator.taste.rank import TasteRanker

__all__ = [
    "SIGNAL_NAMES",
    "TasteProfile",
    "TasteProfileKind",
    "TasteRanker",
    "baseline_weights",
    "default_profile",
]
