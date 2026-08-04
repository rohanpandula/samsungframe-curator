"""Analysis provider boundary + capabilities/probe (M002/S01).

:class:`AnalysisProvider` is the M002 -> M006 seam: any concrete analysis
backend (local CPU, CoreML/Metal, cloud/hybrid) implements the ABC and advertises
its :class:`AnalysisCapabilities` plus a live :meth:`AnalysisProvider.probe`.
:func:`capability_requirements` validates that a provider's declared
capabilities cover a requested profile/backend before a run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from curator.analysis.compute import ComputeBackend
from curator.analysis.profiles import KNOWN_STAGES, AnalysisProfile


@dataclass(frozen=True)
class AnalysisCapabilities:
    """Declared capabilities of an analysis provider."""

    profiles: frozenset[AnalysisProfile]
    backends: frozenset[ComputeBackend]
    stages: frozenset[str] = frozenset(KNOWN_STAGES)
    air_gapped: bool = True
    deterministic: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-encodable dict (enums serialized by their values)."""
        return {
            "profiles": sorted(p.value for p in self.profiles),
            "backends": sorted(b.value for b in self.backends),
            "stages": sorted(self.stages),
            "air_gapped": self.air_gapped,
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True)
class ComputeProbe:
    """A point-in-time health/probe result for one backend."""

    ok: bool
    backend: ComputeBackend
    latency_ms: float = 0.0
    available_backends: frozenset[ComputeBackend] = frozenset()
    message: str | None = None


class AnalysisProvider(ABC):
    """Abstract analysis provider.

    Concrete implementations must report :meth:`capabilities` (static contract)
    and a live :meth:`probe` (runtime health). A provider-specific ``analyze``
    method (not part of this ABC) consumes an image and returns an
    :class:`~curator.analysis.schema.AnalysisResult`.
    """

    @abstractmethod
    def capabilities(self) -> AnalysisCapabilities:
        """Return this provider's declared capabilities."""

    @abstractmethod
    def probe(self) -> ComputeProbe:
        """Return a live health probe for the current backend."""


def capability_requirements(
    capabilities: AnalysisCapabilities,
    profile: AnalysisProfile,
    backend: ComputeBackend,
) -> bool:
    """Return True if *capabilities* cover the requested *profile*/*backend*.

    A provider satisfies the requirement when it advertises the exact requested
    backend and either the requested profile or, for ``CUSTOM``, any set of
    known stages. Returns ``False`` (never raises) so callers can branch on
    capability gaps before dispatching a run.
    """
    if backend not in capabilities.backends:
        return False
    if profile is AnalysisProfile.CUSTOM:
        return bool(capabilities.stages)
    return profile in capabilities.profiles
