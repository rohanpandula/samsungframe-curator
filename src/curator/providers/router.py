"""Hybrid local/cloud analysis provider router (M006/S01, R017).

:class:`HybridRouter` maps analysis *kinds* to a provider — by default every
derivative/quantitative kind stays local (air-gapped) while only semantic,
composition, pairing, and taste ride the cloud (R017). If a cloud call raises
:class:`ProviderOutageError`, the router degrades that single call to local-only
(reporting a pause) and never lets a cloud outage harm approved/local work.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from curator.analysis.profiles import AnalysisProfile
from curator.analysis.provider import AnalysisProvider
from curator.providers.cloud import ProviderOutageError

#: Kinds that stay on the local (air-gapped) provider by default.
LOCAL_KINDS = frozenset(
    {"duplicate_detection", "technical", "embeddings", "saliency", "color_story"}
)

#: Kinds that ride the cloud provider by default (R017 hybrid split).
CLOUD_KINDS = frozenset({"semantic", "composition", "pairing", "taste"})

#: Default routing policy: kind -> provider role (``"local"`` / ``"cloud"``).
DEFAULT_ROUTING: dict[str, str] = {
    kind: "local" for kind in LOCAL_KINDS
}
DEFAULT_ROUTING.update({kind: "cloud" for kind in CLOUD_KINDS})


@dataclass
class Pause:
    """A reported degradation event (cloud outage -> local fallback)."""

    kind: str
    provider: str = "cloud"
    reason: str = "ProviderOutageError"
    degraded_to: str = "local"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "provider": self.provider,
            "reason": self.reason,
            "degraded_to": self.degraded_to,
        }


class HybridRouter:
    """Routes analysis kinds across a local and a cloud provider."""

    def __init__(
        self,
        local_provider: AnalysisProvider,
        cloud_provider: AnalysisProvider,
        policy: Mapping[str, str] | None = None,
    ) -> None:
        self.local_provider = local_provider
        self.cloud_provider = cloud_provider
        self.policy: dict[str, str] = dict(DEFAULT_ROUTING)
        if policy:
            self.policy.update(policy)
        self.pauses: list[Pause] = []

    def route(self, kind: str) -> AnalysisProvider:
        """Return the provider for *kind* (local is the safe default)."""
        role = self.policy.get(kind, "local")
        return self.cloud_provider if role == "cloud" else self.local_provider

    @property
    def pause_count(self) -> int:
        """Number of cloud-outage degradation events reported."""
        return len(self.pauses)

    def run(
        self,
        kind: str,
        source_bytes: bytes | bytearray,
        profile: AnalysisProfile = AnalysisProfile.BALANCED,
        asset_id: str | None = None,
        source_id: str | None = None,
        allowed_components: list[str] | set[str] | None = None,
    ) -> Any:
        """Execute *kind* on the routed provider.

        A cloud :class:`ProviderOutageError` degrades that single call to the local
        provider (reported as a pause) and is never raised to the caller.
        """
        provider = self.route(kind)
        if provider is self.cloud_provider:
            try:
                return self.cloud_provider.analyze(  # type: ignore[attr-defined]
                    source_bytes,
                    profile,
                    asset_id,
                    source_id,
                    allowed_components,
                )
            except ProviderOutageError:
                self.pauses.append(Pause(kind=kind))
                return self.local_provider.analyze(  # type: ignore[attr-defined]
                    source_bytes,
                    profile,
                    asset_id,
                )
        return provider.analyze(  # type: ignore[attr-defined]
            source_bytes,
            profile,
            asset_id,
        )


# Re-export for a single convenient import surface.
__all__ = [
    "CLOUD_KINDS",
    "DEFAULT_ROUTING",
    "HybridRouter",
    "LOCAL_KINDS",
    "Pause",
]
