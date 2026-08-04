"""Cloud/hybrid provider routing, privacy disclosure, and exclusion policy (M006/S01).

This package adds the privacy-preserving cloud/hybrid analysis surface on top of the
M002 :mod:`curator.analysis` provider contract: a JSON-serializable privacy
:class:`~curator.providers.privacy.Disclosure`, an
:class:`~curator.providers.cloud.ExclusionPolicy` that gates per-source/per-image
payload components, a deterministic mock
:class:`~curator.providers.cloud.SyntheticCloudAnalysisRuntime` that records what it
receives, a :class:`~curator.providers.cloud.CloudAnalysisProvider`, and a
:class:`~curator.providers.router.HybridRouter` that degrades cloud outages to local.
"""

from curator.providers.cloud import (
    CLOUD_STAGES,
    COMPONENT_ALL,
    COMPONENT_FACES,
    COMPONENT_FULL_RESOLUTION,
    COMPONENT_GPS,
    CloudAnalysisProvider,
    ExclusionPolicy,
    ProviderOutageError,
    SyntheticCloudAnalysisRuntime,
)
from curator.providers.privacy import (
    Disclosure,
    MachineLeaves,
    default_disclosure,
)
from curator.providers.router import (
    CLOUD_KINDS,
    DEFAULT_ROUTING,
    LOCAL_KINDS,
    HybridRouter,
    Pause,
)

__all__ = [
    "CLOUD_KINDS",
    "CLOUD_STAGES",
    "COMPONENT_ALL",
    "COMPONENT_FACES",
    "COMPONENT_FULL_RESOLUTION",
    "COMPONENT_GPS",
    "DEFAULT_ROUTING",
    "CloudAnalysisProvider",
    "Disclosure",
    "ExclusionPolicy",
    "HybridRouter",
    "LOCAL_KINDS",
    "MachineLeaves",
    "Pause",
    "ProviderOutageError",
    "SyntheticCloudAnalysisRuntime",
    "default_disclosure",
]
