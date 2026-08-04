"""Art Direction Manifest subsystem (M002/S03).

Carries a human's art-direction intent for one curated set — layout treatment,
background, processing intent, and per-target overrides — as a versioned,
JSON-serializable :class:`~curator.artdirection.manifest.ArtDirectionManifest`.
Also exposes the deterministic policy engine (:mod:`curator.artdirection.policy`)
that ranks treatments and materializes manifests from analysis signals.
"""

from curator.artdirection.manifest import (
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
    ManifestError,
    ProcessingIntent,
    SourceRegion,
)
from curator.artdirection.policy import (
    ArtDirectionRequest,
    TreatmentProposal,
    materialize_manifest,
    propose,
    propose_treatments,
)

__all__ = [
    "ArtDirectionManifest",
    "ArtDirectionRequest",
    "BackgroundSpec",
    "LayoutTreatment",
    "ManifestError",
    "ProcessingIntent",
    "SourceRegion",
    "TreatmentProposal",
    "materialize_manifest",
    "propose",
    "propose_treatments",
]
