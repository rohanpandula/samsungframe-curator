"""Art Direction Manifest subsystem (M002/S03).

Carries a human's art-direction intent for one curated set — layout treatment,
background, processing intent, and per-target overrides — as a versioned,
JSON-serializable :class:`~curator.artdirection.manifest.ArtDirectionManifest`.
Also exposes the deterministic policy engine (:mod:`curator.artdirection.policy`)
that ranks treatments and materializes manifests from analysis signals, and the
pure geometry layer (:mod:`curator.artdirection.packing`, M010/S01) that turns
N sources into real output-canvas-pixel cells.
"""

from curator.artdirection.manifest import (
    MAX_LAYOUT_SOURCES,
    MULTI_CELL_TREATMENTS,
    ArtDirectionManifest,
    BackgroundSpec,
    LayoutTreatment,
    ManifestError,
    ProcessingIntent,
    SourceRegion,
)
from curator.artdirection.packing import (
    Cell,
    PackingError,
    WeightedSource,
    equal_cells,
    gutter_for_target,
    resolve_regions,
    slice_cells,
)
from curator.artdirection.policy import (
    ArtDirectionRequest,
    TreatmentProposal,
    materialize_manifest,
    propose,
    propose_treatments,
)

__all__ = [
    "MAX_LAYOUT_SOURCES",
    "MULTI_CELL_TREATMENTS",
    "ArtDirectionManifest",
    "ArtDirectionRequest",
    "BackgroundSpec",
    "Cell",
    "LayoutTreatment",
    "ManifestError",
    "PackingError",
    "ProcessingIntent",
    "SourceRegion",
    "TreatmentProposal",
    "WeightedSource",
    "equal_cells",
    "gutter_for_target",
    "materialize_manifest",
    "propose",
    "propose_treatments",
    "resolve_regions",
    "slice_cells",
]
