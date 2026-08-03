"""Consolidation subsystem — inventory, planning, and execution (S03 / R002).

This package delivers the non-destructive SSD consolidation workflow. The
:mod:`curator.consolidate.plan` module builds a :class:`ConsolidationPlan` from a
**direct directory scan** of a legacy source folder — grouping every file into 8
observable categories (exact dupes, near dupes, higher-res originals, filename
collisions, panels, sidecars, corrupt, missing-date) — reusing S02's ingest
primitives (:func:`~curator.ingest.decode.decode_image`,
:class:`~curator.ingest.clustering.ImageItem`,
:func:`~curator.ingest.clustering.cluster_images`) so there is no CV/logic
duplication.

The plan is built from the directory, **not** from the catalog/IngestReport,
because panel dimensions, sidecar pairing, filename collisions, and missing-date
are directory-inventory concepts the S02 catalog does not capture. A later
executor (:mod:`curator.consolidate.executor`, S03-T3) consumes the same scan to
stage/verify/promote files under the canonical library root.
"""

from __future__ import annotations

from curator.consolidate.executor import ConsolidationExecutor, ConsolidationResult
from curator.consolidate.plan import (
    PANEL_DIMENSIONS,
    SIDECAR_SUFFIXES,
    ConsolidationPlan,
    build_plan,
)

__all__ = [
    "PANEL_DIMENSIONS",
    "SIDECAR_SUFFIXES",
    "ConsolidationPlan",
    "build_plan",
    "ConsolidationExecutor",
    "ConsolidationResult",
]
