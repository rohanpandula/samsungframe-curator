"""Ingest subsystem — decode, cluster, and report (S02).

This package contains the pure algorithmic core (clustering) plus the
orchestration (pipeline, decode) and reporting surfaces that turn a
:class:`~curator.connectors.base.SourceConnector`'s enumerated assets into
clustered, cataloged entries. :mod:`curator.ingest.clustering` is deliberately
pure and stateless so S03's consolidation grouping can reuse it without any
storage or I/O coupling.
"""

from __future__ import annotations

from curator.ingest.clustering import (
    CROP_AR_TOLERANCE,
    PHASH_NEAR_THRESHOLD,
    Cluster,
    ImageItem,
    best_original,
    cluster_images,
    hamming_distance,
)

__all__ = [
    "CROP_AR_TOLERANCE",
    "PHASH_NEAR_THRESHOLD",
    "Cluster",
    "ImageItem",
    "best_original",
    "cluster_images",
    "hamming_distance",
]
