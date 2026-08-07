"""Local embedding provider + storage subsystem (M009/S02).

A CPU-only, offline, deterministic image-embedding provider slotted into the
existing :class:`~curator.analysis.provider.AnalysisProvider`/
:class:`~curator.analysis.compute.ComputeBackend` seam (mirrored, not rebuilt —
see :mod:`curator.taste.embedding.provider`), plus content-scoped,
model-versioned vector storage (:mod:`curator.taste.embedding.store`). The model
file is never downloaded at request time: :func:`resolve_model_path` only ever
resolves a fixed local cache path, and :meth:`OnnxEmbeddingProvider.probe`
reports honestly when nothing is placed there yet — the milestone's documented
early-exit checkpoint (``curator taste embed-status``).

M010/S04 adds :mod:`curator.taste.embedding.grouping` alongside it: bounded-pool
selection of *which* images belong together, kept strictly out of
:mod:`curator.artdirection` so the policy engine's purity survives.
"""

from __future__ import annotations

from curator.taste.embedding.errors import (
    EmbeddingError,
    EmbeddingUnavailableError,
    EmbeddingVersionError,
)
from curator.taste.embedding.grouping import (
    AFFINITY_SOURCE,
    GROUP_SIMILARITY_THRESHOLD,
    MAX_CANDIDATE_POOL,
    GroupCandidate,
    GroupingError,
    GroupSelection,
    resolve_group_pool,
    select_group,
)
from curator.taste.embedding.provider import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_VERSION,
    EmbeddingCapabilities,
    EmbeddingProvider,
    OnnxEmbeddingProvider,
    resolve_expected_sha256,
    resolve_model_path,
)
from curator.taste.embedding.store import EmbeddingStore, StoredEmbedding, cosine_similarity

__all__ = [
    "AFFINITY_SOURCE",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_VERSION",
    "GROUP_SIMILARITY_THRESHOLD",
    "MAX_CANDIDATE_POOL",
    "EmbeddingCapabilities",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingStore",
    "EmbeddingUnavailableError",
    "EmbeddingVersionError",
    "GroupCandidate",
    "GroupSelection",
    "GroupingError",
    "OnnxEmbeddingProvider",
    "StoredEmbedding",
    "cosine_similarity",
    "resolve_expected_sha256",
    "resolve_group_pool",
    "resolve_model_path",
    "select_group",
]
