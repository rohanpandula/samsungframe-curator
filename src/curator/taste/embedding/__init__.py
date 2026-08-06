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
"""

from __future__ import annotations

from curator.taste.embedding.errors import (
    EmbeddingError,
    EmbeddingUnavailableError,
    EmbeddingVersionError,
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

__all__ = [
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL_VERSION",
    "EmbeddingCapabilities",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingUnavailableError",
    "EmbeddingVersionError",
    "OnnxEmbeddingProvider",
    "resolve_expected_sha256",
    "resolve_model_path",
]
