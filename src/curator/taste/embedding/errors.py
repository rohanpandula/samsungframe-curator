"""Typed errors for the embedding subsystem (M009/S02).

All embedding-specific failures derive from :class:`EmbeddingError`, which in turn
derives from the repo-wide :class:`~curator.errors.CuratorError` so callers can
catch a single base. :class:`EmbeddingUnavailableError` is raised whenever the
provider cannot honestly produce a vector (model absent, checksum mismatch, or a
corrupt/unloadable file) — it never falls back to a silent zero vector.
:class:`EmbeddingVersionError` guards every cross-``model_version`` comparison,
mirroring :class:`~curator.analysis.errors.SchemaVersionError`'s "reject rather
than silently mismatch" posture.
"""

from __future__ import annotations

from curator.errors import CuratorError


class EmbeddingError(CuratorError):
    """Base class for all errors raised by the embedding subsystem."""


class EmbeddingUnavailableError(EmbeddingError):
    """Raised when the embedding provider cannot produce a vector.

    Covers a missing model file, a checksum mismatch against a pinned
    ``CURATOR_TASTE_EMBEDDING_MODEL_SHA256``, or a corrupt/unloadable ONNX file —
    never silently returns a zero vector in place of a real one.
    """


class EmbeddingVersionError(EmbeddingError):
    """Raised when comparing embeddings computed under different model versions.

    A vector's numbers are only meaningful relative to the checkpoint that
    produced them; comparing across ``model_version`` values would return a
    plausible-looking but meaningless float if this guard did not exist.
    """
