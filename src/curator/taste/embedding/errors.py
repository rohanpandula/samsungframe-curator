"""Typed errors for the embedding subsystem (M009/S02).

All embedding-specific failures derive from :class:`EmbeddingError`, which in turn
derives from the repo-wide :class:`~curator.errors.CuratorError` so callers can
catch a single base. :class:`EmbeddingUnavailableError` is raised whenever the
provider cannot honestly produce a vector (model absent, checksum mismatch, or a
corrupt/unloadable file) — it never falls back to a silent zero vector.

Cross-``model_version`` safety needs no error class: every read in
:mod:`curator.taste.embedding.store` (``get``/``get_matrix``) is scoped by
``model_version`` at the SQL layer, so a vector from another checkpoint can
never enter a comparison in the first place (M009/M010 audit, 2026-09-02).
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

