"""Typed errors for the analysis subsystem (M002).

All analysis-specific failures derive from :class:`AnalysisError`, which in turn
derives from the repo-wide :class:`~curator.errors.CuratorError` so callers can
catch a single base. :class:`SchemaVersionError` guards the versioned
:mod:`curator.analysis.schema` round-trip; :class:`ComputeBackendError` is raised
by strict-device backend resolution in :mod:`curator.analysis.compute`.
"""

from __future__ import annotations

from curator.errors import CuratorError


class AnalysisError(CuratorError):
    """Base class for all errors raised by the analysis subsystem."""


class SchemaVersionError(AnalysisError):
    """Raised when deserializing an analysis result with an unknown schema version."""


class ComputeBackendError(AnalysisError):
    """Raised when strict-device resolution demands an unavailable backend."""


class CatalogEntryNotFound(AnalysisError):
    """Raised when a source cannot be mapped to an existing catalog entry."""
