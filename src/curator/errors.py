"""Typed error hierarchy for the Curator package.

Every failure raised by curator code derives from :class:`CuratorError` so callers
can catch a single base type, or narrow to the specific subsystem responsible.
"""

from __future__ import annotations


class CuratorError(Exception):
    """Base class for all Curator errors."""


class CatalogError(CuratorError):
    """Raised when a catalog (system of record) operation fails."""


class StorageError(CuratorError):
    """Raised when content-addressed storage fails or a blob is missing."""


class ConnectorError(CuratorError):
    """Raised when a source connector fails to enumerate or read an asset."""


class IngestError(CuratorError):
    """Raised when an ingest pipeline step fails."""
