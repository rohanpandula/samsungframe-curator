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


class ConsolidationError(CuratorError):
    """Raised when a consolidation (SSD execute/archive) step fails.

    Covers the R002 failure surface: staging a source that is not a directory,
    a staged-to-source SHA-256 verification mismatch, trying to archive a source
    folder before every file reached ``promoted``, or archiving a folder that has
    already been archived.
    """


class ApprovalError(CuratorError):
    """Raised when an approval/decision operation is invalid.

    Covers unknown catalog entry ids and undo/redo with nothing to revert or
    re-apply for the given entry.
    """
