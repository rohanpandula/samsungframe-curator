"""JSON-serializable ingest report (S02 / S04-S05 surface).

:class:`IngestReport` is the deliverable of one :class:`IngestPipeline` run. It is
fully JSON-serializable (via :meth:`IngestReport.to_dict` / :meth:`to_json`) so
S04/S05 can persist or display it without bespoke plumbing.

Counts are self-consistent:

- ``total_enumerated``  — assets the connector surfaced this run.
- ``indexed_count``     — assets successfully read + decoded.
- ``unique_clusters``   — distinct dedup clusters == ``exact + near + singles``.
- ``exact_clusters``    — clusters whose members all share one content sha256
                          (identical bytes; a single canonical entry survives).
- ``near_clusters``     — clusters with more than one distinct sha256 (resized /
                          crop / near-edit merged by perceptual hash).

Per-file outcomes (``failures``) capture the explicit-unsupported surface R003
requires: a recognized RAW file is reported with ``status="unsupported"`` and a
corrupt/unreadable file with ``status="corrupt"/"error"`` and its preserved error
text — nothing silently disappears.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReportEntry:
    """One cataloged, clustered entry in an ingest run.

    ``best_original`` is True for the cluster's highest-resolution member;
    ``dupe_of`` names that canonical member's asset_id for every dupe.
    ``phash_distance`` is the Hamming distance from this member's phash to the
    cluster best's phash (0 for the best, ``None`` when unknown).
    """

    asset_id: str
    connector_id: str
    sha256: str
    cluster_id: str | None
    best_original: bool = False
    dupe_of: str | None = None
    width: int | None = None
    height: int | None = None
    phash: str | None = None
    phash_distance: int | None = None


@dataclass(frozen=True)
class ReportIssue:
    """A file that was not indexed, with its explicit reason preserved.

    ``status`` is one of ``"unsupported"`` (recognized RAW, R003), ``"corrupt"``
    (bytes present but did not decode), or ``"error"`` (read failure).
    """

    status: str
    asset_id: str
    connector_id: str
    media_type: str | None = None
    error: str | None = None


@dataclass
class IngestReport:
    """Structured, JSON-serializable result of an ingest run."""

    connector_id: str
    total_enumerated: int = 0
    indexed_count: int = 0
    unique_clusters: int = 0
    exact_clusters: int = 0
    near_clusters: int = 0
    unsupported_count: int = 0
    corrupt_count: int = 0
    error_count: int = 0
    entries: list[ReportEntry] = field(default_factory=list)
    failures: list[ReportIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict (nested dataclasses expanded)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return this report serialized as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
