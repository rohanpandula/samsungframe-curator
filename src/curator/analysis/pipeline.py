"""Analysis orchestration pipeline (M002/S02).

:class:`AnalysisPipeline` drives a provider (:class:`AnalysisProvider`) over a
set of assets that are **already cataloged** (entry ids are supplied by the
caller, which is responsible for mapping sources to catalog entries). It never
re-ingests. For every asset it writes an append-only row to ``analysis_results``:
``ok`` rows store the full JSON analysis, while an :class:`AnalysisError` (corrupt
or undecodable input) is recorded as a ``corrupt`` row with an actionable reason
rather than dropped. The run never aborts on a corrupt file.

History-preserving posture: re-running the pipeline appends new rows rather than
deleting or bespoke-deduping them, so every run's outcome is observable. The
``AnalysisRunReport`` counts ok/corrupt/error results and is fully
JSON-serializable (``to_dict`` / ``from_dict``), mirroring :class:`IngestReport`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from curator.analysis.errors import AnalysisError
from curator.analysis.local import LocalAnalysisProvider
from curator.analysis.profiles import AnalysisProfile
from curator.analysis.schema import AnalysisResult
from curator.catalog import Catalog

#: The only statuses the ``analysis_results.status`` column may hold.
ANALYSIS_STATUS_OK = "ok"
ANALYSIS_STATUS_CORRUPT = "corrupt"


@dataclass(frozen=True)
class AnalysisAsset:
    """A lightweight descriptor of an asset to analyze.

    The caller maps a source (path or encoded bytes) to its catalog entry id;
    the pipeline assumes the asset is already cataloged and does not verify it.
    """

    entry_id: int
    source: str | bytes


@dataclass(frozen=True)
class AnalysisRunEntry:
    """One asset's outcome in an analysis run (part of :class:`AnalysisRunReport`).

    ``status`` is ``ok``, ``corrupt``, or ``error``; ``reason`` carries the
    actionable detail for ``corrupt``/``error`` results. ``summary`` is a compact
    signal excerpt for ``ok`` results (never the full JSON).
    """

    entry_id: int
    status: str
    reason: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisRunReport:
    """JSON-serializable result of one :class:`AnalysisPipeline` run."""

    profile: str
    provider_version: str
    total_assets: int = 0
    analyzed_count: int = 0
    corrupt_count: int = 0
    error_count: int = 0
    entries: list[AnalysisRunEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict (nested dataclasses expanded)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Return this report serialized as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisRunReport:
        """Build an :class:`AnalysisRunReport` from a serialized dict."""
        entries = [
            AnalysisRunEntry(**entry)
            for entry in data.get("entries", [])
        ]
        return cls(
            profile=data["profile"],
            provider_version=data["provider_version"],
            total_assets=data.get("total_assets", 0),
            analyzed_count=data.get("analyzed_count", 0),
            corrupt_count=data.get("corrupt_count", 0),
            error_count=data.get("error_count", 0),
            entries=entries,
        )


class AnalysisPipeline:
    """Orchestrate a provider run over already-cataloged assets.

    ``provider`` defaults to :class:`LocalAnalysisProvider`; ``profile`` defaults
    to :class:`AnalysisProfile.BALANCED`. Assets are supplied to :meth:`run` as
    :class:`AnalysisAsset` values (their catalog entry id + source bytes/path).
    """

    def __init__(
        self,
        catalog: Catalog,
        provider: Any | None = None,
        profile: AnalysisProfile | None = None,
    ) -> None:
        self.catalog = catalog
        self.provider = provider if provider is not None else LocalAnalysisProvider()
        self.profile = profile if profile is not None else AnalysisProfile.BALANCED

    @property
    def provider_version(self) -> str:
        """The provider's engine version, when exposed; else ``"unknown"``."""
        return str(getattr(self.provider, "engine_version", "unknown"))

    def run(
        self,
        assets: Sequence[AnalysisAsset],
        profile: AnalysisProfile | None = None,
    ) -> AnalysisRunReport:
        """Analyze *assets*, writing append-only rows and returning a report.

        ``AnalysisError`` from the provider is recorded as a ``corrupt`` row
        (never aborts). An unexpected exception is counted as an ``error`` and the
        run still continues. All writes are committed at the end of the run.
        """
        profile = profile if profile is not None else self.profile
        report = AnalysisRunReport(
            profile=profile.value,
            provider_version=self.provider_version,
            total_assets=len(assets),
        )

        for asset in assets:
            try:
                asset_id = f"entry-{asset.entry_id}"
                result = self.provider.analyze(asset.source, profile, asset_id=asset_id)
            except AnalysisError as exc:
                reason = str(exc)
                self._record_corrupt(asset.entry_id, profile, reason)
                report.corrupt_count += 1
                report.entries.append(
                    AnalysisRunEntry(
                        entry_id=asset.entry_id,
                        status=ANALYSIS_STATUS_CORRUPT,
                        reason=reason,
                    )
                )
            except Exception as exc:  # unexpected — still continue, never abort
                reason = str(exc)
                report.error_count += 1
                report.entries.append(
                    AnalysisRunEntry(entry_id=asset.entry_id, status="error", reason=reason)
                )
            else:
                analysis_json = json.dumps(result.to_dict())
                self._record_ok(asset.entry_id, profile, analysis_json)
                report.analyzed_count += 1
                report.entries.append(
                    AnalysisRunEntry(
                        entry_id=asset.entry_id,
                        status=ANALYSIS_STATUS_OK,
                        summary=_summarize(result),
                    )
                )

        self.catalog.db.commit()
        return report

    # -- persistence ------------------------------------------------------------

    def _record_ok(
        self,
        entry_id: int,
        profile: AnalysisProfile,
        analysis_json: str,
    ) -> None:
        """Insert an ``ok`` row for *entry_id* with the serialized analysis."""
        self.catalog.db.execute(
            "INSERT INTO analysis_results"
            " (catalog_entry_id, profile, engine_version, analysis_json, status)"
            " VALUES (?, ?, ?, ?, ?)",
            (entry_id, profile.value, self.provider_version, analysis_json, ANALYSIS_STATUS_OK),
        )

    def _record_corrupt(
        self,
        entry_id: int,
        profile: AnalysisProfile,
        reason: str,
    ) -> None:
        """Insert a ``corrupt`` row for *entry_id* with an actionable *reason*."""
        self.catalog.db.execute(
            "INSERT INTO analysis_results"
            " (catalog_entry_id, profile, engine_version, analysis_json, status, corrupt_reason)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                entry_id,
                profile.value,
                self.provider_version,
                "{}",
                ANALYSIS_STATUS_CORRUPT,
                reason,
            ),
        )


def _summarize(result: AnalysisResult) -> dict[str, Any]:
    """Return a compact signal excerpt from an ok analysis result."""
    return {
        "asset_id": result.asset_id,
        "schema_version": result.schema_version,
        "technical_quality": result.quality.technical_quality,
        "aesthetic_quality": result.quality.aesthetic_quality,
        "sharpness": result.quality.sharpness,
        "resolution_sufficient": result.quality.resolution_sufficient,
        "colorfulness": result.color_story.colorfulness,
    }
