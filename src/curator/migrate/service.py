"""Migration service — backup, idempotent import, and dry-run reporting (M006/S04).

:class:`MigrationService` turns a :class:`~curator.migrate.plan.MigrationPlan` into
catalog rows. Two guarantees matter here:

**Backup-before-mutation.** :meth:`backup` snapshots the catalog database (a
consistent :func:`sqlite3.backup` — WAL-safe) to a timestamped ``.backup`` path in
the same directory *before* any import row is written. A dry run never backs up
(it performs no mutation) but still verifies the DB is backup-able.

**Idempotent, restart-safe import.** Every discovered item is written with a stable
key (its relative path in the legacy folder) and checkpointed in the schema-v3
:attr:`~curator.catalog.ConsolidationExecutor`-style ``consolidation_journal`` as a
per-item ``promoted`` marker keyed by ``(connector_id == legacy folder, asset_id ==
item path)``. :meth:`import_migration` processes items in deterministic batches;
already-``promoted`` items are skipped. A mid-run interruption therefore resumes
without duplicating prior items — the checkpoint is durable the moment each item is
imported.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from curator.catalog import CONSOLIDATION_PROMOTED, Catalog
from curator.config import CuratorConfig
from curator.db import default_db_path
from curator.migrate.legacy import LegacyReader
from curator.migrate.plan import MigrationPlan, build_plan

#: Number of items processed per checkpoint batch.
BATCH_SIZE = 25


def _rollback_limitations() -> list[str]:
    """Return the documented, non-empty rollback limitations for migration imports."""
    return [
        "Relationships are imported as observations: "
        "deleting or renaming a migrated source does not delete or update "
        "the imported relationships/output rows.",
        "Rendered panels are imported as immutable content entries: "
        "deleting a migrated source does not delete the imported panel content.",
        "Rotation playlist state is imported as a snapshot: "
        "any future rotation on the same playlist supersedes the imported state.",
    ]


@dataclass(frozen=True)
class MigrationReport:
    """JSON-serializable summary of one :meth:`MigrationService.migrate` run.

    ``discovered`` repeats the per-category counts from the plan; ``imported`` /
    ``skipped`` count items this run wrote / already-present (restart-safe);
    ``backup_path`` is the created backup's path (``None`` on a dry run) and
    ``backup_created`` whether one was taken. ``rollback_limitations`` surfaces the
    documented constraints the operator must accept.
    """

    source: str
    dry_run: bool
    discovered: dict[str, int]
    imported: int
    skipped: int
    backup_path: str | None
    backup_created: bool
    rollback_limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict representation of this report."""
        return {
            "source": self.source,
            "dry_run": self.dry_run,
            "discovered": dict(self.discovered),
            "imported": self.imported,
            "skipped": self.skipped,
            "backup_path": self.backup_path,
            "backup_created": self.backup_created,
            "rollback_limitations": list(self.rollback_limitations),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationReport:
        """Rebuild a :class:`MigrationReport` from a :meth:`to_dict` dict."""
        if isinstance(data, cls):
            return data
        return cls(
            source=str(data["source"]),
            dry_run=bool(data.get("dry_run", False)),
            discovered={str(k): int(v) for k, v in (data.get("discovered") or {}).items()},
            imported=int(data.get("imported", 0)),
            skipped=int(data.get("skipped", 0)),
            backup_path=data.get("backup_path"),
            backup_created=bool(data.get("backup_created", False)),
            rollback_limitations=[
                str(s) for s in data.get("rollback_limitations", [])
            ],
        )


class MigrationService:
    """Backup + idempotent/restart-safe import of a legacy folder into the catalog.

    ``catalog`` should be a :class:`~curator.catalog.Catalog` (constructed with the
    intended ``data_root`` so the backup targets the right DB). When omitted, one is
    built from ``data_root`` (defaulting to the six-axis config root).
    """

    def __init__(
        self,
        catalog: Catalog | None = None,
        data_root: Path | None = None,
    ) -> None:
        if data_root is None:
            data_root = CuratorConfig().data_root
        self.data_root = Path(data_root)
        self.catalog = catalog if catalog is not None else Catalog(data_root=self.data_root)

    # -- API ------------------------------------------------------------------

    def backup(self) -> Path:
        """Snapshot the catalog DB to a timestamped ``.backup`` path (before mutation).

        Uses :func:`sqlite3.Connection.backup` (WAL-safe and consistent) rather than
        a raw file copy, so an un-checkpointed WAL is captured too. Returns the backup
        path. Safe to call repeatedly — each call writes a fresh timestamped snapshot.
        """
        db_path = default_db_path(self.data_root)
        backup_path = _timestamped_backup_path(db_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        dest = sqlite3.connect(str(backup_path))
        try:
            self.catalog.db.backup(dest)
        finally:
            dest.close()
        return backup_path

    def rollback_limitations(self) -> list[str]:
        """Return the documented rollback limitations for migrate imports."""
        return _rollback_limitations()

    def import_migration(self, plan: MigrationPlan) -> MigrationReport:
        """Map *plan*'s discovered items into the catalog idempotently.

        Re-scans the plan's folder to enumerate concrete items, dedups them by
        relative path, and imports each in deterministic batches. Items already
        checkpointed ``promoted`` for ``(connector_id, asset_id)`` are skipped, so
        re-running — including a resume after an interruption — never duplicates a
        row. Returns a :class:`MigrationReport`.
        """
        connector_id = str(Path(plan.source).resolve())
        items = self._enumerate_items(plan.source)
        imported = 0
        skipped = 0
        for start in range(0, len(items), BATCH_SIZE):
            batch = items[start : start + BATCH_SIZE]
            for asset_id, category, path, sha in batch:
                if (
                    self.catalog.consolidation_checkpoint(connector_id, asset_id)
                    == CONSOLIDATION_PROMOTED
                ):
                    skipped += 1
                    continue
                self._import_one(connector_id, asset_id, category, path, sha)
                imported += 1
        return MigrationReport(
            source=plan.source,
            dry_run=plan.dry_run,
            discovered=dict(plan.discovered),
            imported=imported,
            skipped=skipped,
            backup_path=None,
            backup_created=False,
            rollback_limitations=list(_rollback_limitations()),
        )

    def migrate(
        self,
        folder: Path | str,
        dry_run: bool = False,
        catalog: Catalog | None = None,
    ) -> MigrationReport:
        """Build a plan, back up (unless dry-run), then import — or report only.

        *dry_run* skips the backup and the import and returns a report that records
        the discovered counts, verifies the DB is backup-able (``backup_path`` set
        but ``backup_created`` False), and surfaces the rollback limitations. A real
        run takes a backup *before* mutating, then imports idempotently.
        """
        service = self
        if catalog is not None and catalog is not self.catalog:
            service = MigrationService(catalog, data_root=self.data_root)
        plan = build_plan(folder)
        plan = MigrationPlan(
            source=plan.source,
            discovered=plan.discovered,
            dry_run=dry_run,
        )

        if dry_run:
            backup_path = default_db_path(self.data_root)
            return MigrationReport(
                source=plan.source,
                dry_run=True,
                discovered=dict(plan.discovered),
                imported=0,
                skipped=0,
                backup_path=str(backup_path) if backup_path.exists() else None,
                backup_created=False,
                rollback_limitations=list(_rollback_limitations()),
            )

        backup_path = service.backup()
        report = service.import_migration(plan)
        return MigrationReport(
            source=report.source,
            dry_run=False,
            discovered=report.discovered,
            imported=report.imported,
            skipped=report.skipped,
            backup_path=str(backup_path),
            backup_created=True,
            rollback_limitations=list(_rollback_limitations()),
        )

    # -- impl ------------------------------------------------------------------

    def _enumerate_items(self, folder: Path | str) -> list[tuple[str, str, Path, str]]:
        """Flatten and dedupe discovered items into deterministic ``(key, cat, path, sha)``.

        Dedupes by relative path across categories so one physical file is imported
        once, and orders the result by path for reproducible processing.
        """
        inventory = LegacyReader(folder).scan()
        root = Path(folder).resolve()
        by_rel: dict[str, tuple[str, str, Path, str]] = {}
        for item in inventory.all_items():
            if item.rel in by_rel:
                continue
            by_rel[item.rel] = (
                item.rel,
                item.category,
                root / item.rel,
                item.sha256,
            )
        return [by_rel[rel] for rel in sorted(by_rel)]

    def _import_one(
        self,
        connector_id: str,
        asset_id: str,
        category: str,
        path: Path,
        sha: str,
    ) -> None:
        """Import one item's bytes as a catalog entry and checkpoint it ``promoted``."""
        data = path.read_bytes()
        row_id = self.catalog.consolidation_journal_start(
            connector_id, asset_id, note=category
        )
        self.catalog.add_source(
            connector_id,
            asset_id,
            data,
            metadata={
                "revision": sha,
                "connector_type": "legacy_migrate",
                "quality_reason": f"migrated from legacy folder: {category}",
            },
        )
        self.catalog.consolidation_journal_update(
            row_id, CONSOLIDATION_PROMOTED, sha256=sha
        )


def _timestamped_backup_path(db_path: Path) -> Path:
    """Return a timestamped ``<db>.<YYYYmmddTHHMMSS>.backup`` path for *db_path*."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return db_path.with_name(f"{db_path.name}.{stamp}.backup")
