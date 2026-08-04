"""Migration planner — a JSON-serializable snapshot of a legacy-folder scan (M006/S04).

:class:`MigrationPlan` summarizes what a :class:`~curator.migrate.legacy.LegacyReader`
discovered in a legacy Samsung SSD working folder, plus the operator's intent about
whether the import should actually run (``dry_run``). It is deliberately small and
portable: the heavy item detail lives in the inventory returned by the reader, while
the plan is the serializable contract passed across process boundaries / saved for
the record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curator.errors import CuratorError
from curator.migrate.legacy import CATEGORIES, LegacyReader

#: The discovery categories a plan inventories (stable key order for JSON).
DISCOVERED_CATEGORIES: tuple[str, ...] = CATEGORIES


@dataclass(frozen=True)
class MigrationPlan:
    """A frozen, JSON-serializable migration plan for one legacy folder.

    ``source`` is the resolved absolute path of the scanned folder. ``discovered``
    maps each of the five discovery categories to its item count; ``dry_run`` marks
    the plan as a report-only (no import) run.
    """

    source: str
    discovered: dict[str, int] = field(default_factory=dict)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict representation of this plan."""
        return {
            "source": self.source,
            "discovered": dict(self.discovered),
            "dry_run": self.dry_run,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MigrationPlan:
        """Rebuild a :class:`MigrationPlan` from a :meth:`to_dict` dict."""
        if isinstance(data, cls):
            return data
        discovered = data.get("discovered") or {}
        return cls(
            source=str(data["source"]),
            discovered={str(k): int(v) for k, v in discovered.items()},
            dry_run=bool(data.get("dry_run", False)),
        )


def build_plan(folder: Path | str) -> MigrationPlan:
    """Inventory *folder* into a :class:`MigrationPlan` (report-only, ``dry_run`` off).

    Scans *folder* read-only via :class:`LegacyReader` and records the discovered
    per-category counts. Raises :class:`CuratorError` when *folder* is not a
    directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise CuratorError(f"migrate source is not a directory: {folder}")
    inventory = LegacyReader(folder).scan()
    return MigrationPlan(
        source=inventory.source,
        discovered=inventory.counts(),
        dry_run=False,
    )
