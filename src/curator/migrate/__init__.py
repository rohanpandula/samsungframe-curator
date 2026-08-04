"""Migration subsystem — legacy Samsung SSD working-folder import (M006/S04).

This package provides a non-destructive migration toolchain for a legacy Samsung
Frame SSD working folder, which is treated as a **read-only input**:

- :mod:`curator.migrate.legacy`  — :class:`LegacyReader` deterministically discovers
  panels, Samsung Frame manifests, source-to-output relationships, device IDs, and
  rotation playlist state.
- :mod:`curator.migrate.plan`    — :class:`MigrationPlan` (frozen, JSON-serializable)
  summarizes the scan as per-category counts plus a ``dry_run`` intent; :func:`build_plan`
  produces one from a folder.
- :mod:`curator.migrate.service` — :class:`MigrationService` backs up the catalog DB
  before any mutation, then imports discovered items **idempotently** and
  **restart-safely** via a durable per-item ``promoted`` checkpoint, and exposes the
  documented rollback limitations.
"""

from __future__ import annotations

from curator.migrate.legacy import CATEGORIES, LegacyInventory, LegacyItem, LegacyReader
from curator.migrate.plan import DISCOVERED_CATEGORIES, MigrationPlan, build_plan
from curator.migrate.service import MigrationReport, MigrationService

__all__ = [
    "CATEGORIES",
    "DISCOVERED_CATEGORIES",
    "LegacyInventory",
    "LegacyItem",
    "LegacyReader",
    "MigrationPlan",
    "MigrationReport",
    "MigrationService",
    "build_plan",
]
