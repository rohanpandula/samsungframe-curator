"""Tests for the migration subsystem (M006/S04: legacy Samsung SSD import).

Builds a small synthetic legacy Samsung SSD working folder entirely in-process
with Pillow (panels) plus manifest / relationship / device / rotation ``.json``
files, then pins the discovery counts and exercises the non-destructive,
idempotent, restart-safe import path:

  * panels       : 1 x 1920x1080 rendered panel        -> 1
  * manifests    : 1 x Samsung Frame manifest          -> 1
  * relationships: 1 x source->output mapping          -> 1
  * devices      : 1 x device/config file              -> 1
  * rotation     : 1 x rotation/playlist file          -> 1
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

from curator.catalog import Catalog
from curator.migrate import (
    LegacyReader,
    MigrationPlan,
    MigrationReport,
    MigrationService,
    build_plan,
)


def _panel_image() -> Image.Image:
    """A smooth, band-limited 1920x1080 image (Samsung Frame panel)."""
    img = Image.new("RGB", (1920, 1080), (100, 100, 100))
    ImageDraw.Draw(img).ellipse([300, 220, 1620, 860], fill=(150, 150, 150))
    return img.filter(ImageFilter.GaussianBlur(40))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def build_legacy_fixture(root: Path) -> Path:
    """Create a deterministic synthetic legacy folder; return its path."""
    src = root / "legacy-ssd"
    src.mkdir(parents=True, exist_ok=True)
    _panel_image().save(str(src / "panel_01.jpg"), format="JPEG")
    _write_json(
        src / "art_manifest.json",
        {"panel": "1920x1080", "art": "sunset", "samsung": True},
    )
    _write_json(
        src / "render_mapping.json",
        {"source": "IMG_001.jpg", "output": "panel_001.jpg"},
    )
    _write_json(src / "device.json", {"device_id": "frame-123", "serial": "ABC"})
    _write_json(
        src / "rotation_playlist.json",
        {"rotation": "interval", "playlist": ["a", "b"], "interval_seconds": 3600},
    )
    return src


def _snapshot(folder: Path) -> dict[str, bytes]:
    """Snapshot all files under *folder* as ``{relative_path: bytes}``."""
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# reader: discovery + read-only
# ---------------------------------------------------------------------------


def test_scan_discovers_all_categories(tmp_path):
    src = build_legacy_fixture(tmp_path)
    inventory = LegacyReader(src).scan()

    assert inventory.source == str(src.resolve())

    panels = {i.rel for i in inventory.panels}
    manifests = {i.rel for i in inventory.manifests}
    relationships = {i.rel for i in inventory.relationships}
    devices = {i.rel for i in inventory.devices}
    rotation = {i.rel for i in inventory.rotation}

    assert panels == {"panel_01.jpg"}
    assert manifests == {"art_manifest.json"}
    assert relationships == {"render_mapping.json"}
    assert devices == {"device.json"}
    assert rotation == {"rotation_playlist.json"}

    assert inventory.counts() == {
        "panels": 1,
        "manifests": 1,
        "relationships": 1,
        "devices": 1,
        "rotation": 1,
    }


def test_scan_is_read_only(tmp_path):
    src = build_legacy_fixture(tmp_path)
    before = _snapshot(src)
    LegacyReader(src).scan()
    assert _snapshot(src) == before


# ---------------------------------------------------------------------------
# plan: counts + round-trip + dry_run flag
# ---------------------------------------------------------------------------


def test_build_plan_discovered_counts(tmp_path):
    src = build_legacy_fixture(tmp_path)
    plan = build_plan(src)
    assert plan.source == str(src.resolve())
    assert plan.discovered == {
        "panels": 1,
        "manifests": 1,
        "relationships": 1,
        "devices": 1,
        "rotation": 1,
    }
    assert plan.dry_run is False


def test_plan_round_trips(tmp_path):
    src = build_legacy_fixture(tmp_path)
    plan = build_plan(src)
    restored = MigrationPlan.from_dict(plan.to_dict())
    assert restored.source == plan.source
    assert restored.discovered == plan.discovered
    assert restored.dry_run == plan.dry_run


def test_build_plan_rejects_non_directory(tmp_path):
    with pytest.raises(Exception):
        build_plan(tmp_path / "missing")
    with pytest.raises(Exception):
        LegacyReader(tmp_path / "missing").scan()


# ---------------------------------------------------------------------------
# service: backup + dry_run + idempotent/restart-safe import
# ---------------------------------------------------------------------------


def _service(data_root: Path) -> tuple[MigrationService, Catalog]:
    catalog = Catalog(data_root=data_root)
    return MigrationService(catalog=catalog, data_root=data_root), catalog


def test_backup_creates_backup_before_import(tmp_path, data_root):
    src = build_legacy_fixture(tmp_path)
    service, catalog = _service(data_root)
    report = service.migrate(src, dry_run=False)

    backups = list(data_root.glob("*.backup"))
    assert report.backup_created is True
    assert report.backup_path is not None
    assert len(backups) == 1
    assert Path(report.backup_path) == backups[0]
    # Catalog rows were imported after the backup existed.
    assert catalog.count_catalog_entries() == 5


def test_import_idempotent_no_duplicates(tmp_path, data_root):
    src = build_legacy_fixture(tmp_path)
    service, catalog = _service(data_root)
    plan = build_plan(src)

    first = service.import_migration(plan)
    assert first.imported == 5
    assert catalog.count_catalog_entries() == 5

    second = service.import_migration(plan)
    assert second.imported == 0
    assert second.skipped == 5
    assert catalog.count_catalog_entries() == 5  # no duplicates


def test_import_restart_safe_resumes_without_duplicating(tmp_path, data_root):
    src = build_legacy_fixture(tmp_path)
    service, catalog = _service(data_root)
    plan = build_plan(src)

    calls = {"n": 0}
    original = catalog.add_source

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("injected interruption")
        return original(*args, **kwargs)

    catalog.add_source = flaky
    with pytest.raises(RuntimeError):
        service.import_migration(plan)
    catalog.add_source = original

    # Resume: prior fully-imported items are skipped, the interrupted one and the
    # rest complete, and nothing duplicated.
    resumed = service.import_migration(plan)
    assert resumed.skipped == 2  # two items already checkpointed promoted
    assert resumed.imported == 3
    assert catalog.count_catalog_entries() == 5


def test_dry_run_imports_nothing_and_no_backup(tmp_path, data_root):
    src = build_legacy_fixture(tmp_path)
    service, catalog = _service(data_root)

    before = _snapshot(src)
    report = service.migrate(src, dry_run=True)

    assert report.dry_run is True
    assert report.imported == 0
    assert report.backup_created is False
    assert catalog.count_catalog_entries() == 0
    assert list(data_root.glob("*.backup")) == []
    assert _snapshot(src) == before  # sources untouched


def test_sources_untouched_by_real_import(tmp_path, data_root):
    src = build_legacy_fixture(tmp_path)
    service, _ = _service(data_root)
    before = _snapshot(src)
    service.migrate(src, dry_run=False)
    assert _snapshot(src) == before


def test_rollback_limitations_documented(tmp_path, data_root):
    src = build_legacy_fixture(tmp_path)
    service, _ = _service(data_root)
    report = service.migrate(src, dry_run=True)

    limitations = service.rollback_limitations()
    assert len(limitations) >= 3
    assert limitations == report.rollback_limitations
    assert any("observation" in s for s in limitations)
    assert any("snapshot" in s for s in limitations)
    assert all(isinstance(s, str) and s for s in limitations)


def test_report_round_trips(tmp_path, data_root):
    src = build_legacy_fixture(tmp_path)
    service, _ = _service(data_root)
    report = service.migrate(src, dry_run=False)
    restored = MigrationReport.from_dict(report.to_dict())
    assert restored.source == report.source
    assert restored.dry_run == report.dry_run
    assert restored.discovered == report.discovered
    assert restored.imported == report.imported
    assert restored.backup_created is report.backup_created
    assert restored.rollback_limitations == report.rollback_limitations


def test_migrate_report_is_json_serializable(tmp_path, data_root):
    src = build_legacy_fixture(tmp_path)
    service, _ = _service(data_root)
    report = service.migrate(src, dry_run=False)
    json.dumps(report.to_dict())
    # The plan is JSON-serializable too.
    json.dumps(build_plan(src).to_dict())
