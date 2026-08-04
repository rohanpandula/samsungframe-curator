"""Legacy Samsung SSD working-folder discovery (M006/S04).

The :class:`LegacyReader` inspects a legacy Samsung Frame SSD working folder
**read-only** and deterministically inventories its contents into five
observable categories so an operator can migrate them into the catalog:

- ``panels``       — rendered/art image files whose decoded pixel dimensions match
                     a Samsung Frame panel (1920x1080 / 3840x2160).
- ``manifests``    — Samsung Frame manifests: ``.json`` files recognizable by
                     filename convention or by known content keys
                     (``panel``/``art``/``samsung``/``manifest``/``resolution``).
- ``relationships``— source-to-output mapping sidecars: ``.json`` files that map a
                     source to a rendered output (content carries both a ``source``
                     key and an output-like key).
- ``devices``      — device/config files naming ``device_id``/``serial`` or with a
                     ``device``-like filename.
- ``rotation``     — rotation/playlist state files (filename or content naming
                     ``rotation``/``playlist``).

Every discovery is heuristic and content-driven so it behaves deterministically
on any legacy folder. Categories are independent flags — a single file may belong
to more than one (mirroring the :mod:`curator.consolidate.plan` panels/missing-date
overlap posture). The reader only ever *reads*; it never writes to the legacy
folder (the folder is treated as a read-only input).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curator.connectors.local import is_supported_suffix
from curator.consolidate.plan import PANEL_DIMENSIONS
from curator.hashing import sha256_hex
from curator.ingest.decode import DecodeError, decode_image

#: JSON filename conventions / content keys signaling a Samsung Frame manifest.
MANIFEST_KEYS: frozenset[str] = frozenset(
    {"panel", "art", "samsung", "manifest", "resolution"}
)

#: Content keys signaling a source->output relationship mapping (both sides).
RELATIONSHIP_SOURCE_KEY = "source"
RELATIONSHIP_OUTPUT_KEYS: frozenset[str] = frozenset(
    {"output", "rendered", "target"}
)

#: Content keys / filename signals for a device/config file.
DEVICE_CONTENT_KEYS: frozenset[str] = frozenset({"device", "device_id", "serial"})
DEVICE_FILENAME_MARKERS: frozenset[str] = frozenset({"device", "config"})

#: Content keys / filename signals for a rotation/playlist file.
ROTATION_CONTENT_KEYS: frozenset[str] = frozenset(
    {"rotation", "playlist", "interval", "schedule"}
)
ROTATION_FILENAME_MARKERS: frozenset[str] = frozenset({"rotation", "playlist"})

#: JSON filename markers shared by manifest / device / rotation recognizers.
_MANIFEST_FILENAME_MARKERS: frozenset[str] = frozenset(
    {"manifest", "metadata", "art"}
)

#: Categories in deterministic discovery/iteration order.
CATEGORIES: tuple[str, ...] = (
    "panels",
    "manifests",
    "relationships",
    "devices",
    "rotation",
)


@dataclass(frozen=True)
class LegacyItem:
    """One discovered artifact: its category, relative path, and content hash.

    ``rel`` is the POSIX path relative to the scanned folder; ``sha256`` is the
    byte identity used as a stable import key so re-importing never duplicates.
    """

    category: str
    rel: str
    sha256: str


@dataclass(frozen=True)
class LegacyInventory:
    """The deterministic discovery result for one legacy folder.

    ``source`` is the resolved absolute path of the scanned folder. Items are
    grouped by category; :meth:`counts` summarizes them as ``{category: count}``.
    """

    source: str
    panels: list[LegacyItem] = field(default_factory=list)
    manifests: list[LegacyItem] = field(default_factory=list)
    relationships: list[LegacyItem] = field(default_factory=list)
    devices: list[LegacyItem] = field(default_factory=list)
    rotation: list[LegacyItem] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        """Return ``{category: item_count}`` for the five discovery categories."""
        return {
            "panels": len(self.panels),
            "manifests": len(self.manifests),
            "relationships": len(self.relationships),
            "devices": len(self.devices),
            "rotation": len(self.rotation),
        }

    def all_items(self) -> list[LegacyItem]:
        """Return every discovered item in deterministic (category-then-path) order."""
        items: list[LegacyItem] = []
        for category in CATEGORIES:
            items.extend(getattr(self, category))
        return items


class LegacyReader:
    """Read-only, deterministic discoverer of a legacy Samsung SSD folder.

    Instantiate with the folder to scan; :meth:`scan` returns a
    :class:`LegacyInventory`. The reader never writes to the folder.
    """

    def __init__(self, folder: Path | str) -> None:
        self.folder = Path(folder)

    def scan(self) -> LegacyInventory:
        """Scan the folder read-only and return its :class:`LegacyInventory`."""
        folder = self.folder
        if not folder.is_dir():
            raise NotADirectoryError(f"migrate source is not a directory: {folder}")

        panels: list[LegacyItem] = []
        manifests: list[LegacyItem] = []
        relationships: list[LegacyItem] = []
        devices: list[LegacyItem] = []
        rotation: list[LegacyItem] = []

        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(folder).as_posix()
            suffix = path.suffix.lower()

            if is_supported_suffix(suffix):
                item = self._classify_image(path, rel)
                if item is not None and item.category == "panels":
                    panels.append(item)
                continue

            if suffix == ".json":
                is_dev, is_man, is_rel, is_rot = self._classify_json(path, rel)
                if is_man:
                    manifests.append(self._item("manifests", path, rel))
                if is_rel:
                    relationships.append(self._item("relationships", path, rel))
                if is_dev:
                    devices.append(self._item("devices", path, rel))
                if is_rot:
                    rotation.append(self._item("rotation", path, rel))

        return LegacyInventory(
            source=str(folder.resolve()),
            panels=panels,
            manifests=manifests,
            relationships=relationships,
            devices=devices,
            rotation=rotation,
        )

    # -- category recognizers ------------------------------------------------

    def _classify_image(self, path: Path, rel: str) -> LegacyItem | None:
        """Return a ``panels`` item when *path* decodes at a Frame resolution."""
        try:
            sig = decode_image(path.read_bytes())
        except (DecodeError, OSError):
            return None
        if (sig.width, sig.height) in PANEL_DIMENSIONS:
            return self._item("panels", path, rel)
        return None

    def _classify_json(
        self, path: Path, rel: str
    ) -> tuple[bool, bool, bool, bool]:
        """Classify a ``.json`` file into (device, manifest, relationship, rotation).

        A file may satisfy several overlapping categories (independent flags).
        Uses filename markers as well as parsed content keys. Content never
        matches if the file is not a JSON object.
        """
        name = path.name.lower()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, False, False, False
        keys = _key_set(data)

        is_device = bool(keys & DEVICE_CONTENT_KEYS) or any(
            marker in name for marker in DEVICE_FILENAME_MARKERS
        )
        is_manifest = bool(keys & MANIFEST_KEYS) or any(
            marker in name for marker in _MANIFEST_FILENAME_MARKERS
        )
        is_relationship = RELATIONSHIP_SOURCE_KEY in keys and bool(
            keys & RELATIONSHIP_OUTPUT_KEYS
        )
        is_rotation = bool(keys & ROTATION_CONTENT_KEYS) or any(
            marker in name for marker in ROTATION_FILENAME_MARKERS
        )
        return is_device, is_manifest, is_relationship, is_rotation

    def _item(self, category: str, path: Path, rel: str) -> LegacyItem:
        """Build a :class:`LegacyItem` for a file (hashing its bytes)."""
        return LegacyItem(
            category=category,
            rel=rel,
            sha256=_sha256(path),
        )


def _key_set(data: Any) -> set[str]:
    """Return the lowercased top-level keys of a JSON object (empty for non-objects)."""
    if not isinstance(data, dict):
        return set()
    return {str(key).lower() for key in data}


def _sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path*'s bytes."""
    return sha256_hex(path.read_bytes())
