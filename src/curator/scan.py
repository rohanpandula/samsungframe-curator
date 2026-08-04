"""Catalog diff engine — read-only drift detection between a source and the catalog.

``scan`` is the S04 headless "what would need to change" surface: it enumerates a
source connector, reads back the catalog, and produces a :class:`ScanDiff`
classifying every asset as **new** (on disk, not cataloged), **changed** (on disk,
cataloged, but the content digest differs), or **missing** (cataloged but no longer
on disk). The CLI maps ``no_changes`` to its documented exit code 3 so scripts can
branch on "nothing to do" (S04->S05 boundary).

This module is deliberately **pure and read-only**: it calls ``connector.enumerate``
/ ``connector.read_original`` and ``catalog.get_source_asset_ids`` /
``catalog.get_by_source`` — never any write path — so re-scanning is idempotent
(S05 acceptance 2). Classification mirrors what ``IngestPipeline`` would *catalog*:
recognized RAW (explicit-unsupported, R003) and non-decodable (corrupt) files are
excluded, exactly as ingest never gives them a ``catalog_entries`` row. That keeps a
scan immediately after an ingest of an unchanged folder at ``no_changes``.

Diff semantics:

- **new** — enumerated, non-RAW, decodable content with no catalog entry.
- **changed** — cataloged asset whose current on-disk content digest differs from
  its latest stored revision (the catalog stores the content digest as the
  revision, so a changed file surfaces as a digest mismatch).
- **missing** — a cataloged ``asset_id`` the connector no longer enumerates.

``missing`` is computed from *all* enumerated ids (not just catalogable ones): a
file still on disk — even one whose content is now corrupt — is not "missing"; that
is a content change, not a disappearance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from curator.catalog import Catalog
from curator.connectors.base import SourceConnector
from curator.connectors.local import UNSUPPORTED_EXTRA_KEY
from curator.hashing import sha256_hex
from curator.ingest.decode import DecodeError, decode_image


@dataclass
class ScanDiff:
    """A JSON-serializable read-only catalog diff.

    ``new`` / ``changed`` hold per-asset dicts; ``missing`` holds bare asset ids.
    :attr:`no_changes` (a property) drives the CLI's exit-code 3 signal.
    """

    connector_id: str
    new: list[dict[str, Any]] = field(default_factory=list)
    changed: list[dict[str, Any]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def no_changes(self) -> bool:
        """True when there is nothing new, changed, or missing."""
        return not (self.new or self.changed or self.missing)

    def to_dict(self) -> dict[str, Any]:
        """Return the diff as a plain JSON-serializable dict."""
        return {
            "connector_id": self.connector_id,
            "new": self.new,
            "changed": self.changed,
            "missing": self.missing,
            "no_changes": self.no_changes,
        }

    def to_json(self) -> str:
        """Return the diff as pretty-printed JSON."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def scan_connector(connector: SourceConnector, catalog: Catalog) -> ScanDiff:
    """Diff *connector*'s on-disk assets against *catalog*; pure and read-only.

    Enumerates the connector, reads back the catalog for the connector's
    ``connector_id``, and classifies every enumerated asset as new / changed /
    unchanged, plus every cataloged-but-not-enumerated asset as missing. Never
    writes to catalog or journal tables.
    """
    cataloged = catalog.get_source_asset_ids(connector.connector_id)
    diff = ScanDiff(connector_id=connector.connector_id)

    enumerated_ids: set[str] = set()
    for meta in connector.enumerate():
        enumerated_ids.add(meta.asset_id)
        _classify_enumerated(connector, catalog, meta, diff)

    # Anything cataloged but no longer enumerated is missing.
    diff.missing = sorted(cataloged - enumerated_ids)
    return diff


def _classify_enumerated(
    connector: SourceConnector,
    catalog: Catalog,
    meta,
    diff: ScanDiff,
) -> None:
    """Classify one enumerated asset and append it to *diff* if it drifts."""
    # Explicit-unsupported RAW files (R003) are never cataloged by ingest, so they
    # are not drift. Skip them (they cannot be new/changed/missing).
    if meta.extra.get(UNSUPPORTED_EXTRA_KEY):
        return

    data = connector.read_original(meta.asset_id)
    current_digest = sha256_hex(data)

    entry = catalog.get_by_source(connector.connector_id, meta.asset_id)
    if entry is None:
        # No entry yet: a candidate for "new" — but only if it is actually
        # decodable catalog material (a corrupt file is skipped by ingest too).
        try:
            decode_image(data)
        except DecodeError:
            return  # not catalogable -> not drift
        diff.new.append(
            {
                "asset_id": meta.asset_id,
                "revision": current_digest,
                "media_type": meta.media_type,
                "size_bytes": meta.size_bytes,
            }
        )
        return

    stored_revision = entry["revision"]
    if current_digest != stored_revision:
        diff.changed.append(
            {
                "asset_id": meta.asset_id,
                "stored_revision": stored_revision,
                "current_revision": current_digest,
                "media_type": meta.media_type,
            }
        )
    # Otherwise the asset is unchanged: on disk and cataloged with matching bytes.
