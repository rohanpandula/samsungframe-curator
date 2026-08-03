"""Catalog API — the system-of-record access layer (R001).

The :class:`Catalog` is the transactional boundary every ingest path (S02) and the
CLI (T06) call into. It owns the mapping between connector-scoped **source**
identity (``connector_id`` + opaque ``asset_id``) and content-addressed **content**
identity (``sha256``). The two are deliberately distinct (decision D002): a
rename/move/revision in a source never creates duplicate work on the content side
(a ``source_assets`` row is created once per ``(connector_id, asset_id)``), while
identical bytes from any connector converge on a single ``content`` row
(the byte-convergence point).

All mutations are transactional, keyed by the schema's ``UNIQUE(connector_id,
asset_id, revision)`` on ``catalog_entries`` so re-adding the same bytes + the same
(connector, asset, revision) upserts one row (idempotent). Connector-scoped source
identity is guaranteed distinct by the schema's ``UNIQUE(connector_id, asset_id)``
on ``source_assets``.

Failure semantics: storage-layer problems propagate as :class:`StorageError` from
the ContentStore; SQLite/database problems are wrapped and re-raised as
:class:`CatalogError`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from curator import db as _db
from curator.content_store import ContentStore
from curator.errors import CatalogError

# Column order of ``SELECT *`` on catalog_entries (matches schema v1 DDL).
_ENTRY_COLUMNS = [
    "id",
    "connector_id",
    "asset_id",
    "revision",
    "sha256",
    "quality_score",
    "quality_reason",
    "created_at",
    "updated_at",
]

_TIMESTAMP = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"


class Catalog:
    """System-of-record API over a migrated SQLite DB + ContentStore.

    If no connection / store is supplied, one is created from the six-axis config
    (``CURATOR_DATA_ROOT``) and the schema is migrated idempotently to guarantee the
    tables exist before any operation.
    """

    def __init__(
        self,
        conn: sqlite3.Connection | None = None,
        store: ContentStore | None = None,
        data_root: Path | None = None,
    ) -> None:
        if conn is None:
            conn = _db.connect(data_root)
        if store is None:
            store = ContentStore(data_root)
        self.db = conn
        self.content = store
        # Idempotent — safe even when the caller has already migrated.
        _db.migrate(self.db)

    # -- public API -------------------------------------------------------------

    def add_source(
        self,
        connector_id: str,
        asset_id: str,
        data: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Hash and store *data*, then upsert a catalog entry linking it to a source.

        Returns the content SHA-256 digest. Idempotent: re-adding the same bytes for
        the same ``(connector_id, asset_id, revision)`` upserts a single row.

        *metadata* may carry the optional keys:

        - ``connector_type`` — connector instance type used to create the
          ``source_connectors`` row (default ``"local"``).
        - ``revision`` — caller-supplied revision for the entry (default: the content
          SHA-256, so identical bytes + asset map to one revision).
        - ``quality_score`` / ``quality_reason`` — initial quality flags.
        """
        metadata = metadata or {}
        digest = self.content.put(data)  # StorageError propagates untouched
        connector_type = metadata.get("connector_type", "local")
        revision = metadata.get("revision", digest)
        quality_score = metadata.get("quality_score")
        quality_reason = metadata.get("quality_reason")

        try:
            self.db.execute(
                "INSERT OR IGNORE INTO source_connectors(connector_id, connector_type)"
                " VALUES (?, ?)",
                (connector_id, connector_type),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO source_assets(connector_id, asset_id) VALUES (?, ?)",
                (connector_id, asset_id),
            )
            self.db.execute(
                "INSERT OR IGNORE INTO content(sha256, size) VALUES (?, ?)",
                (digest, len(data)),
            )
            self.db.execute(
                "INSERT INTO catalog_entries"
                " (connector_id, asset_id, revision, sha256, quality_score, quality_reason)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(connector_id, asset_id, revision) DO UPDATE SET"
                "   sha256 = excluded.sha256,"
                "   quality_score = excluded.quality_score,"
                "   quality_reason = excluded.quality_reason,"
                f"   updated_at = {_TIMESTAMP}",
                (connector_id, asset_id, revision, digest, quality_score, quality_reason),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CatalogError(
                f"failed to add source entry {connector_id}/{asset_id}: {exc}"
            ) from exc
        return digest

    def get_by_source(self, connector_id: str, asset_id: str) -> dict[str, Any] | None:
        """Return the most recent catalog entry for ``(connector_id, asset_id)``.

        Returns ``None`` when no entry exists for that source.
        """
        rows = self._query(
            "SELECT * FROM catalog_entries"
            " WHERE connector_id = ? AND asset_id = ? ORDER BY id DESC LIMIT 1",
            (connector_id, asset_id),
        )
        return rows[0] if rows else None

    def get_by_hash(self, sha256: str) -> list[dict[str, Any]]:
        """Return all catalog entries whose content matches *sha256*."""
        return self._query(
            "SELECT * FROM catalog_entries WHERE sha256 = ? ORDER BY id", (sha256,)
        )

    def update_quality_flags(
        self,
        connector_id: str,
        asset_id: str,
        quality_score: float | None = None,
        quality_reason: str | None = None,
        revision: str | None = None,
    ) -> None:
        """Set quality flags on a catalog entry.

        When *revision* is given, only that revision's row is updated; otherwise the
        most recent row for ``(connector_id, asset_id)`` is updated. Raises
        :class:`CatalogError` when no matching row exists.
        """
        params: tuple[Any, ...]
        if revision is not None:
            sql = (
                "UPDATE catalog_entries SET quality_score = ?, quality_reason = ?,"
                f" updated_at = {_TIMESTAMP}"
                " WHERE connector_id = ? AND asset_id = ? AND revision = ?"
            )
            params = (quality_score, quality_reason, connector_id, asset_id, revision)
        else:
            sql = (
                "UPDATE catalog_entries SET quality_score = ?, quality_reason = ?,"
                f" updated_at = {_TIMESTAMP}"
                " WHERE id = (SELECT id FROM catalog_entries"
                "             WHERE connector_id = ? AND asset_id = ?"
                "             ORDER BY id DESC LIMIT 1)"
            )
            params = (quality_score, quality_reason, connector_id, asset_id)
        try:
            cur = self.db.execute(sql, params)
            if cur.rowcount == 0:
                self.db.rollback()
                detail = (
                    f" revision={revision!r})"
                    if revision is not None
                    else " (no revision)"
                )
                raise CatalogError(
                    f"no catalog entry for {connector_id}/{asset_id}{detail}"
                )
            self.db.commit()
        except CatalogError:
            raise
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CatalogError(
                f"failed to update quality flags for {connector_id}/{asset_id}: {exc}"
            ) from exc

    # -- impl -------------------------------------------------------------------

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run a SELECT against ``catalog_entries`` and return dict rows."""
        cur = self.db.execute(sql, params)
        rows = cur.fetchall()
        return [dict(zip(_ENTRY_COLUMNS, row)) for row in rows]
