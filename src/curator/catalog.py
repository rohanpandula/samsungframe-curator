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

import json
import sqlite3
from pathlib import Path
from typing import Any

from curator import db as _db
from curator.content_store import ContentStore
from curator.errors import CatalogError

# Column order of ``SELECT *`` on catalog_entries (matches schema v1 + v2 DDL).
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
    # v2 dedup/consolidation columns (migration 2).
    "cluster_id",
    "dupe_of",
    "quality_flags",
    "best_original",
]

# Column order of ``SELECT *`` on content_image (schema v2 DDL).
_IMAGE_COLUMNS = ["sha256", "width", "height", "phash", "created_at"]

# Column order of ``SELECT *`` on consolidation_journal (schema v1 + v3 DDL).
# v3 added the per-file columns after the run-level ones.
_CONSOLIDATION_COLUMNS = [
    "id",
    "status",
    "note",
    "created_at",
    # v3 per-file state-machine columns (migration 3).
    "connector_id",
    "asset_id",
    "sha256",
    "error",
    "started_at",
    "finished_at",
]

# Per-file consolidation_journal status vocabulary (mirrors ingest_journal).
CONSOLIDATION_STARTED = "started"
CONSOLIDATION_STAGED = "staged"
CONSOLIDATION_VERIFIED = "verified"
CONSOLIDATION_PROMOTED = "promoted"
CONSOLIDATION_ERROR = "error"

# The full per-file status vocabulary (for input validation).
CONSOLIDATION_STATUSES = frozenset(
    {
        CONSOLIDATION_STARTED,
        CONSOLIDATION_STAGED,
        CONSOLIDATION_VERIFIED,
        CONSOLIDATION_PROMOTED,
        CONSOLIDATION_ERROR,
    }
)

# Terminal success statuses: a file recorded here is NOT re-attempted on resume,
# while an ``error`` row IS re-attempted (mirror IngestPipeline resume semantics).
_CONSOLIDATION_TERMINAL = {
    CONSOLIDATION_STAGED,
    CONSOLIDATION_VERIFIED,
    CONSOLIDATION_PROMOTED,
}

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
        - ``cluster_id`` / ``dupe_of`` — dedup-cluster identity for this entry.
        - ``best_original`` — 1/True when this entry is its cluster's best-original.
        - ``quality_flags`` — a dict (serialized to JSON) of derived flags.
        """
        metadata = metadata or {}
        digest = self.content.put(data)  # StorageError propagates untouched
        connector_type = metadata.get("connector_type", "local")
        revision = metadata.get("revision", digest)
        quality_score = metadata.get("quality_score")
        quality_reason = metadata.get("quality_reason")
        cluster_id = metadata.get("cluster_id")
        dupe_of = metadata.get("dupe_of")
        best_original = 1 if metadata.get("best_original") else None
        qf = metadata.get("quality_flags")
        quality_flags = json.dumps(qf) if isinstance(qf, dict) else (qf if qf is not None else None)

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
                " (connector_id, asset_id, revision, sha256, quality_score, quality_reason,"
                "  cluster_id, dupe_of, quality_flags, best_original)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(connector_id, asset_id, revision) DO UPDATE SET"
                "   sha256 = excluded.sha256,"
                "   quality_score = excluded.quality_score,"
                "   quality_reason = excluded.quality_reason,"
                "   cluster_id = excluded.cluster_id,"
                "   dupe_of = excluded.dupe_of,"
                "   quality_flags = excluded.quality_flags,"
                "   best_original = excluded.best_original,"
                f"   updated_at = {_TIMESTAMP}",
                (
                    connector_id,
                    asset_id,
                    revision,
                    digest,
                    quality_score,
                    quality_reason,
                    cluster_id,
                    dupe_of,
                    quality_flags,
                    best_original,
                ),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CatalogError(
                f"failed to add source entry {connector_id}/{asset_id}: {exc}"
            ) from exc
        return digest

    def get_source_asset_ids(self, connector_id: str) -> set[str]:
        """Return the set of distinct on-disk ``asset_id`` values cataloged for *connector_id*.

        Used by ``scan`` to compute the ``missing`` side of the catalog diff: any
        cataloged asset id the source no longer enumerates is missing. Queries
        ``catalog_entries`` (the actual catalog) rather than ``source_assets``,
        so only assets that were successfully indexed count as cataloged.
        """
        cur = self.db.execute(
            "SELECT DISTINCT asset_id FROM catalog_entries WHERE connector_id = ?",
            (connector_id,),
        )
        return {str(row[0]) for row in cur.fetchall()}

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

    def set_image_signature(
        self,
        sha256: str,
        width: int,
        height: int,
        phash: str | None = None,
    ) -> None:
        """Upsert the durable image signature (dimensions + phash) for *sha256*.

        ``content_image`` is keyed by the content hash, so re-ingesting the same
        bytes overwrites in place (idempotent). Requires a ``content`` row to exist
        (foreign key); raises :class:`CatalogError` otherwise.
        """
        try:
            self.db.execute(
                "INSERT INTO content_image(sha256, width, height, phash)"
                " VALUES (?, ?, ?, ?)"
                " ON CONFLICT(sha256) DO UPDATE SET"
                "   width = excluded.width,"
                "   height = excluded.height,"
                "   phash = excluded.phash,"
                f"   created_at = {_TIMESTAMP}",
                (sha256, width, height, phash),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CatalogError(
                f"failed to set image signature for {sha256}: {exc}"
            ) from exc

    def get_image_signature(self, sha256: str) -> dict[str, Any] | None:
        """Return the durable image signature for *sha256*, or ``None`` when absent."""
        cur = self.db.execute(
            "SELECT * FROM content_image WHERE sha256 = ?", (sha256,)
        )
        row = cur.fetchone()
        return dict(zip(_IMAGE_COLUMNS, row)) if row else None

    def get_by_cluster(self, cluster_id: str) -> list[dict[str, Any]]:
        """Return all catalog entries assigned to *cluster_id*, oldest first."""
        return self._query(
            "SELECT * FROM catalog_entries WHERE cluster_id = ? ORDER BY id",
            (cluster_id,),
        )

    def count_catalog_entries(self) -> int:
        """Return the total number of rows in ``catalog_entries``.

        Used by the ``health`` surface (CLI + API) to report the catalog's
        entry count. Counts every ``catalog_entries`` row regardless of
        connector or cluster membership.
        """
        row = self.db.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()
        return int(row[0])

    def count_unique_clusters(self) -> int:
        """Return the number of distinct non-NULL ``cluster_id`` values.

        Used by acceptance: unique entries are defined as distinct dedup clusters.
        """
        row = self.db.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM catalog_entries"
            " WHERE cluster_id IS NOT NULL"
        ).fetchone()
        return int(row[0])

    def get_by_phash(self, phash: str) -> list[dict[str, Any]]:
        """Return catalog entries whose content hash has image signature *phash*.

        Joins ``catalog_entries`` to ``content_image`` on sha256.
        """
        return self._query(
            "SELECT e.* FROM catalog_entries e"
            " JOIN content_image i ON i.sha256 = e.sha256"
            " WHERE i.phash = ? ORDER BY e.id",
            (phash,),
        )

    # -- consolidation journal (schema v3) -------------------------------------

    def consolidation_journal_start(
        self, connector_id: str, asset_id: str, note: str | None = None
    ) -> int:
        """Record a ``started`` row in the per-file consolidation_journal.

        ``connector_id`` identifies the consolidation run (typically the resolved
        legacy source path under the canonical data root) and ``asset_id`` the
        individual source file within it. Returns the new row id. Non-terminal — the
        row is always superseded by a ``staged``/``verified``/``promoted``/``error``
        transition via :meth:`consolidation_journal_update`.
        """
        try:
            cur = self.db.execute(
                "INSERT INTO consolidation_journal"
                " (connector_id, asset_id, status, note)"
                " VALUES (?, ?, ?, ?)",
                (connector_id, asset_id, CONSOLIDATION_STARTED, note),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CatalogError(
                f"failed to start consolidation_journal row for"
                f" {connector_id}/{asset_id}: {exc}"
            ) from exc
        row_id = cur.lastrowid
        if row_id is None:
            raise CatalogError("failed to obtain consolidation_journal row id")
        return int(row_id)

    def consolidation_journal_update(
        self,
        row_id: int,
        status: str,
        sha256: str | None = None,
        error: str | None = None,
        note: str | None = None,
    ) -> None:
        """Transition one consolidation_journal row to *status*.

        Writes ``sha256``/``error``/``note`` only when provided (COALESCE keeps prior
        values) and stamps ``finished_at`` on terminal success statuses
        (``staged``/``verified``/``promoted``). Raise :class:`CatalogError` for an
        unknown status. ``error`` rows stay active (re-attempted on resume).
        """
        if status not in CONSOLIDATION_STATUSES:
            raise CatalogError(f"unknown consolidation_journal status {status!r}")
        if status in _CONSOLIDATION_TERMINAL:
            sql = (
                "UPDATE consolidation_journal SET"
                "   sha256 = COALESCE(?, sha256),"
                "   status = ?,"
                "   error = COALESCE(?, error),"
                "   note = COALESCE(?, note),"
                f"   finished_at = {_TIMESTAMP}"
                " WHERE id = ?"
            )
        else:
            sql = (
                "UPDATE consolidation_journal SET"
                "   sha256 = COALESCE(?, sha256),"
                "   status = ?,"
                "   error = COALESCE(?, error),"
                "   note = COALESCE(?, note)"
                " WHERE id = ?"
            )
        try:
            self.db.execute(sql, (sha256, status, error, note, row_id))
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CatalogError(
                f"failed to update consolidation_journal row {row_id}: {exc}"
            ) from exc

    def consolidation_checkpoint(
        self, connector_id: str, asset_id: str
    ) -> str | None:
        """Return the status of the newest journal row for one source file.

        Used by resume to decide whether a file was already promoted/verified
        (skip) or errored (re-attempt). Returns ``None`` when no row exists.
        """
        row = self.db.execute(
            "SELECT status FROM consolidation_journal"
            " WHERE connector_id = ? AND asset_id = ?"
            " ORDER BY id DESC LIMIT 1",
            (connector_id, asset_id),
        ).fetchone()
        return str(row[0]) if row else None

    def consolidation_journal_rows(
        self, connector_id: str
    ) -> list[dict[str, Any]]:
        """Return every journal row for a consolidation run, oldest first.

        Lets the executor reconcile the per-file outcomes of a run (S05: no unique
        source omitted, no source deleted).
        """
        cur = self.db.execute(
            "SELECT * FROM consolidation_journal WHERE connector_id = ? ORDER BY id",
            (connector_id,),
        )
        return [dict(zip(_CONSOLIDATION_COLUMNS, row)) for row in cur.fetchall()]

    # -- impl -------------------------------------------------------------------

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        """Run a SELECT against ``catalog_entries`` and return dict rows."""
        cur = self.db.execute(sql, params)
        rows = cur.fetchall()
        return [dict(zip(_ENTRY_COLUMNS, row)) for row in rows]
