"""SQLite schema v1 for the Curator catalog (system of record, R001).

The catalog is a single-user embedded SQLite database owned entirely by this repo.
Per decision D013, schema changes are **hand-written linear migrations** guarded by
``PRAGMA user_version`` — no Alembic/ORM. Each migration is a ``(version, sql)`` tuple
in :data:`MIGRATIONS`; :func:`curator.db.migrate` applies them in order, idempotently.

Table names in v1 match the Boundary Map contract surface exactly:

- ``source_connectors``      — connector instances (opaque connector_id)
- ``source_assets``          — opaque source assets, scoped by UNIQUE(connector_id, asset_id)
- ``source_observations``    — append-only revision observations with availability/tombstone
- ``source_sync_checkpoints``— per-(connector, asset) enumeration cursor
- ``content``                — sha256 PRIMARY KEY, size; the single byte-convergence point
- ``catalog_entries``        — links an asset<->content per revision + quality flags
- ``ingest_journal``         — append-only ingest run journal
- ``consolidation_journal``  — append-only consolidation run journal (populated in S03)

Keep v1 minimal: analysis / approval / render tables are added by later slices (S02/S03),
not here.
"""

from __future__ import annotations

# ISO-8601 UTC timestamp used as the default for all *_at columns.
_TIMESTAMP = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

SCHEMA_V1_SQL = f"""
CREATE TABLE IF NOT EXISTS source_connectors (
    connector_id   TEXT PRIMARY KEY,
    connector_type TEXT NOT NULL,
    name           TEXT,
    created_at     TEXT NOT NULL DEFAULT ({_TIMESTAMP})
);

CREATE TABLE IF NOT EXISTS source_assets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    connector_id  TEXT NOT NULL REFERENCES source_connectors(connector_id) ON DELETE CASCADE,
    asset_id      TEXT NOT NULL,
    first_seen_at TEXT NOT NULL DEFAULT ({_TIMESTAMP}),
    -- The scoping guarantee: the same opaque asset_id across two connector
    -- instances is TWO rows here, never a collision.
    UNIQUE(connector_id, asset_id)
);

-- Append-only revision observations. History is never rewritten: an unavailable
-- reference is recorded with available = 0 (tombstone), never deleted.
CREATE TABLE IF NOT EXISTS source_observations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    connector_id TEXT NOT NULL REFERENCES source_connectors(connector_id) ON DELETE CASCADE,
    asset_id     TEXT NOT NULL,
    revision     TEXT,
    changed      INTEGER NOT NULL DEFAULT 1,
    available    INTEGER NOT NULL DEFAULT 1,
    observed_at  TEXT NOT NULL DEFAULT ({_TIMESTAMP})
);

CREATE TABLE IF NOT EXISTS source_sync_checkpoints (
    connector_id  TEXT NOT NULL REFERENCES source_connectors(connector_id) ON DELETE CASCADE,
    asset_id      TEXT NOT NULL,
    cursor        TEXT,
    revision      TEXT,
    updated_at    TEXT NOT NULL DEFAULT ({_TIMESTAMP}),
    PRIMARY KEY (connector_id, asset_id)
);

-- Single byte-convergence point: identical bytes always map to one row here.
CREATE TABLE IF NOT EXISTS content (
    sha256        TEXT PRIMARY KEY,
    size          INTEGER NOT NULL,
    first_seen_at TEXT NOT NULL DEFAULT ({_TIMESTAMP})
);

CREATE TABLE IF NOT EXISTS catalog_entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    connector_id  TEXT NOT NULL REFERENCES source_connectors(connector_id) ON DELETE CASCADE,
    asset_id      TEXT NOT NULL,
    revision      TEXT,
    sha256        TEXT NOT NULL REFERENCES content(sha256) ON DELETE RESTRICT,
    quality_score REAL,
    quality_reason TEXT,
    created_at    TEXT NOT NULL DEFAULT ({_TIMESTAMP}),
    updated_at    TEXT,
    -- Idempotent re-add of the same (connector, asset, revision) upserts one row.
    UNIQUE(connector_id, asset_id, revision)
);

CREATE INDEX IF NOT EXISTS idx_catalog_entries_sha256 ON catalog_entries(sha256);

CREATE TABLE IF NOT EXISTS ingest_journal (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    connector_id TEXT NOT NULL REFERENCES source_connectors(connector_id) ON DELETE CASCADE,
    asset_id     TEXT NOT NULL,
    sha256       TEXT,
    status       TEXT NOT NULL DEFAULT 'started',
    error        TEXT,
    started_at   TEXT NOT NULL DEFAULT ({_TIMESTAMP}),
    finished_at  TEXT
);

CREATE TABLE IF NOT EXISTS consolidation_journal (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    status     TEXT,
    note       TEXT,
    created_at TEXT NOT NULL DEFAULT ({_TIMESTAMP})
);
"""

# Ordered hand-written linear migrations: ``(schema_version, ddl)``.
MIGRATIONS: list[tuple[int, str]] = [
    (1, SCHEMA_V1_SQL),
]

# Highest applied schema version (== PRAGMA user_version after migrate()).
SCHEMA_VERSION: int = MIGRATIONS[-1][0]

# The boundary-map table set delivered by v1. Used by tests and diagnostics.
EXPECTED_TABLES: list[str] = [
    "source_connectors",
    "source_assets",
    "source_observations",
    "source_sync_checkpoints",
    "content",
    "catalog_entries",
    "ingest_journal",
    "consolidation_journal",
]
