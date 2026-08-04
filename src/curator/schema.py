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

v2 (migration ``(2, ...)``) extends the boundary for S02's ingest + dedup work:

- ``catalog_entries`` gains per-entry dedup columns populated by the IngestPipeline /
  consolidation: ``cluster_id`` (opaque dedup cluster key), ``dupe_of`` (opaque id of the
  cluster's canonical member), ``quality_flags`` (JSON blob of derived flags, e.g.
  ``highest_res`` / ``crop_candidate``), and ``best_original`` (1 when this entry is the
  cluster's chosen best-original, else 0/NULL).
- ``content_image`` — a durable per-content-hash image signature table holding decoded
  dimensions and a perceptual hash (phash) string for near-dupe clustering.

v3 (migration ``(3, ...)``) expands ``consolidation_journal`` from a run-level table into a
per-file state machine (``started -> staged -> verified -> promoted/error``) mirroring
``ingest_journal`` — added columns: ``connector_id``, ``asset_id``, ``sha256``, ``error``,
``started_at``, ``finished_at`` — so a mid-copy interrupt can be resumed (S03 resume
contract). Existing v1 run-level columns (``status``/``note``/``created_at``) are kept for
backward compatibility.

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

# Migration v2 (S02 ingest + dedup): adds the dedup/consolidation columns to
# ``catalog_entries`` and a per-content-hash image-signature table. SQLite forbids
# adding multiple columns in a single ALTER, so each ADD COLUMN is its own
# statement; ``executescript`` applies them in one migration step.
SCHEMA_V2_SQL = f"""
ALTER TABLE catalog_entries ADD COLUMN cluster_id TEXT;
ALTER TABLE catalog_entries ADD COLUMN dupe_of TEXT;
ALTER TABLE catalog_entries ADD COLUMN quality_flags TEXT;
ALTER TABLE catalog_entries ADD COLUMN best_original INTEGER;

-- Durable per-content-hash image signature: dimensions + perceptual hash.
-- Rows are derived metadata, so they follow their content row via CASCADE.
CREATE TABLE IF NOT EXISTS content_image (
    sha256 TEXT PRIMARY KEY REFERENCES content(sha256) ON DELETE CASCADE,
    width  INTEGER NOT NULL,
    height INTEGER NOT NULL,
    phash  TEXT,
    created_at TEXT NOT NULL DEFAULT ({_TIMESTAMP})
);
"""

# Migration v3 (S03 consolidation): expands ``consolidation_journal`` from a
# run-level table (id, status, note, created_at) into a **per-file state machine**
# mirroring ``ingest_journal`` — ``started -> staged -> verified -> promoted/error`` —
# so a mid-copy interrupt can be resumed and every file's outcome is observable.
# SQLite forbids adding multiple columns in a single ALTER, so each ADD COLUMN is
# its own statement (v2 precedent); ``executescript`` applies them in one step.
# Existing columns (status/note/created_at) are preserved for backward compat with
# any v1/v2 run-level rows.
SCHEMA_V3_SQL = """
ALTER TABLE consolidation_journal ADD COLUMN connector_id TEXT;
ALTER TABLE consolidation_journal ADD COLUMN asset_id TEXT;
ALTER TABLE consolidation_journal ADD COLUMN sha256 TEXT;
ALTER TABLE consolidation_journal ADD COLUMN error TEXT;
ALTER TABLE consolidation_journal ADD COLUMN started_at TEXT;
ALTER TABLE consolidation_journal ADD COLUMN finished_at TEXT;

-- Resume lookups select the latest row per (connector_id, asset_id) / status.
CREATE INDEX IF NOT EXISTS idx_consolidation_journal_conn_asset
    ON consolidation_journal(connector_id, asset_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_journal_status
    ON consolidation_journal(status);
"""

# Migration v4 (M002/S02 analysis): adds the append-only ``analysis_results``
# table capturing one row per analysis run per catalog entry — a history-preserving
# posture mirroring the ingest/consolidation journals. A corrupt/undecodable file
# is recorded as a ``corrupt`` row with an actionable ``corrupt_reason`` rather than
# silently dropped. ``catalog_entry_id`` is a plain INTEGER with an index (not an
# enforced FK) so analysis may append rows without coupling to ingest ordering.
SCHEMA_V4_SQL = """
CREATE TABLE IF NOT EXISTS analysis_results (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    catalog_entry_id  INTEGER NOT NULL,
    profile           TEXT NOT NULL,
    engine_version    TEXT NOT NULL,
    analysis_json     TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'ok',
    corrupt_reason    TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_analysis_results_catalog_entry
    ON analysis_results(catalog_entry_id);
"""

# Migration v5 (M002/S04): adds the append-only ``proposals`` (derived
# treatment recommendations) and ``art_direction_manifests`` (per-entry art-direction
# payloads) tables. Both mirror the ``analysis_results`` posture — a plain INTEGER
# ``catalog_entry_id`` with an index (no enforced FK) so rows append without
# coupling to ingest ordering, and a wall-clock ``created_at`` default.
SCHEMA_V5_SQL = """
CREATE TABLE IF NOT EXISTS proposals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    catalog_entry_id  INTEGER NOT NULL,
    treatment         TEXT NOT NULL,
    score             REAL,
    rationale_json    TEXT,
    evidence_json     TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_proposals_catalog_entry
    ON proposals(catalog_entry_id);

CREATE TABLE IF NOT EXISTS art_direction_manifests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    catalog_entry_id  INTEGER NOT NULL,
    manifest_version  TEXT NOT NULL,
    manifest_json     TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_art_direction_manifests_catalog_entry
    ON art_direction_manifests(catalog_entry_id);
"""

# Migration v6 (M003/S03): adds the append-only ``approvals`` table capturing one
# row per explicit per-entry decision/transition (approve/reject/undo/redo),
# mirroring the ``analysis_results``/``proposals`` posture — a plain INTEGER
# ``catalog_entry_id`` with an index (no enforced FK) so rows append without
# coupling to ingest ordering, and a wall-clock ``created_at`` default. History is
# never erased: "current" for an entry is simply the latest row; undo/redo append
# new transition rows rather than rewriting/deleting prior ones.
SCHEMA_V6_SQL = """
CREATE TABLE IF NOT EXISTS approvals (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    catalog_entry_id  INTEGER NOT NULL,
    decision          TEXT NOT NULL,
    rationale         TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_approvals_catalog_entry
    ON approvals(catalog_entry_id);
"""

# Migration v7 (M004/S03): adds the append-only ``renders`` table capturing one
# row per persisted render artifact, mirroring the ``analysis_results`` /
# ``proposals``/``approvals`` posture — a plain INTEGER ``catalog_entry_id`` with
# an index (nullable, since a render may target an inline source with no catalog
# entry), a wall-clock ``created_at`` default, and a hashed reference to the
# stored artifact rather than a raw FK. ``artifact_sha`` is the byte-deterministic
# SHA-256 of the rendered PNG bytes as stored in the ContentStore.
SCHEMA_V7_SQL = """
CREATE TABLE IF NOT EXISTS renders (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    catalog_entry_id  INTEGER,
    target            TEXT,
    renderer_version  TEXT,
    artifact_sha      TEXT,
    render_json       TEXT,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_renders_catalog_entry
    ON renders(catalog_entry_id);

CREATE INDEX IF NOT EXISTS idx_renders_artifact_sha
    ON renders(artifact_sha);
"""

# Migration v8 (M005/S01): adds the append-only ``dest_journal`` table capturing
# every publish attempt/transition per artifact, mirroring the ingest/consolidation
# journal posture — one row per attempt advanced in place through the per-artifact
# state machine (``staged -> verified -> applied | error``). ``adapter_id`` names the
# :class:`~curator.dest.base.DestinationAdapter` instance, ``artifact_id`` the
# exact-ID destination key, ``op`` the write operation (put/replace/remove), ``sha``
# the published SHA-256, and ``status`` the terminal outcome. ``error`` preserves
# the failure text so a retry can resume from the same row rather than duplicating
# the error. History is never erased.
SCHEMA_V8_SQL = """
CREATE TABLE IF NOT EXISTS dest_journal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    adapter_id  TEXT NOT NULL,
    artifact_id TEXT NOT NULL,
    op          TEXT NOT NULL,
    sha         TEXT,
    status      TEXT NOT NULL,
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_dest_journal_artifact
    ON dest_journal(artifact_id);
"""

# Ordered hand-written linear migrations: ``(schema_version, ddl)``.
MIGRATIONS: list[tuple[int, str]] = [
    (1, SCHEMA_V1_SQL),
    (2, SCHEMA_V2_SQL),
    (3, SCHEMA_V3_SQL),
    (4, SCHEMA_V4_SQL),
    (5, SCHEMA_V5_SQL),
    (6, SCHEMA_V6_SQL),
    (7, SCHEMA_V7_SQL),
    (8, SCHEMA_V8_SQL),
]

# Highest applied schema version (== PRAGMA user_version after migrate()).
SCHEMA_VERSION: int = MIGRATIONS[-1][0]

# The boundary-map table set delivered by the migrations. Used by tests/diagnostics.
EXPECTED_TABLES: list[str] = [
    "source_connectors",
    "source_assets",
    "source_observations",
    "source_sync_checkpoints",
    "content",
    "catalog_entries",
    "ingest_journal",
    "consolidation_journal",
    "content_image",
    "analysis_results",
    "proposals",
    "art_direction_manifests",
    "approvals",
    "renders",
    "dest_journal",
]
