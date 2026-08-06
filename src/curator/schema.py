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

# Migration v9 (M005/S03): adds the append-only ``watcher_queue`` table that
# makes the durable watcher's enqueue/idempotency/reconciliation persistent.
# One row is appended per enqueued path and advanced in place through the
# per-path state machine (``queued -> processing -> done | error``), mirroring
# the ingest/consolidation journal posture. ``path`` is the normalized source
# path, ``sha`` the content SHA-256 at enqueue time, ``size``/``mtime`` the
# stat snapshot used for stabilization, and ``processed_at`` the wall-clock
# completion stamp. An ``error`` row stays re-attemptable; a ``queued`` row
# survives a crash between enqueue and done (reclaimed by the next drain).
SCHEMA_V9_SQL = """
CREATE TABLE IF NOT EXISTS watcher_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    path         TEXT NOT NULL,
    sha          TEXT,
    state        TEXT NOT NULL DEFAULT 'queued',
    size         INTEGER,
    mtime        INTEGER,
    enqueued_at  TEXT NOT NULL DEFAULT (datetime('now')),
    processed_at TEXT
);

    CREATE INDEX IF NOT EXISTS idx_watcher_queue_path
    ON watcher_queue(path);
"""

# Migration v10 (M005/S04): adds the collections/rotation persistence layer. Three
# append-only/registry tables backing user-defined playlists and their deterministic
# rotation engine state, mirroring the v4-v9 posture (plain INTEGER FKs with indexes,
# JSON blobs for per-row config/state, wall-clock ``created_at`` defaults).
#
# - ``playlists``       — one row per playlist; ``config_json`` is the full JSON
#   serialization of :class:`curator.collections.rotation.Playlist` (name apart,
#   kept as a real column for queryability).
# - ``playlist_members``— ordered membership: one row per (playlist, catalog_entry)
#   with an explicit ``position`` so order is preserved and re-shuffled on load.
# - ``rotation_state``  — one row per playlist holding the latest persisted
#   :class:`~curator.collections.rotation.RotationState` (``state_json``). A single
#   row per playlist (via the UNIQUE playlist_id index) is the simplest form that
#   preserves persistence + round-trip; the engine's own ``history`` list inside the
#   JSON keeps the explainability trail without an append-only journal.
SCHEMA_V10_SQL = """
CREATE TABLE IF NOT EXISTS playlists (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    config_json TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_playlists_name
    ON playlists(name);

CREATE TABLE IF NOT EXISTS playlist_members (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id      INTEGER NOT NULL,
    catalog_entry_id INTEGER NOT NULL,
    position         INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_playlist_members_playlist
    ON playlist_members(playlist_id);

CREATE TABLE IF NOT EXISTS rotation_state (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    playlist_id INTEGER NOT NULL,
    state_json  TEXT,
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- One row per playlist: keeps persistence + round-trip simple and lets the
-- store upsert the latest state for a playlist without an append-only journal.
CREATE UNIQUE INDEX IF NOT EXISTS uq_rotation_state_playlist
    ON rotation_state(playlist_id);
"""

# Migration v11 (M005/S05): persists the Immich source connector's sync state so
# :meth:`~curator.connectors.immich.ImmichConnector.sync` is checkpointed,
# idempotent, and isolated per connector instance. Two tables, mirroring the
# v4-v10 posture (plain TEXT/INTEGER columns, INSERT OR IGNORE/upsert friendly):
#
# - ``immich_sync_state``   — one row per connector instance holding the single
#   persisted browse/query **cursor** (a value, not per-asset), so a completed
#   sync resumes exactly where it left off without re-walking known assets.
#   Keyed by PRIMARY KEY(connector_id), the per-instance isolation guarantee.
# - ``immich_asset_state``  — one row per (connector, asset) holding the
#   last-observed ``revision``/``checksum``/``available``. Rows are **never
#   deleted**: an asset that disappears flips ``available = 0`` (a tombstone,
#   mirroring the M001 availability semantics) and stays in the table.
#
# These cannot reuse ``source_sync_checkpoints`` (M001): that table is keyed
# per (connector_id, asset_id) and carries no checksum or availability column,
# so it cannot hold a single-connector query cursor nor the per-asset
# download-verify state this connector needs.
SCHEMA_V11_SQL = """
CREATE TABLE IF NOT EXISTS immich_sync_state (
    connector_id TEXT PRIMARY KEY,
    cursor       TEXT,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS immich_asset_state (
    connector_id TEXT NOT NULL,
    asset_id     TEXT NOT NULL,
    revision     TEXT,
    checksum     TEXT,
    available    INTEGER NOT NULL DEFAULT 1,
    updated_at   TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (connector_id, asset_id)
);
"""

# Migration v12 (M006/S02): adds the append-only ``jobs`` table backing the durable
# job orchestrator — a checkpointed, idempotent executor with classified outcomes. One
# row per job; the ``key`` column holds a content-derived idempotency key (kind +
# canonical payload JSON) with a UNIQUE index so the orchestrator can enqueue a job
# once regardless of how many times the same work is requested. ``state`` walks
# ``queued -> active -> checkpointed -> completed | error | cancelled``; a crash
# between phases leaves ``active``/``checkpointed`` rows that a fresh orchestrator
# rehydrates (``resume_after_restart``) with their ``checkpoint_json`` + ``phase``
# intact, so no phase runs twice and content-addressed art is never duplicated.
# ``phase`` names the current step so a multi-phase job resumes exactly where it
# stopped, and the outcome columns (``outcome``/``reason``/``recovery``/
# ``user_explanation``) record the six-way classified failure surface
# (transient/permanent/policy-blocked/capability-unsupported/unresolved-external/
# user-cancelled).
SCHEMA_V12_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    key              TEXT NOT NULL,
    kind             TEXT NOT NULL,
    payload_json     TEXT,
    state            TEXT NOT NULL,
    phase            TEXT NOT NULL DEFAULT '',
    checkpoint_json  TEXT,
    attempts         INTEGER NOT NULL DEFAULT 0,
    outcome          TEXT,
    reason           TEXT,
    recovery         TEXT,
    user_explanation TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at       TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_key
    ON jobs(key);
"""

# Migration v13 (M007/S01): adds the ``taste_profiles`` and ``taste_preferences``
# tables backing the taste subsystem's profile-driven deterministic reranking.
# Mirroring the v4-v12 posture (plain INTEGER FK/registry rows, JSON blobs for
# per-row config, wall-clock ``created_at`` defaults):
#
# - ``taste_profiles``    — one row per taste profile. ``uid`` holds the profile's
#   string id (UNIQUE); ``kind`` names its scope (personal/household/room/...),
#   ``weights_json`` the full JSON serialization of the per-signal weights, and
#   ``version`` the profile's schema version. An index on ``kind`` supports
#   scope-scoped queries.
# - ``taste_preferences`` — optional per-entry preference rows for S02 (learning),
#   kept separate so renderer/approval paths never depend on them.
SCHEMA_V13_SQL = """
CREATE TABLE IF NOT EXISTS taste_profiles (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    uid          TEXT,
    kind         TEXT NOT NULL,
    name         TEXT,
    version      INTEGER NOT NULL DEFAULT 1,
    weights_json TEXT NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT,
    UNIQUE(uid)
);

CREATE INDEX IF NOT EXISTS idx_taste_profiles_kind
    ON taste_profiles(kind);

CREATE TABLE IF NOT EXISTS taste_preferences (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id       INTEGER NOT NULL,
    catalog_entry_id INTEGER NOT NULL,
    preference       INTEGER NOT NULL DEFAULT 0,
    note             TEXT,
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_taste_preferences_profile
    ON taste_preferences(profile_id);
"""

# Migration v14 (M008/S01/T2): adds the taste dialogue persistence layer — the
# append-only ``taste_observations`` journal and the ``taste_sessions`` registry.
# Mirroring the v4-v13 posture (plain columns, JSON blobs for per-row payloads,
# wall-clock ``created_at`` defaults):
#
# - ``taste_sessions``     — one row per dialogue session. ``id`` is a TEXT
#   primary key supplied by the session layer (a UUID, not an autoincrement),
#   ``kind`` names the surface that opened it (e.g. reaction-room/cli),
#   ``images_json`` the :class:`~curator.taste.dialogue.observation.ImageRef`s
#   the session surfaced, and ``closed_at`` is NULL until the session is closed.
# - ``taste_observations`` — append-only per-statement records: one row per user
#   observation, never updated or deleted. ``verbatim`` is the exact statement
#   text, ``attributes_json`` the extracted tags, ``polarity``/``confidence``
#   the disposition, ``images_json`` the referenced ImageRefs, and ``created_at``
#   the wall-clock receipt stamp. An index on ``session_id`` supports
#   chronological per-session replay.
SCHEMA_V14_SQL = """
CREATE TABLE IF NOT EXISTS taste_sessions (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    images_json TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    closed_at   TEXT
);

CREATE TABLE IF NOT EXISTS taste_observations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    verbatim        TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    polarity        TEXT NOT NULL,
    confidence      REAL NOT NULL,
    images_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_taste_observations_session
    ON taste_observations(session_id);
"""

# Migration v15 (M008/S04/T1): adds the taste-profile document + its append-only
# timeline. Mirrors the v14 posture (JSON blob payloads, wall-clock ``*_at``
# defaults):
#
# - ``taste_profile_doc``    — the single current profile document (``id`` is
#   pinned to 1 by a CHECK constraint). ``profile_json`` is the full
#   :class:`~curator.taste.dialogue.profile.TasteProfile` serialization and
#   ``version`` its monotonic document version. Rebuilding the profile replaces
#   this row; the timeline below is what preserves history.
# - ``taste_profile_events`` — append-only pin/edit/dispute records, never
#   updated or deleted. ``claim_id`` names the affected
#   :class:`~curator.taste.dialogue.profile.TasteClaim`, ``kind`` is
#   pin/edit/dispute and ``detail`` carries the new text (edit) or the evidence
#   marked for re-interpretation (dispute).
SCHEMA_V15_SQL = """
CREATE TABLE IF NOT EXISTS taste_profile_doc (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    profile_json TEXT NOT NULL,
    version      INTEGER NOT NULL,
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT
);

CREATE TABLE IF NOT EXISTS taste_profile_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id   TEXT NOT NULL,
    kind       TEXT NOT NULL,
    detail     TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_taste_profile_events_claim
    ON taste_profile_events(claim_id);
"""

# Migration v16 (M009/S01): adds vote-pairing + retraction columns to
# ``taste_preferences``. This is a mechanical version-allocation correction, not a
# policy change — 01-CONTEXT.md's "Files Likely Touched" note anticipated "v16
# (embeddings BLOB)" for this milestone, but S01's vote-pairing decision claims v16
# first; the embeddings BLOB table (S02) is now v17. ``EXPECTED_TABLES`` is
# unchanged (no new table, only columns).
#
# ``taste_preferences`` (v13) is flat and entry-scoped — (id, profile_id,
# catalog_entry_id, preference, note, created_at) — with no column linking the two
# entries compared in one pairwise vote. Research confirmed the only existing
# writers are test fixtures approximating a vote as two independently-inserted
# rows. This migration adds:
#
# - ``vote_group``   — groups the winner row + loser row of one
#   ``curator taste vote`` call so history is group-recoverable.
# - ``retracted_at`` — NULL = active; a non-NULL timestamp marks the row retracted
#   without deleting it, mirroring the append-only posture of
#   ``taste_observations``/``taste_profile_events``. History is never erased.
#
# Following the v2/v3 "one ADD COLUMN per statement" convention (SQLite forbids a
# multi-column ALTER), each column is its own statement. Both are nullable with no
# DEFAULT, so every existing pre-M009 row — including every test fixture's raw
# 4-column ``INSERT INTO taste_preferences(profile_id, catalog_entry_id,
# preference, note) VALUES (...)`` — reads back NULL for both and is simply
# excluded from the new grouped-replay logic (``vote_group IS NOT NULL`` gate); no
# existing test breaks.
SCHEMA_V16_SQL = """
ALTER TABLE taste_preferences ADD COLUMN vote_group TEXT;
ALTER TABLE taste_preferences ADD COLUMN retracted_at TEXT;

CREATE INDEX IF NOT EXISTS idx_taste_preferences_vote_group
    ON taste_preferences(vote_group);
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
    (9, SCHEMA_V9_SQL),
    (10, SCHEMA_V10_SQL),
    (11, SCHEMA_V11_SQL),
    (12, SCHEMA_V12_SQL),
    (13, SCHEMA_V13_SQL),
    (14, SCHEMA_V14_SQL),
    (15, SCHEMA_V15_SQL),
    (16, SCHEMA_V16_SQL),
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
    "watcher_queue",
    "playlists",
    "playlist_members",
    "rotation_state",
    "immich_sync_state",
    "immich_asset_state",
    "jobs",
    "taste_profiles",
    "taste_preferences",
    "taste_sessions",
    "taste_observations",
    "taste_profile_doc",
    "taste_profile_events",
]
