"""SQLite connection factory and versioned migrations (D013).

Every persistence layer in the curator connects through :func:`connect`, which
resolves the catalog database path from the six-axis config (``CURATOR_DATA_ROOT``)
and enables WAL journaling plus foreign-key enforcement. The schema is applied by
:func:`migrate`, which walks the linear, hand-written migrations in
``curator.schema.MIGRATIONS`` guarded by ``PRAGMA user_version`` (idempotent).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from curator.config import CuratorConfig
from curator.schema import MIGRATIONS


def default_db_path(data_root: Path | None = None) -> Path:
    """Return the catalog database path under *data_root* (``catalog.db``).

    When *data_root* is None it is resolved from ``CURATOR_DATA_ROOT`` via the
    six-axis config (default ``~/.curator``).
    """
    if data_root is None:
        data_root = CuratorConfig().data_root
    return Path(data_root) / "catalog.db"


def connect(data_root: Path | None = None) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection to the catalog database.

    The parent data root is created if missing. WAL journaling and foreign-key
    enforcement are enabled per-connection. Schema is NOT applied here — callers
    must call :func:`migrate` (e.g. ``curator catalog init``).
    """
    path = default_db_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # ``check_same_thread=False`` lets a single Catalog serve the S04 FastAPI app,
    # whose ASGI workers (uvicorn / TestClient) may run requests on a thread other
    # than the one that opened the connection. SQLite WAL serializes concurrent
    # writers at the database level, so this is safe for the single-process,
    # loopback-only API (MEM003). CLI/ingest paths are single-threaded.
    db = sqlite3.connect(str(path), check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def migrate(db: sqlite3.Connection) -> None:
    """Apply any pending linear migrations, idempotently.

    Walks ``schema.MIGRATIONS`` in order; skips versions already reflected in
    ``PRAGMA user_version``. Safe to call repeatedly — the second run is a no-op.
    """
    current = int(db.execute("PRAGMA user_version").fetchone()[0])
    for version, ddl in MIGRATIONS:
        if version > current:
            db.executescript(ddl)
            db.execute(f"PRAGMA user_version = {version}")
    db.commit()


def table_names(db: sqlite3.Connection) -> set[str]:
    """Return the set of user table names present in the database."""
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}
