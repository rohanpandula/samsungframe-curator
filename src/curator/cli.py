"""Command-line interface for the Curator pipeline.

T06 ships a minimal headless CLI (the full surface — scan, health, jobs — arrives
in S04). It exposes a single ``catalog`` subcommand group:

- ``curator catalog init``   — create ``catalog.db`` (WAL mode) at CURATOR_DATA_ROOT
                               by connecting and migrating the v1 schema, then report
                               the database path.
- ``curator catalog add FILE`` — read a local file's bytes and write a catalog entry
                               with the correct SHA-256 and connector-scoped source
                               identity (reusing :meth:`Catalog.add_source`). Re-adding
                               the same file is idempotent (no duplicate row).

The CLI resolves all paths through the six-axis config (``CURATOR_DATA_ROOT``), so it
honors that environment variable wherever it points the data root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from curator import db
from curator.catalog import Catalog
from curator.errors import CuratorError

# Connector instance used for ad-hoc ``catalog add`` invocations. The source identity
# is the normalized absolute file path, consistent with LocalConnector semantics.
_CLI_CONNECTOR_ID = "cli-local"


def build_parser() -> argparse.ArgumentParser:
    """Build the argparse tree for the curator CLI."""
    parser = argparse.ArgumentParser(
        prog="curator",
        description="Samsung Frame curation pipeline — catalog, analysis, approval, render.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    catalog = sub.add_parser("catalog", help="catalog operations")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)

    catalog_sub.add_parser("init", help="create catalog.db (WAL) at CURATOR_DATA_ROOT")
    add = catalog_sub.add_parser("add", help="add a local file to the catalog")
    add.add_argument("file", help="path to the local file to add")

    return parser


def _catalog_init() -> Path:
    """Connect and migrate, creating CURATOR_DATA_ROOT/catalog.db idempotently."""
    conn = db.connect()
    try:
        db.migrate(conn)
    finally:
        conn.close()
    return db.default_db_path()


def _catalog_add(file_path: str) -> str:
    """Read *file_path* and add it to the catalog; return its SHA-256 digest."""
    path = Path(file_path).resolve()
    data = path.read_bytes()
    catalog = Catalog()
    try:
        return catalog.add_source(
            connector_id=_CLI_CONNECTOR_ID,
            asset_id=str(path),
            data=data,
            metadata={"connector_type": "local"},
        )
    finally:
        catalog.db.close()


def _dispatch(args: argparse.Namespace) -> int:
    """Route parsed args to the matching subcommand handler."""
    if args.command == "catalog":
        if args.catalog_command == "init":
            path = _catalog_init()
            print(f"initialized catalog at {path}")
            return 0
        if args.catalog_command == "add":
            digest = _catalog_add(args.file)
            print(f"added {args.file} -> {digest}")
            return 0
    # Unreachable given required subparsers; defensive fallback.
    return 2


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``curator`` console script.

    Returns a process exit code. Errors are reported to stderr and mapped to a
    non-zero exit rather than raising, so callers in-process can assert on the
    returned code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (CuratorError, OSError) as exc:
        print(f"curator: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - interactive entry
    raise SystemExit(main())
