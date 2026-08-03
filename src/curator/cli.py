"""Command-line interface for the Curator pipeline.

T06 ships a minimal headless CLI (the full surface — scan, health, jobs — arrives
in S04). It exposes a ``catalog`` subcommand group plus the ``ingest`` command:

- ``curator catalog init``   — create ``catalog.db`` (WAL mode) at CURATOR_DATA_ROOT
                               by connecting and migrating the v1 schema, then report
                               the database path.
- ``curator catalog add FILE`` — read a local file's bytes and write a catalog entry
                               with the correct SHA-256 and connector-scoped source
                               identity (reusing :meth:`Catalog.add_source`). Re-adding
                               the same file is idempotent (no duplicate row).
- ``curator ingest PATH``     — run the local IngestPipeline over *PATH* (dedup + cluster)
                               and print a human-readable report of unique clusters, exact/
                               near families, best-original flags with phash distances, and
                               the explicit-unsupported (RAW) / corrupt failure surfaces.

The CLI resolves all paths through the six-axis config (``CURATOR_DATA_ROOT``), so it
honors that environment variable wherever it points the data root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from curator import db
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.errors import CuratorError
from curator.ingest.pipeline import IngestPipeline
from curator.ingest.report import IngestReport

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

    ingest = sub.add_parser(
        "ingest", help="ingest a local folder into the catalog (dedup + cluster)"
    )
    ingest.add_argument("path", help="path to the local folder to ingest")
    ingest.add_argument(
        "--resume",
        action="store_true",
        help="resume from the ingest_journal: skip already-indexed assets",
    )

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


def _ingest(path_str: str, resume: bool) -> int:
    """Ingest *path_str* (a local folder) and print a dedup/cluster report.

    Runs the local-folder implementation through the shared SourceConnector
    boundary (LocalConnector), clustering + cataloging via :class:`IngestPipeline`
    against the CURATOR_DATA_ROOT database. Prints a human-readable report
    showing unique clusters, exact/near families, best-original flags with phash
    distances, and the explicit-unsupported (RAW) / corrupt failure surfaces.
    """
    folder = Path(path_str).resolve()
    if not folder.is_dir():
        raise CuratorError(f"ingest source is not a directory: {path_str!r}")
    catalog = Catalog()
    try:
        report = IngestPipeline(LocalConnector(folder), catalog=catalog).run(
            resume=resume
        )
    finally:
        catalog.db.close()
    print(_format_report(report))
    return 0


def _format_report(report: IngestReport) -> str:
    """Render an :class:`IngestReport` as a human-readable dedup summary."""
    lines = [
        f"ingest {report.connector_id}",
        f"  total enumerated : {report.total_enumerated}",
        f"  indexed          : {report.indexed_count}",
        f"  unique clusters  : {report.unique_clusters}"
        f"  (exact={report.exact_clusters}, near={report.near_clusters})",
        f"  unsupported      : {report.unsupported_count}",
        f"  corrupt          : {report.corrupt_count}",
        f"  error            : {report.error_count}",
        "",
        "clusters:",
    ]
    by_cluster: dict[str, list] = {}
    for entry in report.entries:
        if entry.cluster_id is None:
            continue  # unclustered entry (defensive; pipeline always clusters)
        by_cluster.setdefault(entry.cluster_id, []).append(entry)
    for cluster_id in sorted(by_cluster):
        members = by_cluster[cluster_id]
        best = next((m for m in members if m.best_original), members[0])
        lines.append(
            f"  {cluster_id}  best={Path(best.asset_id).name}"
            f"  members={len(members)}"
        )
        for member in sorted(members, key=lambda m: Path(m.asset_id).name):
            flag = "*" if member.best_original else " "
            dist = member.phash_distance if member.phash_distance is not None else "-"
            lines.append(f"      {flag} {Path(member.asset_id).name}  phash_dist={dist}")
    unsupported = [f for f in report.failures if f.status == "unsupported"]
    if unsupported:
        lines.append("")
        lines.append("explicit-unsupported (RAW):")
        for failure in unsupported:
            lines.append(f"  {Path(failure.asset_id).name}")
    corrupt = [f for f in report.failures if f.status == "corrupt"]
    if corrupt:
        lines.append("")
        lines.append("corrupt:")
        for failure in corrupt:
            lines.append(f"  {Path(failure.asset_id).name}  -> {failure.error}")
    return "\n".join(lines)


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
    if args.command == "ingest":
        return _ingest(args.path, resume=args.resume)
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
