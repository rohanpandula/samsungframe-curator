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
- ``curator render SOURCE``   — render an art-direction manifest (``.json``) or a media
                               asset to a target (named ``1080p``/``4k`` or ``WxH``) via the
                               deterministic renderer, printing a summary or ``--json`` result.
- ``curator validate FILE``   — gate a rendered artifact against expected provenance
                               (``--expected-sha`` + ``--target`` dims); exit 0 publishable /
                               1 not / 2 on read or parse error.

The CLI resolves all paths through the six-axis config (``CURATOR_DATA_ROOT``), so it
honors that environment variable wherever it points the data root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from curator import api as api_module
from curator import db
from curator.analysis.cli_utils import resolve_catalog_entry
from curator.analysis.errors import CatalogEntryNotFound
from curator.analysis.local import LocalAnalysisProvider
from curator.analysis.pipeline import AnalysisAsset, AnalysisPipeline
from curator.analysis.profiles import AnalysisProfile
from curator.analysis.schema import AnalysisResult
from curator.approve import ApprovalService
from curator.artdirection.manifest import (
    MANIFEST_VERSION,
    ArtDirectionManifest,
    LayoutTreatment,
    SourceRegion,
)
from curator.artdirection.policy import (
    ArtDirectionRequest,
    TreatmentProposal,
    materialize_manifest,
)
from curator.artdirection.policy import (
    propose as policy_propose,
)
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.consolidate import ConsolidationExecutor, ConsolidationPlan, build_plan
from curator.content_store import ContentStore
from curator.errors import CuratorError
from curator.hashing import sha256_hex
from curator.ingest.pipeline import IngestPipeline
from curator.ingest.report import IngestReport
from curator.render.renderer import DeterministicRenderer, RenderError, RenderResult
from curator.render.validate import ArtifactValidator, ValidationReport
from curator.scan import ScanDiff, scan_connector

# Documented exit-code contract (S04): 0=ok, 1=partial/warnings, 2=fatal,
# 3=no-change. Every subcommand returns one of these; ``main`` maps uncaught
# CuratorError/OSError to ``EXIT_FATAL`` (2).
EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_FATAL = 2
EXIT_NO_CHANGE = 3

# Connector instance used for ad-hoc ``catalog add`` invocations. The source identity
# is the normalized absolute file path, consistent with LocalConnector semantics.
_CLI_CONNECTOR_ID = "cli-local"

# Known render targets (name -> pixel dimensions) for ``propose`` / ``manifest``.
_TARGET_SPECS: dict[str, tuple[int, int]] = {
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

#: Default render target, matching the configured S01 render output profile.
DEFAULT_TARGET = "1080p"


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

    consolidate = sub.add_parser(
        "consolidate",
        help="consolidate a legacy SSD folder (dry-run plan / non-destructive "
        "execute + archive)",
    )
    consolidate.add_argument("path", help="path to the legacy source folder")
    mode = consolidate.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="inventory the source into an 8-group consolidation plan (default)",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="stage/verify/promote every source file into <root>/library/",
    )
    consolidate.add_argument(
        "--resume",
        action="store_true",
        help="resume a prior execute from its consolidation_journal checkpoint",
    )
    consolidate.add_argument(
        "--archive",
        action="store_true",
        help="after execute, move the fully-consolidated source under <root>/archive/",
    )
    consolidate.add_argument(
        "--json",
        action="store_true",
        help="emit structured JSON instead of a human-readable report",
    )

    health = sub.add_parser("health", help="report catalog health (exit 0)")
    health.add_argument(
        "--json",
        action="store_true",
        help="emit structured JSON instead of a human-readable summary",
    )

    scan = sub.add_parser(
        "scan", help="diff a local folder against the catalog (exit 0 changes / 3 none)"
    )
    scan.add_argument("path", help="path to the local folder to scan")
    scan.add_argument(
        "--json",
        action="store_true",
        help="emit the catalog diff as structured JSON",
    )

    analyze = sub.add_parser(
        "analyze",
        help="analyze a folder's cataloged assets (offline local engine)",
    )
    analyze.add_argument("path", help="path to the local folder to analyze")
    analyze.add_argument(
        "--profile",
        choices=("fast", "balanced", "quality", "max"),
        default="balanced",
        help="analysis workload profile (default: balanced)",
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        help="emit the analysis run report as structured JSON",
    )

    propose = sub.add_parser(
        "propose",
        help="rank art-direction treatments for a single cataloged asset",
    )
    propose.add_argument(
        "asset", help="path to a cataloged local media asset"
    )
    propose.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"render target (default: {DEFAULT_TARGET})",
    )
    propose.add_argument(
        "--json",
        action="store_true",
        help="emit the ranked proposals as structured JSON",
    )

    manifest = sub.add_parser(
        "manifest",
        help="build an Art Direction Manifest for a cataloged asset",
    )
    manifest.add_argument(
        "asset", help="path to a cataloged local media asset"
    )
    manifest.add_argument(
        "--treatment",
        help="select a specific treatment from the asset's proposals",
    )
    manifest.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"resolve the manifest for a render target (default: {DEFAULT_TARGET})",
    )
    manifest.add_argument(
        "--json",
        action="store_true",
        help="emit the manifest as structured JSON",
    )

    render = sub.add_parser(
        "render",
        help="render an art-direction manifest (.json) or a media asset to a target",
    )
    render.add_argument(
        "source", help="path to an art-direction manifest (.json) or a media asset"
    )
    render.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="render target: a named target (1080p/4k) or WxH dims "
        f"(default: {DEFAULT_TARGET})",
    )
    render.add_argument(
        "--json",
        action="store_true",
        help="emit the RenderResult as structured JSON",
    )

    validate = sub.add_parser(
        "validate",
        help="validate a rendered artifact against expected provenance",
    )
    validate.add_argument("file", help="path to the rendered artifact file")
    validate.add_argument(
        "--expected-sha",
        required=True,
        help="expected SHA-256 of the artifact bytes",
    )
    validate.add_argument(
        "--target",
        required=True,
        help="expected target dims (named 1080p/4k or WxH)",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        help="emit the ValidationReport as structured JSON",
    )

    review = sub.add_parser(
        "review",
        help="review catalog approval state (list / approve / reject / undo)",
    )
    review.add_argument(
        "--status",
        choices=("approved", "rejected", "pending"),
        help="only list entries whose state matches (default: all)",
    )
    review.add_argument(
        "--json",
        action="store_true",
        help="emit the review list as a JSON array",
    )
    review_sub = review.add_subparsers(dest="review_command")
    review_sub.add_parser("list", help="list catalog entries with their approval state")

    for verb in ("approve", "reject"):
        act = review_sub.add_parser(verb, help=f"{verb} a cataloged asset (or --batch of them)")
        act.add_argument("asset", nargs="?", help="path to the cataloged asset")
        act.add_argument(
            "--batch",
            help="comma-separated cataloged asset paths to act on",
        )
        act.add_argument(
            "--rationale",
            default="",
            help="rationale recorded with the decision",
        )

    undo = review_sub.add_parser("undo", help="revert the latest decision on an asset")
    undo.add_argument("asset", help="path to the cataloged asset")

    headless = sub.add_parser(
        "headless",
        help="start the headless server (launchd/Docker packaging entrypoint; "
        "also reachable as `--headless start`)",
    )
    headless.add_argument(
        "--config",
        help="path to a JSON config file (data_root / secrets overrides)",
    )
    headless_sub = headless.add_subparsers(dest="headless_command", required=True)
    start = headless_sub.add_parser("start", help="start the headless curator API")
    start.add_argument(
        "--accelerator",
        choices=("cpu", "cuda"),
        default=None,
        help="prefer an accelerator (default: CURATOR_ACCELERATOR or cpu)",
    )
    start.add_argument(
        "--check",
        action="store_true",
        help="dry run: resolve config, emit status JSON, exit without binding the "
        "server (deterministic, used by tests and ops preflight)",
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


def _consolidate(args: argparse.Namespace) -> int:
    """Run the S03 consolidate surface: dry-run plan or non-destructive execute.

    ``--dry-run`` (the default) inventories the source directory into an
    :class:`ConsolidationPlan` covering all 8 R002 groups and prints a
    human-readable (or ``--json``) report. ``--execute`` runs the
    :class:`ConsolidationExecutor` to stage/verify/promote every source file,
    optionally resuming from a prior run's ``consolidation_journal`` checkpoint
    (``--resume``) and requesting the explicitly-approved archive step
    (``--archive``) once every file reached ``promoted``.
    """
    source = Path(args.path).resolve()
    if not source.is_dir():
        raise CuratorError(f"consolidate source is not a directory: {args.path!r}")
    if not args.execute:
        return _consolidate_dry_run(args, source)
    return _consolidate_execute(args, source)


def _consolidate_dry_run(args: argparse.Namespace, source: Path) -> int:
    """Inventory *source* into a plan and report it (human-readable or JSON)."""
    plan = build_plan(source)
    if args.json:
        print(plan.to_json())
    else:
        print(_format_plan(plan))
    return 0


def _consolidate_execute(args: argparse.Namespace, source: Path) -> int:
    """Execute (and optionally archive) the consolidation of *source*."""
    executor = ConsolidationExecutor(source)
    result = executor.execute(resume=args.resume)
    archive_path: Path | None = None
    if args.archive:
        archive_path = executor.archive()
    if args.json:
        payload = result.to_dict()
        if archive_path is not None:
            payload["archive_path"] = str(archive_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_format_result(result))
        if archive_path is not None:
            print(f"  archived to {archive_path}")
    return 0


def _health(args: argparse.Namespace) -> int:
    """Report catalog health: status plus the total catalog entry count.

    Opens a :class:`Catalog` (auto-migrating, honoring ``CURATOR_DATA_ROOT``)
    and prints ``{"status": "healthy", "catalog_entries": N}`` when ``--json``
    is given, else a small human-readable summary. Returns 0 on success.
    """
    catalog = Catalog()
    try:
        count = catalog.count_catalog_entries()
    finally:
        catalog.db.close()
    if args.json:
        print(json.dumps({"status": "healthy", "catalog_entries": count}))
    else:
        print(f"curator: healthy ({count} catalog entries)")
    return 0


def _scan(args: argparse.Namespace) -> int:
    """Diff *args.path* against the catalog; return 0 if drift, 3 if none.

    Opens a :class:`Catalog` (auto-migrating, honoring ``CURATOR_DATA_ROOT``),
    builds a :class:`LocalConnector` over *args.path*, and computes a read-only
    :class:`ScanDiff` (never writes to catalog/journal tables). Prints the diff
    as JSON (``--json``) or a human summary. Returns :data:`EXIT_NO_CHANGE` (3)
    when ``no_changes`` is true, else :data:`EXIT_OK` (0).
    """
    folder = Path(args.path).resolve()
    if not folder.is_dir():
        raise CuratorError(f"scan source is not a directory: {args.path!r}")
    catalog = Catalog()
    try:
        diff = scan_connector(LocalConnector(folder), catalog)
    finally:
        catalog.db.close()
    if args.json:
        print(diff.to_json())
    else:
        print(_format_scan(diff))
    return EXIT_NO_CHANGE if diff.no_changes else EXIT_OK


def _format_scan(diff: ScanDiff) -> str:
    """Render a :class:`ScanDiff` as a human-readable drift summary."""
    lines = [
        f"scan {diff.connector_id}",
        f"  new     : {len(diff.new)}",
        f"  changed : {len(diff.changed)}",
        f"  missing : {len(diff.missing)}",
    ]
    if diff.new:
        lines.append("")
        lines.append("new:")
        lines.extend(f"  {Path(e['asset_id']).name}" for e in diff.new)
    if diff.changed:
        lines.append("")
        lines.append("changed:")
        lines.extend(f"  {Path(e['asset_id']).name}" for e in diff.changed)
    if diff.missing:
        lines.append("")
        lines.append("missing:")
        lines.extend(f"  {Path(asset_id).name}" for asset_id in diff.missing)
    if diff.no_changes:
        lines.append("")
        lines.append("no changes")
    return "\n".join(lines)


def _analyze(args: argparse.Namespace) -> int:
    """Run the local analysis engine over *args.path*'s cataloged assets.

    Maps every media file under *args.path* (LocalConnector semantics) to its
    catalog entry id via :func:`resolve_catalog_entry`; assets that have not
    been cataloged (e.g. RAW/corrupt files ingest skipped) are excluded. Runs
    :class:`AnalysisPipeline` over those assets and prints a human-readable
    report (specs via ``--profile``) or the :class:`AnalysisRunReport` as JSON
    (``--json``). Returns :data:`EXIT_OK` (0) when nothing was corrupt/errored,
    else :data:`EXIT_PARTIAL` (1).
    """
    folder = Path(args.path).resolve()
    if not folder.is_dir():
        raise CuratorError(f"analyze source is not a directory: {args.path!r}")
    connector = LocalConnector(folder)
    profile = AnalysisProfile(args.profile)
    catalog = Catalog()
    try:
        assets: list[AnalysisAsset] = []
        for meta in connector.enumerate():
            try:
                entry_id = resolve_catalog_entry(
                    catalog, connector.connector_id, meta.asset_id
                )
            except CatalogEntryNotFound:
                continue  # not cataloged — not part of this run
            assets.append(AnalysisAsset(entry_id=entry_id, source=meta.asset_id))
        report = AnalysisPipeline(
            catalog, provider=LocalAnalysisProvider(), profile=profile
        ).run(assets)
    finally:
        catalog.db.close()
    if args.json:
        print(report.to_json())
    else:
        print(_format_analysis(report))
    return EXIT_PARTIAL if (report.corrupt_count + report.error_count) > 0 else EXIT_OK


def _format_analysis(report) -> str:
    """Render an :class:`AnalysisRunReport` as a human-readable summary."""
    lines = [
        f"analyze {report.profile} ({report.provider_version})",
        f"  total     : {report.total_assets}",
        f"  analyzed  : {report.analyzed_count}",
        f"  corrupt   : {report.corrupt_count}",
        f"  error     : {report.error_count}",
    ]
    corrupt = [e for e in report.entries if e.status in ("corrupt", "error")]
    if corrupt:
        lines.append("")
        lines.append("corrupt / error:")
        for entry in corrupt:
            reason = f"  -> {entry.reason}" if entry.reason else ""
            lines.append(f"  entry-{entry.entry_id} ({entry.status}){reason}")
    return "\n".join(lines)


def _format_plan(plan: ConsolidationPlan) -> str:
    """Render a :class:`ConsolidationPlan` as a human-readable dry-run report."""
    counts = plan.group_counts()
    lines = [
        f"consolidate {plan.source_path} (dry-run)",
        f"  exact_dupes          : {counts['exact_dupes']}",
        f"  near_dupes           : {counts['near_dupes']}",
        f"  higher_res_originals : {counts['higher_res_originals']}",
        f"  filename_collisions  : {counts['filename_collisions']}",
        f"  panels               : {counts['panels']}",
        f"  sidecars             : {counts['sidecars']}",
        f"  corrupt              : {counts['corrupt']}",
        f"  missing_date         : {counts['missing_date']}",
    ]

    if plan.exact_dupes:
        lines.append("")
        lines.append("exact_dupes:")
        for group in plan.exact_dupes:
            lines.append("  " + ", ".join(group))
    if plan.near_dupes:
        lines.append("")
        lines.append("near_dupes:")
        for group in plan.near_dupes:
            lines.append("  " + ", ".join(group))
    if plan.higher_res_originals:
        lines.append("")
        lines.append("higher_res_originals:")
        lines.extend(f"  {rel}" for rel in plan.higher_res_originals)
    if plan.filename_collisions:
        lines.append("")
        lines.append("filename_collisions:")
        for group in plan.filename_collisions:
            lines.append("  " + ", ".join(group))
    if plan.panels:
        lines.append("")
        lines.append("panels:")
        lines.extend(f"  {rel}" for rel in plan.panels)
    if plan.sidecars:
        lines.append("")
        lines.append("sidecars:")
        lines.extend(f"  {rel}" for rel in plan.sidecars)
    if plan.corrupt:
        lines.append("")
        lines.append("corrupt:")
        for entry in plan.corrupt:
            lines.append(f"  {entry.get('path')}  -> {entry.get('error')}")
    if plan.missing_date:
        lines.append("")
        lines.append("missing_date:")
        lines.extend(f"  {rel}" for rel in plan.missing_date)
    return "\n".join(lines)


def _format_result(result) -> str:
    """Render a :class:`ConsolidationResult` as a human-readable execute report."""
    return "\n".join(
        [
            f"consolidate {result.source_path} (execute)",
            f"  staged         : {result.staged}",
            f"  verified       : {result.verified}",
            f"  promoted       : {result.promoted}",
            f"  skipped        : {result.skipped}",
            f"  unique library : {result.unique_library_files}",
        ]
    )


def _target_dims(target: str) -> tuple[int, int]:
    """Return ``(width, height)`` for a known render *target*.

    Raises :class:`CuratorError` for an unknown target so the CLI surfaces a
    fatal (exit 2) rather than silently guessing.
    """
    dims = _TARGET_SPECS.get(target)
    if dims is None:
        raise CuratorError(
            f"unknown target {target!r} (known: {', '.join(sorted(_TARGET_SPECS))})"
        )
    return dims


def _resolve_asset(path: Path) -> tuple[Catalog, int, str]:
    """Return the catalog entry id for *path* plus the owning :class:`Catalog`.

    The asset's parent folder supplies the LocalConnector so the connector id
    matches the one ingest wrote. Raises :class:`CatalogEntryNotFound` (and
    through ``main``, a fatal exit 2) when the asset has not been cataloged.
    """
    connector = LocalConnector(path.parent)
    catalog = Catalog()
    try:
        entry_id = resolve_catalog_entry(catalog, connector.connector_id, str(path))
    except Exception:
        catalog.db.close()
        raise
    return catalog, entry_id, str(path)


def _load_analysis(catalog: Catalog, entry_id: int) -> AnalysisResult | None:
    """Return the newest persisted ``ok`` :class:`AnalysisResult` for *entry_id*.

    Used by ``propose`` / ``manifest`` to reuse a prior deterministic analysis
    (the documented reuse path) instead of re-analyzing, when one exists.
    """
    row = catalog.db.execute(
        "SELECT analysis_json FROM analysis_results"
        " WHERE catalog_entry_id = ? AND status = 'ok' ORDER BY id DESC LIMIT 1",
        (entry_id,),
    ).fetchone()
    if row is None:
        return None
    return AnalysisResult.from_dict(json.loads(row[0]))


def _analyze_or_reuse(
    catalog: Catalog, entry_id: int, source: str, request: ArtDirectionRequest
) -> list[TreatmentProposal]:
    """Run the policy engine, reusing persisted analysis when available.

    Prefers the newest persisted ``ok`` analysis (deterministic + fast), falling
    back to a fresh local analysis of *source* bytes otherwise. Returns the
    ranked :class:`TreatmentProposal` list.
    """
    persisted = _load_analysis(catalog, entry_id)
    if persisted is not None:
        return policy_propose([source], request, analysis=[persisted])
    return policy_propose([source], request, provider=LocalAnalysisProvider())


def _propose(args: argparse.Namespace) -> int:
    """Propose art-direction treatments for a single cataloged asset.

    Resolves *args.asset* to its catalog entry, derives an :class:`AnalysisResult`
    (reusing a persisted one, else analyzing fresh), runs the S03 policy engine,
    persists the resulting proposals to ``proposals`` (append-only), and prints
    them ranked (human or ``--json``). Returns :data:`EXIT_OK` (0) when proposals
    exist, :data:`EXIT_NO_CHANGE` (3) when the policy produced none.
    """
    path = Path(args.asset).resolve()
    if not path.is_file():
        raise CuratorError(f"propose source is not a file: {args.asset!r}")
    width, height = _target_dims(args.target)
    request = ArtDirectionRequest(
        target=args.target,
        target_width=width,
        target_height=height,
        sources=[str(path)],
    )
    catalog, entry_id, source = _resolve_asset(path)
    try:
        proposals = _analyze_or_reuse(catalog, entry_id, source, request)
        _persist_proposals(catalog, entry_id, proposals)
    finally:
        catalog.db.close()
    if not proposals:
        return EXIT_NO_CHANGE
    if args.json:
        print(json.dumps([p.to_dict() for p in proposals], indent=2, ensure_ascii=False))
    else:
        print(_format_proposals(proposals))
    return EXIT_OK


def _manifest(args: argparse.Namespace) -> int:
    """Build and persist an :class:`ArtDirectionManifest` for a cataloged asset.

    Chooses a :class:`TreatmentProposal` (the ``--treatment`` one, else the
    top-ranked) for the asset's catalog entry, materializes a base manifest via
    :func:`materialize_manifest`, attaches deterministic per-target overrides,
    resolves the base for ``--target``, and persists the validated result to
    ``art_direction_manifests`` (append-only). Returns :data:`EXIT_OK` (0).
    """
    path = Path(args.asset).resolve()
    if not path.is_file():
        raise CuratorError(f"manifest source is not a file: {args.asset!r}")
    base_width, base_height = _target_dims(DEFAULT_TARGET)
    request = ArtDirectionRequest(
        target=DEFAULT_TARGET,
        target_width=base_width,
        target_height=base_height,
        sources=[str(path)],
    )
    catalog, entry_id, source = _resolve_asset(path)
    try:
        proposal = _select_proposal(catalog, entry_id, args.treatment, source, request)
        base = _attach_target_overrides(
            materialize_manifest(proposal, request, [source])
        )
        manifest = base.resolved_for(args.target)
        _persist_manifest(catalog, entry_id, manifest)
    finally:
        catalog.db.close()
    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(_format_manifest(manifest))
    return EXIT_OK


def _select_proposal(
    catalog: Catalog,
    entry_id: int,
    treatment: str | None,
    source: str,
    request: ArtDirectionRequest,
) -> TreatmentProposal:
    """Return the chosen proposal for *entry_id*, generating it if needed.

    Prefers an already-persisted proposal (matching *treatment* when given),
    else falls back to running the policy engine (persisting its output).
    Raises :class:`CuratorError` when no proposal can be produced.
    """
    rows = _load_proposals(catalog, entry_id, treatment)
    if not rows:
        fresh = _analyze_or_reuse(catalog, entry_id, source, request)
        _persist_proposals(catalog, entry_id, fresh)
        rows = _load_proposals(catalog, entry_id, treatment)
    if not rows:
        raise CuratorError(f"no proposal available to manifest for catalog entry {entry_id}")
    return rows[0]


def _load_proposals(
    catalog: Catalog, entry_id: int, treatment: str | None = None
) -> list[TreatmentProposal]:
    """Return persisted proposals for *entry_id*, ranked by score descending."""
    if treatment is not None:
        cur = catalog.db.execute(
            "SELECT treatment, score, rationale_json, evidence_json FROM proposals"
            " WHERE catalog_entry_id = ? AND treatment = ? ORDER BY score DESC, id",
            (entry_id, treatment),
        )
    else:
        cur = catalog.db.execute(
            "SELECT treatment, score, rationale_json, evidence_json FROM proposals"
            " WHERE catalog_entry_id = ? ORDER BY score DESC, id",
            (entry_id,),
        )
    out: list[TreatmentProposal] = []
    for row in cur.fetchall():
        out.append(
            TreatmentProposal.from_dict(
                {
                    "treatment": row[0],
                    "score": row[1],
                    "rationale": json.loads(row[2]) if row[2] else [],
                    "evidence": json.loads(row[3]) if row[3] else {},
                }
            )
        )
    return out


def _persist_proposals(
    catalog: Catalog, entry_id: int, proposals: list[TreatmentProposal]
) -> None:
    """Append *proposals* to the append-only ``proposals`` table (schema v5)."""
    for proposal in proposals:
        catalog.db.execute(
            "INSERT INTO proposals"
            " (catalog_entry_id, treatment, score, rationale_json, evidence_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                entry_id,
                proposal.treatment.value,
                proposal.score,
                json.dumps(proposal.rationale),
                json.dumps(proposal.evidence),
            ),
        )
    catalog.db.commit()


def _persist_manifest(
    catalog: Catalog, entry_id: int, manifest: ArtDirectionManifest
) -> None:
    """Append a manifest row to ``art_direction_manifests`` (schema v5)."""
    catalog.db.execute(
        "INSERT INTO art_direction_manifests"
        " (catalog_entry_id, manifest_version, manifest_json) VALUES (?, ?, ?)",
        (entry_id, MANIFEST_VERSION, json.dumps(manifest.to_dict())),
    )
    catalog.db.commit()


def _attach_target_overrides(base: ArtDirectionManifest) -> ArtDirectionManifest:
    """Attach deterministic per-target overrides to *base*.

    A non-default target (currently ``4k``) carries a processing override that
    records the upscaling risk of rendering a 1080p-class source up to 4K, so
    ``resolved_for(target)`` has an observable effect (override precedence).
    """
    base_dims = _TARGET_SPECS[DEFAULT_TARGET]
    overrides: dict[str, dict] = {}
    for name, dims in _TARGET_SPECS.items():
        if dims != base_dims:
            overrides[name] = {"processing_intent": {"upscale_warning": True}}
    if not overrides:
        return base
    data = base.to_dict()
    data["target_overrides"] = overrides
    return ArtDirectionManifest.from_dict(data)


def _format_proposals(proposals: list[TreatmentProposal]) -> str:
    """Render a ranked :class:`TreatmentProposal` list for humans."""
    lines = ["proposals (ranked):"]
    for idx, proposal in enumerate(proposals, 1):
        lines.append(f"  {idx}. {proposal.treatment.value}  score={proposal.score:.4f}")
        lines.extend(f"      - {r}" for r in proposal.rationale)
    return "\n".join(lines)


def _format_manifest(manifest: ArtDirectionManifest) -> str:
    """Render an :class:`ArtDirectionManifest` as a human-readable summary."""
    lines = [
        f"manifest v{manifest.manifest_version}"
        f"  treatment={manifest.layout_treatment.value}",
        f"  sources     : {', '.join(manifest.sources) if manifest.sources else '-'}",
        f"  background  : {manifest.background.background_choice}",
    ]
    if manifest.rationale:
        lines.append("  rationale   :")
        lines.extend(f"      - {r}" for r in manifest.rationale)
    if manifest.target_overrides:
        lines.append(
            "  target overrides: " + ", ".join(sorted(manifest.target_overrides))
        )
    return "\n".join(lines)


def _render_target_dims(target: str) -> tuple[int, int]:
    """Return ``(width, height)`` for a named target or an explicit ``WxH`` string.

    Resolves the known named targets (``1080p``/``4k``) first, then falls back to
    parsing a ``WxH`` pair. Raises :class:`CuratorError` for anything else so the
    CLI surfaces a fatal (exit 2) rather than silently guessing.
    """
    named = _TARGET_SPECS.get(target)
    if named is not None:
        return named
    if "x" in target:
        try:
            w_str, h_str = target.split("x", 1)
            width, height = int(w_str), int(h_str)
            if width > 0 and height > 0:
                return width, height
        except ValueError:
            pass
    raise CuratorError(
        f"unknown target {target!r} (use a named target or WxH dimensions)"
    )


def _render(args: argparse.Namespace) -> int:
    """Render *args.source* to *args.target*; print a summary or ``--json`` result.

    A ``.json`` source is loaded via :class:`ArtDirectionManifest` (resolved for
    *args.target*) with every referenced source sha fetched from the
    :class:`ContentStore`; any other source is treated as a media asset and
    rendered through a deterministic single-full-bleed manifest built from its
    bytes. Maps ``--target`` to pixel dims, renders via
    :class:`DeterministicRenderer`, and returns :data:`EXIT_OK` (0). An unapproved
    upscale raises :class:`RenderError` (R008), which is reported to stderr and
    returned as :data:`EXIT_FATAL` (2).
    """
    path = Path(args.source).resolve()
    if not path.is_file():
        raise CuratorError(f"render source is not a file: {args.source!r}")
    width, height = _render_target_dims(args.target)
    if path.suffix.lower() == ".json":
        try:
            manifest = ArtDirectionManifest.from_dict(
                json.loads(path.read_text(encoding="utf-8"))
            ).resolved_for(args.target)
        except (json.JSONDecodeError, CuratorError, OSError) as exc:
            print(f"curator: error: failed to load manifest {path}: {exc}", file=sys.stderr)
            return EXIT_FATAL
        sources = {sha: ContentStore().get(sha) for sha in manifest.sources}
    else:
        data = path.read_bytes()
        sha = sha256_hex(data)
        manifest = ArtDirectionManifest(
            sources=[sha],
            regions=[SourceRegion(source_sha256=sha)],
            layout_treatment=LayoutTreatment.SINGLE_FULLBLEED,
        )
        sources = {sha: data}
    renderer = DeterministicRenderer()
    try:
        result = renderer.render(manifest, sources, (width, height))
    except RenderError as exc:
        print(f"curator: error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(_format_render(result))
    return EXIT_OK


def _format_render(result: RenderResult) -> str:
    """Render a :class:`RenderResult` as a human-readable summary."""
    return "\n".join(
        [
            f"render {result.target_width}x{result.target_height}",
            f"  treatment : {result.treatment}",
            f"  sha256    : {result.sha256}",
            f"  size      : {result.size_bytes} bytes",
            f"  profile   : {result.color_profile} ({result.color_mode})",
            f"  sources   : {', '.join(result.sources)}",
            f"  upscaled  : {result.upscaled_warning}",
        ]
    )


def _validate(args: argparse.Namespace) -> int:
    """Gate a rendered artifact against expected provenance (dimensions + SHA-256).

    Reads *args.file*'s bytes and runs :class:`ArtifactValidator` against the
    expected SHA-256 and parsed ``--target`` dims. Prints a human-readable summary
    (publishable flag plus every failing check's reason) or the
    :class:`ValidationReport` as JSON. Returns :data:`EXIT_OK` (0) when
    publishable, :data:`EXIT_PARTIAL` (1) otherwise, and :data:`EXIT_FATAL` (2)
    on a read or target parse error.
    """
    path = Path(args.file).resolve()
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(f"curator: error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    try:
        width, height = _render_target_dims(args.target)
    except CuratorError as exc:
        print(f"curator: error: {exc}", file=sys.stderr)
        return EXIT_FATAL
    report = ArtifactValidator().validate(data, args.expected_sha, (width, height))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(_format_validate(report))
    return EXIT_OK if report.publishable else EXIT_PARTIAL


def _format_validate(report: ValidationReport) -> str:
    """Render a :class:`ValidationReport` as a human-readable summary."""
    lines = [f"validate  publishable : {str(report.publishable).lower()}"]
    failed = [c for c in report.checks if not c.passed]
    if failed:
        lines.append("  failing checks:")
        for check in failed:
            lines.append(f"    - {check.name}: {check.reason}")
    return "\n".join(lines)


def _review(args: argparse.Namespace) -> int:
    """Run the review surface: list approval state, or apply a decision.

    With no subcommand (or ``list``), lists every catalog entry with its current
    ApprovalService state (optionally filtered by ``--status``). The
    ``approve``/``reject``/``undo`` subcommands resolve each asset to its catalog
    entry (LocalConnector semantics) and record the transition, printing a
    confirmation. Unknown assets raise :class:`CatalogEntryNotFound` (mapped to a
    fatal exit 2 by ``main``).
    """
    catalog = Catalog()
    try:
        if args.review_command in ("approve", "reject", "undo"):
            return _review_update(args, catalog)
        return _review_list(args, catalog)
    finally:
        catalog.db.close()


def _review_resolve(catalog: Catalog, asset: str) -> int:
    """Return the catalog entry id for *asset* (LocalConnector semantics).

    The asset's parent folder supplies the connector id matching ingest, exactly
    as ``propose``/``manifest`` do via :func:`_resolve_asset`.
    """
    path = Path(asset).resolve()
    connector = LocalConnector(path.parent)
    return resolve_catalog_entry(catalog, connector.connector_id, str(path))


def _review_list(args: argparse.Namespace, catalog: Catalog) -> int:
    """List catalog entries with their current approval state (filterable)."""
    approval = ApprovalService(catalog)
    rows: list[dict] = []
    for entry in catalog.list_entries():
        current = approval.current(entry["id"])
        decision = current.decision.value.lower() if current else None
        rows.append(
            {
                "asset_id": entry["asset_id"],
                "entry_id": entry["id"],
                "decision": decision,
            }
        )
    if args.status:
        status = args.status
        rows = [
            r for r in rows
            if (r["decision"] is None if status == "pending" else r["decision"] == status)
        ]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        if not rows:
            print("review: no catalog entries")
        else:
            print("review")
            for r in rows:
                print(
                    f"  {Path(r['asset_id']).name}  {r['decision'] or 'pending'}"
                    f"  (entry {r['entry_id']})"
                )
    return EXIT_OK


def _review_update(args: argparse.Namespace, catalog: Catalog) -> int:
    """Apply an approve/reject/undo (or --batch) decision and confirm it."""
    approval = ApprovalService(catalog)
    if args.review_command == "undo":
        performed = [(args.asset, _review_resolve(catalog, args.asset))]
        approval.undo(performed[0][1])
    elif args.batch:
        assets = [a.strip() for a in args.batch.split(",") if a.strip()]
        if not assets:
            raise CuratorError(f"review {args.review_command}: --batch provided no assets")
        performed = [(a, _review_resolve(catalog, a)) for a in assets]
        if args.review_command == "approve":
            approval.batch_approve([eid for _, eid in performed], args.rationale)
        else:
            for _a, eid in performed:
                approval.reject(eid, args.rationale)
    else:
        if not args.asset:
            raise CuratorError(
                f"review {args.review_command}: no asset — pass <asset> or --batch"
            )
        performed = [(args.asset, _review_resolve(catalog, args.asset))]
        if args.review_command == "approve":
            approval.approve(performed[0][1], args.rationale)
        else:
            approval.reject(performed[0][1], args.rationale)
    for asset, entry_id in performed:
        event = approval.current(entry_id)
        state = event.decision.value.lower() if event else "pending"
        print(f"review {args.review_command}: {Path(asset).name} (entry {entry_id}) -> {state}")
    return EXIT_OK


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
    if args.command == "consolidate":
        return _consolidate(args)
    if args.command == "health":
        return _health(args)
    if args.command == "scan":
        return _scan(args)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "propose":
        return _propose(args)
    if args.command == "manifest":
        return _manifest(args)
    if args.command == "render":
        return _render(args)
    if args.command == "validate":
        return _validate(args)
    if args.command == "review":
        return _review(args)
    if args.command == "headless":
        return _headless_start(args)
    # Unreachable given required subparsers; defensive fallback.
    return EXIT_FATAL


def _headless_load_config(config_path: str | None) -> dict:
    """Load the optional ``--config`` JSON file as a plain dict (stdlib only).

    The file may carry ``data_root`` and ``secrets`` keys. Unknown keys are
    ignored (mirroring the config module's ``extra="ignore"`` behaviour). A
    ``data_root`` in the file is a fallback that the environment overrides.
    """
    if config_path is None:
        return {}
    path = Path(config_path)
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CuratorError(f"failed to read --config {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CuratorError(f"--config {path}: expected a JSON object")
    return parsed


def _headless_resolve_secrets(config: dict) -> dict[str, str]:
    """Resolve env-backed secret references (``*_TOKEN`` / ``*_API_KEY``).

    Explicit environment variables take precedence over values in the config
    file, matching the config module's env-over-config ordering. Only the
    presence of at least one secret is validated; values are never echoed.
    """
    secrets: dict[str, str] = dict(config.get("secrets", {}) or {})
    for key, value in os.environ.items():
        if key.endswith("_TOKEN") or key.endswith("_API_KEY"):
            secrets[key] = value
    return secrets


def _accelerator_available(name: str) -> bool:
    """Return whether the requested *name* accelerator is usable here.

    ``cuda`` is considered available only when the ``nvidia-smi`` driver tool is
    on PATH. ``cpu`` is always available. This is a conservative, side-effect-free
    probe so the headless start can degrade gracefully without probing GPU libs.
    """
    if name == "cuda":
        return shutil.which("nvidia-smi") is not None
    return True


def _headless_start(args: argparse.Namespace) -> int:
    """Resolve config and start (``--check`` preflight or live) the headless API.

    Configuration is resolved from an optional ``--config`` JSON file and/or the
    ``CURATOR_*`` environment plus ``*_TOKEN`` / ``*_API_KEY`` env-backed secret
    references, without any prompting. Required configuration is validated:
    ``data_root`` must resolve and at least one secret must be configured.

    ``--check`` (documented dry path) resolves configuration, emits the
    machine-readable JSON status, and returns without binding a server — keeping
    tests and ops preflight deterministic. Without ``--check`` the API is served
    (uvicorn, loopback-only) via :func:`curator.api.create_app`.

    When a CUDA accelerator is requested but unavailable, the service degrades to
    CPU (which remains functional): a clear actionable warning is emitted and the
    status reports ``"cpu (fallback)"`` rather than failing silently.

    Returns :data:`EXIT_OK` (0) when ready, :data:`EXIT_FATAL` (2) on invalid or
    missing required configuration.
    """
    config = _headless_load_config(args.config)
    data_root = (
        os.environ.get("CURATOR_DATA_ROOT")
        or (str(config["data_root"]) if "data_root" in config else None)
        or str(Path.home() / ".curator")
    )
    secrets = _headless_resolve_secrets(config)
    api = f"http://{api_module.HOST}:{api_module.PORT}"

    if not secrets:
        print(
            "curator: error: headless start requires at least one secret — set a "
            "*_TOKEN or *_API_KEY environment variable (or a 'secrets' map in --config)",
            file=sys.stderr,
        )
        return EXIT_FATAL

    requested = args.accelerator or os.environ.get("CURATOR_ACCELERATOR") or "cpu"
    ready = True
    if requested != "cpu" and not _accelerator_available(requested):
        print(
            f"curator: warning: accelerator {requested!r} requested but unavailable; "
            "degrading to CPU — CPU remains fully functional",
            file=sys.stderr,
        )
        requested = "cpu (fallback)"

    status = {
        "status": "ok",
        "data_root": data_root,
        "api": api,
        "ready": ready,
        "accelerator": requested,
    }
    print(json.dumps(status, ensure_ascii=False))

    if args.check:
        return EXIT_OK

    import uvicorn

    uvicorn.run(api_module.create_app(), host=api_module.HOST, port=api_module.PORT)
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``curator`` console script.

    Returns a process exit code. Errors are reported to stderr and mapped to a
    non-zero exit rather than raising, so callers in-process can assert on the
    returned code.
    """
    parser = build_parser()
    if argv and argv[0] == "--headless":
        argv = ["headless", *argv[1:]]
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (CuratorError, OSError) as exc:
        print(f"curator: error: {exc}", file=sys.stderr)
        return EXIT_FATAL


if __name__ == "__main__":  # pragma: no cover - interactive entry
    raise SystemExit(main())
