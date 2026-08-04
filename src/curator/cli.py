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
import json
import sys
from pathlib import Path

from curator import db
from curator.analysis.cli_utils import resolve_catalog_entry
from curator.analysis.errors import CatalogEntryNotFound
from curator.analysis.local import LocalAnalysisProvider
from curator.analysis.pipeline import AnalysisAsset, AnalysisPipeline
from curator.analysis.profiles import AnalysisProfile
from curator.analysis.schema import AnalysisResult
from curator.artdirection.manifest import MANIFEST_VERSION, ArtDirectionManifest
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
from curator.errors import CuratorError
from curator.ingest.pipeline import IngestPipeline
from curator.ingest.report import IngestReport
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
    # Unreachable given required subparsers; defensive fallback.
    return EXIT_FATAL


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
        return EXIT_FATAL


if __name__ == "__main__":  # pragma: no cover - interactive entry
    raise SystemExit(main())
