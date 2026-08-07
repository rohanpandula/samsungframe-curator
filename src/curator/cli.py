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
- ``curator propose ASSET...`` — rank art-direction treatments for one or more cataloged
                               assets (M010/S02): the single-source treatments plus, when the
                               group coheres, ``diptych`` at two assets, ``triptych`` at three,
                               ``quad`` at four and ``packed`` at any count up to the cap
                               (M010/S03), whose cells are sized by ``--weights`` or by each
                               asset's aesthetic quality. Over ``MAX_LAYOUT_SOURCES`` assets is
                               a fatal exit 2 — rejected, never truncated.
- ``curator manifest ASSET...`` — build an Art Direction Manifest over one or more cataloged
                               assets, optionally selecting ``--treatment`` (``diptych`` /
                               ``triptych`` / ``quad`` / ``packed`` included) and resolving it
                               for ``--target``; the manifest carries one real cell per source,
                               packed with the weights the chosen proposal recorded.
- ``curator render SOURCE``   — render an art-direction manifest (``.json``) or a media
                               asset to a target (named ``1080p``/``4k`` or ``WxH``) via the
                               deterministic renderer, printing a summary or ``--json`` result.
                               A manifest's sources are resolved by content sha (ContentStore)
                               or by path, so a ``curator manifest`` document renders directly.
- ``curator validate FILE``   — gate a rendered artifact against expected provenance
                               (``--expected-sha`` + ``--target`` dims); exit 0 publishable /
                               1 not / 2 on read or parse error.
- ``curator taste drop PATH... [--note TEXT] [--save]`` — open a reaction-room session
                               over the dropped images: record one extracted reaction
                               (``--note``, exit 0, followed by the "What I learned"
                               delta) or preview its deterministic probing questions
                               (no ``--note``, exit 3). ``--save`` is the explicit
                               choice that promotes ephemeral drops to full resolution
                               in the catalog.
- ``curator taste profile``   — print the taste profile document (vocabulary / patterns /
                               tensions / evolution) with per-claim evidence; ``--json``
                               emits the document, ``--no-seed`` drops the low-provenance
                               approval/pairwise history claims.
- ``curator taste dispute ID`` — remove a claim and mark its evidence for
                               re-interpretation on the append-only timeline (exit 3 when
                               no claim carries that id).
- ``curator taste vote [--prefer a|b] [--note TEXT]`` — preview the current A/B
                               taste comparison ``pairwise.choose_pair`` selects (no
                               flags, exit 3, records nothing) or answer it
                               (``--prefer``, exit 0), persisting the vote to
                               ``taste_preferences`` and the moved profile to
                               ``taste_profiles``.
- ``curator taste votes [--json]`` — list every recorded vote (winner/loser/note/
                               retracted status), oldest first.
- ``curator taste retract VOTE_GROUP`` — retract a vote by its ``vote_group`` id,
                               reversing its effect on the persisted profile without
                               deleting history (exit 3 when no vote carries that id).
- ``curator taste embed-status [--backfill] [--json]`` — report whether the
                               local embedding subsystem (M009/S02) is usable
                               (model placed, checksum verified if pinned) with
                               no network call either way; ``--backfill``
                               computes and stores vectors for every catalog
                               entry lacking a current-model-version embedding.
                               A per-entry failure is recorded
                               (``backfill_error_count``/``backfill_errors``)
                               and the run continues past it, never aborting
                               the whole backfill on one bad entry.
                               Always exits 0 once the probe runs — this is the
                               milestone's documented early-exit checkpoint.
- ``curator taste embedding-head [--json]`` — fit the nonparametric
                               preference head (M009/S03) over every vote
                               resolvable against S02 embeddings and report
                               ``capacity`` alongside the literal
                               retained-parameter count and a zero-vote parity
                               self-check. Always exits 0 (``ok=False`` reports
                               cleanly, same as ``embed-status``, when no model
                               is available).
- ``curator taste embedding-explain PATH_OR_SHA [--json]`` — explain one
                               embedding-head score (M009/S04): fits the head
                               fresh over every resolvable vote, then reports
                               its exact per-vote attribution (summing back to
                               the score), up to three nearest-neighbour
                               exemplars from the user's own liked images, and
                               a deterministic template ``rationale`` — no
                               saliency map, no LLM narration. ``ok=False``
                               reports cleanly (``EXIT_NO_CHANGE``) when no
                               model is available, same as ``embedding-head``.
- ``curator taste compare [--json]`` — compare the lens and embedding heads
                               over recorded votes, with uncertainty (M009/S05):
                               held-out accuracy + promotion-gate result for
                               each head independently, a discordant-pairs
                               head-to-head accuracy with a 95% confidence
                               interval, a learning curve, and a verdict of
                               ``embedding_better``/``lens_better``/``tie``/
                               ``insufficient_evidence`` — never a coin-flip
                               winner, never a path that retires the lens
                               head. ``EXIT_NO_CHANGE`` when the embedding
                               provider is unavailable or fewer than two
                               analyzed candidates exist yet.

The CLI resolves all paths through the six-axis config (``CURATOR_DATA_ROOT``), so it
honors that environment variable wherever it points the data root.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

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
    _TREATMENT_SOURCE_COUNT,
    MANIFEST_VERSION,
    MAX_LAYOUT_SOURCES,
    MULTI_CELL_TREATMENTS,
    ArtDirectionManifest,
    LayoutTreatment,
    SourceRegion,
)
from curator.artdirection.packing import resolve_regions
from curator.artdirection.policy import (
    ArtDirectionRequest,
    TreatmentProposal,
    materialize_manifest,
    propose_treatments,
)
from curator.catalog import Catalog
from curator.config import CuratorConfig
from curator.connectors.local import LocalConnector
from curator.consolidate import ConsolidationExecutor, ConsolidationPlan, build_plan
from curator.content_store import ContentStore
from curator.errors import CuratorError, StorageError
from curator.hashing import sha256_hex
from curator.ingest.pipeline import IngestPipeline
from curator.ingest.report import IngestReport
from curator.render.renderer import DeterministicRenderer, RenderError, RenderResult
from curator.render.validate import ArtifactValidator, ValidationReport
from curator.scan import ScanDiff, scan_connector
from curator.taste.dialogue.extraction import (
    extraction_config_from_env,
    resolve_extraction_provider,
)
from curator.taste.dialogue.observation import create_observation
from curator.taste.dialogue.profile import (
    ColdStartSeeder,
    ProfileBuilder,
    ProfileEvent,
    ProfileStore,
    WhatILearned,
)
from curator.taste.dialogue.profile import (
    TasteProfile as DialogueProfile,
)
from curator.taste.dialogue.retention import save_to_catalog
from curator.taste.dialogue.room import ReactionRoom
from curator.taste.dialogue.session import TasteSession
from curator.taste.dialogue.store import ObservationStore
from curator.taste.embedding.attribution import attribute_score, find_exemplars, render_rationale
from curator.taste.embedding.compare import (
    HELD_OUT_FRACTION,
    MIN_DISCORDANT_PAIRS,
    compare_heads,
)
from curator.taste.embedding.errors import EmbeddingError
from curator.taste.embedding.head import (
    VoteVectors,
    fit_embedding_head,
    resolve_vote_vectors,
)
from curator.taste.embedding.provider import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL_VERSION,
    OnnxEmbeddingProvider,
)
from curator.taste.embedding.store import EmbeddingStore
from curator.taste.pairwise import Scorer
from curator.taste.rank import TasteRanker
from curator.taste.store import TasteVoteStore, VoteRecord, next_pair, resolve_vote_candidates

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

# 64-hex content sha256 (mirrors ``api.py``'s ``_SHA_RE``) — a ``path_or_sha`` argument
# fully matching this pattern is a direct content sha lookup, not a filesystem path.
_SHA_RE = re.compile(r"[0-9a-fA-F]{64}")

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
        help="rank art-direction treatments for one or more cataloged assets",
    )
    propose.add_argument(
        "asset", nargs="+", help="path(s) to cataloged local media asset(s)"
    )
    propose.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help=f"render target (default: {DEFAULT_TARGET})",
    )
    propose.add_argument(
        "--weights",
        help=(
            "comma-separated importance weights, one per asset in the same "
            "order, sizing the packed layout's cells (default: each asset's "
            "analyzed aesthetic quality)"
        ),
    )
    propose.add_argument(
        "--json",
        action="store_true",
        help="emit the ranked proposals as structured JSON",
    )

    manifest = sub.add_parser(
        "manifest",
        help="build an Art Direction Manifest for one or more cataloged assets",
    )
    manifest.add_argument(
        "asset", nargs="+", help="path(s) to cataloged local media asset(s)"
    )
    manifest.add_argument(
        "--treatment",
        help=(
            "select a specific treatment from the assets' proposals: "
            "single_fullbleed, contain_matte, panoramic, square, diptych, "
            "triptych, quad, packed (packed lays out any 2-9 assets, sized by "
            "the weights the proposal recorded)"
        ),
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
        "--manifest",
        help="check the artifact's cell geometry against an Art Direction Manifest",
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

    taste = sub.add_parser(
        "taste", help="taste dialogue: reaction room, profile, disputes"
    )
    taste_sub = taste.add_subparsers(dest="taste_command", required=True)

    drop = taste_sub.add_parser(
        "drop",
        help="record a reaction to dropped image(s), or preview probing questions",
    )
    drop.add_argument(
        "paths",
        nargs="+",
        help="image path(s): a cataloged asset path or a third-party ephemeral file",
    )
    drop.add_argument(
        "--note",
        default=None,
        help="reaction text in the user's own words; omit to preview probing questions",
    )
    drop.add_argument(
        "--save",
        action="store_true",
        help="also save the dropped third-party image(s) into the catalog at full "
        "resolution (retention keeps thumbnails + hashes only without this)",
    )

    taste_profile = taste_sub.add_parser(
        "profile",
        help="print the taste profile (vocabulary / patterns / tensions / evolution)",
    )
    taste_profile.add_argument(
        "--json",
        action="store_true",
        help="emit the profile document as JSON instead of readable text",
    )
    taste_profile.add_argument(
        "--no-seed",
        action="store_true",
        help="omit low-provenance claims seeded from approval/pairwise history",
    )

    dispute = taste_sub.add_parser(
        "dispute",
        help="dispute a profile claim: removes it and marks its evidence for "
        "re-interpretation",
    )
    dispute.add_argument("claim_id", help="profile claim id to dispute")
    dispute.add_argument(
        "--json",
        action="store_true",
        help="emit the recorded dispute event as JSON",
    )

    vote = taste_sub.add_parser(
        "vote",
        help="preview the current A/B taste comparison, or answer it with --prefer",
    )
    vote.add_argument(
        "--prefer",
        choices=["a", "b"],
        default=None,
        help="which shown candidate you prefer; omit to preview the pair without "
        "recording a vote",
    )
    vote.add_argument("--note", default="", help="optional note recorded with the vote")
    vote.add_argument(
        "--json",
        action="store_true",
        help="emit the pair/vote result as JSON",
    )

    votes = taste_sub.add_parser("votes", help="list recorded pairwise votes")
    votes.add_argument(
        "--json",
        action="store_true",
        help="emit the vote history as JSON",
    )

    retract = taste_sub.add_parser(
        "retract",
        help="retract a recorded vote, reversing its effect on the persisted profile",
    )
    retract.add_argument("vote_group", help="vote_group id to retract (see `taste votes`)")
    retract.add_argument(
        "--json",
        action="store_true",
        help="emit the retraction result as JSON",
    )

    embed_status = taste_sub.add_parser(
        "embed-status",
        help="report whether the local embedding subsystem is usable (M009/S02)",
    )
    embed_status.add_argument(
        "--backfill",
        action="store_true",
        help="compute + store embeddings for every catalog entry lacking one "
        "(only runs when the probe reports ok)",
    )
    embed_status.add_argument(
        "--json",
        action="store_true",
        help="emit the probe/backfill result as JSON",
    )

    embedding_head = taste_sub.add_parser(
        "embedding-head",
        help="fit the nonparametric embedding preference head over resolvable "
        "votes (M009/S03)",
    )
    embedding_head.add_argument(
        "--json",
        action="store_true",
        help="emit the fit result as JSON",
    )

    embedding_explain = taste_sub.add_parser(
        "embedding-explain",
        help="explain one embedding-head score: per-vote attribution + nearest "
        "liked exemplars (M009/S04)",
    )
    embedding_explain.add_argument(
        "path_or_sha", help="image path (LocalConnector semantics) or 64-hex content sha256"
    )
    embedding_explain.add_argument(
        "--json",
        action="store_true",
        help="emit the explanation as JSON",
    )

    compare = taste_sub.add_parser(
        "compare",
        help="compare the lens and embedding heads over recorded votes, with "
        "uncertainty (M009/S05)",
    )
    compare.add_argument(
        "--json",
        action="store_true",
        help="emit the comparison report as JSON",
    )

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


def _asset_paths(assets: list[str] | str, command: str) -> list[Path]:
    """Resolve the ``asset`` positional(s) to existing files (M010/S02).

    The :data:`MAX_LAYOUT_SOURCES` cap is checked **first**, before any path is
    touched and long before any catalog resolution or analysis runs, so an
    over-cap request costs zero work — and is rejected with an actionable
    message rather than silently truncated to the first N.
    """
    raw = [assets] if isinstance(assets, str) else list(assets)
    if len(raw) > MAX_LAYOUT_SOURCES:
        raise CuratorError(
            f"{command} accepts at most {MAX_LAYOUT_SOURCES} assets, got {len(raw)} "
            f"— an over-cap request is rejected, never truncated"
        )
    paths: list[Path] = []
    for asset in raw:
        path = Path(asset).resolve()
        if not path.is_file():
            raise CuratorError(f"{command} source is not a file: {asset!r}")
        paths.append(path)
    return paths


def _parse_weights(raw: str | None, count: int) -> list[float] | None:
    """Parse ``propose --weights`` into one float per asset (M010/S03).

    Returns ``None`` when the flag was not given, so the policy engine falls back
    to its default weight source. A count mismatch or an unparseable value is a
    fatal :class:`CuratorError` (exit 2) naming what was wrong — never a padded,
    truncated or silently-dropped weight vector, which would produce a layout the
    caller did not ask for.
    """
    if raw is None:
        return None
    weights: list[float] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        try:
            weights.append(float(text))
        except ValueError:
            raise CuratorError(
                f"--weights value is not a number: {text!r} — pass one "
                f"comma-separated weight per asset (e.g. --weights 0.9,0.4,0.4)"
            ) from None
    if len(weights) != count:
        raise CuratorError(
            f"--weights has {len(weights)} value(s) for {count} asset(s) — pass "
            f"one weight per asset, in the same order"
        )
    return weights


def _resolve_assets(paths: list[Path]) -> tuple[Catalog, list[int], list[str]]:
    """Return the catalog entry ids for *paths* plus the one owning :class:`Catalog`.

    Exactly one :class:`Catalog` is opened for the whole batch (never one per
    asset) and closed on any failure, so a partially-resolved multi-asset request
    leaks no connection. Each asset's parent folder supplies the LocalConnector
    so the connector id matches the one ingest wrote. Raises
    :class:`CatalogEntryNotFound` (and through ``main``, a fatal exit 2) for the
    first asset that has not been cataloged.
    """
    catalog = Catalog()
    entry_ids: list[int] = []
    sources: list[str] = []
    try:
        for path in paths:
            connector = LocalConnector(path.parent)
            entry_ids.append(
                resolve_catalog_entry(catalog, connector.connector_id, str(path))
            )
            sources.append(str(path))
    except Exception:
        catalog.db.close()
        raise
    return catalog, entry_ids, sources


def _resolve_asset(path: Path) -> tuple[Catalog, int, str]:
    """Return the catalog entry id for one *path* plus the owning :class:`Catalog`.

    The single-path delegate over :func:`_resolve_assets`, kept so callers that
    genuinely resolve one asset (``curator review``'s sibling idiom, and M010/S04's
    group seed) keep a signature that cannot be handed a list by accident.
    """
    catalog, entry_ids, sources = _resolve_assets([path])
    return catalog, entry_ids[0], sources[0]


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


def _analyze_all_or_reuse(
    catalog: Catalog,
    entry_ids: list[int],
    sources: list[str],
    request: ArtDirectionRequest,
) -> list[TreatmentProposal]:
    """Run the policy engine over every requested asset (M010/S02).

    Derives one :class:`AnalysisResult` per asset, in request order: the newest
    persisted ``ok`` row when there is one (deterministic and fast), else a fresh
    local analysis of that asset's bytes. Every fresh analysis goes through a
    *single* :class:`LocalAnalysisProvider`, so the policy engine can ask it for
    a real cross-image affinity between assets it analyzed; a pair it never saw
    falls back per comparison to the stored ``pairing.affinity``. Returns the
    ranked :class:`TreatmentProposal` list for the whole group.
    """
    provider = LocalAnalysisProvider()
    results: list[AnalysisResult] = []
    for entry_id, source in zip(entry_ids, sources, strict=True):
        persisted = _load_analysis(catalog, entry_id)
        results.append(persisted if persisted is not None else provider.analyze(source))
    return propose_treatments(results, request, provider=provider)


def _propose(args: argparse.Namespace) -> int:
    """Propose art-direction treatments for one or more cataloged assets.

    Resolves every ``asset`` path to its catalog entry, derives one
    :class:`AnalysisResult` each (reusing persisted ones, else analyzing fresh),
    runs the policy engine over the whole group — single-source treatments from
    the primary plus ``diptych``/``triptych``/``quad`` when the group coheres —
    persists the resulting proposals to ``proposals`` (append-only), and prints
    them ranked (human or ``--json``). Returns :data:`EXIT_OK` (0) when proposals
    exist, :data:`EXIT_NO_CHANGE` (3) when the policy produced none.

    **The primary source owns the persisted row** (Open Question #3): proposals
    are recorded against ``entry_ids[0]``, extending the precedent
    ``api._record_render`` already set for the renders journal. The manifest JSON
    itself carries every source, so no junction table is needed.

    ``--weights`` (M010/S03) overrides the packed layout's default per-source
    importance. It is validated against the asset count *before* any catalog or
    analysis work, and travels in ``request.context`` so the policy engine can
    record it in the proposal's evidence — which is what makes a later
    ``curator manifest`` reproduce the same geometry.
    """
    paths = _asset_paths(args.asset, "propose")
    width, height = _target_dims(args.target)
    weights = _parse_weights(getattr(args, "weights", None), len(paths))
    request = ArtDirectionRequest(
        target=args.target,
        target_width=width,
        target_height=height,
        sources=[str(path) for path in paths],
        context={} if weights is None else {"weights": weights},
    )
    catalog, entry_ids, sources = _resolve_assets(paths)
    try:
        proposals = _analyze_all_or_reuse(catalog, entry_ids, sources, request)
        _persist_proposals(catalog, entry_ids[0], proposals)
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
    """Build and persist an :class:`ArtDirectionManifest` over one or more assets.

    Chooses a :class:`TreatmentProposal` (the ``--treatment`` one, else the
    top-ranked proposal that can actually lay out this many sources),
    materializes a base manifest via :func:`materialize_manifest` — one real cell
    per source — attaches deterministic per-target overrides, resolves the base
    for ``--target``, and persists the validated result to
    ``art_direction_manifests`` (append-only). Returns :data:`EXIT_OK` (0).

    **The primary source owns the persisted row** (Open Question #3): the
    manifest is recorded against ``entry_ids[0]``, extending
    ``api._record_render``'s precedent. Every source is in the manifest JSON.
    """
    paths = _asset_paths(args.asset, "manifest")
    base_width, base_height = _target_dims(DEFAULT_TARGET)
    request = ArtDirectionRequest(
        target=DEFAULT_TARGET,
        target_width=base_width,
        target_height=base_height,
        sources=[str(path) for path in paths],
    )
    catalog, entry_ids, sources = _resolve_assets(paths)
    try:
        proposal = _select_proposal(catalog, entry_ids, args.treatment, sources, request)
        base = _attach_target_overrides(
            materialize_manifest(proposal, request, sources)
        )
        manifest = base.resolved_for(args.target)
        _persist_manifest(catalog, entry_ids[0], manifest)
    finally:
        catalog.db.close()
    if args.json:
        print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(_format_manifest(manifest))
    return EXIT_OK


def _select_proposal(
    catalog: Catalog,
    entry_ids: list[int],
    treatment: str | None,
    sources: list[str],
    request: ArtDirectionRequest,
) -> TreatmentProposal:
    """Return the chosen proposal for the primary entry, generating it if needed.

    Prefers an already-persisted proposal (matching *treatment* when given), else
    falls back to running the policy engine over every asset (persisting its
    output). Candidates whose named template cannot lay out exactly
    ``len(sources)`` sources are skipped rather than selected and then rejected
    by :func:`materialize_manifest` — a diptych is not an answer to three assets.
    Raises :class:`CuratorError` when no proposal can be produced.
    """
    entry_id = entry_ids[0]
    rows = _load_proposals(catalog, entry_id, treatment)
    if not rows:
        fresh = _analyze_all_or_reuse(catalog, entry_ids, sources, request)
        _persist_proposals(catalog, entry_id, fresh)
        rows = _load_proposals(catalog, entry_id, treatment)
    usable = [row for row in rows if _lays_out(row.treatment, len(sources))]
    if not usable:
        raise CuratorError(
            f"no proposal available to manifest {len(sources)} source(s) for "
            f"catalog entry {entry_id}"
        )
    return usable[0]


def _lays_out(treatment: LayoutTreatment, source_count: int) -> bool:
    """True when *treatment* can lay out exactly *source_count* sources.

    A treatment with no entry in ``_TREATMENT_SOURCE_COUNT`` has no fixed width.
    That splits two ways: the single-source treatments render the primary and are
    unconstrained, while M010/S03's ``packed`` is variable *within a range* — it
    is a multi-cell treatment, so it needs at least two sources and at most
    :data:`MAX_LAYOUT_SOURCES`, exactly the bound ``renderer._multi_cell``
    enforces. Without that, a ``packed`` proposal persisted from a five-asset
    ``propose`` would be selected for a later single-asset ``manifest`` and then
    rejected by the materializer — a correct error about the wrong thing.
    """
    required = _TREATMENT_SOURCE_COUNT.get(treatment)
    if required is not None:
        return required == source_count
    if treatment in MULTI_CELL_TREATMENTS:
        return 2 <= source_count <= MAX_LAYOUT_SOURCES
    return True


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
    *args.target*) with every referenced source resolved by
    :func:`_manifest_source_bytes`; any other source is treated as a media asset
    and rendered through a deterministic single-full-bleed manifest built from
    its bytes. Maps ``--target`` to pixel dims, renders via
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
        sources = _manifest_source_bytes(manifest)
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


def _manifest_source_bytes(manifest: ArtDirectionManifest) -> dict[str, bytes]:
    """Return ``{source: bytes}`` for every source *manifest* names (M010/S02).

    A manifest names its sources by whatever identity produced it, and both
    identities are first-class here: ``POST /api/render`` and hand-authored
    manifests name **content shas**, resolved from the :class:`ContentStore`,
    while ``curator manifest`` names the **cataloged asset paths** it was handed,
    read from disk. Resolving both is what makes ``curator manifest ... > m.json
    && curator render m.json`` an actual pipeline instead of two commands that
    speak different identities; a name that is neither is a fatal, actionable
    error rather than a stray ``StorageError``.
    """
    store = ContentStore()
    resolved: dict[str, bytes] = {}
    for source in manifest.sources:
        if len(source) == 64 and all(char in "0123456789abcdef" for char in source):
            resolved[source] = store.get(source)
            continue
        asset = Path(source)
        if not asset.is_file():
            raise CuratorError(
                f"manifest source {source!r} is neither a content sha nor a readable file"
            )
        resolved[source] = asset.read_bytes()
    return resolved


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
    expected SHA-256 and parsed ``--target`` dims. With ``--manifest`` the
    artifact is additionally checked cell by cell (M010/S01): the manifest is
    resolved for ``--target``, its cells resolved via
    :func:`~curator.artdirection.packing.resolve_regions`, and each one checked
    for bounds, sub-pixel extent, unintended cropping and disjointness — the
    production read side of region geometry. Prints a human-readable summary
    (publishable flag plus every failing check's reason) or the
    :class:`ValidationReport` as JSON. Returns :data:`EXIT_OK` (0) when
    publishable, :data:`EXIT_PARTIAL` (1) otherwise, and :data:`EXIT_FATAL` (2)
    on a read, manifest or target parse error.
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
    regions: list[SourceRegion] | None = None
    if args.manifest:
        manifest_path = Path(args.manifest).resolve()
        try:
            manifest = ArtDirectionManifest.from_dict(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            ).resolved_for(args.target)
            regions = resolve_regions(manifest, (width, height))
        except (json.JSONDecodeError, CuratorError, OSError) as exc:
            print(
                f"curator: error: failed to load manifest {manifest_path}: {exc}",
                file=sys.stderr,
            )
            return EXIT_FATAL
    report = ArtifactValidator().validate(
        data, args.expected_sha, (width, height), source_regions=regions
    )
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


def _taste(args: argparse.Namespace) -> int:
    """Route the taste group to its subcommand handler."""
    if args.taste_command == "drop":
        return _taste_drop(args)
    if args.taste_command == "profile":
        return _taste_profile(args)
    if args.taste_command == "dispute":
        return _taste_dispute(args)
    if args.taste_command == "vote":
        return _taste_vote(args)
    if args.taste_command == "votes":
        return _taste_votes(args)
    if args.taste_command == "retract":
        return _taste_retract(args)
    if args.taste_command == "embed-status":
        return _taste_embed_status(args)
    if args.taste_command == "embedding-head":
        return _taste_embedding_head(args)
    if args.taste_command == "embedding-explain":
        return _taste_embedding_explain(args)
    if args.taste_command == "compare":
        return _taste_compare(args)
    return EXIT_FATAL


def _taste_current_profile(catalog: Catalog, *, seed: bool = True) -> DialogueProfile:
    """Return the current taste profile, rebuilt from observations.

    Rebuilds from the append-only observation journal so the document always
    reflects every recorded reaction, re-applies the persisted pin/edit/dispute
    timeline on top (those are the user's corrections, not derived state), and
    optionally appends low-provenance claims seeded from approve/reject +
    pairwise history.
    """
    store = ProfileStore(catalog)
    profile = ProfileBuilder().build(ObservationStore(catalog).all())
    if seed:
        profile = ColdStartSeeder(catalog).seed(profile)
    for event in store.events():
        profile = _apply_profile_event(profile, event)
    return profile


def _apply_profile_event(profile: DialogueProfile, event: ProfileEvent) -> DialogueProfile:
    """Re-apply one timeline *event* to a freshly rebuilt *profile*."""
    store_ops = {
        "pin": lambda c: dataclasses.replace(c, status="pinned"),
        "edit": lambda c: dataclasses.replace(c, text=event.detail, status="edited"),
    }
    if event.kind == "dispute":
        return dataclasses.replace(
            profile,
            patterns=[c for c in profile.patterns if c.id != event.claim_id],
            tensions=[c for c in profile.tensions if c.id != event.claim_id],
        )
    op = store_ops[event.kind]
    return dataclasses.replace(
        profile,
        patterns=[op(c) if c.id == event.claim_id else c for c in profile.patterns],
        tensions=[op(c) if c.id == event.claim_id else c for c in profile.tensions],
    )


def _taste_profile(args: argparse.Namespace) -> int:
    """Print the taste profile document (readable text, or ``--json``)."""
    catalog = Catalog()
    try:
        profile = _taste_current_profile(catalog, seed=not args.no_seed)
        if args.json:
            print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(_format_taste_profile(profile))
        return EXIT_OK
    finally:
        catalog.db.close()


def _taste_dispute(args: argparse.Namespace) -> int:
    """Dispute a profile claim: remove it and mark its evidence for re-interpretation.

    Exits :data:`EXIT_NO_CHANGE` when no claim carries that id — a dispute is only
    recorded against a claim the profile actually makes.
    """
    catalog = Catalog()
    try:
        profile = _taste_current_profile(catalog)
        claim = next(
            (c for c in list(profile.patterns) + list(profile.tensions)
             if c.id == args.claim_id),
            None,
        )
        if claim is None:
            print(f"taste dispute: no claim with id {args.claim_id!r}")
            return EXIT_NO_CHANGE
        store = ProfileStore(catalog)
        store.apply(profile)
        event = store.dispute(args.claim_id)
        if args.json:
            print(json.dumps(event.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"taste dispute: removed claim {args.claim_id}")
            print(f"  was: {claim.text}")
            print(f"  {event.detail}")
        return EXIT_OK
    finally:
        catalog.db.close()


def _taste_pair_candidate(catalog: Catalog, entry_id: int) -> dict[str, Any]:
    """Return ``{entry_id, sha256, asset_id}`` for one catalog entry (pair display)."""
    row = catalog.db.execute(
        "SELECT sha256, asset_id FROM catalog_entries WHERE id = ?", (entry_id,)
    ).fetchone()
    return {
        "entry_id": entry_id,
        "sha256": str(row[0]) if row else "",
        "asset_id": str(row[1]) if row else "",
    }


def _taste_vote(args: argparse.Namespace) -> int:
    """Preview the current A/B taste comparison, or answer it with ``--prefer``.

    Without ``--prefer``, prints the pair :func:`~curator.taste.store.next_pair`
    currently selects and records nothing (exit :data:`EXIT_NO_CHANGE`) — mirrors
    ``_taste_drop``'s no-note preview. With ``--prefer``, records the vote and
    prints the new profile version (exit :data:`EXIT_OK`).
    """
    catalog = Catalog()
    try:
        pair = next_pair(catalog)
        if pair is None:
            print("taste vote: not enough analyzed images with votes remaining to compare")
            return EXIT_NO_CHANGE
        a, b = pair
        if args.prefer is None:
            info = {
                "a": _taste_pair_candidate(catalog, int(a["id"])),
                "b": _taste_pair_candidate(catalog, int(b["id"])),
            }
            if args.json:
                print(json.dumps(info, ensure_ascii=False))
            else:
                print("taste vote: current pair —")
                for label in ("a", "b"):
                    cand = info[label]
                    print(
                        f"  {label.upper()}: entry {cand['entry_id']}"
                        f" sha256={cand['sha256'][:12]} asset={cand['asset_id']}"
                    )
                print("  answer with `curator taste vote --prefer a|b`")
            return EXIT_NO_CHANGE
        winner, loser = (a, b) if args.prefer == "a" else (b, a)
        store = TasteVoteStore(catalog)
        record = store.record_vote(int(winner["id"]), int(loser["id"]), note=args.note)
        profile = store.load_profile()
        if args.json:
            payload = record.to_dict()
            payload["profile_version"] = profile.version
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(
                f"taste vote: recorded {record.vote_group}"
                f" (entry {record.winner_entry_id} over entry {record.loser_entry_id})"
                f" — profile now version {profile.version}"
            )
        return EXIT_OK
    finally:
        catalog.db.close()


def _taste_votes(args: argparse.Namespace) -> int:
    """List every recorded pairwise vote (winner/loser/note/retracted status)."""
    catalog = Catalog()
    try:
        records = TasteVoteStore(catalog).votes()
        if args.json:
            print(json.dumps([r.to_dict() for r in records], ensure_ascii=False))
        elif not records:
            print("taste votes: (none yet — cast one with `curator taste vote`)")
        else:
            for record in records:
                status = "retracted" if record.retracted else "active"
                note = f' "{record.note}"' if record.note else ""
                print(
                    f"  [{record.vote_group}] entry {record.winner_entry_id} over"
                    f" entry {record.loser_entry_id} ({status}){note}"
                )
        return EXIT_OK
    finally:
        catalog.db.close()


def _taste_retract(args: argparse.Namespace) -> int:
    """Retract a recorded vote, reversing its effect on the persisted profile.

    Exits :data:`EXIT_NO_CHANGE` when no vote carries that id (unknown or
    already retracted) — mirrors ``_taste_dispute``'s exit-code idiom. History
    is never deleted: the retracted rows stay visible via ``taste votes``.
    """
    catalog = Catalog()
    try:
        store = TasteVoteStore(catalog)
        changed = store.retract(args.vote_group)
        if not changed:
            print(f"taste retract: no vote with id {args.vote_group!r}")
            return EXIT_NO_CHANGE
        profile = store.load_profile()
        if args.json:
            print(
                json.dumps(
                    {
                        "retracted": True,
                        "vote_group": args.vote_group,
                        "profile_version": profile.version,
                    },
                    ensure_ascii=False,
                )
            )
        else:
            print(
                f"taste retract: reversed {args.vote_group}"
                f" — profile now version {profile.version}"
            )
        return EXIT_OK
    finally:
        catalog.db.close()


def _taste_embed_status(args: argparse.Namespace) -> int:
    """Report whether the local embedding subsystem is usable, optionally backfilling.

    Constructs :class:`~curator.taste.embedding.provider.OnnxEmbeddingProvider`
    with default path resolution and calls :meth:`probe`. With ``--backfill`` and
    a passing probe, computes + stores a vector for every catalog entry lacking
    one under the current model version. A per-entry failure (missing content,
    or the model file swapped/corrupted mid-loop — the same T-09-05 scenario the
    embedding-explain routes guard against) is recorded and the loop continues
    (WR-06), mirroring the established ``IngestReport``/``AnalysisRunReport``
    per-item ``corrupt``/``error`` posture rather than aborting the whole run and
    discarding every already-successful entry's report. Always exits
    :data:`EXIT_OK` once the probe runs — even ``ok=False`` is a successful
    *report*, not a CLI failure; this command's whole job is to answer "is this
    available", which is the milestone's documented early-exit checkpoint.
    """
    provider = OnnxEmbeddingProvider()
    probe = provider.probe()
    result: dict[str, Any] = {
        "ok": probe.ok,
        "backend": probe.backend.value,
        "message": probe.message,
    }
    computed = 0
    errors: list[dict[str, str]] = []
    if probe.ok and args.backfill:
        catalog = Catalog()
        try:
            store = EmbeddingStore(catalog)
            for entry in catalog.list_entries():
                sha = entry["sha256"]
                if store.get(sha, provider.model_version) is not None:
                    continue
                try:
                    data = catalog.content.get(sha)
                    vector = provider.embed(data)
                    store.set(sha, provider.model_version, vector)
                except (StorageError, EmbeddingError) as exc:
                    errors.append({"sha256": sha, "error": str(exc)})
                    continue
                computed += 1
        finally:
            catalog.db.close()
        result["backfilled"] = computed
        result["backfill_error_count"] = len(errors)
        result["backfill_errors"] = errors
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"taste embed-status: ok={result['ok']} backend={result['backend']}")
        print(f"  {result['message']}")
        if probe.ok and args.backfill:
            print(f"  backfilled {computed} embedding(s)")
            if errors:
                print(f"  {len(errors)} entry(ies) failed to backfill:")
                for err in errors:
                    print(f"    {err['sha256'][:12]} — {err['error']}")
    return EXIT_OK


def _taste_embedding_head(args: argparse.Namespace) -> int:
    """Fit the nonparametric embedding head (M009/S03) and report its shape.

    Delegates availability to the same
    :meth:`~curator.taste.embedding.provider.OnnxEmbeddingProvider.probe` S02's
    ``embed-status`` reports — ``ok=False`` reuses its ``message`` field
    verbatim (never a re-worded copy) and exits :data:`EXIT_NO_CHANGE`:
    nothing to fit without a working provider, the visible early-exit
    checkpoint an operator reaches naturally by trying this command. When
    ``ok=True``, resolves every vote S01 recorded against S02's stored
    vectors and fits the head, printing ``capacity`` alongside the literal
    retained-parameter count (``len(head.vote_terms)``) so both are visible
    and can be eyeballed as equal on every invocation — a live,
    human-checkable proof of the R041 contract, not just a test assertion.
    Always exits :data:`EXIT_OK` once the probe runs: reporting
    ``capacity=0`` is a valid, successful report, not a failure.
    """
    provider = OnnxEmbeddingProvider()
    probe = provider.probe()
    if not probe.ok:
        if args.json:
            print(
                json.dumps(
                    {"ok": False, "backend": probe.backend.value, "message": probe.message},
                    ensure_ascii=False,
                )
            )
        else:
            print(f"taste embedding-head: ok=False backend={probe.backend.value}")
            print(f"  {probe.message}")
        return EXIT_NO_CHANGE
    catalog = Catalog()
    try:
        votes = TasteVoteStore(catalog).votes()
        vote_vectors = resolve_vote_vectors(
            votes, EmbeddingStore(catalog), EMBEDDING_MODEL_VERSION
        )
        head = fit_embedding_head(vote_vectors, EMBEDDING_MODEL_VERSION)
    finally:
        catalog.db.close()
    # WR-05: exercise the real fit_embedding_head([]) zero-vote branch — not a
    # hand-rolled EmbeddingHead(vote_terms=(), ...) stand-in, which bypassed
    # fit_embedding_head entirely and was guaranteed 0.0 by score()'s own first
    # branch regardless of what fit_embedding_head's zero-vote path actually did.
    zero_check = fit_embedding_head([], EMBEDDING_MODEL_VERSION).score(
        np.random.RandomState(0).rand(EMBEDDING_DIM).astype(np.float32)
    )
    direction_norm = float(np.linalg.norm(head.effective_direction()))
    result: dict[str, Any] = {
        "ok": True,
        "model_version": head.model_version,
        "capacity": head.capacity,
        "retained_parameters": len(head.vote_terms),
        "effective_direction_norm": direction_norm,
        "zero_vote_parity_ok": zero_check == 0.0,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"taste embedding-head: capacity={result['capacity']}")
        print(
            f"  retained_parameters={result['retained_parameters']}"
            " (structural cross-check: must equal capacity)"
        )
        print(
            "  effective_direction() L2 norm (display summary only, not the head's state):"
            f" {direction_norm:.6f}"
        )
        print(f"  zero-vote parity: {'OK' if zero_check == 0.0 else 'FAIL'} (score={zero_check})")
    return EXIT_OK


def _taste_embedding_resolve(catalog: Catalog, path_or_sha: str) -> tuple[int, str]:
    """Resolve *path_or_sha* to ``(entry_id, sha256)``.

    A 64-hex string is treated as a direct content sha lookup (mirrors ``api.py``'s
    ``_SHA_RE`` usage); anything else is resolved as a filesystem path via
    :func:`resolve_catalog_entry`/:class:`LocalConnector`, the same way
    ``_review_resolve`` does. Raises :class:`CatalogEntryNotFound` when nothing matches
    — caught by :func:`main` and mapped to :data:`EXIT_FATAL`, same as every other
    catalog-lookup miss in this CLI.
    """
    if _SHA_RE.fullmatch(path_or_sha):
        sha = path_or_sha.lower()
        row = catalog.db.execute(
            "SELECT id FROM catalog_entries WHERE sha256 = ? ORDER BY id DESC LIMIT 1",
            (sha,),
        ).fetchone()
        if row is None:
            raise CatalogEntryNotFound(f"no catalog entry for sha256={sha}")
        return int(row[0]), sha
    path = Path(path_or_sha).resolve()
    connector = LocalConnector(path.parent)
    entry_id = resolve_catalog_entry(catalog, connector.connector_id, str(path))
    row = catalog.db.execute(
        "SELECT sha256 FROM catalog_entries WHERE id = ?", (entry_id,)
    ).fetchone()
    return entry_id, str(row[0]) if row else ""


def _taste_liked_shas(
    catalog: Catalog, votes: list[VoteRecord]
) -> tuple[list[str], dict[str, int]]:
    """Return ``(liked_shas, sha_to_entry_id)`` for every non-retracted vote winner.

    ``liked_shas`` is what
    :func:`~curator.taste.embedding.attribution.find_exemplars` restricts its search to
    — resolved here (not inside that otherwise-pure function) so it never needs a hidden
    catalog dependency.
    """
    winner_ids = sorted({v.winner_entry_id for v in votes if not v.retracted})
    if not winner_ids:
        return [], {}
    placeholders = ",".join("?" for _ in winner_ids)
    rows = catalog.db.execute(
        f"SELECT id, sha256 FROM catalog_entries WHERE id IN ({placeholders})",
        winner_ids,
    ).fetchall()
    sha_to_entry_id = {str(sha): int(entry_id) for entry_id, sha in rows}
    return list(sha_to_entry_id.keys()), sha_to_entry_id


def _taste_embedding_explain(args: argparse.Namespace) -> int:
    """Explain one embedding-head score: per-vote attribution + nearest liked exemplars.

    Resolves ``path_or_sha`` to a catalog entry, probes the embedding provider
    (``ok=False`` -> the not-available message, :data:`EXIT_NO_CHANGE`, same pattern as
    ``embedding-head``), fetches (embedding on demand when missing, mirroring
    ``embed-status``'s ``--backfill`` logic) the entry's stored vector, fits the head
    fresh over every resolvable vote (S03 — this command persists nothing, matching
    ``embedding-head``'s own posture), then reports
    :func:`~curator.taste.embedding.attribution.attribute_score`,
    :func:`~curator.taste.embedding.attribution.find_exemplars`, and
    :func:`~curator.taste.embedding.attribution.render_rationale` over it. Always exits
    :data:`EXIT_OK` once the report is produced — a report with zero
    contributions/exemplars is a valid, honest answer, not a failure.
    """
    catalog = Catalog()
    try:
        entry_id, sha = _taste_embedding_resolve(catalog, args.path_or_sha)
        provider = OnnxEmbeddingProvider()
        probe = provider.probe()
        if not probe.ok:
            if args.json:
                print(
                    json.dumps(
                        {"ok": False, "backend": probe.backend.value, "message": probe.message},
                        ensure_ascii=False,
                    )
                )
            else:
                print(f"taste embedding-explain: ok=False backend={probe.backend.value}")
                print(f"  {probe.message}")
            return EXIT_NO_CHANGE
        store = EmbeddingStore(catalog)
        stored = store.get(sha, provider.model_version)
        if stored is None:
            vector = provider.embed(catalog.content.get(sha))
            store.set(sha, provider.model_version, vector)
        else:
            vector = stored.vector
        vote_records = TasteVoteStore(catalog).votes()
        vote_vectors = resolve_vote_vectors(vote_records, store, provider.model_version)
        head = fit_embedding_head(vote_vectors, provider.model_version)
        liked_shas, sha_to_entry_id = _taste_liked_shas(catalog, vote_records)
        attribution = attribute_score(vector, head, vote_vectors)
        exemplars = find_exemplars(
            vector, liked_shas, sha_to_entry_id, store, provider.model_version
        )
        rationale = render_rationale(attribution, exemplars)
    finally:
        catalog.db.close()
    result: dict[str, Any] = {
        "entry_id": entry_id,
        "sha256": sha,
        "score": attribution.score,
        "rationale": rationale,
        "contributions": attribution.contributions,
        "exemplars": [e.to_dict() for e in exemplars],
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"taste embedding-explain: entry {entry_id} sha256={sha[:12]}")
        print(f"  score={result['score']:+.4f} — {rationale}")
        if attribution.contributions:
            print("  contributions:")
            for c in attribution.contributions:
                print(f"    [{c['vote_group']}] {c['contribution']:+.4f}")
        else:
            print("  contributions: (none — no votes resolvable yet)")
        if exemplars:
            print("  exemplars:")
            for e in exemplars:
                print(f"    {e.sha256[:12]} (entry {e.entry_id}, similarity {e.similarity:.3f})")
        else:
            print("  exemplars: (none yet)")
    return EXIT_OK


def _taste_compare_vector_by_asset_id(
    catalog: Catalog,
    candidates: list[dict[str, Any]],
    analysis_map: dict[str, AnalysisResult],
    embedding_store: EmbeddingStore,
    model_version: str,
) -> dict[str, np.ndarray]:
    """Return ``{analysis.asset_id: vector}`` for every candidate with a stored embedding.

    A candidate lacking a stored vector under *model_version* is simply absent
    — the comparison's embedding scorer (see
    ``_taste_compare_embedding_scorer_factory``) treats a missing entry as
    ``0.0`` (no evidence), never a crash, mirroring the zero-vote/no-evidence
    posture the rest of this subsystem already uses.
    """
    entry_ids = [int(c["id"]) for c in candidates]
    if not entry_ids:
        return {}
    placeholders = ",".join("?" for _ in entry_ids)
    rows = catalog.db.execute(
        f"SELECT id, sha256 FROM catalog_entries WHERE id IN ({placeholders})",
        entry_ids,
    ).fetchall()
    sha_by_entry_id = {int(entry_id): str(sha) for entry_id, sha in rows}
    vector_by_asset_id: dict[str, np.ndarray] = {}
    for cid, analysis in analysis_map.items():
        sha = sha_by_entry_id.get(int(cid))
        if sha is None:
            continue
        stored = embedding_store.get(sha, model_version)
        if stored is not None:
            vector_by_asset_id[analysis.asset_id] = stored.vector
    return vector_by_asset_id


def _taste_compare_embedding_scorer_factory(
    vector_by_asset_id: dict[str, np.ndarray], model_version: str
) -> Callable[[Sequence[VoteVectors]], Scorer]:
    """Return a ``training-votes-subset -> Scorer`` factory for ``compare_heads``.

    Refits :func:`~curator.taste.embedding.head.fit_embedding_head` fresh for
    whichever vote subset it is called with — the full training set, or one of
    ``compare_heads``'s learning-curve checkpoints — so ``compare.py`` itself
    never needs to know about ``EmbeddingStore``/``model_version`` (it stays a
    pure statistics module).
    """

    def factory(votes_subset: Sequence[VoteVectors]) -> Scorer:
        head = fit_embedding_head(votes_subset, model_version)

        def scorer(analysis: AnalysisResult) -> float:
            vector = vector_by_asset_id.get(analysis.asset_id)
            return head.score(vector) if vector is not None else 0.0

        return scorer

    return factory


def _taste_compare(args: argparse.Namespace) -> int:
    """Compare the lens and embedding heads over recorded votes, with uncertainty.

    Splits the non-retracted vote history into training/held-out
    (:data:`~curator.taste.embedding.compare.HELD_OUT_FRACTION`, most-recent
    votes held out — deterministic, chronological), fits the lens scorer over
    the current persisted profile and the embedding scorer fresh over the
    training votes, evaluates both against the held-out set via
    :func:`~curator.taste.embedding.compare.compare_heads`, and prints the full
    report. Mirrors ``embedding-head``/``embedding-explain``'s availability
    gate: ``ok=False`` -> :data:`EXIT_NO_CHANGE`. Also exits
    :data:`EXIT_NO_CHANGE` when fewer than two analyzed candidates exist yet —
    nothing to compare, not a failure.
    """
    provider = OnnxEmbeddingProvider()
    probe = provider.probe()
    if not probe.ok:
        if args.json:
            print(
                json.dumps(
                    {"ok": False, "backend": probe.backend.value, "message": probe.message},
                    ensure_ascii=False,
                )
            )
        else:
            print(f"taste compare: ok=False backend={probe.backend.value}")
            print(f"  {probe.message}")
        return EXIT_NO_CHANGE
    catalog = Catalog()
    try:
        candidates, analysis_map = resolve_vote_candidates(catalog)
        if len(candidates) < 2:
            print("taste compare: fewer than two analyzed candidates — nothing to compare yet")
            return EXIT_NO_CHANGE
        all_votes = [v for v in TasteVoteStore(catalog).votes() if not v.retracted]
        n_held_out = max(1, round(len(all_votes) * HELD_OUT_FRACTION))
        split_index = len(all_votes) - n_held_out
        training_records = all_votes[:split_index]
        held_out_records = all_votes[split_index:]
        # A held-out vote's winner/loser must still be a current analyzed
        # candidate — a vote can outlive its entry's analysis (re-analysis,
        # deletion); skip rather than KeyError deep inside evaluate().
        #
        # CR-01 defense-in-depth: position in the (aid, bid, preferred) triple
        # is assigned by a stable, vote-independent key (numeric min/max of the
        # two entry ids), never "winner always in the aid slot" — evaluate()/
        # compare_heads() no longer credit ties either way, but this also keeps
        # the *shape* of held_out_pairs itself uncorrelated with which id won,
        # so no future evaluation mechanism can silently reintroduce this exploit.
        analyzed_ids = set(analysis_map)
        held_out_pairs: list[tuple[str, str, str]] = []
        for v in held_out_records:
            winner_id, loser_id = v.winner_entry_id, v.loser_entry_id
            if str(winner_id) not in analyzed_ids or str(loser_id) not in analyzed_ids:
                continue
            aid, bid = (winner_id, loser_id) if winner_id < loser_id else (loser_id, winner_id)
            held_out_pairs.append((str(aid), str(bid), str(winner_id)))
        embedding_store = EmbeddingStore(catalog)
        training_votes = resolve_vote_vectors(
            training_records, embedding_store, provider.model_version
        )
        vector_by_asset_id = _taste_compare_vector_by_asset_id(
            catalog, candidates, analysis_map, embedding_store, provider.model_version
        )
        current_profile = TasteVoteStore(catalog).load_profile()
        ranker = TasteRanker()

        def lens_scorer(analysis: AnalysisResult) -> float:
            return ranker.personal_delta(analysis, current_profile)[0]

        def baseline_scorer(analysis: AnalysisResult) -> float:
            return 0.0

        embedding_scorer_factory = _taste_compare_embedding_scorer_factory(
            vector_by_asset_id, provider.model_version
        )
        comparison = compare_heads(
            training_votes,
            held_out_pairs,
            candidates,
            analysis_map,
            lens_scorer,
            embedding_scorer_factory,
            baseline_scorer,
            lens_sample_efficiency_pairs=current_profile.version - 1,
        )
    finally:
        catalog.db.close()
    result = comparison.to_dict()
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"taste compare: verdict={comparison.verdict}")
        print(
            f"  lens:      held_out_accuracy={comparison.lens_evidence.held_out_accuracy:.3f}"
            f" promoted={comparison.lens_promoted}"
        )
        print(
            f"  embedding: held_out_accuracy={comparison.embedding_evidence.held_out_accuracy:.3f}"
            f" promoted={comparison.embedding_promoted}"
        )
        print(
            f"  discordant pairs: {comparison.discordant_pairs}"
            f" (embedding correct on {comparison.discordant_correct_embedding})"
        )
        if comparison.head_to_head_accuracy is not None and comparison.head_to_head_ci is not None:
            lo, hi = comparison.head_to_head_ci
            print(
                f"  head-to-head accuracy: {comparison.head_to_head_accuracy:.3f}"
                f" (95% CI [{lo:.3f}, {hi:.3f}])"
            )
        else:
            print(
                "  head-to-head accuracy: insufficient evidence"
                f" (< {MIN_DISCORDANT_PAIRS} discordant pairs)"
            )
        print("  learning curve:")
        for point in comparison.learning_curve:
            print(
                f"    votes={point['votes']}"
                f" embedding_held_out_accuracy={point['embedding_held_out_accuracy']:.3f}"
            )
    return EXIT_OK


def _format_taste_profile(profile: DialogueProfile) -> str:
    """Render the profile as the human-readable document (R035)."""
    lines = [f"Taste Profile (version {profile.version})"]
    lines.append("")
    lines.append("Vocabulary:")
    if not profile.vocabulary:
        lines.append("  (empty — react to some images with `curator taste drop`)")
    for word, entry in profile.vocabulary.items():
        lines.append(
            f"  {word} -> {entry['attribute']} ({entry['usage_count']} uses)"
        )
    for title, claims in (("Patterns", profile.patterns), ("Tensions", profile.tensions)):
        lines.append("")
        lines.append(f"{title}:")
        if not claims:
            lines.append("  (none yet)")
        for claim in claims:
            lines.append(f"  [{claim.id}] {claim.text}")
            lines.append(
                f"    status={claim.status} provenance={claim.provenance}"
                f" evidence={len(claim.evidence)}"
            )
            for ref in claim.evidence:
                lines.append(
                    f"      - {ref.image_sha[:12]} \"{ref.verbatim}\""
                    f" (confidence {ref.confidence:.2f}, {ref.created_at})"
                )
    lines.append("")
    lines.append("Evolution:")
    if not profile.evolution:
        lines.append("  (none yet)")
    for entry in profile.evolution:
        # An empty ``at`` is legitimate (a profile built from zero observations
        # has no stamp to inherit) — print the summary alone rather than ": ...".
        prefix = f"{entry['at']}: " if entry["at"] else ""
        lines.append(f"  {prefix}{entry['summary']}")
    return "\n".join(lines)


def _taste_extraction_config() -> dict[str, Any] | None:
    """Return the taste extraction provider config from env, or ``None`` when off.

    Shared with the API surface so both gate identically — see
    :func:`~curator.taste.dialogue.extraction.extraction_config_from_env`.
    """
    return extraction_config_from_env()


def _taste_drop(args: argparse.Namespace) -> int:
    """Record a reaction to dropped images, or preview probing questions.

    Resolves each PATH as a cataloged asset or an ephemeral third-party file via
    :class:`ReactionRoom`. With ``--note``, one session is opened, the reaction is
    extracted and recorded, and the session is closed (exit 0). Without ``--note``
    the room previews its deterministic probing questions and records nothing
    (exit :data:`EXIT_NO_CHANGE`).
    """
    provider = resolve_extraction_provider(_taste_extraction_config())
    data_root = CuratorConfig().data_root
    catalog = Catalog()
    try:
        room = ReactionRoom(catalog, provider, data_root)
        session = room.start(args.paths)
        if args.save:
            for saved in _taste_save_drops(room, session, args.paths, catalog):
                print(f"taste drop: saved {saved} to the catalog (full resolution)")
        if args.note:
            turn = room.react(session, args.note)
            count = room.finish(session)
            label = "observation" if count == 1 else "observations"
            print(
                f"taste drop: recorded reaction ({count} {label} in session {session.id})"
            )
            if turn.question is not None:
                print(f"  {turn.question.text}")
            # No silent learning (R038): every session reports what it added.
            print()
            print(WhatILearned.delta_after(session.id, ObservationStore(catalog)).summary)
            return EXIT_OK
        questions = room.generator.questions_for(
            create_observation(session_id=session.id, verbatim="")
        )
        room.finish(session)
        if not questions:
            print("taste drop: no probing questions")
        else:
            print("taste drop: probing questions:")
            for question in questions:
                print(f"  - {question.text}")
        return EXIT_NO_CHANGE
    finally:
        catalog.db.close()


def _taste_save_drops(
    room: ReactionRoom,
    session: TasteSession,
    paths: list[str],
    catalog: Catalog,
) -> list[str]:
    """Promote this session's ephemeral drops into the catalog (R034).

    Retention keeps third-party drops as a thumbnail + hash only; ``--save`` is
    the explicit user choice that writes the full-resolution bytes. Already
    cataloged drops are skipped. Returns the promoted shas, shortened.
    """
    by_sha = {ref.sha256: ref for ref in room.session_images(session)}
    saved: list[str] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        data = path.read_bytes()
        ref = by_sha.get(sha256_hex(data))
        if ref is None or not ref.ephemeral or ref.catalog_saved:
            continue
        save_to_catalog(ref, data, catalog)
        saved.append(ref.sha256[:12])
    return saved


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
    if args.command == "taste":
        return _taste(args)
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
