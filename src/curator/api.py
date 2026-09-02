"""Localhost FastAPI application — the S04-S05 operator surface (MEM003).

:func:`create_app` builds a thin FastAPI app that exposes the S04->S05 boundary
endpoints over loopback. Every handler delegates to an existing subsystem and
never reimplements business logic (MEM003 / MEM010 thin-handler rule):

- ``GET /health``             — :meth:`Catalog.count_catalog_entries` ->
  ``{"status": "healthy", "catalog_entries": N}`` (the same payload the
  ``curator health --json`` CLI command emits).
- ``GET /status``             — catalog entry + unique-cluster counts
  (:meth:`Catalog.count_catalog_entries` / :meth:`count_unique_clusters`).
- ``POST /ingest``            — run :class:`~curator.ingest.pipeline.IngestPipeline`
  over a ``LocalConnector`` for ``{"path": ..., "resume": bool}`` and return the
  JSON-serializable :class:`~curator.ingest.report.IngestReport`.
- ``GET /catalog``            — the full catalog via
  :meth:`Catalog.list_entries`.
- ``GET /api/review``         — catalog entries with their current approval
  decision, plus ``POST /api/review/{approve,reject,undo}`` to record/revert
  decisions via :class:`~curator.approve.ApprovalService`.
- ``GET /api/taste/profile``  — the Taste Profile document (vocabulary / patterns /
  tensions / evolution) plus its quotable ``citations``, ``dimensions``, and the
  pin/edit/dispute ``timeline``; ``POST /api/taste/drop`` records one Reaction
  Room turn (503 when no extraction provider is enabled — it never guesses), and
  ``POST /api/taste/{pin,edit,dispute}`` applies a correction to one claim.
  ``POST /api/taste/explain`` returns the rerank explanation citing the profile.
- ``GET /api/taste/pair``     — the current A/B taste comparison
  (M009/S01), or ``{"available": false}`` when fewer than two analyzed
  candidates remain; ``POST /api/taste/vote`` answers it (409 if the pair
  changed since it was fetched), ``GET /api/taste/votes`` lists every recorded
  vote, and ``POST /api/taste/retract`` reverses one without deleting it.
- ``GET /api/taste/embedding-status`` — whether the local embedding subsystem
  (M009/S02) is usable, with no network call either way; read-only (no
  ``--backfill`` equivalent over HTTP — that bulk-recompute loop is a
  CLI-only maintenance operation).
- ``POST /api/taste/embedding-explain`` — explain one embedding-head score
  (M009/S04): the exact per-vote attribution (summing back to the score),
  up to three nearest-neighbour exemplars from the user's own liked images,
  and a deterministic template ``rationale`` — the same shape
  ``curator taste embedding-explain`` prints. 503 when the embedding
  provider itself is unavailable.
- ``GET /api/taste/compare``  — compare the lens and embedding heads over
  recorded votes, with uncertainty (M009/S05): the same report
  ``curator taste compare`` prints — held-out accuracy + promotion-gate
  result per head, a discordant-pairs head-to-head accuracy with a 95%
  confidence interval, a learning curve, and a verdict, never a path that
  retires the lens head. 503 when the embedding provider is unavailable,
  400 when fewer than two analyzed candidates exist yet.
- ``GET /`` + ``/app``        — the review SPA (``webui/``) served by starlette
  ``StaticFiles``.
- ``GET /consolidation-plan`` — a ``?path=`` directory inventory via
  :func:`curator.consolidate.plan.build_plan`, returned as a
  :class:`~curator.consolidate.plan.ConsolidationPlan`.

FastAPI auto-serves the interactive OpenAPI docs at ``/docs`` and the raw schema
at ``/openapi.json`` — the demo gate (``http://127.0.0.1:8765/docs``).

The app binds to **loopback only** (127.0.0.1:8765) by default — air-gapped and
never exposed on a LAN interface. The module-level ``app = create_app()`` is the
``uvicorn curator.api:app`` import target.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import re
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from starlette.responses import FileResponse

from curator.analysis.cli_utils import resolve_catalog_entry
from curator.analysis.errors import AnalysisError, CatalogEntryNotFound
from curator.analysis.local import LocalAnalysisProvider
from curator.analysis.profiles import AnalysisProfile
from curator.analysis.schema import AnalysisResult
from curator.approve import ApprovalError, ApprovalService
from curator.artdirection.manifest import (
    MAX_LAYOUT_SOURCES,
    ArtDirectionManifest,
    ManifestError,
    SourceRegion,
)
from curator.artdirection.packing import resolve_regions
from curator.artdirection.policy import ArtDirectionRequest, propose_treatments
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.consolidate.plan import build_plan
from curator.errors import CuratorError, StorageError
from curator.hashing import sha256_hex
from curator.ingest.pipeline import IngestPipeline
from curator.render.renderer import DeterministicRenderer, RenderError, RenderResult
from curator.render.validate import ArtifactValidator
from curator.taste.dialogue.extraction import (
    extraction_config_from_env,
    resolve_extraction_provider,
)
from curator.taste.dialogue.profile import (
    ColdStartSeeder,
    ProfileBuilder,
    ProfileEvent,
    ProfileStore,
    TasteClaim,
    TasteProfile,
    WhatILearned,
)
from curator.taste.dialogue.retention import save_to_catalog
from curator.taste.dialogue.room import ReactionRoom, ReactionRoomUnavailableError
from curator.taste.dialogue.store import ObservationStore
from curator.taste.dialogue.upstream import (
    citations_for,
    explain_rank,
    profile_dimensions,
)
from curator.taste.embedding.attribution import attribute_score, find_exemplars, render_rationale
from curator.taste.embedding.compare import HELD_OUT_FRACTION, compare_heads
from curator.taste.embedding.errors import EmbeddingError
from curator.taste.embedding.grouping import (
    GROUP_SIMILARITY_THRESHOLD,
    MAX_CANDIDATE_POOL,
    GroupingError,
    resolve_group_pool,
    resolve_group_sources,
    select_group,
)
from curator.taste.embedding.head import VoteVectors, fit_embedding_head, resolve_vote_vectors
from curator.taste.embedding.provider import OnnxEmbeddingProvider
from curator.taste.embedding.store import EmbeddingStore
from curator.taste.pairwise import Scorer
from curator.taste.rank import TasteRanker
from curator.taste.store import TasteVoteStore, VoteRecord, next_pair, resolve_vote_candidates

# Loopback-only bind address (MEM003): never expose the API on a LAN interface.
HOST = "127.0.0.1"
PORT = 8765

# Absolute path to the bundled SPA (repo-root webui/), mounted under /app.
WEBUI_DIR = Path(__file__).resolve().parents[2] / "webui"


class IngestRequest(BaseModel):
    """JSON request body for ``POST /ingest``.

    ``path`` is the local folder to ingest; ``resume`` (default False) resumes a
    prior ingest from the ``ingest_journal`` checkpoint (see IngestPipeline.run).
    """

    path: str
    resume: bool = False


class ReviewActionRequest(BaseModel):
    """JSON request body for ``POST /api/review/{approve,reject,undo}``.

    ``asset`` is the source asset id (LocalConnector normalized path); the
    optional ``entry_id`` short-circuits resolution to the catalog entry
    directly. ``rationale`` records a reason with approve/reject.
    """

    asset: str
    rationale: str = ""
    entry_id: int | None = None


class AnalyzeRequest(BaseModel):
    """JSON request body for ``POST /api/analyze``.

    ``asset`` is a 64-hex content sha256 resolved from the ContentStore, or
    ``bytes`` carries the base64-encoded image payload directly (tests). The
    optional ``profile`` maps to an :class:`AnalysisProfile`.
    """

    asset: str | None = None
    bytes: str | None = None
    profile: str = "balanced"


class ProposeRequest(BaseModel):
    """JSON request body for ``POST /api/propose``.

    One source comes from ``asset``/``bytes`` (primary); ``sources`` may supply
    additional content shas — two enable a diptych, three a triptych and four a
    quad (M010/S02). The target is ``1080p``/``4k`` (or an explicit width/height
    pair).

    ``sources`` is capped at :data:`MAX_LAYOUT_SOURCES` entries. Until M010/S02
    the handler's ``[:2]`` slice was accidentally the *only* bound on this
    field, so the cap lands in the same change that removes the truncation:
    de-truncating without it would turn an accidentally-bounded list into an
    unbounded one (T-10-06).
    """

    asset: str | None = None
    bytes: str | None = None
    sources: list[str] = []
    target: str = "1080p"
    target_width: int | None = None
    target_height: int | None = None
    allow_diptych: bool = True

    @field_validator("sources")
    @classmethod
    def _cap_sources(cls, value: list[str]) -> list[str]:
        """Reject an over-cap source list before the handler body runs (422)."""
        if len(value) > MAX_LAYOUT_SOURCES:
            raise ValueError(
                f"sources has {len(value)} entries, over the "
                f"{MAX_LAYOUT_SOURCES}-source layout cap — an over-cap request is "
                f"rejected, never truncated"
            )
        return value


class GroupRequest(BaseModel):
    """JSON request body for ``POST /api/packing/group`` (M010/S04).

    The seed comes from ``asset`` (a 64-hex content sha) or ``bytes`` (base64),
    resolved by the same ``_resolve_image`` every other route uses. ``size`` is
    the group size **including** the seed, ``pool_size`` bounds the candidate
    pool, and ``threshold`` is the minimum seed-to-candidate cosine.

    None of the three is bounded here: ``select_group``/``resolve_group_pool``
    own those contracts (``GroupingError`` -> 400), so this route goes through the
    exact same single mechanism the CLI does rather than restating a second copy
    of the bounds that could drift from it.
    """

    asset: str | None = None
    bytes: str | None = None
    size: int = 3
    pool_size: int = MAX_CANDIDATE_POOL
    threshold: float = GROUP_SIMILARITY_THRESHOLD


class RenderRequest(BaseModel):
    """JSON request body for ``POST /api/render``.

    One of ``manifest`` (full :class:`ArtDirectionManifest` with source shas),
    ``asset`` (single content sha), or ``bytes`` (single base64 image) must be
    given. ``sources`` maps a sha to base64 bytes to override ContentStore lookups.
    """

    asset: str | None = None
    bytes: str | None = None
    manifest: dict[str, Any] | None = None
    sources: dict[str, str] | None = None
    target: str | dict[str, int] | None = None
    target_width: int | None = None
    target_height: int | None = None


class ValidateRequest(BaseModel):
    """JSON request body for ``POST /api/validate``.

    ``artifact_sha`` resolves rendered bytes from the ContentStore, or
    ``artifact_bytes`` carries them base64-encoded directly (tests).
    ``expected_sha`` is the provenance hash checked against the artifact.
    ``manifest`` (M010/S01) additionally checks the artifact's cell geometry:
    its regions are resolved for the target and validated per cell.
    """

    artifact_sha: str | None = None
    artifact_bytes: str | None = None
    expected_sha: str
    target: str | dict[str, int] | None = None
    target_width: int | None = None
    target_height: int | None = None
    color_mode: str = "RGB"
    color_profile: str = "sRGB"
    manifest: dict[str, Any] | None = None


class TasteDropRequest(BaseModel):
    """JSON request body for ``POST /api/taste/drop``.

    ``images`` carries base64-encoded third-party drops (retained as thumbnail +
    hash only) and ``shas`` names already-cataloged content; at least one is
    required. ``note`` is the user's reaction in their own words — stored
    verbatim. ``save`` is the explicit choice that promotes the dropped images
    into the catalog at full resolution.
    """

    images: list[str] = []
    shas: list[str] = []
    note: str = ""
    save: bool = False


class TasteClaimRequest(BaseModel):
    """JSON request body for ``POST /api/taste/{pin,edit,dispute}``.

    ``text`` is the replacement wording and is required only for ``edit``.
    """

    claim_id: str
    text: str = ""


class TasteVoteRequest(BaseModel):
    """JSON request body for ``POST /api/taste/vote`` (M009/S01).

    ``prefer`` must be ``"a"`` or ``"b"`` (validated explicitly — a non-matching
    string maps to 400, not a raw 422). ``a_entry_id``/``b_entry_id`` are the two
    candidate ids the client is looking at, resolved from the most recent
    ``GET /api/taste/pair`` response — required, not optional, so a stale client
    cannot silently vote on a substitute pair: the handler recomputes the pair
    fresh and 409s on a mismatch instead of trusting these blindly.
    """

    prefer: str
    note: str = ""
    a_entry_id: int
    b_entry_id: int


class TasteRetractRequest(BaseModel):
    """JSON request body for ``POST /api/taste/retract`` (M009/S01)."""

    vote_group: str


def create_app(catalog: Catalog | None = None) -> FastAPI:
    """Build the curator FastAPI application.

    When *catalog* is provided it is used by every handler (tests pass an
    isolated ``data_root``-backed Catalog). Otherwise the default catalog is
    resolved lazily on first request from ``CURATOR_DATA_ROOT`` via :class:`Catalog`
    — so importing this module / creating the module-level ``app`` has no
    side effects (no DB file is created at import time).
    """
    app = FastAPI(
        title="Curator",
        description=(
            "Samsung Frame curation pipeline localhost API — catalog, ingest, "
            "consolidation-planning, and health surfaces (S04-S05 boundary)."
        ),
        version="0.1.0",
    )
    app.state.catalog = catalog

    # -- helpers ---------------------------------------------------------------

    def _catalog(request: Request) -> Catalog:
        """Return the app's Catalog, resolving the default lazily on first use."""
        resolved = request.app.state.catalog
        if resolved is None:
            resolved = Catalog()  # honors CURATOR_DATA_ROOT at request time
            request.app.state.catalog = resolved
        return resolved

    # -- endpoints -------------------------------------------------------------

    @app.get("/health")
    def health(request: Request) -> dict:
        """Report healthy status plus the total catalog entry count."""
        count = _catalog(request).count_catalog_entries()
        return {"status": "healthy", "catalog_entries": count}

    @app.get("/status")
    def status(request: Request) -> dict:
        """Report catalog entry + unique-cluster counts."""
        catalog = _catalog(request)
        return {
            "catalog_entries": catalog.count_catalog_entries(),
            "unique_clusters": catalog.count_unique_clusters(),
        }

    @app.post("/ingest")
    def ingest(body: IngestRequest, request: Request) -> dict:
        """Run the ingest pipeline over *body.path* and return its JSON report."""
        report = IngestPipeline(
            LocalConnector(Path(body.path)),
            catalog=_catalog(request),
        ).run(resume=body.resume)
        return report.to_dict()

    @app.get("/catalog")
    def catalog_list(request: Request) -> list[dict]:
        """Return the full catalog as a JSON list of entries."""
        return _catalog(request).list_entries()

    @app.get("/consolidation-plan")
    def consolidation_plan(path: str, request: Request) -> dict:
        """Inventory *path* into a ConsolidationPlan; raises if not a directory."""
        return build_plan(Path(path)).to_dict()

    # -- SPA -------------------------------------------------------------------

    @app.get("/")
    def spa() -> FileResponse:
        """Serve the review SPA's index.html at the site root."""
        return FileResponse(WEBUI_DIR / "index.html", media_type="text/html")

    app.mount(
        "/app",
        StaticFiles(directory=WEBUI_DIR, html=True),
        name="webui",
    )

    # -- review surfaces --------------------------------------------------------

    def _resolve_entry_id(catalog: Catalog, body: ReviewActionRequest) -> int:
        """Resolve the request to a catalog entry id (asset or direct entry id)."""
        if body.entry_id is not None:
            return body.entry_id
        path = Path(body.asset).resolve()
        connector = LocalConnector(path.parent)
        return resolve_catalog_entry(catalog, connector.connector_id, str(path))

    @app.get("/api/review")
    def review_list(request: Request, status: str | None = None) -> list[dict]:
        """List catalog entries with their current approval decision.

        Optionally filter by *status* (approved | rejected | pending).
        """
        catalog = _catalog(request)
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
        if status:
            status = status.lower()
            rows = [
                r
                for r in rows
                if (r["decision"] is None if status == "pending" else r["decision"] == status)
            ]
        return rows

    def _apply_review(
        catalog: Catalog, body: ReviewActionRequest, action: str
    ) -> dict:
        """Resolve *body* to an entry and apply approve/reject/undo, returning state."""
        approval = ApprovalService(catalog)
        try:
            entry_id = _resolve_entry_id(catalog, body)
            if action == "approve":
                approval.approve(entry_id, body.rationale)
            elif action == "reject":
                approval.reject(entry_id, body.rationale)
            else:
                approval.undo(entry_id)
        except (CatalogEntryNotFound, ApprovalError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        current = approval.current(entry_id)
        return {
            "asset_id": body.asset,
            "entry_id": entry_id,
            "decision": current.decision.value.lower() if current else None,
        }

    @app.post("/api/review/approve")
    def review_approve(body: ReviewActionRequest, request: Request) -> dict:
        """Record an approved decision for the requested entry."""
        return _apply_review(_catalog(request), body, "approve")

    @app.post("/api/review/reject")
    def review_reject(body: ReviewActionRequest, request: Request) -> dict:
        """Record a rejected decision for the requested entry."""
        return _apply_review(_catalog(request), body, "reject")

    @app.post("/api/review/undo")
    def review_undo(body: ReviewActionRequest, request: Request) -> dict:
        """Revert the latest decision for the requested entry (flips active state)."""
        return _apply_review(_catalog(request), body, "undo")

    # -- analyze / propose / render / validate surfaces --------------------------

    #: Well-known render targets: name -> (width, height).
    TARGETS: dict[str, tuple[int, int]] = {"1080p": (1920, 1080), "4k": (3840, 2160)}
    _SHA_RE = re.compile(r"[0-9a-fA-F]{64}")

    def _decode_b64(text: str) -> bytes:
        """Decode base64-encoded payload, raising 400 on malformed input."""
        try:
            return base64.b64decode(text, validate=True)
        except binascii.Error as exc:
            raise HTTPException(status_code=400, detail=f"invalid base64 payload: {exc}") from exc

    def _fetch(store: Any, sha: str) -> bytes:
        """Fetch *sha* from the ContentStore, mapping a miss to 404."""
        try:
            return store.get(sha)
        except StorageError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    def _resolve_image(
        catalog: Catalog, asset: str | None, inline: str | None
    ) -> tuple[bytes, str]:
        """Resolve one source to (bytes, sha) from ``asset`` or base64 ``inline``."""
        if inline is not None:
            data = _decode_b64(inline)
            return data, sha256_hex(data)
        if asset is not None:
            if not _SHA_RE.fullmatch(asset):
                raise HTTPException(
                    status_code=400, detail=f"asset must be a 64-hex content sha, got {asset!r}"
                )
            return _fetch(catalog.content, asset), asset
        raise HTTPException(
            status_code=400, detail="provide 'asset' (content sha) or 'bytes' (base64 image)"
        )

    def _parse_target(
        target: str | dict[str, int] | None,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[int, int]:
        """Resolve a target spec (string, dims dict, or explicit w/h) to a pair."""
        if width is not None and height is not None:
            return int(width), int(height)
        if isinstance(target, dict):
            if "width" not in target or "height" not in target:
                raise HTTPException(
                    status_code=400, detail="target dict requires 'width' and 'height'"
                )
            return int(target["width"]), int(target["height"])
        if isinstance(target, str) and target in TARGETS:
            return TARGETS[target]
        raise HTTPException(
            status_code=400,
            detail=f"unknown target {target!r}; expected one of {sorted(TARGETS)} or a dims dict",
        )

    def _parse_profile(name: str) -> AnalysisProfile:
        """Map a profile string to an :class:`AnalysisProfile`, else 400."""
        try:
            return AnalysisProfile(name.lower())
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"unknown profile {name!r}; expected one of "
                f"{sorted(p.value for p in AnalysisProfile)}",
            ) from None

    @app.post("/api/analyze")
    def analyze(body: AnalyzeRequest, request: Request) -> dict:
        """Analyze one source via LocalAnalysisProvider and return its result."""
        data, sha = _resolve_image(_catalog(request), body.asset, body.bytes)
        profile = _parse_profile(body.profile)
        try:
            return LocalAnalysisProvider().analyze(
                data, profile, asset_id=sha
            ).to_dict()
        except AnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/propose")
    def propose(body: ProposeRequest, request: Request) -> list[dict]:
        """Rank applicable layout treatments for the requested sources.

        Every source is analyzed and every source reaches the policy engine
        (M010/S02) — the ``[:1]``/``[:2]`` slices that used to bound this handler
        silently discarded any third image the caller selected. The bound is now
        explicit: :data:`MAX_LAYOUT_SOURCES`, enforced on ``sources`` by a
        validator (422) and again on the combined list once the primary sha is
        known, before a single image is analyzed (T-10-06).
        """
        catalog = _catalog(request)
        provider = LocalAnalysisProvider()
        sources_sha = list(body.sources)
        primary_sha = None
        primary_data: bytes | None = None
        try:
            if body.asset or body.bytes:
                primary_data, primary_sha = _resolve_image(
                    catalog, body.asset, body.bytes
                )
                if primary_sha not in sources_sha:
                    sources_sha.insert(0, primary_sha)
            if len(sources_sha) > MAX_LAYOUT_SOURCES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"request names {len(sources_sha)} sources, over the "
                        f"{MAX_LAYOUT_SOURCES}-source layout cap — an over-cap "
                        f"request is rejected, never truncated"
                    ),
                )
            # CR-03: build `results` keyed by sha, then read it back out in
            # exactly `sources_sha` order — never "primary first, then the
            # rest". `propose_treatments`'s diptych/triptych/quad/packed blocks
            # zip `results[i]` against `art_request.sources[i]` positionally, so
            # when the caller's JSON body names the primary sha *inside*
            # `sources` at an index other than 0, "primary first" desyncs the
            # two lists and a cell's crop-safety verdict gets attributed to the
            # wrong image's sha in the returned evidence.
            analyzed: dict[str, AnalysisResult] = {}
            if primary_data is not None and primary_sha is not None:
                analyzed[primary_sha] = provider.analyze(
                    primary_data, AnalysisProfile.BALANCED, asset_id=primary_sha
                )
            for other in sources_sha:
                if other in analyzed:
                    continue
                analyzed[other] = provider.analyze(
                    _fetch(catalog.content, other),
                    AnalysisProfile.BALANCED,
                    asset_id=other,
                )
            results = [analyzed[sha] for sha in sources_sha]
        except AnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        target = _parse_target(body.target, body.target_width, body.target_height)
        target_name = body.target if isinstance(body.target, str) else "custom"
        art_request = ArtDirectionRequest(
            target=target_name,
            target_width=target[0],
            target_height=target[1],
            sources=list(sources_sha),
            allow_diptych=body.allow_diptych,
        )
        return [
            proposal.to_dict()
            for proposal in propose_treatments(results, art_request, provider=provider)
        ]

    def _record_render(
        catalog: Catalog,
        manifest: ArtDirectionManifest,
        target: tuple[int, int],
        target_label: str,
        result: RenderResult,
        artifact_sha: str,
    ) -> None:
        """Persist one render row into the ``renders`` table (append-only)."""
        entry_id: int | None = None
        if manifest.sources:
            for entry in catalog.get_by_hash(manifest.sources[0]):
                entry_id = entry["id"]
                break
        catalog.db.execute(
            "INSERT INTO renders"
            " (catalog_entry_id, target, renderer_version, artifact_sha, render_json)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                entry_id,
                target_label,
                result.renderer_version,
                artifact_sha,
                json.dumps(result.to_dict()),
            ),
        )
        catalog.db.commit()

    def _target_label(
        body: RenderRequest, target: tuple[int, int]
    ) -> str:
        """Return a readable target label for persistence (name or ``WxH``)."""
        if isinstance(body.target, str):
            return body.target
        return f"{target[0]}x{target[1]}"

    @app.post("/api/render")
    def render(body: RenderRequest, request: Request) -> dict:
        """Render a manifest/source to a target via DeterministicRenderer.

        The rendered PNG bytes are persisted into the ContentStore (content-
        addressed by their SHA-256) and a ``renders`` journal row is appended.
        The response expands :class:`RenderResult` with the stored ``artifact_sha``.

        The manifest is validated **before** any source bytes are fetched
        (M010/S02): ``DeterministicRenderer`` validates too, but only after this
        handler has already read N blobs out of the ContentStore, so an over-cap
        or malformed manifest would have cost N reads to reject (T-10-07).
        """
        catalog = _catalog(request)
        target = _parse_target(body.target, body.target_width, body.target_height)
        sources_b64 = body.sources or {}
        manifest = _resolve_manifest(catalog, body, sources_b64)
        try:
            manifest.validate()
        except ManifestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        sources = {
            sha: (
                _decode_b64(sources_b64[sha])
                if sha in sources_b64
                else _fetch(catalog.content, sha)
            )
            for sha in manifest.sources
        }
        renderer = DeterministicRenderer()
        try:
            result = renderer.render(manifest, sources, target)
        except RenderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        payload = renderer.render_bytes(manifest, sources, target)
        artifact_sha = catalog.content.put(payload)
        _record_render(
            catalog, manifest, target, _target_label(body, target), result, artifact_sha
        )
        data = result.to_dict()
        data["artifact_sha"] = artifact_sha
        return data

    def _resolve_manifest(
        catalog: Catalog, body: RenderRequest, sources_b64: dict[str, str]
    ) -> ArtDirectionManifest:
        """Build the manifest from ``manifest`` or a minimal single-source one."""
        if body.manifest is not None:
            return ArtDirectionManifest.from_dict(body.manifest)
        data, sha = _resolve_image(catalog, body.asset, body.bytes)
        return ArtDirectionManifest(
            sources=[sha],
            regions=[SourceRegion(source_sha256=sha)],
        )

    @app.post("/api/validate")
    def validate(body: ValidateRequest, request: Request) -> dict:
        """Gate an artifact for publishability via ArtifactValidator.

        A supplied ``manifest`` is resolved for the target and its cells checked
        one by one (M010/S01); an invalid manifest or an unpackable cell list is
        a 400 carrying the error text.
        """
        catalog = _catalog(request)
        if body.artifact_bytes is not None:
            artifact = _decode_b64(body.artifact_bytes)
        elif body.artifact_sha is not None:
            artifact = _fetch(catalog.content, body.artifact_sha)
        else:
            raise HTTPException(
                status_code=400, detail="provide 'artifact_sha' or 'artifact_bytes'"
            )
        target = _parse_target(body.target, body.target_width, body.target_height)
        regions: list[SourceRegion] | None = None
        if body.manifest is not None:
            try:
                manifest = ArtDirectionManifest.from_dict(body.manifest)
                if isinstance(body.target, str):
                    manifest = manifest.resolved_for(body.target)
                regions = resolve_regions(manifest, target)
            except CuratorError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        report = ArtifactValidator().validate(
            artifact,
            body.expected_sha,
            target,
            color_mode=body.color_mode,
            color_profile=body.color_profile,
            source_regions=regions,
        )
        return report.to_dict()

    # -- taste dialogue surface (M008/S07) ---------------------------------------

    def _taste_profile_document(catalog: Catalog, *, seed: bool = True) -> TasteProfile:
        """Rebuild the profile from observations and replay its correction timeline.

        Mirrors the CLI's read path exactly: derive from the append-only
        observation journal, optionally append the low-provenance cold-start
        claims, then re-apply pin/edit/dispute so corrections survive a rebuild.
        """
        profile = ProfileBuilder().build(ObservationStore(catalog).all())
        if seed:
            profile = ColdStartSeeder(catalog).seed(profile)
        for event in ProfileStore(catalog).events():
            profile = _replay_profile_event(profile, event)
        return profile

    def _replay_profile_event(profile: TasteProfile, event: ProfileEvent) -> TasteProfile:
        """Re-apply one timeline event to a freshly rebuilt profile."""
        if event.kind == "dispute":
            return dataclasses.replace(
                profile,
                patterns=[c for c in profile.patterns if c.id != event.claim_id],
                tensions=[c for c in profile.tensions if c.id != event.claim_id],
            )
        if event.kind == "pin":
            def apply(claim: TasteClaim) -> TasteClaim:
                return dataclasses.replace(claim, status="pinned")
        else:
            def apply(claim: TasteClaim) -> TasteClaim:
                return dataclasses.replace(claim, text=event.detail, status="edited")
        return dataclasses.replace(
            profile,
            patterns=[apply(c) if c.id == event.claim_id else c for c in profile.patterns],
            tensions=[apply(c) if c.id == event.claim_id else c for c in profile.tensions],
        )

    def _taste_room(catalog: Catalog) -> ReactionRoom:
        """Build a ReactionRoom over the env-gated extraction provider."""
        provider = resolve_extraction_provider(extraction_config_from_env())
        return ReactionRoom(catalog, provider, catalog.content.root)

    @app.get("/api/taste/profile")
    def taste_profile(request: Request, seed: bool = True) -> dict:
        """Return the taste profile document plus its quotable citations.

        ``seed=false`` omits the low-provenance approval/pairwise history claims.
        ``citations`` is what an explanation would quote — the profile's own
        words, high-provenance first; it is empty for an empty profile.
        """
        profile = _taste_profile_document(_catalog(request), seed=seed)
        payload = profile.to_dict()
        payload["citations"] = [c.to_dict() for c in citations_for(profile)]
        payload["dimensions"] = list(profile_dimensions(profile))
        payload["timeline"] = [e.to_dict() for e in ProfileStore(_catalog(request)).events()]
        return payload

    @app.post("/api/taste/drop")
    def taste_drop(body: TasteDropRequest, request: Request) -> dict:
        """Record one Reaction Room turn over the dropped image(s).

        Images arrive as base64 ``images`` (third-party drops, retained as
        thumbnail + hash) and/or catalog ``shas``. Returns the observation, the
        single probing question this reaction earned, and the "What I learned"
        delta — nothing enters the profile without one.
        """
        catalog = _catalog(request)
        if not body.images and not body.shas:
            raise HTTPException(status_code=400, detail="provide 'images' or 'shas'")
        room = _taste_room(catalog)
        staged: list[Path] = []
        with tempfile.TemporaryDirectory(prefix="curator-taste-drop-") as tmp:
            for index, encoded in enumerate(body.images):
                path = Path(tmp) / f"drop-{index}"
                path.write_bytes(_decode_b64(encoded))
                staged.append(path)
            drops: list[str | Path] = []
            drops.extend(str(p) for p in staged)
            drops.extend(body.shas)
            try:
                session = room.start(drops)
                if body.save:
                    for path in staged:
                        data = path.read_bytes()
                        ref = next(
                            (
                                r
                                for r in room.session_images(session)
                                if r.sha256 == sha256_hex(data)
                            ),
                            None,
                        )
                        if ref is not None and ref.ephemeral and not ref.catalog_saved:
                            save_to_catalog(ref, data, catalog)
                turn = room.react(session, body.note)
                room.finish(session)
            except ReactionRoomUnavailableError as exc:
                # 503: the room is unavailable, not broken — and it never guesses.
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except CuratorError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        learned = WhatILearned.delta_after(session.id, ObservationStore(catalog))
        return {
            "session_id": session.id,
            "observation": turn.observation.to_dict(),
            "question": turn.question.to_dict() if turn.question else None,
            "followups_asked": turn.followups_asked,
            "learned": learned.to_dict(),
        }

    def _taste_claim_action(catalog: Catalog, body: TasteClaimRequest, kind: str) -> dict:
        """Apply pin/edit/dispute to a claim, 404-ing when the profile lacks it."""
        profile = _taste_profile_document(catalog)
        claim = next(
            (
                c
                for c in list(profile.patterns) + list(profile.tensions)
                if c.id == body.claim_id
            ),
            None,
        )
        if claim is None:
            raise HTTPException(
                status_code=404, detail=f"no claim with id {body.claim_id!r}"
            )
        store = ProfileStore(catalog)
        store.apply(profile)
        if kind == "pin":
            event = store.pin(body.claim_id)
        elif kind == "edit":
            if not body.text:
                raise HTTPException(status_code=400, detail="'text' is required to edit")
            event = store.edit(body.claim_id, body.text)
        else:
            event = store.dispute(body.claim_id)
        return {"event": event.to_dict(), "was": claim.to_dict()}

    @app.post("/api/taste/pin")
    def taste_pin(body: TasteClaimRequest, request: Request) -> dict:
        """Pin a claim (it stays, marked, on the append-only timeline)."""
        return _taste_claim_action(_catalog(request), body, "pin")

    @app.post("/api/taste/edit")
    def taste_edit(body: TasteClaimRequest, request: Request) -> dict:
        """Rewrite a claim in the user's own words."""
        return _taste_claim_action(_catalog(request), body, "edit")

    @app.post("/api/taste/dispute")
    def taste_dispute(body: TasteClaimRequest, request: Request) -> dict:
        """Dispute a claim: remove it and mark its evidence for re-interpretation."""
        return _taste_claim_action(_catalog(request), body, "dispute")

    # -- taste vote capture (M009/S01) -------------------------------------------

    def _entry_pair_info(catalog: Catalog, entry_id: int) -> dict[str, Any]:
        """Return ``{entry_id, sha256, asset_id}`` for one catalog entry."""
        row = catalog.db.execute(
            "SELECT sha256, asset_id FROM catalog_entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return {
            "entry_id": entry_id,
            "sha256": str(row[0]) if row else "",
            "asset_id": str(row[1]) if row else "",
        }

    @app.get("/api/taste/pair")
    def taste_pair(request: Request) -> dict:
        """Return the current A/B taste comparison, or ``available: false``.

        This response is the single source of truth the client must echo back
        on vote — ``a.entry_id``/``b.entry_id`` are exactly what
        ``POST /api/taste/vote`` demands and revalidates.
        """
        catalog = _catalog(request)
        pair = next_pair(catalog)
        if pair is None:
            return {
                "available": False,
                "reason": "fewer than two analyzed candidates remaining",
            }
        a, b = pair
        return {
            "available": True,
            "a": _entry_pair_info(catalog, int(a["id"])),
            "b": _entry_pair_info(catalog, int(b["id"])),
        }

    @app.post("/api/taste/vote")
    def taste_vote(body: TasteVoteRequest, request: Request) -> dict:
        """Record a pairwise vote, revalidating the pair the client saw.

        Recomputes ``next_pair`` fresh, independently of whatever the client
        displayed, and 409s on any mismatch against ``a_entry_id``/
        ``b_entry_id`` without recording anything — closes the TOCTOU window
        between ``GET /api/taste/pair`` and this call (catalog state can change
        between the two, e.g. a background watcher analyzing new photos).
        """
        if body.prefer not in ("a", "b"):
            raise HTTPException(status_code=400, detail="'prefer' must be 'a' or 'b'")
        catalog = _catalog(request)
        pair = next_pair(catalog)
        if pair is None:
            raise HTTPException(status_code=400, detail="no pair available")
        a, b = pair
        a_id, b_id = int(a["id"]), int(b["id"])
        if (a_id, b_id) != (body.a_entry_id, body.b_entry_id):
            raise HTTPException(
                status_code=409,
                detail="the compared pair has changed since it was fetched — "
                "GET /api/taste/pair again and retry",
            )
        winner_id, loser_id = (a_id, b_id) if body.prefer == "a" else (b_id, a_id)
        try:
            record = TasteVoteStore(catalog).record_vote(winner_id, loser_id, note=body.note)
        except CuratorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        profile_version = TasteVoteStore(catalog).load_profile().version
        return {
            "vote_group": record.vote_group,
            "winner_entry_id": record.winner_entry_id,
            "loser_entry_id": record.loser_entry_id,
            "profile_version": profile_version,
        }

    @app.get("/api/taste/votes")
    def taste_votes(request: Request) -> dict:
        """List every recorded pairwise vote (winner/loser/note/retracted status)."""
        votes = TasteVoteStore(_catalog(request)).votes()
        return {"votes": [v.to_dict() for v in votes], "count": len(votes)}

    @app.post("/api/taste/retract")
    def taste_retract(body: TasteRetractRequest, request: Request) -> dict:
        """Retract a vote, reversing its effect on the persisted profile.

        Never deletes the underlying rows — 404s when *vote_group* is unknown
        or already retracted rather than silently no-op-succeeding.
        """
        catalog = _catalog(request)
        store = TasteVoteStore(catalog)
        if not store.retract(body.vote_group):
            raise HTTPException(
                status_code=404, detail=f"no vote with id {body.vote_group!r}"
            )
        return {
            "retracted": True,
            "vote_group": body.vote_group,
            "profile_version": store.load_profile().version,
        }

    # -- embedding provider status (M009/S02) ------------------------------------

    @app.get("/api/taste/embedding-status")
    def taste_embedding_status() -> dict:
        """Report whether the local embedding subsystem is usable, with no network call.

        Read-only — no ``--backfill`` equivalent over HTTP; a bulk-recompute loop
        is a CLI-only maintenance operation (avoids adding a long-running
        synchronous mutation route to a loopback API for a milestone that does
        not otherwise need a background-job story — a scope decision, not an
        oversight).
        """
        probe = OnnxEmbeddingProvider().probe()
        return {"ok": probe.ok, "backend": probe.backend.value, "message": probe.message}

    # -- embedding attribution + exemplars (M009/S04) ----------------------------

    def _embedding_liked_shas(
        catalog: Catalog, votes: list[VoteRecord]
    ) -> tuple[list[str], dict[str, int]]:
        """Return ``(liked_shas, sha_to_entry_id)`` for every non-retracted vote winner.

        Mirrors the CLI's ``_taste_liked_shas`` helper exactly (same query, same
        shape) — this project keeps small per-surface duplication for CLI/API glue
        rather than a cross-module shared helper, the same precedent
        ``_entry_pair_info``/``_taste_pair_candidate`` already set.
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

    def _entry_id_for_sha(catalog: Catalog, sha: str) -> int | None:
        """Return the highest-id catalog entry for *sha*, or ``None`` when uncataloged.

        Mirrors the CLI's ``_taste_embedding_resolve`` sha branch exactly (same
        query) — WR-03: a ``bytes``-only request whose content was never
        cataloged legitimately has no entry id, so this returns ``None`` rather
        than raising.
        """
        row = catalog.db.execute(
            "SELECT id FROM catalog_entries WHERE sha256 = ? ORDER BY id DESC LIMIT 1",
            (sha,),
        ).fetchone()
        return int(row[0]) if row is not None else None

    @app.post("/api/taste/embedding-explain")
    def taste_embedding_explain(body: AnalyzeRequest, request: Request) -> dict:
        """Explain one embedding-head score: per-vote attribution + nearest liked exemplars.

        Resolves the target the same way ``/api/taste/explain`` already does
        (``asset``/``bytes`` via ``_resolve_image``), fetches (embedding on demand
        when missing, mirroring the CLI's ``--backfill`` logic) its stored vector,
        fits the head fresh over every resolvable vote (S03 — nothing is
        persisted), then returns ``attribute_score``/``find_exemplars``/
        ``render_rationale`` over it — the identical JSON shape
        ``curator taste embedding-explain`` prints, ``entry_id`` included
        (WR-03: ``None`` for a ``bytes``-only request whose content was never
        cataloged). 503s when the embedding provider itself is unavailable
        (mirrors ``taste_drop``'s "unavailable, not broken" posture for
        ``ReactionRoomUnavailableError``), and again if on-demand embedding
        fails after a passing probe (T-09-05: a model file can be swapped out
        between the probe and this call).
        """
        catalog = _catalog(request)
        data, sha = _resolve_image(catalog, body.asset, body.bytes)
        provider = OnnxEmbeddingProvider()
        probe = provider.probe()
        if not probe.ok:
            raise HTTPException(status_code=503, detail=probe.message)
        store = EmbeddingStore(catalog)
        stored = store.get(sha, provider.model_version)
        try:
            if stored is None:
                vector = provider.embed(data)
                store.set(sha, provider.model_version, vector)
            else:
                vector = stored.vector
        except EmbeddingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        vote_records = TasteVoteStore(catalog).votes()
        vote_vectors = resolve_vote_vectors(vote_records, store, provider.model_version)
        head = fit_embedding_head(vote_vectors, provider.model_version)
        liked_shas, sha_to_entry_id = _embedding_liked_shas(catalog, vote_records)
        attribution = attribute_score(vector, head, vote_vectors)
        exemplars = find_exemplars(
            vector, liked_shas, sha_to_entry_id, store, provider.model_version
        )
        rationale = render_rationale(attribution, exemplars)
        return {
            "entry_id": _entry_id_for_sha(catalog, sha),
            "sha256": sha,
            "score": attribution.score,
            "rationale": rationale,
            "contributions": attribution.contributions,
            "exemplars": [e.to_dict() for e in exemplars],
        }

    # -- embedding-affinity group selection (M010/S04) ---------------------------

    @app.post("/api/packing/group")
    def packing_group(body: GroupRequest, request: Request) -> dict:
        """Propose which cataloged assets belong together with the seed.

        Answers *which* images to group, never how they are arranged — the
        geometry stays in :mod:`curator.artdirection`, which imports nothing from
        ``taste``. Returns the byte-identical JSON shape ``curator group --json``
        prints (``GroupSelection.to_dict()`` plus ``sources``, the ``curator
        propose`` argument list), the house convention for a feature that ships
        on both surfaces.

        An **unavailable** selection — nothing embedded yet, no candidate above
        the threshold — is a normal ``200`` with ``available: false`` and a
        ``reason``, not an error: there is nothing wrong, there is just no group
        to propose. A caller-contract violation (``size`` outside the layout
        bounds, ``pool_size`` over the grouping bound) is a ``400``, and an
        unavailable embedding provider a ``503``, mirroring
        ``/api/taste/embedding-explain``'s posture for the identical condition.
        """
        catalog = _catalog(request)
        _data, sha = _resolve_image(catalog, body.asset, body.bytes)
        provider = OnnxEmbeddingProvider()
        probe = provider.probe()
        if not probe.ok:
            raise HTTPException(status_code=503, detail=probe.message)
        # The seed's own entry id, merged in so ``select_group``'s caller-contract
        # check passes and a seed with no stored vector degrades honestly instead
        # of raising. It is never surfaced (only *members* carry an id), so a
        # ``bytes``-only seed that was never cataloged (WR-03) can use -1.
        seed_entry_id = _entry_id_for_sha(catalog, sha)
        try:
            pool, sha_to_entry_id = resolve_group_pool(
                catalog, provider.model_version, limit=body.pool_size
            )
            selection = select_group(
                sha,
                pool,
                {**sha_to_entry_id, sha: seed_entry_id if seed_entry_id is not None else -1},
                EmbeddingStore(catalog),
                provider.model_version,
                group_size=body.size,
                threshold=body.threshold,
            )
        except GroupingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **selection.to_dict(),
            "sources": resolve_group_sources(catalog, seed_entry_id, selection),
        }

    # -- head-to-head comparison, with uncertainty (M009/S05) --------------------

    def _compare_vector_by_asset_id(
        catalog: Catalog,
        candidates: list[dict[str, Any]],
        analysis_map: dict[str, AnalysisResult],
        embedding_store: EmbeddingStore,
        model_version: str,
    ) -> dict[str, np.ndarray]:
        """Return ``{analysis.asset_id: vector}`` for every candidate with a stored embedding.

        Mirrors the CLI's ``_taste_compare_vector_by_asset_id`` exactly (same
        query, same shape) — the established per-surface duplication precedent
        ``_entry_pair_info``/``_taste_pair_candidate`` and
        ``_embedding_liked_shas``/``_taste_liked_shas`` already set. A
        candidate lacking a stored vector is simply absent — the comparison's
        embedding scorer treats a missing entry as ``0.0`` (no evidence),
        never a crash.
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

    def _compare_embedding_scorer_factory(
        vector_by_asset_id: dict[str, np.ndarray], model_version: str
    ) -> Callable[[Sequence[VoteVectors]], Scorer]:
        """Return a ``training-votes-subset -> Scorer`` factory for ``compare_heads``.

        Refits ``fit_embedding_head`` fresh for whichever vote subset it is
        called with — the full training set, or one of ``compare_heads``'s
        learning-curve checkpoints — so ``compare.py`` itself never needs to
        know about ``EmbeddingStore``/``model_version`` (it stays a pure
        statistics module).
        """

        def factory(votes_subset: Sequence[VoteVectors]) -> Scorer:
            head = fit_embedding_head(votes_subset, model_version)

            def scorer(analysis: AnalysisResult) -> float:
                vector = vector_by_asset_id.get(analysis.asset_id)
                return head.score(vector) if vector is not None else 0.0

            return scorer

        return factory

    @app.get("/api/taste/compare")
    def taste_compare(request: Request) -> dict:
        """Compare the lens and embedding heads over recorded votes, with uncertainty.

        Same pipeline as ``curator taste compare``: splits the non-retracted
        vote history into training/held-out (``HELD_OUT_FRACTION``,
        most-recent votes held out — deterministic, chronological), fits the
        lens scorer over the current persisted profile and the embedding
        scorer fresh over the training votes, evaluates both against the
        held-out set via ``compare_heads``, and returns the full report. 503
        when the embedding provider itself is unavailable (mirrors
        ``taste_embedding_explain``'s posture); 400 when fewer than two
        analyzed candidates exist yet — nothing to compare, not a server
        failure.
        """
        catalog = _catalog(request)
        provider = OnnxEmbeddingProvider()
        probe = provider.probe()
        if not probe.ok:
            raise HTTPException(status_code=503, detail=probe.message)
        candidates, analysis_map = resolve_vote_candidates(catalog)
        if len(candidates) < 2:
            raise HTTPException(
                status_code=400,
                detail="fewer than two analyzed candidates — nothing to compare yet",
            )
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
        vector_by_asset_id = _compare_vector_by_asset_id(
            catalog, candidates, analysis_map, embedding_store, provider.model_version
        )
        current_profile = TasteVoteStore(catalog).load_profile()
        ranker = TasteRanker()

        def lens_scorer(analysis: AnalysisResult) -> float:
            return ranker.personal_delta(analysis, current_profile)[0]

        def baseline_scorer(analysis: AnalysisResult) -> float:
            return 0.0

        embedding_scorer_factory = _compare_embedding_scorer_factory(
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
        return comparison.to_dict()

    @app.post("/api/taste/explain")
    def taste_explain(body: AnalyzeRequest, request: Request) -> dict:
        """Explain one rerank, citing the taste profile by the user's own words.

        This is R036 in the live product: the M007 personal delta now cites the
        real persisted Lens profile (M009/S01 — every recorded vote moves it),
        and the M008 profile supplies the words. With zero votes cast,
        ``TasteVoteStore.load_profile`` reads back byte-identical to
        ``default_profile()``, so the rationale is exactly the uncited baseline
        until a vote actually exists.
        """
        catalog = _catalog(request)
        data, sha = _resolve_image(catalog, body.asset, body.bytes)
        try:
            analysis = LocalAnalysisProvider().analyze(
                data, _parse_profile(body.profile), asset_id=sha
            )
        except AnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        explanation = explain_rank(
            analysis, TasteVoteStore(catalog).load_profile(), _taste_profile_document(catalog)
        )
        return explanation.to_dict()

    return app


# Default import target for ``uvicorn curator.api:app``. Lazy catalog (above)
# means importing this module opens no database.
app = create_app()


if __name__ == "__main__":  # pragma: no cover - interactive/dev entry
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
