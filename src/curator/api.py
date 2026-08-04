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
import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import FileResponse

from curator.analysis.cli_utils import resolve_catalog_entry
from curator.analysis.errors import AnalysisError, CatalogEntryNotFound
from curator.analysis.local import LocalAnalysisProvider
from curator.analysis.profiles import AnalysisProfile
from curator.approve import ApprovalError, ApprovalService
from curator.artdirection.manifest import ArtDirectionManifest, SourceRegion
from curator.artdirection.policy import ArtDirectionRequest, propose_treatments
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.consolidate.plan import build_plan
from curator.errors import StorageError
from curator.hashing import sha256_hex
from curator.ingest.pipeline import IngestPipeline
from curator.render.renderer import DeterministicRenderer, RenderError
from curator.render.validate import ArtifactValidator

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
    additional content shas (a second one enables a diptych proposal). The
    target is ``1080p``/``4k`` (or an explicit width/height pair).
    """

    asset: str | None = None
    bytes: str | None = None
    sources: list[str] = []
    target: str = "1080p"
    target_width: int | None = None
    target_height: int | None = None
    allow_diptych: bool = True


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
    """

    artifact_sha: str | None = None
    artifact_bytes: str | None = None
    expected_sha: str
    target: str | dict[str, int] | None = None
    target_width: int | None = None
    target_height: int | None = None
    color_mode: str = "RGB"
    color_profile: str = "sRGB"


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
        """Rank applicable layout treatments for the requested sources."""
        catalog = _catalog(request)
        provider = LocalAnalysisProvider()
        results = []
        sources_sha = list(body.sources)
        primary_sha = None
        try:
            if body.asset or body.bytes:
                data, primary_sha = _resolve_image(catalog, body.asset, body.bytes)
                results.append(
                    provider.analyze(data, AnalysisProfile.BALANCED, asset_id=primary_sha)
                )
                if primary_sha not in sources_sha:
                    sources_sha.insert(0, primary_sha)
            for other in [s for s in sources_sha if s != primary_sha][:1]:
                results.append(
                    provider.analyze(
                        _fetch(catalog.content, other),
                        AnalysisProfile.BALANCED,
                        asset_id=other,
                    )
                )
        except AnalysisError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        target = _parse_target(body.target, body.target_width, body.target_height)
        target_name = body.target if isinstance(body.target, str) else "custom"
        art_request = ArtDirectionRequest(
            target=target_name,
            target_width=target[0],
            target_height=target[1],
            sources=sources_sha[:2],
            allow_diptych=body.allow_diptych,
        )
        return [
            proposal.to_dict()
            for proposal in propose_treatments(results, art_request, provider=provider)
        ]

    @app.post("/api/render")
    def render(body: RenderRequest, request: Request) -> dict:
        """Render a manifest/source to a target via DeterministicRenderer."""
        catalog = _catalog(request)
        target = _parse_target(body.target, body.target_width, body.target_height)
        sources_b64 = body.sources or {}
        manifest = _resolve_manifest(catalog, body, sources_b64)
        sources = {
            sha: (
                _decode_b64(sources_b64[sha])
                if sha in sources_b64
                else _fetch(catalog.content, sha)
            )
            for sha in manifest.sources
        }
        try:
            return DeterministicRenderer().render(manifest, sources, target).to_dict()
        except RenderError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
        """Gate an artifact for publishability via ArtifactValidator."""
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
        report = ArtifactValidator().validate(
            artifact,
            body.expected_sha,
            target,
            color_mode=body.color_mode,
            color_profile=body.color_profile,
        )
        return report.to_dict()

    return app


# Default import target for ``uvicorn curator.api:app``. Lazy catalog (above)
# means importing this module opens no database.
app = create_app()


if __name__ == "__main__":  # pragma: no cover - interactive/dev entry
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
