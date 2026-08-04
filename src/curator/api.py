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

from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.consolidate.plan import build_plan
from curator.ingest.pipeline import IngestPipeline

# Loopback-only bind address (MEM003): never expose the API on a LAN interface.
HOST = "127.0.0.1"
PORT = 8765


class IngestRequest(BaseModel):
    """JSON request body for ``POST /ingest``.

    ``path`` is the local folder to ingest; ``resume`` (default False) resumes a
    prior ingest from the ``ingest_journal`` checkpoint (see IngestPipeline.run).
    """

    path: str
    resume: bool = False


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

    return app


# Default import target for ``uvicorn curator.api:app``. Lazy catalog (above)
# means importing this module opens no database.
app = create_app()


if __name__ == "__main__":  # pragma: no cover - interactive/dev entry
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
