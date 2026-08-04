"""Tests for the S04 FastAPI app (src/curator/api.py).

Proves the create_app factory's 5 endpoints plus /docs drive the real subsystems
(reusing Catalog / IngestPipeline / build_plan — no reimplemented logic) and that
a TestClient round-trips them end to end: /health, /status, POST /ingest,
/catalog, /consolidation-plan, and the interactive /docs OpenAPI UI. Uses the
shared ``data_root`` fixture so every Catalog resolves an isolated database.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from consolidate_fixture import build_consolidation_fixture
from curator.api import create_app
from curator.catalog import Catalog
from fixture_library import INDEXED_FILES, TOTAL_CLUSTERS, build_fixture


@pytest.fixture
def catalog(data_root):
    """A Catalog backed by a fresh migrated DB under the isolated data root."""
    return Catalog(data_root=data_root)


@pytest.fixture
def client(catalog):
    """A TestClient over a create_app instance bound to the fixture Catalog."""
    return TestClient(create_app(catalog=catalog))


def test_api_get_health(client):
    """GET /health returns the documented healthy payload with the entry count."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy", "catalog_entries": 0}


def test_api_health_reflects_ingest(client, tmp_path):
    """GET /health reports the entry count after a POST /ingest."""
    folder = build_fixture(tmp_path / "fixture").root
    assert client.post("/ingest", json={"path": str(folder)}).status_code == 200
    body = client.get("/health").json()
    assert body == {"status": "healthy", "catalog_entries": INDEXED_FILES}


def test_api_get_status(client, tmp_path):
    """GET /status reports catalog entries + unique clusters after an ingest."""
    folder = build_fixture(tmp_path / "fixture").root
    assert client.post("/ingest", json={"path": str(folder)}).status_code == 200
    body = client.get("/status").json()
    assert body["catalog_entries"] == INDEXED_FILES
    assert body["unique_clusters"] == TOTAL_CLUSTERS


def test_api_post_ingest(client, tmp_path):
    """POST /ingest runs the real IngestPipeline and returns its JSON report."""
    folder = build_fixture(tmp_path / "fixture").root
    resp = client.post("/ingest", json={"path": str(folder)})
    assert resp.status_code == 200
    report = resp.json()
    assert report["connector_id"] == f"local:{folder.resolve()}"
    assert report["total_enumerated"] == 50
    assert report["indexed_count"] == INDEXED_FILES
    assert report["unique_clusters"] == TOTAL_CLUSTERS
    assert len(report["entries"]) == INDEXED_FILES


def test_api_post_ingest_accepts_resume_flag(client, tmp_path):
    """POST /ingest honors the optional resume boolean in the JSON body."""
    folder = build_fixture(tmp_path / "fixture").root
    resp = client.post("/ingest", json={"path": str(folder), "resume": True})
    assert resp.status_code == 200
    assert resp.json()["indexed_count"] == INDEXED_FILES


def test_api_get_catalog(client, tmp_path):
    """GET /catalog lists every catalog_entries row after an ingest."""
    folder = build_fixture(tmp_path / "fixture").root
    assert client.post("/ingest", json={"path": str(folder)}).status_code == 200
    entries = client.get("/catalog").json()
    assert isinstance(entries, list)
    assert len(entries) == INDEXED_FILES
    first = entries[0]
    # The documented catalog_entries columns are present.
    for key in (
        "connector_id",
        "asset_id",
        "revision",
        "sha256",
        "cluster_id",
        "best_original",
    ):
        assert key in first


def test_api_get_consolidation_plan(client, tmp_path):
    """GET /consolidation-plan inventories a legacy folder via build_plan."""
    fixture = build_consolidation_fixture(tmp_path / "consolidation")
    resp = client.get("/consolidation-plan", params={"path": str(fixture.root)})
    assert resp.status_code == 200
    plan = resp.json()
    assert plan["source_path"] == str(fixture.root.resolve())
    # The 8-group plan surface is present and machine-parseable.
    for key in (
        "exact_dupes",
        "near_dupes",
        "higher_res_originals",
        "filename_collisions",
        "panels",
        "sidecars",
        "corrupt",
        "missing_date",
    ):
        assert key in plan
    assert plan["corrupt"] != []  # broken.jpg is inventoried as corrupt


def test_api_docs_served(client):
    """GET /docs serves the interactive OpenAPI UI (demo gate)."""
    resp = client.get("/docs")
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower() or "<html" in resp.text.lower()


def test_api_consolidation_plan_rejects_non_directory(client, catalog, tmp_path):
    """A non-directory ?path= for /consolidation-plan is a fatal subsystem error.

    build_plan raises CuratorError for a non-directory source (matching the CLI);
    with ``raise_server_exceptions=False`` that surfaces as HTTP 500 so operators
    can observe the failure rather than a silently-empty plan.
    """
    failing = TestClient(create_app(catalog=catalog), raise_server_exceptions=False)
    resp = failing.get(
        "/consolidation-plan",
        params={"path": str(tmp_path / "no-such-dir")},
    )
    assert resp.status_code == 500


def test_api_ingest_missing_folder_is_benign(client, tmp_path):
    """POST /ingest over a missing folder returns an empty (non-fatal) report.

    The read-only LocalConnector enumerates nothing for a non-existent path, so the
    pipeline reports zero work rather than raising — an operator can detect the
    empty result instead of crashing the process.
    """
    resp = client.post(
        "/ingest", json={"path": str(tmp_path / "does-not-exist")}
    )
    assert resp.status_code == 200
    report = resp.json()
    assert report["total_enumerated"] == 0
    assert report["indexed_count"] == 0


def test_api_openapi_lists_endpoints(client):
    """The OpenAPI schema exposes all five documented endpoints."""
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"])
    assert {
        "/health",
        "/status",
        "/ingest",
        "/catalog",
        "/consolidation-plan",
    } <= paths
    assert schema["paths"]["/health"]["get"] is not None
