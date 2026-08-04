"""Tests for the SPA serving + review endpoints (src/curator/api.py).

Covers the mounted ``webui/`` SPA (GET / and /app), the review JSON surface
(GET /api/review with optional status filter), the approve/reject/undo round-trip
through ApprovalService, and the 404 path for unknown assets. Reuses the shared
``data_root`` fixture so the Catalog resolves an isolated database.
"""

from __future__ import annotations

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from curator.api import create_app
from curator.artdirection.manifest import ArtDirectionManifest, SourceRegion
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.hashing import sha256_hex
from curator.render.renderer import DeterministicRenderer


@pytest.fixture
def catalog(data_root):
    """A Catalog backed by a fresh migrated DB under the isolated data root."""
    return Catalog(data_root=data_root)


@pytest.fixture
def client(catalog):
    """A TestClient over a create_app instance bound to the fixture Catalog."""
    return TestClient(create_app(catalog=catalog))


def _seed(catalog, tmp_path, name="a.png"):
    """Add one catalog entry via LocalConnector and return its resolved asset path."""
    folder = tmp_path / "review"
    folder.mkdir(exist_ok=True)
    asset = folder / name
    connector = LocalConnector(folder)
    catalog.add_source(connector.connector_id, str(asset.resolve()), b"image-bytes")
    return str(asset.resolve())


def _decision_map(client):
    """Return {asset_id: decision} from GET /api/review."""
    return {r["asset_id"]: r["decision"] for r in client.get("/api/review").json()}


def test_webui_root_serves_spa(client):
    """GET / returns the SPA's index.html as text/html."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "Darkroom Bench" in resp.text


def test_webui_app_mount_serves_html(client):
    """The /app mount serves index.html (directory + explicit resource)."""
    for path in ("/app/", "/app/index.html"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "Darkroom Bench" in resp.text


def test_webui_smoke_markup_has_catalog_actions_and_scripts(client):
    """The served SPA shell has the catalog grid + action buttons and loads app.js.

    This is the S02 smoke gate: the catalog region id, the bench action labels,
    accessible landmark hints, and the bootstrap script reference must all be
    present in the served index.html.
    """
    resp = client.get("/")
    text = resp.text

    assert 'id="catalog"' in text
    assert 'id="catalog-grid"' in text
    assert 'aria-label="Catalog"' in text

    for label in (
        "Analyze",
        "Propose",
        "Render 1080p",
        "Render 4K",
        "Validate",
        "Approve",
        "Reject",
        "Undo",
    ):
        assert label in text

    assert '<script src="/app/app.js">' in text
    assert 'href="/app/styles.css"' in text
    assert 'aria-live="polite"' in text
    assert 'aria-live="assertive"' in text


def test_webui_static_assets_served(client):
    """The mounted SPA assets are reachable."""
    assert client.get("/app/app.js").status_code == 200
    assert client.get("/app/styles.css").status_code == 200


def test_webui_review_pending_before_any_decision(client, catalog, tmp_path):
    """GET /api/review shows pending (None) for a seeded entry."""
    asset = _seed(catalog, tmp_path)
    assert _decision_map(client)[asset] is None


def test_webui_review_approve_reject_undo_roundtrip(client, catalog, tmp_path):
    """Approve/reject/undo toggle the decision through ApprovalService."""
    asset = _seed(catalog, tmp_path)
    entry_id = next(
        r["entry_id"] for r in client.get("/api/review").json() if r["asset_id"] == asset
    )

    approve = client.post("/api/review/approve", json={"asset": asset})
    assert approve.status_code == 200
    assert approve.json() == {"asset_id": asset, "entry_id": entry_id, "decision": "approved"}
    assert _decision_map(client)[asset] == "approved"

    reject = client.post("/api/review/reject", json={"asset": asset})
    assert reject.status_code == 200
    assert reject.json()["decision"] == "rejected"
    assert _decision_map(client)[asset] == "rejected"

    undo = client.post("/api/review/undo", json={"asset": asset})
    assert undo.status_code == 200
    assert undo.json()["decision"] == "approved"
    assert _decision_map(client)[asset] == "approved"


def test_webui_review_status_filter(client, catalog, tmp_path):
    """GET /api/review?status= filters by decision."""
    asset_a = _seed(catalog, tmp_path, "a.png")
    _seed(catalog, tmp_path, "b.png")
    assert client.post("/api/review/approve", json={"asset": asset_a}).status_code == 200

    pending = client.get("/api/review", params={"status": "pending"}).json()
    approved = client.get("/api/review", params={"status": "approved"}).json()
    assert len(pending) == 1
    assert len(approved) == 1
    assert approved[0]["asset_id"] == asset_a


def test_webui_review_approve_unknown_asset_404(client, catalog, tmp_path):
    """POST approve for an unknown asset returns 404 with a JSON error."""
    resp = client.post(
        "/api/review/approve", json={"asset": "/does/not/exist.png"}
    )
    assert resp.status_code == 404
    assert isinstance(resp.json(), dict)


def test_webui_existing_endpoints_still_work(client, catalog, tmp_path):
    """Adding the review surface does not break /health or /catalog."""
    assert client.get("/health").status_code == 200
    _seed(catalog, tmp_path)
    catalog_resp = client.get("/catalog")
    assert catalog_resp.status_code == 200
    assert len(catalog_resp.json()) == 1


# -- analyze / propose / render / validate (M004/S01 T2+T3) --------------------


def _png(width, height, color=(120, 30, 60)):
    """Encode a solid-color synthetic image as PNG bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _b64(data):
    """Base64-encode bytes for an inline JSON payload."""
    return base64.b64encode(data).decode("ascii")


def test_webui_analyze_synthetic_image(client):
    """POST /api/analyze returns an analysis with profile + quality present."""
    resp = client.post(
        "/api/analyze",
        json={"bytes": _b64(_png(320, 240)), "profile": "balanced"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metadata"]["profile"] == "balanced"
    assert data["quality"]["technical_quality"] >= 0
    assert "sharpness" in data["quality"]
    assert data["asset_id"]


def test_webui_analyze_unknown_profile_400(client):
    """An unknown profile maps to a structured 400."""
    resp = client.post(
        "/api/analyze",
        json={"bytes": _b64(_png(64, 64)), "profile": "bogus"},
    )
    assert resp.status_code == 400
    assert "profile" in str(resp.json()["detail"]).lower()


def test_webui_propose_returns_ranked_proposals(client):
    """POST /api/propose returns a ranked proposal list with rationale."""
    resp = client.post("/api/propose", json={"bytes": _b64(_png(320, 240))})
    assert resp.status_code == 200
    proposals = resp.json()
    assert isinstance(proposals, list)
    assert proposals
    for proposal in proposals:
        assert proposal["treatment"]
        assert proposal["rationale"]
        assert isinstance(proposal["rationale"], list)
        assert "score" in proposal
        assert "evidence" in proposal


def test_webui_render_downscale_to_1080p(client, catalog):
    """Rendering a large source down to 1080p succeeds and returns dims + sha."""
    png = _png(3000, 2000)
    sha = catalog.content.put(png)
    resp = client.post("/api/render", json={"asset": sha, "target": "1080p"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_width"] == 1920
    assert data["target_height"] == 1080
    assert data["sha256"]
    assert data["size_bytes"] > 0
    assert data["treatment"] == "single_fullbleed"


def test_webui_render_manifest_source_from_store(client, catalog):
    """A manifest whose source sha is resolved from the ContentStore renders."""
    png = _png(3000, 2000)
    sha = catalog.content.put(png)
    manifest = {
        "manifest_version": "1",
        "sources": [sha],
        "regions": [{"source_sha256": sha}],
        "layout_treatment": "single_fullbleed",
        "processing_intent": {"upscale_warning": False},
    }
    resp = client.post("/api/render", json={"manifest": manifest, "target": "1080p"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["target_width"] == 1920
    assert data["sha256"]
    assert data["sources"] == [sha]


def test_webui_render_tiny_source_to_huge_target_4xx(client, catalog):
    """Upscaling a tiny source into a huge target is refused (R008)."""
    sha = catalog.content.put(_png(120, 80))
    resp = client.post("/api/render", json={"asset": sha, "target": "4k"})
    assert 400 <= resp.status_code < 500
    detail = str(resp.json()["detail"])
    assert "upscale" in detail.lower() or "R008" in detail


def test_webui_validate_correct_artifact(client, catalog):
    """A correctly-rendered artifact validates as publishable."""
    renderer = DeterministicRenderer()
    src = _png(3000, 2000)
    sha = catalog.content.put(src)
    manifest = ArtDirectionManifest(
        sources=[sha], regions=[SourceRegion(source_sha256=sha)]
    )
    raw = renderer.render_bytes(manifest, {sha: src}, (1920, 1080))
    store_sha = catalog.content.put(raw)
    resp = client.post(
        "/api/validate",
        json={
            "artifact_sha": store_sha,
            "expected_sha": sha256_hex(raw),
            "target": "1080p",
        },
    )
    assert resp.status_code == 200
    report = resp.json()
    assert report["publishable"] is True
    assert report["valid"] is True


def test_webui_validate_tampered_bytes(client):
    """Tampered artifact bytes fail validation with a hash check failure."""
    renderer = DeterministicRenderer()
    src = _png(3000, 2000)
    sha = sha256_hex(src)
    manifest = ArtDirectionManifest(
        sources=[sha], regions=[SourceRegion(source_sha256=sha)]
    )
    raw = renderer.render_bytes(manifest, {sha: src}, (1920, 1080))
    tampered = bytes([raw[0] ^ 0xFF]) + raw[1:]
    resp = client.post(
        "/api/validate",
        json={
            "artifact_bytes": _b64(tampered),
            "expected_sha": sha256_hex(raw),
            "target": "1080p",
        },
    )
    assert resp.status_code == 200
    report = resp.json()
    assert report["publishable"] is False
    hash_check = next(c for c in report["checks"] if c["name"] == "hash")
    assert hash_check["passed"] is False
    assert "sha256 mismatch" in hash_check["reason"]


def test_webui_analyze_malformed_body_4xx(client):
    """A body with no source maps to a structured 4xx."""
    resp = client.post("/api/analyze", json={})
    assert 400 <= resp.status_code < 500
    assert "asset" in str(resp.json()["detail"]).lower()
