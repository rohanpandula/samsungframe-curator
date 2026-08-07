"""Tests for the SPA serving + review endpoints (src/curator/api.py).

Covers the mounted ``webui/`` SPA (GET / and /app), the review JSON surface
(GET /api/review with optional status filter), the approve/reject/undo round-trip
through ApprovalService, and the 404 path for unknown assets. Reuses the shared
``data_root`` fixture so the Catalog resolves an isolated database.
"""

from __future__ import annotations

import base64
import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from curator.api import create_app
from curator.artdirection.manifest import (
    MAX_LAYOUT_SOURCES,
    ArtDirectionManifest,
    SourceRegion,
)
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.hashing import sha256_hex
from curator.render.renderer import DeterministicRenderer
from curator.schema import MIGRATIONS, SCHEMA_VERSION


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


# -- review queue UI + accessibility (M004/S03 T2) -----------------------------


def test_webui_review_mode_and_a11y_markup(client):
    """The served SPA exposes the review queue view and a11y affordances."""
    text = client.get("/").text

    assert 'data-review-mode="queue"' in text
    assert 'id="review-view"' in text
    assert 'aria-label="Review queue"' in text

    # navigation with both views + current-state hint
    assert 'aria-label="Sections"' in text
    assert 'id="nav-catalog"' in text
    assert 'id="nav-review"' in text
    assert 'aria-current="page"' in text

    # skip link + landmarks + live regions
    assert 'class="skip-link"' in text
    assert 'href="#main"' in text
    assert 'role="status"' in text
    assert 'aria-live="polite"' in text
    assert 'aria-live="assertive"' in text
    assert 'role="alert"' in text

    # review actions + filter controls labeled
    for label in ("Approve", "Reject", "Undo", "Pending", "Approved", "Rejected", "All"):
        assert label in text
    assert 'aria-label="Filter review queue by decision"' in text
    assert 'aria-label=' in text  # action buttons carry explicit labels


def test_webui_review_status_pending_returns_only_pending(client, catalog, tmp_path):
    """GET /api/review?status=pending returns only undecided entries."""
    asset_a = _seed(catalog, tmp_path, "a.png")
    _seed(catalog, tmp_path, "b.png")
    client.post("/api/review/approve", json={"asset": asset_a})

    pending = client.get("/api/review", params={"status": "pending"}).json()
    assert len(pending) == 1
    assert pending[0]["asset_id"] != asset_a
    assert pending[0]["decision"] is None


def test_webui_review_approve_moves_out_of_pending(client, catalog, tmp_path):
    """Approving an asset surfaces it under approved and removes it from pending."""
    asset = _seed(catalog, tmp_path, "a.png")

    pending_before = client.get(
        "/api/review", params={"status": "pending"}
    ).json()
    assert asset in [r["asset_id"] for r in pending_before]

    resp = client.post("/api/review/approve", json={"asset": asset})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "approved"

    approved = client.get("/api/review", params={"status": "approved"}).json()
    pending = client.get("/api/review", params={"status": "pending"}).json()
    assert asset in [r["asset_id"] for r in approved]
    assert asset not in [r["asset_id"] for r in pending]


def test_webui_review_undo_after_approve_flips_decision(client, catalog, tmp_path):
    """Undo reverts an approve by moving the entry to the opposite decision.

    The backend's undo is a flip (approved->rejected), not a return to pending,
    because approval history is append-only (R010). This documents the actual
    contract so the UI's shared decision state stays in sync.
    """
    asset = _seed(catalog, tmp_path, "a.png")
    client.post("/api/review/approve", json={"asset": asset})

    undo = client.post("/api/review/undo", json={"asset": asset})
    assert undo.status_code == 200
    assert undo.json()["decision"] == "rejected"

    approved = client.get("/api/review", params={"status": "approved"}).json()
    rejected = client.get("/api/review", params={"status": "rejected"}).json()
    assert asset not in [r["asset_id"] for r in approved]
    assert asset in [r["asset_id"] for r in rejected]


def test_webui_review_reject_moves_out_of_pending(client, catalog, tmp_path):
    """Rejecting an asset surfaces it under rejected and removes it from pending."""
    asset = _seed(catalog, tmp_path, "a.png")
    resp = client.post("/api/review/reject", json={"asset": asset})
    assert resp.status_code == 200
    assert resp.json()["decision"] == "rejected"

    rejected = client.get("/api/review", params={"status": "rejected"}).json()
    pending = client.get("/api/review", params={"status": "pending"}).json()
    assert asset in [r["asset_id"] for r in rejected]
    assert asset not in [r["asset_id"] for r in pending]


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


def _kin_png(marker: int, width=1600, height=1200):
    """A near-kin frame: shared palette and subject, distinct bytes (M010/S02).

    Near-kin so ``LocalAnalysisProvider`` derives a real cross-image affinity
    above the N-up threshold — the API path analyzes for real, with no fixture.
    """
    img = Image.new("RGB", (width, height), (60, 90, 170))
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 150, 900, 900], fill=(210, 180, 60))
    draw.rectangle([10, 10, 20 + marker, 20], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_webui_propose_over_cap_sources_is_422(client):
    """An over-cap ``sources`` list is rejected before the handler body runs."""
    resp = client.post(
        "/api/propose",
        json={"sources": [f"{index:064x}" for index in range(MAX_LAYOUT_SOURCES + 1)]},
    )
    assert resp.status_code == 422
    assert "layout cap" in json.dumps(resp.json())


def test_webui_propose_four_sources_ranks_a_quad(client, catalog):
    """Four sources propose a quad — no [:2] truncation to a pair (M010/S02)."""
    shas = [catalog.content.put(_kin_png(index)) for index in range(4)]
    resp = client.post("/api/propose", json={"sources": shas, "target": "1080p"})
    assert resp.status_code == 200
    proposals = resp.json()
    treatments = {proposal["treatment"] for proposal in proposals}
    assert "quad" in treatments
    quad = next(p for p in proposals if p["treatment"] == "quad")
    assert quad["evidence"]["sources"] == 4
    assert len(quad["evidence"]["cells"]) == 4


def test_webui_propose_keeps_crop_safety_attribution_when_primary_reappears_midlist(
    client, catalog, monkeypatch
):
    """CR-03: a primary sha present but not first in ``sources`` must not desync
    ``results`` from ``art_request.sources`` — otherwise a cell's crop-safety
    verdict is attributed to the wrong image's sha in the returned evidence.

    Mirrors the review's reproduction: ``asset`` names the sha of a genuinely
    crop-*risky* source, which is also present (not first) inside ``sources``
    alongside two crop-*safe* ones. ``LocalAnalysisProvider.analyze`` is
    monkeypatched to return canned, deterministic fixtures keyed by the
    ``asset_id`` the handler passes it, isolating this test to the handler's own
    source-ordering bug rather than real per-pixel crop-safety heuristics.
    """
    import curator.api as api_module
    from analysis_factory import crop_risky_result, crop_safe_result

    sha_x = catalog.content.put(_png(1600, 1200, (10, 10, 10)))
    sha_a = catalog.content.put(_png(1600, 1200, (20, 20, 20)))
    sha_b = catalog.content.put(_png(1600, 1200, (30, 30, 30)))
    fixtures = {
        sha_x: crop_risky_result(sha_x),
        sha_a: crop_safe_result(sha_a),
        sha_b: crop_safe_result(sha_b),
    }

    def fake_analyze(self, source, profile=None, asset_id=None):
        return fixtures[asset_id]

    monkeypatch.setattr(api_module.LocalAnalysisProvider, "analyze", fake_analyze)

    resp = client.post(
        "/api/propose",
        json={"asset": sha_x, "sources": [sha_a, sha_x, sha_b], "target": "1080p"},
    )
    assert resp.status_code == 200
    packed = next(p for p in resp.json() if p["treatment"] == "packed")
    cells = {cell["sha"]: cell for cell in packed["evidence"]["cells"]}
    assert set(cells) == {sha_a, sha_x, sha_b}
    assert cells[sha_a]["crop_safe"] is True
    assert cells[sha_x]["crop_safe"] is False
    assert cells[sha_b]["crop_safe"] is True


def test_webui_render_over_cap_manifest_is_400_before_any_fetch(client):
    """An over-cap manifest is rejected before a single blob is read.

    The shas are deliberately absent from the ContentStore: if validation still
    ran after the source comprehension the response would be the 404 of a
    missing blob, so the 400 is itself the proof that nothing was fetched.
    """
    shas = [f"{index:064x}" for index in range(MAX_LAYOUT_SOURCES + 1)]
    resp = client.post(
        "/api/render",
        json={
            "manifest": {"manifest_version": "1", "sources": shas},
            "target": "1080p",
        },
    )
    assert resp.status_code == 400
    assert "layout cap" in resp.json()["detail"]
    assert "never truncated" in resp.json()["detail"]


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


def test_webui_validate_with_manifest_checks_every_cell(client):
    """POST /api/validate with a manifest gates cell geometry too (M010/S01)."""
    from curator.artdirection.manifest import LayoutTreatment
    from curator.artdirection.packing import Cell, equal_cells, gutter_for_target

    renderer = DeterministicRenderer()
    target = (1920, 1080)
    payload = {sha256_hex(src): src for src in (_png(1600, 1200), _png(1400, 1100))}
    shas = list(payload)
    manifest = ArtDirectionManifest(
        sources=shas,
        regions=equal_cells(shas, Cell(0, 0, *target), gap=gutter_for_target(target)),
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=shas,
    )
    raw = renderer.render_bytes(manifest, payload, target)
    resp = client.post(
        "/api/validate",
        json={
            "artifact_bytes": _b64(raw),
            "expected_sha": sha256_hex(raw),
            "target": "1080p",
            "manifest": manifest.to_dict(),
        },
    )
    assert resp.status_code == 200
    report = resp.json()
    names = {c["name"] for c in report["checks"]}
    assert {"source_region[0]", "source_region[1]", "source_regions_disjoint"} <= names
    assert report["publishable"] is True


def test_webui_validate_rejects_invalid_manifest(client):
    """An over-cap manifest is a 400 carrying the cap message, not a 500."""
    raw = _png(1920, 1080)
    resp = client.post(
        "/api/validate",
        json={
            "artifact_bytes": _b64(raw),
            "expected_sha": sha256_hex(raw),
            "target": "1080p",
            "manifest": {"manifest_version": "999", "sources": ["a"]},
        },
    )
    assert resp.status_code == 400
    assert "unsupported manifest version" in str(resp.json()["detail"])


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


# -- render persistence + schema v7 (M004/S03 T1) ------------------------------


def test_webui_render_persists_artifact_and_returns_artifact_sha(client, catalog):
    """A successful render stores the artifact in ContentStore and returns its sha."""
    png = _png(3000, 2000)
    sha = catalog.content.put(png)
    resp = client.post("/api/render", json={"asset": sha, "target": "1080p"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["sha256"]
    assert data["artifact_sha"] == data["sha256"]
    assert data["artifact_sha"] == sha256_hex(catalog.content.get(data["artifact_sha"]))
    assert catalog.content.exists(data["artifact_sha"])


def test_webui_render_persists_renders_row(client, catalog):
    """Rendering appends an entry to the ``renders`` table (schema v7)."""
    png = _png(3000, 2000)
    sha = catalog.content.put(png)
    resp = client.post("/api/render", json={"asset": sha, "target": "1080p"})
    assert resp.status_code == 200
    artifact_sha = resp.json()["artifact_sha"]

    rows = catalog.db.execute(
        "SELECT catalog_entry_id, target, renderer_version, artifact_sha, render_json"
        " FROM renders WHERE artifact_sha = ?",
        (artifact_sha,),
    ).fetchall()
    assert len(rows) == 1
    _, target, renderer_version, stored_sha, render_json = rows[0]
    assert target == "1080p"
    assert renderer_version
    assert stored_sha == artifact_sha
    assert artifact_sha in render_json


def test_webui_validate_by_stored_artifact_sha_publishable(client, catalog):
    """Validation by the stored artifact_sha from render reports publishable."""
    png = _png(3000, 2000)
    sha = catalog.content.put(png)
    render = client.post("/api/render", json={"asset": sha, "target": "1080p"})
    assert render.status_code == 200
    artifact_sha = render.json()["artifact_sha"]

    resp = client.post(
        "/api/validate",
        json={"artifact_sha": artifact_sha, "expected_sha": artifact_sha, "target": "1080p"},
    )
    assert resp.status_code == 200
    report = resp.json()
    assert report["publishable"] is True
    assert report["valid"] is True


def test_webui_validate_stored_artifact_tampered_expected_sha(client, catalog):
    """A mismatched expected_sha against the stored artifact fails publishability."""
    png = _png(3000, 2000)
    sha = catalog.content.put(png)
    render = client.post("/api/render", json={"asset": sha, "target": "1080p"})
    assert render.status_code == 200
    artifact_sha = render.json()["artifact_sha"]

    resp = client.post(
        "/api/validate",
        json={
            "artifact_sha": artifact_sha,
            "expected_sha": "0" * 64,
            "target": "1080p",
        },
    )
    assert resp.status_code == 200
    report = resp.json()
    assert report["publishable"] is False
    hash_check = next(c for c in report["checks"] if c["name"] == "hash")
    assert hash_check["passed"] is False


def test_webui_validate_unknown_artifact_sha_404(client):
    """Validating a sha not present in the ContentStore returns 404."""
    resp = client.post(
        "/api/validate",
        json={"artifact_sha": "f" * 64, "expected_sha": "0" * 64, "target": "1080p"},
    )
    assert resp.status_code == 404


def test_schema_v7_migration_applies_renders_table(data_root, catalog):
    """Migrations v7/v8/v9 are present; the ``renders`` table exists."""
    assert MIGRATIONS[-1][0] == SCHEMA_VERSION
    rows = catalog.db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='renders'"
    ).fetchall()
    assert len(rows) == 1
    ddl = catalog.db.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='renders'"
    ).fetchall()
    assert "catalog_entry_id" in ddl[0][0]
    assert "artifact_sha" in ddl[0][0]
    assert "render_json" in ddl[0][0]
    assert "renderer_version" in ddl[0][0]
