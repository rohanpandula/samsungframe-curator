"""Acceptance gate for the self-hosted WebUI surface (M004/S03 T3).

Each scenario is **self-bootstrapping and air-gapped**: it builds its own
deterministic fixture (Pillow synthetic images + an ingested tree), mints a
:class:`~curator.catalog.Catalog` over the isolated ``data_root``, and drives
the FastAPI app served by :func:`curator.api.create_app` via starlette's
``TestClient`` — never relying on cross-test ordering, a live server, or the
network.

This gate covers the SPA + review/analyze/propose/render/validate endpoints:

* W1 — ``GET /`` serves the SPA as text/html and ``GET /catalog`` lists an
  ingested fixture's entries.
* W2 — the analyze -> propose loop over one cataloged synthetic image returns
  structured analysis JSON and a ranked proposal list with rationale whose
  treatment order is deterministic for a given image.
* W3 — render to 1080p (1920x1080) and 4k (3840x2160) returns dims + a stored
  ``artifact_sha``; validate is publishable for a matching expected sha and not
  publishable for a tampered one.
* W4 — approve surfaces an asset under ``?status=approved``; undo flips the
  decision (approved -> rejected) with append-only history preserved.
* W5 — the served HTML carries WCAG/structural affordances: skip link, nav
  landmarks, live regions, approve/reject/undo labels, and catalog + review
  containers, and references ``app.js`` / ``styles.css``.
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from acceptance_harness import build_ingest_tree, run_cli
from curator.api import create_app
from curator.approve import ApprovalService
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.hashing import sha256_hex


def _png(width: int, height: int, color=(120, 30, 60)) -> bytes:
    """Encode a solid-color synthetic image as PNG bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _client(data_root):
    """A TestClient over a create_app instance bound to an isolated Catalog."""
    return TestClient(create_app(catalog=Catalog(data_root=data_root)))


def _cataloged(catalog, tmp_path, name="a.png", size=(640, 480), color=(120, 30, 60)):
    """Seed one cataloged entry from a synthetic PNG; return (asset_path, content_sha)."""
    data = _png(*size, color)
    sha = catalog.content.put(data)
    folder = tmp_path / "fixture"
    folder.mkdir(exist_ok=True)
    asset = folder / name
    asset.write_bytes(data)
    connector = LocalConnector(folder)
    catalog.add_source(connector.connector_id, str(asset.resolve()), data)
    return str(asset.resolve()), sha


# ---------------------------------------------------------------------------
# W1 — the SPA is served and reflects the catalog
# ---------------------------------------------------------------------------


def test_webui_served_and_catalog(data_root, tmp_path):
    """GET / returns the SPA; an ingested fixture's entries appear via /catalog."""
    tree = build_ingest_tree(tmp_path)
    assert run_cli(["ingest", str(tree)])[0] == 0

    client = _client(data_root)
    root = client.get("/")
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")

    entries = client.get("/catalog").json()
    assert entries  # the 50-file fixture lands at least one catalog entry
    assert all(e["asset_id"] for e in entries)
    assert all(e["id"] for e in entries)


# ---------------------------------------------------------------------------
# W2 — analyze -> propose loop is structured and deterministic
# ---------------------------------------------------------------------------


def test_webui_analyze_propose_loop(data_root, tmp_path):
    """Analyze a cataloged image, then propose ranks treatments + rationale."""
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    asset, sha = _cataloged(catalog, tmp_path, "loop.png", (640, 480))

    analysis = client.post(
        "/api/analyze", json={"asset": sha, "profile": "balanced"}
    )
    assert analysis.status_code == 200
    doc = analysis.json()
    assert doc["metadata"]["profile"] == "balanced"
    assert doc["asset_id"] == sha
    assert "quality" in doc and "sharpness" in doc["quality"]

    def _propose():
        return client.post(
            "/api/propose", json={"asset": sha, "target": "1080p"}
        ).json()

    proposals = _propose()
    assert proposals  # non-empty
    for proposal in proposals:
        assert proposal["treatment"]
        assert proposal["rationale"]
        assert isinstance(proposal["rationale"], list)
        assert "score" in proposal

    # Determinism: the same controlled image yields the same treatment order.
    again = _propose()
    assert [p["treatment"] for p in proposals] == [p["treatment"] for p in again]
    assert asset  # cataloged source is exercised end-to-end


# ---------------------------------------------------------------------------
# W3 — render to 1080p/4k then validate the stored artifact
# ---------------------------------------------------------------------------


def test_webui_render_validate_loop(data_root, tmp_path):
    """Render 1080p+4k with stored artifact sha; validate publishable vs tampered."""
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    png = _png(4000, 3000)
    sha = catalog.content.put(png)

    r1080 = client.post("/api/render", json={"asset": sha, "target": "1080p"})
    assert r1080.status_code == 200
    data = r1080.json()
    assert (data["target_width"], data["target_height"]) == (1920, 1080)
    artifact_sha = data["artifact_sha"]
    assert artifact_sha and len(artifact_sha) == 64
    assert sha256_hex(catalog.content.get(artifact_sha)) == artifact_sha

    r4k = client.post("/api/render", json={"asset": sha, "target": "4k"})
    assert r4k.status_code == 200
    four_k = r4k.json()
    assert (four_k["target_width"], four_k["target_height"]) == (3840, 2160)

    good = client.post(
        "/api/validate",
        json={
            "artifact_sha": artifact_sha,
            "expected_sha": artifact_sha,
            "target": "1080p",
        },
    )
    assert good.status_code == 200
    assert good.json()["publishable"] is True

    tampered = client.post(
        "/api/validate",
        json={
            "artifact_sha": artifact_sha,
            "expected_sha": "0" * 64,
            "target": "1080p",
        },
    )
    assert tampered.status_code == 200
    report = tampered.json()
    assert report["publishable"] is False
    hash_check = next(c for c in report["checks"] if c["name"] == "hash")
    assert hash_check["passed"] is False


# ---------------------------------------------------------------------------
# W4 — approve then undo flips the decision; history preserved
# ---------------------------------------------------------------------------


def test_webui_review_approve_undo(data_root, tmp_path):
    """Approve lists under ?status=approved; undo flips back; history append-only."""
    catalog = Catalog(data_root=data_root)
    client = TestClient(create_app(catalog=catalog))
    asset, _ = _cataloged(catalog, tmp_path, "approve.png")

    approve = client.post("/api/review/approve", json={"asset": asset})
    assert approve.status_code == 200
    assert approve.json()["decision"] == "approved"

    approved = client.get("/api/review", params={"status": "approved"}).json()
    assert asset in [r["asset_id"] for r in approved]

    undo = client.post("/api/review/undo", json={"asset": asset})
    assert undo.status_code == 200
    # Undo is a flip (approved -> rejected), documented backend semantics.
    assert undo.json()["decision"] == "rejected"

    approved_after = client.get(
        "/api/review", params={"status": "approved"}
    ).json()
    assert asset not in [r["asset_id"] for r in approved_after]

    entry_id = next(
        r["entry_id"]
        for r in client.get("/api/review").json()
        if r["asset_id"] == asset
    )
    events = ApprovalService(catalog).history(entry_id)
    assert [e.decision.value.lower() for e in events] == ["approved", "rejected"]


# ---------------------------------------------------------------------------
# W5 — WCAG / structural affordances in the served HTML
# ---------------------------------------------------------------------------


def test_webui_a11y_markup(data_root):
    """The SPA shell carries skip link, landmarks, live regions, and action labels."""
    text = _client(data_root).get("/").text

    assert 'class="skip-link"' in text
    assert 'href="#main"' in text
    assert '<nav aria-label="Sections">' in text

    assert 'aria-live="polite"' in text
    assert 'aria-live="assertive"' in text
    assert 'role="status"' in text
    assert 'role="alert"' in text

    for label in ("Analyze", "Propose", "Approve", "Reject", "Undo"):
        assert label in text

    assert 'id="catalog"' in text
    assert 'id="catalog-grid"' in text
    assert 'id="review-view"' in text
    assert 'aria-label="Review queue"' in text

    assert 'src="/app/app.js"' in text
    assert 'href="/app/styles.css"' in text
