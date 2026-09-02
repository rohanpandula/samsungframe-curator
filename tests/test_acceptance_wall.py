"""M011 acceptance: One Wall — the single-flow page and its glue (R048, R049).

The 12th ``make acceptance`` file. It proves, over fixtures and the isolated
``data_root`` only — never a device, never the network — that the whole flow a
person walks in the browser exists end to end through the real HTTP surface:

* Scenario A (R048) — the page is one flow: the three stage ids exist in the
  served markup with their counts, the old bench survives behind ``More tools``,
  and every id/label the earlier web gates pin is still present.
* Scenario B (R048) — pictures instead of hashes: ``GET /api/thumb/{sha}`` serves a
  cached JPEG for a cataloged photo, and ``GET /api/wall`` carries a thumb URL,
  a decision and a score slot for every entry, in one request.
* Scenario C (R048) — the load and score jobs run to completion through the API
  and the counts move: loaded, then scored, best-first.
* Scenario D (R049) — hanging is reachable from the browser's surface for the first
  time: approve → ``POST /api/publish`` renders each approved photo with the policy
  engine's top treatment, journals it through ``PublishCoordinator``, and lands a
  sha-verified 1920x1080 PNG in a folder a person can copy to a USB stick.
* Scenario E (R049) — honesty at the boundary: an unavailable destination is a 400
  carrying its reason, an over-small photo asked for 4K is refused per photo and
  named, and the same wall hung twice on one destination is skipped, not duplicated.
* Scenario F — this file is air-gapped (AST-checked), like every gate before it.
"""

from __future__ import annotations

import ast
import hashlib
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from acceptance_harness import assert_no_network_imports
from curator.api import create_app
from curator.catalog import Catalog

STAGE_IDS = ("stage-load", "catalog", "stage-hang")
COUNT_IDS = ("count-loaded", "count-scored", "count-approved", "count-hung")
PINNED_FROM_EARLIER_GATES = (
    'id="catalog-grid"',
    'id="review-view"',
    'id="taste-view"',
    'id="reaction-room"',
    'id="taste-profile"',
    '<nav aria-label="Sections">',
    'role="status"',
    'role="alert"',
)


@pytest.fixture
def catalog(data_root):
    return Catalog(data_root=data_root)


@pytest.fixture
def client(catalog):
    return TestClient(create_app(catalog=catalog))


def _frame(width: int, height: int, hue: int) -> bytes:
    img = Image.new("RGB", (width, height), (40 + hue, 70, 150))
    draw = ImageDraw.Draw(img)
    draw.rectangle([width // 6, height // 5, width // 2, height * 4 // 5], fill=(220, 180, 70))
    for x in range(0, width, max(1, width // 12)):
        draw.line([(x, 0), (x, height)], fill=(20 + hue, 40, 90), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _folder(tmp_path, frames: dict[str, bytes]) -> Path:
    folder = tmp_path / "shoot"
    folder.mkdir()
    for name, data in frames.items():
        (folder / name).write_bytes(data)
    return folder


# -- Scenario A: one page, one flow ------------------------------------------------


def test_acceptance_wall_page_is_one_flow_with_the_old_bench_folded_away(client):
    text = client.get("/").text
    for stage in STAGE_IDS:
        assert f'id="{stage}"' in text
    for count in COUNT_IDS:
        assert f'id="{count}"' in text
    for control in ('id="score-button"', 'id="hang-button"', 'id="load-form"'):
        assert control in text
    assert 'id="more-tools"' in text  # the bench lives on, one click away
    for pinned in PINNED_FROM_EARLIER_GATES:
        assert pinned in text
    # A stage's primary verb is a button, never a link — one action per stage.
    assert text.count('class="btn primary') >= 3


# -- Scenario B + C: pictures, counts, jobs ---------------------------------------


def test_acceptance_wall_load_then_score_moves_the_counts(client, tmp_path, data_root):
    folder = _folder(
        tmp_path, {"a.png": _frame(1920, 1080, 0), "b.png": _frame(1920, 1080, 40)}
    )

    empty = client.get("/api/wall").json()
    assert empty["counts"] == {"loaded": 0, "scored": 0, "approved": 0, "rejected": 0, "hung": 0}

    load = client.post("/api/load", json={"path": str(folder), "wait": True}).json()
    assert load["state"] == "done", load
    wall = client.get("/api/wall").json()
    assert wall["counts"]["loaded"] == 2 and wall["counts"]["scored"] == 0
    assert wall["folders"] == [str(folder.resolve())]

    # Pictures: every entry has a thumb URL that serves a real, cached JPEG.
    for entry in wall["entries"]:
        response = client.get(entry["thumb"])
        assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
        assert (data_root / "thumbs" / f"{entry['sha256']}-320.jpg").is_file()

    score = client.post("/api/score", json={"wait": True}).json()
    assert score["state"] == "done" and score["result"] == {"scored": 2, "failed": 0, "total": 2}
    wall = client.get("/api/wall").json()
    assert wall["counts"]["scored"] == 2
    scores = [e["score"] for e in wall["entries"]]
    assert all(isinstance(s, float) for s in scores) and scores == sorted(scores, reverse=True)
    # The page reads job state from the same call, so a reload never loses a running job.
    assert set(wall["jobs"]) == {"load", "score", "publish"}


# -- Scenario D: hanging is reachable from the browser's surface ---------------------


def test_acceptance_wall_hangs_approved_photos_in_a_folder_with_verified_bytes(
    client, tmp_path
):
    folder = _folder(
        tmp_path, {"keep.png": _frame(1920, 1080, 10), "pass.png": _frame(1920, 1080, 80)}
    )
    loaded = client.post("/api/load", json={"path": str(folder), "wait": True}).json()
    assert loaded["state"] == "done"
    wall = client.get("/api/wall").json()
    keep = next(e for e in wall["entries"] if e["name"] == "keep.png")
    approved = client.post(
        "/api/review/approve", json={"asset": keep["asset_id"], "entry_id": keep["entry_id"]}
    )
    assert approved.status_code == 200

    usb = tmp_path / "usb"
    hang = client.post(
        "/api/publish",
        json={"destination": "folder", "output": "1080p", "folder": str(usb), "wait": True},
    ).json()
    assert hang["state"] == "done", hang
    result = hang["result"]
    assert (result["approved"], result["hung"], result["failed"]) == (1, 1, 0)
    item = result["items"][0]
    assert item["status"] == "hung" and item["treatment"]
    written = usb / item["artifact_id"]
    assert written.is_file()
    assert hashlib.sha256(written.read_bytes()).hexdigest() == item["artifact_sha"]
    with Image.open(written) as im:
        assert im.size == (1920, 1080)

    wall = client.get("/api/wall").json()
    assert wall["counts"]["hung"] == 1
    hung = next(e for e in wall["entries"] if e["entry_id"] == keep["entry_id"])
    assert hung["hung"][0]["artifact_id"] == item["artifact_id"]
    assert hung["hung"][0]["adapter_id"].startswith("folder:")


# -- Scenario E: honesty at the boundary ---------------------------------------------


def test_acceptance_wall_refuses_loudly_and_never_duplicates(client, tmp_path):
    folder = _folder(
        tmp_path, {"small.png": _frame(640, 480, 5), "big.png": _frame(1920, 1080, 30)}
    )
    loaded = client.post("/api/load", json={"path": str(folder), "wait": True}).json()
    assert loaded["state"] == "done"
    for entry in client.get("/api/wall").json()["entries"]:
        client.post(
            "/api/review/approve", json={"asset": entry["asset_id"], "entry_id": entry["entry_id"]}
        )

    samsung = client.post("/api/publish", json={"destination": "samsung", "wait": True})
    assert samsung.status_code == 400 and "transport" in samsung.json()["detail"]
    destinations = {d["id"]: d for d in client.get("/api/wall").json()["destinations"]}
    assert destinations["samsung"]["available"] is False and destinations["samsung"]["reason"]

    at_4k = client.post(
        "/api/publish", json={"destination": "simulator", "output": "4k", "wait": True}
    ).json()["result"]
    by_name = {item["name"]: item for item in at_4k["items"]}
    # 640x480 would need an unapproved upscale (R008): refused and named, never hidden.
    assert by_name["small.png"]["status"] == "error" and by_name["small.png"]["error"]
    assert at_4k["failed"] >= 1 and at_4k["hung"] == 0  # 1920x1080 cannot honestly fill 4K either

    usb = tmp_path / "usb"
    body = {"destination": "folder", "output": "1080p", "folder": str(usb), "wait": True}
    first = client.post("/api/publish", json=body).json()["result"]
    second = client.post("/api/publish", json=body).json()["result"]
    # big.png hangs once; small.png is refused both times; nothing is ever duplicated.
    assert (first["hung"], first["skipped"], first["failed"]) == (1, 0, 1)
    assert (second["hung"], second["skipped"], second["failed"]) == (0, 1, 1)
    assert len(list(usb.iterdir())) == 1


# -- Scenario F: air-gapped ------------------------------------------------------------


def test_acceptance_wall_module_is_air_gapped() -> None:
    """This gate never reaches the network, checked on its own AST, not by substring."""
    assert_no_network_imports()
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"requests", "httpx", "urllib", "socket", "aiohttp", "http"}
    assert not (imported & banned), imported & banned
