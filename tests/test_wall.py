"""One Wall backend glue (M011/S01): thumbnails, wall state, the three jobs, publishing.

Everything runs against the isolated ``data_root`` fixture with real PNG bytes,
the real analysis engine, the real renderer and validator, and the real
``dest/`` adapters — the simulator in memory and the filesystem adapter in a
temp folder. No network, no device.
"""

from __future__ import annotations

import hashlib
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from curator.api import create_app
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.wall import JobRunner, thumbnail_bytes


@pytest.fixture
def catalog(data_root):
    return Catalog(data_root=data_root)


@pytest.fixture
def client(catalog):
    return TestClient(create_app(catalog=catalog))


def _frame(width: int, height: int, marker: int = 0) -> bytes:
    """A frame with a subject and a gradient — enough structure for a real score."""
    img = Image.new("RGB", (width, height), (60, 90, 170))
    draw = ImageDraw.Draw(img)
    draw.rectangle([width // 8, height // 8, width // 2, height * 3 // 4], fill=(210, 180, 60))
    for x in range(0, width, max(1, width // 16)):
        draw.line([(x, 0), (x, height)], fill=(40 + marker, 60, 120), width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _seed(catalog: Catalog, tmp_path, name: str, data: bytes) -> tuple[int, str]:
    """Catalog *data* under a real file path; return ``(entry_id, sha)``."""
    folder = tmp_path / "wall-src"
    folder.mkdir(exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    connector = LocalConnector(folder)
    sha = catalog.add_source(connector.connector_id, str(path.resolve()), data)
    entry_id = int(catalog.get_by_hash(sha)[0]["id"])
    return entry_id, sha


# -- thumbnails -----------------------------------------------------------------


def test_thumbnail_bytes_bounds_the_long_side_and_is_jpeg() -> None:
    out = thumbnail_bytes(_frame(1600, 1200), 320)
    with Image.open(io.BytesIO(out)) as im:
        assert im.format == "JPEG"
        assert max(im.size) == 320


def test_thumb_route_serves_cached_jpeg_and_rejects_bad_shas(client, catalog, tmp_path, data_root):
    _, sha = _seed(catalog, tmp_path, "a.png", _frame(1600, 1200))

    first = client.get(f"/api/thumb/{sha}")
    assert first.status_code == 200
    assert first.headers["content-type"] == "image/jpeg"
    assert "immutable" in first.headers["cache-control"]
    with Image.open(io.BytesIO(first.content)) as im:
        assert max(im.size) == 320
    assert (data_root / "thumbs" / f"{sha}-320.jpg").is_file()
    assert client.get(f"/api/thumb/{sha}").content == first.content  # served from the cache

    small = client.get(f"/api/thumb/{sha}?w=64")
    with Image.open(io.BytesIO(small.content)) as im:
        assert max(im.size) == 64
    huge = client.get(f"/api/thumb/{sha}?w=99999")  # clamped, never rejected
    with Image.open(io.BytesIO(huge.content)) as im:
        assert max(im.size) == 1600  # thumbnail() never upscales the 1600 px source

    assert client.get("/api/thumb/" + "0" * 64).status_code == 404
    assert client.get("/api/thumb/not-a-sha").status_code == 400


# -- the job runner ---------------------------------------------------------------


def test_job_runner_reports_done_error_and_running_once() -> None:
    runner = JobRunner()
    assert runner.status("score").state == "idle"
    done = runner.start("score", lambda progress: {"scored": 1}, wait=True)
    assert done.state == "done" and done.result == {"scored": 1}

    def boom(progress):
        raise ValueError("no")

    failed = runner.start("publish", boom, wait=True)
    assert failed.state == "error" and "ValueError: no" in failed.message
    assert runner.status("publish").to_dict()["state"] == "error"


# -- wall state + scoring -----------------------------------------------------------


def test_wall_counts_score_job_and_best_first_order(client, catalog, tmp_path):
    _seed(catalog, tmp_path, "plain.png", _frame(1920, 1080))
    _seed(catalog, tmp_path, "busy.png", _frame(1920, 1080, marker=90))

    wall = client.get("/api/wall").json()
    assert wall["counts"] == {"loaded": 2, "scored": 0, "approved": 0, "rejected": 0, "hung": 0}
    assert wall["jobs"]["score"]["state"] == "idle"
    assert [d["id"] for d in wall["destinations"]] == ["folder", "simulator", "samsung"]
    assert wall["destinations"][2]["available"] is False
    assert all(e["thumb"].startswith("/api/thumb/") for e in wall["entries"])
    assert wall["folders"] == [str((tmp_path / "wall-src").resolve())]

    status = client.post("/api/score", json={"wait": True}).json()
    assert status["state"] == "done"
    assert status["result"] == {"scored": 2, "failed": 0, "total": 2}
    assert client.get("/api/score").json()["done"] == 2

    wall = client.get("/api/wall").json()
    assert wall["counts"]["scored"] == 2
    scores = [e["score"] for e in wall["entries"]]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(s, float) for s in scores)

    # Scoring again has nothing to do — and says so, rather than re-scoring.
    again = client.post("/api/score", json={"wait": True}).json()
    assert again["result"]["total"] == 0


def test_approve_by_entry_id_feeds_the_counts(client, catalog, tmp_path):
    entry_id, _ = _seed(catalog, tmp_path, "a.png", _frame(1920, 1080))
    approved = client.post("/api/review/approve", json={"asset": "", "entry_id": entry_id})
    assert approved.status_code == 200
    wall = client.get("/api/wall").json()
    assert wall["counts"]["approved"] == 1
    assert wall["entries"][0]["decision"] == "approved"


# -- publishing ---------------------------------------------------------------------


def test_publish_hangs_approved_photos_on_the_simulator_and_in_a_folder(
    client, catalog, tmp_path
):
    approved_id, sha = _seed(catalog, tmp_path, "keep.png", _frame(1920, 1080))
    _seed(catalog, tmp_path, "skip.png", _frame(1920, 1080, marker=50))  # never approved
    client.post("/api/review/approve", json={"asset": "", "entry_id": approved_id})

    status = client.post(
        "/api/publish", json={"destination": "simulator", "output": "1080p", "wait": True}
    ).json()
    assert status["state"] == "done", status
    result = status["result"]
    assert (result["approved"], result["hung"], result["skipped"], result["failed"]) == (1, 1, 0, 0)
    item = result["items"][0]
    assert item["status"] == "hung" and item["treatment"]
    assert item["artifact_id"] == f"{approved_id:05d}-{sha[:12]}-1080p.png"

    wall = client.get("/api/wall").json()
    assert wall["counts"]["hung"] == 1
    hung_entry = next(e for e in wall["entries"] if e["entry_id"] == approved_id)
    assert hung_entry["hung"][0]["adapter_id"] == "simulator"
    # The artifact is content-addressed in the store and thumbnail-able like any photo.
    assert client.get(f"/api/thumb/{item['artifact_sha']}").status_code == 200

    folder = tmp_path / "usb"
    status = client.post(
        "/api/publish",
        json={"destination": "folder", "output": "1080p", "folder": str(folder), "wait": True},
    ).json()
    result = status["result"]
    assert result["hung"] == 1 and result["location"] == str(folder)
    written = folder / item["artifact_id"]
    assert written.is_file()
    assert hashlib.sha256(written.read_bytes()).hexdigest() == item["artifact_sha"]
    with Image.open(written) as im:
        assert im.size == (1920, 1080)

    # Publishing the same wall again is idempotent: journaled as applied, skipped.
    again = client.post(
        "/api/publish",
        json={"destination": "folder", "output": "1080p", "folder": str(folder), "wait": True},
    ).json()["result"]
    assert (again["hung"], again["skipped"]) == (0, 1)


def test_publish_refuses_an_upscale_per_photo_and_bad_requests_up_front(
    client, catalog, tmp_path
):
    small_id, _ = _seed(catalog, tmp_path, "small.png", _frame(640, 480))
    client.post("/api/review/approve", json={"asset": "", "entry_id": small_id})

    status = client.post(
        "/api/publish", json={"destination": "simulator", "output": "4k", "wait": True}
    ).json()
    result = status["result"]
    assert result["hung"] == 0 and result["failed"] == 1
    assert result["items"][0]["status"] == "error"
    assert result["items"][0]["error"]

    samsung = client.post("/api/publish", json={"destination": "samsung", "wait": True})
    assert samsung.status_code == 400
    assert "transport" in samsung.json()["detail"]
    assert client.post("/api/publish", json={"output": "8k"}).status_code == 400
    assert client.post("/api/publish", json={"destination": "cloud"}).status_code == 400


def test_load_job_ingests_a_folder_and_rejects_a_non_folder(client, tmp_path):
    folder = tmp_path / "incoming"
    folder.mkdir()
    (folder / "one.png").write_bytes(_frame(800, 600))
    (folder / "two.png").write_bytes(_frame(800, 600, marker=30))

    status = client.post("/api/load", json={"path": f" {folder}/ ", "wait": True}).json()
    assert status["state"] == "done", status
    assert status["result"]["folder"] == str(folder)
    assert client.get("/api/wall").json()["counts"]["loaded"] == 2

    bad = client.post("/api/load", json={"path": str(folder / "one.png"), "wait": True}).json()
    assert bad["state"] == "error" and "not a folder" in bad["message"]
