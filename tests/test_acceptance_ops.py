"""Acceptance gate for the M006/S05 orchestrator/provider/migration/packaging surface.

This module ships the deterministic, air-gapped acceptance gate for the four
interchangeable operational subsystems shipped across M006/S01-S04. Each scenario
is **self-bootstrapping**: it mints its own fixtures over the isolated ``data_root``
(from conftest) and drives the subsystem objects directly — never relying on
cross-test ordering, a live server, or the network.

* S1 — cloud/hybrid routing + privacy disclosure + exclusions: the router pins
  derivative kinds local and semantic stages to the cloud, the disclosure states
  exactly what leaves (and what never does), a per-source exclusion strips the
  disallowed payload components, a cloud outage degrades to local with a recorded
  Pause, and no secret/credential ever reaches the cloud runtime.
* S2 — durable job orchestrator: a multi-phase ingest-like job crashes mid-phase,
  is classified (TRANSIENT), and a fresh orchestrator resumes from the checkpoint,
  completes without duplicating content-addressed art, and never regresses
  last-known-good; PERMANENT and USER_CANCELLED outcomes are also pinned.
* S3 — non-destructive migration: a dry run reports the discovered counts and
  backs up / imports nothing; a real run backs up first and imports idempotently
  (no duplicate rows) while leaving sources untouched, with non-empty rollback
  limitations in the report.
* S4 — packaging + headless preflight: the launchd plist runs ``--headless start``,
  both Dockerfiles carry FROM/COPY/ENTRYPOINT, and ``headless start --check`` exits
  2 without a secret, 0 with JSON status when one is set, and degrades to CPU when
  CUDA is requested but unavailable.
"""

from __future__ import annotations

import io
import json
import os
import plistlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from curator.analysis.local import LocalAnalysisProvider
from curator.analysis.profiles import AnalysisProfile
from curator.catalog import Catalog
from curator.hashing import sha256_hex
from curator.jobs import JobFailure, JobKind, JobOrchestrator, JobOutcome
from curator.migrate import MigrationService, build_plan
from curator.providers import (
    COMPONENT_FACES,
    COMPONENT_GPS,
    CloudAnalysisProvider,
    ExclusionPolicy,
    HybridRouter,
    SyntheticCloudAnalysisRuntime,
)

REPO = Path(__file__).resolve().parents[1]
PLIST = REPO / "packaging" / "launchd" / "com.rohan.curator.plist"
DOCKERFILE = REPO / "packaging" / "docker" / "Dockerfile"
DOCKERFILE_CUDA = REPO / "packaging" / "docker" / "Dockerfile.cuda"

BALANCED = AnalysisProfile.BALANCED
_SECRET_ENV_SUFFIXES = ("_TOKEN", "_API_KEY")


def _make_jpeg(color: tuple[int, int, int] = (120, 130, 140)) -> bytes:
    """Build a deterministic decodable JPEG in memory."""
    img = Image.new("RGB", (800, 600), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _jpg_with_gps() -> bytes:
    """Build a JPEG bearing a synthetic EXIF GPS record."""
    img = Image.new("RGB", (200, 100), (90, 120, 160))
    exif = img.getexif()
    exif.get_ifd(0x8825).update(  # type: ignore[union-attr]
        {1: "N", 2: (52.0, 5.0, 20.0), 3: "E", 4: (4.0, 4.0, 30.0)}
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def _blob_count(data_root: Path) -> int:
    """Count content-addressed blob files under *data_root* (excludes temp)."""
    content_root = data_root / "content"
    if not content_root.exists():
        return 0
    return len(list(content_root.glob("*/*/*")))


def _snapshot(folder: Path) -> dict[str, bytes]:
    """Snapshot every file under *folder* as ``{relative_path: bytes}``."""
    return {
        p.relative_to(folder).as_posix(): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _clear_env_secrets(monkeypatch) -> None:
    """Drop ambient ``*_TOKEN`` / ``*_API_KEY`` vars so preflight is deterministic."""
    for key in list(os.environ):
        if key.endswith(_SECRET_ENV_SUFFIXES):
            monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# S1 — cloud/hybrid routing + privacy disclosure + exclusions
# ---------------------------------------------------------------------------


def test_acceptance_cloud_hybrid_routing_disclosure_exclusion(data_root):
    """Routing, disclosure, per-source exclusion, and outage degrade to local."""
    runtime = SyntheticCloudAnalysisRuntime()
    policy = ExclusionPolicy(
        per_source={"vacation": frozenset({COMPONENT_FACES, COMPONENT_GPS})}
    )
    local = LocalAnalysisProvider()
    cloud = CloudAnalysisProvider(runtime=runtime, policy=policy)
    router = HybridRouter(local, cloud)

    # Route mapping is deterministic: derivatives stay local, semantic stages cloud.
    assert router.route("duplicate_detection") is local
    assert router.route("taste") is cloud
    assert router.route("technical") is local
    assert router.route("pairing") is cloud

    # Disclosure lists exactly what leaves — and what NEVER leaves.
    gps_img = _jpg_with_gps()
    result = router.run("taste", gps_img, profile=BALANCED, asset_id="sunset",
                        source_id="vacation", allowed_components={"faces"})
    assert result is not None
    d = cloud.disclosure()
    assert d.provider == "synthetic-cloud"
    assert "downscaled_derivative" in d.leaves_machine.payload_types
    assert {"asset_id", "source_id", "profile"} <= set(d.leaves_machine.metadata_scope)
    for never in ("original_image", "secrets", "credentials", "tv_ha_credentials",
                  "gps", "faces"):
        assert never in d.leaves_machine.never
        assert never not in d.leaves_machine.payload_types

    # Per-source exclusion stripped the disallowed components from the payload.
    assert len(runtime.received_payloads) == 1
    derivative, received_meta = runtime.received_payloads[0]
    assert derivative != gps_img  # only a downscaled derivative leaves
    assert "gps" not in received_meta
    assert "faces" not in received_meta
    assert "source_resolution" not in received_meta

    # No secret / credential ever appears in what the runtime received.
    all_meta = repr([m for _, m in runtime.received_payloads])
    assert "secret" not in all_meta.lower()
    assert "credential" not in all_meta.lower()
    assert "credentials" not in received_meta and "secrets" not in received_meta

    # Cloud outage: the single cloud call degrades to local, records a Pause.
    runtime.set_down(True)
    degraded = router.run("taste", _make_jpeg(), profile=BALANCED, asset_id="sunset",
                          source_id="vacation")
    assert degraded is not None  # no exception to the caller
    assert router.pause_count == 1
    assert router.pauses[0].kind == "taste"
    assert router.pauses[0].degraded_to == "local"

    # Local/approved work is unaffected by the outage: no new pause, still local.
    before = router.pause_count
    ok = router.run("duplicate_detection", _make_jpeg(), profile=BALANCED,
                    asset_id="dupe")
    assert ok is not None
    assert router.pause_count == before


# ---------------------------------------------------------------------------
# S2 — durable orchestrator: crash-resume, classified outcomes, no dup art
# ---------------------------------------------------------------------------


def test_acceptance_orchestrator_crash_resume_classified(data_root):
    """A crash mid-ingest is classified; resume completes without dup art/regress."""
    catalog = Catalog(data_root=data_root)
    orch = JobOrchestrator(catalog)
    art = b"ingest-render-artifact-bytes"
    sha = sha256_hex(art)
    crash_state = {"p2": 0}

    def p1(job, o):
        o.protect_art(sha, art)
        o.set_result("p1", "art-stored", verified=True)
        return "p2"

    def p2(job, o):
        o.protect_art(sha, art)
        o.set_result("p2", "analysis-ok", verified=True)
        crash_state["p2"] += 1
        if crash_state["p2"] == 1:
            raise RuntimeError("simulated crash mid-ingest")
        return None

    phase_map = {JobKind.INGEST: {"p1": p1, "p2": p2}}

    def transient_hook(job, exc):
        return JobFailure(
            outcome=JobOutcome.TRANSIENT,
            reason=str(exc),
            recovery_action="retry",
            user_explanation="A transient failure occurred; resuming.",
        )

    orch.enqueue(JobKind.INGEST, {"source": "wall-1"})

    # p1 checkpoints at p2; p2 crashes on its first run -> classified TRANSIENT.
    orch.process_next(phase_map)
    failed = orch.process_next(phase_map, fail_hook=transient_hook)
    assert failed.state == "error"
    assert failed.phase == "p2"
    failure = orch.get_failure(failed.id)
    assert failure is not None
    assert failure.outcome is JobOutcome.TRANSIENT
    assert "simulated crash" in failure.reason
    assert failure.recovery_action == "retry"

    # Fresh orchestrator resumes from the checkpoint; no art was duplicated.
    orch2 = JobOrchestrator(catalog)
    resumed = orch2.resume_after_restart()
    assert len(resumed) == 1
    assert resumed[0].state == "queued" and resumed[0].phase == "p2"
    assert resumed[0].checkpoint["known_good"]["p1"] == "art-stored"
    assert resumed[0].checkpoint["known_good"]["p2"] == "analysis-ok"
    assert _blob_count(data_root) == 1

    done = orch2.process_next(phase_map, fail_hook=transient_hook)
    assert done.state == "completed"
    # Last-known-good preserved — re-running p2 never regressed the later value.
    assert done.checkpoint["known_good"]["p2"] == "analysis-ok"
    assert done.checkpoint["known_good"]["p1"] == "art-stored"
    assert catalog.content.exists(sha)
    assert _blob_count(data_root) == 1  # single content-addressed blob

    # PERMANENT classification is distinct from TRANSIENT.
    def permanent_hook(job, exc):
        return JobFailure(outcome=JobOutcome.PERMANENT, reason=str(exc),
                          recovery_action="skip", user_explanation="Permanent.")

    perm = orch2.enqueue(JobKind.PUBLISH, {"case": "perm"})
    pred_hook = permanent_hook
    orch2.process_next({JobKind.PUBLISH: {"p": lambda j, o: (_ for _ in ()).throw(
        RuntimeError("corrupt source"))}}, fail_hook=pred_hook)
    pf = orch2.get_failure(perm.id)
    assert pf is not None and pf.outcome is JobOutcome.PERMANENT
    assert pf.recovery_action == "skip"

    # USER_CANCELLED maps to the `cancelled` state.
    cancelled_job = orch2.enqueue(JobKind.ANALYZE, {"case": "cancel"})

    def cancel_hook(job, exc):
        return JobFailure(outcome=JobOutcome.USER_CANCELLED, reason="operator stop",
                          recovery_action="none", user_explanation="Stopped.")

    orch2.process_next({JobKind.ANALYZE: {"p": lambda j, o: (_ for _ in ()).throw(
        RuntimeError("stop"))}}, fail_hook=cancel_hook)
    row = catalog.db.execute(
        "SELECT state, outcome FROM jobs WHERE id = ?", (cancelled_job.id,)
    ).fetchone()
    assert row is not None and row[0] == "cancelled" and row[1] == "user_cancelled"


# ---------------------------------------------------------------------------
# S3 — non-destructive migration: backup + idempotent import
# ---------------------------------------------------------------------------


def _panel_image() -> Image.Image:
    """A smooth band-limited 1920x1080 Samsung Frame panel image."""
    img = Image.new("RGB", (1920, 1080), (100, 100, 100))
    ImageDraw.Draw(img).ellipse([300, 220, 1620, 860], fill=(150, 150, 150))
    return img.filter(ImageFilter.GaussianBlur(40))


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _build_legacy_folder(root: Path) -> Path:
    """Create a deterministic synthetic legacy folder; return its path."""
    src = root / "legacy-ssd"
    src.mkdir(parents=True, exist_ok=True)
    _panel_image().save(str(src / "panel_01.jpg"), format="JPEG")
    _write_json(src / "art_manifest.json", {"panel": "1920x1080", "samsung": True})
    _write_json(src / "render_mapping.json", {"source": "IMG.jpg", "output": "o.jpg"})
    _write_json(src / "device.json", {"device_id": "frame-123", "serial": "ABC"})
    _write_json(src / "rotation_playlist.json", {"rotation": "interval",
                                                 "playlist": ["a", "b"]})
    return src


def test_acceptance_migration_non_destructive_backed_up(data_root, tmp_path):
    """Dry run reports + imports nothing; real run backs up and imports idempotently."""
    src = _build_legacy_folder(tmp_path)
    catalog = Catalog(data_root=data_root)
    service = MigrationService(catalog=catalog, data_root=data_root)

    expected_counts = {
        "panels": 1,
        "manifests": 1,
        "relationships": 1,
        "devices": 1,
        "rotation": 1,
    }

    # Dry run: reports discovered counts, imports nothing, writes no backup.
    dry = service.migrate(src, dry_run=True)
    assert dry.dry_run is True
    assert dry.discovered == expected_counts
    assert dry.imported == 0
    assert dry.backup_created is False
    assert catalog.count_catalog_entries() == 0
    assert list(data_root.glob("*.backup")) == []

    # Real run: backs up first, then imports every discovered item.
    before_src = _snapshot(src)
    live = service.migrate(src, dry_run=False)
    assert live.backup_created is True
    assert live.backup_path is not None
    backups = list(data_root.glob("*.backup"))
    assert len(backups) == 1 and Path(live.backup_path) == backups[0]
    assert live.imported == 5
    assert catalog.count_catalog_entries() == 5

    # Re-running imports nothing new — no duplicate rows.
    again = service.import_migration(build_plan(src))
    assert again.imported == 0
    assert again.skipped == 5
    assert catalog.count_catalog_entries() == 5

    # Sources are untouched by any migration.
    assert _snapshot(src) == before_src

    # Rollback limitations are documented and surfaced in the report.
    limitations = service.rollback_limitations()
    assert limitations and limitations == live.rollback_limitations
    assert all(isinstance(s, str) and s for s in limitations)


# ---------------------------------------------------------------------------
# S4 — packaging artifacts + headless preflight
# ---------------------------------------------------------------------------


def test_acceptance_packaging_artifacts_and_headless(data_root, monkeypatch, capsys):
    """Plist/Dockerfile artifacts are valid; headless start --check gates correctly."""
    # launchd plist runs `--headless start`.
    plist = plistlib.loads(PLIST.read_bytes())
    args = plist["ProgramArguments"]
    assert plist["Label"] == "com.rohan.curator"
    assert plist["KeepAlive"] is True
    assert any("curator" in os.path.basename(arg) for arg in args)
    assert "--headless" in args and "start" in args
    assert args.index("--headless") < args.index("start")

    # Both Dockerfiles carry the documented directives.
    for path in (DOCKERFILE, DOCKERFILE_CUDA):
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert lines
        joined = "\n".join(lines)
        assert any(ln.startswith("FROM") for ln in lines)
        assert "COPY" in joined
        assert "ENTRYPOINT" in joined
        assert any("--headless" in ln and "start" in ln for ln in lines)

    from curator import cli

    # Missing required secret -> exit 2 with a clear error.
    _clear_env_secrets(monkeypatch)
    assert cli.main(["--headless", "start", "--check"]) == 2
    err = capsys.readouterr().err
    assert "secret" in err and ("API_KEY" in err or "TOKEN" in err)

    # A required env secret set -> exit 0 with JSON status.
    monkeypatch.setenv("SAMSUNG_API_TOKEN", "test-token")
    assert cli.main(["--headless", "start", "--check"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["status"] == "ok" and doc["ready"] is True
    assert doc["data_root"] == str(data_root)

    # CUDA requested but unavailable -> status reports CPU fallback + clear message.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # still no nvidia-smi
    assert cli.main(["--headless", "start", "--check", "--accelerator", "cuda"]) == 0
    captured = capsys.readouterr()
    cuda_doc = json.loads(captured.out)
    assert cuda_doc["ready"] is True
    assert cuda_doc["accelerator"] == "cpu (fallback)"
    assert "degrading to CPU" in captured.err
