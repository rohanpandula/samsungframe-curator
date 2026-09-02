"""One Wall — the browser flow's backend glue (M011/S01).

The web page is one flow, **Load → Score and pick → Hang**, and this module is
everything that flow needs that the engines did not already expose to a browser:

- :func:`cached_thumbnail` — a JPEG thumbnail per content sha, cached on disk, so
  the page shows pictures instead of hashes (HEIC included: importing
  :mod:`curator.ingest.decode` registers the HEIF opener).
- :func:`wall_state` — every entry with its decision, score and hang status, plus
  the four counts the page shows on its stages, in **one** request.
- :class:`JobRunner` — the smallest possible in-process background runner, so
  scoring a library or publishing a wall never blocks the page (Doherty); a
  ``wait=True`` start runs inline for tests and scripts.
- :func:`score_unscored`, :func:`load_folder`, :func:`publish_approved` — the
  three jobs. Publishing is the milestone's real gap-closer: ``dest/`` was
  complete and tested with **zero** callers until this module, and nothing in
  the product ever wrote a rendered file where a person could reach it.

Every job opens its own :class:`~curator.catalog.Catalog` (own SQLite
connection; WAL serializes writers), so the request thread never shares a
cursor with a worker. Nothing here reads a taste signal; the treatment each
approved photo hangs with is whatever :func:`~curator.artdirection.policy.propose_treatments`
ranks first for it — the same answer ``curator propose`` gives.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

import curator.ingest.decode  # noqa: F401 — registers the HEIF opener for thumbnails
from curator.analysis.local import LocalAnalysisProvider
from curator.analysis.pipeline import AnalysisAsset, AnalysisPipeline
from curator.analysis.profiles import AnalysisProfile
from curator.analysis.schema import AnalysisResult
from curator.approve.approval import ApprovalService, Decision
from curator.artdirection.manifest import ArtDirectionManifest, ManifestError
from curator.artdirection.packing import resolve_regions
from curator.artdirection.policy import (
    ArtDirectionRequest,
    materialize_manifest,
    propose_treatments,
)
from curator.catalog import Catalog
from curator.connectors.local import LocalConnector
from curator.dest.base import STATUS_APPLIED, DestinationAdapter, DestinationError
from curator.dest.filesystem import FilesystemDestinationAdapter
from curator.dest.publish import publish
from curator.dest.simulator import SimulatorDestinationAdapter
from curator.errors import CuratorError
from curator.hashing import sha256_hex
from curator.ingest.pipeline import IngestPipeline
from curator.render.renderer import DeterministicRenderer, RenderError, RenderResult
from curator.render.validate import ArtifactValidator

#: The two Frame panel sizes, by the names the CLI, the API and the page all use.
OUTPUT_TARGETS: dict[str, tuple[int, int]] = {"1080p": (1920, 1080), "4k": (3840, 2160)}

#: Thumbnail long-side bounds; anything outside is clamped, never rejected.
THUMB_MIN_SIDE = 64
THUMB_MAX_SIDE = 2048
THUMB_DEFAULT_SIDE = 320

#: The folder destination's default root under the data root.
WALL_DIRNAME = "wall"

Progress = Callable[[int, int, str], None]


# -- thumbnails -----------------------------------------------------------------


def thumbnail_bytes(data: bytes, max_side: int) -> bytes:
    """Return a JPEG thumbnail of *data* with its long side at most *max_side*.

    Orientation-corrected like the analysis engine's decode, RGB, quality 82.
    Undecodable bytes raise :class:`CuratorError` with the decoder's reason.
    """
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except Exception as exc:  # PIL raises many types for malformed input
        raise CuratorError(f"cannot decode image for thumbnail: {exc}") from exc
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    return buffer.getvalue()


def clamp_thumb_side(requested: int | None) -> int:
    """Clamp a requested long side into ``THUMB_MIN_SIDE..THUMB_MAX_SIDE``."""
    if requested is None:
        return THUMB_DEFAULT_SIDE
    return max(THUMB_MIN_SIDE, min(THUMB_MAX_SIDE, int(requested)))


def cached_thumbnail(data_root: Path, store: Any, sha: str, max_side: int) -> bytes:
    """Return the cached thumbnail for *sha* at *max_side*, rendering it on a miss.

    Cache path: ``<data_root>/thumbs/<sha>-<max_side>.jpg``, written via a
    per-writer temp file and ``os.replace`` so two concurrent misses cannot
    leave a torn file. A sha the store does not hold raises
    :class:`~curator.errors.StorageError` unchanged (the API maps it to 404).
    """
    path = data_root / "thumbs" / f"{sha}-{max_side}.jpg"
    if path.is_file():
        return path.read_bytes()
    data = thumbnail_bytes(store.get(sha), max_side)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return data


# -- jobs -----------------------------------------------------------------------


@dataclass
class JobStatus:
    """Progress and outcome of one named background job (JSON via ``to_dict``)."""

    name: str
    state: str = "idle"  # idle | running | done | error
    done: int = 0
    total: int = 0
    current: str = ""
    message: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    result: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "done": self.done,
            "total": self.total,
            "current": self.current,
            "message": self.message,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": dict(self.result),
        }


class JobRunner:
    """One background job per name; starting a running name just reports it.

    ponytail: a dict and a daemon thread per job, in-process, forgotten on
    restart. Upgrade to :mod:`curator.jobs` (durable, resumable) if a wall ever
    has to survive a server restart mid-publish.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobStatus] = {}

    def status(self, name: str) -> JobStatus:
        with self._lock:
            return self._jobs.get(name) or JobStatus(name=name)

    def start(
        self, name: str, work: Callable[[Progress], dict[str, Any]], *, wait: bool = False
    ) -> JobStatus:
        with self._lock:
            existing = self._jobs.get(name)
            if existing is not None and existing.state == "running":
                return existing
            status = JobStatus(name=name, state="running", started_at=time.time())
            self._jobs[name] = status

        def progress(done: int, total: int, current: str = "") -> None:
            status.done, status.total, status.current = done, total, current

        def run() -> None:
            try:
                status.result = work(progress)
                status.state = "done"
            except Exception as exc:  # a job's failure is reported, never raised
                status.state = "error"
                status.message = f"{type(exc).__name__}: {exc}"
            finally:
                status.finished_at = time.time()

        if wait:
            run()
        else:
            threading.Thread(target=run, name=f"wall-{name}", daemon=True).start()
        return status


# -- state ----------------------------------------------------------------------


def latest_scores(catalog: Catalog) -> dict[int, dict[str, Any]]:
    """Return ``{entry_id: {"aesthetic", "technical", "engine"}}`` from the newest ok analysis."""
    rows = catalog.db.execute(
        "SELECT r.catalog_entry_id, r.engine_version, r.analysis_json FROM analysis_results r"
        " JOIN (SELECT catalog_entry_id, MAX(id) AS id FROM analysis_results"
        "       WHERE status = 'ok' GROUP BY catalog_entry_id) m ON m.id = r.id"
    ).fetchall()
    scores: dict[int, dict[str, Any]] = {}
    for entry_id, engine, analysis_json in rows:
        quality = json.loads(analysis_json).get("quality", {})
        scores[int(entry_id)] = {
            "aesthetic": quality.get("aesthetic_quality"),
            "technical": quality.get("technical_quality"),
            "engine": engine,
        }
    return scores


def hung_artifacts(catalog: Catalog) -> dict[int, list[dict[str, Any]]]:
    """Return ``{entry_id: [{"artifact_id", "adapter_id", "target"}]}`` for applied publishes."""
    rows = catalog.db.execute(
        "SELECT rd.catalog_entry_id, j.artifact_id, j.adapter_id, rd.target"
        " FROM dest_journal j JOIN renders rd ON rd.artifact_sha = j.sha"
        " WHERE j.status = ? AND rd.catalog_entry_id IS NOT NULL"
        " ORDER BY j.id",
        (STATUS_APPLIED,),
    ).fetchall()
    hung: dict[int, list[dict[str, Any]]] = {}
    for entry_id, artifact_id, adapter_id, target in rows:
        hung.setdefault(int(entry_id), []).append(
            {"artifact_id": artifact_id, "adapter_id": adapter_id, "target": target}
        )
    return hung


def wall_state(catalog: Catalog) -> dict[str, Any]:
    """Everything the page needs, best-first: entries, counts, folders, targets."""
    approval = ApprovalService(catalog)
    scores = latest_scores(catalog)
    hung = hung_artifacts(catalog)
    entries: list[dict[str, Any]] = []
    for row in catalog.list_entries():
        entry_id = int(row["id"])
        current = approval.current(entry_id)
        decision = current.decision.value.lower() if current else "pending"
        score = scores.get(entry_id)
        asset = str(row["asset_id"])
        entries.append(
            {
                "entry_id": entry_id,
                "asset_id": asset,
                "name": Path(asset).name,
                "folder": str(Path(asset).parent),
                "sha256": row["sha256"],
                "decision": decision,
                "scored": score is not None,
                "score": None if score is None else score["aesthetic"],
                "technical": None if score is None else score["technical"],
                "hung": hung.get(entry_id, []),
                "thumb": f"/api/thumb/{row['sha256']}",
            }
        )
    # Best first; unscored photos sink to the bottom in a stable, name order.
    entries.sort(key=lambda e: (-(e["score"] if e["score"] is not None else -1.0), e["name"]))
    counts = {
        "loaded": len(entries),
        "scored": sum(1 for e in entries if e["scored"]),
        "approved": sum(1 for e in entries if e["decision"] == "approved"),
        "rejected": sum(1 for e in entries if e["decision"] == "rejected"),
        "hung": sum(1 for e in entries if e["hung"]),
    }
    return {
        "entries": entries,
        "counts": counts,
        "folders": sorted({e["folder"] for e in entries}),
        "targets": list(OUTPUT_TARGETS),
    }


# -- destinations ---------------------------------------------------------------


def destinations(data_root: Path) -> list[dict[str, Any]]:
    """The destinations the page can offer, each with an honest ``available`` flag."""
    return [
        {
            "id": "folder",
            "label": "A folder (copy to a USB stick for the Frame)",
            "available": True,
            "location": str(data_root / WALL_DIRNAME),
        },
        {
            "id": "simulator",
            "label": "Simulator (a test target, in memory)",
            "available": True,
            "location": "in-memory",
        },
        {
            "id": "samsung",
            "label": "Samsung Frame over the network",
            "available": False,
            "reason": (
                "no network transport is implemented yet — only the simulator "
                "transport exists; export to a folder and use a USB stick"
            ),
        },
    ]


def make_destination(
    kind: str,
    data_root: Path,
    folder: str | None = None,
    simulator: SimulatorDestinationAdapter | None = None,
) -> tuple[DestinationAdapter, str, str]:
    """Return ``(adapter, adapter_id, location)`` for *kind*; unknown/unavailable raise."""
    if kind == "folder":
        root = Path(folder).expanduser() if folder else data_root / WALL_DIRNAME
        return FilesystemDestinationAdapter(root), f"folder:{root}", str(root)
    if kind == "simulator":
        return simulator or SimulatorDestinationAdapter(), "simulator", "in-memory"
    if kind == "samsung":
        raise CuratorError(
            "Samsung Frame over the network is not available: no network transport "
            "is implemented yet — export to a folder and use a USB stick"
        )
    raise CuratorError(f"unknown destination {kind!r} (known: folder, simulator)")


# -- the jobs -------------------------------------------------------------------


def _unscored(catalog: Catalog) -> list[tuple[int, str, str]]:
    return [
        (int(entry_id), str(sha), str(asset))
        for entry_id, sha, asset in catalog.db.execute(
            "SELECT id, sha256, asset_id FROM catalog_entries WHERE id NOT IN"
            " (SELECT catalog_entry_id FROM analysis_results WHERE status = 'ok')"
            " ORDER BY id"
        ).fetchall()
    ]


def score_unscored(data_root: Path, progress: Progress) -> dict[str, Any]:
    """Score every entry without an ok analysis, one persisted row each.

    Reads bytes from the content store (the source folder may be unmounted),
    runs the pipeline per asset so progress is per photo and a corrupt file is
    recorded rather than fatal, and reports counts.
    """
    catalog = Catalog(data_root=data_root)
    try:
        pending = _unscored(catalog)
        pipeline = AnalysisPipeline(catalog, provider=LocalAnalysisProvider())
        scored = failed = 0
        progress(0, len(pending), "")
        for index, (entry_id, sha, asset) in enumerate(pending, start=1):
            data = catalog.content.get(sha)
            report = pipeline.run([AnalysisAsset(entry_id=entry_id, source=data)])
            scored += report.analyzed_count
            failed += report.corrupt_count + report.error_count
            progress(index, len(pending), Path(asset).name)
        return {"scored": scored, "failed": failed, "total": len(pending)}
    finally:
        catalog.db.close()


def load_folder(data_root: Path, path: str, progress: Progress) -> dict[str, Any]:
    """Ingest *path* (a local folder) through the ingest pipeline; report its counts."""
    folder = Path(path.strip()).expanduser()
    if not folder.is_dir():
        raise CuratorError(f"not a folder: {path!r}")
    catalog = Catalog(data_root=data_root)
    try:
        progress(0, 0, str(folder))
        report = IngestPipeline(LocalConnector(folder), catalog=catalog).run()
        summary = report.to_dict()
        summary["folder"] = str(folder)
        return summary
    finally:
        catalog.db.close()


def latest_analysis(catalog: Catalog, entry_id: int) -> AnalysisResult | None:
    """Return the newest ok :class:`AnalysisResult` for *entry_id*, if any."""
    row = catalog.db.execute(
        "SELECT analysis_json FROM analysis_results"
        " WHERE catalog_entry_id = ? AND status = 'ok' ORDER BY id DESC LIMIT 1",
        (entry_id,),
    ).fetchone()
    return None if row is None else AnalysisResult.from_dict(json.loads(row[0]))


def record_render(
    catalog: Catalog,
    manifest: ArtDirectionManifest,
    target_label: str,
    result: RenderResult,
    artifact_sha: str,
) -> None:
    """Append one ``renders`` row (shared by ``POST /api/render`` and the publish job)."""
    entry_id: int | None = None
    if manifest.sources:
        for entry in catalog.get_by_hash(manifest.sources[0]):
            entry_id = int(entry["id"])
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


def render_entry(
    catalog: Catalog,
    entry_id: int,
    sha: str,
    output: str,
    provider: LocalAnalysisProvider,
    renderer: DeterministicRenderer,
) -> tuple[bytes, ArtDirectionManifest, RenderResult]:
    """Render one cataloged photo at *output* with the policy engine's top treatment.

    Reuses the persisted analysis when there is one (deterministic, and what the
    score stage wrote), else analyzes the bytes once. Raises
    :class:`~curator.render.renderer.RenderError` when the render would need an
    unapproved upscale (R008) — the caller reports that photo, never hides it.
    """
    width, height = OUTPUT_TARGETS[output]
    data = catalog.content.get(sha)
    analysis = latest_analysis(catalog, entry_id)
    if analysis is None:
        analysis = provider.analyze(data, AnalysisProfile.BALANCED, asset_id=sha)
    request = ArtDirectionRequest(
        target=output, target_width=width, target_height=height, sources=[sha]
    )
    proposals = propose_treatments([analysis], request, provider=provider)
    if not proposals:
        raise CuratorError("the policy engine proposed no treatment for this photo")
    manifest = materialize_manifest(proposals[0], request, [sha])
    manifest.validate()
    sources = {sha: data}
    result = renderer.render(manifest, sources, (width, height))
    payload = renderer.render_bytes(manifest, sources, (width, height))
    return payload, manifest, result


def publish_approved(
    data_root: Path,
    destination: str,
    output: str,
    progress: Progress,
    *,
    folder: str | None = None,
    simulator: SimulatorDestinationAdapter | None = None,
) -> dict[str, Any]:
    """Hang every approved photo: render → validate → publish, one report per photo.

    Refuses (per photo, loudly) anything the validator calls unpublishable or
    that would need an unapproved upscale; ``dest_journal`` keeps the
    verify-before-replace record exactly as :class:`~curator.dest.publish.PublishCoordinator`
    writes it, and the rendered bytes land in the content store with a
    ``renders`` row so the page can show what hangs where.
    """
    if output not in OUTPUT_TARGETS:
        raise CuratorError(f"unknown output {output!r} (known: {', '.join(OUTPUT_TARGETS)})")
    catalog = Catalog(data_root=data_root)
    try:
        adapter, adapter_id, location = make_destination(
            destination, data_root, folder=folder, simulator=simulator
        )
        approval = ApprovalService(catalog)
        approved = [
            row
            for row in catalog.list_entries()
            if (current := approval.current(int(row["id"]))) is not None
            and current.decision == Decision.APPROVED
        ]
        provider = LocalAnalysisProvider()
        renderer = DeterministicRenderer()
        validator = ArtifactValidator()
        items: list[dict[str, Any]] = []
        published = skipped = failed = 0
        progress(0, len(approved), "")
        for index, row in enumerate(approved, start=1):
            entry_id, sha, asset = int(row["id"]), str(row["sha256"]), str(row["asset_id"])
            name = Path(asset).name
            item: dict[str, Any] = {"entry_id": entry_id, "name": name}
            try:
                payload, manifest, result = render_entry(
                    catalog, entry_id, sha, output, provider, renderer
                )
                artifact_sha = sha256_hex(payload)
                report = validator.validate(
                    payload,
                    artifact_sha,
                    OUTPUT_TARGETS[output],
                    source_regions=resolve_regions(manifest, OUTPUT_TARGETS[output]),
                )
                if not report.publishable:
                    failed += 1
                    item.update(
                        status="unpublishable",
                        reasons=[c.reason for c in report.checks if not c.passed and c.reason],
                    )
                    items.append(item)
                    continue
                catalog.content.put(payload)
                record_render(catalog, manifest, output, result, artifact_sha)
                artifact_id = f"{entry_id:05d}-{sha[:12]}-{output}.png"
                outcome = publish(
                    adapter,
                    catalog.db,
                    artifact_id,
                    payload,
                    meta={"entry_id": entry_id, "source_sha256": sha, "target": output},
                    adapter_id=adapter_id,
                )
                if outcome.skipped:
                    skipped += 1
                else:
                    published += 1
                item.update(
                    status="skipped" if outcome.skipped else "hung",
                    artifact_id=artifact_id,
                    artifact_sha=artifact_sha,
                    treatment=manifest.layout_treatment.value,
                )
            except (RenderError, ManifestError, DestinationError, CuratorError) as exc:
                failed += 1
                item.update(status="error", error=str(exc))
            items.append(item)
            progress(index, len(approved), name)
        return {
            "destination": destination,
            "adapter_id": adapter_id,
            "location": location,
            "output": output,
            "approved": len(approved),
            "hung": published,
            "skipped": skipped,
            "failed": failed,
            "items": items,
        }
    finally:
        catalog.db.close()
