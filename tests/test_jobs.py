"""Tests for the durable job orchestrator (M006/S02).

Proves the ``jobs`` table (schema v12) and the
:class:`~curator.jobs.orchestrator.JobOrchestrator` contract:

- enqueue is idempotent by content-derived key (same kind + payload -> one row);
- :class:`Job` / :class:`JobFailure` JSON round-trip;
- a multi-phase job advances its phases and reaches ``completed``;
- a crash mid-phase is classified and recorded, then a fresh orchestrator resumes
  from the checkpoint without duplicating content-addressed art or regressing a
  previously-successful later known-good value;
- all six :class:`JobOutcome` values are recorded with reason + recovery_action;
- identical inputs produce identical job / state transitions (deterministic).

Tests use the ``data_root`` fixture with a real Catalog + ContentStore so durability
and content-addressed dedup are exercised against the actual store.
"""

from __future__ import annotations

import pytest

from curator import db, schema
from curator.catalog import Catalog
from curator.hashing import sha256_hex
from curator.jobs import Job, JobFailure, JobKind, JobOrchestrator, JobOutcome


@pytest.fixture
def catalog(data_root):
    """A Catalog backed by a fresh migrated DB + ContentStore under the data root."""
    return Catalog(data_root=data_root)


def _blob_count(data_root) -> int:
    """Count content-addressed blob files under the data root (excludes temp)."""
    content_root = data_root / "content"
    if not content_root.exists():
        return 0
    return len(list(content_root.glob("*/*/*")))


# -- enqueue idempotency ---------------------------------------------------------


def test_enqueue_idempotent_by_content_key(catalog):
    orch = JobOrchestrator(catalog)
    payload = {"source": "frame-1", "width": 1920}

    first = orch.enqueue(JobKind.RENDER, payload)
    second = orch.enqueue(JobKind.RENDER, payload)

    assert first.id == second.id
    assert first.state == "queued"
    rows = catalog.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    assert rows == 1


def test_enqueue_distinct_payloads_create_distinct_jobs(catalog):
    orch = JobOrchestrator(catalog)
    a = orch.enqueue(JobKind.RENDER, {"source": "a"})
    b = orch.enqueue(JobKind.RENDER, {"source": "b"})
    assert a.id != b.id
    assert catalog.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


# -- JSON round-trip --------------------------------------------------------------


def test_job_to_dict_from_dict_roundtrip():
    job = Job(
        id=7,
        key="render:{'source':'x'}",
        kind=JobKind.ANALYZE,
        payload={"source": "x"},
        state="checkpointed",
        phase="p2",
        checkpoint={"results": {"p1": 1}, "known_good": {"p1": 1}},
        attempts=2,
        created_at="2020-01-01T00:00:00Z",
        updated_at="2020-01-01T00:00:01Z",
    )
    restored = Job.from_dict(job.to_dict())
    assert restored == job
    assert restored.kind is JobKind.ANALYZE


def test_job_failure_to_dict_from_dict_roundtrip():
    failure = JobFailure(
        outcome=JobOutcome.POLICY_BLOCKED,
        reason="aspect ratio not allowed",
        recovery_action="choose a 16:9 image",
        user_explanation="This image is not a supported shape.",
    )
    restored = JobFailure.from_dict(failure.to_dict())
    assert restored == failure
    assert restored.outcome is JobOutcome.POLICY_BLOCKED


# -- schema ----------------------------------------------------------------------


def test_v12_jobs_table_exists(data_root):
    conn = db.connect()
    try:
        db.migrate(conn)
        assert "jobs" in db.table_names(conn)
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        assert version >= 12
    finally:
        conn.close()


def test_jobs_expected_in_boundary_tables(data_root):
    assert "jobs" in schema.EXPECTED_TABLES


# -- phase processing ------------------------------------------------------------


def _advance_map(steps: dict[str, object]):
    """Build a PhaseMap for ROTATE kind from an ordered {phase: value} dict.

    Each phase records its result and advances to the next key, with the final phase
    returning None (completes the job).
    """
    order = list(steps)

    def make(phase):
        def phase_fn(job, orch):
            orch.set_result(phase, steps[phase])
            idx = order.index(phase)
            return order[idx + 1] if idx + 1 < len(order) else None

        return phase_fn

    return {JobKind.ROTATE: {phase: make(phase) for phase in order}}


def test_phase_processing_advances_and_completes(catalog):
    orch = JobOrchestrator(catalog)
    phase_map = _advance_map({"p1": 10, "p2": 20, "p3": 30})
    job = orch.enqueue(JobKind.ROTATE, {"seq": 1})

    j1 = orch.process_next(phase_map)
    assert j1.state == "checkpointed"
    assert j1.phase == "p2"
    assert j1.checkpoint["known_good"]["p1"] == 10

    j2 = orch.process_next(phase_map)
    assert j2.state == "checkpointed"
    assert j2.phase == "p3"

    j3 = orch.process_next(phase_map)
    assert j3.state == "completed"
    assert j3.checkpoint["known_good"]["p2"] == 20
    assert j3.checkpoint["known_good"]["p3"] == 30

    # Nothing left to run.
    assert orch.process_next(phase_map) is None
    assert catalog.db.execute(
        "SELECT state FROM jobs WHERE id = ?", (job.id,)
    ).fetchone()[0] == "completed"


# -- crash mid-phase + resume ----------------------------------------------------


def test_crash_midphase_resume_no_duplicate_art_no_regress(catalog, data_root):
    orch = JobOrchestrator(catalog)
    art = b"the-render-artifact-bytes"
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
            raise RuntimeError("simulated crash at phase p2")
        return None

    phase_map = {JobKind.RENDER: {"p1": p1, "p2": p2}}

    def fail_hook(job, exc):
        return JobFailure(
            outcome=JobOutcome.TRANSIENT,
            reason=str(exc),
            recovery_action="retry",
            user_explanation="A transient failure occurred; resuming.",
        )

    orch.enqueue(JobKind.RENDER, {"artifact": "wall-1"})

    # First orchestrator: run p1 (checkpoint at p2) then p2 (crashes -> classified).
    orch.process_next(phase_map)
    failed = orch.process_next(phase_map, fail_hook=fail_hook)
    assert failed.state == "error"
    assert failed.phase == "p2"
    failure = orch.get_failure(failed.id)
    assert failure is not None
    assert failure.outcome is JobOutcome.TRANSIENT
    assert "simulated crash" in failure.reason
    assert failure.recovery_action == "retry"

    # A fresh orchestrator on the same catalog resumes from the checkpoint.
    orch2 = JobOrchestrator(catalog)
    resumed = orch2.resume_after_restart()
    assert len(resumed) == 1
    assert resumed[0].state == "queued"
    assert resumed[0].phase == "p2"
    # Checkpoint survived the crash with both phase results.
    assert resumed[0].checkpoint["known_good"]["p1"] == "art-stored"
    assert resumed[0].checkpoint["known_good"]["p2"] == "analysis-ok"

    done = orch2.process_next(phase_map, fail_hook=fail_hook)
    assert done.state == "completed"
    # Last-known-good preserved (not regressed by re-running p2).
    assert done.checkpoint["known_good"]["p2"] == "analysis-ok"
    assert done.checkpoint["known_good"]["p1"] == "art-stored"

    # No duplicate art: protect_art was called in p1 and twice in p2 (crash + resume),
    # but content-addressed put-if-missing stored exactly one blob.
    assert _blob_count(data_root) == 1
    assert catalog.content.exists(sha)


# -- six-way outcome classification ----------------------------------------------


def test_all_six_outcomes_recorded(catalog):
    orch = JobOrchestrator(catalog)

    cases = [
        (JobKind.INGEST, JobOutcome.TRANSIENT, "network blip"),
        (JobKind.ANALYZE, JobOutcome.PERMANENT, "corrupt file"),
        (JobKind.RENDER, JobOutcome.POLICY_BLOCKED, "aspect ratio"),
        (JobKind.PUBLISH, JobOutcome.CAPABILITY_UNSUPPORTED, "no hdr"),
        (JobKind.ROTATE, JobOutcome.UNRESOLVED_EXTERNAL, "api down"),
        (JobKind.INGEST, JobOutcome.USER_CANCELLED, "operator stop"),
    ]

    def make_phases(case_outcome, tag):
        def boom(job, o):
            raise RuntimeError(f"{tag}:boom")

        def fail_hook(job, exc):
            return JobFailure(
                outcome=case_outcome,
                reason=str(exc),
                recovery_action=f"recover-{tag}",
                user_explanation=f"explained-{tag}",
            )

        return {job.kind: {"run": boom}}, boom, fail_hook

    for idx, (kind, outcome, tag) in enumerate(cases):
        job = orch.enqueue(kind, {"case": idx})
        phase_map, boom, fail_hook = make_phases(outcome, tag)
        failed = orch.process_next(phase_map, fail_hook=fail_hook)
        assert failed.state == ("cancelled" if outcome is JobOutcome.USER_CANCELLED else "error")
        failure = orch.get_failure(failed.id)
        assert failure is not None
        assert failure.outcome is outcome
        assert f"{tag}:boom" == failure.reason
        assert failure.recovery_action == f"recover-{tag}"
        assert failure.user_explanation == f"explained-{tag}"


# -- determinism -----------------------------------------------------------------


def test_determinism_same_inputs_same_transitions(tmp_path):
    """Same inputs across two independent roots yield identical job/state transitions."""

    def run(root):
        cat = Catalog(data_root=root)
        orch = JobOrchestrator(cat)
        phase_map = _advance_map({"p1": "a", "p2": "b"})
        orch.enqueue(JobKind.ROTATE, {"seq": 42})
        j1 = orch.process_next(phase_map)
        j2 = orch.process_next(phase_map)
        return [j1.to_dict(), j2.to_dict()]

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    out_a = run(root_a)
    out_b = run(root_b)

    assert [j["state"] for j in out_a] == [j["state"] for j in out_b]
    assert [j["phase"] for j in out_a] == [j["phase"] for j in out_b]
    assert [j["key"] for j in out_a] == [j["key"] for j in out_b]
    assert out_b[0]["checkpoint"] == out_a[0]["checkpoint"]
