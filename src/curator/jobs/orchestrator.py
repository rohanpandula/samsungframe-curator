"""Durable job orchestrator — checkpoints, crash-resume, classified outcomes (M006/S02).

:class:`JobOrchestrator` executes units of work from the ``jobs`` table (schema v12)
as a checkpointed state machine over injectable per-kind phase functions, so tests
and callers control exactly which phases run and how failures are classified.

Guarantees:

- **Idempotent enqueue** — :meth:`enqueue` derives a content ``key`` from kind +
  canonical payload and inserts a single ``queued`` row; re-enqueuing the same work
  returns the existing job while it is ``queued``/``active``/``completed``.
- **Checkpointed phases** — :meth:`process_next` runs the job's current phase and,
  on success, persists the phase result to ``checkpoint_json`` plus advances the
  ``phase``, leaving the row ``checkpointed`` so a crash never loses progress.
- **Crash-resume without duplicate art / regressing last-known-good** —
  :meth:`resume_after_restart` rehydrates ``active``/``checkpointed``/``error`` rows
  back to ``queued`` (retaining ``checkpoint_json`` + ``phase``); :meth:`protect_art`
  stores bytes content-addressed (put-if-missing, so a blob is written once), and
  :meth:`set_result` keeps a per-phase ``known_good`` map so a later verified result
  on resume never overwrites a previously-successful later value.
- **Classified outcomes** — a phase exception is classified by an injected
  ``fail_hook`` (or a transient/permanent default) into one of the six
  :class:`~curator.jobs.model.JobOutcome` values recorded via :meth:`fail`.

Everything is synchronous and takes injectable phase maps / fail hooks, keeping
tests deterministic (no sleeps, no wall-clock coupling).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from curator.content_store import ContentStore
from curator.errors import CuratorError
from curator.hashing import sha256_hex
from curator.jobs.model import Job, JobFailure, JobKind, JobOutcome

if TYPE_CHECKING:
    from curator.catalog import Catalog

# Phase function: runs the current phase for a job and returns the next phase name,
# or ``None`` when the whole job is complete.
PhaseFn = Callable[[Job, "JobOrchestrator"], str | None]
# Per-kind phase functions keyed by phase name (``kind -> {phase -> fn}``).
PhaseMap = dict[JobKind, dict[str, PhaseFn]]
# Failure hook: classify a phase exception; return a JobFailure or None to use the
# transient/permanent default.
FailHook = Callable[[Job, Exception], JobFailure | None]

_TIMESTAMP = "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')"

_JOB_COLUMNS = [
    "id",
    "key",
    "kind",
    "payload_json",
    "state",
    "phase",
    "checkpoint_json",
    "attempts",
    "outcome",
    "reason",
    "recovery",
    "user_explanation",
    "created_at",
    "updated_at",
]

# States a fresh orchestrator may resume after a restart.
_RESUMABLE = ("active", "checkpointed", "error")

# States for which enqueue is idempotent (re-enqueue returns the existing job).
_IDEMPOTENT_STATES = ("queued", "active", "completed")


class TransientJobError(CuratorError):
    """Default marker for a retryable (transient) phase failure."""


@dataclass(frozen=True)
class JobReport:
    """JSON-serializable status snapshot of the job table (drain/status surface)."""

    total: int
    by_state: dict[str, int] = field(default_factory=dict)
    jobs: list[Job] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return {
            "total": self.total,
            "by_state": dict(self.by_state),
            "jobs": [job.to_dict() for job in self.jobs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobReport:
        """Reconstruct a :class:`JobReport` from a :meth:`JobReport.to_dict` dict."""
        return cls(
            total=int(data["total"]),
            by_state=dict(data.get("by_state") or {}),
            jobs=[Job.from_dict(j) for j in data.get("jobs") or []],
        )


class JobOrchestrator:
    """Checkpointed, idempotent executor over the ``jobs`` table.

    Takes either a :class:`~curator.catalog.Catalog` (reusing its connection and
    ContentStore) or a raw ``sqlite3.Connection`` (with a ContentStore resolved from
    config).
    """

    def __init__(self, source: Catalog | sqlite3.Connection) -> None:
        if isinstance(source, sqlite3.Connection):
            self.db = source
            self.content = ContentStore()
        else:
            self.db = source.db
            self.content = source.content
        self._current: Job | None = None

    # -- enqueue / idempotency ---------------------------------------------------

    @staticmethod
    def _derive_key(kind: JobKind, payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return f"{kind.value}:{canonical}"

    def enqueue(self, kind: JobKind, payload: dict[str, Any]) -> Job:
        """Enqueue *payload* for *kind*, returning a persistent :class:`Job`.

        Idempotent: if a ``queued``/``active``/``completed`` job with the same
        content ``key`` already exists, returns it instead of inserting a duplicate.
        """
        key = self._derive_key(kind, payload)
        placeholders = ",".join("?" for _ in range(len(_IDEMPOTENT_STATES)))
        existing = self.db.execute(
            "SELECT * FROM jobs WHERE key = ? AND state IN (" + placeholders + ") LIMIT 1",
            (key, *map(str, _IDEMPOTENT_STATES)),
        ).fetchone()
        if existing is not None:
            return self._job_from_row(existing)
        try:
            cur = self.db.execute(
                "INSERT INTO jobs(key, kind, payload_json, state)"
                " VALUES (?, ?, ?, 'queued')",
                (key, kind.value, json.dumps(payload)),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CuratorError(f"failed to enqueue job: {exc}") from exc
        if cur.lastrowid is None:
            raise CuratorError("failed to obtain new job id")
        return self._reload(int(cur.lastrowid))

    # -- processing --------------------------------------------------------------

    def process_next(
        self,
        phase_fns: PhaseMap | None = None,
        fail_hook: FailHook | None = None,
    ) -> Job | None:
        """Run the next runnable job's current phase and return the updated job.

        Pulls the oldest ``queued`` **or** ``checkpointed`` job (a checkpointed job is
        still runnable until it completes), marks it ``active``, runs its current phase
        via the per-kind phase function map (*phase_fns*), then either completes it
        (phase fn returned ``None``) or persists a ```checkpointed``` state + advanced
        phase. A phase exception is classified via *fail_hook* (or a transient/permanent
        default) and recorded with :meth:`fail`. Returns ``None`` when no job is
        runnable.
        """
        phase_fns = phase_fns or {}
        row = self.db.execute(
            "SELECT * FROM jobs WHERE state IN ('queued', 'checkpointed')"
            " ORDER BY id LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        job = self._job_from_row(row)
        self._current = job
        try:
            self._mark_active(job)
            if job.kind not in phase_fns:
                raise KeyError(f"no phase functions registered for kind {job.kind.value}")
            phase_map = phase_fns[job.kind]
            phase = job.phase or next(iter(phase_map))
            job = replace(job, phase=phase)
            self._current = job
            self._update(job.id, phase=phase)
            next_phase = phase_map[phase](job, self)
            if next_phase is None:
                self._complete(job)
            else:
                self._checkpoint(job, next_phase)
            result = self._reload(job.id)
            self._current = None
            return result
        except Exception as exc:  # noqa: BLE001 - any phase failure is classified
            failure = fail_hook(job, exc) if fail_hook else None
            if failure is None:
                outcome = (
                    JobOutcome.TRANSIENT
                    if isinstance(exc, TransientJobError)
                    else JobOutcome.PERMANENT
                )
                failure = JobFailure(
                    outcome=outcome,
                    reason=str(exc),
                    recovery_action="retry",
                )
            job = self._current or job
            self.fail(job, failure)
            self._current = None
            return self._reload(job.id)

    def resume_after_restart(self) -> list[Job]:
        """Rehydrate resumable jobs back to ``queued`` (checkpoint + phase retained).

        Moves every ``active``/``checkpointed``/``error`` row back to ``queued`` so a
        fresh orchestrator on the same DB resumes them exactly where they stopped.
        Returns the rehydrated jobs.
        """
        ids = [
            int(r[0])
            for r in self.db.execute(
                "SELECT id FROM jobs WHERE state IN ("
                + ",".join("?" for _ in range(len(_RESUMABLE)))
                + ")",
                (*map(str, _RESUMABLE),),
            ).fetchall()
        ]
        for job_id in ids:
            self._update(job_id, state="queued")
        self.db.commit()
        return [self._reload(job_id) for job_id in ids]

    # -- outcome + durable-art helpers -------------------------------------------

    def fail(self, job: Job, failure: JobFailure) -> None:
        """Record a classified :class:`JobFailure`; state becomes error/cancelled."""
        state = "cancelled" if failure.outcome is JobOutcome.USER_CANCELLED else "error"
        try:
            self.db.execute(
                "UPDATE jobs SET state = ?, outcome = ?, reason = ?, recovery = ?,"
                f" user_explanation = ?, updated_at = {_TIMESTAMP}"
                " WHERE id = ?",
                (
                    state,
                    failure.outcome.value,
                    failure.reason,
                    failure.recovery_action,
                    failure.user_explanation,
                    job.id,
                ),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CuratorError(f"failed to record job failure for {job.id}: {exc}") from exc

    def get_failure(self, job_id: int) -> JobFailure | None:
        """Return the recorded :class:`JobFailure` for *job_id*, or ``None``."""
        row = self.db.execute(
            "SELECT outcome, reason, recovery, user_explanation FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return JobFailure(
            outcome=JobOutcome(row[0]),
            reason=str(row[1] or ""),
            recovery_action=str(row[2] or ""),
            user_explanation=str(row[3] or ""),
        )

    def protect_art(self, sha: str, data: bytes) -> str:
        """Store *data* content-addressed under *sha* (put-if-missing).

        Reuses the ContentStore, which writes a blob only when its hash is absent, so
        a resumed job that re-protects the same art never duplicates the blob. Verifies
        *data* hashes to *sha* before storing.
        """
        if sha256_hex(data) != sha:
            raise ValueError("art bytes do not match the provided content sha")
        return self.content.put(data)

    def set_result(self, phase: str, result: Any, verified: bool = True) -> None:
        """Record *result* for *phase* into the current job's checkpoint.

        Every call appends to the raw ``results`` map; a *verified* result is also
        written to the per-phase ``known_good`` map. Because ``known_good`` is keyed
        by phase, re-running or recording an earlier phase never overwrites or
        regresses a previously-successful later known-good value.
        """
        if self._current is None:
            raise RuntimeError("set_result must be called from within a phase function")
        cp = dict(self._current.checkpoint)
        results = dict(cp.get("results", {}))
        results[phase] = result
        cp["results"] = results
        if verified:
            known_good = dict(cp.get("known_good", {}))
            known_good[phase] = result
            cp["known_good"] = known_good
        self._update(self._current.id, checkpoint_json=json.dumps(cp))
        self._current = replace(self._current, checkpoint=cp)

    def last_known_good(self, phase: str | None = None) -> Any:
        """Return the known-good result for *phase* (or the highest phase)."""
        cp = self._current.checkpoint if self._current is not None else {}
        known_good = cp.get("known_good", {})
        if phase is not None:
            return known_good.get(phase)
        if not known_good:
            return None
        return known_good[max(known_good)]

    # -- status ------------------------------------------------------------------

    def report(self) -> JobReport:
        """Return a :class:`JobReport` snapshot of every job, oldest first."""
        rows = self.db.execute("SELECT * FROM jobs ORDER BY id").fetchall()
        jobs = [self._job_from_row(r) for r in rows]
        by_state: dict[str, int] = {}
        for job in jobs:
            by_state[job.state] = by_state.get(job.state, 0) + 1
        return JobReport(total=len(jobs), by_state=by_state, jobs=jobs)

    # -- persistence helpers -----------------------------------------------------

    def _update(self, job_id: int, **cols: Any) -> None:
        if not cols:
            return
        sets = ", ".join(f"{col} = ?" for col in cols)
        params = list(cols.values())
        try:
            self.db.execute(
                f"UPDATE jobs SET {sets}, updated_at = {_TIMESTAMP} WHERE id = ?",
                (*params, job_id),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CuratorError(f"failed to update job {job_id}: {exc}") from exc

    def _mark_active(self, job: Job) -> None:
        try:
            self.db.execute(
                "UPDATE jobs SET state = 'active', attempts = attempts + 1,"
                f" updated_at = {_TIMESTAMP} WHERE id = ?",
                (job.id,),
            )
            self.db.commit()
        except sqlite3.Error as exc:
            self.db.rollback()
            raise CuratorError(f"failed to activate job {job.id}: {exc}") from exc

    def _checkpoint(self, job: Job, next_phase: str) -> None:
        cp = self._current.checkpoint if self._current else job.checkpoint
        self._update(
            job.id,
            state="checkpointed",
            phase=next_phase,
            checkpoint_json=json.dumps(cp),
        )

    def _complete(self, job: Job) -> None:
        cp = self._current.checkpoint if self._current else job.checkpoint
        self._update(
            job.id,
            state="completed",
            checkpoint_json=json.dumps(cp),
        )

    def _job_from_row(self, row: sqlite3.Row | tuple[Any, ...]) -> Job:
        d = dict(zip(_JOB_COLUMNS, row))
        return Job(
            id=int(d["id"]),
            key=str(d["key"]),
            kind=JobKind(d["kind"]),
            payload=json.loads(d["payload_json"]) if d["payload_json"] else {},
            state=str(d["state"]),
            phase=str(d["phase"] or ""),
            checkpoint=json.loads(d["checkpoint_json"]) if d["checkpoint_json"] else {},
            attempts=int(d["attempts"] or 0),
            created_at=str(d["created_at"] or ""),
            updated_at=str(d["updated_at"] or ""),
        )

    def _reload(self, job_id: int) -> Job:
        row = self.db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise CuratorError(f"no job with id {job_id}")
        return self._job_from_row(row)
