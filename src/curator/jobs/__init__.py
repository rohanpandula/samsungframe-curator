"""Durable job orchestration — checkpoints and classified outcomes (M006/S02).

The orchestrator executes units of work from the ``jobs`` table (schema v12) as a
checkpointed state machine: :class:`~curator.jobs.orchestrator.JobOrchestrator`
enqueues idempotently by content key, advances jobs phase-by-phase with durable
checkpoints, resumes after a crash without duplicating content-addressed art or
regressing last-known-good results, and classifies every failure into one of the six
:class:`~curator.jobs.model.JobOutcome` values.

- :mod:`curator.jobs.model`        — :class:`Job`, :class:`JobFailure`, and the
  :class:`JobKind` / :class:`JobOutcome` enums, all JSON-serializable.
- :mod:`curator.jobs.orchestrator` — the checkpointed executor and its
  :class:`JobReport` status surface.
"""

from __future__ import annotations

from curator.jobs.model import Job, JobFailure, JobKind, JobOutcome
from curator.jobs.orchestrator import JobOrchestrator, JobReport

__all__ = [
    "Job",
    "JobFailure",
    "JobKind",
    "JobOrchestrator",
    "JobOutcome",
    "JobReport",
]
