"""Durable job model — kinds, classified outcomes, and JSON round-trip (M006/S02).

The orchestrator (``curator.jobs.orchestrator``) persists :class:`Job` rows to the
``jobs`` table (schema v12) and classifies any failure as a :class:`JobFailure`
carrying one of the six :class:`JobOutcome` values, so operators and tests can
distinguish retryable infrastructure blips from permanent, policy-blocked,
capability-unsupported, unresolved-external, or user-cancelled work.

Both job kinds and outcomes are enums serialized by their string ``.value`` so the
model stays trivially JSON-serializable via :meth:`Job.to_dict` /
:meth:`Job.from_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JobKind(Enum):
    """The discrete kinds of durable work the orchestrator can run."""

    INGEST = "ingest"
    ANALYZE = "analyze"
    RENDER = "render"
    PUBLISH = "publish"
    ROTATE = "rotate"


class JobOutcome(Enum):
    """Classified terminal failure outcome for a job.

    - ``TRANSIENT``              — retryable infrastructure blip (network, lock).
    - ``PERMANENT``              — deterministic failure that will not resolve on retry.
    - ``POLICY_BLOCKED``         — a rule/policy rejected the work.
    - ``CAPABILITY_UNSUPPORTED`` — the renderer/destination lacks the required feature.
    - ``UNRESOLVED_EXTERNAL``    — an external dependency failed and cannot be resolved.
    - ``USER_CANCELLED``         — the operator cancelled the job.
    """

    TRANSIENT = "transient"
    PERMANENT = "permanent"
    POLICY_BLOCKED = "policy_blocked"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    UNRESOLVED_EXTERNAL = "unresolved_external"
    USER_CANCELLED = "user_cancelled"


@dataclass(frozen=True)
class Job:
    """A durable unit of work persisted to the ``jobs`` table.

    ``key`` is the content-derived idempotency key (kind + canonical payload), used
    by :meth:`~curator.jobs.orchestrator.JobOrchestrator.enqueue` to enqueue the same
    work only once. ``state`` walks ``queued -> active -> checkpointed -> completed``
    or ``error``/``cancelled``. ``checkpoint`` is a JSON dict holding per-phase
    results; ``phase`` names the current step so a crash-resumed job continues where
    it stopped without re-running completed phases.
    """

    id: int
    key: str
    kind: JobKind
    payload: dict[str, Any]
    state: str
    phase: str = ""
    checkpoint: dict[str, Any] = field(default_factory=dict)
    attempts: int = 0
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict (enum expanded to its string value)."""
        return {
            "id": self.id,
            "key": self.key,
            "kind": self.kind.value,
            "payload": self.payload,
            "state": self.state,
            "phase": self.phase,
            "checkpoint": self.checkpoint,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        """Reconstruct a :class:`Job` from a :meth:`Job.to_dict` dict."""
        return cls(
            id=int(data["id"]),
            key=data["key"],
            kind=JobKind(data["kind"]),
            payload=dict(data.get("payload") or {}),
            state=data["state"],
            phase=str(data.get("phase") or ""),
            checkpoint=dict(data.get("checkpoint") or {}),
            attempts=int(data.get("attempts") or 0),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass(frozen=True)
class JobFailure:
    """A classified failure attached to a job.

    ``outcome`` is one of the six :class:`JobOutcome` values; ``reason`` the raw
    failure text, ``recovery_action`` the recommended remediation, and
    ``user_explanation`` an end-user-readable summary. Persisted to the
    ``jobs`` outcome columns by :meth:`~curator.jobs.orchestrator.JobOrchestrator.fail`.
    """

    outcome: JobOutcome
    reason: str
    recovery_action: str
    user_explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return {
            "outcome": self.outcome.value,
            "reason": self.reason,
            "recovery_action": self.recovery_action,
            "user_explanation": self.user_explanation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobFailure:
        """Reconstruct a :class:`JobFailure` from a :meth:`JobFailure.to_dict` dict."""
        return cls(
            outcome=JobOutcome(data["outcome"]),
            reason=data["reason"],
            recovery_action=data["recovery_action"],
            user_explanation=data.get("user_explanation", ""),
        )
