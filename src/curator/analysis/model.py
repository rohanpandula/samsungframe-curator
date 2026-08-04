"""Model specification + runner contracts (M002/S01).

:class:`ModelSpec` is a pure value object describing one analyzable model.
:class:`ModelRunner` is the capability-probed execution seam; a runner's
:meth:`ModelRunner.available` reports whether it can run here.
:class:`CpuReferenceRunner` is the always-available golden reference every
provider falls back to.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class ModelPrecision(enum.StrEnum):
    """Numeric precision a model is served at."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dataclass(frozen=True)
class ModelSpec:
    """A pinned, reproducible model specification."""

    name: str
    version: str
    family: str
    backend: str = "cpu"
    precision: ModelPrecision | str = ModelPrecision.FP32
    sha256: str | None = None
    memory_estimate_mb: int | None = None
    task: str = "analysis"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return {
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "backend": self.backend,
            "precision": getattr(self.precision, "value", self.precision),
            "sha256": self.sha256,
            "memory_estimate_mb": self.memory_estimate_mb,
            "task": self.task,
        }


class ModelRunner(ABC):
    """Capability-probed executor for a :class:`ModelSpec`.

    Implementing subclasses report :meth:`available` and perform a single
    :meth:`run`. The contract is deliberately minimal so local (CPU/CoreML/Metal)
    and remote (cloud/hybrid) runners share one seam.
    """

    @abstractmethod
    def available(self) -> bool:
        """Return True if this runner can execute on the current host."""

    @abstractmethod
    def run(self, spec: ModelSpec, payload: Any) -> Any:
        """Execute *spec* against *payload* and return the model output."""


class CpuReferenceRunner(ModelRunner):
    """The always-available golden-reference runner.

    Always reports ``available()`` True regardless of host, so a deterministic
    CPU result is never unavailable (R029 baseline).
    """

    def available(self) -> bool:
        return True

    def run(self, spec: ModelSpec, payload: Any) -> Any:
        return payload
