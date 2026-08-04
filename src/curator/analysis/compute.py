"""Compute backends + deterministic, strict-device resolution (M002/S01).

:func:`resolve_backend` is the pure policy seam (R029): given a requested
:class:`ComputeBackend`, the set of backends a host actually exposes, and a
``strict`` flag, it deterministically picks a backend. Strict-device demand for
an unavailable accelerator raises :class:`ComputeBackendError`; otherwise it
falls back to :data:`ComputeBackend.CPU` (the always-available golden reference).
"""

from __future__ import annotations

import enum
from collections.abc import Iterable

from curator.analysis.errors import ComputeBackendError


class ComputeBackend(enum.Enum):
    """A compute device an analysis model may run on."""

    AUTO = "auto"
    CPU = "cpu"
    METAL = "metal"
    COREML = "coreml"
    CUDA = "cuda"


#: Deterministic preference order for AUTO resolution: highest-tier accelerator
#: first, CPU (always available) as the terminal fallback.
_PREFERENCE_ORDER: tuple[ComputeBackend, ...] = (
    ComputeBackend.METAL,
    ComputeBackend.COREML,
    ComputeBackend.CUDA,
    ComputeBackend.CPU,
)


def _as_backend(value: ComputeBackend | str) -> ComputeBackend:
    """Coerce *value* (enum member or its string name) to a :class:`ComputeBackend`."""
    if isinstance(value, ComputeBackend):
        return value
    return ComputeBackend(value)


def resolve_backend(
    requested: ComputeBackend | str,
    available: Iterable[ComputeBackend | str],
    strict: bool = False,
) -> tuple[ComputeBackend, bool]:
    """Determine the effective compute backend for a run.

    *requested* is the desired backend (or ``AUTO``); *available* is the set of
    backends the host reports; *strict* demands that the requested backend be
    honored exactly. Returns ``(backend, effective_strict)``:

    - ``AUTO`` — picks the best available backend preferentially; the returned
      ``strict`` is ``False`` (auto never demands a specific device).
    - A specific backend that is available — returns ``(requested, strict)``.
    - A specific backend that is unavailable with ``strict`` — raises
      :class:`ComputeBackendError`.
    - A specific backend that is unavailable without ``strict`` — falls back to
      ``(CPU, False)``.

    Deterministic: identical inputs always produce identical outputs.
    """
    requested_backend = _as_backend(requested)
    available_set: set[ComputeBackend] = {
        _as_backend(item) for item in available
    }
    available_set.add(ComputeBackend.CPU)  # CPU is always available (golden).

    if requested_backend is ComputeBackend.AUTO:
        for candidate in _PREFERENCE_ORDER:
            if candidate in available_set:
                return candidate, False
        return ComputeBackend.CPU, False

    if requested_backend in available_set:
        return requested_backend, strict

    if strict:
        raise ComputeBackendError(
            f"strict device {requested_backend.value!r} is not available "
            f"(available: {sorted(b.value for b in available_set)})"
        )
    return ComputeBackend.CPU, False


def strict_device(
    requested: ComputeBackend | str,
    available: Iterable[ComputeBackend | str],
) -> ComputeBackend:
    """Resolve *requested* strictly, raising :class:`ComputeBackendError` if unmet.

    Convenience wrapper over :func:`resolve_backend` with ``strict=True`` that
    returns just the chosen backend on success.
    """
    backend, _ = resolve_backend(requested, available, strict=True)
    return backend
