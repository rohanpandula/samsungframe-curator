"""Local ONNX embedding provider (M009/S02): offline, deterministic, capability-probed.

:class:`OnnxEmbeddingProvider` implements the :class:`EmbeddingProvider` ABC and
slots into the existing :class:`~curator.analysis.provider.AnalysisProvider`/
:class:`~curator.analysis.compute.ComputeBackend`/:class:`~curator.analysis.model.ModelSpec`
seam (reused directly, not rebuilt) — this milestone's own dependency decision:
``onnxruntime`` (CPU-only, 19.1MB, MIT-licensed, no torch) is the first ML
dependency this repo takes on, and it is allowed to conclude "not worth it."

The model file is never downloaded at request time. :func:`resolve_model_path`
resolves a fixed local cache path (env override, else
``<data_root>/models/embedding/{EMBEDDING_MODEL_VERSION}.onnx``);
:meth:`OnnxEmbeddingProvider.probe` reports ``ok=False`` with a clean, actionable
message when nothing is placed there — never an exception, never a fetch. When a
checksum is pinned via ``CURATOR_TASTE_EMBEDDING_MODEL_SHA256``, the file is
verified before every load (:meth:`OnnxEmbeddingProvider._ensure_session`), not
just at construction/probe time, so a file swapped out after a successful probe
is still caught before inference (T-09-05).

Inference is pinned to ``CPUExecutionProvider`` only, with
``use_deterministic_compute``, a fixed ``intra_op_num_threads``, and
``ORT_SEQUENTIAL`` execution — measured bit-identical across repeated calls and
fresh sessions on CPU (never :data:`~curator.analysis.compute.ComputeBackend.AUTO`,
which this provider never accepts or passes through).
"""

from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

from curator.analysis.compute import ComputeBackend
from curator.analysis.model import ModelPrecision, ModelSpec
from curator.analysis.provider import ComputeProbe
from curator.config import CuratorConfig
from curator.hashing import sha256_hex
from curator.taste.embedding.errors import EmbeddingUnavailableError

#: Pinned checkpoint + revision this provider's vectors are versioned against.
#: Comparing vectors across different values of this string is a bug — every
#: :class:`~curator.taste.embedding.store.EmbeddingStore` read is scoped by it.
EMBEDDING_MODEL_VERSION = "clip-vit-b-32-laion2b-1"

#: Output vector dimensionality for :data:`EMBEDDING_MODEL_VERSION`.
EMBEDDING_DIM = 512

#: Descriptive metadata for the pinned checkpoint, using the existing
#: :class:`~curator.analysis.model.ModelSpec` seam directly (R040/CONTEXT: reuse,
#: never a parallel bespoke spec type). Not threaded through a ``ModelRunner`` in
#: this slice — :class:`OnnxEmbeddingProvider` manages its ONNX Runtime session
#: directly, the same way :class:`~curator.analysis.local.LocalAnalysisProvider`
#: keeps ``model_spec`` as descriptive metadata rather than routing every call
#: through a runner.
_MODEL_SPEC = ModelSpec(
    name="clip-vit-b-32-laion2b",
    version=EMBEDDING_MODEL_VERSION,
    family="clip",
    backend="cpu",
    precision=ModelPrecision.FP32,
    task="embedding",
)

# CLIP's fixed preprocessing normalization constants (the values the reference
# encoder was trained with), applied to pixels scaled to [0, 1].
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

#: Square input side length the encoder's fixed ``(1, 3, 224, 224)`` contract expects.
_INPUT_SIZE = 224


@dataclass(frozen=True)
class EmbeddingCapabilities:
    """Declared capabilities of an embedding provider."""

    dim: int
    backends: frozenset[ComputeBackend]
    air_gapped: bool = True
    deterministic: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict (enums serialized by their values)."""
        return {
            "dim": self.dim,
            "backends": sorted(b.value for b in self.backends),
            "air_gapped": self.air_gapped,
            "deterministic": self.deterministic,
        }


class EmbeddingProvider(ABC):
    """Abstract embedding provider.

    Mirrors :class:`~curator.analysis.provider.AnalysisProvider`'s two-method
    shape: :meth:`capabilities` (static contract) and a live :meth:`probe`
    (runtime health). A provider-specific ``embed`` method (not part of this ABC,
    since its exact signature is provider-specific) turns image bytes into a
    vector.
    """

    @abstractmethod
    def capabilities(self) -> EmbeddingCapabilities:
        """Return this provider's declared capabilities."""

    @abstractmethod
    def probe(self) -> ComputeProbe:
        """Return a live health probe for the current backend."""


def resolve_model_path(data_root: Path | None = None) -> Path:
    """Return the local cache path the embedding model is loaded from.

    ``CURATOR_TASTE_EMBEDDING_MODEL_PATH`` takes precedence when set (a flat
    ``os.environ.get(...)`` operational knob, mirroring
    :func:`resolve_expected_sha256` — not a ``CuratorConfig`` axis); otherwise
    resolves to ``<data_root or CuratorConfig().data_root>/models/embedding/
    {EMBEDDING_MODEL_VERSION}.onnx``, mirroring
    :func:`curator.db.default_db_path`'s exact data-root-or-config-default
    pattern. Never downloads anything — this is a path computation only.
    """
    override = os.environ.get("CURATOR_TASTE_EMBEDDING_MODEL_PATH", "").strip()
    if override:
        return Path(override)
    if data_root is None:
        data_root = CuratorConfig().data_root
    return Path(data_root) / "models" / "embedding" / f"{EMBEDDING_MODEL_VERSION}.onnx"


def resolve_expected_sha256() -> str | None:
    """Return the pinned model checksum from the environment, or ``None``.

    ``CURATOR_TASTE_EMBEDDING_MODEL_SHA256`` is an optional operational knob (a
    single value, not a config axis) — mirrors
    :func:`~curator.taste.dialogue.extraction.extraction_config_from_env`'s flat
    ``os.environ.get(...)`` idiom rather than the nested ``CuratorConfig``
    pydantic-settings pattern. When set, :meth:`OnnxEmbeddingProvider._verify`
    refuses to load a file whose hash doesn't match.
    """
    value = os.environ.get("CURATOR_TASTE_EMBEDDING_MODEL_SHA256", "").strip()
    return value or None


class OnnxEmbeddingProvider(EmbeddingProvider):
    """CPU-only, deterministic, offline ONNX Runtime image-embedding provider.

    The model file crosses from disk into an executing runtime only at the path
    :func:`resolve_model_path` computes itself — never a request- or
    CLI-argument-supplied path (T-09-05, the only trust boundary this module
    owns). Construction does no I/O beyond a path computation (matches
    ``api.py``'s lazy-``Catalog`` "importing/constructing has no side effects"
    posture); the ONNX Runtime session is built lazily on first :meth:`embed`
    call and cached.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        model_version: str = EMBEDDING_MODEL_VERSION,
        expected_sha256: str | None = None,
    ) -> None:
        self.model_path = model_path if model_path is not None else resolve_model_path()
        self.model_version = model_version
        self.expected_sha256 = (
            expected_sha256 if expected_sha256 is not None else resolve_expected_sha256()
        )
        self._session: Any | None = None

    # -- EmbeddingProvider ABC ---------------------------------------------------

    def capabilities(self) -> EmbeddingCapabilities:
        return EmbeddingCapabilities(dim=EMBEDDING_DIM, backends=frozenset({ComputeBackend.CPU}))

    def _verify(self) -> str | None:
        """Return an error message when the model is unusable, else ``None``.

        Checked in order: the file must exist at :attr:`model_path`; when
        :attr:`expected_sha256` is set, the file's sha256 must match it exactly.
        This is the T-09-05 mitigation — the model is loaded only from a fixed
        local path this code resolves itself, and only after an exact hash match
        when one is pinned. Never raises.
        """
        if not self.model_path.exists():
            return (
                f"embedding model not available at {self.model_path}"
                " — see docs for manual placement"
            )
        if self.expected_sha256:
            actual = sha256_hex(self.model_path.read_bytes())
            if actual != self.expected_sha256:
                return (
                    f"embedding model checksum mismatch at {self.model_path}"
                    " — refusing to load"
                )
        return None

    def probe(self) -> ComputeProbe:
        """Return a live, cheap health probe. Never raises, never downloads."""
        message = self._verify()
        if message is None:
            return ComputeProbe(
                ok=True,
                backend=ComputeBackend.CPU,
                available_backends=frozenset({ComputeBackend.CPU}),
                message="embedding model available (deterministic, air-gapped)",
            )
        return ComputeProbe(
            ok=False,
            backend=ComputeBackend.CPU,
            available_backends=frozenset(),
            message=message,
        )

    # -- inference ----------------------------------------------------------------

    def _ensure_session(self) -> Any:
        """Return the cached ONNX Runtime session, verifying + building it lazily.

        Re-runs :meth:`_verify` on every call (not just at construction/probe
        time) — a file swapped out after :meth:`probe` succeeded but before
        :meth:`embed` is called must still be caught. Raises
        :class:`EmbeddingUnavailableError` on any verification failure.
        """
        message = self._verify()
        if message is not None:
            raise EmbeddingUnavailableError(message)
        if self._session is None:
            sess_options = ort.SessionOptions()
            sess_options.use_deterministic_compute = True
            sess_options.intra_op_num_threads = 1
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    def embed(self, image_bytes: bytes) -> np.ndarray:
        """Return the L2-normalized ``(EMBEDDING_DIM,)`` float32 embedding for *image_bytes*.

        Decodes + preprocesses (EXIF-corrected, resized so the shorter side is
        224px, center-cropped to exactly 224x224, CLIP-normalized, CHW +
        batch-dim) then runs the ONNX session. Raises
        :class:`EmbeddingUnavailableError` — never returns garbage or a silent
        zero vector — when the model cannot be loaded.
        """
        session = self._ensure_session()
        with Image.open(io.BytesIO(image_bytes)) as img:
            # Same EXIF-orientation idiom as ingest/decode.py's decode_image, but
            # this path needs real pixel data (not just a signature), so it does
            # not call decode_image itself.
            oriented = ImageOps.exif_transpose(img)
            rgb = oriented.convert("RGB")
            resized = _resize_shorter_side(rgb, _INPUT_SIZE)
            cropped = _center_crop(resized, _INPUT_SIZE, _INPUT_SIZE)
            array = np.asarray(cropped, dtype=np.float32) / 255.0
        normalized = (array - _CLIP_MEAN) / _CLIP_STD
        chw = normalized.transpose(2, 0, 1).astype(np.float32)
        batch = chw[np.newaxis, :, :, :]
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: batch})
        vector = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


def _resize_shorter_side(img: Image.Image, size: int) -> Image.Image:
    """Resize *img* so its shorter side is exactly *size* px, preserving aspect ratio.

    Matches CLIP's reference preprocessing (``Resize(size, interpolation=BICUBIC)``
    given a single int) — deliberately not this codebase's usual
    ``Image.Resampling.LANCZOS`` (used elsewhere for thumbnail/render quality, an
    unrelated context); BICUBIC is what the real encoder was calibrated against.
    """
    width, height = img.size
    if width <= height:
        new_width = size
        new_height = round(height * (size / width))
    else:
        new_height = size
        new_width = round(width * (size / height))
    return img.resize((new_width, new_height), Image.Resampling.BICUBIC)


def _center_crop(img: Image.Image, width: int, height: int) -> Image.Image:
    """Return the exact *width*x*height* center crop of *img*."""
    src_width, src_height = img.size
    left = (src_width - width) // 2
    top = (src_height - height) // 2
    return img.crop((left, top, left + width, top + height))
