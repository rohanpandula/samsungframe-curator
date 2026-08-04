"""Versioned, JSON-serializable analysis result schema (M002/S01).

``AnalysisResult`` is the provider-neutral currency exchanged across local /
cloud / hybrid :class:`~curator.analysis.provider.AnalysisProvider`
implementations (R005). It is a frozen dataclass tree — every nested signal
group is itself a frozen dataclass — that round-trips losslessly through JSON via
:meth:`AnalysisResult.to_dict` / :meth:`AnalysisResult.from_dict`.

Versioning is lenient **forward-compatible on keys**: unknown keys present in
serialized input are preserved (not dropped) so a newer producer's data survives
an older consumer. An unknown *schema version*, however, is a hard error and
raises :class:`SchemaVersionError` because it signals incompatible semantics.

Every signal dataclass exposes ``to_dict()`` / ``from_dict()``; importing
:data:`SCHEMA_VERSION` from this module pins the current contract version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from curator.analysis.errors import SchemaVersionError

#: Current serialization contract version. Consumers reject anything newer.
SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class SchemaBase:
    """Base for all schema dataclasses: JSON round-trip + forward-compat.

    ``_extra`` (omitted from IR by :meth:`to_dict`) carries any unknown keys found
    during deserialization so a newer producer's fields survive the round-trip.
    Subclasses are frozen dataclasses; this base provides the generic
    ``to_dict()`` / ``from_dict()`` machinery.
    """

    _extra: dict[str, Any] = field(
        default_factory=dict, repr=False, init=False
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict (nested dataclasses expanded).

        Any forward-compat keys preserved in ``_extra`` are re-emitted as
        top-level keys so the dumped form matches what a newer producer wrote.
        """
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "_extra":
                continue
            out[f.name] = _serialize(getattr(self, f.name))
        out.update(self._extra)
        return out

    def to_json(self, indent: int = 2) -> str:
        """Return this schema object serialized as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SchemaBase:
        """Build an instance from a dict, preserving unknown keys as ``_extra``."""
        if isinstance(data, cls):
            return data
        return _build(cls, data)


# -- primitive value objects ------------------------------------------------


@dataclass(frozen=True)
class Point(SchemaBase):
    """A 2D coordinate, e.g. a saliency focal point."""

    x: float = 0.0
    y: float = 0.0


@dataclass(frozen=True)
class BoundingBox(SchemaBase):
    """An axis-aligned box in normalized [0, 1] image coordinates."""

    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0


# -- nested signal groups ---------------------------------------------------


@dataclass(frozen=True)
class QualitySignals(SchemaBase):
    """Technical and aesthetic quality assessment of one image."""

    technical_quality: float = 0.0
    aesthetic_quality: float = 0.0
    sharpness: float = 0.0
    exposure: float = 0.0
    contrast: float = 0.0
    resolution_sufficient: bool = False


@dataclass(frozen=True)
class Saliency(SchemaBase):
    """Saliency map summary: size, detected subjects, and focal point."""

    map_size: tuple[int, int] = (0, 0)
    subjects: list[BoundingBox] = field(default_factory=list)
    focal_point: Point = field(default_factory=Point)


@dataclass(frozen=True)
class CropSafety(SchemaBase):
    """Per-direction safe-to-crop flags and their margin ratios (Samsung frame crop)."""

    safe_north: bool = True
    safe_south: bool = True
    safe_east: bool = True
    safe_west: bool = True
    margin_north: float = 0.0
    margin_south: float = 0.0
    margin_east: float = 0.0
    margin_west: float = 0.0


@dataclass(frozen=True)
class ColorStory(SchemaBase):
    """Dominant color palette, harmony, and pickable background candidates."""

    dominant_colors: list[dict[str, Any]] = field(default_factory=list)
    colorfulness: float = 0.0
    harmony: float = 0.0
    background_candidates: list[str] = field(default_factory=list)
    background_choice: str | None = None


@dataclass(frozen=True)
class Pairing(SchemaBase):
    """Affinity between two images for pairing as a set (e.g. a pair of frames)."""

    affinity: float = 0.0
    phash_distance: int | None = None
    palette_distance: float | None = None
    date_proximity: float | None = None
    orientation_match: bool = False


@dataclass(frozen=True)
class PerceptualRepresentation(SchemaBase):
    """A fixed-length embedding vector plus the method that produced it."""

    method: str = ""
    dim: int = 0
    vector: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class AnalysisMetadata(SchemaBase):
    """Provenance about how an analysis was produced."""

    profile: str = ""
    compute_backend: str = ""
    model_spec: dict[str, Any] = field(default_factory=dict)
    engine_version: str = ""
    deterministic: bool = True
    timing_ms: float = 0.0


# -- root result ------------------------------------------------------------


@dataclass(frozen=True)
class AnalysisResult(SchemaBase):
    """The normalized, versioned output of a single image analysis (R005).

    ``asset_id`` keys the analyzed asset. Nested signal groups default to empty
    so a partially-populated result is representable (S03 policy may only consume
    a subset of signals). ``schema_version`` pins the contract.
    """

    asset_id: str
    schema_version: str = SCHEMA_VERSION
    quality: QualitySignals = field(default_factory=QualitySignals)
    saliency: Saliency = field(default_factory=Saliency)
    crop_safety: CropSafety = field(default_factory=CropSafety)
    color_story: ColorStory = field(default_factory=ColorStory)
    pairing: Pairing = field(default_factory=Pairing)
    perceptual: PerceptualRepresentation = field(default_factory=PerceptualRepresentation)
    metadata: AnalysisMetadata = field(default_factory=AnalysisMetadata)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        """Build an :class:`AnalysisResult`, rejecting an unknown schema version.

        A missing ``schema_version`` is treated as the current version
        (backward-compat for older/partial payloads). Any other value raises
        :class:`SchemaVersionError`.
        """
        version = data.get("schema_version", SCHEMA_VERSION)
        if str(version) != SCHEMA_VERSION:
            raise SchemaVersionError(
                f"unsupported analysis schema version: {version!r} "
                f"(expected {SCHEMA_VERSION!r})"
            )
        return _build(cls, data)

    @classmethod
    def from_json(cls, text: str) -> AnalysisResult:
        """Deserialize an :class:`AnalysisResult` from a JSON string."""
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------------
# generic (de)serialization helpers
# ---------------------------------------------------------------------------


def _serialize(value: Any) -> Any:
    """Recursively convert a schema value into JSON-encodable primitives."""
    if isinstance(value, SchemaBase):
        return value.to_dict()
    if isinstance(value, (list, tuple, set)):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize(v) for k, v in value.items()}
    return value


def _build(cls: Any, data: dict[str, Any]) -> Any:
    """Construct *cls* from *data* using the dataclass field hints.

    Values for nested dataclass / container fields are coerced recursively via
    :func:`_convert`; keys that are not dataclass fields are preserved as
    ``_extra`` (forward-compat).
    """
    if not is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name == "_extra":
            continue
        if f.name in data:
            kwargs[f.name] = _convert(data[f.name], hints.get(f.name, Any))
    obj = cast(type, cls)(**kwargs)
    extra = {k: v for k, v in data.items() if k not in known}
    object.__setattr__(obj, "_extra", extra)
    return obj


def _convert(value: Any, type_hint: Any) -> Any:
    """Coerce *value* to *type_hint*, recursing into nested containers.

    ``Any``/scalar/dict hints pass the value through unchanged; dataclass hints
    build a nested instance; ``list``/``set``/``tuple`` and ``Optional``/``Union``
    hints rewrite their contents. Silently passes through on ambiguity so a
    partially-serialized newer payload degrades gracefully.
    """
    if value is None:
        return None
    if is_dataclass(type_hint):
        return _build(type_hint, value)
    origin = get_origin(type_hint)
    if origin is None:
        return value
    args = get_args(type_hint)
    if origin in (Union, UnionType):
        for arg in args:
            try:
                return _convert(value, arg)
            except (TypeError, ValueError):
                continue
        return value
    if origin is dict:
        return value
    if origin is tuple:
        elem = args[0] if args and args[0] is not Ellipsis else Any
        converted = [_convert(item, elem) for item in value]
        return tuple(converted)
    if origin in (list, set):
        elem = args[0] if args else Any
        converted = [_convert(item, elem) for item in value]
        return set(converted) if origin is set else converted
    return value
