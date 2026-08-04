"""Versioned, JSON-serializable Art Direction Manifest (M002/S03).

``ArtDirectionManifest`` captures a curator's art-direction intent for one
curated set: which source images feed it, how they are laid out on the frame,
the background, the intended processing, and any per-target overrides. It is a
frozen dataclass tree that round-trips losslessly through JSON via
:meth:`ArtDirectionManifest.to_dict` / :meth:`ArtDirectionManifest.from_dict`.

Versioning is strict on the schema version: deserializing a payload whose
``manifest_version`` is not the current :data:`MANIFEST_VERSION` raises
:class:`ManifestError`. Per-target overrides are schema-validated lazily by
:meth:`ArtDirectionManifest.resolved_for`, which rejects override fields that
are not known manifest fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, Union, cast, get_args, get_origin, get_type_hints

from curator.errors import CuratorError

MANIFEST_VERSION = "1"


class ManifestError(CuratorError):
    """Raised for an invalid or version-mismatched Art Direction Manifest."""


class LayoutTreatment(Enum):
    """How source content is placed on the frame surface."""

    SINGLE_FULLBLEED = "single_fullbleed"
    CONTAIN_MATTE = "contain_matte"
    PANORAMIC = "panoramic"
    SQUARE = "square"
    DIPTYCH = "diptych"


_NESTED_DICT_FIELDS = {"background", "processing_intent"}


@dataclass(frozen=True)
class SourceRegion:
    """A normalized source region used as art, keyed by content identity."""

    source_sha256: str
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    crop: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceRegion:
        if isinstance(data, cls):
            return data
        return _build(cls, data)


@dataclass(frozen=True)
class BackgroundSpec:
    """Background / matte treatment: choice and the matte border spec."""

    background_choice: str = "none"
    color: str | None = None
    width: int | None = None
    style: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BackgroundSpec:
        if isinstance(data, cls):
            return data
        return _build(cls, data)


@dataclass(frozen=True)
class ProcessingIntent:
    """Intended color processing for the rendered set."""

    color_profile: str = "srgb"
    upscale_warning: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessingIntent:
        if isinstance(data, cls):
            return data
        return _build(cls, data)


@dataclass(frozen=True)
class ArtDirectionManifest:
    """The versioned art-direction intent for one curated set.

    ``target_overrides`` is keyed by target name (e.g. ``"1080p"``, ``"4k"``,
    ``"custom:<name>"``); each value is a subset of manifest fields that may
    differ per target, kept in plain serialized form. Override schema validity
    is enforced lazily by :meth:`resolved_for`.
    """

    manifest_version: str = MANIFEST_VERSION
    sources: list[str] = field(default_factory=list)
    regions: list[SourceRegion] = field(default_factory=list)
    layout_treatment: LayoutTreatment = LayoutTreatment.SINGLE_FULLBLEED
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    processing_intent: ProcessingIntent = field(default_factory=ProcessingIntent)
    pairing_order: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    target_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _to_plain(getattr(self, f.name)) for f in fields(self)}

    def validate(self) -> None:
        """Check internal invariants, raising :class:`ManifestError` on failure.

        Enforces a current ``manifest_version``, a valid ``layout_treatment``,
        and string-keyed ``target_overrides``. Unknown *override* fields are
        rejected later by :meth:`resolved_for`.
        """
        if str(self.manifest_version) != MANIFEST_VERSION:
            raise ManifestError(
                f"unsupported manifest version {self.manifest_version!r} "
                f"(expected {MANIFEST_VERSION!r})"
            )
        if not isinstance(self.layout_treatment, LayoutTreatment):
            raise ManifestError(
                f"layout_treatment must be a LayoutTreatment, got {self.layout_treatment!r}"
            )
        if not self.sources:
            raise ManifestError("manifest requires at least one source sha")
        for target, overrides in self.target_overrides.items():
            if not isinstance(target, str):
                raise ManifestError(
                    f"target override key must be a string, got {target!r}"
                )
            if not isinstance(overrides, dict):
                raise ManifestError(
                    f"target override for {target!r} must be a dict of fields"
                )

    def resolved_for(self, target: str) -> ArtDirectionManifest:
        """Return a copy with ``target_overrides[target]`` applied over the base.

        A missing target returns the base manifest unchanged. Otherwise every
        override field is validated against the known manifest fields (unknown
        fields raise :class:`ManifestError`), applied over the serialized base —
        scalar fields such as ``layout_treatment`` are replaced, while nested
        dict fields such as ``background`` / ``processing_intent`` are shallow
        merged so a partial override wins per key. The base manifest is never
        mutated.
        """
        override = self.target_overrides.get(target)
        if override is None:
            return self

        _validate_override_fields(override)

        base = self.to_dict()
        base_overrides = base.pop("target_overrides", {})
        merged: dict[str, Any] = dict(base)
        for key, value in override.items():
            if (
                key in _NESTED_DICT_FIELDS
                and isinstance(merged.get(key), dict)
                and isinstance(value, dict)
            ):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        merged["target_overrides"] = base_overrides
        return ArtDirectionManifest.from_dict(merged)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtDirectionManifest:
        if isinstance(data, cls):
            return data
        version = data.get("manifest_version", MANIFEST_VERSION)
        if str(version) != MANIFEST_VERSION:
            raise ManifestError(
                f"unsupported manifest version {version!r} (expected {MANIFEST_VERSION!r})"
            )
        obj = _build(cls, data)
        normalized = {str(t): _to_plain(d) for t, d in obj.target_overrides.items()}
        object.__setattr__(obj, "target_overrides", normalized)
        return obj


def _validate_override_fields(override: dict[str, Any]) -> None:
    known = {f.name for f in fields(ArtDirectionManifest)}
    unknown = sorted(set(override) - known)
    if unknown:
        raise ManifestError(
            f"unknown target override field(s): {', '.join(unknown)}"
        )


def _to_plain(value: Any) -> Any:
    """Recursively turn a manifest value into JSON-encodable primitives."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _build(cls: Any, data: dict[str, Any]) -> Any:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in data:
            kwargs[f.name] = _convert(data[f.name], hints.get(f.name, Any))
    return cls(**kwargs)


def _convert(value: Any, hint: Any) -> Any:
    if value is None:
        return None
    if isinstance(hint, type) and is_dataclass(hint):
        return cast(Any, hint).from_dict(value)
    if isinstance(hint, type) and issubclass(hint, Enum):
        if isinstance(value, hint):
            return value
        try:
            return hint(value)
        except (ValueError, KeyError):
            raise ManifestError(f"invalid {hint.__name__} value: {value!r}") from None
    origin = get_origin(hint)
    if origin is None:
        return value
    args = get_args(hint)
    if origin in (Union, UnionType):
        for arg in args:
            try:
                return _convert(value, arg)
            except ManifestError:
                continue
        return value
    if origin in (list, set):
        elem = args[0] if args else Any
        if not isinstance(value, list):
            return value
        converted = [_convert(item, elem) for item in value]
        return set(converted) if origin is set else converted
    if origin is dict:
        return value
    return value
