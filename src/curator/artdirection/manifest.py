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
    """How source content is placed on the frame surface.

    Members serialize by ``.value``, so adding one is schema-compatible with the
    current :data:`MANIFEST_VERSION` — M010/S02's ``TRIPTYCH`` / ``QUAD`` needed
    no version bump. Adding a member *does* require a
    ``policy._TREATMENT_RANK`` entry; that module fails loudly at import time
    when one is missing.
    """

    SINGLE_FULLBLEED = "single_fullbleed"
    CONTAIN_MATTE = "contain_matte"
    PANORAMIC = "panoramic"
    SQUARE = "square"
    DIPTYCH = "diptych"
    TRIPTYCH = "triptych"
    QUAD = "quad"
    #: Arbitrary N within :data:`MAX_LAYOUT_SOURCES`, packed by weight (M010/S03).
    #: Deliberately absent from :data:`_TREATMENT_SOURCE_COUNT` — a variable
    #: source count is the whole point of it.
    PACKED = "packed"


_NESTED_DICT_FIELDS = {"background", "processing_intent"}

#: Maximum number of sources a single manifest may lay out (M010/S01).
#:
#: A **stated, revisable engineering default** (01-ROADMAP.md's "stated,
#: revisable engineering default" decision), *not* a researched legibility
#: ceiling — no source gives a real upper bound on how many images stay readable
#: on one frame. Over-cap manifests are rejected by
#: :meth:`ArtDirectionManifest.validate`, never silently truncated, and because
#: ``DeterministicRenderer._render`` calls ``validate()`` as its first statement
#: the cap fires before any image is decoded.
MAX_LAYOUT_SOURCES = 9

#: Treatments that occupy more than one cell of the output canvas (M010/S01).
#:
#: The single place the "does this treatment need more than one cell?" question
#: is answered — read by ``policy.materialize_manifest`` (which cell order to
#: record) and by ``renderer._render`` (which branch renders it). M010/S02 added
#: the two named N-up templates; M010/S03 added ``PACKED``, which needed no new
#: render branch — only a set member.
MULTI_CELL_TREATMENTS: frozenset[LayoutTreatment] = frozenset(
    {
        LayoutTreatment.DIPTYCH,
        LayoutTreatment.TRIPTYCH,
        LayoutTreatment.QUAD,
        LayoutTreatment.PACKED,
    }
)

#: How many sources each fixed-size named template requires, exactly (M010/S02).
#:
#: The single exact-count table both the policy engine
#: (``materialize_manifest`` -> :class:`ManifestError`) and the renderer
#: (``_multi_cell`` -> ``RenderError``) read, so "reject, never truncate" is
#: enforced from one place rather than restated per layer. A treatment absent
#: from this table has no fixed count — that is how a variable-width template
#: (M010/S03's ``PACKED``) opts out.
_TREATMENT_SOURCE_COUNT: dict[LayoutTreatment, int] = {
    LayoutTreatment.DIPTYCH: 2,
    LayoutTreatment.TRIPTYCH: 3,
    LayoutTreatment.QUAD: 4,
}

#: The one crop mode: scale to fill the cell and center-crop the overflow (M010/S05).
#:
#: Renders through ``renderer._center_crop_fill`` — the codebase's single crop,
#: already used by ``SINGLE_FULLBLEED`` — never a second cropping heuristic.
CROP_FILL = "fill"

#: The accepted vocabulary for :attr:`SourceRegion.crop` (M010/S05).
#:
#: ``None`` (and the empty string) mean **letterbox**: fit the source inside its
#: cell preserving aspect and pad the remainder with the manifest background.
#: That is the default every multi-region treatment has always used and the only
#: fit any manifest written before M010/S05 can carry. Anything outside this
#: frozenset is rejected by ``policy.materialize_manifest`` (:class:`ManifestError`)
#: and by ``renderer._multi_cell`` (``RenderError``) — an unrecognized fit
#: directive is never silently downgraded to a letterbox, because silently
#: discarding a caller's stated intent is the failure mode M010 exists to remove.
VALID_CROP_MODES: frozenset[str] = frozenset({CROP_FILL})


@dataclass(frozen=True)
class SourceRegion:
    """One source's cell on the output canvas, keyed by content identity.

    ``x`` / ``y`` / ``w`` / ``h`` are **output-canvas pixels** of the render
    target the manifest was materialized against (M010/S01) — floats for schema
    compatibility, but pixel counts, never fractions of the canvas. That is the
    unit its only reader already assumes (``ArtifactValidator`` compares these
    against the target dims) and the unit the renderer's paste box
    ``(bx, by, bw, bh)`` is expressed in, so a region is a paste box without
    conversion.

    An **all-zero** region (``x == y == w == h == 0.0``) means *unset — no
    geometry declared*, never "a zero-sized cell"; see :attr:`is_unset`. Every
    manifest persisted before M010 carries all-zero regions, so this reading is
    what keeps that history valid.

    ``crop`` is this cell's **fit** — how the source is placed inside the
    geometry, as opposed to what the geometry is. Its vocabulary is
    :data:`VALID_CROP_MODES`, with ``None``/``""`` meaning letterbox (M010/S05).
    The field has existed since M002; M010/S05 gives it its first production
    writer (``policy.materialize_manifest``) and its first production reader
    (``renderer._multi_cell``), rather than adding a parallel one.
    """

    source_sha256: str
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    crop: str | None = None

    @property
    def is_unset(self) -> bool:
        """True when this region declares no geometry (x, y, w and h all zero).

        Deliberately a property and never a field: :func:`_to_plain` walks
        ``dataclasses.fields``, so a property stays invisible to
        :meth:`to_dict` / :meth:`from_dict` and the persisted schema is
        unchanged (M010/S01).
        """
        return self.x == 0.0 and self.y == 0.0 and self.w == 0.0 and self.h == 0.0

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
        and string-keyed ``target_overrides``. M010/S01 adds three N-source
        invariants, all of which used to pass silently: the
        :data:`MAX_LAYOUT_SOURCES` cap, one region per source whenever
        ``regions`` is populated, and every region referencing a sha that is
        actually in ``sources``. Regions are deliberately *not* required to be
        uniformly set or uniformly unset — resolving a partially-populated
        manifest is render-time work (``packing.resolve_regions``), not a data
        model rule. Unknown *override* fields are rejected later by
        :meth:`resolved_for`.
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
        if len(self.sources) > MAX_LAYOUT_SOURCES:
            raise ManifestError(
                f"manifest has {len(self.sources)} source(s), over the "
                f"{MAX_LAYOUT_SOURCES}-source layout cap — an over-cap request is "
                f"rejected, never truncated"
            )
        if self.regions:
            if len(self.regions) != len(self.sources):
                raise ManifestError(
                    f"manifest has {len(self.sources)} source(s) but "
                    f"{len(self.regions)} region(s) — one region per source is required"
                )
            unknown = sorted({r.source_sha256 for r in self.regions} - set(self.sources))
            if unknown:
                raise ManifestError(
                    f"region references source(s) not in manifest: {unknown}"
                )
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
