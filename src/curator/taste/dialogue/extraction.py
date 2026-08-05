"""Attribute/polarity extraction provider boundary for taste dialogue (M008/S02).

A :class:`TasteObservation`'s verbatim text is turned into a closed set of
:data:`CONTROLLED_VOCABULARY` attribute tags plus a :class:`Polarity` by an
:class:`ExtractionProvider`. Providers advertise an
:class:`ExtractionCapabilities` contract and either produce an
:class:`ExtractionResult` or raise :class:`ExtractionUnavailableError` — the
caller-facing helper :func:`extract_or_unavailable` converts that into ``None``
and *never* falls back to keyword guessing (the no-silent-fallback contract).

The cloud realization (:class:`CloudExtractionProvider`) is air-gapped: it talks
to :class:`SyntheticExtractionRuntime`, a deterministic mock, and sends only the
verbatim text plus a downscaled image thumbnail and policy-gated metadata —
never originals, secrets, GPS, or faces (M006 privacy shape).
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from curator.errors import CuratorError
from curator.providers.cloud import (
    COMPONENT_FACES,
    COMPONENT_FULL_RESOLUTION,
    COMPONENT_GPS,
    ExclusionPolicy,
)
from curator.providers.privacy import Disclosure, MachineLeaves
from curator.taste.dialogue.observation import ImageRef, Polarity, TasteObservation

#: Closed, deterministic set of attribute tags extraction may emit.
CONTROLLED_VOCABULARY: tuple[str, ...] = (
    "negative-space",
    "muted-palette",
    "lone-subject",
    "symmetry",
    "warm-tones",
    "high-contrast",
    "breathing-room",
    "texture",
    "motion",
    "repetition",
    "minimal",
    "dense",
    "geometric",
    "organic",
    "nostalgic",
    "quiet",
    "heavy",
    "light",
)

#: Ordered keyword -> attribute-tag rules for the deterministic mock runtime.
_ATTRIBUTE_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("quiet",), ("negative-space", "muted-palette", "quiet")),
    (("negative space",), ("negative-space",)),
    (("lone",), ("lone-subject",)),
    (("symmetry", "symmetric"), ("symmetry",)),
    (("warm",), ("warm-tones",)),
    (("contrast",), ("high-contrast",)),
    (("breathing",), ("breathing-room",)),
    (("texture", "textured"), ("texture",)),
    (("motion", "movement"), ("motion",)),
    (("repetition", "repeating"), ("repetition",)),
    (("minimal",), ("minimal",)),
    (("busy", "crowded", "chaotic"), ("dense",)),
    (("geometric",), ("geometric",)),
    (("organic",), ("organic",)),
    (("nostalgic",), ("nostalgic",)),
    (("heavy",), ("heavy",)),
    (("light",), ("light",)),
)

#: DISLIKE cues are checked before LIKE cues so "don't like" reads as dislike.
_DISLIKE_CUES = ("hate", "don't", "dislike", "ugly", "awful", "bad")
_LIKE_CUES = ("love", "like", "beautiful", "great", "gorgeous")


class ExtractionUnavailableError(CuratorError):
    """Raised when extraction is unavailable (disabled, unconfigured, or down).

    The ``reason`` explains why, so callers can log or surface it without
    guessing from the observation text.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ExtractionProbe:
    """A point-in-time health/probe result for one extraction provider."""

    ok: bool
    kind: str
    message: str | None = None


@dataclass(frozen=True)
class ExtractionCapabilities:
    """Declared capabilities of an extraction provider."""

    enabled: bool
    kind: str
    supports_local: bool
    supports_cloud: bool
    disclosure_available: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return {
            "enabled": self.enabled,
            "kind": self.kind,
            "supports_local": self.supports_local,
            "supports_cloud": self.supports_cloud,
            "disclosure_available": self.disclosure_available,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ExtractionCapabilities:
        """Build an :class:`ExtractionCapabilities` from a dict (lenient)."""
        if isinstance(data, ExtractionCapabilities):
            return data
        fields = dict(data)
        return cls(
            enabled=bool(fields.get("enabled", False)),
            kind=str(fields.get("kind", "local")),
            supports_local=bool(fields.get("supports_local", False)),
            supports_cloud=bool(fields.get("supports_cloud", False)),
            disclosure_available=bool(fields.get("disclosure_available", False)),
        )


@dataclass(frozen=True)
class ExtractionResult:
    """The outcome of one extraction: attributes + polarity + confidence.

    ``attributes`` is a subset of :data:`CONTROLLED_VOCABULARY`; ``verbatim`` is
    always the exact input text, and ``provider`` names which provider produced it.
    """

    attributes: list[str]
    polarity: Polarity
    confidence: float
    verbatim: str
    provider: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence!r} out of range [0.0, 1.0]")
        unknown = [a for a in self.attributes if a not in CONTROLLED_VOCABULARY]
        if unknown:
            raise ValueError(
                f"attributes not in controlled vocabulary: {sorted(unknown)}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return {
            "attributes": list(self.attributes),
            "polarity": self.polarity.value,
            "confidence": self.confidence,
            "verbatim": self.verbatim,
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ExtractionResult:
        """Build an :class:`ExtractionResult` from a dict (lenient)."""
        if isinstance(data, ExtractionResult):
            return data
        fields = dict(data)
        return cls(
            attributes=list(fields.get("attributes", ())),
            polarity=_coerce_polarity(
                fields.get("polarity", Polarity.CONFLICTED.value)
            ),
            confidence=float(fields.get("confidence", 0.5)),
            verbatim=str(fields.get("verbatim", "")),
            provider=str(fields.get("provider", "")),
        )


class ExtractionProvider(ABC):
    """Abstract attribute/polarity extraction provider.

    Concrete implementations advertise :meth:`capabilities` and a live
    :meth:`probe`, and implement :meth:`extract` to turn a
    :class:`TasteObservation` into an :class:`ExtractionResult`, raising
    :class:`ExtractionUnavailableError` when they cannot run.
    """

    @abstractmethod
    def capabilities(self) -> ExtractionCapabilities:
        """Return this provider's declared capabilities."""

    @abstractmethod
    def probe(self) -> ExtractionProbe:
        """Return a live health probe for the current backend."""

    @abstractmethod
    def extract(self, observation: TasteObservation) -> ExtractionResult:
        """Extract attributes/polarity from *observation*, or raise."""


class SyntheticExtractionRuntime:
    """Deterministic mock cloud extraction runtime (acceptance, no network).

    Records every payload it receives so tests can assert the provider never
    sent originals or secrets. :meth:`extract` runs a fixed keyword rule engine
    over the verbatim and returns ``(attributes, polarity, confidence)``.
    """

    def __init__(self, engine_version: str = "extraction-1.0.0") -> None:
        self.engine_version = engine_version
        self.down: bool = False
        self.received_payloads: list[tuple[str, dict[str, Any]]] = []

    def set_down(self, down: bool) -> None:
        """Simulate a cloud outage when *down* is True."""
        self.down = down

    def extract(
        self, verbatim: str, image_context: dict[str, Any]
    ) -> tuple[list[str], Polarity, float]:
        """Extract from *verbatim* and record exactly what was received."""
        if self.down:
            raise ExtractionUnavailableError("cloud extraction runtime is down")
        self.received_payloads.append((verbatim, dict(image_context)))
        return self._rule_engine(verbatim)

    @staticmethod
    def _rule_engine(verbatim: str) -> tuple[list[str], Polarity, float]:
        lowered = verbatim.lower()
        attributes: list[str] = []
        seen: set[str] = set()
        for keywords, tags in _ATTRIBUTE_RULES:
            if any(keyword in lowered for keyword in keywords):
                for tag in tags:
                    if tag not in seen:
                        seen.add(tag)
                        attributes.append(tag)
        if any(cue in lowered for cue in _DISLIKE_CUES):
            polarity, confidence = Polarity.DISLIKE, 0.85
        elif any(cue in lowered for cue in _LIKE_CUES):
            polarity, confidence = Polarity.LIKE, 0.9
        else:
            polarity, confidence = Polarity.CONFLICTED, 0.3
        return attributes, polarity, confidence


class CloudExtractionProvider(ExtractionProvider):
    """Privacy-preserving cloud extraction wrapping a runtime + exclusion policy."""

    def __init__(
        self,
        runtime: SyntheticExtractionRuntime | None = None,
        policy: ExclusionPolicy | None = None,
        provider_name: str = "synthetic-cloud",
        enabled: bool = True,
    ) -> None:
        self.runtime = (
            runtime if runtime is not None else SyntheticExtractionRuntime()
        )
        self.policy = policy if policy is not None else ExclusionPolicy()
        self.provider_name = provider_name
        self.enabled = enabled

    def capabilities(self) -> ExtractionCapabilities:
        return ExtractionCapabilities(
            enabled=self.enabled,
            kind="cloud",
            supports_local=False,
            supports_cloud=True,
            disclosure_available=True,
        )

    def probe(self) -> ExtractionProbe:
        ok = self.enabled and not self.runtime.down
        return ExtractionProbe(
            ok=ok,
            kind="cloud",
            message=(
                "cloud extraction runtime available"
                if ok
                else "cloud extraction runtime unavailable"
            ),
        )

    def disclosure(self) -> Disclosure:
        """Return the privacy disclosure reflecting the live exclusion policy."""
        return extraction_default_disclosure(
            self.provider_name, exclusions=self._declared_exclusions()
        )

    def _declared_exclusions(self) -> list[str]:
        declared: set[str] = set()
        for values in self.policy.per_source.values():
            declared.update(values)
        for values in self.policy.per_image.values():
            declared.update(values)
        return sorted(declared)

    def extract(self, observation: TasteObservation) -> ExtractionResult:
        """Extract *observation*, sending ONLY verbatim + downscaled thumbnails.

        Each referenced image is reduced to a downscaled derivative and
        policy-allowed metadata (faces/GPS/full-resolution components are gated
        by the :class:`ExclusionPolicy` and dropped before sending). The runtime
        result is wrapped with the verbatim preserved exactly.
        """
        if not self.enabled:
            raise ExtractionUnavailableError("cloud extraction disabled in config")
        if self.runtime.down:
            raise ExtractionUnavailableError("cloud extraction runtime is down")
        image_contexts = [
            ctx
            for ctx in (self._image_context(ref) for ref in observation.images)
            if ctx is not None
        ]
        attributes, polarity, confidence = self.runtime.extract(
            observation.verbatim, {"images": image_contexts}
        )
        return ExtractionResult(
            attributes=attributes,
            polarity=polarity,
            confidence=confidence,
            verbatim=observation.verbatim,
            provider=self.provider_name,
        )

    def _image_context(self, ref: ImageRef) -> dict[str, Any] | None:
        """Build the privacy-preserving per-image context for *ref*.

        Returns ``None`` when the thumbnail is missing/unreadable so nothing is
        sent for that image. The derivative is always downscaled, never the
        original bytes.
        """
        if ref.thumb_path is None:
            return None
        try:
            derivative = _downscale_thumbnail(Path(ref.thumb_path).read_bytes())
        except (OSError, ValueError):
            return None
        asset_id = ref.sha256
        source_id = None
        ctx: dict[str, Any] = {
            "sha256": asset_id,
            "width": derivative["width"],
            "height": derivative["height"],
            "thumb_bytes": derivative["bytes"],
        }
        if self.policy.allows(source_id, asset_id, COMPONENT_FACES):
            ctx["faces_present"] = _faces_present(derivative["pil"])
        if self.policy.allows(source_id, asset_id, COMPONENT_GPS):
            gps = _extract_gps(derivative["pil"])
            if gps is not None:
                ctx["gps"] = gps
        if self.policy.allows(source_id, asset_id, COMPONENT_FULL_RESOLUTION):
            ctx["source_resolution"] = [
                derivative["source_width"],
                derivative["source_height"],
            ]
        return ctx


class LocalExtractionSlot(ExtractionProvider):
    """Reserved local extraction slot; a local model is not yet configured."""

    def capabilities(self) -> ExtractionCapabilities:
        return ExtractionCapabilities(
            enabled=False,
            kind="local",
            supports_local=True,
            supports_cloud=False,
            disclosure_available=False,
        )

    def probe(self) -> ExtractionProbe:
        return ExtractionProbe(
            ok=False, kind="local", message="local extraction model not configured"
        )

    def extract(self, observation: TasteObservation) -> ExtractionResult:
        raise ExtractionUnavailableError("local model not configured")


def resolve_extraction_provider(
    config: dict[str, Any] | None,
) -> ExtractionProvider | None:
    """Pick an extraction provider from *config*, or ``None`` when nothing is enabled.

    *config* may carry ``{"enabled": True, "provider": "cloud"|"local"}`` and an
    optional ``{"policy": ExclusionPolicy}`` for the cloud provider.
    """
    if not config:
        return None
    if not config.get("enabled"):
        return None
    kind = config.get("provider")
    if kind == "cloud":
        policy = config.get("policy")
        if isinstance(policy, ExclusionPolicy):
            return CloudExtractionProvider(enabled=True, policy=policy)
        return CloudExtractionProvider(enabled=True)
    if kind == "local":
        return LocalExtractionSlot()
    return None


def extract_or_unavailable(
    provider: ExtractionProvider | None, observation: TasteObservation
) -> ExtractionResult | None:
    """Run *provider* if available; return ``None`` when it is not.

    Only :class:`ExtractionUnavailableError` is swallowed; the result is never
    derived by guessing keywords from the observation (no-silent-fallback).
    """
    if provider is None:
        return None
    try:
        return provider.extract(observation)
    except ExtractionUnavailableError:
        return None


def extraction_default_disclosure(
    provider: str = "synthetic-cloud",
    exclusions: list[str] | None = None,
) -> Disclosure:
    """Canonical privacy :class:`Disclosure` for cloud taste extraction.

    What leaves is the verbatim text, a downscaled thumbnail, and minimal
    metadata; originals, secrets, credentials, GPS, and face data never leave.
    """
    return Disclosure(
        statement=(
            "When cloud taste-extraction runs, only the user's verbatim text, a "
            "low-resolution downscaled image thumbnail, and minimal "
            "non-identifying metadata are sent to the cloud provider for "
            "attribute and polarity extraction. Original images, secrets, "
            "TV/Home Assistant credentials, GPS, and face data never leave the "
            "device."
        ),
        leaves_machine=MachineLeaves(
            payload_types=["verbatim", "downscaled_thumbnail"],
            metadata_scope=["sha256", "dimensions"],
            never=[
                "original_image",
                "secrets",
                "credentials",
                "tv_ha_credentials",
                "gps",
                "faces",
            ],
        ),
        exclusions=list(exclusions or []),
        provider=provider,
    )


# -- derivative / metadata helpers (mirror CloudAnalysisProvider, M006) -----


def _downscale_thumbnail(data: bytes, max_edge: int = 512) -> dict[str, Any]:
    """Return a deterministic downscaled JPEG derivative dict for *data*."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            pil = ImageOps.exif_transpose(im).convert("RGB")
            source_width, source_height = pil.size
    except Exception as exc:
        raise ValueError(
            f"failed to decode extraction thumbnail ({type(exc).__name__})"
        ) from exc
    scale = min(1.0, max_edge / max(source_width, source_height))
    width = max(1, int(round(source_width * scale)))
    height = max(1, int(round(source_height * scale)))
    thumb = (
        pil.resize((width, height), Image.Resampling.LANCZOS)
        if (width, height) != (source_width, source_height)
        else pil
    )
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return {
        "bytes": buf.getvalue(),
        "pil": thumb,
        "width": width,
        "height": height,
        "source_width": source_width,
        "source_height": source_height,
    }


def _faces_present(pil: Image.Image) -> bool:
    """Identity-free skin-tone presence hint (never identifies anyone)."""
    small = pil.convert("RGB").resize((48, 48))
    arr = np.asarray(small, dtype=int).reshape(-1, 3)
    if arr.shape[0] == 0:
        return False
    r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
    skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (r - g > 15)
    return bool(int(skin.sum()) / arr.shape[0] > 0.02)


def _extract_gps(pil: Image.Image) -> dict[str, float] | None:
    """Return ``{"lat": .., "lon": ..}`` from EXIF GPS if present, else None."""
    try:
        gps_ifd = pil.getexif().get_ifd(0x8825)
    except Exception:
        return None
    if not gps_ifd:
        return None
    lat = gps_ifd.get(2)
    lon = gps_ifd.get(4)
    if lat is None or lon is None:
        return None
    lat_ref = gps_ifd.get(1)
    lon_ref = gps_ifd.get(3)
    return {
        "lat": (-1.0 if lat_ref in ("S",) else 1.0) * _dms_to_degrees(lat),
        "lon": (-1.0 if lon_ref in ("W",) else 1.0) * _dms_to_degrees(lon),
    }


def _dms_to_degrees(value: Any) -> float:
    """Convert an EXIF DMS tuple ``(deg, min, sec)`` to decimal degrees."""
    if isinstance(value, (list, tuple)):
        deg, *rest = value
        minute, second = (rest + [0, 0])[:2]
        return float(deg) + float(minute) / 60.0 + float(second) / 3600.0
    return float(value)


def _coerce_polarity(value: Any) -> Polarity:
    """Return *value* as a :class:`Polarity`, coercing strings or enum instances."""
    if isinstance(value, Polarity):
        return value
    try:
        return Polarity(value)
    except ValueError as exc:
        raise ValueError(f"invalid polarity {value!r}") from exc


__all__ = [
    "CONTROLLED_VOCABULARY",
    "CloudExtractionProvider",
    "ExtractionCapabilities",
    "ExtractionProbe",
    "ExtractionProvider",
    "ExtractionResult",
    "ExtractionUnavailableError",
    "LocalExtractionSlot",
    "SyntheticExtractionRuntime",
    "extract_or_unavailable",
    "extraction_default_disclosure",
    "resolve_extraction_provider",
]
