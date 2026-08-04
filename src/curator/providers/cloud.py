"""Cloud/hybrid analysis provider + privacy-preserving exclusion policy (M006/S01).

:class:`ExclusionPolicy` gates which per-image / per-source components (faces, GPS,
full resolution, or everything via ``all``) may be sent off-device.
:class:`SyntheticCloudAnalysisRuntime` is a deterministic mock cloud backend used by
acceptance/tests — it records every payload it receives so callers can assert the
provider NEVER sent originals or secrets. :class:`CloudAnalysisProvider` wraps the
runtime + policy: it always emits a downscaled derivative and only policy-allowed
metadata, and raises :class:`ProviderOutageError` when the runtime is down (R017).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from PIL import Image, ImageOps

from curator.analysis.compute import ComputeBackend
from curator.analysis.errors import AnalysisError
from curator.analysis.profiles import AnalysisProfile
from curator.analysis.provider import (
    AnalysisCapabilities,
    AnalysisProvider,
    ComputeProbe,
)
from curator.analysis.schema import SCHEMA_VERSION, AnalysisResult
from curator.providers.privacy import Disclosure, default_disclosure

#: ComputeBackend has no ``CLOUD`` member (M002 enums cover local devices only);
#: the cloud provider represents its backend with the opaque ``"cloud"`` id.
_CLOUD_BACKEND = cast(ComputeBackend, "cloud")

#: Stages served by the cloud provider (R017 hybrid split).
CLOUD_STAGES = frozenset({"semantic", "composition", "pairing", "taste"})

#: Payload components a privacy policy may exclude for a source/asset.
COMPONENT_ALL = "all"
COMPONENT_FULL_RESOLUTION = "full_resolution"
COMPONENT_FACES = "faces"
COMPONENT_GPS = "gps"


class ProviderOutageError(AnalysisError):
    """Raised when the cloud provider/runtime is down (R017 outage degradation)."""


@dataclass(frozen=True)
class ExclusionPolicy:
    """Per-source and per-image exclusion policy for cloud payloads.

    ``per_source`` maps a ``source_id`` to the set of excluded components and
    ``per_image`` maps an ``asset_id`` to the same. An ``all`` exclusion means the
    whole payload for that source/asset is blocked.
    """

    per_source: Mapping[str, frozenset[str]] = field(default_factory=dict)
    per_image: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def exclusions_for(
        self, source_id: str | None, asset_id: str
    ) -> frozenset[str]:
        """Return the merged exclusion set applicable to ``(source_id, asset_id)``."""
        merged: set[str] = set()
        if source_id is not None and source_id in self.per_source:
            merged.update(self.per_source[source_id])
        if asset_id in self.per_image:
            merged.update(self.per_image[asset_id])
        return frozenset(merged)

    def allows(
        self, source_id: str | None, asset_id: str, payload_component: str
    ) -> bool:
        """Return True if *payload_component* may be sent for the given source/asset.

        An ``all`` exclusion blocks every component; otherwise the component is
        blocked only when it appears in the merged per-source/per-image set.
        """
        exclusions = self.exclusions_for(source_id, asset_id)
        if COMPONENT_ALL in exclusions:
            return False
        return payload_component not in exclusions


class SyntheticCloudAnalysisRuntime:
    """Deterministic mock cloud analysis runtime (acceptance, no network)."""

    def __init__(self, engine_version: str = "cloud-1.0.0") -> None:
        self.engine_version = engine_version
        self.down: bool = False
        self.received_payloads: list[tuple[bytes, dict[str, Any]]] = []

    def set_down(self, down: bool) -> None:
        """Simulate a cloud outage when *down* is True."""
        self.down = down

    def analyze(self, derivative_bytes: bytes, meta: dict[str, Any]) -> AnalysisResult:
        """Analyze a delivered derivative and return a deterministic result.

        The exact bytes/meta it received are recorded (for privacy assertions).
        """
        if self.down:
            raise ProviderOutageError("cloud analysis runtime is down")
        self.received_payloads.append((derivative_bytes, dict(meta)))
        return AnalysisResult.from_dict(
            {
                "asset_id": str(meta.get("asset_id", "asset")),
                "schema_version": SCHEMA_VERSION,
                "metadata": {
                    "profile": meta.get("profile", ""),
                    "compute_backend": "cloud",
                    "engine_version": self.engine_version,
                    "deterministic": False,
                },
            }
        )

    def capabilities(self) -> AnalysisCapabilities:
        """Report the cloud provider's declared capabilities (R017)."""
        return AnalysisCapabilities(
            profiles=frozenset(AnalysisProfile),
            backends=frozenset({_CLOUD_BACKEND}),
            stages=CLOUD_STAGES,
            air_gapped=False,
            deterministic=False,
        )


class CloudAnalysisProvider(AnalysisProvider):
    """Privacy-preserving cloud analysis provider wrapping a runtime + exclusion policy."""

    def __init__(
        self,
        runtime: SyntheticCloudAnalysisRuntime | None = None,
        policy: ExclusionPolicy | None = None,
        provider_name: str = "synthetic-cloud",
    ) -> None:
        self.runtime = (
            runtime if runtime is not None else SyntheticCloudAnalysisRuntime()
        )
        self.policy = policy if policy is not None else ExclusionPolicy()
        self.provider_name = provider_name

    def capabilities(self) -> AnalysisCapabilities:
        return self.runtime.capabilities()

    def probe(self) -> ComputeProbe:
        return ComputeProbe(
            ok=not self.runtime.down,
            backend=_CLOUD_BACKEND,
            latency_ms=0.0,
            available_backends=frozenset({_CLOUD_BACKEND}),
            message=(
                "cloud analysis runtime available"
                if not self.runtime.down
                else "cloud analysis runtime down"
            ),
        )

    def disclosure(self) -> Disclosure:
        """Return the privacy disclosure reflecting the live exclusion policy."""
        return default_disclosure(
            self.provider_name, exclusions=self._declared_exclusions()
        )

    def _declared_exclusions(self) -> list[str]:
        declared: set[str] = set()
        for values in self.policy.per_source.values():
            declared.update(values)
        for values in self.policy.per_image.values():
            declared.update(values)
        return sorted(declared)

    def analyze(
        self,
        source_bytes: bytes | bytearray,
        profile: AnalysisProfile = AnalysisProfile.BALANCED,
        asset_id: str | None = None,
        source_id: str | None = None,
        allowed_components: list[str] | set[str] | None = None,
    ) -> AnalysisResult:
        """Analyze *source_bytes*, sending ONLY a downscaled derivative + allowed metadata.

        The derivative is always downscaled (never the original), and GPS/faces/
        full-resolution metadata are included only when the :class:`ExclusionPolicy`
        and *allowed_components* both permit them.
        """
        data = bytes(source_bytes)
        derivative = self._downscale(data)
        asset_id = asset_id or "asset-" + hashlib.sha256(data).hexdigest()[:16]

        meta: dict[str, Any] = {
            "asset_id": asset_id,
            "source_id": source_id,
            "profile": profile.value,
            "width": derivative["width"],
            "height": derivative["height"],
        }

        def want(component: str) -> bool:
            if allowed_components is not None and component not in allowed_components:
                return False
            return self.policy.allows(source_id, asset_id, component)

        if want(COMPONENT_GPS):
            gps = self._extract_gps(derivative["pil"])
            if gps is not None:
                meta["gps"] = gps
        if want(COMPONENT_FACES):
            meta["faces"] = self._faces_present(derivative["pil"])
        if want(COMPONENT_FULL_RESOLUTION):
            meta["source_resolution"] = [
                derivative["source_width"],
                derivative["source_height"],
            ]
        return self.runtime.analyze(derivative["bytes"], meta)

    # -- derivative / metadata helpers ---------------------------------------

    @staticmethod
    def _downscale(data: bytes, max_edge: int = 512) -> dict[str, Any]:
        try:
            with Image.open(io.BytesIO(data)) as im:
                pil = ImageOps.exif_transpose(im).convert("RGB")
                source_width, source_height = pil.size
        except Exception as exc:
            raise AnalysisError(
                f"failed to decode cloud derivative from {len(data)} bytes "
                f"({type(exc).__name__})"
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

    @staticmethod
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

    @staticmethod
    def _faces_present(pil: Image.Image) -> bool:
        """Identity-free skin-tone presence hint (never identifies anyone)."""
        small = pil.convert("RGB").resize((48, 48))
        arr = np.asarray(small, dtype=int).reshape(-1, 3)
        if arr.shape[0] == 0:
            return False
        r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
        skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r > b) & (r - g > 15)
        return bool(int(skin.sum()) / arr.shape[0] > 0.02)


def _dms_to_degrees(value: Any) -> float:
    """Convert an EXIF DMS tuple ``(deg, min, sec)`` to decimal degrees."""
    if isinstance(value, (list, tuple)):
        deg, *rest = value
        minute, second = (rest + [0, 0])[:2]
        return float(deg) + float(minute) / 60.0 + float(second) / 3600.0
    return float(value)
