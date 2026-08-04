"""Artifact validator / publish gate (M003/S02).

:class:`ArtifactValidator` gates whether a rendered artifact may be published:
it checks exact target dimensions, color mode (RGB), an sRGB color profile,
content SHA-256 vs the expected provenance, source-region math being in-bounds,
and that the source region is not unintentionally cropped. It returns a
JSON-serializable :class:`ValidationReport` with a per-check ``{name, passed,
reason}`` list and ``publishable`` (True only when every check passes), where
every failing check carries an actionable reason (R009).

The validator is pure and deterministic: the same inputs always produce the
same report.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from curator.artdirection.manifest import SourceRegion
from curator.errors import CuratorError
from curator.hashing import sha256_hex


class ValidationError(CuratorError):
    """Raised when an artifact cannot be decoded for validation."""


@dataclass(frozen=True)
class ValidationCheck:
    """The result of one validation check."""

    name: str
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationCheck:
        if isinstance(data, cls):
            return data
        return cls(
            name=data["name"],
            passed=data["passed"],
            reason=data.get("reason", ""),
        )


@dataclass(frozen=True)
class ValidationReport:
    """JSON-serializable validation outcome with per-check results.

    ``valid`` / ``publishable`` are the AND of all checks, so both are True only
    when every check passes. They are derived, not stored.
    """

    checks: list[ValidationCheck] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True when every check passes."""
        return all(check.passed for check in self.checks)

    @property
    def publishable(self) -> bool:
        """True when the artifact may be published (all checks pass)."""
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        return {
            "checks": [check.to_dict() for check in self.checks],
            "valid": self.valid,
            "publishable": self.publishable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationReport:
        if isinstance(data, cls):
            return data
        return cls(checks=[ValidationCheck.from_dict(c) for c in data["checks"]])


class ArtifactValidator:
    """Stateless validator gating publishability of a rendered artifact."""

    def validate(
        self,
        artifact_bytes: bytes,
        expected_sha256: str,
        target_dims: tuple[int, int],
        color_mode: str = "RGB",
        color_profile: str = "sRGB",
        source_region: SourceRegion | None = None,
    ) -> ValidationReport:
        """Validate *artifact_bytes* and return a :class:`ValidationReport`.

        Checks exact target dimensions, color mode, sRGB color profile, content
        SHA-256 vs *expected_sha256*, and (when given) that *source_region* is
        in-bounds and not unintentionally cropped against *target_dims*.
        """
        checks: list[ValidationCheck] = []
        try:
            with Image.open(io.BytesIO(artifact_bytes)) as img:
                img.load()
                checks.append(self._check_dimensions(img, target_dims))
                checks.append(self._check_color_mode(img, color_mode))
                checks.append(self._check_color_profile(img, color_profile))
        except Exception as exc:  # noqa: BLE001 - PIL raises varied types
            msg = (
                "cannot decode artifact "
                f"({type(exc).__name__}): {exc}"
            )
            checks.append(ValidationCheck("dimensions", False, msg))
            checks.append(ValidationCheck("color_mode", False, msg))
            checks.append(ValidationCheck("color_profile", False, msg))

        checks.append(self._check_hash(artifact_bytes, expected_sha256))

        if source_region is not None:
            checks.append(self._check_source_region(source_region, target_dims))
            checks.append(
                self._check_no_unintended_crop(source_region, target_dims)
            )

        return ValidationReport(checks=checks)

    def _check_dimensions(
        self, img: Image.Image, target_dims: tuple[int, int]
    ) -> ValidationCheck:
        tw, th = target_dims
        if (img.width, img.height) == (tw, th):
            return ValidationCheck("dimensions", True, "")
        if img.width != tw:
            return ValidationCheck(
                "dimensions",
                False,
                f"target width {img.width} != expected {tw}",
            )
        return ValidationCheck(
            "dimensions",
            False,
            f"target height {img.height} != expected {th}",
        )

    def _check_color_mode(
        self, img: Image.Image, color_mode: str
    ) -> ValidationCheck:
        if img.mode == color_mode:
            return ValidationCheck("color_mode", True, "")
        return ValidationCheck(
            "color_mode",
            False,
            f"color mode {img.mode!r} != expected {color_mode!r}",
        )

    def _check_color_profile(
        self, img: Image.Image, color_profile: str
    ) -> ValidationCheck:
        icc = img.info.get("icc_profile")
        if icc is None:
            return ValidationCheck("color_profile", True, "")
        if _is_srgb_profile(icc):
            return ValidationCheck("color_profile", True, "")
        return ValidationCheck(
            "color_profile",
            False,
            f"color profile is not {color_profile!r} "
            f"(embedded {len(icc)}-byte ICC profile)",
        )

    def _check_hash(
        self, artifact_bytes: bytes, expected_sha256: str
    ) -> ValidationCheck:
        actual = sha256_hex(artifact_bytes)
        if actual == expected_sha256:
            return ValidationCheck("hash", True, "")
        return ValidationCheck(
            "hash",
            False,
            f"sha256 mismatch: got {actual}, expected {expected_sha256}",
        )

    def _check_source_region(
        self, region: SourceRegion, target_dims: tuple[int, int]
    ) -> ValidationCheck:
        tw, th = target_dims
        x, y, w, h = region.x, region.y, region.w, region.h
        in_bounds = bool(
            0.0 <= x <= tw and 0.0 <= y <= th and 0.0 < w <= tw and 0.0 < h <= th
        )
        if in_bounds:
            return ValidationCheck("source_region", True, "")
        return ValidationCheck(
            "source_region",
            False,
            f"source region out of bounds: "
            f"(x={x}, y={y}, w={w}, h={h}) not within target {target_dims}",
        )

    def _check_no_unintended_crop(
        self, region: SourceRegion, target_dims: tuple[int, int]
    ) -> ValidationCheck:
        tw, th = target_dims
        x, y, w, h = region.x, region.y, region.w, region.h
        inside = bool(x >= 0 and y >= 0 and (x + w) <= tw and (y + h) <= th)
        if inside:
            return ValidationCheck("no_unintended_crop", True, "")
        return ValidationCheck(
            "no_unintended_crop",
            False,
            f"source region would crop outside target: "
            f"(x+w={x + w}, y+h={y + h}) exceeds {target_dims}",
        )


def _is_srgb_profile(icc: bytes) -> bool:
    """Return True when *icc* is the standard sRGB profile."""
    try:
        from PIL import ImageCms

        srgb = ImageCms.core.profile_tobytes(ImageCms.createProfile("sRGB"))
        if icc == srgb:
            return True
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        desc = (prof.profile_description or "").lower()
        return "srgb" in desc
    except Exception:  # noqa: BLE001 - profile introspection is best-effort
        return False
