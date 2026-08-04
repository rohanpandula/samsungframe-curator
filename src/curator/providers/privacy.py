"""Privacy disclosure for cloud/hybrid analysis providers (M006/S01).

:class:`Disclosure` is a frozen, JSON-serializable plain-language statement plus a
machine-readable :attr:`Disclosure.leaves_machine` breakdown of exactly what data
leaves the device when a cloud provider runs, what metadata is shared, and what is
*never* sent (originals, secrets, TV/HA credentials, GPS, faces-per-policy). It
mirrors the :mod:`curator.analysis.provider` convention of frozen dataclasses with
``to_dict`` / ``from_dict``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Payloads that may legitimately leave the device on a cloud analysis call.
_DEFAULT_PAYLOAD_TYPES = ["downscaled_derivative"]

#: Metadata scope shared with the cloud provider (identity + rendering hints only).
_DEFAULT_METADATA_SCOPE = ["asset_id", "source_id", "profile", "dimensions"]

#: What is never sent, per privacy policy (originals/secrets/credentials/GPS/faces).
_DEFAULT_NEVER = [
    "original_image",
    "secrets",
    "credentials",
    "tv_ha_credentials",
    "gps",
    "faces",
]


@dataclass(frozen=True)
class MachineLeaves:
    """Structured record of what data leaves the machine on a cloud call."""

    payload_types: list[str] = field(default_factory=list)
    metadata_scope: list[str] = field(default_factory=list)
    never: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return {
            "payload_types": list(self.payload_types),
            "metadata_scope": list(self.metadata_scope),
            "never": list(self.never),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MachineLeaves:
        """Build a :class:`MachineLeaves` from a dict (lenient on missing keys)."""
        return cls(
            payload_types=list(data.get("payload_types", [])),
            metadata_scope=list(data.get("metadata_scope", [])),
            never=list(data.get("never", [])),
        )


@dataclass(frozen=True)
class Disclosure:
    """Plain-language + structured privacy disclosure for one cloud provider."""

    statement: str
    leaves_machine: MachineLeaves = field(default_factory=MachineLeaves)
    exclusions: list[str] = field(default_factory=list)
    provider: str = "cloud"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-encodable dict."""
        return {
            "statement": self.statement,
            "leaves_machine": self.leaves_machine.to_dict(),
            "exclusions": list(self.exclusions),
            "provider": self.provider,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Disclosure:
        """Build a :class:`Disclosure` from a dict (lenient on missing keys)."""
        return cls(
            statement=str(data.get("statement", "")),
            leaves_machine=MachineLeaves.from_dict(
                data.get("leaves_machine", {}) or {}
            ),
            exclusions=list(data.get("exclusions", [])),
            provider=str(data.get("provider", "cloud")),
        )


def default_disclosure(
    provider: str = "cloud",
    exclusions: list[str] | None = None,
) -> Disclosure:
    """Build the canonical privacy :class:`Disclosure` for a cloud provider.

    *exclusions* are the per-source/per-image exclusions currently in effect and
    are surfaced in the disclosure so it always reflects the live policy.
    """
    return Disclosure(
        statement=(
            "When cloud analysis runs, only a low-resolution downscaled derivative "
            "and minimal non-identifying metadata are sent to the cloud provider for "
            "semantic, composition, pairing, and taste stages. Original images, "
            "secrets, TV/Home Assistant credentials, GPS, and face data never leave "
            "the device."
        ),
        leaves_machine=MachineLeaves(
            payload_types=list(_DEFAULT_PAYLOAD_TYPES),
            metadata_scope=list(_DEFAULT_METADATA_SCOPE),
            never=list(_DEFAULT_NEVER),
        ),
        exclusions=list(exclusions or []),
        provider=provider,
    )
