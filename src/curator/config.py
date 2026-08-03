"""Six-axis configuration skeleton (R022).

Configuration is read from ``CURATOR_*`` environment variables via pydantic-settings.
Flat settings (``CURATOR_DATA_ROOT``) are top level; per-axis overrides use the nested
delimiter (``CURATOR_SOURCE__TYPE``).

Only the **source** axis is populated in S01 (path + type=local). The other five axes
are typed placeholder models with sensible defaults — they exist so downstream slices
can add fields without bespoke axis code paths, and so every valid axis combination
travels through the same catalog / analysis / manifest / renderer / job-journal surface
(R022).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SourceAxis(BaseModel):
    """Source of incoming art/photos. In S01 the only supported type is `local`."""

    model_config = ConfigDict(extra="ignore")

    path: Path = Field(default_factory=lambda: Path.home() / "Pictures")
    type: str = "local"
    poll_interval_seconds: float = 300.0


class IntelligenceProviderAxis(BaseModel):
    """Local/cloud/hybrid inference provider. Placeholder in S01."""

    model_config = ConfigDict(extra="ignore")

    provider: str = "local"
    model: str = ""


class InterfaceAxis(BaseModel):
    """Web UI / CLI / API. Placeholder in S01."""

    model_config = ConfigDict(extra="ignore")

    mode: str = "cli"


class RuntimeAxis(BaseModel):
    """One-shot / scheduled / watcher execution. Placeholder in S01."""

    model_config = ConfigDict(extra="ignore")

    mode: str = "one-shot"


class RenderTargetAxis(BaseModel):
    """1080p / 4K / custom output profile. Placeholder in S01."""

    model_config = ConfigDict(extra="ignore")

    profile: str = "1080p"
    width: int = 1920
    height: int = 1080


class DestinationAxis(BaseModel):
    """Filesystem / static URL / Samsung Art API / HA-coordinated Samsung. Placeholder."""

    model_config = ConfigDict(extra="ignore")

    type: str = "filesystem"


class CuratorConfig(BaseSettings):
    """Root configuration object for the Curator pipeline (R022)."""

    model_config = SettingsConfigDict(
        env_prefix="CURATOR_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    data_root: Path = Field(default_factory=lambda: Path.home() / ".curator")

    source: SourceAxis = SourceAxis()
    intelligence_provider: IntelligenceProviderAxis = IntelligenceProviderAxis()
    interface: InterfaceAxis = InterfaceAxis()
    runtime: RuntimeAxis = RuntimeAxis()
    render_target: RenderTargetAxis = RenderTargetAxis()
    destination: DestinationAxis = DestinationAxis()
