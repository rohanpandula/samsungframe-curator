"""Analysis subsystem — provider-neutral contract surface (M002/S01).

This package ships the analysis boundary: the versioned JSON result schema
(:mod:`curator.analysis.schema`), the profile contract
(:mod:`curator.analysis.profiles`), compute-backend resolution
(:mod:`curator.analysis.compute`), model/runner contracts
(:mod:`curator.analysis.model`), and the provider ABC with capabilities/probe
(:mod:`curator.analysis.provider`). No heavy-ML runtime, no network.
"""

from __future__ import annotations

from curator.analysis.compute import ComputeBackend, resolve_backend, strict_device
from curator.analysis.errors import (
    AnalysisError,
    ComputeBackendError,
    SchemaVersionError,
)
from curator.analysis.model import (
    CpuReferenceRunner,
    ModelPrecision,
    ModelRunner,
    ModelSpec,
)
from curator.analysis.profiles import (
    KNOWN_STAGES,
    PROFILE_ORDER,
    AnalysisProfile,
    StageSpec,
    custom_profile,
    profile_order,
    profile_specs,
)
from curator.analysis.provider import (
    AnalysisCapabilities,
    AnalysisProvider,
    ComputeProbe,
    capability_requirements,
)
from curator.analysis.schema import (
    SCHEMA_VERSION,
    AnalysisMetadata,
    AnalysisResult,
    BoundingBox,
    ColorStory,
    CropSafety,
    Pairing,
    PerceptualRepresentation,
    Point,
    QualitySignals,
    Saliency,
    SchemaBase,
)

__all__ = [
    "AnalysisCapabilities",
    "AnalysisError",
    "AnalysisMetadata",
    "AnalysisProfile",
    "AnalysisProvider",
    "AnalysisResult",
    "BoundingBox",
    "ColorStory",
    "ComputeBackend",
    "ComputeBackendError",
    "ComputeProbe",
    "CpuReferenceRunner",
    "CropSafety",
    "KNOWN_STAGES",
    "ModelPrecision",
    "ModelRunner",
    "ModelSpec",
    "PROFILE_ORDER",
    "Pairing",
    "PerceptualRepresentation",
    "Point",
    "QualitySignals",
    "SCHEMA_VERSION",
    "Saliency",
    "SchemaBase",
    "SchemaVersionError",
    "StageSpec",
    "capability_requirements",
    "custom_profile",
    "profile_order",
    "profile_specs",
    "resolve_backend",
    "strict_device",
]
