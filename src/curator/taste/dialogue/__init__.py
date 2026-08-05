"""Taste dialogue subsystem (M008): interactive art discussion.

Third-party images surfaced during a dialogue are retained as evidence only
(thumbnail + content hash) and promoted into the catalog via an explicit action
— see :mod:`curator.taste.dialogue.retention`.
"""

from __future__ import annotations

from curator.taste.dialogue.extraction import (
    CONTROLLED_VOCABULARY,
    CloudExtractionProvider,
    ExtractionCapabilities,
    ExtractionProbe,
    ExtractionProvider,
    ExtractionResult,
    ExtractionUnavailableError,
    LocalExtractionSlot,
    SyntheticExtractionRuntime,
    extract_or_unavailable,
    extraction_default_disclosure,
    resolve_extraction_provider,
)
from curator.taste.dialogue.observation import (
    ImageRef,
    ObservationError,
    Polarity,
    TasteObservation,
    create_observation,
)
from curator.taste.dialogue.retention import (
    retain_ephemeral,
    retention_policy,
    save_to_catalog,
)
from curator.taste.dialogue.session import TasteSession
from curator.taste.dialogue.store import ObservationStore, SessionStore

__all__ = [
    "CONTROLLED_VOCABULARY",
    "CloudExtractionProvider",
    "ExtractionCapabilities",
    "ExtractionProbe",
    "ExtractionProvider",
    "ExtractionResult",
    "ExtractionUnavailableError",
    "ImageRef",
    "LocalExtractionSlot",
    "ObservationError",
    "Polarity",
    "SyntheticExtractionRuntime",
    "TasteObservation",
    "create_observation",
    "extract_or_unavailable",
    "extraction_default_disclosure",
    "resolve_extraction_provider",
    "retain_ephemeral",
    "retention_policy",
    "save_to_catalog",
    "TasteSession",
    "ObservationStore",
    "SessionStore",
]
