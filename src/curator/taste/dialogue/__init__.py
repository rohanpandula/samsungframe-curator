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
from curator.taste.dialogue.profile import (
    ColdStartSeeder,
    EvidenceRef,
    HistoryDecision,
    ProfileBuilder,
    ProfileEvent,
    ProfileStore,
    TasteClaim,
    TasteProfile,
    WhatILearned,
)
from curator.taste.dialogue.retention import (
    retain_ephemeral,
    retention_policy,
    save_to_catalog,
)
from curator.taste.dialogue.session import TasteSession
from curator.taste.dialogue.store import ObservationStore, SessionStore
from curator.taste.dialogue.upstream import (
    CORROBORATING_WEIGHT,
    ProfileCitation,
    RankExplanation,
    citations_for,
    explain_rank,
    familiar_surprising_dimensions,
    pairing_rationale,
    profile_dimensions,
    profile_fit,
)

__all__ = [
    "CONTROLLED_VOCABULARY",
    "CORROBORATING_WEIGHT",
    "CloudExtractionProvider",
    "ColdStartSeeder",
    "EvidenceRef",
    "ExtractionCapabilities",
    "ExtractionProbe",
    "ExtractionProvider",
    "ExtractionResult",
    "ExtractionUnavailableError",
    "HistoryDecision",
    "ImageRef",
    "LocalExtractionSlot",
    "ObservationError",
    "Polarity",
    "ProfileBuilder",
    "ProfileCitation",
    "ProfileEvent",
    "ProfileStore",
    "RankExplanation",
    "SyntheticExtractionRuntime",
    "TasteClaim",
    "TasteObservation",
    "TasteProfile",
    "WhatILearned",
    "citations_for",
    "create_observation",
    "explain_rank",
    "extract_or_unavailable",
    "extraction_default_disclosure",
    "familiar_surprising_dimensions",
    "pairing_rationale",
    "profile_dimensions",
    "profile_fit",
    "resolve_extraction_provider",
    "retain_ephemeral",
    "retention_policy",
    "save_to_catalog",
    "TasteSession",
    "ObservationStore",
    "SessionStore",
]
