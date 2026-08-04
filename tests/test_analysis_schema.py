"""Tests for the AnalysisResult schema round-trip (M002/S01-T1).

Covers identity round-trips (to_dict -> from_dict -> to_dict), JSON-serializability
via json.dumps, empty/partial results, unknown-version rejection, and lenient
forward-compat (unknown keys preserved through a round-trip).
"""

from __future__ import annotations

import json

import pytest

from analysis_factory import full_result
from curator.analysis.schema import (
    SCHEMA_VERSION,
    AnalysisResult,
    QualitySignals,
    Saliency,
)
from curator.errors import CuratorError


def test_roundtrip_identity_full_result() -> None:
    result = full_result()
    restored = AnalysisResult.from_dict(result.to_dict())
    assert restored == result
    assert restored.to_dict() == result.to_dict()


def test_json_dumps_serializable() -> None:
    result = full_result()
    payload = json.dumps(result.to_dict())
    assert isinstance(payload, str)
    restored = AnalysisResult.from_json(payload)
    assert restored == result


def test_empty_result_roundtrip() -> None:
    result = AnalysisResult(asset_id="bare")
    restored = AnalysisResult.from_dict(result.to_dict())
    assert restored == result


def test_partial_result_allows_missing_groups() -> None:
    result = AnalysisResult.from_dict(
        {"asset_id": "partial", "quality": {"technical_quality": 0.5}}
    )
    assert result.quality.technical_quality == 0.5
    assert result.quality.aesthetic_quality == 0.0  # default filled in
    assert result.saliency == Saliency()  # other groups default


def test_unknown_schema_version_raises() -> None:
    with pytest.raises(CuratorError) as exc:
        AnalysisResult.from_dict({"asset_id": "x", "schema_version": "999"})
    assert exc.value.__class__.__name__ == "SchemaVersionError"


def test_missing_schema_version_treated_as_current() -> None:
    result = AnalysisResult.from_dict({"asset_id": "legacy"})
    assert result.schema_version == SCHEMA_VERSION


def test_forward_compat_preserves_unknown_top_level_key() -> None:
    result = AnalysisResult.from_dict(
        {"asset_id": "x", "future_field": {"nested": 1}}
    )
    assert result.to_dict()["future_field"] == {"nested": 1}


def test_forward_compat_preserves_unknown_nested_key() -> None:
    result = AnalysisResult.from_dict(
        {"asset_id": "x", "quality": {"technical_quality": 1.0, "future_signal": 7}}
    )
    assert result.quality.technical_quality == 1.0
    assert result.to_dict()["quality"]["future_signal"] == 7


def test_nested_signal_to_dict_from_dict() -> None:
    q = QualitySignals(sharpness=0.9)
    assert QualitySignals.from_dict(q.to_dict()) == q
