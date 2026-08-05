"""Tests for src/curator/taste (M007/S01 profile + isolated deterministic rerank).

Covers the identity default profile, JSON round-trip, disable-restores-baseline
isolation, deterministic trained reranking with explainable contributions, kind
isolation, and the schema v13 ``taste_profiles`` tables.
"""

from __future__ import annotations

import json

from curator import db as _db
from curator.analysis.schema import (
    AnalysisResult,
    ColorStory,
    Pairing,
    QualitySignals,
)
from curator.schema import EXPECTED_TABLES, SCHEMA_VERSION
from curator.taste import (
    SIGNAL_NAMES,
    TasteProfile,
    TasteProfileKind,
    TasteRanker,
    baseline_weights,
    default_profile,
)


def _result(
    aesthetic: float = 0.5,
    technical: float = 0.5,
    colorfulness: float = 0.5,
    harmony: float = 0.5,
    affinity: float = 0.5,
) -> AnalysisResult:
    return AnalysisResult(
        asset_id="asset",
        quality=QualitySignals(
            aesthetic_quality=aesthetic, technical_quality=technical
        ),
        color_story=ColorStory(colorfulness=colorfulness, harmony=harmony),
        pairing=Pairing(affinity=affinity),
    )


def _profile(
    kind: TasteProfileKind,
    weights: dict[str, float],
    version: int = 1,
) -> TasteProfile:
    return TasteProfile(
        id=f"p-{kind.value}",
        kind=kind,
        name=kind.value,
        weights=weights,
        version=version,
    )


# ---------------------------------------------------------------------------
# profiles: baseline identity + default profile
# ---------------------------------------------------------------------------


def test_baseline_weights_all_zero_identity():
    weights = baseline_weights()
    assert set(weights) == set(SIGNAL_NAMES)
    assert all(v == 0.0 for v in weights.values())
    # Deterministic identity: reproducible across calls.
    assert baseline_weights() == weights


def test_default_profile_preserves_baseline():
    prof = default_profile()
    assert prof.kind is TasteProfileKind.PERSONAL
    assert prof.weights == baseline_weights()
    assert prof.version == 1
    other = default_profile(kind=TasteProfileKind.ROOM, name="studio")
    assert other.kind is TasteProfileKind.ROOM
    assert other.name == "studio"
    assert other.weights == baseline_weights()


# ---------------------------------------------------------------------------
# profiles: JSON round-trip + versioning
# ---------------------------------------------------------------------------


def test_taste_profile_round_trip():
    prof = _profile(
        TasteProfileKind.HOUSEHOLD,
        {"aesthetic_quality": 1.5, "harmony": -0.5, "vibrancy": 2.0},
        version=4,
    )
    rebuilt = TasteProfile.from_dict(json.loads(json.dumps(prof.to_dict())))
    assert rebuilt == prof
    assert rebuilt.kind is TasteProfileKind.HOUSEHOLD
    assert rebuilt.weights == {"aesthetic_quality": 1.5, "harmony": -0.5, "vibrancy": 2.0}
    assert rebuilt.version == 4


def test_taste_profile_kind_coerced_from_string():
    rebuilt = TasteProfile.from_dict(
        {
            "id": "x",
            "kind": "season",  # string form, not the enum
            "name": "winter",
            "weights": baseline_weights(),
            "version": 2,
        }
    )
    assert rebuilt.kind is TasteProfileKind.SEASON
    assert rebuilt.version == 2
    assert rebuilt.id == "x"


def test_version_increments_round_trip():
    prof = _profile(TasteProfileKind.EXPERIMENTAL, {"colorfulness": 1.0}, version=1)
    assert prof.version == 1
    bumped = TasteProfile.from_dict({**prof.to_dict(), "version": 7})
    assert bumped.version == 7
    assert bumped.weights == prof.weights


# ---------------------------------------------------------------------------
# ranker: disabled -> exact baseline order (isolation)
# ---------------------------------------------------------------------------


def test_none_profile_reproduces_baseline_exactly():
    ranker = TasteRanker()
    cands = [
        {"id": "a", "baseline": 1.0},
        {"id": "b", "baseline": 2.0},
        {"id": "c", "baseline": 3.0},
    ]
    amap = {"a": _result(), "b": _result(), "c": _result()}
    assert ranker.rank(cands, None, analysis_map=amap) == cands
    assert ranker.is_enabled(None) is False


def test_all_zero_profile_reproduces_baseline_exactly():
    ranker = TasteRanker()
    cands = [
        {"id": "z", "baseline": 0.5},
        {"id": "y", "baseline": 9.0},
        {"id": "x", "baseline": 3.0},
    ]
    amap = {c["id"]: _result() for c in cands}
    prof = _profile(TasteProfileKind.PERSONAL, baseline_weights())
    assert ranker.is_enabled(prof) is False
    assert ranker.rank(cands, prof, analysis_map=amap) == cands


def test_default_profile_reproduces_baseline_exactly():
    ranker = TasteRanker()
    cands = [
        {"id": "a", "baseline": 5.0},
        {"id": "b", "baseline": 1.0},
    ]
    amap = {c["id"]: _result() for c in cands}
    prof = default_profile()
    assert ranker.is_enabled(prof) is False
    assert ranker.rank(cands, prof, analysis_map=amap) == cands


# ---------------------------------------------------------------------------
# ranker: trained profile reranks deterministically
# ---------------------------------------------------------------------------


def test_trained_profile_reranks_deterministically():
    ranker = TasteRanker()
    cands = [
        {"id": "hi", "baseline": 1.0},
        {"id": "mid", "baseline": 1.0},
        {"id": "lo", "baseline": 1.0},
    ]
    amap = {
        "hi": _result(aesthetic=1.0),
        "mid": _result(aesthetic=0.75),
        "lo": _result(aesthetic=0.0),
    }
    prof = _profile(
        TasteProfileKind.PERSONAL,
        {"aesthetic_quality": 1.0, **{k: 0.0 for k in SIGNAL_NAMES if k != "aesthetic_quality"}},
    )
    assert ranker.is_enabled(prof) is True
    first = ranker.rank(cands, prof, analysis_map=amap)
    second = ranker.rank(cands, prof, analysis_map=amap)
    assert [c["id"] for c in first] == ["hi", "mid", "lo"]
    assert first == second


def test_signal_weight_shifts_candidates():
    ranker = TasteRanker()
    cands = [
        {"id": "aesthetic", "baseline": 1.0},
        {"id": "technical", "baseline": 1.0},
    ]
    amap = {
        "aesthetic": _result(aesthetic=1.0, technical=0.0),
        "technical": _result(aesthetic=0.0, technical=1.0),
    }
    aesthetic_prof = _profile(
        TasteProfileKind.PERSONAL,
        {"aesthetic_quality": 1.0, **{k: 0.0 for k in SIGNAL_NAMES if k != "aesthetic_quality"}},
    )
    technical_prof = _profile(
        TasteProfileKind.PERSONAL,
        {"technical_quality": 1.0, **{k: 0.0 for k in SIGNAL_NAMES if k != "technical_quality"}},
    )
    assert [c["id"] for c in ranker.rank(cands, aesthetic_prof, analysis_map=amap)] == [
        "aesthetic",
        "technical",
    ]
    assert [c["id"] for c in ranker.rank(cands, technical_prof, analysis_map=amap)] == [
        "technical",
        "aesthetic",
    ]


def test_contributions_sum_to_delta():
    ranker = TasteRanker()
    prof = _profile(
        TasteProfileKind.HOUSEHOLD,
        {"aesthetic_quality": 2.0, "colorfulness": -1.0, "harmony": 0.5},
    )
    analysis = _result(aesthetic=0.25, colorfulness=0.5, harmony=0.8)
    delta, contributions = ranker.personal_delta(analysis, prof)
    expected = 2.0 * 0.25 + (-1.0) * 0.5 + 0.5 * 0.8
    assert delta == expected
    assert all(c["contribution"] == c["weight"] * c["value"] for c in contributions)
    assert sum(c["contribution"] for c in contributions) == delta
    cont_map = {c["signal"]: c for c in contributions}
    assert cont_map["aesthetic_quality"]["weight"] == 2.0
    assert cont_map["colorfulness"]["value"] == 0.5
    # Every canonical signal is explained (zeros included) and none are missing.
    assert {c["signal"] for c in contributions} == set(SIGNAL_NAMES)


# ---------------------------------------------------------------------------
# kind isolation
# ---------------------------------------------------------------------------


def test_kind_isolation_independent_rankings():
    ranker = TasteRanker()
    cands = [
        {"id": "a", "baseline": 1.0},
        {"id": "b", "baseline": 1.0},
    ]
    amap = {
        "a": _result(aesthetic=1.0, technical=0.0),
        "b": _result(aesthetic=0.0, technical=1.0),
    }
    personal = _profile(
        TasteProfileKind.PERSONAL,
        {"aesthetic_quality": 1.0, **{k: 0.0 for k in SIGNAL_NAMES if k != "aesthetic_quality"}},
    )
    household = _profile(
        TasteProfileKind.HOUSEHOLD,
        {"technical_quality": 1.0, **{k: 0.0 for k in SIGNAL_NAMES if k != "technical_quality"}},
    )
    p_rank = [c["id"] for c in ranker.rank(cands, personal, analysis_map=amap)]
    h_rank = [c["id"] for c in ranker.rank(cands, household, analysis_map=amap)]
    assert p_rank == ["a", "b"]
    assert h_rank == ["b", "a"]
    # Ranking with one profile did not perturb the other (pure/stateless).
    personal_expected = {
        "aesthetic_quality": 1.0,
        **{k: 0.0 for k in SIGNAL_NAMES if k != "aesthetic_quality"},
    }
    household_expected = {
        "technical_quality": 1.0,
        **{k: 0.0 for k in SIGNAL_NAMES if k != "technical_quality"},
    }
    assert personal.weights == personal_expected
    assert household.weights == household_expected


def test_mutating_input_weights_does_not_affect_frozen_profile():
    weights_in = {"colorfulness": 1.0, "harmony": 0.0}
    prof = TasteProfile(
        id="guard", kind=TasteProfileKind.ROOM, name="guard", weights=weights_in
    )
    weights_in["colorfulness"] = 999.0
    weights_in["harmony"] = 123.0
    assert prof.weights["colorfulness"] == 1.0
    assert prof.weights["harmony"] == 0.0


# ---------------------------------------------------------------------------
# schema v13
# ---------------------------------------------------------------------------


def test_schema_v13_taste_tables_exist(data_root):
    db = _db.connect(data_root)
    _db.migrate(db)
    tables = _db.table_names(db)
    assert SCHEMA_VERSION >= 14
    assert "taste_profiles" in tables
    assert "taste_preferences" in tables
    assert "taste_profiles" in EXPECTED_TABLES
    assert "taste_preferences" in EXPECTED_TABLES
    cols = {row[1] for row in db.execute("PRAGMA table_info(taste_profiles)")}
    assert {"kind", "version", "weights_json"} <= cols
    db.close()


# ---------------------------------------------------------------------------
# determinism across calls
# ---------------------------------------------------------------------------


def test_deterministic_across_repeated_calls():
    ranker = TasteRanker()
    cands = [{"id": f"c{i}", "baseline": float(i)} for i in range(5)]
    amap = {f"c{i}": _result(aesthetic=float(i) / 5.0) for i in range(5)}
    prof = _profile(
        TasteProfileKind.PERSONAL,
        {"aesthetic_quality": 1.0, **{k: 0.0 for k in SIGNAL_NAMES if k != "aesthetic_quality"}},
    )
    results = [ranker.rank(cands, prof, analysis_map=amap) for _ in range(3)]
    assert results[0] == results[1] == results[2]
    assert [c["id"] for c in results[0]] == ["c4", "c3", "c2", "c1", "c0"]
