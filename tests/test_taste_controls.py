"""Tests for src/curator/taste/controls (M007/S03 blend/veto/history/isolation).

Synthetic profiles and signals; all deterministic with no RNG. The renderer
check builds a real tiny PNG source so it exercises the actual deterministic
renderer path (taste-layer isolation regression).
"""

from __future__ import annotations

import io
import json

from PIL import Image

from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.artdirection.manifest import ArtDirectionManifest, ProcessingIntent
from curator.hashing import sha256_hex
from curator.render.renderer import DeterministicRenderer
from curator.taste.controls import (
    DROP_THRESHOLD,
    VETO_THRESHOLD,
    ProfileHistory,
    apply_veto,
    approved_manifest_unchanged,
    blend,
    veto,
)
from curator.taste.profiles import (
    SIGNAL_NAMES,
    TasteProfile,
    TasteProfileKind,
    baseline_weights,
    default_profile,
)


def _profile(kind: TasteProfileKind, weights: dict[str, float], version: int = 1) -> TasteProfile:
    return TasteProfile(
        id=f"p-{kind.value}",
        kind=kind,
        name=kind.value,
        weights={**baseline_weights(), **weights},
        version=version,
    )


def _result(colorfulness: float, aesthetic: float = 0.5) -> AnalysisResult:
    return AnalysisResult(
        asset_id="asset",
        quality=QualitySignals(aesthetic_quality=aesthetic),
        color_story=ColorStory(colorfulness=colorfulness),
    )


def _signals(amap) -> dict[str, dict[str, float]]:
    from curator.taste.rank import TasteRanker

    r = TasteRanker()
    return {i: r.signal_values(a) for i, a in amap.items()}


# ---------------------------------------------------------------------------
# blend
# ---------------------------------------------------------------------------


def test_blend_weights_weighted_combination():
    a = _profile(TasteProfileKind.PERSONAL, {"colorfulness": 1.0, "harmony": 0.5})
    b = _profile(TasteProfileKind.HOUSEHOLD, {"colorfulness": -1.0, "aesthetic_quality": 2.0})
    out = blend(a, b, w=0.5)
    assert out.weights["colorfulness"] == 0.5 * 1.0 + 0.5 * (-1.0)
    assert out.weights["harmony"] == 0.5 * 0.5 + 0.5 * 0.0
    assert out.weights["aesthetic_quality"] == 0.5 * 0.0 + 0.5 * 2.0
    assert set(out.weights) == set(SIGNAL_NAMES)


def test_blend_returns_new_profile_and_bumps_version():
    a = _profile(TasteProfileKind.PERSONAL, {"colorfulness": 2.0}, version=3)
    b = _profile(TasteProfileKind.HOUSEHOLD, {"harmony": 1.0}, version=5)
    out = blend(a, b, w=0.25)
    assert out is not a and out is not b
    assert out.version == max(a.version, b.version) + 1 == 6
    assert out.kind is TasteProfileKind.PERSONAL
    assert out.id == f"blend-{a.id}-{b.id}"
    assert a.version == 3 and b.version == 5  # inputs untouched


def test_blend_deterministic():
    a = _profile(TasteProfileKind.PERSONAL, {"colorfulness": 1.0})
    b = _profile(TasteProfileKind.PERSONAL, {"harmony": 2.0})
    first = blend(a, b, 0.4)
    second = blend(a, b, 0.4)
    assert first == second


def test_blend_extremes_reduce_to_inputs():
    a = _profile(TasteProfileKind.PERSONAL, {"colorfulness": 3.0})
    b = _profile(TasteProfileKind.HOUSEHOLD, {"colorfulness": -1.0})
    assert blend(a, b, 1.0).weights["colorfulness"] == a.weights["colorfulness"]
    assert blend(a, b, 0.0).weights["colorfulness"] == b.weights["colorfulness"]


# ---------------------------------------------------------------------------
# veto + apply_veto
# ---------------------------------------------------------------------------


def test_veto_flags_strongly_negative_signal():
    primary = _profile(TasteProfileKind.PERSONAL, {"colorfulness": 1.0, "harmony": 0.5})
    vetoer = _profile(TasteProfileKind.HOUSEHOLD, {"colorfulness": -5.0})
    out = veto(primary, vetoer)
    assert out.weights["colorfulness"] == -5.0  # veto overrides primary
    assert out.weights["harmony"] == 0.5  # untouched keep primary
    assert out.version == primary.version + 1
    assert out.kind is primary.kind


def test_veto_no_strong_negative_leaves_primary_weights():
    primary = _profile(TasteProfileKind.PERSONAL, {"harmony": 1.0})
    mild = _profile(TasteProfileKind.HOUSEHOLD, {"colorfulness": -1.0})
    assert veto(primary, mild).weights["harmony"] == 1.0
    assert veto(primary, mild).weights["colorfulness"] == 0.0
    assert VETO_THRESHOLD == -2.0


def test_apply_veto_demotes_strongly_disliked():
    cands = [
        {"id": "lo", "baseline": 1.0},
        {"id": "hi", "baseline": 1.0},
        {"id": "mid", "baseline": 1.0},
    ]
    amap = {
        "lo": _result(colorfulness=0.0),
        "hi": _result(colorfulness=1.0),
        "mid": _result(colorfulness=0.1),
    }
    sig = _signals(amap)
    vetoer = _profile(TasteProfileKind.HOUSEHOLD, {"colorfulness": -5.0})
    # hi delta -5.0 < DROP_THRESHOLD (-1.0); mid delta -0.5 is kept. Demote hi.
    ranked = apply_veto(cands, sig, vetoer)
    assert [c["id"] for c in ranked] == ["lo", "mid", "hi"]
    assert DROP_THRESHOLD == -1.0


def test_apply_veto_drop_removes_strongly_disliked():
    cands = [
        {"id": "lo", "baseline": 1.0},
        {"id": "hi", "baseline": 1.0},
    ]
    amap = {
        "lo": _result(colorfulness=0.0),
        "hi": _result(colorfulness=1.0),
    }
    sig = _signals(amap)
    vetoer = _profile(TasteProfileKind.HOUSEHOLD, {"colorfulness": -5.0})
    ranked = apply_veto(cands, sig, vetoer, policy="drop")
    assert [c["id"] for c in ranked] == ["lo"]


def test_apply_veto_deterministic():
    cands = [
        {"id": f"c{i}", "baseline": 1.0} for i in range(6)
    ]
    amap = {f"c{i}": _result(colorfulness=i / 5.0) for i in range(6)}
    sig = _signals(amap)
    vetoer = _profile(TasteProfileKind.HOUSEHOLD, {"colorfulness": -5.0})
    first = apply_veto(cands, sig, vetoer)
    second = apply_veto(cands, sig, vetoer)
    assert first == second


def test_apply_veto_unknown_policy_raises():
    cands = [{"id": "a", "baseline": 1.0}]
    sig = {"a": {name: 0.0 for name in SIGNAL_NAMES}}
    vetoer = _profile(TasteProfileKind.HOUSEHOLD, {"colorfulness": -5.0})
    try:
        apply_veto(cands, sig, vetoer, policy="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# ---------------------------------------------------------------------------
# ProfileHistory
# ---------------------------------------------------------------------------


def test_history_snapshot_replay_restores_exact_profile():
    history = ProfileHistory()
    prof = _profile(TasteProfileKind.PERSONAL, {"colorfulness": 1.5, "harmony": -0.5}, version=4)
    version = history.snapshot(prof)
    assert version == 4
    restored = history.replay(4)
    assert restored == prof
    assert restored.weights == prof.weights
    assert restored.version == prof.version


def test_history_snapshot_is_immutable_deep_copy():
    history = ProfileHistory()
    weights = {"colorfulness": 2.0}
    prof = _profile(TasteProfileKind.PERSONAL, weights, version=1)
    history.snapshot(prof)
    weights["colorfulness"] = 999.0
    assert history.replay(1).weights["colorfulness"] == 2.0


def test_history_undo_returns_previous_version():
    history = ProfileHistory()
    v1 = _profile(TasteProfileKind.PERSONAL, {"colorfulness": 1.0}, version=1)
    v2 = _profile(TasteProfileKind.PERSONAL, {"colorfulness": 2.0}, version=2)
    history.snapshot(v1)
    history.snapshot(v2)
    previous = history.undo()
    assert previous.version == 1
    assert previous.weights["colorfulness"] == 1.0
    assert history.head() == 2


def test_history_undo_nothing_raises():
    history = ProfileHistory()
    history.snapshot(_profile(TasteProfileKind.PERSONAL, {}, version=1))
    try:
        history.undo()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_history_append_only_does_not_overwrite():
    history = ProfileHistory()
    history.snapshot(_profile(TasteProfileKind.PERSONAL, {"colorfulness": 1.0}, version=2))
    history.snapshot(_profile(TasteProfileKind.PERSONAL, {"colorfulness": 99.0}, version=2))
    assert history.replay(2).weights["colorfulness"] == 1.0


def test_history_reset_returns_default_profile():
    history = ProfileHistory()
    out = history.reset(_profile(TasteProfileKind.ROOM, {"colorfulness": 3.0}))
    assert out.weights == baseline_weights()
    assert out.kind is TasteProfileKind.ROOM
    assert out == default_profile(kind=TasteProfileKind.ROOM)


def test_history_export_round_trips_portable_json_no_secrets():
    history = ProfileHistory()
    prof = _profile(TasteProfileKind.HOUSEHOLD, {"aesthetic_quality": 1.5}, version=6)
    exported = history.export_profile(prof)
    rebuilt = TasteProfile.from_dict(json.loads(json.dumps(exported)))
    assert rebuilt == prof
    assert "token" not in exported and "secret" not in exported
    # Portable and lossless across a JSON boundary.
    assert "kind" in exported and "weights" in exported and "version" in exported


def test_history_delete_removes_snapshot():
    history = ProfileHistory()
    history.snapshot(_profile(TasteProfileKind.PERSONAL, {"colorfulness": 1.0}, version=1))
    history.snapshot(_profile(TasteProfileKind.PERSONAL, {"colorfulness": 2.0}, version=2))
    history.delete(1)
    assert 1 not in history.snapshots
    assert 2 in history.snapshots
    try:
        history.replay(1)
    except KeyError:
        return
    raise AssertionError("expected KeyError")


def test_history_delete_profiles_multiple():
    history = ProfileHistory()
    for v in (1, 2, 3):
        history.snapshot(_profile(TasteProfileKind.PERSONAL, {"colorfulness": float(v)}, version=v))
    history.delete_profiles([1, 3, 99])
    assert set(history.snapshots) == {2}


# ---------------------------------------------------------------------------
# approved_manifest_unchanged: renderer-path isolation regression
# ---------------------------------------------------------------------------


def _png_bytes(w: int = 120, h: int = 68) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 90, 60)).save(buf, format="PNG")
    return buf.getvalue()


def _small_manifest() -> tuple[ArtDirectionManifest, dict[str, bytes]]:
    source = _png_bytes()
    sha = sha256_hex(source)
    manifest = ArtDirectionManifest(
        sources=[sha],
        layout_treatment=ArtDirectionManifest.from_dict(
            {"layout_treatment": "single_fullbleed"}
        ).layout_treatment,
        processing_intent=ProcessingIntent(upscale_warning=True),
    )
    return manifest, {sha: source}


def test_approved_manifest_unchanged_true_and_deterministic():
    manifest, sources = _small_manifest()
    once = approved_manifest_unchanged(manifest, sources)
    twice = approved_manifest_unchanged(manifest, sources)
    assert once is True
    assert twice is True


def test_renderer_path_deterministic_regardless_of_taste():
    manifest, sources = _small_manifest()
    renderer = DeterministicRenderer()
    r1 = renderer.render(manifest, sources, (1920, 1080))
    r2 = renderer.render(manifest, sources, (1920, 1080))
    assert r1.sha256 == r2.sha256
    assert approved_manifest_unchanged(manifest, sources, (1920, 1080)) is True
