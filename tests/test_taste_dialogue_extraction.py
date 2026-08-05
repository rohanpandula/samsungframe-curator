"""Tests for src/curator/taste/dialogue/extraction (M008/S02).

Deterministic and air-gapped: the cloud runtime is synthetic, so every privacy
claim is asserted against what the runtime actually received. Covers the
capabilities/result JSON round-trips, provider resolution from config, the
no-silent-fallback contract of :func:`extract_or_unavailable`, the disclosure,
the per-image exclusion policy dropping components before sending, the local
slot's clean unavailability, cloud outage degradation, and determinism.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

from curator.providers import ExclusionPolicy
from curator.taste.dialogue import (
    CONTROLLED_VOCABULARY,
    CloudExtractionProvider,
    ExtractionCapabilities,
    ExtractionResult,
    ExtractionUnavailableError,
    ImageRef,
    LocalExtractionSlot,
    Polarity,
    SyntheticExtractionRuntime,
    TasteObservation,
    extract_or_unavailable,
    resolve_extraction_provider,
)

#: A sentinel credential that must never appear in any received payload.
TEST_SECRET = "samsung_tv_credential_XY1234"


def _thumb(
    tmp_path,
    color: tuple[int, int, int] = (120, 130, 140),
    width: int = 600,
    height: int = 400,
    name: str = "img.thumb.jpg",
) -> Path:
    """Write a deterministic JPEG thumbnail file and return its path."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    path = tmp_path / name
    path.write_bytes(buf.getvalue())
    return path


def _observation(
    verbatim: str,
    thumb_path=None,
    sha: str = "a" * 64,
) -> TasteObservation:
    images = [ImageRef(sha256=sha, thumb_path=thumb_path)] if thumb_path else []
    return TasteObservation(
        session_id="sess-1",
        verbatim=verbatim,
        attributes=[],
        polarity=Polarity.LIKE,
        confidence=0.5,
        images=images,
    )


# ---------------------------------------------------------------------------
# T1: capabilities / result round-trips
# ---------------------------------------------------------------------------


def test_capabilities_round_trip() -> None:
    caps = ExtractionCapabilities(
        enabled=True,
        kind="cloud",
        supports_local=False,
        supports_cloud=True,
        disclosure_available=True,
    )
    rebuilt = ExtractionCapabilities.from_dict(json.loads(json.dumps(caps.to_dict())))
    assert rebuilt == caps
    assert ExtractionCapabilities.from_dict(caps) is caps
    assert ExtractionCapabilities.from_dict(caps.to_dict()) == caps


def test_result_round_trip() -> None:
    result = ExtractionResult(
        attributes=["negative-space", "quiet"],
        polarity=Polarity.LIKE,
        confidence=0.9,
        verbatim="i love the quiet negative space",
        provider="synthetic-cloud",
    )
    rebuilt = ExtractionResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert rebuilt == result
    assert rebuilt.polarity is Polarity.LIKE
    assert ExtractionResult.from_dict(result) is result
    assert ExtractionResult.from_dict(result.to_dict()) == result
    assert rebuilt.to_dict() == result.to_dict()


def test_controlled_vocabulary_closed_and_unique() -> None:
    assert len(CONTROLLED_VOCABULARY) == len(set(CONTROLLED_VOCABULARY))
    assert all(isinstance(tag, str) and tag for tag in CONTROLLED_VOCABULARY)


def test_result_rejects_out_of_vocabulary() -> None:
    with pytest.raises(ValueError):
        ExtractionResult(
            attributes=["made-up-tag"],
            polarity=Polarity.LIKE,
            confidence=0.5,
            verbatim="x",
            provider="p",
        )


def test_result_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        ExtractionResult(
            attributes=[],
            polarity=Polarity.CONFLICTED,
            confidence=1.5,
            verbatim="x",
            provider="p",
        )


# ---------------------------------------------------------------------------
# T2: provider resolution from config
# ---------------------------------------------------------------------------


def test_resolve_cloud_enabled() -> None:
    provider = resolve_extraction_provider({"enabled": True, "provider": "cloud"})
    assert isinstance(provider, CloudExtractionProvider)
    assert provider.enabled is True


def test_resolve_cloud_with_policy() -> None:
    policy = ExclusionPolicy(per_image={"a" * 64: frozenset({"faces"})})
    provider = resolve_extraction_provider(
        {"enabled": True, "provider": "cloud", "policy": policy}
    )
    assert isinstance(provider, CloudExtractionProvider)
    assert provider.policy is policy


def test_resolve_local_enabled() -> None:
    provider = resolve_extraction_provider({"enabled": True, "provider": "local"})
    assert isinstance(provider, LocalExtractionSlot)


def test_resolve_none_when_nothing_enabled() -> None:
    assert resolve_extraction_provider(None) is None
    assert resolve_extraction_provider({}) is None
    assert resolve_extraction_provider({"enabled": False, "provider": "cloud"}) is None
    assert resolve_extraction_provider({"enabled": True, "provider": "unknown"}) is None


# ---------------------------------------------------------------------------
# T3: cloud extraction — happy path + no-secrets guarantee
# ---------------------------------------------------------------------------


def test_cloud_extract_happy_path_and_no_secrets(tmp_path) -> None:
    thumb = _thumb(tmp_path)
    sha = "a" * 64
    policy = ExclusionPolicy(per_image={sha: frozenset({"faces", "gps"})})
    runtime = SyntheticExtractionRuntime()
    provider = CloudExtractionProvider(runtime=runtime, policy=policy)
    verbatim = "i love the quiet negative space"
    obs = _observation(verbatim, thumb, sha)

    result = provider.extract(obs)

    assert result.polarity is Polarity.LIKE
    assert result.confidence >= 0.8
    assert "negative-space" in result.attributes
    assert "quiet" in result.attributes
    assert all(tag in CONTROLLED_VOCABULARY for tag in result.attributes)
    assert result.verbatim == verbatim
    assert result.provider == "synthetic-cloud"

    assert len(runtime.received_payloads) == 1
    verbatim_sent, context = runtime.received_payloads[0]
    assert verbatim_sent == verbatim
    assert len(context["images"]) == 1
    img = context["images"][0]
    assert img["sha256"] == sha
    assert img["width"] <= 512 and img["height"] <= 512
    assert "faces_present" not in img
    assert "gps" not in img
    original = thumb.read_bytes()
    assert original not in img["thumb_bytes"]
    assert img["thumb_bytes"] != original
    assert TEST_SECRET.encode() not in img["thumb_bytes"]
    assert TEST_SECRET not in repr(context)


def test_gps_never_leaves_even_when_allowed(tmp_path) -> None:
    thumb = _thumb(tmp_path)
    provider = CloudExtractionProvider()  # no exclusions: faces allowed
    provider.extract(_observation("i love the quiet", thumb, "a" * 64))
    img = provider.runtime.received_payloads[0][1]["images"][0]
    assert "gps" not in img


def test_neutral_verbatim_conflicted_low_confidence() -> None:
    result = CloudExtractionProvider().extract(
        _observation("just a photo on the wall")
    )
    assert result.polarity is Polarity.CONFLICTED
    assert result.confidence <= 0.5
    assert result.attributes == []


def test_dislike_detected_over_like_for_dont_like() -> None:
    result = CloudExtractionProvider().extract(_observation("i don't like it"))
    assert result.polarity is Polarity.DISLIKE
    assert result.confidence >= 0.8


# ---------------------------------------------------------------------------
# T4: disclosure
# ---------------------------------------------------------------------------


def test_cloud_disclosure_lists_what_leaves() -> None:
    d = CloudExtractionProvider().disclosure()
    leaves = d.leaves_machine
    assert "verbatim" in leaves.payload_types
    assert "downscaled_thumbnail" in leaves.payload_types
    assert "sha256" in leaves.metadata_scope
    assert "dimensions" in leaves.metadata_scope
    for bad in ("original_image", "secrets", "credentials", "tv_ha_credentials", "gps", "faces"):
        assert bad not in leaves.payload_types
        assert bad not in leaves.metadata_scope
        assert bad in leaves.never


def test_cloud_disclosure_reflects_exclusions() -> None:
    policy = ExclusionPolicy(per_image={"a" * 64: frozenset({"faces"})})
    d = CloudExtractionProvider(policy=policy).disclosure()
    assert set(d.exclusions) == {"faces"}
    assert d.statement and "verbatim" in d.statement
    assert "thumbnail" in d.statement


# ---------------------------------------------------------------------------
# T5: exclusion policy drops components before sending
# ---------------------------------------------------------------------------


def test_exclusion_policy_drops_faces_component(tmp_path) -> None:
    skin_tone = (200, 120, 90)  # triggers the skin-tone presence heuristic
    thumb = _thumb(tmp_path, color=skin_tone, width=300, height=200)
    sha = "a" * 64
    obs = _observation("a busy crowd", thumb, sha)

    runtime_default = SyntheticExtractionRuntime()
    CloudExtractionProvider(runtime=runtime_default).extract(obs)
    img_default = runtime_default.received_payloads[0][1]["images"][0]
    assert img_default["faces_present"] is True  # would be sent by default

    runtime_excluded = SyntheticExtractionRuntime()
    policy = ExclusionPolicy(per_image={sha: frozenset({"faces"})})
    CloudExtractionProvider(runtime=runtime_excluded, policy=policy).extract(obs)
    img_excluded = runtime_excluded.received_payloads[0][1]["images"][0]
    assert "faces_present" not in img_excluded  # dropped before sending


# ---------------------------------------------------------------------------
# T6: no-silent-fallback + local slot
# ---------------------------------------------------------------------------


def test_no_model_enabled_returns_none_no_guess() -> None:
    obs = _observation("i love the quiet negative space")
    assert extract_or_unavailable(None, obs) is None

    disabled = CloudExtractionProvider(enabled=False)
    assert disabled.capabilities().enabled is False
    result = extract_or_unavailable(disabled, obs)
    assert result is None  # never a keyword-derived guess
    with pytest.raises(ExtractionUnavailableError) as excinfo:
        disabled.extract(obs)
    assert excinfo.value.reason
    assert "disabled" in excinfo.value.reason


def test_local_slot_raises_cleanly() -> None:
    slot = LocalExtractionSlot()
    caps = slot.capabilities()
    assert caps.enabled is False
    assert caps.kind == "local"
    assert caps.supports_local is True
    assert caps.supports_cloud is False
    assert caps.disclosure_available is False
    obs = _observation("i love the quiet negative space")
    with pytest.raises(ExtractionUnavailableError) as excinfo:
        slot.extract(obs)
    assert "local model not configured" in excinfo.value.reason
    assert extract_or_unavailable(slot, obs) is None


# ---------------------------------------------------------------------------
# T7: outage + determinism
# ---------------------------------------------------------------------------


def test_outage_returns_none_no_crash(tmp_path) -> None:
    thumb = _thumb(tmp_path)
    runtime = SyntheticExtractionRuntime()
    provider = CloudExtractionProvider(runtime=runtime)
    runtime.set_down(True)
    obs = _observation("i love the quiet negative space", thumb, "a" * 64)
    assert extract_or_unavailable(provider, obs) is None
    assert runtime.received_payloads == []
    with pytest.raises(ExtractionUnavailableError):
        provider.extract(obs)


def test_deterministic_same_verbatim_same_result() -> None:
    provider = CloudExtractionProvider()
    obs = _observation("i love the quiet negative space")
    first = provider.extract(obs)
    second = provider.extract(obs)
    assert first == second
    assert first.to_dict() == second.to_dict()
    assert first.attributes == second.attributes
    assert first.polarity is second.polarity
    assert first.confidence == second.confidence
