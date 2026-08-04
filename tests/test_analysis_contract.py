"""Contract tests for M002/S01 — profiles, backends, models, provider (T2/T3/T4).

Verifies: strict profile ordering + monotonic stage nesting; deterministic /
strict-device / CPU-fallback backend resolution; the always-available CPU
reference runner; and a FakeAnalysisProvider that satisfies the
:class:`AnalysisProvider` ABC, round-trips a full result, and whose capabilities
cover a requested profile/backend.
"""

from __future__ import annotations

import pytest

from analysis_factory import full_result
from curator.analysis.compute import (
    ComputeBackend,
    ComputeBackendError,
    resolve_backend,
    strict_device,
)
from curator.analysis.errors import AnalysisError
from curator.analysis.model import CpuReferenceRunner, ModelPrecision, ModelSpec
from curator.analysis.profiles import (
    KNOWN_STAGES,
    AnalysisProfile,
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
from curator.analysis.schema import AnalysisResult

FAST = AnalysisProfile.FAST
BALANCED = AnalysisProfile.BALANCED
QUALITY = AnalysisProfile.QUALITY
MAX = AnalysisProfile.MAX
CUSTOM = AnalysisProfile.CUSTOM


# ---------------------------------------------------------------------------
# T2: profiles
# ---------------------------------------------------------------------------


def test_profile_order_strict_less_than() -> None:
    assert profile_order(FAST, BALANCED) == -1
    assert profile_order(BALANCED, QUALITY) == -1
    assert profile_order(QUALITY, MAX) == -1
    assert profile_order(MAX, FAST) == 1
    assert profile_order(QUALITY, QUALITY) == 0


def test_profile_specs_monotonic_nesting() -> None:
    specs = {p: [s.stage for s in profile_specs(p)] for p in (FAST, BALANCED, QUALITY, MAX)}
    assert specs[FAST] == ["perceptual", "technical"]
    # Each higher profile is a strict prefix-extension of the previous.
    ordered = [FAST, BALANCED, QUALITY, MAX]
    for lower, higher in zip(ordered, ordered[1:]):
        assert specs[lower] == specs[higher][: len(specs[lower])]
        assert len(specs[higher]) > len(specs[lower])
    # MAX covers every known stage.
    assert set(specs[MAX]) == KNOWN_STAGES


def test_profile_specs_rank_ordering() -> None:
    ranks = [s.rank for s in profile_specs(MAX)]
    assert ranks == sorted(ranks)


def test_profile_specs_custom_requires_explicit_stages() -> None:
    with pytest.raises(AnalysisError):
        profile_specs(CUSTOM)


def test_custom_profile_validates_known_stages() -> None:
    specs = custom_profile(["perceptual", "colorstory"])
    assert [s.stage for s in specs] == ["perceptual", "colorstory"]
    with pytest.raises(AnalysisError):
        custom_profile(["perceptual", "not_a_real_stage"])


# ---------------------------------------------------------------------------
# T3: compute backends + models
# ---------------------------------------------------------------------------


def test_resolve_backend_deterministic() -> None:
    a = resolve_backend("cpu", ["cpu"])
    b = resolve_backend("cpu", ["cpu"])
    assert a == b


def test_resolve_backend_matches_available() -> None:
    backend, strict = resolve_backend(ComputeBackend.METAL, ["metal", "cpu"])
    assert backend is ComputeBackend.METAL
    assert strict is False


def test_resolve_backend_strict_failure() -> None:
    with pytest.raises(ComputeBackendError):
        resolve_backend("cuda", ["cpu"], strict=True)


def test_resolve_backend_cpu_fallback() -> None:
    backend, strict = resolve_backend("cuda", ["cpu"])
    assert backend is ComputeBackend.CPU
    assert strict is False


def test_strict_device_succeeds_when_available() -> None:
    assert strict_device("cpu", ["cpu"]) is ComputeBackend.CPU


def test_resolve_backend_auto_picks_available() -> None:
    backend, strict = resolve_backend("auto", ["metal", "cpu"])
    assert backend is ComputeBackend.METAL
    assert strict is False


def test_cpu_reference_runner_always_available() -> None:
    runner = CpuReferenceRunner()
    assert runner.available() is True
    spec = ModelSpec(name="ref", version="1", family="test")
    assert runner.run(spec, {"x": 1}) == {"x": 1}


def test_model_spec_to_dict_serializes_precision() -> None:
    spec = ModelSpec(
        name="n", version="1", family="f",
        precision=ModelPrecision.FP16, sha256="abc", task="quality",
    )
    d = spec.to_dict()
    assert d["precision"] == "fp16"
    assert d["sha256"] == "abc"
    assert d["task"] == "quality"


# ---------------------------------------------------------------------------
# T4: provider ABC + fake provider integration
# ---------------------------------------------------------------------------


class FakeAnalysisProvider(AnalysisProvider):
    """A test double that always provides CPU analysis of a full result."""

    def capabilities(self) -> AnalysisCapabilities:
        return AnalysisCapabilities(
            profiles=frozenset({FAST, BALANCED, QUALITY, MAX}),
            backends=frozenset({ComputeBackend.CPU}),
            air_gapped=True,
            deterministic=True,
        )

    def probe(self) -> ComputeProbe:
        return ComputeProbe(
            ok=True,
            backend=ComputeBackend.CPU,
            latency_ms=1.5,
            available_backends=frozenset({ComputeBackend.CPU}),
            message="fake ok",
        )

    def analyze(self) -> AnalysisResult:
        return full_result("sd_asset")


def test_fake_provider_satisfies_abc() -> None:
    provider = FakeAnalysisProvider()
    provider_caps = provider.capabilities()
    assert isinstance(provider, AnalysisProvider)
    assert isinstance(provider_caps, AnalysisCapabilities)
    probe = provider.probe()
    assert probe.ok is True
    assert probe.backend is ComputeBackend.CPU


def test_fake_provider_result_roundtrips() -> None:
    result = FakeAnalysisProvider().analyze()
    restored = AnalysisResult.from_dict(result.to_dict())
    assert restored == result
    assert restored.asset_id == "sd_asset"


def test_capabilities_cover_requested_profile_backend() -> None:
    cap_satisfied = FakeAnalysisProvider().capabilities()
    assert capability_requirements(cap_satisfied, MAX, ComputeBackend.CPU) is True


def test_capability_requirements_reject_missing_backend() -> None:
    caps = FakeAnalysisProvider().capabilities()
    assert capability_requirements(caps, FAST, ComputeBackend.CUDA) is False


def test_capability_requirements_reject_missing_profile() -> None:
    caps = AnalysisCapabilities(
        profiles=frozenset({FAST}),
        backends=frozenset({ComputeBackend.CPU}),
    )
    assert capability_requirements(caps, MAX, ComputeBackend.CPU) is False


def test_capabilities_to_dict_serializes() -> None:
    d = FakeAnalysisProvider().capabilities().to_dict()
    assert "cpu" in d["backends"]
    assert d["deterministic"] is True
