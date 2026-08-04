"""Tests for M006/S01 — cloud/hybrid routing, privacy disclosure, exclusion policy.

Deterministic and air-gapped: the cloud runtime is synthetic (no network), so every
test asserts provider behavior against what the runtime was actually sent. Covers
the privacy :class:`Disclosure`, per-source/per-image :class:`ExclusionPolicy`
gating, the :class:`CloudAnalysisProvider` payload shape, outage degradation via the
:class:`HybridRouter`, and the credential-never-sent guarantee.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from curator.analysis.compute import ComputeBackend
from curator.analysis.profiles import AnalysisProfile
from curator.analysis.provider import (
    AnalysisCapabilities,
    AnalysisProvider,
    ComputeProbe,
)
from curator.analysis.schema import AnalysisResult
from curator.providers import (
    COMPONENT_FACES,
    COMPONENT_FULL_RESOLUTION,
    COMPONENT_GPS,
    CloudAnalysisProvider,
    Disclosure,
    ExclusionPolicy,
    HybridRouter,
    MachineLeaves,
    ProviderOutageError,
    SyntheticCloudAnalysisRuntime,
    default_disclosure,
)

BALANCED = AnalysisProfile.BALANCED
MAX = AnalysisProfile.MAX

#: A sentinel credential that must never appear in any received payload.
TEST_SECRET = "tv_ha_credential_AB12cd34"


def make_image_bytes(width: int = 800, height: int = 600) -> bytes:
    """Build a deterministic in-memory JPEG image."""
    img = Image.new("RGB", (width, height), (120, 130, 140))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def img_bytes_with_gps() -> bytes:
    """Build a JPEG bearing a synthetic EXIF GPS record."""
    img = Image.new("RGB", (200, 100), (90, 120, 160))
    exif = img.getexif()
    exif[0x0112] = 1  # orientation
    gps_ifd = {
        1: "N",
        2: (52.0, 5.0, 20.0),
        3: "E",
        4: (4.0, 4.0, 30.0),
    }
    exif.get_ifd(0x8825).update(gps_ifd)  # type: ignore[union-attr]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


class FakeLocalProvider(AnalysisProvider):
    """Minimal local provider for router tests (returns a fixed local result)."""

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str | None]] = []

    def capabilities(self) -> AnalysisCapabilities:
        return AnalysisCapabilities(
            profiles=frozenset(AnalysisProfile),
            backends=frozenset({ComputeBackend.CPU}),
            air_gapped=True,
            deterministic=True,
        )

    def probe(self) -> ComputeProbe:
        return ComputeProbe(
            ok=True,
            backend=ComputeBackend.CPU,
            available_backends=frozenset({ComputeBackend.CPU}),
            message="local ok",
        )

    def analyze(
        self,
        source: bytes | bytearray,
        profile: AnalysisProfile = BALANCED,
        asset_id: str | None = None,
    ) -> AnalysisResult:
        self.calls.append((bytes(source), asset_id))
        return AnalysisResult(asset_id=asset_id or "local_asset")


def _provider(
    policy: ExclusionPolicy | None = None,
) -> tuple[CloudAnalysisProvider, SyntheticCloudAnalysisRuntime]:
    runtime = SyntheticCloudAnalysisRuntime()
    provider = CloudAnalysisProvider(runtime=runtime, policy=policy or ExclusionPolicy())
    return provider, runtime


# ---------------------------------------------------------------------------
# T1: privacy disclosure
# ---------------------------------------------------------------------------


def test_default_disclosure_lists_what_leaves() -> None:
    d = default_disclosure("synthetic-cloud")
    assert d.provider == "synthetic-cloud"
    assert d.statement
    leaves: MachineLeaves = d.leaves_machine
    assert "downscaled_derivative" in leaves.payload_types
    assert "asset_id" in leaves.metadata_scope
    assert "original_image" in leaves.never
    assert "secrets" in leaves.never
    assert "credentials" in leaves.never
    assert "tv_ha_credentials" in leaves.never
    assert "gps" in leaves.never
    assert "faces" in leaves.never


def test_disclosure_roundtrips() -> None:
    d = default_disclosure("synthetic-cloud", exclusions=["faces", "gps"])
    restored = Disclosure.from_dict(d.to_dict())
    assert restored == d
    assert restored.exclusions == ["faces", "gps"]
    assert restored.leaves_machine.to_dict() == d.leaves_machine.to_dict()


def test_disclosure_never_includes_originals_or_secrets() -> None:
    for bad in ("original_image", "secrets", "credentials", "tv_ha_credentials", "gps", "faces"):
        assert bad not in default_disclosure().leaves_machine.payload_types
        assert bad not in default_disclosure().leaves_machine.metadata_scope


def test_cloud_provider_disclosure_reflects_live_policy() -> None:
    policy = ExclusionPolicy(per_source={"srcA": frozenset({"faces", "gps"})})
    provider, _ = _provider(policy)
    d = provider.disclosure()
    assert d.provider == "synthetic-cloud"
    assert set(d.exclusions) == {"faces", "gps"}
    assert "faces" in d.leaves_machine.never


# ---------------------------------------------------------------------------
# T2: exclusion policy
# ---------------------------------------------------------------------------


def test_exclusion_policy_no_exclusions_allows_everything() -> None:
    policy = ExclusionPolicy()
    assert policy.allows(None, "asset", COMPONENT_FACES) is True
    assert policy.allows("srcA", "asset", COMPONENT_GPS) is True
    assert policy.allows("srcA", "asset", COMPONENT_FULL_RESOLUTION) is True


def test_exclusion_policy_per_source_full_resolution_excluded() -> None:
    policy = ExclusionPolicy(per_source={"srcA": frozenset({COMPONENT_FULL_RESOLUTION})})
    assert policy.allows("srcA", "asset", COMPONENT_FULL_RESOLUTION) is False
    assert policy.allows("srcA", "asset", COMPONENT_FACES) is True
    assert policy.allows("srcB", "asset", COMPONENT_FULL_RESOLUTION) is True


def test_exclusion_policy_per_image_faces_gps_excluded() -> None:
    policy = ExclusionPolicy(per_image={"img1": frozenset({"faces", "gps"})})
    assert policy.allows(None, "img1", "faces") is False
    assert policy.allows(None, "img1", "gps") is False
    assert policy.allows(None, "img1", COMPONENT_FULL_RESOLUTION) is True
    assert policy.allows(None, "img2", "faces") is True


def test_exclusion_policy_all_excludes_everything() -> None:
    policy = ExclusionPolicy(per_source={"srcA": frozenset({"all"})})
    assert policy.allows("srcA", "asset", COMPONENT_FACES) is False
    assert policy.allows("srcA", "asset", COMPONENT_GPS) is False
    assert policy.allows("srcA", "asset", COMPONENT_FULL_RESOLUTION) is False
    assert policy.allows("srcB", "asset", COMPONENT_FACES) is True


def test_exclusion_policy_merges_source_and_image() -> None:
    policy = ExclusionPolicy(
        per_source={"srcA": frozenset({"faces"})},
        per_image={"img1": frozenset({"gps"})},
    )
    assert policy.allows("srcA", "img1", "faces") is False
    assert policy.allows("srcA", "img1", "gps") is False
    assert policy.allows("srcA", "img2", "gps") is True


# ---------------------------------------------------------------------------
# T3: cloud provider payload shape + credential guarantee
# ---------------------------------------------------------------------------


def test_cloud_provider_capabilities_and_probe() -> None:
    provider, _ = _provider()
    caps = provider.capabilities()
    assert caps.air_gapped is False
    assert caps.deterministic is False
    assert "cloud" in caps.backends
    assert {"semantic", "composition", "pairing", "taste"} <= caps.stages
    assert provider.probe().ok is True


def test_cloud_analyze_sends_only_downscaled_derivative_and_metadata() -> None:
    provider, runtime = _provider()
    original = make_image_bytes(800, 600)
    result = provider.analyze(
        original, profile=BALANCED, asset_id="img_abc", source_id="srcA"
    )
    assert len(runtime.received_payloads) == 1
    derivative_bytes, meta = runtime.received_payloads[0]
    # The payload is NOT the original bytes.
    assert derivative_bytes != original
    assert len(derivative_bytes) < len(original)
    assert meta["asset_id"] == "img_abc"
    assert meta["source_id"] == "srcA"
    assert meta["profile"] == BALANCED.value
    assert "original_image" not in meta and "secrets" not in meta
    # Result is a valid, round-trippable AnalysisResult.
    assert isinstance(result, AnalysisResult)
    restored = AnalysisResult.from_dict(result.to_dict())
    assert restored.asset_id == "img_abc"


def test_cloud_analyze_excludes_full_resolution_payload() -> None:
    policy = ExclusionPolicy(
        per_source={"srcA": frozenset({COMPONENT_FULL_RESOLUTION})}
    )
    provider, runtime = _provider(policy)
    provider.analyze(
        make_image_bytes(800, 600), profile=BALANCED, asset_id="img", source_id="srcA"
    )
    _, meta = runtime.received_payloads[0]
    # Downscaled derivative only; no source-resolution claim and size is reduced.
    assert "source_resolution" not in meta
    assert runtime.received_payloads[0][0] != make_image_bytes(800, 600)


def test_cloud_analyze_drops_faces_and_gps_components() -> None:
    policy = ExclusionPolicy(
        per_source={"srcA": frozenset({"faces", "gps"})}
    )
    provider, runtime = _provider(policy)
    provider.analyze(
        img_bytes_with_gps(), profile=BALANCED, asset_id="img", source_id="srcA"
    )
    _, meta = runtime.received_payloads[0]
    assert "faces" not in meta
    assert "gps" not in meta


def test_cloud_analyze_includes_faces_and_gps_when_allowed() -> None:
    provider, runtime = _provider()
    provider.analyze(
        img_bytes_with_gps(), profile=BALANCED, asset_id="img", source_id="srcA"
    )
    _, meta = runtime.received_payloads[0]
    assert "faces" in meta
    assert "gps" in meta


def test_credentials_never_sent_to_any_provider() -> None:
    policy = ExclusionPolicy(
        per_image={"img": frozenset({"gps"})},
    )
    provider, runtime = _provider(policy)
    provider.analyze(
        make_image_bytes(), profile=BALANCED, asset_id="img", source_id="srcA"
    )
    all_bytes = b"".join(p for p, _ in runtime.received_payloads)
    all_meta = repr([m for _, m in runtime.received_payloads])
    assert TEST_SECRET.encode() not in all_bytes
    assert TEST_SECRET not in all_meta


# ---------------------------------------------------------------------------
# T4: outage + hybrid routing
# ---------------------------------------------------------------------------


def test_cloud_analyze_raises_provider_outage_when_down() -> None:
    provider, runtime = _provider()
    runtime.set_down(True)
    with pytest.raises(ProviderOutageError):
        provider.analyze(make_image_bytes(), profile=BALANCED, asset_id="img")
    assert runtime.received_payloads == []


def test_routing_defaults() -> None:
    router = HybridRouter(FakeLocalProvider(), CloudAnalysisProvider())
    for kind in ("duplicate_detection", "technical", "embeddings", "saliency", "color_story"):
        assert router.route(kind) is router.local_provider
    for kind in ("semantic", "composition", "pairing", "taste"):
        assert router.route(kind) is router.cloud_provider
    # Unknown kinds safely default to local (air-gapped).
    assert router.route("unknown_kind") is router.local_provider


def test_routing_deterministic_and_override() -> None:
    router = HybridRouter(
        FakeLocalProvider(), CloudAnalysisProvider(), policy={"taste": "local"}
    )
    assert router.route("taste") is router.local_provider
    assert router.route("pairing") is router.cloud_provider
    assert router.route("taste") is router.route("taste")


def test_router_cloud_kind_degrades_to_local_on_outage() -> None:
    local = FakeLocalProvider()
    cloud_provider, runtime = _provider()
    runtime.set_down(True)
    router = HybridRouter(local, cloud_provider)
    result = router.run("taste", make_image_bytes(), profile=MAX, asset_id="img")
    # No exception to the caller; the cloud call was satisfied locally.
    assert result.asset_id == "img"
    assert router.pause_count == 1
    assert router.pauses[0].kind == "taste"
    assert router.pauses[0].degraded_to == "local"
    assert local.calls and local.calls[-1][1] == "img"


def test_router_local_kinds_unaffected_by_cloud_outage() -> None:
    local = FakeLocalProvider()
    cloud_provider, runtime = _provider()
    runtime.set_down(True)
    router = HybridRouter(local, cloud_provider)
    result = router.run("technical", make_image_bytes(), profile=MAX, asset_id="tech")
    assert result.asset_id == "tech"
    assert router.pause_count == 0
    assert len(local.calls) == 1


def test_router_cloud_kind_success_uses_cloud() -> None:
    local = FakeLocalProvider()
    cloud_provider, runtime = _provider()
    router = HybridRouter(local, cloud_provider)
    result = router.run("pairing", make_image_bytes(), asset_id="pair")
    assert result.metadata.compute_backend == "cloud"
    assert router.pause_count == 0
    assert len(runtime.received_payloads) == 1
