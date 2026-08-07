"""Tests for the deterministic art-direction policy engine (M002/S03 T2+T3)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from analysis_factory import (
    crop_risky_result,
    crop_safe_result,
    paired_result,
    square_result,
    wide_result,
)
from curator.analysis.local import LocalAnalysisProvider
from curator.artdirection.manifest import (
    MANIFEST_VERSION,
    ArtDirectionManifest,
    LayoutTreatment,
    ManifestError,
)
from curator.artdirection.policy import (
    _TREATMENT_RANK,
    DIPTYCH_AFFINITY,
    NUP_AFFINITY,
    PANORAMIC_MIN_ASPECT,
    SQUARE_ASPECT_MAX,
    SQUARE_ASPECT_MIN,
    ArtDirectionRequest,
    TreatmentProposal,
    materialize_manifest,
    propose,
    propose_treatments,
)


def _request(
    source_shas: list[str],
    *,
    allow_diptych: bool = True,
    width: int = 1920,
    height: int = 1080,
    target: str = "1080p",
) -> ArtDirectionRequest:
    return ArtDirectionRequest(
        target=target,
        target_width=width,
        target_height=height,
        sources=source_shas,
        allow_diptych=allow_diptych,
    )


def _first(
    results: list[TreatmentProposal], treatment: LayoutTreatment
) -> TreatmentProposal | None:
    return next((p for p in results if p.treatment is treatment), None)


def test_crop_safe_high_aesthetic_proposes_fullbleed() -> None:
    req = _request(["aaa111"])
    proposals = propose_treatments(crop_safe_result(), req)
    fb = _first(proposals, LayoutTreatment.SINGLE_FULLBLEED)
    assert fb is not None
    assert fb.rationale
    assert fb.evidence.get("min_margin", 0.0) >= 0.05
    assert fb.evidence.get("aesthetic_quality") is not None


def test_crop_risky_proposes_contain_matte() -> None:
    req = _request(["bbb222"])
    proposals = propose_treatments(crop_risky_result(), req)
    cm = _first(proposals, LayoutTreatment.CONTAIN_MATTE)
    assert cm is not None
    assert cm.evidence.get("unsafe_directions") == ["south"]
    enhancer = propose_treatments(crop_safe_result(), req)
    assert _first(enhancer, LayoutTreatment.SINGLE_FULLBLEED) is not None


def test_risky_does_not_propose_fullbleed() -> None:
    req = _request(["bbb222"])
    proposals = propose_treatments(crop_risky_result(), req)
    assert _first(proposals, LayoutTreatment.SINGLE_FULLBLEED) is None


def test_wide_aspect_proposes_panoramic() -> None:
    req = _request(["ccc333"])
    proposals = propose_treatments(wide_result(), req)
    pan = _first(proposals, LayoutTreatment.PANORAMIC)
    assert pan is not None
    aspect = pan.evidence["aspect"]
    assert aspect >= PANORAMIC_MIN_ASPECT


def test_near_square_proposes_square() -> None:
    req = _request(["ddd444"])
    proposals = propose_treatments(square_result(), req)
    sq = _first(proposals, LayoutTreatment.SQUARE)
    assert sq is not None
    aspect = sq.evidence["aspect"]
    assert SQUARE_ASPECT_MIN <= aspect <= SQUARE_ASPECT_MAX


def test_diptych_high_affinity_and_allowed() -> None:
    req = _request(["eee555", "fff666"], allow_diptych=True)
    proposals = propose_treatments(
        [paired_result("a", affinity=0.9), paired_result("b", affinity=0.9)], req
    )
    assert _first(proposals, LayoutTreatment.DIPTYCH) is not None


def test_diptych_disallowed() -> None:
    req = _request(["eee555", "fff666"], allow_diptych=False)
    proposals = propose_treatments(
        [paired_result("a", affinity=0.9), paired_result("b", affinity=0.9)], req
    )
    assert _first(proposals, LayoutTreatment.DIPTYCH) is None


def test_diptych_low_affinity() -> None:
    req = _request(["eee555", "fff666"], allow_diptych=True)
    proposals = propose_treatments(
        [paired_result("a", affinity=DIPTYCH_AFFINITY - 0.4),
         paired_result("b", affinity=DIPTYCH_AFFINITY - 0.4)],
        req,
    )
    assert _first(proposals, LayoutTreatment.DIPTYCH) is None


def test_single_source_never_diptych() -> None:
    req = _request(["ggg777"], allow_diptych=True)
    proposals = propose_treatments(paired_result("a", affinity=0.99), req)
    assert _first(proposals, LayoutTreatment.DIPTYCH) is None


def test_deterministic_repeats() -> None:
    results = [crop_risky_result("x"), paired_result("b", affinity=0.9)]
    req = _request(["aaa111", "bbb222"])
    first = propose_treatments(results, req)
    second = propose_treatments(results, req)
    tuples = lambda ps: [(p.treatment, tuple(p.rationale), p.score) for p in ps]  # noqa: E731
    assert tuples(first) == tuples(second)
    assert [p.treatment for p in first] == [
        p.treatment for p in sorted(first, key=lambda p: -p.score)
    ]


def test_request_roundtrip() -> None:
    req = _request(["a", "b"], width=3840, height=2160, target="4k",
                   allow_diptych=False)
    req = ArtDirectionRequest(
        target="4k", target_width=3840, target_height=2160,
        sources=["a", "b"], allow_diptych=False, context={"note": "hi"},
    )
    assert ArtDirectionRequest.from_dict(req.to_dict()) == req


def test_proposal_roundtrip() -> None:
    prop = TreatmentProposal(
        treatment=LayoutTreatment.PANORAMIC,
        rationale=["wide aspect 4.00 >= 2.0"],
        score=0.9,
        evidence={"aspect": 4.0, "tags": ["a", "b"]},
    )
    assert TreatmentProposal.from_dict(prop.to_dict()) == prop


def _synthetic_crop_risky(path: str) -> None:
    arr = np.zeros((1080, 1920, 3), dtype=np.uint8)
    arr[0:800, 0:500] = 200
    Image.fromarray(arr).save(path)


def test_lifecycle_factory_results() -> None:
    results = [crop_risky_result("src_a")]
    req = _request(["src_a"])
    proposals = propose(["src_a"], req, analysis=results)
    prop = _first(proposals, LayoutTreatment.CONTAIN_MATTE)
    assert prop is not None
    manifest = materialize_manifest(prop, req, ["src_a"])
    assert manifest.manifest_version == MANIFEST_VERSION
    assert manifest.sources == ["src_a"]
    assert manifest.layout_treatment is LayoutTreatment.CONTAIN_MATTE
    assert manifest.rationale == prop.rationale
    assert manifest.background.background_choice == "#eeeeee"
    assert ArtDirectionManifest.from_dict(manifest.to_dict()) == manifest


def test_lifecycle_provider_synthetic(tmp_path) -> None:
    img = tmp_path / "source.png"
    _synthetic_crop_risky(str(img))
    req = _request([str(img)])
    proposals = propose([str(img)], req, provider=LocalAnalysisProvider())
    assert proposals
    prop = _first(proposals, LayoutTreatment.CONTAIN_MATTE)
    assert prop is not None
    manifest = materialize_manifest(prop, req, [str(img)])
    assert manifest.layout_treatment is LayoutTreatment.CONTAIN_MATTE
    assert ArtDirectionManifest.from_dict(manifest.to_dict()) == manifest


def test_lifecycle_diptych() -> None:
    results = [paired_result("a", affinity=0.9), paired_result("b", affinity=0.9)]
    req = _request(["a", "b"])
    proposals = propose(["a", "b"], req, analysis=results)
    prop = _first(proposals, LayoutTreatment.DIPTYCH)
    assert prop is not None
    manifest = materialize_manifest(prop, req, ["a", "b"])
    assert manifest.pairing_order == ["a", "b"]
    assert manifest.rationale == prop.rationale


# ---------------------------------------------------------------------------
# M010/S02: named N-up templates (triptych / quad)
# ---------------------------------------------------------------------------


def _group(shas: list[str], affinity: float) -> list:
    """One paired AnalysisResult per sha, all carrying *affinity*."""
    return [paired_result(sha, affinity=affinity) for sha in shas]


def test_triptych_proposed_at_three_sources_above_threshold() -> None:
    """Three coherent sources rank a triptych — the first N-up template."""
    shas = ["a", "b", "c"]
    req = _request(shas)
    proposals = propose_treatments(_group(shas, 0.9), req)
    tri = _first(proposals, LayoutTreatment.TRIPTYCH)
    assert tri is not None
    assert tri.score == 0.9
    assert tri.evidence["sources"] == 3
    assert "3 sources" in tri.rationale[0]
    assert f">= {NUP_AFFINITY}" in tri.rationale[0]


def test_triptych_not_proposed_below_threshold() -> None:
    """Below NUP_AFFINITY the group is not coherent enough for an N-up."""
    shas = ["a", "b", "c"]
    proposals = propose_treatments(
        _group(shas, NUP_AFFINITY - 0.2), _request(shas)
    )
    assert _first(proposals, LayoutTreatment.TRIPTYCH) is None


def test_quad_proposed_at_four_sources() -> None:
    """Four coherent sources rank a quad, never a triptych."""
    shas = ["a", "b", "c", "d"]
    proposals = propose_treatments(_group(shas, 0.8), _request(shas))
    quad = _first(proposals, LayoutTreatment.QUAD)
    assert quad is not None
    assert quad.evidence["sources"] == 4
    assert len(quad.evidence["cells"]) == 4
    assert _first(proposals, LayoutTreatment.TRIPTYCH) is None


def test_five_sources_propose_no_named_template_yet() -> None:
    """N=5 has no named template until M010/S03 adds PACKED."""
    shas = ["a", "b", "c", "d", "e"]
    proposals = propose_treatments(_group(shas, 0.95), _request(shas))
    assert _first(proposals, LayoutTreatment.TRIPTYCH) is None
    assert _first(proposals, LayoutTreatment.QUAD) is None


def test_nup_ignores_allow_diptych_flag() -> None:
    """``allow_diptych`` stays DIPTYCH-specific — it never gates an N-up."""
    shas = ["a", "b", "c"]
    proposals = propose_treatments(
        _group(shas, 0.9), _request(shas, allow_diptych=False)
    )
    assert _first(proposals, LayoutTreatment.DIPTYCH) is None
    assert _first(proposals, LayoutTreatment.TRIPTYCH) is not None


def test_treatment_rank_covers_every_layout_treatment() -> None:
    """The sort key is total: a rank-less treatment is a KeyError at sort time."""
    assert set(_TREATMENT_RANK) == set(LayoutTreatment)
    assert _TREATMENT_RANK[LayoutTreatment.DIPTYCH] < _TREATMENT_RANK[
        LayoutTreatment.TRIPTYCH
    ]
    assert _TREATMENT_RANK[LayoutTreatment.TRIPTYCH] < _TREATMENT_RANK[
        LayoutTreatment.QUAD
    ]
    # The four single-source members shifted by +3 but kept their relative order.
    singles = [
        LayoutTreatment.SINGLE_FULLBLEED,
        LayoutTreatment.CONTAIN_MATTE,
        LayoutTreatment.PANORAMIC,
        LayoutTreatment.SQUARE,
    ]
    ranks = [_TREATMENT_RANK[t] for t in singles]
    assert ranks == sorted(ranks)


def test_nup_evidence_names_affinity_source_and_matches_manifest_cells() -> None:
    """Evidence is machine-checkable and agrees with the materialized regions."""
    shas = ["a", "b", "c"]
    req = _request(shas)
    proposals = propose_treatments(_group(shas, 0.9), req)
    tri = _first(proposals, LayoutTreatment.TRIPTYCH)
    assert tri is not None
    assert tri.evidence["affinity_source"] == "pairing.affinity"
    assert tri.evidence["group_affinity"] == 0.9
    assert tri.evidence["pairwise_affinity"] == {"b": 0.9, "c": 0.9}
    assert tri.evidence["cut_axis"] == "vertical"
    assert tri.evidence["gap"] == 33

    cells = tri.evidence["cells"]
    assert len(cells) == 3
    assert all(cell["w"] > 0 and cell["h"] > 0 for cell in cells)

    manifest = materialize_manifest(tri, req, shas)
    assert [
        {
            "sha": region.source_sha256,
            "x": region.x,
            "y": region.y,
            "w": region.w,
            "h": region.h,
        }
        for region in manifest.regions
    ] == cells
    assert manifest.pairing_order == shas
    assert manifest.validate() is None


def test_nup_evidence_roundtrips_through_json() -> None:
    """A triptych proposal survives the proposals-table JSON round trip."""
    shas = ["a", "b", "c"]
    proposals = propose_treatments(_group(shas, 0.9), _request(shas))
    tri = _first(proposals, LayoutTreatment.TRIPTYCH)
    assert tri is not None
    assert TreatmentProposal.from_dict(tri.to_dict()) == tri


def test_materialize_rejects_diptych_with_three_sources() -> None:
    """A fixed-size template with the wrong source count is rejected, not truncated."""
    prop = TreatmentProposal(treatment=LayoutTreatment.DIPTYCH, score=0.9)
    with pytest.raises(ManifestError) as excinfo:
        materialize_manifest(prop, _request(["a", "b", "c"]), ["a", "b", "c"])
    message = str(excinfo.value)
    assert "exactly 2" in message
    assert "got 3" in message
    assert "never truncated" in message


def test_materialize_rejects_triptych_with_two_sources() -> None:
    """The same exact-count contract rejects an under-count triptych."""
    prop = TreatmentProposal(treatment=LayoutTreatment.TRIPTYCH, score=0.9)
    with pytest.raises(ManifestError):
        materialize_manifest(prop, _request(["a", "b"]), ["a", "b"])


def test_nup_deterministic_repeats() -> None:
    """Identical N-up input yields an identical, identically-ordered result."""
    shas = ["a", "b", "c", "d"]
    results = _group(shas, 0.85)
    req = _request(shas)
    first = propose_treatments(results, req)
    second = propose_treatments(results, req)
    assert [p.to_dict() for p in first] == [p.to_dict() for p in second]
