"""Acceptance gate for the M010 Arbitrary Packing milestone (R044-R047).

This module ships the **11th** deterministic, air-gapped acceptance file
(M010/S06). Each scenario is **self-bootstrapping**: it mints its own sources,
analysis fixtures and vectors over the isolated ``data_root`` (from conftest) and
drives the real subsystem objects — ``propose_treatments`` /
``materialize_manifest``, ``DeterministicRenderer``, ``ArtifactValidator``,
``packing``, ``select_group`` — plus the real CLI in-process through
``acceptance_harness.run_cli``. Never a live server, never a subprocess, never
the network, and never a model download: the grouping scenario hand-seeds
vectors into an ``EmbeddingStore`` rather than running inference at all.

* Scenario A (R044) — the manifest carries real, invariant-checked geometry:
  triptych/quad/packed cells tile the canvas exactly, ``validate()`` bites on all
  three M010/S01 invariants, a legacy all-zero-region manifest still validates
  *and* still renders, and ``ArtifactValidator(source_regions=...)`` passes a real
  render while failing a hand-made overlapping pair.
* Scenario B (R044) — N cells render: exact dims at 1080p and 4K with every
  source reported (not the first two), a 3-source ``DIPTYCH`` rejected loudly, and
  the R008 upscale gate firing through **both** the letterbox path and the
  crop-to-fill path — with a control proving the fill scale is what trips it.
* Scenario C (R046) — determinism and engine purity: byte-identical manifests and
  byte-identical rendered PNG bytes on repeat at 1080p **and** 4K, the stated
  ``ceil(N/2)`` tie-break for N in 2..9, and an **AST-parsed** purity scan (never a
  substring scan, which would trip on the rule's own statement of itself — D027).
* Scenario D (R045) — bounded-pool grouping and honest degradation: a closer
  vector outside the pool is never returned, another ``model_version`` is
  invisible, a zero-norm row never surfaces as the most similar member, the two
  affinity signals stay named apart (Parallel-not-Replace), and an empty store
  reports unavailable while a caller-supplied group still proposes and renders.
* Scenario E (R047) — reachability, strictly stronger than M009's. M009 asserts
  ``symbol in cli_source or symbol in api_source``; R047's whole point is that the
  API is *not* sufficient, so this scenario never reads ``api.py`` and asserts, as
  data, that it cannot.
* Scenario F — the air gap plus the full user loop through the real CLI at N=2,
  N=3 and N=5: propose -> manifest -> render -> validate, exit 0 at every step.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from acceptance_harness import assert_no_network_imports, run_cli
from analysis_factory import crop_risky_result, make_result
from curator.artdirection.manifest import (
    CROP_FILL,
    MAX_LAYOUT_SOURCES,
    ArtDirectionManifest,
    LayoutTreatment,
    ManifestError,
    ProcessingIntent,
    SourceRegion,
)
from curator.artdirection.packing import (
    Cell,
    equal_cells,
    gutter_for_target,
    resolve_regions,
)
from curator.artdirection.policy import (
    ArtDirectionRequest,
    TreatmentProposal,
    materialize_manifest,
    propose_treatments,
)
from curator.catalog import Catalog
from curator.hashing import sha256_hex
from curator.render.renderer import DeterministicRenderer, RenderError
from curator.render.validate import ArtifactValidator
from curator.taste.embedding.grouping import AFFINITY_SOURCE, select_group
from curator.taste.embedding.provider import EMBEDDING_DIM
from curator.taste.embedding.store import EmbeddingStore

TARGET_1080P = (1920, 1080)
TARGET_4K = (3840, 2160)

#: Every scenario's default source size.
#:
#: Chosen so a crop-to-fill cell never upscales at **either** target: the tallest
#: 4K cell a triptych produces is 2160px, and the fill scale is
#: ``max(cell_w / sw, cell_h / sh)``, so a source shorter than 2160 would trip the
#: R008 gate at 4K and mask what these scenarios are actually asserting. The
#: upscale gate gets its own deliberately-undersized fixtures in Scenario B.
SOURCE_SIZE = (2400, 2400)

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# fixtures — every scenario builds its own world
# ---------------------------------------------------------------------------


def _png(width: int, height: int, color: tuple[int, int, int]) -> tuple[str, bytes]:
    """Return ``(sha256, PNG bytes)`` for a solid deterministic image.

    ``tests/test_renderer.py``'s ``make_source`` idiom: the content sha is the
    identity every manifest, region and render result is keyed by, so it is
    derived from the bytes rather than invented.
    """
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    return sha256_hex(data), data


def _sources(
    count: int, size: tuple[int, int] = SOURCE_SIZE
) -> tuple[list[str], dict[str, bytes]]:
    """Mint *count* distinct sources; return their shas in order plus the bytes map."""
    shas: list[str] = []
    payloads: dict[str, bytes] = {}
    for index in range(count):
        sha, data = _png(size[0], size[1], (40 + 20 * index, 90, 170))
        shas.append(sha)
        payloads[sha] = data
    return shas, payloads


def _analysis(
    shas: list[str], *, size: tuple[int, int] = SOURCE_SIZE, affinity: float = 0.9
) -> list:
    """One crop-safe, well-margined ``AnalysisResult`` per sha, in source order.

    ``affinity`` is what the policy engine reads from ``results[1:]`` for the
    group-cohesion gate, so 0.9 clears both ``NUP_AFFINITY`` (0.6) and
    ``DIPTYCH_AFFINITY`` (0.75) without a provider.
    """
    return [
        make_result(asset_id=sha, map_size=size, affinity=affinity) for sha in shas
    ]


def _request(
    shas: list[str], target: tuple[int, int] = TARGET_1080P, **context: object
) -> ArtDirectionRequest:
    return ArtDirectionRequest(
        target="4k" if target == TARGET_4K else "1080p",
        target_width=target[0],
        target_height=target[1],
        sources=list(shas),
        context=dict(context),
    )


def _pick(proposals: list[TreatmentProposal], treatment: LayoutTreatment) -> TreatmentProposal:
    """Return the *treatment* proposal, or fail naming what was actually proposed."""
    for proposal in proposals:
        if proposal.treatment is treatment:
            return proposal
    raise AssertionError(
        f"{treatment.value} not proposed; got {[p.treatment.value for p in proposals]}"
    )


def _materialize(
    treatment: LayoutTreatment,
    shas: list[str],
    results: list,
    target: tuple[int, int] = TARGET_1080P,
    **context: object,
) -> tuple[TreatmentProposal, ArtDirectionManifest]:
    """Drive the **real** propose -> materialize path and return both halves."""
    request = _request(shas, target, **context)
    proposal = _pick(propose_treatments(results, request), treatment)
    return proposal, materialize_manifest(proposal, request, list(shas))


def _overlap(a: SourceRegion, b: SourceRegion) -> bool:
    """True when two cells share any area (touching edges do not count)."""
    return bool(
        a.x < b.x + b.w and b.x < a.x + a.w and a.y < b.y + b.h and b.y < a.y + a.h
    )


def _assert_tiles(regions: list[SourceRegion], target: tuple[int, int]) -> None:
    """Assert *regions* are real, in-bounds, disjoint cells that tile *target* exactly."""
    tw, th = target
    assert regions
    for region in regions:
        assert region.is_unset is False, "a materialized cell must declare geometry"
        assert region.w >= 1.0 and region.h >= 1.0
        assert 0.0 <= region.x and region.x + region.w <= tw
        assert 0.0 <= region.y and region.y + region.h <= th
    for index, first in enumerate(regions):
        for second in regions[index + 1 :]:
            assert not _overlap(first, second), f"cells overlap: {first} / {second}"
    assert max(region.x + region.w for region in regions) == tw
    assert max(region.y + region.h for region in regions) == th


# ---------------------------------------------------------------------------
# Scenario A (R044) — the manifest carries real, invariant-checked geometry
# ---------------------------------------------------------------------------


def test_acceptance_packing_manifest_cells_are_real_geometry() -> None:
    """Triptych, quad and packed manifests tile the canvas with one real cell each."""
    for count, treatment in (
        (3, LayoutTreatment.TRIPTYCH),
        (4, LayoutTreatment.QUAD),
        (5, LayoutTreatment.PACKED),
    ):
        shas, _payloads = _sources(count)
        _proposal, manifest = _materialize(treatment, shas, _analysis(shas))

        assert manifest.validate() is None
        assert len(manifest.regions) == len(manifest.sources) == count
        assert [region.source_sha256 for region in manifest.regions] == shas
        _assert_tiles(manifest.regions, TARGET_1080P)


def test_acceptance_packing_manifest_invariants_bite() -> None:
    """Each of M010/S01's three invariants raises, and the message names the problem.

    All three passed silently before M010: a manifest could carry one region for
    three sources, a region naming a sha nothing else mentions, or an over-cap
    source list the renderer would then truncate.
    """
    shas, _payloads = _sources(3)
    _proposal, manifest = _materialize(LayoutTreatment.TRIPTYCH, shas, _analysis(shas))

    with pytest.raises(ManifestError) as short:
        replace(manifest, regions=manifest.regions[:1]).validate()
    assert "one region per source is required" in str(short.value)

    dangling = replace(manifest.regions[0], source_sha256="f" * 64)
    with pytest.raises(ManifestError) as unknown:
        replace(manifest, regions=[dangling, *manifest.regions[1:]]).validate()
    assert "region references source(s) not in manifest" in str(unknown.value)

    with pytest.raises(ManifestError) as over_cap:
        ArtDirectionManifest(
            sources=[f"{index:064d}" for index in range(MAX_LAYOUT_SOURCES + 1)]
        ).validate()
    message = str(over_cap.value)
    assert f"{MAX_LAYOUT_SOURCES}-source layout cap" in message
    assert "never truncated" in message


def test_acceptance_packing_legacy_all_zero_regions_still_validate_and_render() -> None:
    """Open Question #2, checked rather than assumed: unset != a zero-sized cell.

    Every ``art_direction_manifests`` row persisted before M010 carries four
    zeros per region. ``is_unset`` is what keeps that history valid, and
    ``resolve_regions`` recomputes real cells for it at render time.
    """
    shas, payloads = _sources(2)
    legacy = ArtDirectionManifest(
        sources=shas,
        regions=[SourceRegion(source_sha256=sha) for sha in shas],
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=shas,
    )
    assert all(region.is_unset for region in legacy.regions)
    assert legacy.validate() is None

    result = DeterministicRenderer().render(legacy, payloads, TARGET_1080P)
    assert (result.target_width, result.target_height) == TARGET_1080P
    assert result.sources == shas
    assert result.upscaled_warning is False
    _assert_tiles(resolve_regions(legacy, TARGET_1080P), TARGET_1080P)


def test_acceptance_packing_validator_checks_every_cell() -> None:
    """The validator's N-cell path passes a real render and fails an overlapping pair."""
    shas, payloads = _sources(3)
    _proposal, manifest = _materialize(LayoutTreatment.TRIPTYCH, shas, _analysis(shas))
    payload = DeterministicRenderer().render_bytes(manifest, payloads, TARGET_1080P)

    report = ArtifactValidator().validate(
        payload,
        sha256_hex(payload),
        TARGET_1080P,
        source_regions=resolve_regions(manifest, TARGET_1080P),
    )
    names = {check.name for check in report.checks}
    assert report.publishable is True
    assert "source_regions_disjoint" in names
    assert {f"source_region[{index}]" for index in range(3)} <= names
    assert {f"no_unintended_crop[{index}]" for index in range(3)} <= names

    overlapping = [
        SourceRegion(source_sha256=shas[0], x=0, y=0, w=1000, h=1080),
        SourceRegion(source_sha256=shas[1], x=500, y=0, w=1000, h=1080),
    ]
    bad = ArtifactValidator().validate(
        payload, sha256_hex(payload), TARGET_1080P, source_regions=overlapping
    )
    disjoint = next(c for c in bad.checks if c.name == "source_regions_disjoint")
    assert bad.publishable is False
    assert disjoint.passed is False
    assert "overlaps" in disjoint.reason


# ---------------------------------------------------------------------------
# Scenario B (R044) — N cells render, nothing dropped, nothing silently upscaled
# ---------------------------------------------------------------------------


def test_acceptance_packing_n_cells_render_at_both_targets() -> None:
    """A triptych and a quad render at exact dims with **every** source reported.

    ``RenderResult.sources`` under-reporting its own layout was the reporting
    half of the silent-truncation bug M010/S02 closed, so it is asserted here
    alongside the dimensions rather than inferred from them.
    """
    for count, treatment in ((3, LayoutTreatment.TRIPTYCH), (4, LayoutTreatment.QUAD)):
        shas, payloads = _sources(count)
        results = _analysis(shas)
        for target in (TARGET_1080P, TARGET_4K):
            _proposal, manifest = _materialize(treatment, shas, results, target)
            result = DeterministicRenderer().render(manifest, payloads, target)
            assert (result.target_width, result.target_height) == target
            assert result.sources == shas
            assert len(result.sources) == count
            assert result.treatment == treatment.value
            assert result.upscaled_warning is False


def test_acceptance_packing_over_count_diptych_is_loud() -> None:
    """The verified silent third-source drop, now an error naming both counts."""
    shas, payloads = _sources(3)
    gap = gutter_for_target(TARGET_1080P)
    manifest = ArtDirectionManifest(
        sources=shas,
        regions=equal_cells(shas, Cell(0, 0, *TARGET_1080P), gap=gap),
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=shas,
    )

    with pytest.raises(RenderError) as excinfo:
        DeterministicRenderer().render(manifest, payloads, TARGET_1080P)
    message = str(excinfo.value)
    assert "requires exactly 2 sources" in message
    assert "got 3" in message
    assert "never truncated" in message


def _two_cell_manifest(
    shas: list[str], crops: list[str | None], *, approved: bool
) -> ArtDirectionManifest:
    """A hand-authored two-cell manifest with an explicit per-cell fit each."""
    gap = gutter_for_target(TARGET_1080P)
    cells = equal_cells(shas, Cell(0, 0, *TARGET_1080P), gap=gap)
    return ArtDirectionManifest(
        sources=shas,
        regions=[
            replace(cell, crop=crop) for cell, crop in zip(cells, crops, strict=True)
        ],
        layout_treatment=LayoutTreatment.DIPTYCH,
        pairing_order=shas,
        processing_intent=ProcessingIntent(upscale_warning=approved),
    )


def test_acceptance_packing_letterbox_upscale_is_gated() -> None:
    """A cell that would upscale while letterboxing is blocked, then renders once approved."""
    small_sha, small = _png(100, 100, (10, 20, 30))
    big_sha, big = _png(2400, 2400, (200, 180, 60))
    payloads = {small_sha: small, big_sha: big}
    shas = [small_sha, big_sha]

    with pytest.raises(RenderError) as blocked:
        DeterministicRenderer().render(
            _two_cell_manifest(shas, [None, None], approved=False),
            payloads,
            TARGET_1080P,
        )
    assert "R008" in str(blocked.value)
    assert "upscale" in str(blocked.value).lower()

    approved = DeterministicRenderer().render(
        _two_cell_manifest(shas, [None, None], approved=True), payloads, TARGET_1080P
    )
    assert approved.upscaled_warning is True
    assert approved.sources == shas


def test_acceptance_packing_crop_to_fill_upscale_is_gated() -> None:
    """The gate sees the **fill** scale, and that claim is falsifiable, not asserted.

    The 2000x400 source in a 943x1080 cell is chosen because the two scales
    *disagree*: the letterbox fit scale is 0.47 (no upscale) while the fill scale
    is 2.7. Rendering the identical manifest with the fill directive removed is
    what makes the difference attributable to the crop path rather than to the
    fixture.
    """
    wide_sha, wide = _png(2000, 400, (30, 120, 90))
    big_sha, big = _png(4000, 3000, (200, 180, 60))
    payloads = {wide_sha: wide, big_sha: big}
    shas = [wide_sha, big_sha]

    with pytest.raises(RenderError) as blocked:
        DeterministicRenderer().render(
            _two_cell_manifest(shas, [CROP_FILL, None], approved=False),
            payloads,
            TARGET_1080P,
        )
    assert "R008" in str(blocked.value)

    filled = DeterministicRenderer().render(
        _two_cell_manifest(shas, [CROP_FILL, None], approved=True),
        payloads,
        TARGET_1080P,
    )
    assert filled.upscaled_warning is True

    # The control: the same sources and the same cells, letterboxed. No upscale,
    # so the fill directive is the only cause — and no approval is needed.
    letterboxed = DeterministicRenderer().render(
        _two_cell_manifest(shas, [None, None], approved=False), payloads, TARGET_1080P
    )
    assert letterboxed.upscaled_warning is False


def test_acceptance_packing_a_crop_risky_source_never_fills() -> None:
    """No proposal marks a crop-risky source ``fill``, and the manifest agrees."""
    shas, _payloads = _sources(4)
    results = _analysis(shas)
    results[2] = crop_risky_result(asset_id=shas[2])

    proposal, manifest = _materialize(LayoutTreatment.QUAD, shas, results)

    cells = proposal.evidence["cells"]
    assert len(cells) == 4
    assert cells[2]["crop"] is None
    assert cells[2]["crop_safe"] is False
    assert any(cell["crop"] == CROP_FILL for cell in cells), (
        "the fixture must contain at least one filling cell for the negative to bite"
    )
    # The materialized manifest carries exactly the proposal's verdicts, keyed by
    # sha — a crop-safe source's `fill` never lands on the risky source's cell.
    by_sha = {region.source_sha256: region.crop for region in manifest.regions}
    assert by_sha == {cell["sha"]: cell["crop"] for cell in cells}
    assert by_sha[shas[2]] is None


# ---------------------------------------------------------------------------
# Scenario C (R046) — determinism and engine purity
# ---------------------------------------------------------------------------


def test_acceptance_packing_is_byte_identical_at_1080p_and_4k() -> None:
    """Identical sources and weights produce identical manifests and identical bytes.

    Asserted at **both** targets rather than one, extending
    ``test_acceptance_render.py``'s ``test_render_determinism_1080p_and_4k``
    style: per-cell rounding drift can appear at one target and not the other, so
    a single-resolution check could not catch it.
    """
    shas, payloads = _sources(3)
    weights = [0.9, 0.4, 0.4]
    renderer = DeterministicRenderer()

    for target in (TARGET_1080P, TARGET_4K):
        first_proposal, first = _materialize(
            LayoutTreatment.PACKED, shas, _analysis(shas), target, weights=weights
        )
        _second_proposal, second = _materialize(
            LayoutTreatment.PACKED, shas, _analysis(shas), target, weights=weights
        )
        as_json = json.dumps(first.to_dict(), sort_keys=True)
        assert as_json == json.dumps(second.to_dict(), sort_keys=True)
        assert first_proposal.evidence["weights"] == weights

        payload_a = renderer.render_bytes(first, payloads, target)
        payload_b = renderer.render_bytes(second, payloads, target)
        assert payload_a == payload_b
        assert sha256_hex(payload_a) == sha256_hex(payload_b)
        _assert_tiles(first.regions, target)


def test_acceptance_packing_tie_break_puts_ceil_half_on_the_left() -> None:
    """Uniform weights split the list at ``ceil(N / 2)`` for every N in 2..9.

    Stated in public terms — the root cut of a landscape box is vertical, so the
    first ``ceil(N / 2)`` cells all sit strictly left of the rest. That is the
    rule ``equal_cells`` documented in M010/S01 and the one M010/S03's weighted
    bisection was chosen to reproduce.
    """
    gap = gutter_for_target(TARGET_1080P)
    for count in range(2, MAX_LAYOUT_SOURCES + 1):
        shas = [f"{index:064d}" for index in range(count)]
        regions = equal_cells(shas, Cell(0, 0, *TARGET_1080P), gap=gap)
        split = math.ceil(count / 2)
        boundary = regions[split].x
        assert all(r.x + r.w <= boundary for r in regions[:split]), count
        assert all(r.x >= boundary for r in regions[split:]), count
        _assert_tiles(regions, TARGET_1080P)


#: Top-level modules the packing/policy engines must never import.
#:
#: ``PIL`` and ``sqlite3`` would make the geometry impure; ``random``, ``time``
#: and ``datetime`` would break determinism; ``curator.taste`` would breach the
#: locked "treatment-level taste is out of scope" boundary.
_ENGINE_BANNED_IMPORTS = ("random", "time", "datetime", "PIL", "sqlite3")

#: Search-optimizer vocabulary, checked as **identifiers** rather than substrings.
#:
#: ``packing.py``'s own module docstring states the rule ("no iteration count, no
#: temperature schedule"), so a substring scan would fail on the engine's
#: statement of the very constraint it is honoring — the D027 self-reference trap.
#: An AST identifier walk sees ``temperature = 0.9`` and never sees a docstring.
_SEARCH_LOOP_TOKENS = ("temperature", "anneal", "iterations")


def _imported_names(tree: ast.AST) -> set[str]:
    """Every module name *tree* imports, absolute and dotted."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _identifiers(tree: ast.AST) -> set[str]:
    """Every identifier *tree* binds or references (never a docstring or comment)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.keyword) and node.arg:
            names.add(node.arg)
    return names


def test_acceptance_packing_engines_are_pure() -> None:
    """``packing.py`` and ``policy.py`` import nothing that could make them impure."""
    for module in ("packing.py", "policy.py"):
        path = REPO_ROOT / "src" / "curator" / "artdirection" / module
        imported = _imported_names(ast.parse(path.read_text(encoding="utf-8")))
        offenders = sorted(
            name
            for name in imported
            if name.split(".")[0] in _ENGINE_BANNED_IMPORTS
            or name.startswith("curator.taste")
        )
        assert offenders == [], f"{module} imports {offenders}"


def test_acceptance_packing_has_no_search_optimizer() -> None:
    """The locked no-search-optimizer decision, restated as a test.

    No iteration count, no temperature schedule, no candidate-layout list: the
    packer is a single deterministic pass, and R046's byte-determinism guarantee
    is what depends on it staying that way.
    """
    path = REPO_ROOT / "src" / "curator" / "artdirection" / "packing.py"
    identifiers = _identifiers(ast.parse(path.read_text(encoding="utf-8")))
    assert not (identifiers & set(_SEARCH_LOOP_TOKENS))


# ---------------------------------------------------------------------------
# Scenario D (R045) — bounded-pool grouping and honest degradation
# ---------------------------------------------------------------------------

#: This module's own model version label. Never a real checkpoint: every vector
#: below is hand-built, so grouping is exercised with **zero** inference.
MODEL_VERSION = "acceptance-m010"


def _unit(index: int) -> np.ndarray:
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _at_cosine(value: float) -> np.ndarray:
    """A unit vector whose cosine similarity to ``_unit(0)`` is exactly *value*."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = value
    vector[1] = math.sqrt(max(0.0, 1.0 - value * value))
    return vector


def _register_content(catalog: Catalog, sha: str) -> None:
    """Insert the minimal ``content`` row an embedding row's FK requires."""
    catalog.db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (sha, 100))
    catalog.db.commit()


def test_acceptance_packing_group_is_bounded_versioned_and_nan_free(data_root) -> None:
    """One world, four traps: the pool bound, cross-version scoping, zero-norm, NaN.

    ``outside`` is numerically **identical** to the seed (perfect cosine) but is
    deliberately absent from the candidate pool; ``other_version`` is a perfect
    match stored under a different checkpoint; ``zero_norm`` is an all-zero row
    whose cosine is undefined and which would rank above everything as a NaN. The
    only correct answer is the partly-aligned pooled candidate.
    """
    catalog = Catalog(data_root=data_root)
    try:
        store = EmbeddingStore(catalog)
        seed, pooled = "a" * 64, "b" * 64
        outside, other_version, zero_norm = "c" * 64, "d" * 64, "e" * 64
        mapping: dict[str, int] = {}
        for index, (sha, vector) in enumerate(
            (
                (seed, _unit(0)),
                (pooled, _at_cosine(0.8)),
                (outside, _unit(0)),
                (zero_norm, np.zeros(EMBEDDING_DIM, dtype=np.float32)),
            ),
            start=1,
        ):
            _register_content(catalog, sha)
            store.set(sha, MODEL_VERSION, vector)
            mapping[sha] = index
        _register_content(catalog, other_version)
        store.set(other_version, "a-different-checkpoint", _unit(0))
        mapping[other_version] = 99

        pool = [pooled, other_version, zero_norm]
        selection = select_group(
            seed, pool, mapping, store, MODEL_VERSION, group_size=4
        )

        assert selection.available is True
        assert [member.sha256 for member in selection.members] == [pooled]
        assert selection.shas == [seed, pooled]
        # Every member came from the pool the caller offered, and only from it.
        assert set(selection.shas) <= set(pool) | {seed}
        cosines = selection.evidence["pairwise_cosine"]
        assert outside not in cosines, "a closer vector outside the pool leaked in"
        assert other_version not in cosines, "another checkpoint's vector was visible"
        assert zero_norm not in cosines, "an undefined cosine was ranked"
        assert not any(math.isnan(value) for value in cosines.values())
        assert not any(math.isnan(member.similarity) for member in selection.members)
        assert selection.evidence["model_version"] == MODEL_VERSION
    finally:
        catalog.db.close()


def test_acceptance_packing_the_two_affinity_signals_stay_named_apart(data_root) -> None:
    """Parallel-not-Replace, reconciled in one place.

    Grouping answers *which* images belong together and says so
    (``embedding_cosine``); the policy engine independently answers *whether a
    template applies* and says so (``pairing.affinity``) — on the packed proposal
    **and** on the diptych, whose score is itself an affinity. The two are never
    blended into one number and never share a label.
    """
    catalog = Catalog(data_root=data_root)
    try:
        store = EmbeddingStore(catalog)
        seed, pooled = "a" * 64, "b" * 64
        for sha, vector in ((seed, _unit(0)), (pooled, _at_cosine(0.95))):
            _register_content(catalog, sha)
            store.set(sha, MODEL_VERSION, vector)
        selection = select_group(
            seed, [pooled], {seed: 1, pooled: 2}, store, MODEL_VERSION
        )
        assert selection.evidence["affinity_source"] == AFFINITY_SOURCE == "embedding_cosine"
    finally:
        catalog.db.close()

    shas, _payloads = _sources(2)
    proposals = propose_treatments(_analysis(shas), _request(shas))
    diptych = _pick(proposals, LayoutTreatment.DIPTYCH).evidence
    packed = _pick(proposals, LayoutTreatment.PACKED).evidence
    assert diptych["affinity_source"] == "pairing.affinity"
    assert packed["affinity_source"] == "pairing.affinity"
    # No embedding signal ever reaches an art-direction proposal.
    assert AFFINITY_SOURCE not in json.dumps(proposals[0].evidence)
    assert AFFINITY_SOURCE not in json.dumps(diptych)


def test_acceptance_packing_with_no_vectors_reports_unavailable(data_root) -> None:
    """No embeddings stored: honest "not yet", never a fabricated group.

    And the fallback the reason itself names must actually work — a
    caller-supplied group of the same shas still proposes, materializes and
    renders, so auto-grouping is a convenience and never a dependency.
    """
    shas, payloads = _sources(3)
    catalog = Catalog(data_root=data_root)
    try:
        for sha in shas:
            _register_content(catalog, sha)
        store = EmbeddingStore(catalog)
        mapping = {sha: index for index, sha in enumerate(shas, start=1)}

        selection = select_group(shas[0], shas[1:], mapping, store, MODEL_VERSION)

        assert selection.available is False
        assert selection.members == []
        assert selection.reason
        assert selection.shas == [shas[0]]
        assert selection.evidence["selected_group_size"] == 0
    finally:
        catalog.db.close()

    _proposal, manifest = _materialize(
        LayoutTreatment.TRIPTYCH, shas, _analysis(shas)
    )
    assert manifest.validate() is None
    _assert_tiles(manifest.regions, TARGET_1080P)
    result = DeterministicRenderer().render(manifest, payloads, TARGET_1080P)
    assert result.sources == shas
    assert (result.target_width, result.target_height) == TARGET_1080P


def test_acceptance_packing_artdirection_imports_nothing_from_taste() -> None:
    """The geometry engine never learns taste, enforced by an AST import scan.

    Two ``artdirection`` docstrings *state* this boundary in prose, so a
    substring grep would trip on the rule's own statement of itself (D027). The
    real invariant is what each module imports.
    """
    package = REPO_ROOT / "src" / "curator" / "artdirection"
    offenders: list[str] = []
    for module in sorted(package.glob("*.py")):
        for name in _imported_names(ast.parse(module.read_text(encoding="utf-8"))):
            if "taste" in name or "embedding" in name:
                offenders.append(f"{module.name}: {name}")
    assert offenders == []


# ---------------------------------------------------------------------------
# Scenario E (R047) — reachability, strictly stronger than M009's
# ---------------------------------------------------------------------------

#: The surfaces a human can actually drive. **This tuple is the scenario.**
#:
#: M009's Scenario F asserts ``symbol in cli_source or symbol in api_source``
#: (``tests/test_acceptance_taste_embedding.py``). R047 exists precisely because
#: the API is *not* sufficient — diptych was API-only for three milestones — so an
#: ``or api.py`` clause here would let the exact gap this milestone was born to
#: close pass green. The exclusion is asserted below as a property of this data
#: structure rather than as a substring scan of this file's own source, which
#: could not distinguish the check from the thing being checked.
_NON_API_SURFACES = ("src/curator/cli.py", "webui/app.js")

#: ``api.py`` is skipped when walking ``cli.py``'s import closure, and the skip is
#: load-bearing: ``cli.py`` really does ``from curator import api as api_module``
#: (for ``curator headless start``), so without it the closure would swallow
#: ``api.py`` and this scan would decay into M009's ``or api_source`` clause.
_REACHABILITY_EXCLUDED = "curator.api"

#: M010 symbols a user names directly on a surface: treatments they type after
#: ``--treatment``, flags they pass, and the functions the CLI calls by name.
_SURFACE_SYMBOLS = (
    "resolve_regions",
    "select_group",
    "triptych",
    "quad",
    "packed",
    "source_regions",
    "weights",
    "diptych",
)

#: M010 symbols reached *through* the policy engine the CLI drives. A user never
#: types ``slice_cells``; ``curator manifest`` calls ``materialize_manifest``,
#: which calls it. Proved by a real call site inside the api-free import closure —
#: a stronger claim than "the name appears somewhere", since a docstring mention
#: cannot satisfy an ``ast.Call`` scan.
_ENGINE_SYMBOLS = ("equal_cells", "slice_cells")


def _called_names(tree: ast.AST) -> set[str]:
    """Every name *tree* actually calls (never a mention in prose)."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def _defined_names(tree: ast.AST) -> set[str]:
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _module_source(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin).read_text(encoding="utf-8")


def _cli_import_closure() -> dict[str, tuple[set[str], set[str]]]:
    """``{module: (called names, defined names)}`` reachable from ``cli.py``, api-free."""
    index: dict[str, tuple[set[str], set[str]]] = {}
    frontier = ["curator.cli"]
    while frontier:
        name = frontier.pop()
        if name in index:
            continue
        if name == _REACHABILITY_EXCLUDED or name.startswith(f"{_REACHABILITY_EXCLUDED}."):
            continue
        source = _module_source(name)
        if source is None:
            continue
        tree = ast.parse(source)
        index[name] = (_called_names(tree), _defined_names(tree))
        for imported in _imported_names(tree):
            if imported.split(".")[0] == "curator":
                frontier.append(imported)
    return index


def _production_callers(
    symbol: str, index: dict[str, tuple[set[str], set[str]]]
) -> list[str]:
    """Modules in the api-free CLI closure that call *symbol* without defining it."""
    return sorted(
        module
        for module, (called, defined) in index.items()
        if symbol in called and symbol not in defined
    )


def test_acceptance_packing_reachability_never_consults_the_api() -> None:
    """Every M010 symbol is reachable from a surface a human can drive — never api.py.

    Three tiers, each stated for a reason: the surface list is asserted as data
    so it cannot silently grow an API entry; the vocabulary symbols must appear on
    a surface; and the engine symbols must have a real call site inside the
    import closure rooted at ``cli.py`` with ``api.py`` excluded.
    """
    # 1. The escape hatch does not exist, asserted structurally.
    assert not any(surface.endswith("api.py") for surface in _NON_API_SURFACES)
    assert len(_NON_API_SURFACES) == len(set(_NON_API_SURFACES))

    sources = {
        surface: REPO_ROOT.joinpath(*surface.split("/")).read_text(encoding="utf-8")
        for surface in _NON_API_SURFACES
    }
    surface_text = "\n".join(sources.values())
    cli_source = sources["src/curator/cli.py"]

    # 2. Vocabulary: what a user types, or what the surface calls by name.
    for symbol in _SURFACE_SYMBOLS:
        assert symbol in surface_text, (
            f"{symbol} is unreachable from {list(_NON_API_SURFACES)}"
        )

    # The two named gaps this milestone opened with, closed:
    # `grep -c diptych src/curator/cli.py` returned 0, and the validator's region
    # path had no production caller outside tests/ (research finding N8).
    assert "diptych" in cli_source
    assert "source_regions" in cli_source

    # 3. Engine symbols: a real call site in the api-free closure.
    index = _cli_import_closure()
    assert _REACHABILITY_EXCLUDED not in index
    assert "curator.artdirection.policy" in index
    for symbol in _ENGINE_SYMBOLS:
        callers = _production_callers(symbol, index)
        assert callers, f"{symbol} has no production caller reachable from cli.py"

    # Non-vacuity: the scan can fail. A name M010 never added is reachable from
    # neither tier, so a green run above is evidence rather than a tautology.
    absent = "a_symbol_m010_never_added"
    assert absent not in surface_text
    assert _production_callers(absent, index) == []


def test_acceptance_packing_region_geometry_has_a_runtime_writer() -> None:
    """Stronger than any source scan: exercise the writer instead of matching it.

    A manifest materialized through the **real** policy engine declares geometry
    on every cell, and a crop-safe source in a mismatched cell materializes with
    ``crop="fill"`` — ``SourceRegion.crop``'s first real writer, run rather than
    grepped.
    """
    shas, _payloads = _sources(4)
    _proposal, manifest = _materialize(LayoutTreatment.QUAD, shas, _analysis(shas))

    assert manifest.regions
    assert all(region.is_unset is False for region in manifest.regions)

    filled = [region for region in manifest.regions if region.crop == CROP_FILL]
    assert filled, "no cell exercised SourceRegion.crop's production writer"
    for region in filled:
        cell_aspect = region.w / region.h
        # The fixture's sources are square; every quad cell departs from that by
        # far more than CELL_CROP_ASPECT_TOLERANCE, which is why they fill.
        assert abs(cell_aspect - 1.0) > 0.1


# ---------------------------------------------------------------------------
# Scenario F — the air gap, and the full user loop through the real CLI
# ---------------------------------------------------------------------------

#: Model-checkpoint suffixes, assembled from parts at import time.
#:
#: Deliberately not spelled as whole literals: a literal checkpoint extension
#: written out in this module would itself be a string constant ending in a
#: checkpoint suffix, so the scan below would fail on its own banned-token list —
#: the D027 self-reference trap, avoided the same way `_SEARCH_LOOP_TOKENS` is.
_CHECKPOINT_SUFFIXES = tuple(
    "." + extension
    for extension in ("onnx", "pt", "pth", "safetensors", "ckpt", "bin")
)

#: Network client modules. Checked against this file's **imports**, never against
#: its text, so this tuple cannot trip the check it defines.
_NETWORK_BANNED = ("requests", "urllib", "huggingface_hub", "socket", "httpx", "aiohttp")


def test_acceptance_packing_module_is_air_gapped() -> None:
    """This gate never reaches the network and never names a model checkpoint."""
    assert_no_network_imports()

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    tops = {name.split(".")[0] for name in _imported_names(tree)}
    assert not (tops & set(_NETWORK_BANNED))

    checkpoints = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.endswith(_CHECKPOINT_SUFFIXES)
    ]
    assert checkpoints == []


def _kin_folder(tmp_path: Path, count: int, name: str, tint: int) -> list[str]:
    """Ingest *count* near-kin frames and return their resolved asset paths.

    Deliberate near-kin frames (shared palette and subject, a per-frame marker
    for a distinct sha) so ``LocalAnalysisProvider`` derives a genuine
    ``pairing.affinity`` above the N-up gate — real analysis, never a seeded row.
    ``tint`` keeps each folder's frames distinct from every other folder's.
    """
    folder = tmp_path / name
    folder.mkdir()
    paths: list[str] = []
    for index in range(count):
        asset = folder / f"frame{index}.png"
        img = Image.new("RGB", (1600, 1200), (60, 90 + tint, 170))
        draw = ImageDraw.Draw(img)
        draw.rectangle([200, 150, 900, 900], fill=(210, 180, 60))
        draw.rectangle([10, 10, 20 + index, 20], fill=(0, 0, 0))
        img.save(asset)
        paths.append(str(asset.resolve()))
    assert run_cli(["ingest", str(folder)])[0] == 0
    return paths


def test_acceptance_packing_full_cli_loop_at_n2_n3_and_n5(data_root, tmp_path) -> None:
    """propose -> manifest -> render -> validate, in-process, exit 0 at every step.

    N=2 is diptych's end-to-end reachability proof (API-only before M010/S02) and
    N=5 is the general packer's (no named template exists at that count).
    """
    renderer = DeterministicRenderer()
    for tint, (count, treatment) in enumerate(
        ((2, "diptych"), (3, "triptych"), (5, "packed"))
    ):
        assets = _kin_folder(tmp_path, count, f"kin{count}", tint * 7)

        rc, out = run_cli(["propose", *assets, "--json"])
        assert rc == 0, f"propose failed at N={count}"
        assert treatment in {p["treatment"] for p in json.loads(out)}

        rc, out = run_cli(["manifest", *assets, "--treatment", treatment, "--json"])
        assert rc == 0, f"manifest failed at N={count}"
        document = json.loads(out)
        assert document["layout_treatment"] == treatment
        assert document["sources"] == assets
        manifest = ArtDirectionManifest.from_dict(document)
        assert len(manifest.regions) == count
        _assert_tiles(manifest.regions, TARGET_1080P)

        manifest_path = tmp_path / f"manifest-{count}.json"
        manifest_path.write_text(json.dumps(document), encoding="utf-8")

        rc, out = run_cli(["render", str(manifest_path), "--target", "1080p", "--json"])
        assert rc == 0, f"render failed at N={count}"
        rendered = json.loads(out)
        assert (rendered["target_width"], rendered["target_height"]) == TARGET_1080P
        assert rendered["treatment"] == treatment
        assert rendered["sources"] == assets
        assert rendered["upscaled_warning"] is False

        # `curator render` prints a summary rather than writing the artifact, so
        # the bytes are materialized in-process and proved to be the CLI's own by
        # sha before `curator validate` gates them (M010/S01's idiom).
        payload = renderer.render_bytes(
            manifest.resolved_for("1080p"),
            {source: Path(source).read_bytes() for source in manifest.sources},
            TARGET_1080P,
        )
        assert sha256_hex(payload) == rendered["sha256"]
        artifact = tmp_path / f"artifact-{count}.png"
        artifact.write_bytes(payload)

        rc, out = run_cli(
            [
                "validate",
                str(artifact),
                "--expected-sha",
                rendered["sha256"],
                "--target",
                "1080p",
                "--manifest",
                str(manifest_path),
                "--json",
            ]
        )
        assert rc == 0, f"validate failed at N={count}"
        report = json.loads(out)
        assert report["publishable"] is True
        names = {check["name"] for check in report["checks"]}
        assert {f"source_region[{index}]" for index in range(count)} <= names
        assert {f"no_unintended_crop[{index}]" for index in range(count)} <= names
        assert "source_regions_disjoint" in names
