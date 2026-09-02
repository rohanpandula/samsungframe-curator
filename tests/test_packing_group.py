"""Tests for bounded-pool embedding-affinity group selection (M010/S04, R045).

Closes a Wave 0 gap: nothing in this suite exercised "which images belong
together" before this slice. Every fit here uses a real (isolated)
:class:`~curator.taste.embedding.store.EmbeddingStore` seeded with hand-built
vectors, the same fixture idiom ``tests/test_taste_embedding_attribution.py``
established — no ONNX inference and no fixture model is needed for the retrieval
half, because ``select_group`` is a pure numpy operation over stored rows.

The invariants asserted here are the ones that cannot be recovered later: the
pool is caller-bounded (a numerically closer vector *outside* it is never
returned), vectors under another ``model_version`` are invisible, a zero-norm row
on either side degrades cleanly rather than surfacing a ``NaN`` as the most
similar member, ties resolve by the stated rule rather than numpy's unspecified
one, and an absent model produces an honest "unavailable" instead of a
fabricated group.
"""

from __future__ import annotations

import ast
import io
import json
import math
import shlex
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from acceptance_harness import run_cli
from curator import cli
from curator.api import create_app
from curator.artdirection.manifest import MAX_LAYOUT_SOURCES
from curator.catalog import Catalog
from curator.cli import EXIT_FATAL, EXIT_NO_CHANGE, EXIT_OK
from curator.connectors.local import LocalConnector
from curator.taste.embedding.grouping import (
    AFFINITY_SOURCE,
    GROUP_SIMILARITY_THRESHOLD,
    MAX_CANDIDATE_POOL,
    GroupCandidate,
    GroupingError,
    GroupSelection,
    resolve_group_pool,
    select_group,
)
from curator.taste.embedding.provider import EMBEDDING_DIM, EMBEDDING_MODEL_VERSION
from curator.taste.embedding.store import EmbeddingStore

MODEL_VERSION = "v1"

FIXTURE_MODEL = Path(__file__).parent / "fixtures" / "tiny_embedding_model.onnx"


@pytest.fixture
def catalog(data_root) -> Catalog:
    """A migrated Catalog under an isolated data root (schema v17 present)."""
    cat = Catalog(data_root=data_root)
    yield cat
    cat.db.close()


@pytest.fixture
def store(catalog: Catalog) -> EmbeddingStore:
    return EmbeddingStore(catalog)


def _sha(marker: str) -> str:
    """A deterministic 64-hex content sha built from a single marker character."""
    return marker * 64


def _register_content(catalog: Catalog, sha: str) -> None:
    """Insert a minimal ``content`` row so FK-constrained embedding inserts succeed."""
    catalog.db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (sha, 100))
    catalog.db.commit()


def _unit(index: int) -> np.ndarray:
    """The *index*-th basis vector of the embedding space."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _at_cosine(value: float) -> np.ndarray:
    """A unit vector whose cosine similarity to ``_unit(0)`` is exactly *value*."""
    vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vector[0] = value
    vector[1] = math.sqrt(max(0.0, 1.0 - value * value))
    return vector


def _seed(
    catalog: Catalog,
    store: EmbeddingStore,
    rows: dict[str, np.ndarray],
    model_version: str = MODEL_VERSION,
) -> dict[str, int]:
    """Store every ``sha -> vector`` row and return the matching entry-id mapping."""
    mapping: dict[str, int] = {}
    for index, (sha, vector) in enumerate(rows.items(), start=1):
        _register_content(catalog, sha)
        store.set(sha, model_version, vector)
        mapping[sha] = index
    return mapping


def _no_nan(selection: GroupSelection) -> None:
    """Assert no ``NaN`` reached any float this selection reports."""
    assert all(not math.isnan(m.similarity) for m in selection.members)
    assert all(
        not math.isnan(value) for value in selection.evidence["pairwise_cosine"].values()
    )


# ---------------------------------------------------------------------------
# The bounded pool: members come from the pool, and only from the pool
# ---------------------------------------------------------------------------


def test_members_never_fall_outside_the_candidate_pool(catalog, store) -> None:
    seed, close, far, outside = _sha("a"), _sha("b"), _sha("c"), _sha("d")
    mapping = _seed(
        catalog,
        store,
        {
            seed: _unit(0),
            close: _at_cosine(0.99),
            far: _at_cosine(0.9),
            outside: _unit(0),
        },
    )

    selection = select_group(seed, [close, far], mapping, store, MODEL_VERSION, group_size=3)

    assert selection.available is True
    assert selection.shas == [seed, close, far]
    assert all(m.sha256 in {close, far} for m in selection.members)


def test_a_closer_vector_outside_the_pool_is_never_returned(catalog, store) -> None:
    """The bounded-pool proof, mirroring ``find_exemplars``' own exclusion test.

    ``outside`` is numerically *identical* to the seed (perfect cosine) while the
    pooled candidate is only partly aligned — the pooled one must still be the
    only answer, and the excluded one must not even appear in the evidence.
    """
    seed, pooled, outside = _sha("a"), _sha("b"), _sha("c")
    mapping = _seed(
        catalog,
        store,
        {seed: _unit(0), pooled: _at_cosine(0.8), outside: _unit(0)},
    )

    selection = select_group(seed, [pooled], mapping, store, MODEL_VERSION, group_size=3)

    assert [m.sha256 for m in selection.members] == [pooled]
    assert outside not in selection.evidence["pairwise_cosine"]
    assert selection.evidence["pool_size"] == 1


def test_the_seed_is_never_returned_as_its_own_companion(catalog, store) -> None:
    seed, other = _sha("a"), _sha("b")
    mapping = _seed(catalog, store, {seed: _unit(0), other: _at_cosine(0.9)})

    selection = select_group(seed, [seed, other], mapping, store, MODEL_VERSION)

    assert [m.sha256 for m in selection.members] == [other]
    assert seed not in selection.evidence["pairwise_cosine"]


# ---------------------------------------------------------------------------
# T-10-17: version scoping is the only cross-checkpoint detection mechanism
# ---------------------------------------------------------------------------


def test_vectors_stored_under_another_model_version_are_invisible(catalog, store) -> None:
    """A perfect match under a *different* checkpoint must not join the group.

    Two 512-dim vectors from different checkpoints are indistinguishable by shape,
    so ``get_matrix(model_version)``'s scoping is the only thing that can catch
    this — and no path in ``select_group`` bypasses it.
    """
    seed, same_version, other_version = _sha("a"), _sha("b"), _sha("c")
    mapping = _seed(catalog, store, {seed: _unit(0), same_version: _at_cosine(0.7)})
    _register_content(catalog, other_version)
    store.set(other_version, "a-different-checkpoint", _unit(0))
    mapping[other_version] = 99

    selection = select_group(
        seed, [same_version, other_version], mapping, store, MODEL_VERSION, group_size=3
    )

    assert [m.sha256 for m in selection.members] == [same_version]
    assert other_version not in selection.evidence["pairwise_cosine"]
    assert selection.evidence["model_version"] == MODEL_VERSION


def test_a_cross_version_query_returns_no_group_rather_than_a_plausible_one(
    catalog, store
) -> None:
    seed, other = _sha("a"), _sha("b")
    mapping = _seed(catalog, store, {seed: _unit(0), other: _unit(0)})

    selection = select_group(seed, [other], mapping, store, "a-different-checkpoint")

    assert selection.available is False
    assert selection.members == []
    assert selection.reason
    assert "a-different-checkpoint" in selection.reason


# ---------------------------------------------------------------------------
# T-10-18 / IN-01: zero-norm rows never become a NaN "most similar" member
# ---------------------------------------------------------------------------


def test_a_zero_norm_seed_degrades_cleanly_with_no_nan(catalog, store) -> None:
    seed, other = _sha("a"), _sha("b")
    mapping = _seed(
        catalog,
        store,
        {seed: np.zeros(EMBEDDING_DIM, dtype=np.float32), other: _unit(0)},
    )

    selection = select_group(seed, [other], mapping, store, MODEL_VERSION)

    assert selection.available is False
    assert selection.members == []
    assert "zero" in selection.reason
    _no_nan(selection)


def test_a_zero_norm_stored_row_is_excluded_before_the_division(catalog, store) -> None:
    """A stored zero vector is dropped *before* dividing, never surfaced as ``NaN``.

    ``NaN`` compares greater than everything under ``numpy.argsort`` — the trap
    this guard (and the explicit sort that replaced it) exists to close.
    """
    seed, zero_row, real = _sha("a"), _sha("b"), _sha("c")
    mapping = _seed(
        catalog,
        store,
        {
            seed: _unit(0),
            zero_row: np.zeros(EMBEDDING_DIM, dtype=np.float32),
            real: _at_cosine(0.9),
        },
    )

    selection = select_group(seed, [zero_row, real], mapping, store, MODEL_VERSION)

    assert [m.sha256 for m in selection.members] == [real]
    assert zero_row not in selection.evidence["pairwise_cosine"]
    assert selection.evidence["considered"] == 1
    _no_nan(selection)


def test_every_candidate_zero_norm_degrades_cleanly(catalog, store) -> None:
    seed, zero_a, zero_b = _sha("a"), _sha("b"), _sha("c")
    zeros = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    mapping = _seed(catalog, store, {seed: _unit(0), zero_a: zeros, zero_b: zeros})

    selection = select_group(seed, [zero_a, zero_b], mapping, store, MODEL_VERSION)

    assert selection.available is False
    assert selection.members == []
    assert "zero" in selection.reason
    _no_nan(selection)


# ---------------------------------------------------------------------------
# T-10-19 / T-10-20: caller-contract violations raise, never truncate
# ---------------------------------------------------------------------------


def test_a_pool_over_the_bound_raises_rather_than_sweeping_it(catalog, store) -> None:
    seed = _sha("a")
    mapping = {seed: 1}
    over_cap = [f"{index:064x}" for index in range(MAX_CANDIDATE_POOL + 1)]
    mapping.update({sha: 2 for sha in over_cap})

    with pytest.raises(GroupingError) as excinfo:
        select_group(seed, over_cap, mapping, store, MODEL_VERSION)

    assert str(MAX_CANDIDATE_POOL) in str(excinfo.value)
    assert "never truncated" in str(excinfo.value)


def test_a_pool_exactly_at_the_bound_is_accepted(catalog, store) -> None:
    """The bound rejects *over* the cap, not at it — an off-by-one here would
    silently shrink every pool by one candidate."""
    seed = _sha("a")
    mapping = {seed: 1}
    at_cap = [f"{index:064x}" for index in range(MAX_CANDIDATE_POOL)]
    mapping.update({sha: 2 for sha in at_cap})

    selection = select_group(seed, at_cap, mapping, store, MODEL_VERSION)

    assert selection.available is False  # empty store — but it did not raise
    assert selection.evidence["pool_size"] == MAX_CANDIDATE_POOL


@pytest.mark.parametrize("group_size", [1, 0, -1, MAX_LAYOUT_SOURCES + 1, 12])
def test_an_out_of_range_group_size_raises(catalog, store, group_size: int) -> None:
    seed = _sha("a")

    with pytest.raises(GroupingError) as excinfo:
        select_group(seed, [], {seed: 1}, store, MODEL_VERSION, group_size=group_size)

    assert f"2..{MAX_LAYOUT_SOURCES}" in str(excinfo.value)


def test_a_seed_absent_from_the_mapping_raises(catalog, store) -> None:
    with pytest.raises(GroupingError) as excinfo:
        select_group(_sha("a"), [], {}, store, MODEL_VERSION)

    assert "sha_to_entry_id" in str(excinfo.value)


def test_a_pool_sha_absent_from_the_mapping_raises(catalog, store) -> None:
    seed, unmapped = _sha("a"), _sha("b")

    with pytest.raises(GroupingError) as excinfo:
        select_group(seed, [unmapped], {seed: 1}, store, MODEL_VERSION)

    assert "sha_to_entry_id" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Honest degradation: an empty store is a report, never a crash or a fabrication
# ---------------------------------------------------------------------------


def test_an_empty_store_returns_unavailable_with_a_reason_and_no_members(store) -> None:
    seed = _sha("a")

    selection = select_group(seed, [_sha("b")], {seed: 1, _sha("b"): 2}, store, MODEL_VERSION)

    assert selection.available is False
    assert selection.members == []
    assert selection.reason
    assert selection.shas == [seed]
    assert selection.evidence["affinity_source"] == AFFINITY_SOURCE
    assert selection.evidence["selected_group_size"] == 0


def test_a_seed_with_no_stored_vector_is_unavailable_not_an_error(catalog, store) -> None:
    seed, other = _sha("a"), _sha("b")
    mapping = _seed(catalog, store, {other: _unit(0)})
    mapping[seed] = 42

    selection = select_group(seed, [other], mapping, store, MODEL_VERSION)

    assert selection.available is False
    assert "seed" in selection.reason


def test_a_pool_with_no_stored_vectors_is_unavailable(catalog, store) -> None:
    seed, other = _sha("a"), _sha("b")
    mapping = _seed(catalog, store, {seed: _unit(0)})
    mapping[other] = 42

    selection = select_group(seed, [other], mapping, store, MODEL_VERSION)

    assert selection.available is False
    assert selection.members == []


def test_no_candidate_above_the_threshold_returns_unavailable_but_shows_the_evidence(
    catalog, store
) -> None:
    """The rejected candidate is still reported — the evidence shows what was
    passed over, not only what was chosen."""
    seed, weak = _sha("a"), _sha("b")
    mapping = _seed(catalog, store, {seed: _unit(0), weak: _at_cosine(0.5)})

    selection = select_group(seed, [weak], mapping, store, MODEL_VERSION)

    assert selection.available is False
    assert selection.reason == "no candidate met the similarity threshold"
    assert selection.members == []
    assert weak in selection.evidence["pairwise_cosine"]
    assert selection.evidence["pairwise_cosine"][weak] == pytest.approx(0.5, abs=1e-6)


def test_a_shortfall_is_recorded_never_padded(catalog, store) -> None:
    """Fewer companions than requested is a valid, available answer."""
    seed, close, weak = _sha("a"), _sha("b"), _sha("c")
    mapping = _seed(
        catalog, store, {seed: _unit(0), close: _at_cosine(0.95), weak: _at_cosine(0.2)}
    )

    selection = select_group(seed, [close, weak], mapping, store, MODEL_VERSION, group_size=4)

    assert selection.available is True
    assert [m.sha256 for m in selection.members] == [close]
    assert selection.evidence["requested_group_size"] == 4
    assert selection.evidence["selected_group_size"] == 2
    assert selection.evidence["considered"] == 2


def test_more_candidates_than_requested_are_capped_at_group_size_minus_one(
    catalog, store
) -> None:
    seed = _sha("a")
    rows = {seed: _unit(0)}
    companions = [_sha(marker) for marker in "bcde"]
    for index, sha in enumerate(companions):
        rows[sha] = _at_cosine(0.99 - index * 0.01)
    mapping = _seed(catalog, store, rows)

    selection = select_group(seed, companions, mapping, store, MODEL_VERSION, group_size=3)

    assert len(selection.members) == 2
    assert selection.evidence["selected_group_size"] == 3
    assert selection.evidence["considered"] == 4


def test_a_custom_threshold_widens_the_group(catalog, store) -> None:
    seed, weak = _sha("a"), _sha("b")
    mapping = _seed(catalog, store, {seed: _unit(0), weak: _at_cosine(0.5)})

    strict = select_group(seed, [weak], mapping, store, MODEL_VERSION)
    lenient = select_group(seed, [weak], mapping, store, MODEL_VERSION, threshold=0.4)

    assert strict.available is False
    assert lenient.available is True
    assert lenient.evidence["threshold"] == 0.4


# ---------------------------------------------------------------------------
# R046: the tie-break is stated, not numpy's unspecified order
# ---------------------------------------------------------------------------


def test_identical_similarities_resolve_by_sha_ascending(catalog, store) -> None:
    seed, later, earlier = _sha("a"), _sha("c"), _sha("b")
    identical = _at_cosine(0.9)
    mapping = _seed(
        catalog, store, {seed: _unit(0), later: identical.copy(), earlier: identical.copy()}
    )

    selection = select_group(seed, [later, earlier], mapping, store, MODEL_VERSION, group_size=3)

    assert [m.sha256 for m in selection.members] == [earlier, later]


def test_a_tie_decides_which_candidate_makes_a_group_of_two(catalog, store) -> None:
    """With one slot and two exactly-equal candidates, the stated rule is what
    picks the winner — numpy's tie order would be unspecified."""
    seed, later, earlier = _sha("a"), _sha("c"), _sha("b")
    identical = _at_cosine(0.9)
    mapping = _seed(
        catalog, store, {seed: _unit(0), later: identical.copy(), earlier: identical.copy()}
    )

    selection = select_group(seed, [later, earlier], mapping, store, MODEL_VERSION, group_size=2)

    assert [m.sha256 for m in selection.members] == [earlier]


def test_repeat_calls_return_equal_selections(catalog, store) -> None:
    seed = _sha("a")
    rows = {seed: _unit(0)}
    companions = [_sha(marker) for marker in "bcd"]
    for index, sha in enumerate(companions):
        rows[sha] = _at_cosine(0.9 - index * 0.01)
    mapping = _seed(catalog, store, rows)

    first = select_group(seed, companions, mapping, store, MODEL_VERSION, group_size=4)
    second = select_group(seed, companions, mapping, store, MODEL_VERSION, group_size=4)

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_members_are_ordered_by_descending_similarity(catalog, store) -> None:
    seed = _sha("a")
    rows = {seed: _unit(0)}
    companions = [_sha(marker) for marker in "bcd"]
    for index, sha in enumerate(companions):
        rows[sha] = _at_cosine(0.7 + index * 0.1)
    mapping = _seed(catalog, store, rows)

    selection = select_group(seed, companions, mapping, store, MODEL_VERSION, group_size=4)

    similarities = [m.similarity for m in selection.members]
    assert similarities == sorted(similarities, reverse=True)
    assert selection.shas[0] == seed


# ---------------------------------------------------------------------------
# Evidence contract: named affinity source, never an opaque cluster id
# ---------------------------------------------------------------------------


def test_evidence_names_the_affinity_source_and_covers_every_considered_candidate(
    catalog, store
) -> None:
    seed = _sha("a")
    rows = {seed: _unit(0)}
    companions = [_sha(marker) for marker in "bcd"]
    for index, sha in enumerate(companions):
        rows[sha] = _at_cosine(0.9 - index * 0.2)
    mapping = _seed(catalog, store, rows)

    selection = select_group(seed, companions, mapping, store, MODEL_VERSION, group_size=3)
    evidence = selection.evidence

    assert evidence["affinity_source"] == "embedding_cosine"
    assert evidence["threshold"] == GROUP_SIMILARITY_THRESHOLD
    assert evidence["pool_size"] == 3
    assert evidence["considered"] == 3
    # Every considered candidate, including the one below the threshold that was
    # rejected — the evidence shows what was passed over.
    assert set(evidence["pairwise_cosine"]) == set(companions)
    assert len(selection.members) == 2
    assert "cluster_id" not in evidence


def test_the_selection_is_json_serializable_and_round_trips(catalog, store) -> None:
    seed, close = _sha("a"), _sha("b")
    mapping = _seed(catalog, store, {seed: _unit(0), close: _at_cosine(0.9)})

    selection = select_group(seed, [close], mapping, store, MODEL_VERSION)
    rebuilt = GroupSelection.from_dict(json.loads(json.dumps(selection.to_dict())))

    assert rebuilt == selection
    assert rebuilt.shas == selection.shas


def test_group_candidate_round_trips(catalog) -> None:
    candidate = GroupCandidate(sha256=_sha("a"), entry_id=7, similarity=0.42)

    assert GroupCandidate.from_dict(json.loads(json.dumps(candidate.to_dict()))) == candidate


# ---------------------------------------------------------------------------
# T-10-21 boundary guard: artdirection/ imports nothing from taste/
# ---------------------------------------------------------------------------


def test_artdirection_imports_nothing_from_taste_or_embedding() -> None:
    """The policy engine's purity, enforced by the suite rather than by discipline.

    Parsed from each module's AST rather than grepped as a substring: two
    ``artdirection`` docstrings *state* this boundary in prose ("no import from
    the taste or rendering package", "never an embedding"), so a naive substring
    scan would trip on the rule's own statement of itself — the same
    self-reference trap ``assert_no_network_imports`` avoids the same way (D027).
    """
    package = Path(__file__).resolve().parents[1] / "src" / "curator" / "artdirection"
    offenders: list[str] = []
    for module in sorted(package.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if "taste" in name or "embedding" in name:
                    offenders.append(f"{module.name}: {name}")
    assert offenders == []


# ---------------------------------------------------------------------------
# resolve_group_pool: the module's only database access, bounded by SQL
# ---------------------------------------------------------------------------


def test_resolve_group_pool_returns_the_pool_and_its_entry_ids(catalog, store) -> None:
    shas = [_sha(marker) for marker in "abc"]
    for sha in shas:
        _register_content(catalog, sha)
        store.set(sha, MODEL_VERSION, _unit(0))
    entry_ids = {sha: _catalog_entry(catalog, sha) for sha in shas}

    pool, mapping = resolve_group_pool(catalog, MODEL_VERSION)

    assert set(pool) == set(shas)
    assert mapping == entry_ids


def test_resolve_group_pool_skips_other_model_versions(catalog, store) -> None:
    current, other = _sha("a"), _sha("b")
    for sha in (current, other):
        _register_content(catalog, sha)
        _catalog_entry(catalog, sha)
    store.set(current, MODEL_VERSION, _unit(0))
    store.set(other, "a-different-checkpoint", _unit(0))

    pool, mapping = resolve_group_pool(catalog, MODEL_VERSION)

    assert pool == [current]
    assert other not in mapping


def test_resolve_group_pool_applies_the_limit_most_recent_first(catalog, store) -> None:
    shas = [_sha(marker) for marker in "abc"]
    ids = {}
    for sha in shas:
        _register_content(catalog, sha)
        store.set(sha, MODEL_VERSION, _unit(0))
        ids[sha] = _catalog_entry(catalog, sha)

    pool, mapping = resolve_group_pool(catalog, MODEL_VERSION, limit=2)

    assert len(pool) == 2
    # The two highest catalog_entries.id rows — most recently cataloged first.
    assert set(pool) == {shas[-1], shas[-2]}
    assert mapping[shas[-1]] == ids[shas[-1]]


def test_resolve_group_pool_never_exceeds_the_grouping_bound(catalog) -> None:
    """The SQL ``LIMIT`` carries the same bound ``select_group`` enforces, so no
    surface can ask the database for a pool grouping would then refuse."""
    with pytest.raises(GroupingError) as excinfo:
        resolve_group_pool(catalog, MODEL_VERSION, limit=MAX_CANDIDATE_POOL + 1)

    assert str(MAX_CANDIDATE_POOL) in str(excinfo.value)
    assert "never" in str(excinfo.value)


def test_resolve_group_pool_rejects_a_non_positive_limit(catalog) -> None:
    with pytest.raises(GroupingError):
        resolve_group_pool(catalog, MODEL_VERSION, limit=0)


# ---------------------------------------------------------------------------
# The two surfaces: `curator group` and POST /api/packing/group
# ---------------------------------------------------------------------------


def _use_fixture_model(monkeypatch) -> None:
    """Point the provider at the tiny committed ONNX fixture (never a download)."""
    monkeypatch.setenv("CURATOR_TASTE_EMBEDDING_MODEL_PATH", str(FIXTURE_MODEL))


def _no_model(monkeypatch) -> None:
    """Leave the isolated data root's model path empty — the honest-degradation case."""
    monkeypatch.delenv("CURATOR_TASTE_EMBEDDING_MODEL_PATH", raising=False)


def _frame_bytes(shift: int) -> bytes:
    """One near-kin frame: same composition, mean colour shifted by *shift*.

    Square so CLIP's center crop keeps the whole composition — the fixture model
    is a global average pool, so the frames' embeddings differ exactly as much as
    their mean colour does (measured cosine 0.93-0.99 across shifts 0..24).
    """
    img = Image.new("RGB", (800, 800), (60 + shift, 90, 170))
    draw = ImageDraw.Draw(img)
    draw.rectangle([200, 200, 600, 600], fill=(210, 180, 60 + shift))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _unrelated_bytes() -> bytes:
    """A frame with a deliberately different mean colour (measured cosine -0.04)."""
    buf = io.BytesIO()
    Image.new("RGB", (800, 800), (10, 220, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _catalog_entry(catalog: Catalog, sha: str) -> int:
    """Register a catalog entry for an already-inserted ``content`` row."""
    connector_id = "grouping-fixture"
    catalog.db.execute(
        "INSERT OR IGNORE INTO source_connectors(connector_id, connector_type, name)"
        " VALUES (?, 'local', 'grouping fixture')",
        (connector_id,),
    )
    cursor = catalog.db.execute(
        "INSERT INTO catalog_entries(connector_id, asset_id, revision, sha256)"
        " VALUES (?, ?, '1', ?)",
        (connector_id, f"asset-{sha[:8]}", sha),
    )
    catalog.db.commit()
    return int(cursor.lastrowid or 0)


def _write_asset(catalog: Catalog, tmp_path, name: str, data: bytes) -> tuple[str, str]:
    """Write *data* to a file, catalog it, and return ``(resolved path, sha256)``."""
    folder = tmp_path / "frames"
    folder.mkdir(exist_ok=True)
    asset = folder / name
    asset.write_bytes(data)
    connector = LocalConnector(folder)
    sha = catalog.add_source(connector.connector_id, str(asset.resolve()), data)
    return str(asset.resolve()), sha


def _seeded_surface(catalog, tmp_path):
    """Catalog three assets and store hand-built vectors under the real model version.

    Returns ``(seed_path, seed_sha, close_sha, weak_sha)``. Hand-built rather than
    inferred so the cosines are exact: ``close`` at 0.95 clears the default
    threshold and ``weak`` at 0.2 does not.
    """
    store = EmbeddingStore(catalog)
    seed_path, seed_sha = _write_asset(catalog, tmp_path, "seed.png", _frame_bytes(0))
    _, close_sha = _write_asset(catalog, tmp_path, "close.png", _frame_bytes(8))
    _, weak_sha = _write_asset(catalog, tmp_path, "weak.png", _unrelated_bytes())
    store.set(seed_sha, EMBEDDING_MODEL_VERSION, _unit(0))
    store.set(close_sha, EMBEDDING_MODEL_VERSION, _at_cosine(0.95))
    store.set(weak_sha, EMBEDDING_MODEL_VERSION, _at_cosine(0.2))
    return seed_path, seed_sha, close_sha, weak_sha


def test_cli_group_returns_a_group_and_exits_zero(data_root, tmp_path, catalog, monkeypatch):
    _use_fixture_model(monkeypatch)
    seed_path, seed_sha, close_sha, weak_sha = _seeded_surface(catalog, tmp_path)

    rc, out = run_cli(["group", seed_path, "--size", "3", "--json"])

    assert rc == EXIT_OK
    payload = json.loads(out)
    assert payload["available"] is True
    assert payload["seed_sha256"] == seed_sha
    assert [m["sha256"] for m in payload["members"]] == [close_sha]
    assert payload["evidence"]["affinity_source"] == "embedding_cosine"
    # The rejected candidate is still reported — what was passed over is visible.
    assert weak_sha in payload["evidence"]["pairwise_cosine"]


def test_cli_group_prints_the_propose_bridge(data_root, tmp_path, catalog, monkeypatch):
    """The group answers in shas; ``sources`` answers in the paths ``propose`` takes.

    M010 milestone audit: ``curator group`` and ``curator propose`` spoke
    different identities (sha vs path) with no bridge between them.
    """
    _use_fixture_model(monkeypatch)
    seed_path, _seed_sha, _close_sha, _weak = _seeded_surface(catalog, tmp_path)
    close_path = str((tmp_path / "frames" / "close.png").resolve())

    rc, out = run_cli(["group", seed_path, "--size", "3", "--json"])
    assert rc == EXIT_OK
    assert json.loads(out)["sources"] == [seed_path, close_path]

    rc, out = run_cli(["group", seed_path, "--size", "3"])
    assert rc == EXIT_OK
    assert shlex.join(["curator", "propose", seed_path, close_path]) in out


def test_cli_group_human_output_names_the_members(data_root, tmp_path, catalog, monkeypatch):
    _use_fixture_model(monkeypatch)
    seed_path, seed_sha, close_sha, _weak = _seeded_surface(catalog, tmp_path)

    rc, out = run_cli(["group", seed_path])

    assert rc == EXIT_OK
    assert seed_sha[:12] in out
    assert close_sha[:12] in out
    assert "embedding_cosine" in out


def test_cli_group_end_to_end_over_a_real_backfill(data_root, tmp_path, monkeypatch, catalog):
    """Ingest, embed for real through the fixture ONNX model, then group.

    No hand-seeded vectors anywhere on this path: ``curator taste embed-status
    --backfill`` computes every vector through the real provider, and the group
    that comes back is the one those vectors actually imply — the four near-kin
    frames, with the deliberately unrelated one rejected by the threshold.
    """
    _use_fixture_model(monkeypatch)
    folder = tmp_path / "kin"
    folder.mkdir()
    paths = []
    for index, shift in enumerate((0, 8, 16, 24)):
        asset = folder / f"frame{index}.png"
        asset.write_bytes(_frame_bytes(shift))
        paths.append(str(asset.resolve()))
    unrelated = folder / "unrelated.png"
    unrelated.write_bytes(_unrelated_bytes())

    assert run_cli(["ingest", str(folder)])[0] == EXIT_OK
    assert run_cli(["taste", "embed-status", "--backfill", "--json"])[0] == EXIT_OK

    rc, out = run_cli(["group", paths[0], "--size", "4", "--json"])

    assert rc == EXIT_OK
    payload = json.loads(out)
    assert payload["available"] is True
    assert payload["evidence"]["affinity_source"] == "embedding_cosine"
    assert payload["evidence"]["selected_group_size"] == 4
    assert len(payload["members"]) == 3
    # Descending cosine: the nearest shifts first, the unrelated frame never in.
    similarities = [m["similarity"] for m in payload["members"]]
    assert similarities == sorted(similarities, reverse=True)
    assert all(s >= GROUP_SIMILARITY_THRESHOLD for s in similarities)
    assert len(payload["evidence"]["pairwise_cosine"]) == 4
    assert min(payload["evidence"]["pairwise_cosine"].values()) < GROUP_SIMILARITY_THRESHOLD


def test_cli_group_with_no_embedding_model_exits_three_naming_propose(
    data_root, tmp_path, catalog, monkeypatch
):
    """The headline honest-degradation case: no model, no group, no crash."""
    _no_model(monkeypatch)
    seed_path, _seed_sha, _close, _weak = _seeded_surface(catalog, tmp_path)

    rc, out = run_cli(["group", seed_path, "--size", "4"])

    assert rc == EXIT_NO_CHANGE
    assert out.strip()
    assert "curator propose" in out
    assert "member" not in out


def test_cli_group_with_nothing_embedded_yet_exits_three(
    data_root, tmp_path, catalog, monkeypatch
):
    """A model but no vectors: still nothing to propose, still not an error."""
    _use_fixture_model(monkeypatch)
    seed_path, _sha256 = _write_asset(catalog, tmp_path, "lonely.png", _frame_bytes(0))

    rc, out = run_cli(["group", seed_path])

    assert rc == EXIT_NO_CHANGE
    assert "unavailable" in out
    assert "curator propose" in out


def test_cli_group_below_the_threshold_exits_three_without_a_group(
    data_root, tmp_path, catalog, monkeypatch
):
    _use_fixture_model(monkeypatch)
    seed_path, _seed_sha, close_sha, _weak = _seeded_surface(catalog, tmp_path)

    rc, out = run_cli(["group", seed_path, "--threshold", "0.99"])

    assert rc == EXIT_NO_CHANGE
    assert "no candidate met the similarity threshold" in out
    assert close_sha[:12] not in out


def test_cli_group_size_over_the_layout_cap_exits_two(
    data_root, tmp_path, catalog, monkeypatch, capsys
):
    """A caller-contract violation is exit 2 — a different thing from "nothing yet"."""
    _use_fixture_model(monkeypatch)
    seed_path, _seed_sha, _close, _weak = _seeded_surface(catalog, tmp_path)

    assert cli.main(["group", seed_path, "--size", str(MAX_LAYOUT_SOURCES + 3)]) == EXIT_FATAL
    assert f"2..{MAX_LAYOUT_SOURCES}" in capsys.readouterr().err


def test_cli_group_pool_size_over_the_bound_exits_two(
    data_root, tmp_path, catalog, monkeypatch, capsys
):
    _use_fixture_model(monkeypatch)
    seed_path, _seed_sha, _close, _weak = _seeded_surface(catalog, tmp_path)

    assert cli.main(["group", seed_path, "--pool-size", "9999"]) == EXIT_FATAL
    assert str(MAX_CANDIDATE_POOL) in capsys.readouterr().err


def test_cli_json_and_api_response_are_equal_dicts(data_root, tmp_path, catalog, monkeypatch):
    """Both surfaces return the byte-identical shape — the M009 house convention."""
    _use_fixture_model(monkeypatch)
    seed_path, seed_sha, _close, _weak = _seeded_surface(catalog, tmp_path)

    rc, out = run_cli(["group", seed_path, "--size", "3", "--json"])
    assert rc == EXIT_OK

    client = TestClient(create_app(catalog=catalog))
    response = client.post("/api/packing/group", json={"asset": seed_sha, "size": 3})

    assert response.status_code == 200
    assert response.json() == json.loads(out)


def test_api_group_unavailable_is_a_200_with_available_false(
    data_root, tmp_path, catalog, monkeypatch
):
    """Nothing to group is a normal answer, not an error."""
    _use_fixture_model(monkeypatch)
    _path, sha = _write_asset(catalog, tmp_path, "lonely.png", _frame_bytes(0))
    client = TestClient(create_app(catalog=catalog))

    response = client.post("/api/packing/group", json={"asset": sha})

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["reason"]
    assert payload["members"] == []


def test_api_group_size_over_the_layout_cap_is_400(data_root, tmp_path, catalog, monkeypatch):
    _use_fixture_model(monkeypatch)
    _seed_path, seed_sha, _close, _weak = _seeded_surface(catalog, tmp_path)
    client = TestClient(create_app(catalog=catalog))

    response = client.post(
        "/api/packing/group", json={"asset": seed_sha, "size": MAX_LAYOUT_SOURCES + 1}
    )

    assert response.status_code == 400
    assert f"2..{MAX_LAYOUT_SOURCES}" in response.json()["detail"]


def test_api_group_pool_size_over_the_bound_is_400(data_root, tmp_path, catalog, monkeypatch):
    _use_fixture_model(monkeypatch)
    _seed_path, seed_sha, _close, _weak = _seeded_surface(catalog, tmp_path)
    client = TestClient(create_app(catalog=catalog))

    response = client.post(
        "/api/packing/group", json={"asset": seed_sha, "pool_size": MAX_CANDIDATE_POOL + 1}
    )

    assert response.status_code == 400


def test_api_group_without_an_embedding_model_is_503(data_root, tmp_path, catalog, monkeypatch):
    """Mirrors ``/api/taste/embedding-explain``'s posture for the same condition."""
    _no_model(monkeypatch)
    _seed_path, seed_sha, _close, _weak = _seeded_surface(catalog, tmp_path)
    client = TestClient(create_app(catalog=catalog))

    response = client.post("/api/packing/group", json={"asset": seed_sha})

    assert response.status_code == 503
