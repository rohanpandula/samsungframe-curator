"""Bounded-pool group selection over M009's embedding vectors (M010/S04, R045).

This module answers "**which** images belong together" — never "how are N chosen
images arranged". The geometry question belongs to
:mod:`curator.artdirection.packing`, which imports nothing from this package and
never will: grouping lives here beside
:mod:`curator.taste.embedding.attribution` precisely so the policy engine stays
pure and the locked "treatment-level taste is out of scope" boundary holds. The
only thing that crosses is a list of content shas, chosen here and handed to
``curator propose``.

Three properties this module is built around:

**The pool is always caller-bounded, and no all-pairs sweep exists.**
:func:`select_group` compares the seed against a caller-supplied candidate pool
and nothing else. A pool larger than :data:`MAX_CANDIDATE_POOL` is *rejected*,
never truncated, and :func:`resolve_group_pool` applies the same bound as a SQL
``LIMIT`` so neither production surface can exceed it. Whole-library group
discovery is explicitly out of scope — see :data:`MAX_CANDIDATE_POOL` for the
materialization hazard the bound exists to keep out of reach.

**:class:`GroupingError` is raised only for caller-contract violations** — an
over-cap pool, an out-of-range ``group_size``, or a sha the caller's own
``sha_to_entry_id`` mapping does not cover. Those are programming errors. *Data*
conditions — no embedding model, no stored vectors, no candidate above the
threshold — never raise: they return an unavailable :class:`GroupSelection`
carrying a ``reason``, mirroring
:class:`~curator.analysis.provider.ComputeProbe` and ``GET /api/taste/pair``'s
``{"available": False, "reason": ...}`` idiom. That split is the difference
between a bug and honest degradation, and it is what lets ``curator group`` exit
3 ("nothing to act on yet") rather than 2 when no model is installed.

**Parallel, not Replace (locked decision).** M002's hand-crafted
``pairing.affinity`` (phash / palette / date / orientation) remains the diptych
gate and the N-up group-cohesion signal in
:mod:`curator.artdirection.policy`; embedding cosine answers the separate
question of *which* images to put in a group. The two are never blended into one
scalar — every proposal and every selection names which one produced it through
``evidence["affinity_source"]`` (``"pairing.affinity"`` there,
:data:`AFFINITY_SOURCE` here).

The retrieval itself copies
:func:`~curator.taste.embedding.attribution.find_exemplars`'s *current*, hardened
shape (M009/S04, D025 and review fix IN-01): bulk-load the matrix, bound the pool
*first*, exclude zero-norm rows *before* the division, brute-force cosine, rank,
take k, and return an honest empty result when there is nothing to find. No
fitting, no clustering, no opaque cluster id, and no hidden catalog access —
:func:`resolve_group_pool` is the module's only database read and it is a
separate function for exactly that reason.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from curator.artdirection.manifest import MAX_LAYOUT_SOURCES
from curator.catalog import Catalog
from curator.errors import CuratorError
from curator.taste.embedding.store import EmbeddingStore

#: The literal recorded in ``evidence["affinity_source"]`` by every selection this
#: module produces, so a reader never has to guess which of the two parallel
#: affinity signals is speaking (``policy.py`` records ``"pairing.affinity"`` for
#: the other one). Named, never blended.
AFFINITY_SOURCE = "embedding_cosine"

#: Minimum seed-to-candidate cosine similarity for a candidate to join a group.
#:
#: A **stated, revisable engineering default — not a researched number**, the same
#: honesty :data:`~curator.artdirection.manifest.MAX_LAYOUT_SOURCES` gets. Nothing
#: measured 0.6 as the point where a group stops cohering; it is a deliberately
#: selective starting value, and it is a per-call parameter (``--threshold``)
#: precisely because the right value is an open question.
GROUP_SIMILARITY_THRESHOLD = 0.6

#: Hard upper bound on a candidate pool handed to :func:`select_group`.
#:
#: Grouping operates on **bounded pools only**. Whole-library group discovery is
#: explicitly out of scope for M010, and this bound is what keeps it out of
#: reach: at a pool of tens-to-low-hundreds the seed-to-candidate similarities
#: are a few hundred floats and even a hypothetical ``K x K`` all-pairs matrix is
#: a few hundred thousand — but a whole-library sweep at 100k vectors would be
#: 10^10 floats (~40GB in float32). A future all-pairs feature has to confront
#: that materialization hazard deliberately rather than rediscover it; this bound
#: is what stops it being reached by accident. An over-cap pool is rejected with
#: :class:`GroupingError`, never silently truncated ("clear status, never
#: silent").
MAX_CANDIDATE_POOL = 256


class GroupingError(CuratorError):
    """Raised for a caller-contract violation in this module — never for a data condition.

    Over-cap pool, out-of-range ``group_size``, or a sha the caller's own
    ``sha_to_entry_id`` mapping does not cover. "No model installed", "no stored
    vectors" and "nothing above the threshold" are *not* errors: they return an
    unavailable :class:`GroupSelection` (see the module docstring).
    """


@dataclass(frozen=True)
class GroupCandidate:
    """One companion image selected for a group, with the cosine that put it there.

    Mirrors :class:`~curator.taste.embedding.attribution.ExemplarResult`
    field-for-field — same three fields, same ``to_dict``/``from_dict`` shape —
    because it is the same kind of answer (a sha, its catalog identity, and the
    similarity that ranked it) to a different question.
    """

    sha256: str
    entry_id: int
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "entry_id": self.entry_id, "similarity": self.similarity}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupCandidate:
        """Round-trip inverse of :meth:`to_dict`."""
        if isinstance(data, cls):
            return data
        return cls(
            sha256=str(data["sha256"]),
            entry_id=int(data["entry_id"]),
            similarity=float(data["similarity"]),
        )


@dataclass(frozen=True)
class GroupSelection:
    """A seed plus the companions embedding cosine put with it, or an honest "not yet".

    ``available``/``reason`` is the established honest-degradation pair
    (:class:`~curator.analysis.provider.ComputeProbe`, ``GET /api/taste/pair``):
    ``available=False`` always carries a non-empty, actionable ``reason`` and an
    empty ``members`` list — a group is never fabricated to fill the shape.

    ``available=True`` with **fewer** members than requested is a valid answer:
    the shortfall is recorded in ``evidence`` (``requested_group_size`` vs
    ``selected_group_size``) and never padded with candidates below the
    threshold.
    """

    seed_sha256: str
    members: list[GroupCandidate]
    available: bool
    reason: str
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        # Defensive copies, mirroring ``AttributionResult.__post_init__`` — a
        # frozen dataclass whose list/dict fields alias the caller's own objects
        # is only shallowly frozen.
        object.__setattr__(self, "members", list(self.members))
        object.__setattr__(self, "evidence", dict(self.evidence))

    @property
    def shas(self) -> list[str]:
        """The ordered group, ready to hand to ``curator propose``.

        The seed first, then each member in descending similarity — the same
        order the packer will lay the cells out in. Empty of companions (just the
        seed) when this selection is unavailable.
        """
        return [self.seed_sha256] + [m.sha256 for m in self.members]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_sha256": self.seed_sha256,
            "members": [m.to_dict() for m in self.members],
            "available": self.available,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GroupSelection:
        """Round-trip inverse of :meth:`to_dict`."""
        if isinstance(data, cls):
            return data
        return cls(
            seed_sha256=str(data["seed_sha256"]),
            members=[GroupCandidate.from_dict(m) for m in data.get("members", [])],
            available=bool(data["available"]),
            reason=str(data.get("reason", "")),
            evidence=dict(data.get("evidence", {})),
        )


def _evidence(
    *,
    model_version: str,
    threshold: float,
    requested_group_size: int,
    selected_group_size: int,
    pool_size: int,
    considered: int,
    pairwise_cosine: dict[str, float],
) -> dict[str, Any]:
    """Build the machine-readable evidence block every selection carries.

    ``pool_size`` is the *caller's* pool (what was offered), ``considered`` is how
    many of those had a usable — stored, current-version, non-zero-norm — vector,
    and ``pairwise_cosine`` maps **every considered candidate** to its
    seed-to-candidate cosine, rejected ones included, so the evidence shows what
    was passed over and not only what was chosen. Both group sizes count the seed.

    Despite the name, no ``K x K`` matrix is ever materialized: these are the
    ``K`` seed-to-candidate similarities the ranking already computed.
    """
    return {
        "affinity_source": AFFINITY_SOURCE,
        "model_version": model_version,
        "threshold": threshold,
        "requested_group_size": requested_group_size,
        "selected_group_size": selected_group_size,
        "pool_size": pool_size,
        "considered": considered,
        "pairwise_cosine": dict(pairwise_cosine),
    }


def select_group(
    seed_sha: str,
    candidate_pool: Sequence[str],
    sha_to_entry_id: dict[str, int],
    embedding_store: EmbeddingStore,
    model_version: str,
    *,
    group_size: int = 3,
    threshold: float = GROUP_SIMILARITY_THRESHOLD,
) -> GroupSelection:
    """Select up to ``group_size - 1`` companions for *seed_sha* from *candidate_pool*.

    The parameter order deliberately mirrors
    :func:`~curator.taste.embedding.attribution.find_exemplars` — including the
    **explicit caller-supplied** *sha_to_entry_id* (D025), so this function never
    queries the catalog and the "otherwise pure numpy operation" claim stays
    true. :func:`resolve_group_pool` is what builds both *candidate_pool* and
    *sha_to_entry_id*, from one query, for both production surfaces.

    Raises :class:`GroupingError` — and only for a caller-contract violation — on
    an over-cap pool, a ``group_size`` outside ``2..MAX_LAYOUT_SOURCES``, or a
    seed/pool sha the caller's own mapping does not cover. Every *data* condition
    returns an unavailable :class:`GroupSelection` with a non-empty ``reason``.

    ``embedding_store.get_matrix(model_version)`` is the **only** retrieval path.
    That version scoping is the only mechanism that can prevent a
    cross-checkpoint mismatch — two 512-dim vectors from different checkpoints
    are indistinguishable by shape — so no path here bypasses it (T-10-17).
    """
    if len(candidate_pool) > MAX_CANDIDATE_POOL:
        raise GroupingError(
            f"candidate pool has {len(candidate_pool)} entries, over the "
            f"{MAX_CANDIDATE_POOL}-candidate grouping bound — an over-cap pool is "
            f"rejected, never truncated; grouping operates on bounded pools only"
        )
    if group_size < 2 or group_size > MAX_LAYOUT_SOURCES:
        raise GroupingError(
            f"group_size {group_size} is outside 2..{MAX_LAYOUT_SOURCES} — a group "
            f"the packer and the renderer would refuse can never be requested"
        )
    if seed_sha not in sha_to_entry_id:
        raise GroupingError(
            f"seed {seed_sha[:12]} is absent from sha_to_entry_id — this function "
            f"never queries the catalog, so the caller must supply the mapping"
        )
    unmapped = [sha for sha in candidate_pool if sha not in sha_to_entry_id]
    if unmapped:
        raise GroupingError(
            f"{len(unmapped)} candidate pool sha(s) are absent from sha_to_entry_id "
            f"(first: {unmapped[0][:12]}) — the caller supplies both, from one query"
        )

    pool_size = len(candidate_pool)

    def unavailable(
        reason: str,
        *,
        considered: int = 0,
        pairwise_cosine: dict[str, float] | None = None,
    ) -> GroupSelection:
        return GroupSelection(
            seed_sha256=seed_sha,
            members=[],
            available=False,
            reason=reason,
            evidence=_evidence(
                model_version=model_version,
                threshold=threshold,
                requested_group_size=group_size,
                selected_group_size=0,
                pool_size=pool_size,
                considered=considered,
                pairwise_cosine=pairwise_cosine or {},
            ),
        )

    shas, matrix = embedding_store.get_matrix(model_version)
    if not shas:
        return unavailable(
            f"no embedding vectors are stored under model version {model_version!r}"
            " — run `curator taste embed-status --backfill` first"
        )
    try:
        seed_index = shas.index(seed_sha)
    except ValueError:
        return unavailable(
            f"the seed has no stored vector under model version {model_version!r}"
            " — run `curator taste embed-status --backfill` first"
        )
    vector = matrix[seed_index]

    # Bound the pool BEFORE any similarity math (the find_exemplars shape): only
    # rows the caller offered, never the seed itself.
    pool = set(candidate_pool) - {seed_sha}
    keep = [i for i, sha in enumerate(shas) if sha in pool]
    if not keep:
        return unavailable(
            "no candidate in the pool has a stored vector under model version"
            f" {model_version!r}"
        )

    # Zero-norm guards BEFORE any division (IN-01). A NaN from a zero-by-zero
    # division has no defined cosine and would rank as greater-than-everything,
    # silently surfacing as the *most* similar member.
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        return unavailable(
            "the seed's stored vector is all zeros — no defined direction to compare"
        )
    kept_shas = [shas[i] for i in keep]
    kept_matrix = matrix[keep]
    row_norms = np.linalg.norm(kept_matrix, axis=1)
    nonzero = row_norms > 0
    if not np.any(nonzero):
        return unavailable(
            "every candidate's stored vector is all zeros — no defined direction to compare"
        )
    kept_shas = [sha for sha, usable in zip(kept_shas, nonzero, strict=True) if usable]
    kept_matrix = kept_matrix[nonzero]
    row_norms = row_norms[nonzero]

    similarities = (kept_matrix @ vector) / (row_norms * vector_norm)

    # R046 requires every genuine tie to be resolved by a *stated* rule, so this
    # ranks with a Python sort on (-similarity, sha256) rather than
    # ``numpy.argsort``: numpy's order among equal values is unspecified (its
    # default kind is not stable), where two equal cosines here always resolve by
    # sha lexicographically. Deliberately different from ``find_exemplars``,
    # whose sort predates that requirement.
    ranked = sorted(
        ((sha, float(sim)) for sha, sim in zip(kept_shas, similarities, strict=True)),
        key=lambda pair: (-pair[1], pair[0]),
    )
    pairwise_cosine = dict(ranked)
    members = [
        GroupCandidate(sha256=sha, entry_id=sha_to_entry_id[sha], similarity=similarity)
        for sha, similarity in ranked
        if similarity >= threshold
    ][: group_size - 1]
    if not members:
        return unavailable(
            "no candidate met the similarity threshold",
            considered=len(ranked),
            pairwise_cosine=pairwise_cosine,
        )
    return GroupSelection(
        seed_sha256=seed_sha,
        members=members,
        available=True,
        reason=(
            f"{len(members)} of {group_size - 1} requested companion(s) at or above"
            f" cosine {threshold:.2f}"
        ),
        evidence=_evidence(
            model_version=model_version,
            threshold=threshold,
            requested_group_size=group_size,
            selected_group_size=len(members) + 1,
            pool_size=pool_size,
            considered=len(ranked),
            pairwise_cosine=pairwise_cosine,
        ),
    )


def resolve_group_pool(
    catalog: Catalog, model_version: str, *, limit: int = MAX_CANDIDATE_POOL
) -> tuple[list[str], dict[str, int]]:
    """Return ``(pool_shas, sha_to_entry_id)`` — the bounded pool :func:`select_group` needs.

    One join from ``catalog_entries`` to ``content_embeddings`` on the content
    sha, filtered to *model_version*, one row per sha (its highest
    ``catalog_entries.id``, mirroring the ``ORDER BY id DESC LIMIT 1`` idiom the
    CLI/API already use to resolve a sha to an entry), most recent first, limited
    to *limit*.

    **This is the module's only database access**, deliberately isolated in its
    own function so :func:`select_group` stays a pure numpy operation with no
    hidden catalog dependency (D025's rule, restated). It follows M009/S01's
    :func:`~curator.taste.store.resolve_vote_candidates` precedent: a
    module-level ``resolve_*`` helper taking an explicit :class:`Catalog`, shared
    by the CLI and the API so "the pool" is defined once rather than per surface.

    *limit* carries the same bound :func:`select_group` enforces (T-10-19): a
    *limit* outside ``1..MAX_CANDIDATE_POOL`` is a caller-contract violation and
    raises :class:`GroupingError` here, before the query runs — so no surface can
    ask the database for a pool the grouping bound would then refuse, and the
    over-cap request is rejected rather than quietly clamped.
    """
    if limit < 1 or limit > MAX_CANDIDATE_POOL:
        raise GroupingError(
            f"pool limit {limit} is outside 1..{MAX_CANDIDATE_POOL} — grouping "
            f"operates on bounded pools only; an over-cap pool is rejected, never "
            f"truncated"
        )
    rows = catalog.db.execute(
        "SELECT e.sha256, MAX(e.id) AS entry_id"
        " FROM catalog_entries e"
        " JOIN content_embeddings c ON c.sha256 = e.sha256"
        " WHERE c.model_version = ?"
        " GROUP BY e.sha256"
        " ORDER BY entry_id DESC"
        " LIMIT ?",
        (model_version, limit),
    ).fetchall()
    pool = [str(sha) for sha, _ in rows]
    return pool, {str(sha): int(entry_id) for sha, entry_id in rows}
