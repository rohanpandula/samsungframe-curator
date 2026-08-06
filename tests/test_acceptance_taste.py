"""Acceptance gate for the M007/S05 Taste Lens surface.

This module ships the deterministic, air-gapped acceptance gate for the Taste
Lens features delivered across M007/S01-S04. Each scenario is
**self-bootstrapping**: it mints its own synthetic profiles, analysis signals,
and creator catalogs over the isolated ``data_root`` (from conftest) and drives
the subsystem objects directly — never relying on cross-test ordering, a live
server, or the network.

* S1 — profile baseline isolation: a default/``None`` profile ranks exactly as
  the baseline, a trained (``apply_preference``-trained) profile shifts the order
  deterministically, and disabling restores the exact baseline order.
* S2 — pairwise evidence + promotion: the most-informative pair is chosen,
  preferences fold in deterministically (version bump, weight shift), ``evaluate``
  emits held-out accuracy + ranking lift vs baseline with the trained profile
  beating the baseline, and ``promote_if_valid`` gates promotion.
* S3 — controls + manifest isolation: deterministic blend, veto demotion,
  snapshot/replay/undo/export/reset/delete history, and an unchanged
  byte-deterministic render with taste on/off.
* S4 — discover safety: deterministic feed composition, familiar↔surprising
  bias, artist spotlight, outage isolation, publication denial, and action
  parity across button/keyboard/gesture channels.
"""

from __future__ import annotations

import io
import json

from PIL import Image

from curator.analysis.schema import AnalysisResult, ColorStory, Pairing, QualitySignals
from curator.artdirection.manifest import ArtDirectionManifest, LayoutTreatment, ProcessingIntent
from curator.hashing import sha256_hex
from curator.taste.controls import (
    ProfileHistory,
    apply_veto,
    approved_manifest_unchanged,
    blend,
    veto,
)
from curator.taste.discover import (
    Creator,
    CreatorCatalog,
    PublicationDenied,
    TasteDeck,
    Work,
)
from curator.taste.pairwise import (
    PairwiseEvidence,
    apply_preference,
    choose_pair,
    evaluate,
    promote_if_valid,
    uncertainty_score,
)
from curator.taste.profiles import (
    SIGNAL_NAMES,
    TasteProfile,
    TasteProfileKind,
    baseline_weights,
    default_profile,
)
from curator.taste.rank import TasteRanker


def _result(
    aesthetic: float = 0.5,
    technical: float = 0.5,
    colorfulness: float = 0.5,
    harmony: float = 0.5,
    affinity: float = 0.5,
) -> AnalysisResult:
    """A synthetic M002-style analysis result with explicit signal values."""
    return AnalysisResult(
        asset_id="asset",
        quality=QualitySignals(aesthetic_quality=aesthetic, technical_quality=technical),
        color_story=ColorStory(colorfulness=colorfulness, harmony=harmony),
        pairing=Pairing(affinity=affinity),
    )


def _profile(weights: dict[str, float], version: int = 1) -> TasteProfile:
    """A synthetic taste profile over the baseline (all-signal) weights."""
    return TasteProfile(
        id="p-personal",
        kind=TasteProfileKind.PERSONAL,
        name="personal",
        weights={**baseline_weights(), **weights},
        version=version,
    )


def _signals(cands, amap) -> dict[str, dict[str, float]]:
    """Extract per-candidate signal-value maps (for freeze-free determinstic use)."""
    ranker = TasteRanker()
    return {c["id"]: ranker.signal_values(amap[c["id"]]) for c in cands}


def _all_pairs(ids: list[str]) -> list[tuple[str, str]]:
    return [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]


# ---------------------------------------------------------------------------
# S1 — profile baseline isolation
# ---------------------------------------------------------------------------


def test_acceptance_taste_baseline_isolation(data_root):
    """Default/None ranks == baseline; trained reranks; disabling restores it."""
    ranker = TasteRanker()

    # Five candidates whose baselines are distinct, with M002-style signals.
    cands = [
        {"id": "c0", "baseline": 1.0},
        {"id": "c1", "baseline": 3.0},
        {"id": "c2", "baseline": 2.0},
        {"id": "c3", "baseline": 5.0},
        {"id": "c4", "baseline": 4.0},
    ]
    amap = {
        "c0": _result(colorfulness=0.1, aesthetic=0.8),
        "c1": _result(colorfulness=0.3, aesthetic=0.6),
        "c2": _result(colorfulness=0.5, aesthetic=0.5),
        "c3": _result(colorfulness=0.7, aesthetic=0.3),
        "c4": _result(colorfulness=0.9, aesthetic=0.2),
    }
    baseline_ids = [c["id"] for c in cands]

    # A default / None profile is inert: it reproduces the baseline order exactly.
    assert [c["id"] for c in ranker.rank(cands, None, analysis_map=amap)] == baseline_ids
    assert [c["id"] for c in ranker.rank(
        cands, default_profile(), analysis_map=amap
    )] == baseline_ids

    # A trained profile (fold in a few "prefer more colorful" preferences) shifts
    # the order deterministically toward the colorful end.
    sv = _signals(cands, amap)
    prof = _profile({})
    used: set[frozenset] = set()
    holdout_set = {frozenset(("c0", "c4"))}
    pool = [p for p in _all_pairs([c["id"] for c in cands]) if frozenset(p) not in holdout_set]
    for _ in range(min(len(pool), 4)):
        a, b = choose_pair(cands, prof, analysis_map=amap, rng_seed=0,
                           exclude=used | holdout_set)
        aid, bid = a["id"], b["id"]
        prefer_a = amap[aid].color_story.colorfulness > amap[bid].color_story.colorfulness
        prof = apply_preference(prof, sv[aid], sv[bid], prefer_a=prefer_a)
        used.add(frozenset((aid, bid)))

    assert ranker.is_enabled(prof) is True
    trained_ids = [c["id"] for c in ranker.rank(cands, prof, analysis_map=amap)]
    assert trained_ids != baseline_ids
    # Deterministic: the same trained profile ranks identically every time.
    assert trained_ids == [c["id"] for c in ranker.rank(cands, prof, analysis_map=amap)]

    # Disabling (None) restores the exact baseline order — no reordering.
    assert [c["id"] for c in ranker.rank(cands, None, analysis_map=amap)] == baseline_ids


# ---------------------------------------------------------------------------
# S2 — pairwise evidence + promotion
# ---------------------------------------------------------------------------


def test_acceptance_taste_pairwise_evidence_and_promotion(data_root):
    """Informed pairing, deterministic weight update, evidence, and promotion gate."""
    cands = [
        {"id": "x", "baseline": 1.0},
        {"id": "y", "baseline": 1.0},
        {"id": "z", "baseline": 1.0},
    ]
    amap = {
        "x": _result(colorfulness=0.5, aesthetic=0.5),
        "y": _result(colorfulness=0.5, aesthetic=0.5),
        "z": _result(colorfulness=1.0, aesthetic=0.0),
    }
    prof = _profile({"colorfulness": 1.0})

    # The near-tied pair (x, y) is the highest-uncertainty / most informative one.
    assert uncertainty_score(
        _signals(cands, amap)["x"], _signals(cands, amap)["y"], prof
    ) > uncertainty_score(_signals(cands, amap)["x"], _signals(cands, amap)["z"], prof)
    a, b = choose_pair(cands, prof, analysis_map=amap, rng_seed=0)
    assert {a["id"], b["id"]} == {"x", "y"}
    # choose_pair is deterministic for identical inputs.
    a2, b2 = choose_pair(cands, prof, analysis_map=amap, rng_seed=0)
    assert (a["id"], b["id"]) == (a2["id"], b2["id"])

    # apply_preference nudges weights toward the preferred signal and bumps version.
    colorful = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 1.0}
    muted = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 0.0}
    updated = apply_preference(prof, colorful, muted, prefer_a=True)
    assert updated.version == prof.version + 1
    assert updated.weights["colorfulness"] > prof.weights["colorfulness"]
    assert updated is not prof and prof.weights["colorfulness"] == 1.0
    again = apply_preference(prof, colorful, muted, prefer_a=True)
    assert updated.weights == again.weights  # deterministic

    # Train a profile on the fuller pool; a matching model beats the baseline.
    all_cands = [{"id": f"c{i}", "baseline": 1.0} for i in range(6)]
    all_map = {f"c{i}": _result(colorfulness=i / 5.0) for i in range(6)}
    sv = _signals(all_cands, all_map)
    holdouts = [("c0", "c5", "c5")]
    holdout_set = {frozenset(h) for h in holdouts}
    pool = [p for p in _all_pairs([c["id"] for c in all_cands]) if frozenset(p) not in holdout_set]
    base = _profile({})
    used: set[frozenset] = set()
    for _ in range(len(pool)):
        p, q = choose_pair(all_cands, base, analysis_map=all_map, rng_seed=0,
                           exclude=used | holdout_set)
        pid, qid = p["id"], q["id"]
        prefer_p = all_map[pid].color_story.colorfulness > all_map[qid].color_story.colorfulness
        base = apply_preference(base, sv[pid], sv[qid], prefer_a=prefer_p)
        used.add(frozenset((pid, qid)))

    ev = evaluate(
        lambda a: TasteRanker().personal_delta(a, base)[0],
        lambda a: 0.0,
        all_cands,
        all_map,
        holdouts,
        sample_efficiency_pairs=base.version - 1,
    )
    assert ev.held_out_pairs == 1
    assert ev.held_out_accuracy == 1.0
    assert ev.ranking_lift_vs_baseline > 0.0
    assert ev.sample_efficiency_pairs >= 1

    # The trained (matching) model scores higher than the inert baseline.
    base_ev = evaluate(
        lambda a: 0.0,
        lambda a: 0.0,
        all_cands,
        all_map,
        holdouts,
        sample_efficiency_pairs=default_profile().version - 1,
    )
    assert ev.held_out_accuracy > base_ev.held_out_accuracy
    assert ev.ranking_lift_vs_baseline > base_ev.ranking_lift_vs_baseline

    # promote_if_valid gates promotion: validated + threshold, else rejected.
    assert promote_if_valid(ev) is True
    low = PairwiseEvidence(10, 0.4, 0.3, 8)
    assert promote_if_valid(low) is False
    unvalidated = PairwiseEvidence(10, 0.95, 0.3, 8, requires_validation=False)
    assert promote_if_valid(unvalidated) is False


# ---------------------------------------------------------------------------
# S3 — controls + manifest isolation
# ---------------------------------------------------------------------------


def _small_manifest() -> tuple[ArtDirectionManifest, dict[str, bytes]]:
    """A tiny deterministic PNG source + manifest for the renderer check."""
    buf = io.BytesIO()
    Image.new("RGB", (120, 68), (120, 90, 60)).save(buf, format="PNG")
    source = buf.getvalue()
    manifest = ArtDirectionManifest(
        sources=[sha256_hex(source)],
        layout_treatment=LayoutTreatment.SINGLE_FULLBLEED,
        processing_intent=ProcessingIntent(upscale_warning=True),
    )
    return manifest, {manifest.sources[0]: source}


def test_acceptance_taste_controls_and_manifest_isolation(data_root):
    """Deterministic blend, veto demotion, history, and render-path isolation."""
    # blend: deterministic weighted combination, fresh profile + version bump.
    a = _profile({"colorfulness": 1.0, "harmony": 0.5}, version=3)
    b = _profile({"colorfulness": -1.0, "aesthetic_quality": 2.0}, version=5)
    out = blend(a, b, w=0.5)
    assert out.weights["colorfulness"] == 0.5 * 1.0 + 0.5 * (-1.0)
    assert out.weights["aesthetic_quality"] == 0.5 * 0.0 + 0.5 * 2.0
    assert out.version == max(a.version, b.version) + 1 == 6
    assert blend(a, b, 0.5) == out  # deterministic

    # veto: a strongly-negative signal overrides primary to demote a candidate.
    primary = _profile({"colorfulness": 1.0, "harmony": 0.5})
    vetoer = _profile({"colorfulness": -5.0})
    vetoed = veto(primary, vetoer)
    assert vetoed.weights["colorfulness"] == -5.0
    assert vetoed.weights["harmony"] == 0.5
    vetoer_sig = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 1.0}
    kept_sig = {name: 0.5 for name in SIGNAL_NAMES} | {"colorfulness": 0.0}
    sig_map = {"hi": vetoer_sig, "lo": kept_sig}
    ranked = apply_veto(
        [{"id": "hi", "baseline": 1.0}, {"id": "lo", "baseline": 1.0}],
        sig_map,
        vetoer,
    )
    assert [c["id"] for c in ranked] == ["lo", "hi"]  # disliked candidate demoted

    # ProfileHistory: snapshot -> replay restores the exact profile; undo returns
    # the previous version; export round-trips portable JSON; reset -> default;
    # delete removes the snapshot.
    history = ProfileHistory()
    v1 = _profile({"colorfulness": 1.0}, version=1)
    v2 = _profile({"colorfulness": 2.0}, version=2)
    assert history.snapshot(v1) == 1
    assert history.snapshot(v2) == 2
    assert history.replay(2) == v2
    assert history.undo().version == 1
    assert history.undo().weights["colorfulness"] == 1.0
    exported = history.export_profile(v2)
    rebuilt = TasteProfile.from_dict(json.loads(json.dumps(exported)))
    assert rebuilt == v2
    assert history.reset(v2).weights == baseline_weights()
    history.delete(1)
    assert 1 not in history.snapshots and 2 in history.snapshots

    # approved_manifest_unchanged: byte-identical renders with taste on/off.
    manifest, sources = _small_manifest()
    assert approved_manifest_unchanged(manifest, sources) is True


# ---------------------------------------------------------------------------
# S4 — discover safety
# ---------------------------------------------------------------------------


def _creator_signals(colorfulness: float) -> dict[str, float]:
    return {
        "aesthetic_quality": 0.5,
        "technical_quality": 0.5,
        "colorfulness": colorfulness,
        "harmony": 0.5,
        "pairing_affinity": 0.5,
        "vibrancy": colorfulness,
    }


def _discover_catalog() -> CreatorCatalog:
    creators = [
        Creator(id="alice", name="Alice", genre="landscape"),
        Creator(id="bob", name="Bob", genre="portrait"),
        Creator(id="carol", name="Carol", genre="abstract"),
        Creator(id="dave", name="Dave", genre="minimal"),
    ]
    fingerprints = {
        "alice": [0.9, 0.8, 0.7, 0.6],
        "bob": [0.2, 0.25, 0.3, 0.35],
        "carol": [0.95, 0.9, 0.85, 0.8],
        "dave": [0.05, 0.1, 0.15, 0.2],
    }
    works: list[Work] = []
    for creator in creators:
        for i, color in enumerate(fingerprints[creator.id]):
            works.append(
                Work(
                    id=f"{creator.id}-w{i}",
                    creator_id=creator.id,
                    title=f"{creator.name} work {i}",
                    baseline=0.5,
                    signals=_creator_signals(color),
                )
            )
    return CreatorCatalog(creators, works)


def _deck(colorfulness: float = 2.0) -> TasteDeck:
    prof = TasteProfile(
        id="p-personal",
        kind=TasteProfileKind.PERSONAL,
        name="personal",
        weights={**baseline_weights(), "colorfulness": colorfulness},
    )
    return TasteDeck(_discover_catalog(), profile=prof)


def test_acceptance_taste_discover_safety(data_root):
    """Deterministic feeds, spotlight, outage isolation, denial, action parity."""
    # Same seed -> same feed; different seeds -> different wildcard feed.
    deck = _deck()
    first = deck.compose("likely", rng_seed=0)
    second = deck.compose("likely", rng_seed=0)
    assert [w.id for w in first] == [w.id for w in second]
    assert [w.id for w in _deck().compose("wildcard", rng_seed=1)] != [
        w.id for w in _deck().compose("wildcard", rng_seed=99)
    ]

    # familiar vs surprising bias produce different compositions.
    familiar = _deck().compose("likely", familiar_surprising=-0.8)
    surprising = _deck().compose("likely", familiar_surprising=0.8)
    assert [w.id for w in familiar] != [w.id for w in surprising]
    assert surprising[0].signals["colorfulness"] < 0.5

    # artist_spotlight surfaces the spotlight creator's works at the top.
    feed = _deck().compose("likely", artist_spotlight="bob")
    assert feed[0].creator_id == "bob"

    # set_down(creator) is an outage: the deck still composes, minus that creator.
    deck = _deck()
    deck.catalog.set_down("alice")
    out_feed = deck.compose("likely", rng_seed=0)
    assert out_feed
    assert all(w.creator_id != "alice" for w in out_feed)

    # A denied creator's works never appear in any feed.
    policy = PublicationDenied()
    policy.deny("carol")
    denied_deck = TasteDeck(_discover_catalog(), profile=_deck().profile,
                            publication_policy=policy)
    denied_feed = denied_deck.compose("likely", rng_seed=0)
    assert all(w.creator_id != "carol" for w in denied_feed)

    # deck_action parity: button/keyboard/gesture return identical results.
    ids = []
    for channel in ("button", "keyboard", "gesture"):
        d = _deck()
        d.compose("likely", rng_seed=0)
        ids.append(d.deck_action("next", channel=channel).id)
    assert ids[0] == ids[1] == ids[2]

    # Calibration via the action path folds a preference and bumps the version.
    cal_deck = _deck()
    cal_deck.compose("likely", rng_seed=0)
    original = cal_deck.profile.weights["colorfulness"]
    result = cal_deck.deck_action(
        "calibrate_prefer_a",
        channel="button",
        profile=cal_deck.profile,
        work_a_id="alice-w0",
        work_b_id="bob-w0",
    )
    assert result.profile.version == 2
    assert result.profile.weights["colorfulness"] > original
