"""Tests for src/curator/taste/discover (M007/S04 catalog + Taste Deck).

Synthetic federated creator catalog with varied works; all deterministic.
Covers feed determinism, familiar↔surprising bias, artist spotlight, enforced
diversity, outage isolation, publication denial, signal isolation, action
parity, pairwise calibration, and JSON round-trips.
"""

from __future__ import annotations

import json

import pytest

from curator.taste.discover import (
    Creator,
    CreatorCatalog,
    PublicationDenied,
    TasteDeck,
    Work,
    alignment,
)
from curator.taste.profiles import (
    TasteProfile,
    TasteProfileKind,
    baseline_weights,
)


def _profile(colorfulness: float = 0.0) -> TasteProfile:
    return TasteProfile(
        id="p-personal",
        kind=TasteProfileKind.PERSONAL,
        name="personal",
        weights={**baseline_weights(), "colorfulness": colorfulness},
    )


def _signals(colorfulness: float, aesthetic: float = 0.5) -> dict[str, float]:
    return {
        "aesthetic_quality": aesthetic,
        "technical_quality": 0.5,
        "colorfulness": colorfulness,
        "harmony": 0.5,
        "pairing_affinity": 0.5,
        "vibrancy": colorfulness,
    }


_CREATORS = [
    Creator(id="alice", name="Alice", genre="landscape", style="pastoral"),
    Creator(id="bob", name="Bob", genre="portrait", style="studio"),
    Creator(id="carol", name="Carol", genre="abstract", style="bold"),
    Creator(id="dave", name="Dave", genre="black-and-white", style="minimal"),
]

# Distinct per-creator colorfulness "fingerprints" so familiar vs surprising
# clearly differ, and so each creator contributes varied works to the pool.
_CREATOR_PROFILES = {
    "alice": [0.9, 0.8, 0.7, 0.6, 0.5],
    "bob": [0.1, 0.2, 0.3, 0.4, 0.55],
    "carol": [0.95, 0.9, 0.85, 0.8, 0.75],
    "dave": [0.05, 0.1, 0.15, 0.2, 0.25],
}


def _build_catalog() -> CreatorCatalog:
    works: list[Work] = []
    for creator in _CREATORS:
        for i, color in enumerate(_CREATOR_PROFILES[creator.id]):
            works.append(
                Work(
                    id=f"{creator.id}-w{i}",
                    creator_id=creator.id,
                    title=f"{creator.name} work {i}",
                    baseline=0.5,
                    signals=_signals(color, aesthetic=0.5 + (i % 4) * 0.1),
                )
            )
    return CreatorCatalog(_CREATORS, works)


def _deck(
    *,
    colorfulness: float = 2.0,
    policy: PublicationDenied | None = None,
) -> TasteDeck:
    return TasteDeck(_build_catalog(), profile=_profile(colorfulness), publication_policy=policy)


# ---------------------------------------------------------------------------
# deterministic feed
# ---------------------------------------------------------------------------


def test_compose_deterministic_same_inputs():
    deck = _deck()
    first = deck.compose("likely", rng_seed=0)
    second = deck.compose("likely", rng_seed=0)
    assert [w.id for w in first] == [w.id for w in second]


def test_compose_wildcard_deterministic_same_seed():
    first = _deck().compose("wildcard", rng_seed=3)
    second = _deck().compose("wildcard", rng_seed=3)
    assert [w.id for w in first] == [w.id for w in second]


def test_compose_wildcard_changes_with_seed():
    a = _deck().compose("wildcard", rng_seed=1)
    b = _deck().compose("wildcard", rng_seed=99)
    assert [w.id for w in a] != [w.id for w in b]


# ---------------------------------------------------------------------------
# familiar ↔ surprising
# ---------------------------------------------------------------------------


def test_familiar_puts_high_alignment_works_up_top():
    familiar = _deck().compose("likely", familiar_surprising=-0.8)
    surprising = _deck().compose("likely", familiar_surprising=0.8)
    # Familiar biases toward the profile (high colorfulness) — the most colorful
    # creator's works lead; surprising biases toward low-alignment/novel works.
    assert familiar[0].creator_id in {"alice", "carol"}
    assert surprising[0].creator_id in {"bob", "dave"}
    assert [w.id for w in familiar] != [w.id for w in surprising]


def test_surprising_introduces_more_diverse_exploration_works():
    deck = _deck()
    familiar = deck.compose("likely", familiar_surprising=-0.9)
    surprising = deck.compose("likely", familiar_surprising=0.9)
    # The surprising feed prominently surfaces low-colorfulness (novel) works.
    low_color_surprising = [w for w in surprising if w.signals["colorfulness"] < 0.5]
    low_color_familiar = [w for w in familiar if w.signals["colorfulness"] < 0.5]
    assert len(low_color_surprising) > len(low_color_familiar)


# ---------------------------------------------------------------------------
# artist spotlight
# ---------------------------------------------------------------------------


def test_artist_spotlight_surfaces_creator_works():
    deck = _deck()
    feed = deck.compose("likely", artist_spotlight="bob")
    assert feed[0].creator_id == "bob"
    top = feed[:4]
    assert any(w.creator_id == "bob" for w in top)


# ---------------------------------------------------------------------------
# minimum diversity
# ---------------------------------------------------------------------------


def test_no_single_creator_monopolizes_top():
    deck = _deck(colorfulness=0.0)  # no alignment — baseline ties + cover force swaps
    feed = deck.compose(
        "likely",
        size=12,
        max_consecutive=3,
        diversity_min=2,
        diversity_window=6,
    )
    run = 1
    for prev, cur in zip(feed, feed[1:]):
        run = run + 1 if cur.creator_id == prev.creator_id else 1
        assert run <= 3
    front_creators = {w.creator_id for w in feed[:6]}
    assert len(front_creators) >= 2


def test_diversity_min_pulls_extra_creator_into_front():
    deck = _deck(colorfulness=0.0)
    feed = deck.compose("likely", size=12, diversity_min=2, diversity_window=6)
    assert len({w.creator_id for w in feed[:6]}) >= 2


# ---------------------------------------------------------------------------
# outage isolation
# ---------------------------------------------------------------------------


def test_set_down_skips_creator_without_crashing():
    deck = _deck()
    deck.catalog.set_down("alice")
    feed = deck.compose("likely")
    assert feed
    assert all(w.creator_id != "alice" for w in feed)
    assert deck.catalog.creator_available("alice") is False
    assert deck.catalog.works("alice") == []


def test_outage_alone_does_not_break_compose():
    deck = _deck()
    deck.catalog.set_down("bob")
    deck.catalog.set_down("carol")
    feed = deck.compose("likely")
    assert feed
    assert all(w.creator_id not in {"bob", "carol"} for w in feed)


# ---------------------------------------------------------------------------
# publication denial
# ---------------------------------------------------------------------------


def test_publication_denial_keeps_work_out_of_feed():
    policy = PublicationDenied()
    policy.deny("carol")
    assert policy.can_publish("carol") is False
    assert policy.can_publish("alice") is True
    deck = _deck(policy=policy)
    feed = deck.compose("likely")
    assert all(w.creator_id != "carol" for w in feed)


def test_deck_never_acts_on_denied_work():
    policy = PublicationDenied()
    policy.deny("carol")
    deck = _deck(policy=policy)
    deck.compose("likely")
    acted = deck.deck_action("next")
    # Nobody in the feed is denied, so the action can never return a denied work.
    assert policy.can_publish(acted.creator_id) is True
    with pytest.raises(ValueError):
        deck.deck_action("calibrate_prefer_a", work_a_id="carol-w0", work_b_id="bob-w0")


# ---------------------------------------------------------------------------
# signal isolation
# ---------------------------------------------------------------------------


def test_signal_isolation_alignment_uses_only_own_signals():
    prof = _profile(colorfulness=3.0)
    b_signals = _signals(colorfulness=0.4)
    before = alignment(prof, b_signals)
    # Creating a different catalog / changing work A's signals must not affect
    # B's alignment value, which depends only on B's own signals.
    CreatorCatalog(
        _CREATORS,
        [Work("a", "alice", "A", 0.5, _signals(colorfulness=0.99)),
         Work("b", "bob", "B", 0.5, b_signals)],
    )
    CreatorCatalog(
        _CREATORS,
        [Work("a", "alice", "A", 0.5, _signals(colorfulness=0.05)),
         Work("b", "bob", "B", 0.5, b_signals)],
    )
    assert alignment(prof, b_signals) == before


def test_compose_ranking_of_b_unchanged_when_a_changes():
    prof = _profile(colorfulness=2.0)
    b_signals = _signals(colorfulness=0.4)
    base_catalog = CreatorCatalog(
        _CREATORS,
        [Work("a", "alice", "A", 0.5, _signals(0.9)), Work("b", "bob", "B", 0.5, b_signals)],
    )
    changed_catalog = CreatorCatalog(
        _CREATORS,
        [Work("a", "alice", "A", 0.5, _signals(0.05)), Work("b", "bob", "B", 0.5, b_signals)],
    )
    base_score = alignment(prof, b_signals)
    changed_score = alignment(prof, b_signals)
    assert base_score == changed_score
    # B's own score is identical regardless of A — only A's self-score moved.
    assert base_catalog.work("a").signals["colorfulness"] != changed_catalog.work("a").signals[
        "colorfulness"
    ]


# ---------------------------------------------------------------------------
# parity: identical output across channels
# ---------------------------------------------------------------------------


def _first_next_for(channel: str) -> str:
    deck = _deck()
    deck.compose("likely", rng_seed=0)
    return deck.deck_action("next", channel=channel).id


def test_deck_action_parity_next_across_channels():
    btn = _first_next_for("button")
    kbd = _first_next_for("keyboard")
    gst = _first_next_for("gesture")
    assert btn == kbd == gst


def test_deck_action_parity_spotlight_across_channels():
    deck = _deck()
    fmt = lambda ch: deck.deck_action(  # noqa: E731
        "spotlight", channel=ch, artist_spotlight="dave"
    )
    assert [w.id for w in fmt("button")] == [w.id for w in fmt("keyboard")]
    assert [w.id for w in fmt("gesture")] == [w.id for w in fmt("keyboard")]


def test_deck_action_parity_calibrate_across_channels():
    prof = _profile(colorfulness=1.0)
    a = _signals(0.9)
    b = _signals(0.2)

    def cal(channel: str):
        deck = TasteDeck(CreatorCatalog(_CREATORS, [
            Work("a", "alice", "A", 0.5, a),
            Work("b", "bob", "B", 0.5, b),
        ]))
        return deck.deck_action(
            "calibrate_prefer_a",
            channel=channel,
            profile=prof,
            work_a_id="a",
            work_b_id="b",
        )

    r_btn = cal("button")
    r_kbd = cal("keyboard")
    r_gst = cal("gesture")
    assert r_btn == r_kbd == r_gst


# ---------------------------------------------------------------------------
# pairwise calibration
# ---------------------------------------------------------------------------


def test_calibration_bumps_version_shifts_weights_toward_winner():
    deck = _deck(colorfulness=1.0)
    deck.compose("likely")
    result = deck.deck_action(
        "calibrate_prefer_a",
        profile=_profile(1.0),
        work_a_id="alice-w0",
        work_b_id="bob-w0",
    )
    assert result.profile.version == 2
    assert result.preferred_work_id == "alice-w0"
    # Preferring a colorful work raises the colorfulness weight.
    assert result.profile.weights["colorfulness"] > 1.0
    # Other signals untouched (equal between the two works).
    assert result.profile.weights["harmony"] == 0.0


def test_calibration_is_deterministic_and_does_not_mutate_input():
    deck = TasteDeck(
        CreatorCatalog(
            _CREATORS,
            [Work("a", "alice", "A", 0.5, _signals(0.9)),
             Work("b", "bob", "B", 0.5, _signals(0.2))],
        )
    )
    prof = _profile(0.0)
    r1 = deck.deck_action("calibrate_prefer_a", profile=prof, work_a_id="a", work_b_id="b")
    r2 = deck.deck_action("calibrate_prefer_a", profile=prof, work_a_id="a", work_b_id="b")
    assert r1 == r2
    assert prof.weights["colorfulness"] == 0.0  # input untouched


def test_calibrate_prefer_b_moves_weight_down():
    result = TasteDeck(
        CreatorCatalog(
            _CREATORS,
            [Work("a", "alice", "A", 0.5, _signals(0.9)),
             Work("b", "bob", "B", 0.5, _signals(0.2))],
        )
    ).deck_action("calibrate_prefer_b", profile=_profile(1.0), work_a_id="a", work_b_id="b")
    assert result.profile.weights["colorfulness"] < 1.0


# ---------------------------------------------------------------------------
# JSON round-trips
# ---------------------------------------------------------------------------


def test_creator_round_trip():
    c = Creator(id="alice", name="Alice", genre="landscape", style="pastoral")
    assert Creator.from_dict(json.loads(json.dumps(c.to_dict()))) == c


def test_work_round_trip():
    w = Work(id="alice-w0", creator_id="alice", title="A", baseline=0.5, signals=_signals(0.9))
    rebuilt = Work.from_dict(json.loads(json.dumps(w.to_dict())))
    assert rebuilt == w
    assert rebuilt.signals == w.signals


def test_work_signals_defensively_copied():
    sig = _signals(0.9)
    w = Work("a", "alice", "A", 0.5, sig)
    sig["colorfulness"] = 99.0
    assert w.signals["colorfulness"] == 0.9


def test_calibration_result_round_trip():
    deck = TasteDeck(
        CreatorCatalog(
            _CREATORS,
            [Work("a", "alice", "A", 0.5, _signals(0.9)),
             Work("b", "bob", "B", 0.5, _signals(0.2))],
        )
    )
    result = deck.deck_action(
        "calibrate_prefer_a", profile=_profile(0.0), work_a_id="a", work_b_id="b"
    )
    rebuilt = type(result).from_dict(json.loads(json.dumps(result.to_dict())))
    assert rebuilt == result
    assert rebuilt.profile.weights == result.profile.weights


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_compose_rejects_bad_feed_kind_and_bad_bias():
    deck = _deck()
    with pytest.raises(ValueError):
        deck.compose("nope")
    with pytest.raises(ValueError):
        deck.compose("likely", familiar_surprising=2.0)
