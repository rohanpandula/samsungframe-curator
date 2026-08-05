"""Tests for src/curator/taste/dialogue/profile.py (M008/S04/T1+T2).

Covers the TasteProfile/TasteClaim/EvidenceRef/ProfileEvent/WhatILearned JSON
round-trips, the deterministic ProfileBuilder (vocabulary counts, traceable
pattern evidence, surfaced-not-smoothed tensions, evolution against a previous
profile), the append-only ProfileStore timeline (pin/edit/dispute with no
erasure and evidence marked for re-interpretation on dispute), and the
no-silent-learning WhatILearned delta.
"""

from __future__ import annotations

import json

import pytest

from curator import cli
from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.catalog import Catalog
from curator.taste.dialogue.observation import ImageRef, Polarity, TasteObservation
from curator.taste.dialogue.profile import (
    ColdStartSeeder,
    EvidenceRef,
    ProfileBuilder,
    ProfileEvent,
    ProfileStore,
    TasteClaim,
    TasteProfile,
    WhatILearned,
)
from curator.taste.dialogue.store import ObservationStore


def _image(sha: str) -> ImageRef:
    return ImageRef(
        sha256=sha,
        thumb_path=f"/tmp/thumbs/{sha[:8]}.jpg",
        ephemeral=True,
    )


def _observation(
    session_id: str,
    verbatim: str,
    attributes: list[str],
    *,
    polarity: Polarity = Polarity.LIKE,
    confidence: float = 0.9,
    images: list[ImageRef] | None = None,
    created_at: str = "2026-08-04T10:00:00Z",
) -> TasteObservation:
    return TasteObservation(
        session_id=session_id,
        verbatim=verbatim,
        attributes=attributes,
        polarity=polarity,
        confidence=confidence,
        images=images if images is not None else [_image(f"{session_id}-{verbatim[:8]}")],
        created_at=created_at,
    )


def _store(data_root) -> ObservationStore:
    return ObservationStore(Catalog(data_root=data_root))


# -- round-trips -------------------------------------------------------------


def test_evidence_ref_round_trip():
    ref = EvidenceRef(
        image_sha="a" * 64,
        verbatim="i love the quiet empty scene",
        confidence=0.9,
        created_at="2026-08-04T10:00:00Z",
    )
    rebuilt = EvidenceRef.from_dict(json.loads(json.dumps(ref.to_dict())))
    assert rebuilt == ref
    assert EvidenceRef.from_dict(rebuilt) == ref


def test_taste_claim_round_trip():
    claim = TasteClaim(
        id="pattern:negative-space",
        text="you favor negative-space — quiet, empty (2 uses)",
        evidence=[
            EvidenceRef(
                image_sha="b" * 64,
                verbatim="i love the quiet empty scene",
                confidence=0.9,
                created_at="2026-08-04T10:00:00Z",
            )
        ],
        status="active",
        provenance="high",
        created_at="2026-08-04T10:00:00Z",
    )
    rebuilt = TasteClaim.from_dict(json.loads(json.dumps(claim.to_dict())))
    assert rebuilt == claim
    assert TasteClaim.from_dict(rebuilt) == claim
    assert rebuilt.evidence[0].image_sha == "b" * 64
    assert rebuilt.evidence[0].verbatim == claim.evidence[0].verbatim


def test_profile_event_round_trip():
    event = ProfileEvent(
        claim_id="pattern:negative-space",
        kind="pin",
        detail="pattern:negative-space",
        created_at="2026-08-04T10:00:00Z",
    )
    rebuilt = ProfileEvent.from_dict(json.loads(json.dumps(event.to_dict())))
    assert rebuilt == event
    assert ProfileEvent.from_dict(rebuilt) == event


def test_taste_profile_round_trip():
    profile = ProfileBuilder().build(
        [
            _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
            _observation("s2", "so much negative space", ["negative-space"]),
        ]
    )
    rebuilt = TasteProfile.from_dict(json.loads(json.dumps(profile.to_dict())))
    assert rebuilt == profile
    assert TasteProfile.from_dict(rebuilt) == profile
    assert rebuilt.vocabulary["quiet"]["attribute"] == "negative-space"
    assert rebuilt.patterns[0].evidence[0].image_sha


def test_what_i_learned_round_trip():
    learned = WhatILearned(
        summary="Learned from 1 new reactions.\nadded 'quiet' to vocabulary.",
        added=["added 'quiet' to vocabulary"],
        version=2,
    )
    rebuilt = WhatILearned.from_dict(json.loads(json.dumps(learned.to_dict())))
    assert rebuilt == learned
    assert WhatILearned.from_dict(rebuilt) == learned


def test_invalid_claim_status_and_event_kind_rejected():
    with pytest.raises(ValueError):
        TasteClaim(id="x", text="t", evidence=[], status="bogus")
    with pytest.raises(ValueError):
        ProfileEvent(claim_id="x", kind="bogus", detail="", created_at="")


# -- ProfileBuilder ----------------------------------------------------------


def test_vocabulary_counts_and_attribute_mapping():
    obs = [
        _observation("s1", "i love the quiet scene", ["quiet"]),
        _observation("s2", "another quiet calm image", ["quiet"]),
        _observation("s3", "quiet again", ["quiet"]),
        _observation("s4", "the warm glow", ["warm-tones"]),
        _observation("s5", "so warm here", ["warm-tones"]),
        _observation("s6", "the empty horizon", ["breathing-room"]),
    ]
    profile = ProfileBuilder().build(obs)
    assert profile.vocabulary["quiet"]["usage_count"] == 3
    assert profile.vocabulary["quiet"]["attribute"] == "negative-space"
    assert profile.vocabulary["warm"]["usage_count"] == 2
    assert profile.vocabulary["warm"]["attribute"] == "warm-tones"
    assert profile.vocabulary["empty"]["usage_count"] == 1
    assert profile.vocabulary["empty"]["attribute"] == "negative-space"


def test_patterns_have_traceable_evidence():
    obs = [
        _observation(
            "s1", "i love the quiet empty scene", ["negative-space", "quiet"],
            images=[_image("1" * 64)],
        ),
        _observation(
            "s2", "so much negative space", ["negative-space"],
            images=[_image("2" * 64)],
        ),
        _observation(
            "s3", "negative space again", ["negative-space"],
            images=[_image("3" * 64), _image("4" * 64)],
        ),
    ]
    profile = ProfileBuilder().build(obs)
    pattern = next(c for c in profile.patterns if c.id == "pattern:negative-space")
    assert pattern.status == "active"
    assert pattern.provenance == "high"
    assert "3 uses" in pattern.text
    assert pattern.evidence
    assert len(pattern.evidence) == 4  # 1 + 1 + 2 image refs
    for ref in pattern.evidence:
        assert ref.image_sha
        assert ref.verbatim
        assert 0.0 <= ref.confidence <= 1.0
        assert ref.created_at
    shas = {ref.image_sha for ref in pattern.evidence}
    assert {"1" * 64, "2" * 64, "3" * 64, "4" * 64} <= shas
    verbatims = {ref.verbatim for ref in pattern.evidence}
    assert {
        "i love the quiet empty scene",
        "so much negative space",
        "negative space again",
    } == verbatims


def test_single_use_attribute_produces_no_pattern():
    obs = [
        _observation("s1", "i love the quiet scene", ["quiet"]),
        _observation("s2", "the warm glow", ["warm-tones"]),
    ]
    profile = ProfileBuilder().build(obs)
    assert profile.patterns == []
    assert profile.tensions == []


def test_tension_polarity_conflict_surfaced_not_smoothed():
    obs = [
        _observation("s1", "i love negative space", ["negative-space"], polarity=Polarity.LIKE),
        _observation("s2", "negative space again", ["negative-space"], polarity=Polarity.LIKE),
        _observation("s3", "i hate negative space", ["negative-space"], polarity=Polarity.DISLIKE),
    ]
    profile = ProfileBuilder().build(obs)
    tension = next(c for c in profile.tensions if c.id == "tension:negative-space:polarity")
    assert "both liked (2 uses) and disliked (1 uses)" in tension.text
    assert "surfaced, not smoothed" in tension.text
    assert tension.evidence
    for ref in tension.evidence:
        assert ref.image_sha and ref.verbatim
    assert "pattern:negative-space" in {c.id for c in profile.patterns}


def test_tension_contrast_pair_both_liked():
    obs = [
        _observation("s1", "i love minimal rooms", ["minimal"], polarity=Polarity.LIKE),
        _observation("s2", "minimal and clean", ["minimal"], polarity=Polarity.LIKE),
        _observation("s3", "the dense city", ["dense"], polarity=Polarity.LIKE),
        _observation("s4", "so dense", ["dense"], polarity=Polarity.LIKE),
    ]
    profile = ProfileBuilder().build(obs)
    tension = next(c for c in profile.tensions if c.id == "tension:minimal:dense")
    assert "minimal" in tension.text and "dense" in tension.text
    assert "surfaced, not smoothed" in tension.text
    assert tension.evidence


def test_evolution_reflects_prior_profile():
    builder = ProfileBuilder()
    first = builder.build(
        [_observation("s1", "i love the quiet scene", ["quiet"])]
    )
    second = builder.build(
        [
            _observation("s1", "i love the quiet scene", ["quiet"]),
            _observation("s2", "the warm glow", ["warm-tones"]),
            _observation("s3", "warm and sunny", ["warm-tones"]),
            _observation("s4", "so warm", ["warm-tones"]),
        ],
        previous=first,
    )
    summaries = [e["summary"] for e in second.evolution]
    assert any("added 'warm' to vocabulary" in s for s in summaries)
    assert any("new pattern: warm-tones" in s for s in summaries)
    assert second.version == first.version + 1
    assert all(e["at"] == second.created_at for e in second.evolution)


def test_builder_is_deterministic():
    obs = [
        _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
        _observation("s2", "so much negative space", ["negative-space"]),
        _observation("s3", "the warm glow", ["warm-tones"]),
    ]
    first = ProfileBuilder().build(obs).to_dict()
    second = ProfileBuilder().build(obs).to_dict()
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_evolution_initial_profile_and_no_change():
    builder = ProfileBuilder()
    profile = builder.build([])
    assert profile.evolution[0]["summary"].startswith("initial profile:")
    assert profile.version == 1
    again = builder.build([], previous=profile)
    assert any(e["summary"] == "no material change" for e in again.evolution)
    assert again.version == 2


# -- ProfileStore ------------------------------------------------------------


def test_pin_persists_claim_status(data_root):
    profile = ProfileBuilder().build(
        [
            _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
            _observation("s2", "so much negative space", ["negative-space"]),
        ]
    )
    store = ProfileStore(Catalog(data_root=data_root))
    store.apply(profile)

    event = store.pin("pattern:negative-space")
    assert event.kind == "pin"
    assert event.claim_id == "pattern:negative-space"

    loaded = store.load()
    assert loaded is not None
    pattern = next(c for c in loaded.patterns if c.id == "pattern:negative-space")
    assert pattern.status == "pinned"


def test_edit_updates_text_and_status(data_root):
    profile = ProfileBuilder().build(
        [
            _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
            _observation("s2", "so much negative space", ["negative-space"]),
        ]
    )
    store = ProfileStore(Catalog(data_root=data_root))
    store.apply(profile)

    event = store.edit("pattern:negative-space", "I actually prefer breathing room.")
    assert event.kind == "edit"
    assert event.detail == "I actually prefer breathing room."

    loaded = store.load()
    pattern = next(c for c in loaded.patterns if c.id == "pattern:negative-space")
    assert pattern.text == "I actually prefer breathing room."
    assert pattern.status == "edited"


def test_events_append_only_no_erasure(data_root):
    profile = ProfileBuilder().build(
        [
            _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
            _observation("s2", "so much negative space", ["negative-space"]),
        ]
    )
    store = ProfileStore(Catalog(data_root=data_root))
    store.apply(profile)

    store.pin("pattern:negative-space")
    store.edit("pattern:negative-space", "edited text")
    store.dispute("pattern:negative-space")

    events = store.events()
    assert [e.kind for e in events] == ["pin", "edit", "dispute"]
    assert [e.claim_id for e in events] == ["pattern:negative-space"] * 3
    assert all(e.created_at for e in events)
    pin, edit, dispute = events
    assert pin.detail == "pattern:negative-space"
    assert edit.detail == "edited text"
    assert "re-interpretation" in dispute.detail


def test_dispute_removes_claim_and_marks_evidence(data_root):
    profile = ProfileBuilder().build(
        [
            _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
            _observation("s2", "so much negative space", ["negative-space"]),
        ]
    )
    store = ProfileStore(Catalog(data_root=data_root))
    store.apply(profile)

    event = store.dispute("pattern:negative-space")
    assert event.kind == "dispute"
    for sha in ("s1-i love t", "s2-so much "):
        assert sha in event.detail

    loaded = store.load()
    assert "pattern:negative-space" not in {c.id for c in loaded.patterns}
    assert "pattern:negative-space" not in {c.id for c in loaded.tensions}
    assert loaded.vocabulary["quiet"]["usage_count"] == 1


def test_events_persist_across_store_instances(data_root):
    profile = ProfileBuilder().build(
        [
            _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
            _observation("s2", "so much negative space", ["negative-space"]),
        ]
    )
    first = ProfileStore(Catalog(data_root=data_root))
    first.apply(profile)
    first.pin("pattern:negative-space")

    second = ProfileStore(Catalog(data_root=data_root))
    assert [e.kind for e in second.events()] == ["pin"]
    loaded = second.load()
    assert next(c for c in loaded.patterns if c.id == "pattern:negative-space").status == "pinned"


def test_store_load_returns_none_without_apply(data_root):
    store = ProfileStore(Catalog(data_root=data_root))
    assert store.load() is None
    assert store.events() == []


def test_mutation_without_applied_profile_still_records_event(data_root):
    store = ProfileStore(Catalog(data_root=data_root))
    event = store.pin("pattern:missing")
    assert event.kind == "pin"
    assert [e.kind for e in store.events()] == ["pin"]
    assert store.load() is None


# -- WhatILearned ------------------------------------------------------------


def test_delta_after_session_with_observations_is_non_empty(data_root):
    store = _store(data_root)
    for obs in (
        _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
        _observation("s2", "so much negative space", ["negative-space"]),
    ):
        store.add(obs)
    store.add(
        _observation("s3", "the warm glow", ["warm-tones"], created_at="2026-08-04T11:00:00Z")
    )

    learned = WhatILearned.delta_after("s3", store)
    assert learned.summary
    assert learned.added
    assert any("warm" in item for item in learned.added)
    assert learned.version >= 1


def test_delta_after_first_session_lists_additions(data_root):
    store = _store(data_root)
    store.add(
        _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"])
    )
    store.add(
        _observation("s1", "so much negative space", ["negative-space"])
    )

    learned = WhatILearned.delta_after("s1", store)
    assert learned.summary
    assert learned.added
    assert any("added 'quiet' to vocabulary" in item for item in learned.added)
    assert any("new pattern: negative-space" in item for item in learned.added)
    assert learned.version == 1


def test_delta_after_session_with_no_observations_is_noop(data_root):
    store = _store(data_root)
    store.add(
        _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"])
    )

    learned = WhatILearned.delta_after("never-existed", store)
    assert learned.added == []
    assert learned.version == 0
    assert "No new observations" in learned.summary


def test_delta_never_empty_when_session_added_observations(data_root):
    store = _store(data_root)
    for obs in (
        _observation("s1", "i love the quiet empty scene", ["negative-space", "quiet"]),
        _observation("s2", "so much negative space", ["negative-space"]),
    ):
        store.add(obs)

    learned = WhatILearned.delta_after("s1", store)
    assert learned.summary
    assert learned.added

# -- ColdStartSeeder ---------------------------------------------------------


def _seed_history(
    data_root,
    *,
    approvals: dict[int, tuple[str, str]] | None = None,
    preferences: list[tuple[int, int, str]] | None = None,
    liked_colorfulness: float = 0.9,
    other_colorfulness: float = 0.1,
    liked_entries: tuple[int, ...] = (1, 2, 3),
    entries: int = 6,
) -> Catalog:
    """Build a catalog with analyzed entries plus approval/pairwise history.

    Entries in *liked_entries* analyze as colorful, the rest as drab, so the
    seeder has a real signal separation to find. Returns an open Catalog.
    """
    catalog = Catalog(data_root=data_root)
    db = catalog.db
    db.execute(
        "INSERT INTO source_connectors(connector_id, connector_type, name)"
        " VALUES ('local', 'local', 'local')"
    )
    for i in range(1, entries + 1):
        sha = f"{i:064d}"
        db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (sha, 100))
        db.execute(
            "INSERT INTO catalog_entries(connector_id, asset_id, revision, sha256)"
            " VALUES ('local', ?, '1', ?)",
            (f"asset-{i}", sha),
        )
        analysis = AnalysisResult(
            asset_id=sha,
            quality=QualitySignals(technical_quality=0.5, aesthetic_quality=0.5),
            color_story=ColorStory(
                colorfulness=(
                    liked_colorfulness if i in liked_entries else other_colorfulness
                ),
                harmony=0.5,
            ),
        )
        db.execute(
            "INSERT INTO analysis_results(catalog_entry_id, profile, engine_version,"
            " analysis_json, status) VALUES (?, 'standard', 'v1', ?, 'ok')",
            (i, json.dumps(analysis.to_dict())),
        )
    for entry_id, (decision, rationale) in (approvals or {}).items():
        db.execute(
            "INSERT INTO approvals(catalog_entry_id, decision, rationale)"
            " VALUES (?, ?, ?)",
            (entry_id, decision, rationale),
        )
    for entry_id, preference, note in preferences or []:
        db.execute(
            "INSERT INTO taste_preferences(profile_id, catalog_entry_id, preference, note)"
            " VALUES (1, ?, ?, ?)",
            (entry_id, preference, note),
        )
    db.commit()
    return catalog


_APPROVED_HISTORY = {
    1: ("APPROVED", "love the color"),
    2: ("APPROVED", ""),
    3: ("APPROVED", ""),
    4: ("REJECTED", ""),
    5: ("REJECTED", ""),
    6: ("REJECTED", ""),
}


def test_seeder_recovers_latest_decision_per_entry(data_root):
    catalog = _seed_history(data_root, approvals=_APPROVED_HISTORY)
    # Entry 4 flips: the newest row (APPROVED) is the one that counts.
    catalog.db.execute(
        "INSERT INTO approvals(catalog_entry_id, decision, rationale)"
        " VALUES (4, 'APPROVED', 'changed my mind')"
    )
    catalog.db.commit()

    decisions = ColdStartSeeder(catalog).decisions()
    by_entry = {d.catalog_entry_id: d for d in decisions if d.source == "approval"}
    assert len(by_entry) == 6
    assert by_entry[4].liked is True
    assert by_entry[4].note == "changed my mind"
    assert by_entry[5].liked is False


def test_seeder_reads_pairwise_rows_and_skips_abstentions(data_root):
    catalog = _seed_history(
        data_root,
        approvals=_APPROVED_HISTORY,
        preferences=[(1, 1, "picked A"), (2, 1, ""), (4, -1, ""), (5, 0, "abstained")],
    )
    pairwise = [d for d in ColdStartSeeder(catalog).decisions() if d.source == "pairwise"]
    assert [d.catalog_entry_id for d in pairwise] == [1, 2, 4]
    assert [d.liked for d in pairwise] == [True, True, False]


def test_seeder_claims_are_low_provenance_with_traceable_evidence(data_root):
    catalog = _seed_history(
        data_root,
        approvals=_APPROVED_HISTORY,
        preferences=[(1, 1, "picked A"), (2, 1, ""), (4, -1, "")],
    )
    claims = ColdStartSeeder(catalog).claims()
    assert claims
    assert all(c.provenance == "low" for c in claims)
    ids = {c.id for c in claims}
    assert "history:approval:colorfulness" in ids
    assert "history:pairwise:colorfulness" in ids

    claim = next(c for c in claims if c.id == "history:approval:colorfulness")
    assert "high colorfulness" in claim.text
    assert "not your words" in claim.text
    assert len(claim.evidence) == 3  # the three approved entries
    assert {ref.image_sha for ref in claim.evidence} == {f"{i:064d}" for i in (1, 2, 3)}
    assert all(ref.confidence == 0.3 for ref in claim.evidence)


def test_seeder_evidence_verbatim_marks_absent_user_words(data_root):
    catalog = _seed_history(data_root, approvals=_APPROVED_HISTORY)
    claim = next(
        c for c in ColdStartSeeder(catalog).claims()
        if c.id == "history:approval:colorfulness"
    )
    verbatims = {ref.verbatim for ref in claim.evidence}
    assert "love the color" in verbatims  # a recorded rationale is used as-is
    assert any("no verbatim" in v for v in verbatims)  # the rest are marked


def test_seeder_below_minimum_samples_makes_no_claims(data_root):
    catalog = _seed_history(
        data_root, approvals={1: ("APPROVED", ""), 4: ("REJECTED", "")}
    )
    assert ColdStartSeeder(catalog).claims() == []


def test_seeder_on_empty_catalog_is_a_noop(data_root):
    catalog = Catalog(data_root=data_root)
    seeder = ColdStartSeeder(catalog)
    assert seeder.decisions() == []
    assert seeder.claims() == []

    base = ProfileBuilder().build([])
    assert seeder.seed(base) == base
    assert seeder.seed() == ProfileBuilder().build([])


def test_seed_labels_low_and_high_provenance_with_high_first(data_root):
    catalog = _seed_history(data_root, approvals=_APPROVED_HISTORY)
    observed = ProfileBuilder().build(
        [
            _observation("s1", "i love the quiet empty scene", ["negative-space"]),
            _observation("s2", "so much negative space", ["negative-space"]),
        ]
    )
    seeded = ColdStartSeeder(catalog).seed(observed)

    provenances = [c.provenance for c in seeded.patterns]
    assert provenances[0] == "high"
    assert "low" in provenances
    # Reaction Room claims keep their position; history is appended after.
    assert seeded.patterns[0].id == "pattern:negative-space"
    assert all(c.id.startswith("history:") for c in seeded.patterns[1:])
    # Seeding is never silent.
    assert any("seeded" in e["summary"] for e in seeded.evolution)


def test_seeder_is_deterministic(data_root):
    catalog = _seed_history(
        data_root, approvals=_APPROVED_HISTORY, preferences=[(1, 1, ""), (2, 1, "")]
    )
    seeder = ColdStartSeeder(catalog)
    first = [c.to_dict() for c in seeder.claims()]
    second = [c.to_dict() for c in ColdStartSeeder(catalog).claims()]
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# -- CLI ---------------------------------------------------------------------


def _write_observations(data_root) -> None:
    """Persist two dialogue observations, then release the connection."""
    catalog = Catalog(data_root=data_root)
    store = ObservationStore(catalog)
    store.add(_observation("s1", "i love the quiet empty scene", ["negative-space"]))
    store.add(_observation("s2", "so much negative space", ["negative-space"]))
    catalog.db.close()


def test_cli_profile_renders_document_after_observations(data_root, capsys):
    _write_observations(data_root)

    assert cli.main(["taste", "profile"]) == 0
    out = capsys.readouterr().out
    for section in ("Vocabulary:", "Patterns:", "Tensions:", "Evolution:"):
        assert section in out
    assert "pattern:negative-space" in out
    assert "i love the quiet empty scene" in out  # evidence opens the verbatim


def test_cli_profile_json_carries_the_document(data_root, capsys):
    _write_observations(data_root)

    assert cli.main(["taste", "profile", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) >= {
        "vocabulary", "patterns", "tensions", "evolution", "version"
    }
    assert payload["vocabulary"]["quiet"]["attribute"] == "negative-space"
    assert payload["patterns"][0]["id"] == "pattern:negative-space"


def test_cli_profile_no_seed_omits_history_claims(data_root, capsys):
    catalog = _seed_history(data_root, approvals=_APPROVED_HISTORY)
    catalog.db.close()

    assert cli.main(["taste", "profile", "--json"]) == 0
    seeded = json.loads(capsys.readouterr().out)
    assert any(c["id"].startswith("history:") for c in seeded["patterns"])

    assert cli.main(["taste", "profile", "--json", "--no-seed"]) == 0
    unseeded = json.loads(capsys.readouterr().out)
    assert not any(c["id"].startswith("history:") for c in unseeded["patterns"])


def test_cli_dispute_removes_claim_and_survives_rebuild(data_root, capsys):
    _write_observations(data_root)

    assert cli.main(["taste", "dispute", "pattern:negative-space"]) == 0
    out = capsys.readouterr().out
    assert "pattern:negative-space" in out
    assert "re-interpretation" in out

    assert cli.main(["taste", "profile", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "pattern:negative-space" not in {c["id"] for c in payload["patterns"]}


def test_cli_dispute_json_emits_the_event(data_root, capsys):
    _write_observations(data_root)

    assert cli.main(["taste", "dispute", "pattern:negative-space", "--json"]) == 0
    event = json.loads(capsys.readouterr().out)
    assert event["kind"] == "dispute"
    assert event["claim_id"] == "pattern:negative-space"
    assert event["created_at"]


def test_cli_dispute_unknown_claim_reports_no_change(data_root, capsys):
    _write_observations(data_root)

    assert cli.main(["taste", "dispute", "pattern:nope"]) == 3
    assert "pattern:nope" in capsys.readouterr().out
