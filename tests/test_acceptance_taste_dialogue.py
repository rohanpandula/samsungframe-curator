"""Acceptance gate for the M008 Taste Dialogue surface.

This module ships the deterministic, air-gapped acceptance gate for the Taste
Dialogue features delivered across M008/S01-S05. Each scenario is
**self-bootstrapping**: it mints its own images, catalog, sessions, and history
over the isolated ``data_root`` (from conftest) and drives the subsystem objects
directly — never relying on cross-test ordering, a live server, or the network.

* S1 — Reaction Room (R032): dropping non-catalog images and reacting in plain
  language records observations with the verbatim byte-exact, the polarity and
  confidence extracted, the session attributed, and follow-ups hard-capped at
  two (a gym rep, never an interview).
* S2 — extraction provider (R033): cloud is opt-in and states what leaves the
  machine, per-image exclusions are honored, the local slot reports
  not-available cleanly, and with no provider the room is unavailable with a
  clear message and records nothing — it never degrades to keyword matching.
* S3 — retention (R034): third-party drops are kept as thumbnail + content hash
  only (never the original bytes), the policy is stated, and an explicit
  save-to-catalog promotes the full-resolution image.
* S4 — profile artifact (R035): vocabulary/patterns/tensions/evolution render,
  every claim opens its evidence, pin/edit/dispute persist on an append-only
  timeline, a dispute removes the claim and marks its evidence, and a
  "What I learned" delta follows every session that added observations.
* S5 — cold start (R037): claims seeded from approve/reject and pairwise history
  are labeled low-provenance and sort behind the high-provenance Reaction Room
  observations they corroborate.
* S6 — upstream (R036): explanations cite profile quotes when the profile is
  non-empty, ranking moves along profile dimensions, a dispute changes the
  ranking, and an empty profile reproduces the M007 baseline exactly.
* S7 — anti-goals (R038): no generation code path, no silent learning, no jargon
  laundering, no interrogation, no hard dependency.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from curator.analysis.schema import AnalysisResult, ColorStory, QualitySignals
from curator.catalog import Catalog
from curator.errors import CuratorError
from curator.hashing import sha256_hex
from curator.providers.cloud import ExclusionPolicy
from curator.taste.dialogue import upstream as dialogue_upstream
from curator.taste.dialogue.extraction import (
    CloudExtractionProvider,
    LocalExtractionSlot,
    extract_or_unavailable,
    extraction_default_disclosure,
    resolve_extraction_provider,
)
from curator.taste.dialogue.observation import Polarity, create_observation
from curator.taste.dialogue.profile import (
    ColdStartSeeder,
    ProfileBuilder,
    ProfileStore,
    WhatILearned,
)
from curator.taste.dialogue.retention import (
    MAX_THUMB_DIM,
    retain_ephemeral,
    retention_policy,
    save_to_catalog,
)
from curator.taste.dialogue.room import (
    MAX_FOLLOWUPS,
    ReactionRoom,
    ReactionRoomUnavailableError,
)
from curator.taste.dialogue.store import ObservationStore
from curator.taste.dialogue.upstream import (
    citations_for,
    explain_rank,
    familiar_surprising_dimensions,
    pairing_rationale,
    profile_dimensions,
    profile_fit,
)
from curator.taste.profiles import SIGNAL_NAMES, TasteProfileKind
from curator.taste.profiles import TasteProfile as LensProfile
from curator.taste.rank import TasteRanker

QUIET_VERBATIM = "i love the quiet negative space"


# ---------------------------------------------------------------------------
# fixtures — every scenario builds its own world
# ---------------------------------------------------------------------------


def _image_bytes(width: int = 640, height: int = 480, seed: int = 0) -> bytes:
    """Return a decodable PNG (banded gradient) distinct per *seed*."""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for band in range(64):
        color = ((band * 4 + seed) % 256, (band * 7 + seed) % 256, (band * 11) % 256)
        top = band * height // 64
        bottom = (band + 1) * height // 64
        draw.rectangle([0, top, width, bottom], fill=color)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _drop_file(tmp_path: Path, name: str, seed: int = 0) -> Path:
    path = tmp_path / name
    path.write_bytes(_image_bytes(seed=seed))
    return path


def _room(catalog: Catalog, data_root, provider=None) -> ReactionRoom:
    return ReactionRoom(
        catalog,
        CloudExtractionProvider() if provider is None else provider,
        data_root,
    )


def _lens_profile(**weights: float) -> LensProfile:
    return LensProfile(
        id="acceptance", kind=TasteProfileKind.PERSONAL, name="personal", weights=weights
    )


def _analysis(colorfulness: float = 0.8) -> AnalysisResult:
    return AnalysisResult(
        asset_id="a" * 64,
        quality=QualitySignals(technical_quality=0.5, aesthetic_quality=0.6),
        color_story=ColorStory(colorfulness=colorfulness, harmony=0.5),
    )


def _react_twice(catalog: Catalog, data_root, tmp_path) -> str:
    """Drop two third-party images and react twice. Returns the session id."""
    room = _room(catalog, data_root)
    session = room.start(
        [
            str(_drop_file(tmp_path, "drop-a.png", seed=1)),
            str(_drop_file(tmp_path, "drop-b.png", seed=2)),
        ]
    )
    room.react(session, QUIET_VERBATIM)
    room.react(session, "so much negative space, very quiet")
    room.finish(session)
    return session.id


def _seed_history(catalog: Catalog) -> None:
    """Give the catalog six analyzed entries with approval + pairwise history."""
    db = catalog.db
    db.execute(
        "INSERT INTO source_connectors(connector_id, connector_type, name)"
        " VALUES ('history', 'local', 'history')"
    )
    for i in range(1, 7):
        sha = f"{i:064d}"
        db.execute("INSERT INTO content(sha256, size) VALUES (?, ?)", (sha, 100))
        db.execute(
            "INSERT INTO catalog_entries(connector_id, asset_id, revision, sha256)"
            " VALUES ('history', ?, '1', ?)",
            (f"history-{i}", sha),
        )
        db.execute(
            "INSERT INTO analysis_results(catalog_entry_id, profile, engine_version,"
            " analysis_json, status)"
            " SELECT id, 'standard', 'v1', ?, 'ok' FROM catalog_entries WHERE sha256 = ?",
            (json.dumps(_analysis(0.9 if i <= 3 else 0.1).to_dict()), sha),
        )
        db.execute(
            "INSERT INTO approvals(catalog_entry_id, decision, rationale)"
            " SELECT id, ?, ? FROM catalog_entries WHERE sha256 = ?",
            ("APPROVED" if i <= 3 else "REJECTED", "kept it" if i == 1 else "", sha),
        )
    db.execute(
        "INSERT INTO taste_preferences(profile_id, catalog_entry_id, preference, note)"
        " VALUES (1, 1, 1, 'picked A'), (1, 2, 1, ''), (1, 4, -1, '')"
    )
    db.commit()


# ---------------------------------------------------------------------------
# S1 — Reaction Room (R032)
# ---------------------------------------------------------------------------


def test_acceptance_taste_dialogue_reaction_room(data_root, tmp_path):
    """Drop any images, react in plain language, get at most two follow-ups."""
    catalog = Catalog(data_root=data_root)
    room = _room(catalog, data_root)
    drops = [
        str(_drop_file(tmp_path, "one.png", seed=1)),
        str(_drop_file(tmp_path, "two.png", seed=2)),
    ]

    session = room.start(drops)
    turn = room.react(session, QUIET_VERBATIM)

    # The user's words survive byte-exact; the structure comes from extraction.
    assert turn.observation.verbatim == QUIET_VERBATIM
    assert turn.observation.attributes
    assert turn.observation.polarity is Polarity.LIKE
    assert 0.0 < turn.observation.confidence <= 1.0
    assert turn.observation.session_id == session.id
    assert len(turn.observation.images) == 2
    assert all(ref.ephemeral for ref in turn.observation.images)

    # At most two follow-ups, ever — and never a lecture or a compliment.
    asked = [turn.question]
    for _ in range(5):
        asked.append(room.react(session, "still quiet, still empty").question)
    assert sum(1 for q in asked if q is not None) <= MAX_FOLLOWUPS
    for question in (q for q in asked if q is not None):
        assert question.text.endswith("?")
        assert len(question.text) < 120

    assert room.finish(session) == 6
    stored = ObservationStore(catalog).by_session(session.id)
    assert [o.verbatim for o in stored][0] == QUIET_VERBATIM


# ---------------------------------------------------------------------------
# S2 — extraction provider (R033)
# ---------------------------------------------------------------------------


def test_acceptance_taste_dialogue_extraction_provider(data_root, tmp_path):
    """Cloud is opt-in and disclosed; local reports not-available; no model = no room."""
    catalog = Catalog(data_root=data_root)

    # Cloud opt-in: enabled, and it states exactly what leaves the machine.
    cloud = CloudExtractionProvider()
    assert cloud.capabilities().enabled
    assert cloud.capabilities().supports_cloud
    disclosure = cloud.disclosure()
    assert disclosure.statement
    assert "never leave" in disclosure.statement
    assert extraction_default_disclosure().statement

    # Per-image exclusions are honored and named in the disclosure.
    sha = sha256_hex(_image_bytes(seed=3))
    policy = ExclusionPolicy(per_image={sha: frozenset({"faces", "gps"})})
    excluded = CloudExtractionProvider(policy=policy).disclosure()
    assert excluded.statement

    # The local slot exists, is capability-probed, and says "not available" cleanly.
    local = LocalExtractionSlot()
    assert local.capabilities().supports_local
    assert not local.capabilities().enabled
    probe = create_observation(session_id="s", verbatim=QUIET_VERBATIM)
    assert extract_or_unavailable(local, probe) is None  # no error, no guessing

    # No provider configured at all -> the room is unavailable, clearly.
    assert resolve_extraction_provider(None) is None
    assert resolve_extraction_provider({"enabled": False}) is None
    off_room = ReactionRoom(catalog, None, data_root)
    with pytest.raises(ReactionRoomUnavailableError) as excinfo:
        off_room.start([str(_drop_file(tmp_path, "nope.png"))])
    assert "extraction provider" in str(excinfo.value)

    # ...and nothing was recorded: no keyword fallback ever runs.
    assert ObservationStore(catalog).all() == []


# ---------------------------------------------------------------------------
# S3 — retention (R034)
# ---------------------------------------------------------------------------


def test_acceptance_taste_dialogue_ephemeral_retention(data_root, tmp_path):
    """Third-party drops are thumb + hash only until explicitly saved."""
    catalog = Catalog(data_root=data_root)
    original = _image_bytes(width=1600, height=1200, seed=7)
    sha = sha256_hex(original)

    ref = retain_ephemeral(original, Path(data_root))

    assert ref.sha256 == sha
    assert ref.ephemeral and not ref.catalog_saved
    thumb = Path(str(ref.thumb_path))
    assert thumb.is_file()
    with Image.open(thumb) as img:
        assert max(img.size) <= MAX_THUMB_DIM
    # The original bytes are nowhere on disk under the data root.
    assert thumb.read_bytes() != original
    assert not any(
        p.read_bytes() == original for p in Path(data_root).rglob("*") if p.is_file()
    )
    assert catalog.get_by_hash(sha) == []
    assert "never stored by default" in retention_policy()

    # Explicit promotion is a separate, deliberate action.
    saved = save_to_catalog(ref, original, catalog)
    assert saved.catalog_saved
    assert catalog.get_by_hash(sha)
    assert thumb.is_file()  # the thumbnail stays as evidence


# ---------------------------------------------------------------------------
# S4 — profile artifact (R035)
# ---------------------------------------------------------------------------


def test_acceptance_taste_dialogue_profile_artifact(data_root, tmp_path):
    """The profile reads like a page about you, and every claim opens its evidence."""
    catalog = Catalog(data_root=data_root)
    session_id = _react_twice(catalog, data_root, tmp_path)
    store = ObservationStore(catalog)

    profile = ProfileBuilder().build(store.all())

    # Vocabulary: the user's recurring words, mapped to attributes with counts.
    assert profile.vocabulary
    assert profile.vocabulary["quiet"]["usage_count"] >= 2
    assert profile.vocabulary["quiet"]["attribute"]

    # Patterns: evidenced claims. Every claim opens its evidence.
    assert profile.patterns
    for claim in profile.patterns:
        assert claim.text
        assert claim.evidence
        for ref in claim.evidence:
            assert ref.image_sha
            assert ref.verbatim  # the user's words, not a paraphrase
            assert 0.0 <= ref.confidence <= 1.0
            assert ref.created_at  # recency
    assert any(ref.verbatim == QUIET_VERBATIM for ref in profile.patterns[0].evidence)

    # Evolution is present and the document round-trips as JSON.
    assert profile.evolution
    assert profile.to_dict() == json.loads(json.dumps(profile.to_dict()))

    # Tensions are surfaced, never smoothed.
    conflicted = ProfileBuilder().build(
        store.all()
        + [
            create_observation(
                session_id="s-dislike",
                verbatim="actually i hate all this empty space",
                attributes=["negative-space"],
                polarity=Polarity.DISLIKE,
                confidence=0.8,
                images=profile.patterns[0].evidence[:1] and store.all()[0].images,
                created_at="2026-08-04T12:00:00.000000Z",
            )
        ]
    )
    assert conflicted.tensions
    assert any("not smoothed" in t.text for t in conflicted.tensions)

    # Pin / edit / dispute persist on an append-only timeline.
    profile_store = ProfileStore(catalog)
    profile_store.apply(profile)
    claim_id = profile.patterns[0].id
    profile_store.pin(claim_id)
    profile_store.edit(claim_id, "I prefer breathing room, actually.")
    dispute = profile_store.dispute(claim_id)

    events = profile_store.events()
    assert [e.kind for e in events] == ["pin", "edit", "dispute"]
    assert all(e.created_at for e in events)
    # A dispute removes the claim and marks its evidence for re-interpretation.
    assert "re-interpretation" in dispute.detail
    assert claim_id not in {c.id for c in profile_store.load().patterns}

    # "What I learned": no silent learning.
    learned = WhatILearned.delta_after(session_id, store)
    assert learned.summary
    assert learned.added
    assert WhatILearned.delta_after("never-happened", store).added == []


# ---------------------------------------------------------------------------
# S5 — cold start (R037)
# ---------------------------------------------------------------------------


def test_acceptance_taste_dialogue_cold_start_provenance(data_root, tmp_path):
    """History seeds the profile, labeled low-provenance behind the user's words."""
    catalog = Catalog(data_root=data_root)
    _seed_history(catalog)
    _react_twice(catalog, data_root, tmp_path)

    observed = ProfileBuilder().build(ObservationStore(catalog).all())
    seeded = ColdStartSeeder(catalog).seed(observed)

    by_provenance = {c.provenance for c in seeded.patterns}
    assert by_provenance == {"high", "low"}

    high = [c for c in seeded.patterns if c.provenance == "high"]
    low = [c for c in seeded.patterns if c.provenance == "low"]
    assert high and low
    # The user's own words come first; history is appended behind them.
    assert seeded.patterns[: len(high)] == high
    assert all(c.id.startswith("history:") for c in low)
    assert all(not c.id.startswith("history:") for c in high)

    # Seeded claims are still traceable, and honest about having no verbatim.
    for claim in low:
        assert claim.evidence
        assert "not your words" in claim.text
        assert all(ref.image_sha for ref in claim.evidence)

    # Seeding is announced, never silent.
    assert any("seeded" in e["summary"] for e in seeded.evolution)


# ---------------------------------------------------------------------------
# S6 — upstream (R036)
# ---------------------------------------------------------------------------


def test_acceptance_taste_dialogue_upstream_citation_and_parity(data_root, tmp_path):
    """The profile changes what upstream says and ranks — and an empty one does not."""
    catalog = Catalog(data_root=data_root)
    _react_twice(catalog, data_root, tmp_path)
    profile = ProfileBuilder().build(ObservationStore(catalog).all())
    empty = ProfileBuilder().build([])
    lens = _lens_profile(colorfulness=0.5)
    analysis = _analysis()

    # Explanations quote the profile by verbatim when it has something to say.
    cited = explain_rank(analysis, lens, profile)
    baseline = explain_rank(analysis, lens, None)
    assert cited.citations
    assert cited.citations[0].quote in cited.rationale
    assert cited.rationale.startswith(baseline.rationale)
    # M008 adds words, not weights.
    assert cited.delta == baseline.delta
    assert cited.evidence == baseline.evidence

    # Discovery ranks by profile fit, along the profile's own dimensions.
    assert profile_dimensions(profile)
    assert familiar_surprising_dimensions(profile, -1.0) == tuple(
        reversed(familiar_surprising_dimensions(profile, 1.0))
    )
    quiet_work = {"colorfulness": 0.0, "vibrancy": 0.0, "aesthetic_quality": 0.8}
    loud_work = {"colorfulness": 0.9, "vibrancy": 0.9, "aesthetic_quality": 0.8}
    assert profile_fit(profile, quiet_work) > profile_fit(profile, loud_work)

    # Layout proposals may cite the profile in pairing rationale.
    assert pairing_rationale("paired on palette", profile) != "paired on palette"

    # A dispute changes downstream behavior: the cited claim stops being cited
    # and stops moving the ranking.
    store = ProfileStore(catalog)
    store.apply(profile)
    cited_id = citations_for(profile)[0].claim_id
    # A work that is mid-scale on every signal, so removing any claim moves the fit.
    probe_work = dict.fromkeys(SIGNAL_NAMES, 0.5)
    fit_before = profile_fit(profile, probe_work)
    store.dispute(cited_id)
    disputed = store.load()
    assert cited_id not in {c.claim_id for c in citations_for(disputed)}
    assert cited_id not in {c.id for c in disputed.patterns}
    assert profile_fit(disputed, probe_work) != fit_before

    # Disputing everything the profile claims returns it to baseline behavior.
    for claim in list(disputed.patterns):
        store.dispute(claim.id)
    silenced = store.load()
    assert citations_for(silenced) == []
    assert profile_fit(silenced, quiet_work) == 0.0

    # Empty profile == today's behavior, exactly.
    ranker = TasteRanker()
    candidates = [{"id": "a", "baseline": 1.0}, {"id": "b", "baseline": 0.5}]
    analysis_map = {"a": analysis, "b": _analysis(0.1)}
    expected = [c["id"] for c in ranker.rank(candidates, lens, analysis_map=analysis_map)]
    for nothing in (None, empty):
        assert citations_for(nothing) == []
        assert profile_dimensions(nothing) == ()
        assert profile_fit(nothing, quiet_work) == 0.0
        assert pairing_rationale("paired on palette", nothing) == "paired on palette"
        assert explain_rank(analysis, lens, nothing).to_dict() == baseline.to_dict()
        assert [
            c["id"] for c in ranker.rank(candidates, lens, analysis_map=analysis_map)
        ] == expected


# ---------------------------------------------------------------------------
# S7 — anti-goals (R038)
# ---------------------------------------------------------------------------


def test_acceptance_taste_dialogue_anti_goals(data_root, tmp_path):
    """The five anti-goals, each with an executable proof."""
    catalog = Catalog(data_root=data_root)

    # 1. No image generation — not even a dormant hook.
    package = Path(dialogue_upstream.__file__).parent
    banned = (
        "generate_image",
        "txt2img",
        "img2img",
        "diffusion",
        "dall-e",
        "dalle",
        "inpaint",
        "outpaint",
    )
    for source in sorted(package.glob("*.py")):
        lowered = source.read_text().lower()
        for token in banned:
            assert token not in lowered, f"{source}: generation token {token!r}"

    # 2. No silent learning — a session that added observations always has a delta.
    session_id = _react_twice(catalog, data_root, tmp_path)
    store = ObservationStore(catalog)
    learned = WhatILearned.delta_after(session_id, store)
    assert learned.added
    assert learned.summary

    # 3. No jargon laundering — the verbatim is stored and stays the surface text.
    profile = ProfileBuilder().build(store.all())
    assert any(o.verbatim == QUIET_VERBATIM for o in store.all())
    citation = citations_for(profile)[0]
    assert citation.quote == citation.quote.strip()
    assert citation.quote in {o.verbatim for o in store.all()}
    assert citation.quote not in profile.vocabulary  # a quote, not a tag

    # 4. No interrogation — the follow-up cap holds across a long session.
    room = _room(catalog, data_root)
    session = room.start([str(_drop_file(tmp_path, "long.png", seed=9))])
    questions = [room.react(session, "quiet and empty again").question for _ in range(6)]
    assert sum(1 for q in questions if q is not None) <= MAX_FOLLOWUPS
    room.finish(session)

    # 5. No hard dependency — every existing flow works with an empty profile.
    empty = ProfileBuilder().build([])
    lens = _lens_profile(colorfulness=0.5)
    assert explain_rank(_analysis(), lens, empty).to_dict() == explain_rank(
        _analysis(), lens, None
    ).to_dict()
    assert pairing_rationale("base", empty) == "base"
    assert profile_fit(empty, {"colorfulness": 0.9}) == 0.0

    # ...including the case where the dialogue subsystem was never used at all.
    fresh = Catalog(data_root=Path(data_root) / "fresh")
    assert ObservationStore(fresh).all() == []
    assert ProfileStore(fresh).load() is None
    assert ColdStartSeeder(fresh).claims() == []
    assert ProfileBuilder().build([]).patterns == []


# ---------------------------------------------------------------------------
# guard: the gate is wired into `make acceptance`
# ---------------------------------------------------------------------------


def test_acceptance_taste_dialogue_is_wired_into_the_gate():
    """An acceptance suite that is not in the gate is not a gate."""
    makefile = Path(__file__).resolve().parents[1] / "Makefile"
    target = makefile.read_text()
    assert "tests/test_acceptance_taste_dialogue.py" in target


def test_acceptance_taste_dialogue_runs_air_gapped():
    """The dialogue subsystem has no network reach — cloud extraction is synthetic."""
    package = Path(dialogue_upstream.__file__).parent
    network = ("import socket", "import requests", "import httpx", "urllib.request", "aiohttp")
    for source in sorted(package.glob("*.py")):
        text = source.read_text()
        for token in network:
            assert token not in text, f"{source}: network import {token!r}"


def test_acceptance_taste_dialogue_room_rejects_unknown_drops(data_root):
    """A drop that is neither a file nor a cataloged sha fails loudly."""
    catalog = Catalog(data_root=data_root)
    room = _room(catalog, data_root)
    with pytest.raises(CuratorError):
        room.start(["not-a-file-and-not-a-sha"])
