"""Tests for src/curator/taste/dialogue/room.py (M008/S03).

Covers the Reaction Room flow end-to-end: session start with image resolution
(ephemeral third-party retention vs catalog-sha mapping), the react loop with
extraction + the hard two-follow-up cap (no nagging), the unavailable surface
(no provider / local slot / extraction down never keyword-degrades), the
deterministic ProbeGenerator, and the CLI taste drop/profile/dispute surface.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from curator import cli
from curator.catalog import Catalog
from curator.hashing import sha256_hex
from curator.taste.dialogue import (
    CloudExtractionProvider,
    ExtractionCapabilities,
    ExtractionProbe,
    ExtractionProvider,
    ExtractionUnavailableError,
    LocalExtractionSlot,
    ObservationStore,
    Polarity,
    SessionStore,
    create_observation,
)
from curator.taste.dialogue.room import (
    MAX_FOLLOWUPS,
    ProbeGenerator,
    ProbeQuestion,
    ReactionRoom,
    ReactionRoomUnavailableError,
    RoomTurn,
)

VERBATIM = "i love the quiet negative space"


def _image_bytes(
    width: int = 512, height: int = 384, seed: int = 0
) -> bytes:
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


def _write_image(path: Path, seed: int = 0) -> Path:
    path.write_bytes(_image_bytes(seed=seed))
    return path


class _DownStub(ExtractionProvider):
    """An 'enabled' provider whose extraction is down (runtime unreachable)."""

    def capabilities(self) -> ExtractionCapabilities:
        return ExtractionCapabilities(
            enabled=True,
            kind="cloud",
            supports_local=False,
            supports_cloud=True,
            disclosure_available=True,
        )

    def probe(self) -> ExtractionProbe:
        return ExtractionProbe(ok=False, kind="cloud", message="stub runtime down")

    def extract(self, observation):
        raise ExtractionUnavailableError("stub runtime down")


# ---------------------------------------------------------------------------
# ProbeQuestion + ProbeGenerator
# ---------------------------------------------------------------------------


def test_probe_question_round_trip() -> None:
    q = ProbeQuestion(
        text="Is it the emptiness on the left, or the symmetry?",
        target_attribute="symmetry",
    )
    rebuilt = ProbeQuestion.from_dict(json.loads(json.dumps(q.to_dict())))
    assert rebuilt == q
    assert ProbeQuestion.from_dict(q) is q
    opening = ProbeQuestion(text="What draws you to this image?", target_attribute=None)
    assert ProbeQuestion.from_dict(opening.to_dict()).target_attribute is None


def test_probe_generator_deterministic_and_bounded() -> None:
    gen = ProbeGenerator()
    base = dict(
        session_id="sess-1",
        verbatim=VERBATIM,
        attributes=["negative-space", "muted-palette", "quiet"],
        polarity=Polarity.LIKE,
        confidence=0.9,
    )
    first = gen.questions_for(create_observation(**base))
    second = gen.questions_for(create_observation(**base))
    assert [q.to_dict() for q in first] == [q.to_dict() for q in second]
    assert len(first) <= MAX_FOLLOWUPS
    assert all(q.text and q.target_attribute for q in first)
    assert first[0].text == "Is it the emptiness on the left, or the symmetry?"


def test_probe_generator_rich_attributes_yield_fewer() -> None:
    gen = ProbeGenerator()
    sparse = gen.questions_for(
        create_observation(
            session_id="s",
            verbatim="x",
            attributes=["negative-space", "muted-palette", "quiet"],
        )
    )
    rich = gen.questions_for(
        create_observation(
            session_id="s",
            verbatim="x",
            attributes=[
                "negative-space", "muted-palette", "quiet", "symmetry",
                "lone-subject", "texture", "warm-tones", "breathing-room",
                "geometric", "dense", "high-contrast", "organic",
                "repetition", "motion", "light",
            ],
        )
    )
    assert len(sparse) > 0
    assert len(rich) < len(sparse)


def test_probe_generator_empty_attributes_opening_question() -> None:
    gen = ProbeGenerator()
    questions = gen.questions_for(
        create_observation(session_id="s", verbatim="just a photo on the wall")
    )
    assert len(questions) == 1
    assert questions[0].text == "What draws you to this image?"
    assert questions[0].target_attribute is None
    assert "keep training" not in questions[0].text.lower()


# ---------------------------------------------------------------------------
# Reaction Room: full flow + follow-up cap
# ---------------------------------------------------------------------------


def test_full_flow(data_root, tmp_path) -> None:
    img = _write_image(tmp_path / "art.png")
    catalog = Catalog(data_root=data_root)
    room = ReactionRoom(catalog, CloudExtractionProvider(), data_root)
    try:
        session = room.start([str(img)])
        turn = room.react(session, VERBATIM)

        assert isinstance(turn, RoomTurn)
        assert turn.observation.session_id == session.id
        assert turn.observation.verbatim == VERBATIM
        assert "negative-space" in turn.observation.attributes
        assert "quiet" in turn.observation.attributes
        assert turn.observation.polarity is Polarity.LIKE
        assert turn.observation.confidence >= 0.8
        assert turn.question is not None
        assert turn.followups_asked == 1
        [ref] = turn.observation.images
        assert ref.ephemeral is True
        assert ref.thumb_path is not None and ref.thumb_path.exists()

        count = room.finish(session)
        assert count == 1

        store = ObservationStore(catalog)
        [rebuilt] = store.by_session(session.id)
        assert rebuilt.verbatim == VERBATIM
        assert rebuilt.attributes == turn.observation.attributes
        assert rebuilt.polarity is Polarity.LIKE
    finally:
        catalog.db.close()


def test_followup_cap_and_no_nagging(data_root, tmp_path) -> None:
    img = _write_image(tmp_path / "art.png")
    catalog = Catalog(data_root=data_root)
    room = ReactionRoom(catalog, CloudExtractionProvider(), data_root)
    try:
        session = room.start([str(img)])
        t1 = room.react(session, VERBATIM)
        t2 = room.react(session, "i love the symmetry here")
        t3 = room.react(session, "the texture is great")

        assert t1.question is not None
        assert t2.question is not None
        assert t3.question is None
        assert t1.followups_asked == 1
        assert t2.followups_asked == 2
        assert t3.followups_asked == 2

        asked = [q.text for q in (t1.question, t2.question) if q is not None]
        assert asked
        assert all("keep training" not in q.lower() for q in asked)
        assert all(q.endswith("?") for q in asked)

        store = ObservationStore(catalog)
        assert store.count() == 3
        assert len(store.by_session(session.id)) == 3
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# Unavailable surface: never keyword-degrades
# ---------------------------------------------------------------------------


def test_unavailable_no_provider_on_start(data_root, tmp_path) -> None:
    img = _write_image(tmp_path / "art.png")
    catalog = Catalog(data_root=data_root)
    room = ReactionRoom(catalog, None, data_root)
    try:
        with pytest.raises(ReactionRoomUnavailableError):
            room.start([str(img)])
    finally:
        catalog.db.close()


def test_unavailable_local_slot_on_start(data_root, tmp_path) -> None:
    img = _write_image(tmp_path / "art.png")
    catalog = Catalog(data_root=data_root)
    room = ReactionRoom(catalog, LocalExtractionSlot(), data_root)
    try:
        with pytest.raises(ReactionRoomUnavailableError):
            room.start([str(img)])
    finally:
        catalog.db.close()


def test_react_unavailable_no_provider_never_keyword_degrades(data_root) -> None:
    catalog = Catalog(data_root=data_root)
    room = ReactionRoom(catalog, None, data_root)
    session = SessionStore(catalog).new_session("reaction-room")
    try:
        with pytest.raises(ReactionRoomUnavailableError):
            room.react(session, VERBATIM)
        assert ObservationStore(catalog).count() == 0
    finally:
        catalog.db.close()


def test_react_unavailable_down_provider_no_observation(data_root, tmp_path) -> None:
    img = _write_image(tmp_path / "art.png")
    catalog = Catalog(data_root=data_root)
    room = ReactionRoom(catalog, _DownStub(), data_root)
    try:
        session = room.start([str(img)])  # enabled at capability level
        with pytest.raises(ReactionRoomUnavailableError):
            room.react(session, VERBATIM)
        # No observation and no keyword-derived attributes were recorded.
        assert ObservationStore(catalog).count() == 0
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# Image resolution: ephemeral third-party vs cataloged
# ---------------------------------------------------------------------------


def test_ephemeral_third_party_retained_catalog_sha_not(data_root, tmp_path) -> None:
    third = _write_image(tmp_path / "thirdparty.png", seed=1)
    cataloged = _write_image(tmp_path / "cataloged.png", seed=2)
    catalog = Catalog(data_root=data_root)
    digest = catalog.add_source("cli-local", str(cataloged), cataloged.read_bytes())
    room = ReactionRoom(catalog, CloudExtractionProvider(), data_root)
    try:
        session = room.start([str(third), digest])
        turn = room.react(session, VERBATIM)
        refs = turn.observation.images
        assert len(refs) == 2

        third_ref = next(r for r in refs if r.sha256 == sha256_hex(third.read_bytes()))
        assert third_ref.ephemeral is True
        assert third_ref.catalog_saved is False
        assert third_ref.thumb_path is not None and third_ref.thumb_path.exists()

        cat_ref = next(r for r in refs if r.sha256 == digest)
        assert cat_ref.ephemeral is False
        assert cat_ref.catalog_saved is True
        assert cat_ref.thumb_path is None

        thumbs = [p for p in data_root.rglob("*") if p.suffix == ".jpg"]
        assert len(thumbs) == 1
        assert thumbs[0] == third_ref.thumb_path
    finally:
        catalog.db.close()


def test_catalog_asset_path_maps_to_catalog_not_ephemeral(data_root, tmp_path) -> None:
    cataloged = _write_image(tmp_path / "cataloged.png", seed=3)
    catalog = Catalog(data_root=data_root)
    catalog.add_source("cli-local", str(cataloged), cataloged.read_bytes())
    room = ReactionRoom(catalog, CloudExtractionProvider(), data_root)
    try:
        session = room.start([str(cataloged)])
        turn = room.react(session, VERBATIM)
        [ref] = turn.observation.images
        assert ref.ephemeral is False
        assert ref.catalog_saved is True
        assert ref.thumb_path is None
    finally:
        catalog.db.close()


def test_unknown_image_reference_raises(data_root) -> None:
    catalog = Catalog(data_root=data_root)
    room = ReactionRoom(catalog, CloudExtractionProvider(), data_root)
    try:
        with pytest.raises(Exception):
            room.start(["not-a-file-not-a-sha"])
    finally:
        catalog.db.close()


# ---------------------------------------------------------------------------
# CLI: taste drop / profile / dispute
# ---------------------------------------------------------------------------


def test_cli_drop_with_note_records_observation(data_root, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CURATOR_TASTE_EXTRACTION_ENABLED", "1")
    img = _write_image(tmp_path / "drop.png")
    rc = cli.main(
        ["taste", "drop", str(img), "--note", "i love the quiet negative space"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "recorded" in out.lower()
    assert "1 observation" in out

    catalog = Catalog(data_root=data_root)
    try:
        store = ObservationStore(catalog)
        assert store.count() == 1
        [obs] = store.all()
        assert obs.verbatim == "i love the quiet negative space"
        assert "negative-space" in obs.attributes
        assert "quiet" in obs.attributes
        assert obs.polarity is Polarity.LIKE
    finally:
        catalog.db.close()


def test_cli_drop_prints_what_i_learned_delta(data_root, tmp_path, capsys, monkeypatch):
    """No silent learning (R038/R035): a session always reports what it added."""
    monkeypatch.setenv("CURATOR_TASTE_EXTRACTION_ENABLED", "1")
    img = _write_image(tmp_path / "drop.png")

    assert cli.main(["taste", "drop", str(img), "--note", VERBATIM]) == 0

    out = capsys.readouterr().out
    assert "Learned from 1 new reactions" in out
    assert "Nothing enters the profile without appearing here." in out
    assert "vocabulary" in out or "new pattern" in out


def test_cli_drop_asks_the_reactions_own_follow_up(data_root, tmp_path, capsys, monkeypatch):
    """The probe generated from the reaction is shown, not discarded."""
    monkeypatch.setenv("CURATOR_TASTE_EXTRACTION_ENABLED", "1")
    img = _write_image(tmp_path / "drop.png")

    assert cli.main(["taste", "drop", str(img), "--note", VERBATIM]) == 0

    out = capsys.readouterr().out
    assert "?" in out
    generator = ProbeGenerator()
    expected = generator.questions_for(
        create_observation(
            session_id="s",
            verbatim=VERBATIM,
            attributes=["negative-space", "muted-palette", "quiet"],
        )
    )
    assert expected[0].text in out


def test_cli_drop_without_save_keeps_third_party_ephemeral(
    data_root, tmp_path, capsys, monkeypatch
):
    """R034: retention is thumb + hash only until the user explicitly saves."""
    monkeypatch.setenv("CURATOR_TASTE_EXTRACTION_ENABLED", "1")
    img = _write_image(tmp_path / "drop.png")
    sha = sha256_hex(img.read_bytes())

    assert cli.main(["taste", "drop", str(img), "--note", VERBATIM]) == 0

    catalog = Catalog(data_root=data_root)
    try:
        assert catalog.get_by_hash(sha) == []
    finally:
        catalog.db.close()


def test_cli_drop_with_save_promotes_to_catalog(data_root, tmp_path, capsys, monkeypatch):
    """R034: --save is the explicit user choice that stores full resolution."""
    monkeypatch.setenv("CURATOR_TASTE_EXTRACTION_ENABLED", "1")
    img = _write_image(tmp_path / "drop.png")
    sha = sha256_hex(img.read_bytes())

    assert cli.main(["taste", "drop", str(img), "--note", VERBATIM, "--save"]) == 0

    out = capsys.readouterr().out
    assert "saved" in out.lower()
    assert sha[:12] in out

    catalog = Catalog(data_root=data_root)
    try:
        assert catalog.get_by_hash(sha)
    finally:
        catalog.db.close()


def test_cli_drop_save_skips_already_cataloged_images(
    data_root, tmp_path, capsys, monkeypatch
):
    """A drop that is already in the catalog is not re-saved."""
    monkeypatch.setenv("CURATOR_TASTE_EXTRACTION_ENABLED", "1")
    img = _write_image(tmp_path / "drop.png")
    catalog = Catalog(data_root=data_root)
    try:
        catalog.add_source("local", "asset-1", img.read_bytes())
    finally:
        catalog.db.close()

    assert cli.main(["taste", "drop", str(img), "--note", VERBATIM, "--save"]) == 0
    assert "saved" not in capsys.readouterr().out.lower()


def test_cli_drop_without_note_previews_questions(data_root, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("CURATOR_TASTE_EXTRACTION_ENABLED", "1")
    img = _write_image(tmp_path / "drop.png")
    rc = cli.main(["taste", "drop", str(img)])
    assert rc == 3
    out = capsys.readouterr().out
    assert "probing questions" in out.lower()
    assert "What draws you to this image?" in out


def test_cli_drop_unavailable_without_provider(data_root, tmp_path, capsys):
    img = _write_image(tmp_path / "drop.png")
    rc = cli.main(["taste", "drop", str(img), "--note", "i love it"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "error" in err.lower()


def test_cli_profile_renders_empty_profile(data_root, capsys):
    """An empty profile still renders the document (no hard dependency, R038)."""
    rc = cli.main(["taste", "profile"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Taste Profile" in out
    for section in ("Vocabulary:", "Patterns:", "Tensions:", "Evolution:"):
        assert section in out


def test_cli_dispute_unknown_claim_reports_no_change(data_root, capsys):
    """Disputing a claim the profile never made changes nothing (exit 3)."""
    rc = cli.main(["taste", "dispute", "claim-123"])
    assert rc == 3
    out = capsys.readouterr().out
    assert "claim-123" in out
