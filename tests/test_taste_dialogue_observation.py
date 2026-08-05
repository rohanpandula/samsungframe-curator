"""Tests for src/curator/taste/dialogue/observation (M008/S01 T1).

Covers the full to_dict/from_dict round-trip of a fully-populated
TasteObservation, byte-exact verbatim preservation, polarity coercion (and
rejection of invalid polarity), attribute/confidence/image preservation, JSON
serializability, the ImageRef round-trip including ephemeral/catalog_saved
flags, and the create_observation helper defaults.
"""

from __future__ import annotations

import json

import pytest

from curator.errors import CuratorError
from curator.taste.dialogue import (
    ImageRef,
    ObservationError,
    Polarity,
    TasteObservation,
    create_observation,
)


def _observation() -> TasteObservation:
    return TasteObservation(
        session_id="sess-1",
        verbatim="the fog makes it feel private\n  (quiet)",
        attributes=["negative-space", "muted-palette", "lone-subject"],
        polarity=Polarity.LIKE,
        confidence=0.87,
        images=[
            ImageRef(sha256="a" * 64),
            ImageRef(
                sha256="b" * 64,
                thumb_path="/tmp/thumbs/b.jpg",
                ephemeral=True,
            ),
        ],
        id=42,
        created_at="2026-08-04T10:00:00Z",
    )


def test_full_round_trip():
    obs = _observation()
    rebuilt = TasteObservation.from_dict(json.loads(json.dumps(obs.to_dict())))
    assert rebuilt == obs
    assert rebuilt.id == 42
    assert rebuilt.session_id == "sess-1"
    assert rebuilt.polarity is Polarity.LIKE
    assert rebuilt.confidence == 0.87
    assert rebuilt.created_at == "2026-08-04T10:00:00Z"
    assert rebuilt.images == obs.images


def test_verbatim_preserved_byte_exact():
    verbatim = "  The fog makes it feel private...\n\t(quiet, no caps)  "
    obs = TasteObservation(
        session_id="s",
        verbatim=verbatim,
        attributes=[],
        polarity=Polarity.LIKE,
        confidence=0.5,
        images=[],
    )
    rebuilt = TasteObservation.from_dict(json.loads(json.dumps(obs.to_dict())))
    assert rebuilt.verbatim == verbatim
    assert rebuilt.verbatim.encode("utf-8") == verbatim.encode("utf-8")


def test_polarity_coerced_from_string():
    rebuilt = TasteObservation.from_dict(
        {
            "session_id": "s",
            "verbatim": "hate it",
            "attributes": [],
            "polarity": "dislike",
            "confidence": 0.9,
            "images": [],
        }
    )
    assert rebuilt.polarity is Polarity.DISLIKE
    assert rebuilt.to_dict()["polarity"] == "dislike"


def test_polarity_coerced_from_enum_instance():
    obs = _observation()
    rebuilt = TasteObservation.from_dict(obs)
    assert rebuilt == obs


def test_invalid_polarity_raises_clear_error():
    with pytest.raises(ObservationError) as excinfo:
        TasteObservation.from_dict(
            {
                "session_id": "s",
                "verbatim": "??",
                "attributes": [],
                "polarity": "meh",
                "confidence": 0.5,
                "images": [],
            }
        )
    message = str(excinfo.value)
    assert "polarity" in message
    assert "meh" in message
    assert isinstance(excinfo.value, CuratorError)


def test_attributes_confidence_images_preserved():
    obs = _observation()
    d = obs.to_dict()
    assert d["attributes"] == ["negative-space", "muted-palette", "lone-subject"]
    assert d["confidence"] == 0.87
    assert d["images"][0]["sha256"] == "a" * 64
    rebuilt = TasteObservation.from_dict(d)
    assert rebuilt.attributes == obs.attributes
    assert rebuilt.confidence == obs.confidence
    assert rebuilt.images == obs.images


def test_json_serializable():
    obs = _observation()
    text = json.dumps(obs.to_dict())
    assert isinstance(text, str)
    rebuilt = TasteObservation.from_dict(json.loads(text))
    assert rebuilt == obs


def test_imageref_round_trip_flags():
    ref = ImageRef(
        sha256="c" * 64,
        thumb_path=None,
        ephemeral=True,
        catalog_saved=True,
    )
    rebuilt = ImageRef.from_dict(json.loads(json.dumps(ref.to_dict())))
    assert rebuilt == ref
    assert rebuilt.ephemeral is True
    assert rebuilt.catalog_saved is True
    assert rebuilt.thumb_path is None


def test_imageref_defaults():
    ref = ImageRef(sha256="d" * 64)
    assert ref.ephemeral is False
    assert ref.catalog_saved is False
    assert ref.thumb_path is None
    rebuilt = ImageRef.from_dict(ref.to_dict())
    assert rebuilt == ref
    assert ImageRef.from_dict(rebuilt) == ref


def test_create_observation_defaults():
    obs = create_observation(session_id="s2", verbatim="brilliant")
    assert isinstance(obs, TasteObservation)
    assert obs.polarity is Polarity.LIKE
    assert obs.confidence == 0.5
    assert obs.attributes == []
    assert obs.images == []
    assert obs.id is None
    assert obs.created_at == ""


def test_create_observation_accepts_iterables():
    obs = create_observation(
        session_id="s3",
        verbatim="both are noisy",
        attributes=("noisy", "street"),
        images=(ImageRef(sha256="e" * 64),),
        polarity=Polarity.CONFLICTED,
        confidence=0.4,
        created_at="2026-08-04T11:00:00Z",
    )
    assert obs.attributes == ["noisy", "street"]
    assert [ref.sha256 for ref in obs.images] == ["e" * 64]
    assert obs.polarity is Polarity.CONFLICTED
    rebuilt = TasteObservation.from_dict(json.loads(json.dumps(obs.to_dict())))
    assert rebuilt == obs


def test_confidence_out_of_range_raises():
    with pytest.raises(ObservationError):
        TasteObservation(
            session_id="s",
            verbatim="nope",
            attributes=[],
            polarity=Polarity.DISLIKE,
            confidence=1.5,
            images=[],
        )
