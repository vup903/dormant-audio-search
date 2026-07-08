"""Hermetic tests: no network, no models. The schema is the contract."""
import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schema" / "record.schema.json").read_text(encoding="utf-8"))
VOCAB = json.loads((ROOT / "vocab.json").read_text(encoding="utf-8"))


def make_record(**overrides):
    record = {
        "id": "test-item",
        "title": "Test Broadcast",
        "program": "Test Program",
        "date": "1944-06-06",
        "decade": "1940s",
        "source_url": "https://archive.org/details/test-item",
        "rights": "public domain",
        "audio_file": "test-item.mp3",
        "duration_s": 300.0,
        "asr": {"model": "faster-whisper-base", "confidence": 0.85, "ai_generated": True},
        "transcript": "This is a test.",
        "segments": [{"start": 0.0, "end": 3.2, "text": "This is a test.", "confidence": 0.85}],
        "tags": [{"term": "war", "score": 0.61, "ai_generated": True}],
    }
    record.update(overrides)
    return record


def test_schema_tag_enum_matches_vocab():
    enum = SCHEMA["properties"]["tags"]["items"]["properties"]["term"]["enum"]
    assert sorted(enum) == sorted(VOCAB["terms"].keys())


def test_valid_record_passes():
    jsonschema.validate(make_record(), SCHEMA)


def test_out_of_vocab_tag_rejected():
    bad = make_record(tags=[{"term": "elections", "score": 0.9, "ai_generated": True}])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, SCHEMA)


def test_tag_must_be_marked_ai_generated():
    bad = make_record(tags=[{"term": "war", "score": 0.9, "ai_generated": False}])
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, SCHEMA)


def test_source_url_required():
    bad = make_record()
    del bad["source_url"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, SCHEMA)
