from __future__ import annotations

"""Unit tests for audio analysis, Whisper routing, and acoustic fact extraction."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from modules.audio_analysis import _is_likely_speech, load_audio, transcribe_audio
from modules.llm import LLMError
from modules.reasoning import (
    _rule_based_audio_facts,
    extract_audio_events,
    extract_audio_facts,
)


def test_load_audio_validation(tmp_path):
    """Test load_audio error cases for missing and short files."""
    missing = tmp_path / "missing.wav"
    with pytest.raises(ValueError, match="does not exist"):
        load_audio(missing)


def test_is_likely_speech():
    """Verify that pure tones and silence are identified as non-speech."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)

    assert not _is_likely_speech(sine, sr=sr)
    assert not _is_likely_speech(silence, sr=sr)


def test_rule_based_audio_facts_empty_transcript():
    """Verify deterministic fallback for silent/inaudible audio."""
    facts = _rule_based_audio_facts("")
    assert facts["who"] == "Unknown"
    assert facts["location_inference"]["label"] == "Unknown"
    assert len(facts["hypotheses"]) == 3
    prob_sum = round(sum(h["probability"] for h in facts["hypotheses"]), 3)
    assert prob_sum == 1.0
    assert any("Inaudible" in h["title"] for h in facts["hypotheses"])


def test_extract_audio_facts_mock_llm(monkeypatch):
    """Verify LLM extraction and validation for audio transcript facts."""
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: True)

    mock_llm_output = {
        "who": "Security Guard",
        "where": "Office entrance",
        "when": "Tonight",
        "suspect_description": "Unknown",
        "location_inference": {
            "label": "office",
            "confidence": 0.85,
            "evidence": "office entrance mentioned",
        },
        "hypotheses": [
            {"title": "Intrusion Attempt", "explanation": "Breach reported at entrance.", "probability": 0.6},
            {"title": "False Alarm", "explanation": "Staff badge misread.", "probability": 0.3},
            {"title": "Routine Patrol", "explanation": "Standard check-in.", "probability": 0.1},
        ],
    }

    with patch("modules.reasoning.chat_json", return_value=mock_llm_output):
        facts = extract_audio_facts(
            transcript="Intruder at the office entrance, requesting backup.",
            segments=[{"start_raw": 0.0, "end_raw": 3.0, "text": "Intruder at office"}],
        )

    assert facts["who"] == "Security Guard"
    assert facts["location_inference"]["label"] == "office"
    assert facts["location_inference"]["confidence"] == 0.85
    assert len(facts["hypotheses"]) == 3
    prob_sum = round(sum(h["probability"] for h in facts["hypotheses"]), 3)
    assert prob_sum == 1.0


def test_extract_audio_facts_fallback_on_error(monkeypatch):
    """Verify graceful fallback when LLM encounters an exception during audio analysis."""
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: True)

    with patch("modules.reasoning.chat_json", side_effect=LLMError("Groq 500 error")):
        facts = extract_audio_facts(
            transcript="Help, someone is trying to break in!",
            segments=[{"start_raw": 0.0, "end_raw": 2.0, "text": "Help"}],
        )

    assert facts["who"] == "Unknown"
    assert len(facts["hypotheses"]) == 3
    prob_sum = round(sum(h["probability"] for h in facts["hypotheses"]), 3)
    assert prob_sum == 1.0
    assert any("rule-based fallback" in h["explanation"] for h in facts["hypotheses"])


def test_extract_audio_events_mock_llm_overlapping_merge(mocker):
    """Verify overlapping audio events of the same category merge correctly."""
    mocker.patch("modules.reasoning.is_configured", return_value=True)

    mock_llm_response = {
        "events": [
            {
                "segment_indices": [0],
                "category": "Aggression",
                "description": "Suspect pushed victim",
                "who": "Unknown",
                "where": "Unknown",
                "confidence": 0.8,
            },
            {
                "segment_indices": [1],
                "category": "Aggression",
                "description": "Suspect physically assaulted victim near doorway",
                "who": "Suspect",
                "where": "Doorway",
                "confidence": 0.95,
            },
        ]
    }
    mocker.patch("modules.reasoning.chat_json", return_value=mock_llm_response)

    segments = [
        {"start_raw": 1.0, "end_raw": 3.0, "text": "He pushed me"},
        {"start_raw": 2.5, "end_raw": 5.0, "text": "He assaulted me at the doorway"},
    ]

    events = extract_audio_events(segments)

    # Must merge into exactly 1 event because they overlap and share category
    assert len(events) == 1
    ev = events[0]
    assert ev["id"] == "A1"
    assert ev["category"] == "Aggression"
    assert ev["start_raw"] == 1.0
    assert ev["end_raw"] == 5.0
    assert ev["start_raw"] <= ev["end_raw"]
    assert ev["confidence"] == 0.95
    # Must keep the longer description
    assert ev["description"] == "Suspect physically assaulted victim near doorway"


def test_extract_audio_events_fallback_keyword(mocker):
    """Verify keyword-based fallback event extraction when LLM is unconfigured."""
    mocker.patch("modules.reasoning.is_configured", return_value=False)

    segments = [
        {"start_raw": 1.5, "end_raw": 3.5, "text": "Watch out, he had a knife!"},
    ]

    events = extract_audio_events(segments)

    assert len(events) >= 1
    ev = events[0]
    assert ev["id"].startswith("A")
    assert ev["start_raw"] <= ev["end_raw"]
    assert ev["confidence"] == 0.4
    assert "knife" in ev["description"].lower()

