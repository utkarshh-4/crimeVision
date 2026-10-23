from __future__ import annotations

"""Unit tests for image analysis, LLM client, and forensic reasoning."""

import pytest
from unittest.mock import MagicMock, patch

from modules.llm import LLMError, chat_json
from modules.reasoning import (
    _compute_relative_position,
    _rule_based_image_hypotheses,
    generate_image_hypotheses,
)


def test_compute_relative_position():
    """Verify relative spatial positioning calculations."""
    # Top-left box
    pos = _compute_relative_position([10.0, 10.0, 50.0, 80.0], width=1000, height=1000)
    assert pos == "left, far"

    # Bottom-right box
    pos = _compute_relative_position([700.0, 700.0, 950.0, 900.0], width=1000, height=1000)
    assert pos == "right, near"

    # Center box
    pos = _compute_relative_position([400.0, 400.0, 600.0, 500.0], width=1000, height=1000)
    assert pos == "center, mid-ground"


def test_rule_based_image_hypotheses():
    """Verify deterministic fallback hypotheses generation and probability sums."""
    sample = {
        "people_count": 3,
        "weapon_count": 1,
        "scene": {"label": "road"},
        "caption": "people on street",
    }
    hypotheses = _rule_based_image_hypotheses(sample)
    assert len(hypotheses) == 3
    prob_sum = round(sum(h["probability"] for h in hypotheses), 3)
    assert prob_sum == 1.0
    assert all(bool(h["title"]) for h in hypotheses)
    assert all("rule-based fallback" in h["explanation"] for h in hypotheses)


def test_generate_image_hypotheses_mock_llm(monkeypatch):
    """Verify LLM hypothesis processing, validation, and probability normalization."""
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: True)

    mock_response = {
        "hypotheses": [
            {"title": "Armed Standoff", "explanation": "Subjects holding weapons in street.", "probability": 0.6},
            {"title": "Lawful Intervention", "explanation": "Security forces controlling area.", "probability": 0.3},
            {"title": "Civilian Disturbance", "explanation": "Unplanned confrontation in roadway.", "probability": 0.1},
        ]
    }

    with patch("modules.reasoning.chat_json", return_value=mock_response):
        hypotheses = generate_image_hypotheses({
            "width": 800,
            "height": 600,
            "people_count": 2,
            "weapon_count": 1,
            "detections": [],
            "scene": {"label": "road"},
            "caption": "two people on road",
        })

    assert len(hypotheses) == 3
    prob_sum = round(sum(h["probability"] for h in hypotheses), 3)
    assert prob_sum == 1.0
    assert hypotheses[0]["title"] == "Armed Standoff"
    assert hypotheses[0]["probability"] == 0.6


def test_generate_image_hypotheses_fallback_on_llm_error(monkeypatch):
    """Verify automatic fallback when LLM encounters an exception."""
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: True)

    with patch("modules.reasoning.chat_json", side_effect=LLMError("API rate limit")):
        hypotheses = generate_image_hypotheses({
            "width": 800,
            "height": 600,
            "people_count": 1,
            "weapon_count": 0,
            "detections": [],
            "scene": {"label": "office"},
            "caption": "person in office",
        })

    assert len(hypotheses) == 3
    prob_sum = round(sum(h["probability"] for h in hypotheses), 3)
    assert prob_sum == 1.0
    assert any("rule-based fallback" in h["explanation"] for h in hypotheses)


def test_chat_json_parser_strips_markdown():
    """Verify chat_json correctly strips markdown fences and extracts inner JSON."""
    raw_json = '```json\n{"status": "ok", "count": 3}\n```'

    with patch("modules.llm.get_groq_client") as mock_get_client:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = raw_json
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_get_client.return_value = mock_client

        parsed = chat_json(messages=[{"role": "user", "content": "test"}])
        assert parsed == {"status": "ok", "count": 3}
