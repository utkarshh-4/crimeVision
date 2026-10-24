from __future__ import annotations

"""Unit tests for multimodal fusion engine, cross-modal corroboration, and conflict detection."""

import pytest

from modules.fusion import fuse_session


def test_multimodal_fusion_corroboration_and_anchors(monkeypatch):
    """Verify corroboration of compatible audio/video events into anchors with boosted confidence."""
    # Ensure deterministic rule-based narrative for hypotheses check
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: False)

    session = {
        "video": {
            "duration": 7.0,
            "frames": [
                {"t": 0.0, "people_count": 0, "weapon_count": 0, "detections": []},
                {"t": 1.0, "people_count": 1, "weapon_count": 0, "detections": [{"label": "Human"}]},
                {"t": 5.0, "people_count": 1, "weapon_count": 1, "detections": [{"label": "Weapon", "raw_label": "knife", "is_weapon": True}]},
            ],
            "events": [
                {
                    "id": "V1",
                    "category": "Arrival",
                    "description": "Person entered scene (people count: 1)",
                    "start_raw": 1.0,
                    "end_raw": 1.0,
                    "confidence": 0.8,
                },
                {
                    "id": "V2",
                    "category": "Threat",
                    "description": "Weapon appeared in scene",
                    "start_raw": 5.0,
                    "end_raw": 5.0,
                    "confidence": 0.75,
                },
            ],
        },
        "audio": {
            "duration": 7.0,
            "transcript": "Someone came in. Drop the weapon right now!",
            "events": [
                {
                    "id": "A1",
                    "category": "Arrival",
                    "description": "Person arrived at entrance",
                    "start_raw": 0.5,
                    "end_raw": 0.5,
                    "confidence": 0.7,
                },
                {
                    "id": "A2",
                    "category": "Threat",
                    "description": "Verbal threat: drop the weapon",
                    "start_raw": 4.0,
                    "end_raw": 6.0,
                    "confidence": 0.7,
                },
            ],
        },
    }

    fusion = fuse_session(session)

    timeline = fusion["timeline"]
    fused_events = [ev for ev in timeline if ev["modality"] == "fused"]

    # 1. Assert: two fused events
    assert len(fused_events) == 2

    # 2. Assert: both are anchors
    assert all(ev["is_anchor"] is True for ev in fused_events)
    fused_ids = {ev["id"] for ev in fused_events}
    assert fused_ids.issubset(set(fusion["anchor_events"]))

    # 3. Assert: confidence of Threat/Weapon fused event > each source (ca=0.7, cv=0.75)
    threat_fused = next(ev for ev in fused_events if ev["category"] == "Threat")
    # Expected: 1 - (1 - 0.7) * (1 - 0.75) = 1 - 0.075 = 0.925
    assert threat_fused["confidence"] > 0.70
    assert threat_fused["confidence"] > 0.75
    assert abs(threat_fused["confidence"] - 0.925) < 0.01

    # 4. Assert: second_index keys are ints-as-strings
    second_index = fusion["second_index"]
    assert len(second_index) > 0
    for key in second_index.keys():
        assert isinstance(key, str)
        assert key.isdigit()
        # Verify active events list
        assert isinstance(second_index[key], list)

    # 5. Assert: hypotheses sum to 1.0 with LLM unconfigured
    hypotheses = fusion["hypotheses"]
    assert len(hypotheses) == 3
    total_prob = sum(float(h["probability"]) for h in hypotheses)
    assert abs(total_prob - 1.0) < 1e-3

    # Summary exists and is <= 120 words
    summary = fusion["summary"]
    assert isinstance(summary, str) and len(summary) > 0
    assert len(summary.split()) <= 120


def test_fusion_conflict_weapon_discrepancy(monkeypatch):
    """Verify conflict detection when audio mentions knife but video has zero weapons."""
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: False)

    session = {
        "video": {
            "duration": 5.0,
            "frames": [
                {"t": 0.0, "people_count": 1, "weapon_count": 0, "detections": [{"label": "Human"}]},
                {"t": 1.0, "people_count": 1, "weapon_count": 0, "detections": [{"label": "Human"}]},
                {"t": 2.0, "people_count": 1, "weapon_count": 0, "detections": [{"label": "Human"}]},
            ],
            "events": [],
        },
        "audio": {
            "duration": 5.0,
            "transcript": "Look out, he has a knife!",
            "events": [
                {
                    "id": "A1",
                    "category": "Threat",
                    "description": "Witness shouted: he has a knife",
                    "start_raw": 1.0,
                    "end_raw": 3.0,
                    "confidence": 0.8,
                }
            ],
        },
    }

    fusion = fuse_session(session)

    # Assert: conflict produced
    assert len(fusion["conflicts"]) >= 1
    conflict = fusion["conflicts"][0]
    assert conflict["type"] == "weapon_presence_conflict"
    assert "A1" in conflict["involved_events"]

    # Assert: event flagged with conflict=True and confidence lowered
    # Original conf = 0.8. Uncorroborated penalty = 0.8 * 0.85 = 0.68.
    # Conflict penalty = 0.68 - 0.15 = 0.53.
    a1_event = next(ev for ev in fusion["timeline"] if ev["id"] == "A1")
    assert a1_event["conflict"] is True
    assert a1_event["confidence"] < 0.68
    assert abs(a1_event["confidence"] - 0.53) < 0.01


def test_fusion_hypotheses_mocked_llm(monkeypatch):
    """Verify normalization and summary handling when LLM returns non-normalized probabilities."""
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: True)

    mock_response = {
        "hypotheses": [
            {"title": "Armed Threat and Intrusion", "explanation": "Subject brandished weapon.", "probability": 0.8},
            {"title": "Heated Verbal Altercation", "explanation": "Dispute without injury.", "probability": 0.4},
            {"title": "Benign Gathering", "explanation": "No malicious intent.", "probability": 0.2},
        ],
        "summary": "Corroborated evidence confirms suspect arrival at 1.0s followed by weapon display at 5.0s.",
    }

    monkeypatch.setattr("modules.reasoning.chat_json", lambda **kwargs: mock_response)

    session = {
        "video": {
            "duration": 5.0,
            "frames": [{"t": 1.0, "people_count": 1, "weapon_count": 1, "detections": []}],
            "events": [{"id": "V1", "category": "Threat", "description": "Weapon appeared", "start_raw": 1.0, "end_raw": 1.0, "confidence": 0.8}],
        }
    }

    fusion = fuse_session(session)

    hypotheses = fusion["hypotheses"]
    assert len(hypotheses) == 3
    # Probabilities normalized to sum to 1.0
    total_prob = sum(float(h["probability"]) for h in hypotheses)
    assert abs(total_prob - 1.0) < 1e-3
    assert fusion["summary"] == mock_response["summary"]


def test_fusion_safe_handling_of_empty_or_none_session():
    """Verify fuse_session safely returns complete schema even on empty or None input."""
    for empty_input in ({}, None):
        res = fuse_session(empty_input)
        assert isinstance(res, dict)
        assert "timeline" in res
        assert "second_index" in res
        assert "anchor_events" in res
        assert "conflicts" in res
        assert "overall_confidence" in res
        assert "hypotheses" in res
        assert "summary" in res
        assert "warnings" in res
        assert len(res["hypotheses"]) == 3
        assert abs(sum(h["probability"] for h in res["hypotheses"]) - 1.0) < 1e-3
