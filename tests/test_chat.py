from __future__ import annotations

"""Unit tests for forensic chat intent classification, time/general querying, and session persistence."""

import json
import pytest

from modules.chat import answer_general, answer_time_query, chat, classify_intent
from modules.session_store import (
    list_sessions,
    load_session,
    new_session,
    save_session,
    update_progress,
    update_session,
)


def test_classify_intent():
    """Verify classification of time-based queries (MM:SS, sec, seconds) vs general inquiries."""
    # 1. MM:SS format
    res1 = classify_intent("What happened at 00:05?")
    assert res1["type"] == "time"
    assert res1["t"] == 5.0

    # 2. Number + sec format
    res2 = classify_intent("who was there at 12 sec")
    assert res2["type"] == "time"
    assert res2["t"] == 12.0

    # 3. General query without timestamp
    res3 = classify_intent("summarise the incident")
    assert res3["type"] == "general"
    assert res3["t"] is None

    # Additional patterns
    res4 = classify_intent("at t=7")
    assert res4["type"] == "time"
    assert res4["t"] == 7.0

    res5 = classify_intent("second 15 details")
    assert res5["type"] == "time"
    assert res5["t"] == 15.0


def test_session_store_lifecycle_and_validation(tmp_path, monkeypatch):
    """Verify session_store round trip, atomic write, update, progress, and invalid ID handling."""
    # Redirect session dir to temporary path
    monkeypatch.setattr("config.SESSION_DIR", tmp_path)

    # 1. Create new session with media
    session = new_session({"video": "test.mp4", "audio": "test.wav"})
    sid = session["id"]
    assert len(sid) == 32
    assert session["status"] == "queued"
    assert session["progress"]["video"] == "pending"
    assert session["progress"]["audio"] == "pending"
    assert session["progress"]["image"] == "skipped"

    # 2. Load session round trip
    loaded = load_session(sid)
    assert loaded is not None
    assert loaded["id"] == sid
    assert loaded["created_at"] == session["created_at"]

    # 3. Update session fields
    updated = update_session(sid, status="completed", crime_type="Armed Threat")
    assert updated is not None
    assert updated["status"] == "completed"
    assert updated["crime_type"] == "Armed Threat"

    # Re-verify persistence on disk
    reloaded = load_session(sid)
    assert reloaded["status"] == "completed"
    assert reloaded["crime_type"] == "Armed Threat"

    # 4. Update progress
    prog = update_progress(sid, "video", "done")
    assert prog is not None
    assert prog["progress"]["video"] == "done"

    # 5. List sessions
    all_sessions = list_sessions()
    assert len(all_sessions) >= 1
    assert any(s["session_id"] == sid for s in all_sessions)

    # 6. Invalid session IDs prevent path traversal and return None
    assert load_session("../../etc/passwd") is None
    assert load_session("short_id") is None
    assert load_session("not-a-valid-hex-uuid-string-too-long") is None
    assert load_session("00000000000000000000000000000000") is None  # Non-existent valid hex returns None

    # 7. Corrupt JSON returns None
    corrupt_file = tmp_path / f"{'1' * 32}.json"
    corrupt_file.write_text("corrupted non-json text", encoding="utf-8")
    assert load_session("1" * 32) is None


def test_answer_time_query_unconfigured_llm(monkeypatch):
    """Verify answer_time_query retrieves raw evidence facts and includes people count in fallback."""
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: False)

    session = {
        "video": {
            "duration": 10.0,
            "frames": [
                {
                    "t": 5.0,
                    "people_count": 2,
                    "weapon_count": 1,
                    "detections": [
                        {"label": "Human", "raw_label": "person", "confidence": 0.9},
                        {"label": "Human", "raw_label": "person", "confidence": 0.88},
                        {"label": "Weapon", "raw_label": "knife", "is_weapon": True, "confidence": 0.85},
                    ],
                    "caption": "Two people standing in hallway, one with a knife.",
                }
            ],
            "events": [
                {"id": "V1", "category": "Threat", "description": "Weapon appeared", "start_raw": 4.5, "end_raw": 5.5}
            ],
        }
    }

    res = answer_time_query(session, 5.0)

    # Assert: contains the people count
    assert "2 people" in res["answer"] or "2" in res["answer"]
    assert "knife" in res["answer"] or "weapon" in res["answer"]
    assert "5.0s" in res["sources"]
    assert "V1" in res["sources"]
    assert len(res["facts"]) >= 1


def test_answer_general_mocked_llm(monkeypatch):
    """Verify answer_general calls LLM with strict system prompt rule and temperature <= 0.2."""
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: True)

    captured_call = {}

    def mock_chat_text(messages, temperature=0.1, max_tokens=600):
        captured_call["messages"] = messages
        captured_call["temperature"] = temperature
        return "Based on evidence [V1] at [5.0s], a weapon was detected."

    monkeypatch.setattr("modules.reasoning.chat_text", mock_chat_text)

    session = {
        "fusion": {
            "summary": "Corroborated incident at 5.0s.",
            "hypotheses": [{"title": "Threat", "explanation": "Weapon display", "probability": 0.5}],
            "timeline": [{"id": "V1", "start_raw": 5.0, "end_raw": 5.0, "category": "Threat", "description": "Weapon appeared"}],
        }
    }

    res = answer_general(session, "What weapon was observed?", history=[])

    # Assert: chat_text was called
    assert "messages" in captured_call
    messages = captured_call["messages"]

    # Assert: system prompt contains the strict rule
    strict_rule = (
        "You are CrimeVision's forensic assistant. Answer ONLY from the EVIDENCE JSON below. "
        "If the answer is not in the evidence, reply exactly: 'The analysed evidence does not contain that information.' "
        "Do not speculate beyond the three hypotheses provided, and label them as hypotheses. "
        "Cite timestamps like [5.0s] when available. Be concise."
    )
    system_content = messages[0]["content"]
    assert strict_rule in system_content

    # Assert: temperature <= 0.2 was passed
    assert captured_call["temperature"] <= 0.2

    # Assert: answer and sources returned
    assert "weapon was detected" in res["answer"]
    assert "V1" in res["sources"] or "5.0s" in res["sources"]


def test_chat_interface_validation_and_history(tmp_path, monkeypatch):
    """Verify chat() interface handles invalid session, empty messages, and updates session history."""
    monkeypatch.setattr("config.SESSION_DIR", tmp_path)
    monkeypatch.setattr("modules.reasoning.is_configured", lambda: False)

    # 1. Invalid session raises KeyError
    with pytest.raises(KeyError, match="Session '.*' not found"):
        chat("0" * 32, "Hello")

    # 2. Empty message raises ValueError
    session = new_session({})
    sid = session["id"]

    with pytest.raises(ValueError, match="Message cannot be empty"):
        chat(sid, "")

    with pytest.raises(ValueError, match="Message cannot be empty"):
        chat(sid, "   ")

    # 3. Successful chat updates history on session
    resp = chat(sid, "Summarise the events")
    assert isinstance(resp, dict)
    assert "answer" in resp
    assert resp["intent"]["type"] == "general"

    # Verify session has updated chat history
    reloaded = load_session(sid)
    assert len(reloaded["chat_history"]) == 2
    assert reloaded["chat_history"][0]["role"] == "user"
    assert reloaded["chat_history"][1]["role"] == "assistant"
