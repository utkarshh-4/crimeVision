from __future__ import annotations

"""Unit and integration tests for Flask API endpoints, validations, and synchronous pipeline execution."""

import io
import re
from unittest.mock import patch

import pytest

import config
from modules import pipeline, session_store
from tests.schema_utils import assert_session_schema

_SID_RE = re.compile(r"^[a-f0-9]{32}$")


def _get_mock_image_result():
    """Return a schema-valid mock image analysis result."""
    return {
        "width": 64,
        "height": 64,
        "people_count": 1,
        "weapon_count": 0,
        "detections": [
            {
                "label": "Human",
                "raw_label": "person",
                "confidence": 0.92,
                "box": [10.0, 10.0, 50.0, 50.0],
                "is_weapon": False,
            }
        ],
        "scene": {"label": "hall", "confidence": 0.85, "top_k": []},
        "firearm_check": {"firearm_likely": False, "confidence": 0.05},
        "caption": "a person in the hallway",
        "hypotheses": [
            {"title": "Civilian Presence", "explanation": "Subject in hallway.", "probability": 0.7},
            {"title": "Unidentified Subject", "explanation": "Person observed.", "probability": 0.2},
            {"title": "Authorized Personnel", "explanation": "Routine staff entry.", "probability": 0.1},
        ],
    }


def test_api_health(client):
    """Verify /api/health endpoint structure and status."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert "device" in data
    assert "groq_configured" in data


def test_api_analyze_validation(client):
    """Verify /api/analyze input validation for missing media, invalid languages, and bad extensions."""
    # 1. No files provided -> 400
    resp_no_files = client.post("/api/analyze", data={})
    assert resp_no_files.status_code == 400
    assert "at least one media file" in resp_no_files.get_json()["error"].lower()

    # 2. Unsupported language -> 400
    resp_lang = client.post(
        "/api/analyze",
        data={
            "image": (io.BytesIO(b"fake_image"), "test.jpg"),
            "language": "unsupported_lang",
        },
        content_type="multipart/form-data",
    )
    assert resp_lang.status_code == 400
    assert "unsupported language" in resp_lang.get_json()["error"].lower()

    # 3. Bad extension -> 400
    resp_bad_ext = client.post(
        "/api/analyze",
        data={
            "image": (io.BytesIO(b"fake_image"), "malicious.exe"),
        },
        content_type="multipart/form-data",
    )
    assert resp_bad_ext.status_code == 400
    assert "invalid image extension" in resp_bad_ext.get_json()["error"].lower()


def test_api_analyze_oversized_413(client):
    """Verify /api/analyze returns 413 when uploaded payload exceeds MAX_CONTENT_LENGTH."""
    # Temporarily set max content length small (100 bytes) on the test app
    client.application.config["MAX_CONTENT_LENGTH"] = 100

    try:
        large_data = b"x" * 500
        resp = client.post(
            "/api/analyze",
            data={
                "image": (io.BytesIO(large_data), "large.jpg"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 413
        assert "exceeds maximum upload limit" in resp.get_json()["error"].lower()
    finally:
        client.application.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024


def test_api_analyze_valid_upload_and_synchronous_poll(client, make_synthetic_image, monkeypatch):
    """Verify valid upload returns 202 with 32-hex ID, and polling session yields complete fusion block."""
    # Monkeypatch analyze_image to return mock result quickly
    monkeypatch.setattr("modules.pipeline.analyze_image", lambda path: _get_mock_image_result())

    # Monkeypatch start_pipeline to run synchronously
    def _sync_start_pipeline(sid, language="auto"):
        pipeline.run_pipeline(sid, language=language)

    monkeypatch.setattr("routes.api.pipeline.start_pipeline", _sync_start_pipeline)

    img_path = make_synthetic_image("test_sample.jpg")
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    resp = client.post(
        "/api/analyze",
        data={
            "image": (io.BytesIO(img_bytes), "test_sample.jpg"),
            "language": "en",
        },
        content_type="multipart/form-data",
    )

    assert resp.status_code == 202
    data = resp.get_json()
    assert "session_id" in data
    sid = data["session_id"]
    assert _SID_RE.match(sid)

    # Poll status
    status_resp = client.get(f"/api/status/{sid}")
    assert status_resp.status_code == 200
    status_data = status_resp.get_json()
    assert status_data["status"] == "done"
    assert status_data["progress"]["image"] == "done"
    assert status_data["progress"]["fusion"] == "done"

    # Get session details
    sess_resp = client.get(f"/api/session/{sid}")
    assert sess_resp.status_code == 200
    session_data = sess_resp.get_json()

    assert session_data["status"] == "done"
    assert session_data["image"] is not None
    assert session_data["fusion"] is not None
    assert "timeline" in session_data["fusion"]
    assert "hypotheses" in session_data["fusion"]

    # Canonical schema assertion
    assert_session_schema(session_data)


def test_api_chat_flow(client, tmp_data_dirs):
    """Verify /api/chat error handling (404 on missing, 400 on empty) and valid 200 response with answer."""
    # 1. Unknown session -> 404
    resp_404 = client.post("/api/chat", json={"session_id": "0" * 32, "message": "hello"})
    assert resp_404.status_code == 404

    # 2. Empty message -> 400
    sess = session_store.new_session()
    sid = sess["id"]
    resp_empty = client.post("/api/chat", json={"session_id": sid, "message": "   "})
    assert resp_empty.status_code == 400

    # 3. Valid chat query on initialized session
    session_store.update_session(
        sid,
        status="done",
        index={"events": [], "transcript_segments": [], "frames": [], "summary": "Forensic test scene"},
        fusion={"timeline": [], "conflicts": [], "hypotheses": [], "summary": "Forensic test scene", "overall_confidence": 0.9},
    )

    resp_valid = client.post("/api/chat", json={"session_id": sid, "message": "What happened?"})
    assert resp_valid.status_code == 200
    chat_data = resp_valid.get_json()
    assert "answer" in chat_data
    assert isinstance(chat_data["answer"], str)
    assert len(chat_data["answer"]) > 0


def test_api_media_path_traversal(client):
    """Verify /api/media rejects path traversal attempts with 400 or 404."""
    # Kind containing path traversal -> 400
    resp1 = client.get(f"/api/media/{'0'*32}/../x")
    assert resp1.status_code in (400, 404)

    resp2 = client.get(f"/api/media/{'0'*32}/..%2fimage")
    assert resp2.status_code in (400, 404)

    # Path traversal directly on the media route
    resp3 = client.get("/api/media/../x")
    assert resp3.status_code == 404
