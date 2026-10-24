from __future__ import annotations

"""End-to-end integration tests for the forensic analysis orchestration pipeline."""

from unittest.mock import patch
import pytest

from modules import pipeline, session_store
from tests.schema_utils import assert_session_schema


def _mock_image_payload():
    return {
        "width": 64,
        "height": 64,
        "people_count": 1,
        "weapon_count": 0,
        "detections": [
            {
                "label": "Human",
                "raw_label": "person",
                "confidence": 0.95,
                "box": [10.0, 10.0, 40.0, 50.0],
                "is_weapon": False,
            }
        ],
        "scene": {"label": "office", "confidence": 0.88, "top_k": []},
        "firearm_check": {"firearm_likely": False, "confidence": 0.05},
        "caption": "a person in an office",
        "hypotheses": [
            {"title": "Office Worker", "explanation": "Subject at desk.", "probability": 0.7},
            {"title": "Visitor", "explanation": "Unidentified visitor.", "probability": 0.2},
            {"title": "Security", "explanation": "Security patrol.", "probability": 0.1},
        ],
    }


def _mock_audio_payload():
    return {
        "transcript": "Drop the weapon now! Hands in the air.",
        "language": "en",
        "segments": [
            {"id": "S1", "start_raw": 1.0, "end_raw": 3.0, "text": "Drop the weapon now!"},
            {"id": "S2", "start_raw": 3.2, "end_raw": 5.0, "text": "Hands in the air."},
        ],
        "events": [
            {
                "id": "A1",
                "start_raw": 1.0,
                "end_raw": 3.0,
                "category": "Threat",
                "description": "Verbal order to drop weapon",
                "who": "Guard",
                "where": "Office",
                "confidence": 0.95,
            }
        ],
        "facts": {
            "who": "Guard",
            "where": "Office",
            "when": "Now",
            "suspect_description": "Unknown",
            "location_inference": {"label": "office", "confidence": 0.9, "evidence": "office mention"},
            "hypotheses": [
                {"title": "Armed Intervention", "explanation": "Guard issuing verbal command", "probability": 0.8},
                {"title": "Drill Simulation", "explanation": "Practice scenario", "probability": 0.15},
                {"title": "Misinterpretation", "explanation": "Non-threatening audio", "probability": 0.05},
            ],
        },
    }


def _mock_video_payload():
    return {
        "fps": 5.0,
        "frame_count": 25,
        "duration": 5.0,
        "frames": [
            {
                "t": 1.0,
                "people_count": 2,
                "weapon_count": 1,
                "detections": [
                    {"label": "Human", "raw_label": "person", "confidence": 0.9, "box": [10.0, 10.0, 40.0, 50.0], "is_weapon": False},
                    {"label": "Weapon", "raw_label": "knife", "confidence": 0.85, "box": [20.0, 20.0, 30.0, 40.0], "is_weapon": True},
                ],
            },
            {
                "t": 2.0,
                "people_count": 2,
                "weapon_count": 1,
                "detections": [
                    {"label": "Human", "raw_label": "person", "confidence": 0.92, "box": [12.0, 10.0, 42.0, 50.0], "is_weapon": False},
                ],
            },
        ],
        "events": [
            {
                "id": "V1",
                "start_raw": 1.0,
                "end_raw": 3.0,
                "category": "Threat",
                "description": "Subject holding edged weapon in room",
                "confidence": 0.9,
            }
        ],
        "classification": {
            "primary": "Armed Threat",
            "confidence": 0.85,
            "reasoning": "Weapon observed in confrontation.",
            "probabilities": {
                "Armed Threat": 0.85,
                "Assault": 0.10,
                "Theft": 0.0,
                "Robbery": 0.0,
                "Vandalism": 0.0,
                "Trespassing": 0.0,
                "Kidnapping": 0.0,
                "No Crime Detected": 0.05,
            },
        },
        "crime_type": "Armed Threat",
        "crime_confidence": 0.85,
    }


def test_pipeline_end_to_end_all_modalities_success(tmp_data_dirs):
    """Verify end-to-end pipeline execution with image+audio+video creates complete fusion and index."""
    sess = session_store.new_session()
    sid = sess["id"]

    # Provide mocked media paths
    session_store.update_session(
        sid,
        media={
            "image": "dummy/path/image.jpg",
            "audio": "dummy/path/audio.wav",
            "video": "dummy/path/video.mp4",
        },
    )

    with patch("modules.pipeline.analyze_image", return_value=_mock_image_payload()), \
         patch("modules.pipeline.analyze_audio", return_value=_mock_audio_payload()), \
         patch("modules.pipeline.analyze_video", return_value=_mock_video_payload()):
        pipeline.run_pipeline(sid, language="en")

    result_session = session_store.load_session(sid)
    assert result_session is not None
    assert result_session["status"] == "done"

    # All stage progress must be done
    assert result_session["progress"]["image"] == "done"
    assert result_session["progress"]["audio"] == "done"
    assert result_session["progress"]["video"] == "done"
    assert result_session["progress"]["fusion"] == "done"

    # Fusion and index blocks must exist
    assert result_session["fusion"] is not None
    assert len(result_session["fusion"]["timeline"]) > 0
    assert result_session["index"] is not None
    assert "facts" in result_session["index"]

    # Validate against canonical session schema
    assert_session_schema(result_session)


def test_pipeline_end_to_end_partial_failure_resilience(tmp_data_dirs):
    """Verify that if one modality raises an exception, the pipeline still completes with overall 'done' status."""
    sess = session_store.new_session()
    sid = sess["id"]

    session_store.update_session(
        sid,
        media={
            "image": "dummy/path/image.jpg",
            "audio": "dummy/path/audio.wav",
            "video": "dummy/path/video.mp4",
        },
    )

    # Audio fails with an error; image and video succeed
    with patch("modules.pipeline.analyze_image", return_value=_mock_image_payload()), \
         patch("modules.pipeline.analyze_audio", side_effect=RuntimeError("Whisper decode failure")), \
         patch("modules.pipeline.analyze_video", return_value=_mock_video_payload()):
        pipeline.run_pipeline(sid, language="auto")

    result_session = session_store.load_session(sid)
    assert result_session is not None

    # Overall status should still be done because image and video succeeded
    assert result_session["status"] == "done"
    assert result_session["progress"]["image"] == "done"
    assert result_session["progress"]["video"] == "done"
    assert result_session["progress"]["audio"] == "error"
    assert result_session["progress"]["fusion"] == "done"

    # Error must be recorded
    assert len(result_session["errors"]) >= 1
    assert any("Whisper decode failure" in err for err in result_session["errors"])

    # Fusion block and index must still be generated from surviving modalities
    assert result_session["fusion"] is not None
    assert result_session["index"] is not None

    assert_session_schema(result_session)


def test_pipeline_end_to_end_all_modalities_error(tmp_data_dirs):
    """Verify pipeline transitions to status 'error' when every modality fails."""
    sess = session_store.new_session()
    sid = sess["id"]

    session_store.update_session(
        sid,
        media={
            "image": "dummy/image.jpg",
            "audio": "dummy/audio.wav",
        },
    )

    with patch("modules.pipeline.analyze_image", side_effect=RuntimeError("Image corrupted")), \
         patch("modules.pipeline.analyze_audio", side_effect=RuntimeError("Audio corrupted")):
        pipeline.run_pipeline(sid)

    result_session = session_store.load_session(sid)
    assert result_session is not None
    assert result_session["status"] == "error"
    assert result_session["progress"]["image"] == "error"
    assert result_session["progress"]["audio"] == "error"
    assert len(result_session["errors"]) == 2
