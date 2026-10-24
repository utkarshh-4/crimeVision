from __future__ import annotations

"""Schema validation utilities for CrimeVision AI forensic data structures."""

from typing import Any, Dict, List


def assert_session_schema(session: Dict[str, Any]) -> None:
    """Validate canonical schema for a CrimeVision AI session.
    
    Checks required keys, types, boundary ranges (confidence in [0, 1],
    start_raw <= end_raw, hypotheses probability sums ≈ 1.0), and modality payloads.
    """
    assert isinstance(session, dict), f"Session must be a dict, got {type(session)}"
    
    # 1. Top-level required keys
    required_keys = {"id", "created_at", "status", "progress", "errors"}
    for k in required_keys:
        assert k in session, f"Missing required top-level key: {k}"
    
    assert isinstance(session["id"], str) and len(session["id"]) == 32, (
        f"Invalid session id format: {session.get('id')}"
    )
    assert session["status"] in ("queued", "running", "done", "error"), (
        f"Unexpected status: {session.get('status')}"
    )
    assert isinstance(session["progress"], dict), "progress must be a dict"
    for stage, state in session["progress"].items():
        assert state in ("pending", "running", "done", "error", "skipped"), (
            f"Invalid progress state '{state}' for stage '{stage}'"
        )
    assert isinstance(session["errors"], list), "errors must be a list"

    # 2. Media map (if present)
    if "media" in session and session["media"] is not None:
        assert isinstance(session["media"], dict), "media must be a dict"

    # 3. Image modality block (if present)
    if session.get("image") is not None:
        assert_image_schema(session["image"])

    # 4. Audio modality block (if present)
    if session.get("audio") is not None:
        assert_audio_schema(session["audio"])

    # 5. Video modality block (if present)
    if session.get("video") is not None:
        assert_video_schema(session["video"])

    # 6. Fusion block (if present)
    if session.get("fusion") is not None:
        assert_fusion_schema(session["fusion"])

    # 7. Index block (if present)
    if session.get("index") is not None:
        idx = session["index"]
        assert isinstance(idx, dict), "index must be a dict"
        assert "by_second" in idx or "facts" in idx, "index must contain by_second or facts"


def assert_image_schema(img: Dict[str, Any]) -> None:
    """Validate schema of an image analysis result dictionary."""
    assert isinstance(img, dict), "image result must be a dict"
    for key in ("width", "height", "people_count", "weapon_count", "detections", "scene", "caption", "hypotheses"):
        assert key in img, f"image result missing key '{key}'"

    assert isinstance(img["width"], int) and img["width"] > 0
    assert isinstance(img["height"], int) and img["height"] > 0
    assert isinstance(img["people_count"], int) and img["people_count"] >= 0
    assert isinstance(img["weapon_count"], int) and img["weapon_count"] >= 0
    assert isinstance(img["detections"], list)

    for det in img["detections"]:
        assert det["label"] in ("Human", "Weapon", "Object"), f"Unexpected label: {det.get('label')}"
        assert 0.0 <= det["confidence"] <= 1.0, f"Confidence out of range: {det.get('confidence')}"
        assert len(det["box"]) == 4, f"Box must have 4 coords, got: {det.get('box')}"
        assert all(isinstance(coord, (int, float)) for coord in det["box"])
        assert isinstance(det["is_weapon"], bool)

    assert isinstance(img["scene"], dict)
    assert "label" in img["scene"]
    assert 0.0 <= img["scene"].get("confidence", 0.0) <= 1.0

    assert isinstance(img["hypotheses"], list)
    if len(img["hypotheses"]) > 0:
        assert len(img["hypotheses"]) == 3, f"Expected 3 image hypotheses, got {len(img['hypotheses'])}"
        for h in img["hypotheses"]:
            assert "title" in h and "explanation" in h and "probability" in h
            assert 0.0 <= h["probability"] <= 1.0
        prob_sum = sum(h["probability"] for h in img["hypotheses"])
        assert abs(prob_sum - 1.0) < 0.05, f"Hypotheses probabilities sum to {prob_sum}, expected ≈ 1.0"


def assert_audio_schema(aud: Dict[str, Any]) -> None:
    """Validate schema of an audio analysis result dictionary."""
    assert isinstance(aud, dict), "audio result must be a dict"
    assert "transcript" in aud
    assert "language" in aud

    if "segments" in aud and isinstance(aud["segments"], list):
        for seg in aud["segments"]:
            assert seg["start_raw"] <= seg["end_raw"], f"Segment start > end: {seg}"
            assert seg["start_raw"] >= 0.0

    if "events" in aud and isinstance(aud["events"], list):
        for ev in aud["events"]:
            assert ev["start_raw"] <= ev["end_raw"], f"Audio event start > end: {ev}"
            assert 0.0 <= ev["confidence"] <= 1.0

    if "facts" in aud and isinstance(aud["facts"], dict):
        facts = aud["facts"]
        if "hypotheses" in facts and isinstance(facts["hypotheses"], list) and len(facts["hypotheses"]) > 0:
            for h in facts["hypotheses"]:
                assert 0.0 <= h["probability"] <= 1.0
            prob_sum = sum(h["probability"] for h in facts["hypotheses"])
            assert abs(prob_sum - 1.0) < 0.05, f"Audio hypotheses sum to {prob_sum}, expected ≈ 1.0"


def assert_video_schema(vid: Dict[str, Any]) -> None:
    """Validate schema of a video analysis result dictionary."""
    assert isinstance(vid, dict), "video result must be a dict"
    assert "fps" in vid and vid["fps"] > 0
    assert "frame_count" in vid and vid["frame_count"] >= 0
    assert "duration" in vid and vid["duration"] >= 0.0

    if "events" in vid and isinstance(vid["events"], list):
        for ev in vid["events"]:
            assert ev["start_raw"] <= ev["end_raw"], f"Video event start > end: {ev}"
            assert 0.0 <= ev["confidence"] <= 1.0

    if "classification" in vid and isinstance(vid["classification"], dict):
        probs = vid["classification"].get("probabilities", {})
        if probs:
            for p in probs.values():
                assert 0.0 <= p <= 1.0
            prob_sum = sum(probs.values())
            assert abs(prob_sum - 1.0) < 0.05, f"Video classification probabilities sum to {prob_sum}"


def assert_fusion_schema(fus: Dict[str, Any]) -> None:
    """Validate schema of a multimodal fusion result dictionary."""
    assert isinstance(fus, dict), "fusion result must be a dict"
    for key in ("timeline", "conflicts", "hypotheses", "summary", "overall_confidence"):
        assert key in fus, f"fusion result missing key '{key}'"

    assert 0.0 <= fus["overall_confidence"] <= 1.0, (
        f"overall_confidence out of bounds: {fus.get('overall_confidence')}"
    )
    assert isinstance(fus["timeline"], list)
    for ev in fus["timeline"]:
        assert "id" in ev and "modality" in ev and "category" in ev and "description" in ev
        assert ev["start_raw"] <= ev["end_raw"], f"Timeline event start > end: {ev}"
        assert 0.0 <= ev["confidence"] <= 1.0
        assert ev["modality"] in ("image", "video", "audio", "fused"), f"Unknown modality: {ev.get('modality')}"

    assert isinstance(fus["hypotheses"], list)
    if len(fus["hypotheses"]) > 0:
        assert len(fus["hypotheses"]) == 3, f"Expected 3 fusion hypotheses, got {len(fus['hypotheses'])}"
        for h in fus["hypotheses"]:
            assert "title" in h and "explanation" in h and "probability" in h
            assert 0.0 <= h["probability"] <= 1.0
        prob_sum = sum(h["probability"] for h in fus["hypotheses"])
        assert abs(prob_sum - 1.0) < 0.05, f"Fusion hypotheses sum to {prob_sum}, expected ≈ 1.0"
