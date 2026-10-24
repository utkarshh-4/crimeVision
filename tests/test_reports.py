from __future__ import annotations

"""Unit and integration tests for forensic PDF report generation."""

import pathlib
import pytest

from modules import reports, session_store


def test_build_pdf_full_session(
    tmp_data_dirs,
    make_synthetic_image,
    make_synthetic_wav,
    make_synthetic_mp4,
):
    """Verify build_pdf creates a valid forensic PDF > 5 KB starting with %PDF for full multimodal session."""
    img_path = make_synthetic_image("crime_scene.jpg", size=(120, 120))
    aud_path = make_synthetic_wav("dispatch.wav", duration_s=1.0)
    # Generate 3-second video @ 5 fps (15 frames)
    vid_path = make_synthetic_mp4("cctv_feed.mp4", num_frames=15, fps=5.0, size=(120, 120))

    sess = session_store.new_session()
    sid = sess["id"]

    full_session = {
        "id": sid,
        "session_id": sid,
        "created_at": "2026-10-24T17:30:00Z",
        "status": "done",
        "progress": {
            "image": "done",
            "audio": "done",
            "video": "done",
            "fusion": "done",
            "report": "done",
        },
        "media": {
            "image": str(img_path),
            "audio": str(aud_path),
            "video": str(vid_path),
        },
        "errors": [],
        "image": {
            "width": 120,
            "height": 120,
            "people_count": 2,
            "weapon_count": 1,
            "detections": [
                {
                    "label": "Human",
                    "raw_label": "person",
                    "confidence": 0.94,
                    "box": [10.0, 10.0, 50.0, 90.0],
                    "is_weapon": False,
                },
                {
                    "label": "Weapon",
                    "raw_label": "knife",
                    "confidence": 0.88,
                    "box": [55.0, 40.0, 75.0, 65.0],
                    "is_weapon": True,
                },
                {
                    "label": "Object",
                    "raw_label": "backpack",
                    "confidence": 0.76,
                    "box": [80.0, 70.0, 110.0, 110.0],
                    "is_weapon": False,
                },
            ],
            "scene": {"label": "hall", "confidence": 0.82},
            "caption": "a person holding a knife in a hallway",
            "hypotheses": [
                {"title": "Armed Intrusion", "explanation": "Subject brandishing weapon in corridor.", "probability": 0.7},
                {"title": "Security Training", "explanation": "Controlled exercise with dummy weapon.", "probability": 0.2},
                {"title": "Unverified Altercation", "explanation": "Indeterminate physical dispute.", "probability": 0.1},
            ],
        },
        "audio": {
            "transcript": "Stay back! Drop the knife immediately!",
            "language": "en",
            "segments": [
                {"id": "S1", "start_raw": 0.2, "end_raw": 1.4, "text": "Stay back!"},
                {"id": "S2", "start_raw": 1.6, "end_raw": 2.8, "text": "Drop the knife immediately!"},
            ],
            "events": [
                {
                    "id": "A1",
                    "start_raw": 0.2,
                    "end_raw": 2.8,
                    "category": "Threat",
                    "description": "Verbal demand to disarm",
                    "who": "Officer",
                    "where": "Hallway",
                    "confidence": 0.95,
                }
            ],
            "facts": {
                "who": "Officer",
                "where": "Hallway",
                "suspect_description": "Armed subject",
                "location_inference": {"label": "hallway", "confidence": 0.85},
            },
        },
        "video": {
            "fps": 5.0,
            "frame_count": 15,
            "duration": 3.0,
            "crime_type": "Armed Threat",
            "crime_confidence": 0.89,
            "classification": {
                "primary": "Armed Threat",
                "confidence": 0.89,
                "reasoning": "Brandishing edged weapon during verbal challenge.",
                "probabilities": {
                    "Armed Threat": 0.89,
                    "Assault": 0.08,
                    "Robbery": 0.02,
                    "No Crime Detected": 0.01,
                },
            },
            "frames": [
                {"t": 0.0, "people_count": 1, "weapon_count": 0, "detections": []},
                {
                    "t": 1.0,
                    "people_count": 2,
                    "weapon_count": 1,
                    "detections": [
                        {"label": "Human", "confidence": 0.92, "box": [10.0, 10.0, 50.0, 90.0]},
                        {"label": "Weapon", "confidence": 0.88, "box": [55.0, 40.0, 75.0, 65.0]},
                    ],
                },
                {
                    "t": 2.0,
                    "people_count": 2,
                    "weapon_count": 1,
                    "detections": [
                        {"label": "Human", "confidence": 0.91, "box": [12.0, 10.0, 52.0, 90.0]},
                    ],
                },
            ],
            "events": [
                {
                    "id": "V1",
                    "start_raw": 0.5,
                    "end_raw": 2.5,
                    "category": "Weapon",
                    "description": "Edged weapon visible in right hand",
                    "confidence": 0.91,
                }
            ],
        },
        "fusion": {
            "summary": "Corroborated armed confrontation in hallway with visible edged weapon matching acoustic command.",
            "overall_confidence": 0.93,
            "anchor_events": ["A1", "V1"],
            "conflicts": [],
            "timeline": [
                {
                    "id": "F1",
                    "start_raw": 0.2,
                    "end_raw": 2.8,
                    "modality": "fused",
                    "category": "Threat",
                    "description": "Verbal disarm order corroborated with knife detection",
                    "confidence": 0.94,
                    "is_anchor": True,
                },
                {
                    "id": "F2",
                    "start_raw": 1.0,
                    "end_raw": 2.0,
                    "modality": "video",
                    "category": "Weapon",
                    "description": "Knife drawn in hallway",
                    "confidence": 0.88,
                    "is_anchor": True,
                },
            ],
            "hypotheses": [
                {"title": "Armed Threat", "explanation": "Subject brandishing knife confronted by guard.", "probability": 0.75},
                {"title": "Defensive Standoff", "explanation": "Subject holding item in defense.", "probability": 0.18},
                {"title": "Simulation Drill", "explanation": "Training scenario.", "probability": 0.07},
            ],
        },
    }

    session_store.save_session(full_session)

    # Execute PDF generation
    pdf_path = reports.build_pdf(sid)

    assert isinstance(pdf_path, pathlib.Path)
    assert pdf_path.exists(), "Generated PDF file does not exist on disk"
    size = pdf_path.stat().st_size
    assert size > 5000, f"Expected PDF size > 5 KB, got {size} bytes"

    # Validate PDF signature
    with open(pdf_path, "rb") as f:
        header = f.read(5)
    assert header.startswith(b"%PDF"), f"Invalid PDF header: {header}"


def test_build_pdf_missing_session_raises_keyerror():
    """Verify build_pdf raises KeyError when session ID is missing."""
    with pytest.raises(KeyError, match="not found"):
        reports.build_pdf("0" * 32)


def test_build_pdf_empty_modalities(tmp_data_dirs):
    """Verify build_pdf gracefully renders when all modalities are None."""
    sess = session_store.new_session()
    sid = sess["id"]
    session_store.update_session(
        sid,
        status="done",
        fusion={
            "summary": "No evidence was submitted for analysis.",
            "overall_confidence": 0.0,
            "anchor_events": [],
            "conflicts": [],
            "timeline": [],
            "hypotheses": [
                {"title": "Insufficient Evidence", "explanation": "No files uploaded.", "probability": 1.0},
            ],
        },
    )

    pdf_path = reports.build_pdf(sid)
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 3000
    with open(pdf_path, "rb") as f:
        header = f.read(5)
    assert header.startswith(b"%PDF")


def test_api_report_download_route(client, tmp_data_dirs):
    """Verify /api/report/<sid> endpoint returns 404 for unknown session and 200 attachment for valid session."""
    # 1. Unknown session -> 404
    resp_404 = client.get(f"/api/report/{'0'*32}")
    assert resp_404.status_code == 404

    # 2. Valid session -> 200 with application/pdf attachment
    sess = session_store.new_session()
    sid = sess["id"]
    session_store.update_session(
        sid,
        status="done",
        fusion={
            "summary": "Forensic test session",
            "overall_confidence": 0.85,
            "anchor_events": [],
            "conflicts": [],
            "timeline": [],
            "hypotheses": [],
        },
    )

    resp_valid = client.get(f"/api/report/{sid}")
    assert resp_valid.status_code == 200
    assert resp_valid.mimetype == "application/pdf"
    assert "attachment" in resp_valid.headers.get("Content-Disposition", "")
    assert f"CrimeVision_Report_{sid[:8]}.pdf" in resp_valid.headers.get("Content-Disposition", "")
    assert resp_valid.data.startswith(b"%PDF")
