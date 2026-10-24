from __future__ import annotations

"""Unit tests for video probing, sampling, and visual event extraction."""

import pathlib
import pytest

from modules.video_analysis import (
    _center_dist,
    _compute_iou,
    _extract_visual_events,
    probe_video,
)


def test_probe_video_missing_file(tmp_path):
    """Ensure missing video files raise ValueError with clear message."""
    missing = tmp_path / "non_existent.mp4"
    with pytest.raises(ValueError, match="Cannot open video"):
        probe_video(missing)


def test_compute_iou_and_center_dist():
    """Verify geometric calculations for bounding boxes."""
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [5.0, 0.0, 15.0, 10.0]

    # Intersection width=5, height=10 => area=50. Union=150. IoU=50/150=1/3
    iou = _compute_iou(box_a, box_b)
    assert abs(iou - 0.3333) < 0.01

    # Center dist: (5, 5) to (10, 5) => dist = 5.0
    dist = _center_dist(box_a, box_b)
    assert abs(dist - 5.0) < 0.001


def test_extract_visual_events_debounces_flicker():
    """Verify that 1-frame spikes/dips are ignored by debouncing."""
    frames = [
        {"t": 0.0, "people_count": 1, "weapon_count": 0, "detections": []},
        {"t": 1.0, "people_count": 2, "weapon_count": 0, "detections": []},  # 1-frame flicker
        {"t": 2.0, "people_count": 1, "weapon_count": 0, "detections": []},
        {"t": 3.0, "people_count": 1, "weapon_count": 0, "detections": []},
    ]

    events = _extract_visual_events(frames)
    # The temporary jump at t=1.0 should be debounced and not generate Person entered/left
    person_events = [e for e in events if "Person" in e["description"]]
    assert len(person_events) == 0


def test_extract_visual_events_transitions():
    """Verify detection of Person entered, Weapon appeared/removed, and Crowd formed."""
    frames = [
        {"t": 0.0, "people_count": 0, "weapon_count": 0, "detections": []},
        {"t": 1.0, "people_count": 1, "weapon_count": 0, "detections": [{"label": "Human", "confidence": 0.9, "box": [0,0,10,10]}]},
        {"t": 2.0, "people_count": 1, "weapon_count": 1, "detections": [{"label": "Weapon", "is_weapon": True, "confidence": 0.85, "box": [0,0,5,5]}]},
        {"t": 3.0, "people_count": 3, "weapon_count": 1, "detections": []},
        {"t": 4.0, "people_count": 3, "weapon_count": 0, "detections": []},
        {"t": 5.0, "people_count": 0, "weapon_count": 0, "detections": []},
    ]

    events = _extract_visual_events(frames)
    categories = [e["category"] for e in events]

    assert "Arrival" in categories  # Person entered
    assert "Threat" in categories   # Weapon appeared
    assert "Other" in categories    # Crowd formed or weapon removed
    assert "Departure" in categories  # Person departed

    # Sequential IDs
    assert [e["id"] for e in events] == [f"V{i+1}" for i in range(len(events))]
    # All start_raw <= end_raw
    assert all(e["start_raw"] <= e["end_raw"] for e in events)


def test_probe_sample_video():
    """Probe the downloaded test sample if present."""
    sample_path = pathlib.Path("data/uploads/sample.mp4")
    if sample_path.exists():
        info = probe_video(sample_path)
        assert info["width"] > 0
        assert info["height"] > 0
        assert info["fps"] > 0
        assert info["frame_count"] > 0
        assert info["duration"] > 0


def test_classify_crime_llm_normalization_and_primary_fix(monkeypatch):
    """Test LLM probabilities normalization and primary argmax override when LLM primary is inconsistent."""
    from modules.reasoning import classify_crime
    import config

    monkeypatch.setattr("modules.reasoning.is_configured", lambda: True)

    # LLM returns probabilities summing to 1.4, missing taxonomy labels, and primary != argmax
    mock_llm_response = {
        "probabilities": {
            "Assault": 0.9,
            "Robbery": 0.3,
            "Weapon Possession": 0.2,
        },
        "primary": "Robbery",  # Inconsistent with argmax (Assault is 0.9)
        "reasoning": "Physical violence observed between subjects.",
    }

    monkeypatch.setattr("modules.reasoning.chat_json", lambda **kwargs: mock_llm_response)

    frames = [{"t": 0.0, "people_count": 2, "weapon_count": 0, "detections": []}]
    events = []

    res = classify_crime(frames, events)

    # 1. Primary must be overridden to Assault (argmax)
    assert res["crime_type"] == "Assault"

    # 2. All taxonomy keys must be present
    probs = res["crime_probabilities"]
    for cat in config.CRIME_TAXONOMY:
        assert cat in probs

    # 3. Probabilities must sum to 1.0
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-3

    # 4. Confidence equals max probability
    assert res["crime_confidence"] == probs["Assault"]
    assert res["crime_confidence"] > probs["Robbery"]


def test_classify_crime_fallback_heuristic(monkeypatch):
    """Verify deterministic fallback heuristics when LLM is unavailable."""
    from modules.reasoning import classify_crime

    monkeypatch.setattr("modules.reasoning.is_configured", lambda: False)

    # Case A: Weapon present for >= 2s and >= 2 people -> Armed Threat (0.5)
    frames_armed = [
        {"t": 0.0, "people_count": 2, "weapon_count": 1},
        {"t": 1.0, "people_count": 2, "weapon_count": 1},
        {"t": 2.0, "people_count": 2, "weapon_count": 0},
    ]
    res_armed = classify_crime(frames_armed, [])
    assert res_armed["crime_type"] == "Armed Threat"
    assert abs(res_armed["crime_confidence"] - 0.5) < 0.01
    assert abs(sum(res_armed["crime_probabilities"].values()) - 1.0) < 1e-3

    # Case B: Physical contact event -> Assault (0.5)
    frames_assault = [
        {"t": 0.0, "people_count": 2, "weapon_count": 0},
    ]
    events_assault = [
        {
            "id": "V1",
            "category": "Aggression",
            "description": "Possible physical contact between subjects",
            "start_raw": 0.0,
            "end_raw": 1.0,
        }
    ]
    res_assault = classify_crime(frames_assault, events_assault)
    assert res_assault["crime_type"] == "Assault"
    assert abs(res_assault["crime_confidence"] - 0.5) < 0.01
    assert abs(sum(res_assault["crime_probabilities"].values()) - 1.0) < 1e-3

    # Case C: Routine peaceful video -> No Crime Detected (0.6)
    frames_peaceful = [
        {"t": 0.0, "people_count": 1, "weapon_count": 0},
        {"t": 1.0, "people_count": 1, "weapon_count": 0},
    ]
    res_peaceful = classify_crime(frames_peaceful, [])
    assert res_peaceful["crime_type"] == "No Crime Detected"
    assert abs(res_peaceful["crime_confidence"] - 0.6) < 0.01
    assert abs(sum(res_peaceful["crime_probabilities"].values()) - 1.0) < 1e-3


def test_build_evidence_index_modalities():
    """Verify build_evidence_index with video only, audio only, and multimodal session."""
    from modules.chat import build_evidence_index

    # 1. Video only session
    video_session = {
        "video": {
            "frames": [
                {
                    "t": 5.0,
                    "people_count": 2,
                    "weapon_count": 1,
                    "detections": [
                        {"label": "Human", "raw_label": "person", "confidence": 0.92},
                        {"label": "Human", "raw_label": "person", "confidence": 0.88},
                        {"label": "Weapon", "raw_label": "knife", "is_weapon": True, "confidence": 0.85},
                    ],
                    "caption": "A person holding a knife near another person.",
                }
            ],
            "events": [
                {
                    "id": "V1",
                    "category": "Threat",
                    "description": "Weapon appeared in scene",
                    "start_raw": 4.8,
                    "end_raw": 6.0,
                }
            ],
            "crime_type": "Armed Threat",
            "crime_confidence": 0.5,
        }
    }

    idx_v = build_evidence_index(video_session)
    assert "5" in idx_v["by_second"]
    v_sec5 = idx_v["by_second"]["5"]
    assert v_sec5["video"] is not None
    assert v_sec5["video"]["people_count"] == 2
    assert v_sec5["video"]["weapon_count"] == 1
    assert "knife" in v_sec5["video"]["detections"]
    assert "V1" in v_sec5["events"]
    assert v_sec5["audio"] == []

    # Check fact format: "At 5s: 2 people, 1 weapon (knife)"
    assert any("At 5s: 2 people, 1 weapon (knife)" in f for f in idx_v["facts"])
    assert "knife" in idx_v["keywords"]
    assert 5 in idx_v["keywords"]["knife"]

    # 2. Audio only session
    audio_session = {
        "audio": {
            "segments": [
                {"id": 1, "start_raw": 2.0, "end_raw": 4.0, "text": "Put your hands up!"}
            ],
            "events": [
                {
                    "id": "A1",
                    "category": "Threat",
                    "description": "Verbal threat directed at victim",
                    "who": "Suspect",
                    "where": "Alleyway",
                    "confidence": 0.85,
                    "start_raw": 2.0,
                    "end_raw": 4.0,
                }
            ],
            "facts": {
                "who": "Suspect",
                "where": "Alleyway",
                "when": "Night",
                "suspect_description": "Tall male in dark hoodie",
            },
            "transcript": "Put your hands up!",
        }
    }

    idx_a = build_evidence_index(audio_session)
    assert "2" in idx_a["by_second"]
    assert "3" in idx_a["by_second"]
    a_sec2 = idx_a["by_second"]["2"]
    assert a_sec2["video"] is None
    assert len(a_sec2["audio"]) > 0
    assert "A1" in a_sec2["events"]
    assert any("Audio suspect description: Tall male in dark hoodie" in f for f in idx_a["facts"])
    assert "threat" in idx_a["keywords"]

    # 3. Both video and audio combined
    combined_session = {
        "video": video_session["video"],
        "audio": audio_session["audio"],
    }
    idx_both = build_evidence_index(combined_session)
    assert "5" in idx_both["by_second"]
    assert "2" in idx_both["by_second"]
    assert idx_both["by_second"]["5"]["video"] is not None
    assert idx_both["by_second"]["2"]["audio"] is not None

    # 4. Graceful handling of missing modalities
    empty_idx = build_evidence_index({})
    assert empty_idx == {"by_second": {}, "facts": [], "keywords": {}}
    none_idx = build_evidence_index(None)
    assert none_idx == {"by_second": {}, "facts": [], "keywords": {}}

