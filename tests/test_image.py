from __future__ import annotations

"""Unit tests for image analysis, RT-DETR detection, CLIP/BLIP mocking, and forensic reasoning."""

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

import config
from modules.image_analysis import (
    analyze_image,
    analyze_pil_frame,
    check_firearm,
    classify_scene,
    detect_objects,
    load_image,
)
from modules.llm import LLMError, chat_json
from modules.reasoning import (
    _compute_relative_position,
    _rule_based_image_hypotheses,
    generate_image_hypotheses,
)
from tests.schema_utils import assert_image_schema


def _make_mock_rtdetr(boxes_tensor, scores_tensor, labels_tensor, id2label=None):
    """Create mock processor and model returning specified detection outputs."""
    if id2label is None:
        id2label = {0: "person", 1: "knife", 2: "chair"}

    mock_proc = MagicMock()
    mock_model = MagicMock()
    mock_model.config.id2label = id2label

    # Inputs mocking
    mock_inputs = MagicMock()
    mock_inputs.to.return_value = mock_inputs
    mock_proc.return_value = mock_inputs

    # Outputs mocking
    mock_outputs = MagicMock()
    mock_outputs.logits.device = torch.device("cpu")
    mock_model.return_value = mock_outputs

    # Post processing
    mock_proc.post_process_object_detection.return_value = [
        {
            "boxes": boxes_tensor,
            "scores": scores_tensor,
            "labels": labels_tensor,
        }
    ]
    return mock_proc, mock_model


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


def test_detect_objects_label_mapping_and_counts(make_synthetic_image):
    """Verify person -> Human, knife -> Weapon, chair -> Object mapping and counts."""
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],
        [60.0, 60.0, 80.0, 80.0],
        [100.0, 100.0, 150.0, 150.0],
    ])
    scores = torch.tensor([0.95, 0.88, 0.72])
    labels = torch.tensor([0, 1, 2])
    id2label = {0: "person", 1: "knife", 2: "chair"}

    fake_proc, fake_model = _make_mock_rtdetr(boxes, scores, labels, id2label)

    img_path = make_synthetic_image()
    img = load_image(img_path)

    with patch("modules.image_analysis.get_rtdetr", return_value=(fake_proc, fake_model)):
        frame_res = analyze_pil_frame(img)

    assert frame_res["people_count"] == 1
    assert frame_res["weapon_count"] == 1
    assert len(frame_res["detections"]) == 3

    dets = frame_res["detections"]
    assert dets[0]["label"] == "Human"
    assert dets[0]["raw_label"] == "person"
    assert dets[0]["is_weapon"] is False

    assert dets[1]["label"] == "Weapon"
    assert dets[1]["raw_label"] == "knife"
    assert dets[1]["is_weapon"] is True

    assert dets[2]["label"] == "Object"
    assert dets[2]["raw_label"] == "chair"
    assert dets[2]["is_weapon"] is False


def test_detect_objects_nms_removes_duplicates(make_synthetic_image):
    """Verify NMS removes duplicate overlapping bounding boxes."""
    # Two identical overlapping boxes for "person"
    boxes = torch.tensor([
        [20.0, 20.0, 60.0, 60.0],
        [20.0, 20.0, 60.0, 60.0],
    ])
    scores = torch.tensor([0.95, 0.82])
    labels = torch.tensor([0, 0])
    fake_proc, fake_model = _make_mock_rtdetr(boxes, scores, labels, {0: "person"})

    img = Image.new("RGB", (100, 100))
    with patch("modules.image_analysis.get_rtdetr", return_value=(fake_proc, fake_model)):
        dets = detect_objects(img)

    assert len(dets) == 1
    assert dets[0]["confidence"] == 0.95
    assert dets[0]["label"] == "Human"


def test_check_firearm_and_synthetic_detection(make_synthetic_image):
    """Verify firearm synthetic detection is injected when CLIP predicts high probability."""
    # Mock CLIP to return high firearm probability
    mock_clip_proc = MagicMock()
    mock_clip_inputs = MagicMock()
    mock_clip_inputs.to.return_value = mock_clip_inputs
    mock_clip_proc.return_value = mock_clip_inputs

    mock_clip_model = MagicMock()
    mock_clip_out = MagicMock()
    # High positive logits for firearm prompts (index 0 and 1)
    mock_clip_out.logits_per_image = torch.tensor([[10.0, 10.0, -10.0, -10.0]])
    mock_clip_model.return_value = mock_clip_out

    # Mock RT-DETR with empty detections
    fake_proc, fake_model = _make_mock_rtdetr(
        torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    )

    img_path = make_synthetic_image()
    with patch("modules.image_analysis.get_rtdetr", return_value=(fake_proc, fake_model)), \
         patch("modules.image_analysis.get_clip", return_value=(mock_clip_proc, mock_clip_model)), \
         patch("modules.image_analysis.caption_image", return_value="a scene"):
        res = analyze_image(img_path)

    assert res["firearm_check"]["firearm_likely"] is True
    assert res["weapon_count"] == 1
    assert any(d.get("raw_label") == "firearm (CLIP)" for d in res["detections"])
    assert_image_schema(res)


def test_analyze_image_caption_model_failure_resilience(make_synthetic_image):
    """Verify analyze_image returns schema-valid dict with non-empty warnings if caption model fails."""
    fake_proc, fake_model = _make_mock_rtdetr(
        torch.zeros((0, 4)), torch.zeros((0,)), torch.zeros((0,), dtype=torch.long)
    )

    img_path = make_synthetic_image()
    with patch("modules.image_analysis.get_rtdetr", return_value=(fake_proc, fake_model)), \
         patch("modules.image_analysis.classify_scene", return_value={"label": "hall", "confidence": 0.8, "top_k": []}), \
         patch("modules.image_analysis.check_firearm", return_value={"firearm_likely": False, "confidence": 0.1}), \
         patch("modules.image_analysis.caption_image", side_effect=RuntimeError("BLIP weights unavailable")):
        res = analyze_image(img_path)

    assert_image_schema(res)
    assert res["caption"] == ""
    assert "warnings" in res
    assert any("Image captioning failed" in w for w in res["warnings"])
