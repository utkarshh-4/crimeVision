from __future__ import annotations

"""Image analysis module implementing RT-DETR detection, CLIP classification, and BLIP captioning."""

import logging
import pathlib
from typing import Any, Dict, List, Tuple

from PIL import Image
import torch
import torchvision.ops

import config
from modules.device import get_device
from modules.model_registry import get_blip, get_clip, get_rtdetr
from modules.reasoning import generate_image_hypotheses
from modules.utils import round_conf, to_jsonable

logger = logging.getLogger("crimevision.image")


def load_image(path: str | pathlib.Path) -> Image.Image:
    """Validate, open, and downscale an RGB image so its longest side is at most 1600 px.

    Attaches original dimensions to image.info['original_size'].
    """
    img_path = pathlib.Path(path)
    if not img_path.is_file():
        raise ValueError(f"Image file does not exist: {img_path}")

    try:
        with Image.open(img_path) as opened_img:
            img = opened_img.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Unable to read or parse image at '{img_path}': {exc}") from exc

    orig_w, orig_h = img.size
    max_side = max(orig_w, orig_h)
    if max_side > 1600:
        scale = 1600.0 / float(max_side)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    img.info["original_size"] = (orig_w, orig_h)
    setattr(img, "original_size", (orig_w, orig_h))
    return img


def detect_objects(image: Image.Image) -> List[Dict[str, Any]]:
    """Detect objects using RT-DETR, applying NMS and label mapping according to forensic rules."""
    orig_w, orig_h = getattr(image, "original_size", image.info.get("original_size", (image.width, image.height)))
    processor, model = get_rtdetr()
    device = get_device()

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        target_sizes = torch.tensor([(orig_h, orig_w)], device=outputs.logits.device)
        results = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=config.DET_THRESHOLD,
        )[0]

    boxes = results["boxes"]
    scores = results["scores"]
    labels = results["labels"]

    if len(boxes) > 0:
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold=config.NMS_IOU)
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

    id2label = getattr(model.config, "id2label", {})
    detections: List[Dict[str, Any]] = []

    for box, score, label_id in zip(boxes, scores, labels):
        raw_label = str(id2label.get(int(label_id.item()), f"class_{int(label_id.item())}")).lower()

        if raw_label == "person":
            label = "Human"
            is_weapon = False
        elif raw_label in config.WEAPON_COCO_LABELS:
            label = "Weapon"
            is_weapon = True
        else:
            label = "Object"
            is_weapon = False

        x1 = max(0.0, min(float(orig_w), round(float(box[0].item()), 2)))
        y1 = max(0.0, min(float(orig_h), round(float(box[1].item()), 2)))
        x2 = max(0.0, min(float(orig_w), round(float(box[2].item()), 2)))
        y2 = max(0.0, min(float(orig_h), round(float(box[3].item()), 2)))

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        detections.append({
            "label": label,
            "raw_label": raw_label,
            "confidence": round_conf(score.item()),
            "box": [x1, y1, x2, y2],
            "is_weapon": is_weapon,
        })

    return detections


def classify_scene(image: Image.Image) -> Dict[str, Any]:
    """Classify scene environment using zero-shot CLIP over predefined SCENE_LABELS."""
    processor, model = get_clip()
    device = get_device()
    prompts = [f"a photo of a {label}" for label in config.SCENE_LABELS]

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=-1)

    k = min(3, len(config.SCENE_LABELS))
    top_probs, top_indices = torch.topk(probs, k=k)

    top_k: List[Dict[str, Any]] = []
    for prob, idx in zip(top_probs, top_indices):
        top_k.append({
            "label": config.SCENE_LABELS[int(idx.item())],
            "confidence": round_conf(prob.item()),
        })

    return {
        "label": top_k[0]["label"],
        "confidence": top_k[0]["confidence"],
        "top_k": top_k,
    }


def check_firearm(image: Image.Image) -> Dict[str, Any]:
    """Check for firearm presence using zero-shot CLIP contrastive prompts."""
    processor, model = get_clip()
    device = get_device()

    pos_prompts = ["a photo of a person holding a gun", "a photo of a firearm"]
    neg_prompts = ["a photo of a person with empty hands", "a photo of an ordinary room"]
    prompts = pos_prompts + neg_prompts

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]
        probs = torch.softmax(logits, dim=-1)
        firearm_prob = float((probs[0] + probs[1]).item())

    conf = round_conf(firearm_prob)
    firearm_likely = bool(conf >= config.FIREARM_THRESHOLD)
    return {
        "firearm_likely": firearm_likely,
        "confidence": conf,
    }


def caption_image(image: Image.Image) -> str:
    """Generate image caption using BLIP with conditional fallback if initial output is too brief."""
    processor, model = get_blip()
    device = get_device()

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=60, num_beams=3)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()

        if len(caption.split()) < 3:
            inputs_cond = processor(images=image, text="a photograph of", return_tensors="pt").to(device)
            out_cond = model.generate(**inputs_cond, max_new_tokens=60, num_beams=3)
            caption_cond = processor.decode(out_cond[0], skip_special_tokens=True).strip()
            if len(caption_cond.split()) >= len(caption.split()):
                caption = caption_cond

    return caption


def analyze_pil_frame(image: Image.Image) -> Dict[str, Any]:
    """Analyze a single PIL frame for object detections and counts (reusable for video frames)."""
    detections = detect_objects(image)
    people_count = sum(1 for d in detections if d.get("label") == "Human")
    weapon_count = sum(1 for d in detections if d.get("is_weapon", False))

    return {
        "width": image.width,
        "height": image.height,
        "people_count": people_count,
        "weapon_count": weapon_count,
        "detections": detections,
    }


def analyze_image(path: str | pathlib.Path) -> Dict[str, Any]:
    """Perform full forensic image analysis orchestrating RT-DETR, CLIP, and BLIP."""
    image = load_image(path)
    orig_w, orig_h = getattr(image, "original_size", image.info.get("original_size", (image.width, image.height)))
    warnings: List[str] = []

    # 1. Object detection
    try:
        detections = detect_objects(image)
    except Exception as exc:
        logger.exception("Object detection failed: %s", exc)
        detections = []
        warnings.append(f"Object detection failed: {exc}")

    # 2. Scene classification
    try:
        scene = classify_scene(image)
    except Exception as exc:
        logger.exception("Scene classification failed: %s", exc)
        scene = {"label": "unknown", "confidence": 0.0, "top_k": []}
        warnings.append(f"Scene classification failed: {exc}")

    # 3. Firearm check
    try:
        firearm_check = check_firearm(image)
    except Exception as exc:
        logger.exception("Firearm check failed: %s", exc)
        firearm_check = {"firearm_likely": False, "confidence": 0.0}
        warnings.append(f"Firearm check failed: {exc}")

    # 4. Image captioning
    try:
        caption = caption_image(image)
    except Exception as exc:
        logger.exception("Image captioning failed: %s", exc)
        caption = ""
        warnings.append(f"Image captioning failed: {exc}")

    # 5. Synthetic firearm detection injection if likely
    if firearm_check.get("firearm_likely", False):
        detections.append({
            "label": "Weapon",
            "raw_label": "firearm (CLIP)",
            "confidence": firearm_check["confidence"],
            "box": [0.0, 0.0, float(orig_w), float(orig_h)],
            "is_weapon": True,
        })

    # 6. People & Weapon counts
    people_count = sum(1 for d in detections if d.get("label") == "Human")
    weapon_count = sum(1 for d in detections if d.get("is_weapon", False))

    result: Dict[str, Any] = {
        "width": int(orig_w),
        "height": int(orig_h),
        "people_count": int(people_count),
        "weapon_count": int(weapon_count),
        "detections": detections,
        "scene": scene,
        "firearm_check": firearm_check,
        "caption": caption,
        "hypotheses": [],
    }

    # 7. Generate scenario hypotheses
    try:
        result["hypotheses"] = generate_image_hypotheses(result)
    except Exception as exc:
        logger.exception("Hypothesis generation failed: %s", exc)
        result["hypotheses"] = []
        warnings.append(f"Hypothesis generation failed: {exc}")

    if warnings:
        result["warnings"] = warnings

    return to_jsonable(result)
