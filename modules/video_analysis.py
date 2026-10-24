from __future__ import annotations

"""Video analysis module for temporal sampling, object tracking, and visual event extraction."""

import logging
import math
import pathlib
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

import config
from modules.image_analysis import analyze_pil_frame, caption_image, check_firearm
from modules.reasoning import classify_crime
from modules.utils import round_conf, to_jsonable

logger = logging.getLogger("crimevision.video")


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two [x1, y1, x2, y2] bounding boxes."""
    x_a = max(box1[0], box2[0])
    y_a = max(box1[1], box2[1])
    x_b = min(box1[2], box2[2])
    y_b = min(box1[3], box2[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h

    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _center_dist(box1: List[float], box2: List[float]) -> float:
    """Compute Euclidean distance between the centers of two bounding boxes."""
    c1x = (box1[0] + box1[2]) / 2.0
    c1y = (box1[1] + box1[3]) / 2.0
    c2x = (box2[0] + box2[2]) / 2.0
    c2y = (box2[1] + box2[3]) / 2.0
    return math.hypot(c1x - c2x, c1y - c2y)


def probe_video(path: str | pathlib.Path) -> Dict[str, Any]:
    """Inspect video file metadata using OpenCV."""
    video_path = pathlib.Path(path)
    if not video_path.is_file():
        raise ValueError(f"Cannot open video: file does not exist at '{video_path}'")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: stream could not be opened for '{video_path}'")

    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 25.0 if (raw_fps <= 0 or math.isnan(raw_fps) or math.isinf(raw_fps)) else float(raw_fps)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"Cannot open video: invalid stream properties (frames={frame_count}, {width}x{height})"
            )

        duration = round(float(frame_count) / fps, 3)
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
        }
    finally:
        cap.release()


def sample_frames(
    path: str | pathlib.Path,
    hz: int = config.VIDEO_SAMPLE_HZ,
    max_seconds: int = config.MAX_VIDEO_SECONDS,
) -> Iterator[Tuple[float, Image.Image]]:
    """Sample video frames at a constant rate, resizing to max 960 px while preserving aspect ratio."""
    info = probe_video(path)
    fps = info["fps"]
    duration = info["duration"]
    orig_w = info["width"]
    orig_h = info["height"]

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video for sampling: {path}")

    effective_duration = min(duration, float(max_seconds))
    step = 1.0 / max(1, hz)
    t = 0.0

    try:
        while t <= effective_duration:
            frame_idx = int(round(t * fps))
            if frame_idx >= info["frame_count"]:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Unreadable frame at t=%.2fs (idx=%d), skipping.", t, frame_idx)
                t += step
                continue

            # Convert BGR to RGB PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            # Resize keeping aspect ratio so longest side <= 960
            max_side = max(orig_w, orig_h)
            if max_side > 960:
                scale = 960.0 / float(max_side)
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

            pil_img.info["original_size"] = (orig_w, orig_h)
            setattr(pil_img, "original_size", (orig_w, orig_h))

            yield (round(t, 3), pil_img)
            t += step
    finally:
        cap.release()


def _extract_visual_events(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate visual anchor events from sequential frames with single-frame debounce."""
    if not frames:
        return []

    # 1. Debounce detector flicker (1-frame spikes/dips that immediately revert)
    debounced_people = [f["people_count"] for f in frames]
    debounced_weapons = [f["weapon_count"] for f in frames]

    for i in range(1, len(frames) - 1):
        if (
            debounced_people[i] != debounced_people[i - 1]
            and debounced_people[i + 1] == debounced_people[i - 1]
        ):
            debounced_people[i] = debounced_people[i - 1]
        if (
            debounced_weapons[i] != debounced_weapons[i - 1]
            and debounced_weapons[i + 1] == debounced_weapons[i - 1]
        ):
            debounced_weapons[i] = debounced_weapons[i - 1]

    events: List[Dict[str, Any]] = []
    crowd_formed = False
    active_weapon_event: Optional[Dict[str, Any]] = None

    for i in range(len(frames)):
        curr_t = float(frames[i]["t"])
        curr_p = debounced_people[i]
        curr_w = debounced_weapons[i]
        detections = frames[i]["detections"]

        # Derive confidence from detections
        conf_scores = [d.get("confidence", 0.7) for d in detections]
        frame_conf = float(np.mean(conf_scores)) if conf_scores else 0.7

        # Event: Crowd formed (first time people >= 3)
        if curr_p >= 3 and not crowd_formed:
            crowd_formed = True
            events.append({
                "category": "Other",
                "description": f"Crowd formed with {curr_p} individuals present",
                "start_raw": curr_t,
                "end_raw": curr_t,
                "confidence": round_conf(frame_conf),
            })

        if i > 0:
            prev_t = float(frames[i - 1]["t"])
            prev_p = debounced_people[i - 1]
            prev_w = debounced_weapons[i - 1]

            # Event: Person entered
            if curr_p > prev_p:
                events.append({
                    "category": "Arrival",
                    "description": f"Person entered scene (people count: {curr_p})",
                    "start_raw": curr_t,
                    "end_raw": curr_t,
                    "confidence": round_conf(frame_conf),
                })
            # Event: Person left
            elif curr_p < prev_p:
                events.append({
                    "category": "Departure",
                    "description": f"Person departed scene (people count: {curr_p})",
                    "start_raw": curr_t,
                    "end_raw": curr_t,
                    "confidence": round_conf(frame_conf),
                })

            # Event: Weapon appeared
            if prev_w == 0 and curr_w > 0:
                weapon_conf = [d.get("confidence", 0.7) for d in detections if d.get("is_weapon")]
                w_conf = float(np.mean(weapon_conf)) if weapon_conf else frame_conf
                ev = {
                    "category": "Threat",
                    "description": "Weapon appeared in scene",
                    "start_raw": curr_t,
                    "end_raw": curr_t,
                    "confidence": round_conf(w_conf),
                }
                events.append(ev)
                active_weapon_event = ev
            # Extend weapon presence while sustained
            elif curr_w > 0 and active_weapon_event is not None:
                active_weapon_event["end_raw"] = curr_t
            # Event: Weapon removed
            elif prev_w > 0 and curr_w == 0:
                events.append({
                    "category": "Other",
                    "description": "Weapon removed from scene",
                    "start_raw": curr_t,
                    "end_raw": curr_t,
                    "confidence": round_conf(frame_conf),
                })
                active_weapon_event = None

            # Event: Possible physical contact between two persons
            prev_persons = [d["box"] for d in frames[i - 1]["detections"] if d.get("label") == "Human"]
            curr_persons = [d["box"] for d in frames[i]["detections"] if d.get("label") == "Human"]

            if len(curr_persons) >= 2 and len(prev_persons) >= 2:
                for j in range(len(curr_persons)):
                    for k in range(j + 1, len(curr_persons)):
                        b1 = curr_persons[j]
                        b2 = curr_persons[k]
                        iou = _compute_iou(b1, b2)
                        if iou > 0.15:
                            curr_dist = _center_dist(b1, b2)
                            prev_dist = _center_dist(prev_persons[0], prev_persons[1])
                            if curr_dist < prev_dist:
                                events.append({
                                    "category": "Aggression",
                                    "description": "Possible physical contact between subjects",
                                    "start_raw": prev_t,
                                    "end_raw": curr_t,
                                    "confidence": round_conf(max(frame_conf, 0.75)),
                                })
                                break

    # Sort events by start_raw and assign sequential IDs
    events.sort(key=lambda e: (float(e["start_raw"]), float(e["end_raw"])))

    final_events: List[Dict[str, Any]] = []
    for idx, ev in enumerate(events, start=1):
        ev["id"] = f"V{idx}"
        ev["start_raw"] = round(float(ev["start_raw"]), 3)
        ev["end_raw"] = round(float(ev["end_raw"]), 3)
        ev["confidence"] = round_conf(ev["confidence"])
        final_events.append(ev)

    return final_events


def analyze_video(
    path: str | pathlib.Path,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Perform temporal video analysis extracting sampled frames, object counts, and visual events."""
    info = probe_video(path)
    width = info["width"]
    height = info["height"]
    fps = info["fps"]
    duration = min(info["duration"], float(config.MAX_VIDEO_SECONDS))

    hz = max(1, config.VIDEO_SAMPLE_HZ)
    expected_frames = max(1, int(round(duration * hz)))

    frames: List[Dict[str, Any]] = []
    done_frames = 0
    prev_counts: Optional[Tuple[int, int]] = None

    for frame_idx, (t, pil_img) in enumerate(
        sample_frames(path, hz=hz, max_seconds=config.MAX_VIDEO_SECONDS)
    ):
        # 1. Detection via analyze_pil_frame
        frame_analysis = analyze_pil_frame(pil_img)
        detections = frame_analysis["detections"]
        people_count = frame_analysis["people_count"]
        weapon_count = frame_analysis["weapon_count"]

        # 2. Firearm check every 5th frame or when people are detected
        if frame_idx % 5 == 0 or people_count > 0:
            try:
                firearm_check = check_firearm(pil_img)
                if firearm_check.get("firearm_likely", False):
                    detections.append({
                        "label": "Weapon",
                        "raw_label": "firearm (CLIP)",
                        "confidence": firearm_check["confidence"],
                        "box": [0.0, 0.0, float(width), float(height)],
                        "is_weapon": True,
                    })
                    weapon_count = sum(1 for d in detections if d.get("is_weapon", False))
            except Exception as exc:
                logger.warning("Video frame CLIP firearm check failed: %s", exc)

        # 3. Captioning conditionally: every N frames or on people/weapon count change
        counts_changed = (
            prev_counts is not None
            and (people_count != prev_counts[0] or weapon_count != prev_counts[1])
        )
        if frame_idx % max(1, config.CAPTION_EVERY_N_FRAMES) == 0 or counts_changed:
            try:
                caption: Optional[str] = caption_image(pil_img)
            except Exception as exc:
                logger.warning("Video frame captioning failed: %s", exc)
                caption = None
        else:
            caption = None

        prev_counts = (people_count, weapon_count)

        frames.append({
            "t": round(float(t), 3),
            "people_count": int(people_count),
            "weapon_count": int(weapon_count),
            "detections": detections,
            "caption": caption,
        })

        done_frames += 1
        if progress_cb is not None:
            try:
                progress_cb(done_frames, expected_frames)
            except Exception as exc:
                logger.warning("Progress callback raised exception: %s", exc)

    # 4. Extract visual anchor events
    events = _extract_visual_events(frames)

    # 5. Classify crime category
    crime_res = classify_crime(frames, events)

    result: Dict[str, Any] = {
        "duration": round(info["duration"], 3),
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "sample_rate_hz": hz,
        "frames": frames,
        "crime_type": crime_res["crime_type"],
        "crime_confidence": crime_res["crime_confidence"],
        "crime_probabilities": crime_res["crime_probabilities"],
        "events": events,
    }

    return to_jsonable(result)
