from __future__ import annotations

"""Forensic reasoning engine generating scenario hypotheses, event timelines, and LLM inferences."""

import json
import logging
from typing import Any, Dict, List, Set

import config
from modules.llm import LLMError, chat_json, chat_text, is_configured
from modules.utils import round_conf, safe_float

logger = logging.getLogger("crimevision.reasoning")


# ============================================================================
# --- IMAGE REASONING & HYPOTHESIS GENERATION ---
# ============================================================================

def _compute_relative_position(box: List[float], width: int, height: int) -> str:
    """Derive relative spatial position (e.g., 'left, near') from bounding box and frame bounds."""
    w = max(1, width)
    h = max(1, height)
    center_x = (box[0] + box[2]) / 2.0
    bottom_y = box[3]

    if center_x < w / 3.0:
        h_pos = "left"
    elif center_x > (2.0 * w) / 3.0:
        h_pos = "right"
    else:
        h_pos = "center"

    if bottom_y > (2.0 * h) / 3.0:
        v_pos = "near"
    elif bottom_y < h / 3.0:
        v_pos = "far"
    else:
        v_pos = "mid-ground"

    return f"{h_pos}, {v_pos}"


def _build_compact_image_evidence(image_result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract a concise evidence summary for LLM prompt context."""
    width = int(image_result.get("width", 1000))
    height = int(image_result.get("height", 1000))

    scene_label = image_result.get("scene", {}).get("label", "unknown")
    people_count = int(image_result.get("people_count", 0))
    weapon_count = int(image_result.get("weapon_count", 0))
    caption = image_result.get("caption", "")

    detections = image_result.get("detections", [])
    weapon_labels = [
        d.get("raw_label", "weapon")
        for d in detections
        if d.get("is_weapon", False)
    ]

    key_detections = []
    for d in detections:
        box = d.get("box", [0.0, 0.0, 0.0, 0.0])
        pos = _compute_relative_position(box, width, height)
        key_detections.append({
            "label": d.get("label", "Object"),
            "raw_label": d.get("raw_label", "object"),
            "confidence": d.get("confidence", 0.0),
            "position": pos,
        })

    return {
        "scene": scene_label,
        "people_count": people_count,
        "weapon_count": weapon_count,
        "weapons": weapon_labels,
        "caption": caption,
        "key_detections": key_detections,
    }


def _rule_based_image_hypotheses(
    image_result: Dict[str, Any],
    reason: str = "rule-based fallback",
) -> List[Dict[str, Any]]:
    """Generate 3 deterministic, rule-based forensic hypotheses when LLM is unavailable."""
    people = int(image_result.get("people_count", 0))
    weapons = int(image_result.get("weapon_count", 0))
    scene = str(image_result.get("scene", {}).get("label", "scene"))
    caption = str(image_result.get("caption", "visual evidence"))

    if weapons > 0 and people >= 2:
        return [
            {
                "title": "Armed Confrontation",
                "explanation": (
                    f"Multiple subjects ({people}) and weapon indicators present at {scene}. "
                    f"Evidence indicates an active armed confrontation or standoff ({reason})."
                ),
                "probability": 0.5,
            },
            {
                "title": "Defensive Struggle",
                "explanation": (
                    f"Weapon presence in {scene} with {people} individuals suggests a defensive "
                    f"intervention or struggle over a weapon ({reason})."
                ),
                "probability": 0.3,
            },
            {
                "title": "Staged / Non-Criminal Scenario",
                "explanation": (
                    f"Posture in {scene} ({caption}) may represent lawful training, theatrical display, "
                    f"or non-violent possession ({reason})."
                ),
                "probability": 0.2,
            },
        ]
    elif weapons > 0:
        return [
            {
                "title": "Armed Threat or Trespass",
                "explanation": (
                    f"Weapon indicators in {scene} with {people} subject(s) indicate armed presence "
                    f"or hostile positioning ({reason})."
                ),
                "probability": 0.5,
            },
            {
                "title": "Weapon Abandonment / Stashing",
                "explanation": (
                    f"Weapon observed in {scene} without an aggressive group indicates possible "
                    f"contraband abandonment or concealment ({reason})."
                ),
                "probability": 0.3,
            },
            {
                "title": "Lawful Weapon Possession",
                "explanation": (
                    f"Weapon presence in {scene} may reflect lawful security, hunting equipment, "
                    f"or benign tool usage ({reason})."
                ),
                "probability": 0.2,
            },
        ]
    elif people >= 2:
        return [
            {
                "title": "Unarmed Dispute or Altercation",
                "explanation": (
                    f"Presence of {people} individuals in {scene} without visible weapons indicates "
                    f"potential verbal dispute or physical altercation ({reason})."
                ),
                "probability": 0.5,
            },
            {
                "title": "Routine Gathering",
                "explanation": (
                    f"Group of {people} people observed at {scene} ({caption}), consistent with "
                    f"normal lawful interaction or transit ({reason})."
                ),
                "probability": 0.3,
            },
            {
                "title": "Unauthorized Loitering or Trespassing",
                "explanation": (
                    f"Multiple persons grouped in {scene} could indicate trespassing, illicit meetup, "
                    f"or reconnaissance ({reason})."
                ),
                "probability": 0.2,
            },
        ]
    elif people == 1:
        return [
            {
                "title": "Solo Trespassing or Intrusion",
                "explanation": (
                    f"Single person observed in {scene} without visible weapon, consistent with "
                    f"unauthorized entry or scouting ({reason})."
                ),
                "probability": 0.5,
            },
            {
                "title": "Authorized Solo Activity",
                "explanation": (
                    f"Single individual in {scene} ({caption}), consistent with lawful routine occupancy "
                    f"or bystander presence ({reason})."
                ),
                "probability": 0.3,
            },
            {
                "title": "Undetermined Subject Activity",
                "explanation": (
                    f"Single subject in {scene} lacks overt criminal indicators; motives remain "
                    f"unverified ({reason})."
                ),
                "probability": 0.2,
            },
        ]
    else:
        return [
            {
                "title": "Routine Unoccupied Scene",
                "explanation": (
                    f"No human or weapon indicators detected in {scene}. Consistent with normal "
                    f"inactive conditions ({reason})."
                ),
                "probability": 0.5,
            },
            {
                "title": "Post-Incident Vacated Site",
                "explanation": (
                    f"Unoccupied {scene} may indicate that involved parties have already fled or departed "
                    f"prior to capture ({reason})."
                ),
                "probability": 0.3,
            },
            {
                "title": "Inconclusive Visual Evidence",
                "explanation": (
                    f"Scene {scene} shows neither subjects nor contraband; physical forensic evidence "
                    f"is required ({reason})."
                ),
                "probability": 0.2,
            },
        ]


def generate_image_hypotheses(image_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate exactly 3 distinct forensic hypotheses for an analyzed image.

    Utilizes Groq LLM if configured, otherwise falls back gracefully to a deterministic rule-based set.
    """
    if not is_configured():
        logger.info("GROQ_API_KEY not configured. Using rule-based fallback for image hypotheses.")
        return _rule_based_image_hypotheses(image_result, reason="rule-based fallback: LLM unconfigured")

    evidence = _build_compact_image_evidence(image_result)
    system_prompt = (
        "You are a forensic analyst. Use ONLY the supplied evidence. "
        "Never invent objects, people, or events not in the evidence. "
        'Output JSON: {"hypotheses":[{"title":str,"explanation":str,"probability":float}, ...exactly 3]} '
        "with probabilities summing to 1.0. Hypotheses must be distinct (e.g., defensive struggle, "
        "planned tactical entry, non-criminal/benign scenario) and each explanation <= 60 words, "
        "citing which evidence supports it."
    )
    user_prompt = f"Analyze the following evidence and generate 3 hypotheses:\n{json.dumps(evidence, indent=2)}"

    try:
        response = chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=800,
        )

        raw_hypotheses = response.get("hypotheses", [])
        if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) < 3:
            raise LLMError(f"LLM returned invalid hypotheses count: {len(raw_hypotheses) if isinstance(raw_hypotheses, list) else 0}")

        # Validate titles and extract probabilities
        valid_items = []
        raw_probs = []
        for h in raw_hypotheses[:3]:
            title = str(h.get("title", "")).strip()
            explanation = str(h.get("explanation", "")).strip()
            if not title:
                raise LLMError("Hypothesis title cannot be empty.")
            prob = safe_float(h.get("probability", 0.0), 0.0)
            valid_items.append({"title": title, "explanation": explanation})
            raw_probs.append(prob)

        total_prob = sum(raw_probs)
        if total_prob <= 0.0:
            norm_probs = [0.5, 0.3, 0.2]
        else:
            norm_probs = [p / total_prob for p in raw_probs]

        p0 = round_conf(norm_probs[0])
        p1 = round_conf(norm_probs[1])
        p2 = round_conf(max(0.0, 1.0 - p0 - p1))

        # Ensure exact 1.0 sum
        final_probs = [p0, p1, round(1.0 - p0 - p1, 3)]

        hypotheses = [
            {
                "title": valid_items[i]["title"],
                "explanation": valid_items[i]["explanation"],
                "probability": float(final_probs[i]),
            }
            for i in range(3)
        ]
        return hypotheses

    except Exception as exc:
        logger.warning("LLM hypothesis generation failed (%s). Falling back to rule-based engine.", exc)
        return _rule_based_image_hypotheses(image_result, reason=f"rule-based fallback: LLM error ({exc})")


# ============================================================================
# --- AUDIO REASONING & FACT EXTRACTION ---
# ============================================================================

def _rule_based_audio_facts(
    transcript: str,
    reason: str = "rule-based fallback",
) -> Dict[str, Any]:
    """Provide deterministic forensic facts and hypotheses for audio evidence."""
    clean_text = transcript.strip()

    if not clean_text:
        return {
            "who": "Unknown",
            "where": "Unknown",
            "when": "Unknown",
            "suspect_description": "Unknown",
            "location_inference": {
                "label": "Unknown",
                "confidence": 0.0,
                "evidence": f"No speech detected in audio stream ({reason})",
            },
            "hypotheses": [
                {
                    "title": "Inaudible / Ambient Audio",
                    "explanation": f"No discernible speech detected in recording ({reason}).",
                    "probability": 0.5,
                },
                {
                    "title": "Silent Monitoring Interval",
                    "explanation": f"Acoustic channel contains background ambiance without spoken interactions ({reason}).",
                    "probability": 0.3,
                },
                {
                    "title": "Acoustic Interference / Signal Tone",
                    "explanation": f"Input waveform contains synthetic or non-vocal acoustic artifacts ({reason}).",
                    "probability": 0.2,
                },
            ],
        }

    # If transcript has speech
    return {
        "who": "Unknown",
        "where": "Unknown",
        "when": "Unknown",
        "suspect_description": "Unknown",
        "location_inference": {
            "label": "Unknown",
            "confidence": 0.0,
            "evidence": f"Transcript lacks explicit localized scene identifiers ({reason})",
        },
        "hypotheses": [
            {
                "title": "Reported Incident / Emergency Call",
                "explanation": f"Spoken content indicates situational reporting or verbal engagement ({reason}).",
                "probability": 0.5,
            },
            {
                "title": "Witness or Subject Inquiries",
                "explanation": f"Dialogue represents interview, inquiry, or operational dispatch ({reason}).",
                "probability": 0.3,
            },
            {
                "title": "Routine Non-Criminal Conversation",
                "explanation": f"Verbal dialogue is consistent with everyday communication without confirmed threat ({reason}).",
                "probability": 0.2,
            },
        ],
    }


def extract_audio_facts(
    transcript: str,
    segments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Extract structured forensic facts and scenario hypotheses from an audio transcript."""
    clean_text = transcript.strip()
    if not clean_text:
        return _rule_based_audio_facts(clean_text, reason="empty audio transcript")

    if not is_configured():
        logger.info("GROQ_API_KEY not configured. Using rule-based fallback for audio facts.")
        return _rule_based_audio_facts(clean_text, reason="rule-based fallback: LLM unconfigured")

    system_prompt = (
        "You are a forensic linguist. The transcript may be Telugu, Hindi, French or English. "
        "Use ONLY the transcript. If a field is not stated, output \"Unknown\" — never guess. "
        "Output JSON: {\"who\": str, \"where\": str, \"when\": str, \"suspect_description\": str, "
        "\"location_inference\": {\"label\": one of config.SCENE_LABELS or \"Unknown\", \"confidence\": 0-1, "
        "\"evidence\": short quote/paraphrase}, \"hypotheses\": [exactly 3 {title, explanation, probability}]}. "
        "Answer text fields in English."
    )

    segment_sample = [
        {"t": f"{s.get('start_raw')}s-{s.get('end_raw')}s", "text": s.get("text")}
        for s in segments[:15]
    ]

    user_payload = {
        "transcript": clean_text,
        "segments": segment_sample,
        "allowed_scene_labels": config.SCENE_LABELS + ["Unknown"],
    }
    user_prompt = f"Analyze the following audio evidence:\n{json.dumps(user_payload, indent=2)}"

    try:
        response = chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=900,
        )

        who = str(response.get("who", "Unknown")).strip() or "Unknown"
        where = str(response.get("where", "Unknown")).strip() or "Unknown"
        when = str(response.get("when", "Unknown")).strip() or "Unknown"
        suspect = str(response.get("suspect_description", "Unknown")).strip() or "Unknown"

        raw_loc = response.get("location_inference", {})
        if not isinstance(raw_loc, dict):
            raw_loc = {}

        loc_label = str(raw_loc.get("label", "Unknown")).strip()
        if loc_label not in config.SCENE_LABELS:
            loc_label = "Unknown"

        loc_conf = round_conf(safe_float(raw_loc.get("confidence", 0.0), 0.0))
        loc_evidence = str(raw_loc.get("evidence", "None")).strip() or "None"

        location_inference = {
            "label": loc_label,
            "confidence": loc_conf,
            "evidence": loc_evidence,
        }

        # Hypotheses validation & normalization
        raw_hypo = response.get("hypotheses", [])
        if not isinstance(raw_hypo, list) or len(raw_hypo) < 3:
            raise LLMError("LLM returned fewer than 3 hypotheses for audio.")

        valid_items = []
        raw_probs = []
        for h in raw_hypo[:3]:
            title = str(h.get("title", "")).strip()
            explanation = str(h.get("explanation", "")).strip()
            if not title:
                raise LLMError("Audio hypothesis title cannot be empty.")
            prob = safe_float(h.get("probability", 0.0), 0.0)
            valid_items.append({"title": title, "explanation": explanation})
            raw_probs.append(prob)

        total_prob = sum(raw_probs)
        if total_prob <= 0.0:
            norm_probs = [0.5, 0.3, 0.2]
        else:
            norm_probs = [p / total_prob for p in raw_probs]

        p0 = round_conf(norm_probs[0])
        p1 = round_conf(norm_probs[1])
        p2 = round_conf(max(0.0, 1.0 - p0 - p1))
        final_probs = [p0, p1, round(1.0 - p0 - p1, 3)]

        hypotheses = [
            {
                "title": valid_items[i]["title"],
                "explanation": valid_items[i]["explanation"],
                "probability": float(final_probs[i]),
            }
            for i in range(3)
        ]

        return {
            "who": who,
            "where": where,
            "when": when,
            "suspect_description": suspect,
            "location_inference": location_inference,
            "hypotheses": hypotheses,
        }

    except Exception as exc:
        logger.warning("Audio facts extraction failed (%s). Falling back to rule-based engine.", exc)
        return _rule_based_audio_facts(clean_text, reason=f"rule-based fallback: LLM error ({exc})")


AUDIO_EVENT_CATEGORIES: Set[str] = {
    "Aggression",
    "Identification",
    "Spatial Movement",
    "Threat",
    "Arrival",
    "Departure",
    "Other",
}


def _dedupe_and_merge_audio_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge events with the same category that overlap or are within 1.0s."""
    if not raw_events:
        return []

    sorted_events = sorted(
        raw_events,
        key=lambda e: (float(e.get("start_raw", 0.0)), float(e.get("end_raw", 0.0))),
    )

    merged: List[Dict[str, Any]] = []
    for ev in sorted_events:
        start = max(0.0, float(ev.get("start_raw", 0.0)))
        end = max(start, float(ev.get("end_raw", start)))
        cat = str(ev.get("category", "Other")).strip()
        if cat not in AUDIO_EVENT_CATEGORIES:
            cat = "Other"

        item = {
            "category": cat,
            "description": str(ev.get("description", "")).strip(),
            "who": str(ev.get("who", "Unknown")).strip() or "Unknown",
            "where": str(ev.get("where", "Unknown")).strip() or "Unknown",
            "confidence": round_conf(safe_float(ev.get("confidence", 0.4), 0.4)),
            "start_raw": start,
            "end_raw": end,
        }

        matched_idx = -1
        for idx, prev in enumerate(merged):
            if prev["category"] == item["category"]:
                # Check overlap or gap <= 1.0s
                if item["start_raw"] <= prev["end_raw"] + 1.0 and prev["start_raw"] <= item["end_raw"] + 1.0:
                    matched_idx = idx
                    break

        if matched_idx != -1:
            prev = merged[matched_idx]
            prev["start_raw"] = min(prev["start_raw"], item["start_raw"])
            prev["end_raw"] = max(prev["end_raw"], item["end_raw"])
            prev["confidence"] = max(prev["confidence"], item["confidence"])
            if len(item["description"]) > len(prev["description"]):
                prev["description"] = item["description"]
            if prev["who"] == "Unknown" and item["who"] != "Unknown":
                prev["who"] = item["who"]
            if prev["where"] == "Unknown" and item["where"] != "Unknown":
                prev["where"] = item["where"]
        else:
            merged.append(item)

    merged.sort(key=lambda e: float(e["start_raw"]))

    final_events: List[Dict[str, Any]] = []
    for i, ev in enumerate(merged, start=1):
        words = ev["description"].split()
        if len(words) > 25:
            ev["description"] = " ".join(words[:25])

        ev["id"] = f"A{i}"
        ev["confidence"] = round_conf(ev["confidence"])
        ev["start_raw"] = round(float(ev["start_raw"]), 3)
        ev["end_raw"] = round(float(ev["end_raw"]), 3)
        final_events.append(ev)

    return final_events


def _fallback_audio_events(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keyword-based deterministic fallback classifying forensic events across languages."""
    raw_events: List[Dict[str, Any]] = []

    threat_kw = {"knife", "gun", "kill", "shoot", "threat", "weapon", "बंदूक", "चाकू", "मारूंगा", "couteau", "arme", "tuer", "tupaki", "kathi"}
    aggression_kw = {"hit", "push", "beat", "stab", "fight", "attack", "मारा", "पीटा", "frappé", "attaqué", "kottu"}
    arrival_kw = {"came", "entered", "arrived", "inside", "आया", "घुसा", "entré", "venu", "vachadu"}
    departure_kw = {"ran", "fled", "escaped", "left", "outside", "भाग", "भागा", "fui", "parti", "poyadu"}
    other_kw = {"help", "police", "ambulance", "मदद", "पुलिस", "aide", "sahayam"}

    for s in segments:
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        text_lower = text.lower()
        words = text.split()
        desc = " ".join(words[:25])
        start_raw = float(s.get("start_raw", 0.0))
        end_raw = float(s.get("end_raw", start_raw))

        matched = False
        if any(kw in text_lower for kw in threat_kw):
            raw_events.append({
                "category": "Threat",
                "description": desc,
                "who": "Unknown",
                "where": "Unknown",
                "confidence": 0.4,
                "start_raw": start_raw,
                "end_raw": end_raw,
            })
            matched = True
        if any(kw in text_lower for kw in aggression_kw):
            raw_events.append({
                "category": "Aggression",
                "description": desc,
                "who": "Unknown",
                "where": "Unknown",
                "confidence": 0.4,
                "start_raw": start_raw,
                "end_raw": end_raw,
            })
            matched = True
        if any(kw in text_lower for kw in arrival_kw):
            raw_events.append({
                "category": "Arrival",
                "description": desc,
                "who": "Unknown",
                "where": "Unknown",
                "confidence": 0.4,
                "start_raw": start_raw,
                "end_raw": end_raw,
            })
            matched = True
        if any(kw in text_lower for kw in departure_kw):
            raw_events.append({
                "category": "Departure",
                "description": desc,
                "who": "Unknown",
                "where": "Unknown",
                "confidence": 0.4,
                "start_raw": start_raw,
                "end_raw": end_raw,
            })
            matched = True
        if not matched and any(kw in text_lower for kw in other_kw):
            raw_events.append({
                "category": "Other",
                "description": desc,
                "who": "Unknown",
                "where": "Unknown",
                "confidence": 0.4,
                "start_raw": start_raw,
                "end_raw": end_raw,
            })

    return _dedupe_and_merge_audio_events(raw_events)


def extract_audio_events(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and chronologically anchor forensic audio events from transcript segments."""
    if not segments:
        return []

    if not is_configured():
        logger.info("GROQ_API_KEY not configured. Using keyword fallback for audio events.")
        return _fallback_audio_events(segments)

    truncated_segments = segments[:120]
    segment_lines = []
    for idx, s in enumerate(truncated_segments):
        start = s.get("start_raw", 0.0)
        end = s.get("end_raw", 0.0)
        txt = s.get("text", "")
        segment_lines.append(f"[{idx}] {start}s-{end}s: {txt}")

    system_prompt = (
        "Classify forensic events from timestamped witness-statement segments. "
        "Use ONLY the segments. Categories: Aggression, Identification, Spatial Movement, Threat, "
        "Arrival, Departure, Other. "
        'Output JSON {"events":[{"segment_indices":[int], "category": str, "description": str, '
        '"who": str, "where": str, "confidence": float}]}. '
        "Description <= 25 words in English. "
        "who or where default to 'Unknown'. "
        "Only include meaningful events (threats, physical struggle, suspect mentions, arrivals, weapon mentions). "
        "Do not invent events."
    )
    user_prompt = "Classify events from the following segments:\n" + "\n".join(segment_lines)

    try:
        response = chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=1000,
        )

        raw_events_llm = response.get("events", [])
        if not isinstance(raw_events_llm, list):
            raise LLMError("LLM output missing valid events list.")

        parsed_events: List[Dict[str, Any]] = []
        num_segments = len(truncated_segments)

        for ev in raw_events_llm:
            indices = ev.get("segment_indices", [])
            if not isinstance(indices, list):
                continue
            valid_indices = [idx for idx in indices if isinstance(idx, int) and 0 <= idx < num_segments]
            if not valid_indices:
                continue

            ref_segments = [truncated_segments[idx] for idx in valid_indices]
            start_raw = min(float(s.get("start_raw", 0.0)) for s in ref_segments)
            end_raw = max(float(s.get("end_raw", start_raw)) for s in ref_segments)

            cat = str(ev.get("category", "Other")).strip()
            if cat not in AUDIO_EVENT_CATEGORIES:
                cat = "Other"

            words = str(ev.get("description", "")).strip().split()
            desc = " ".join(words[:25]) if words else "Audio forensic event"

            who = str(ev.get("who", "Unknown")).strip() or "Unknown"
            where = str(ev.get("where", "Unknown")).strip() or "Unknown"
            conf = round_conf(safe_float(ev.get("confidence", 0.5), 0.5))

            parsed_events.append({
                "category": cat,
                "description": desc,
                "who": who,
                "where": where,
                "confidence": conf,
                "start_raw": start_raw,
                "end_raw": end_raw,
            })

        return _dedupe_and_merge_audio_events(parsed_events)

    except Exception as exc:
        logger.warning("LLM audio events extraction failed (%s). Falling back to keyword engine.", exc)
        return _fallback_audio_events(segments)


# ============================================================================
# --- VIDEO REASONING & CRIME CLASSIFICATION ---
# ============================================================================

def _build_compact_video_sequence(
    video_frames: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> str:
    """Build a concise, grouped sequence of frame observations and events (<= 150 lines)."""
    if not video_frames:
        return "No video frames available."

    grouped: List[Dict[str, Any]] = []
    for f in video_frames:
        t = float(f.get("t", 0.0))
        p = int(f.get("people_count", 0))
        w = int(f.get("weapon_count", 0))
        cap = str(f.get("caption") or "").strip()

        if (
            grouped
            and grouped[-1]["p"] == p
            and grouped[-1]["w"] == w
            and grouped[-1]["cap"] == cap
        ):
            grouped[-1]["end_t"] = t
        else:
            grouped.append({"start_t": t, "end_t": t, "p": p, "w": w, "cap": cap})

    lines: List[str] = []
    for item in grouped[:130]:
        st = int(round(item["start_t"]))
        et = int(round(item["end_t"]))
        t_str = f"t={st}s" if st == et else f"t={st}-{et}s"
        cap_str = f' caption="{item["cap"]}"' if item["cap"] else ""
        lines.append(f"{t_str} people={item['p']} weapons={item['w']}{cap_str}")

    if events:
        lines.append("\nVisual Anchor Events:")
        for ev in events[:20]:
            ev_id = ev.get("id", "V")
            cat = ev.get("category", "Other")
            st = ev.get("start_raw", 0.0)
            et = ev.get("end_raw", st)
            desc = ev.get("description", "")
            lines.append(f"[{ev_id}] {st}s-{et}s ({cat}): {desc}")

    return "\n".join(lines[:150])


def _rule_based_crime_classification(
    video_frames: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
    reason: str = "rule-based fallback",
) -> Dict[str, Any]:
    """Heuristic fallback for crime taxonomy classification."""
    weapon_seconds = sum(1 for f in video_frames if f.get("weapon_count", 0) > 0)
    has_two_people = any(f.get("people_count", 0) >= 2 for f in video_frames)

    has_physical_contact = any(
        "possible physical contact" in str(ev.get("description", "")).lower()
        or "physical contact" in str(ev.get("description", "")).lower()
        or ev.get("category") == "Aggression"
        for ev in events
    )

    if weapon_seconds >= 2 and has_two_people:
        primary = "Armed Threat"
        primary_prob = 0.5
        expl = f"Weapon sustained for >=2s with multiple persons ({reason})"
    elif has_physical_contact:
        primary = "Assault"
        primary_prob = 0.5
        expl = f"Possible physical contact or aggression detected between subjects ({reason})"
    else:
        primary = "No Crime Detected"
        primary_prob = 0.6
        expl = f"No sustained weapons or physical contact observed ({reason})"

    remaining_mass = 1.0 - primary_prob
    other_labels = [c for c in config.CRIME_TAXONOMY if c != primary]
    per_other = remaining_mass / float(len(other_labels))

    probs: Dict[str, float] = {}
    for cat in config.CRIME_TAXONOMY:
        if cat == primary:
            probs[cat] = round_conf(primary_prob)
        else:
            probs[cat] = round_conf(per_other)

    diff = round(1.0 - sum(probs.values()), 3)
    if diff != 0.0:
        probs[primary] = round_conf(probs[primary] + diff)

    return {
        "crime_type": primary,
        "crime_confidence": probs[primary],
        "crime_probabilities": probs,
        "reasoning": expl,
    }


def classify_crime(
    video_frames: List[Dict[str, Any]],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Classify video sequence against crime taxonomy using LLM with deterministic fallback."""
    if not is_configured():
        logger.info("GROQ_API_KEY not configured. Using rule-based fallback for crime classification.")
        return _rule_based_crime_classification(
            video_frames, events, reason="rule-based fallback: LLM unconfigured"
        )

    sequence_text = _build_compact_video_sequence(video_frames, events)
    system_prompt = (
        "You are a forensic classifier. Based ONLY on the sequence, estimate probabilities over "
        f"exactly this taxonomy: {config.CRIME_TAXONOMY}. Prefer 'No Crime Detected' unless there is clear "
        'supporting evidence (weapon, physical contact, forced entry, taking items). '
        'Output JSON {"probabilities": {label: float}, "primary": label, "reasoning": str}. '
        'reasoning <= 40 words.'
    )
    user_prompt = f"Analyze this sequence:\n{sequence_text}"

    try:
        response = chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=800,
        )

        raw_probs = response.get("probabilities", {})
        if not isinstance(raw_probs, dict):
            raise LLMError("LLM response missing probabilities dictionary.")

        cleaned_probs: Dict[str, float] = {}
        for cat in config.CRIME_TAXONOMY:
            cleaned_probs[cat] = safe_float(raw_probs.get(cat, 0.0), 0.0)

        total_prob = sum(cleaned_probs.values())
        if total_prob <= 0.0:
            cleaned_probs["No Crime Detected"] = 1.0
            total_prob = 1.0

        norm_probs: Dict[str, float] = {}
        for cat in config.CRIME_TAXONOMY:
            norm_probs[cat] = round_conf(cleaned_probs[cat] / total_prob)

        diff = round(1.0 - sum(norm_probs.values()), 3)
        max_k = max(norm_probs, key=lambda k: norm_probs[k])
        norm_probs[max_k] = round_conf(norm_probs[max_k] + diff)

        primary = max(norm_probs, key=lambda k: norm_probs[k])
        confidence = norm_probs[primary]

        reasoning = str(response.get("reasoning", "")).strip()
        words = reasoning.split()
        if len(words) > 40:
            reasoning = " ".join(words[:40])

        return {
            "crime_type": primary,
            "crime_confidence": confidence,
            "crime_probabilities": norm_probs,
            "reasoning": reasoning or f"Classified as {primary}",
        }

    except Exception as exc:
        logger.warning("Crime classification LLM failed (%s). Falling back to rule-based engine.", exc)
        return _rule_based_crime_classification(
            video_frames, events, reason=f"rule-based fallback: LLM error ({exc})"
        )


# ============================================================================
# --- FUSION REASONING & NARRATIVE GENERATION ---
# ============================================================================

def _rule_based_fusion_narrative(
    session_without_fusion: Dict[str, Any],
    fusion_core: Dict[str, Any],
    reason: str = "rule-based fallback",
) -> Dict[str, Any]:
    """Deterministic fallback for fusion narrative and hypotheses."""
    timeline = fusion_core.get("timeline", [])
    anchor_ids = set(fusion_core.get("anchor_events", []))
    anchors = [ev for ev in timeline if ev.get("id") in anchor_ids or ev.get("is_anchor")]

    anchors_sorted = sorted(anchors, key=lambda e: float(e.get("start_raw", 0.0)))
    top_5 = anchors_sorted[:5]
    if not top_5 and timeline:
        top_5 = sorted(timeline, key=lambda e: float(e.get("start_raw", 0.0)))[:5]

    if top_5:
        parts = [f"At {ev.get('start_raw')}s: {ev.get('description')}" for ev in top_5]
        summary = ". ".join(parts) + "."
    else:
        summary = "No significant forensic events detected across recorded modalities."

    conflicts = fusion_core.get("conflicts", [])
    if conflicts:
        c_types = ", ".join(c.get("type", "discrepancy") for c in conflicts)
        summary += f" Noted evidence discrepancies: {c_types}."

    words = summary.split()
    if len(words) > 120:
        summary = " ".join(words[:120])

    has_threat = any(
        ev.get("category") in ("Threat", "Aggression")
        or "weapon" in ev.get("description", "").lower()
        or "contact" in ev.get("description", "").lower()
        for ev in timeline
    )

    if has_threat:
        hypotheses = [
            {
                "title": "Escalated Confrontation with Threat Indicators",
                "explanation": f"Corroborated multimodal timeline indicates active threat or confrontation ({reason}).",
                "probability": 0.5,
            },
            {
                "title": "Premeditated Aggression or Ambush",
                "explanation": f"Rapid presentation of weapons or physical contact suggests intentional confrontation ({reason}).",
                "probability": 0.3,
            },
            {
                "title": "Defensive Interaction during Non-Lethal Dispute",
                "explanation": f"Actions observed may represent reactive defense rather than unprovoked criminal hostility ({reason}).",
                "probability": 0.2,
            },
        ]
    else:
        hypotheses = [
            {
                "title": "Routine Non-Criminal Activity",
                "explanation": f"Observed timeline exhibits normal activity without verified weapons or physical force ({reason}).",
                "probability": 0.5,
            },
            {
                "title": "Incident Prior or Subsequent to Recording",
                "explanation": f"Subjects entered or departed scene without overt violence during captured interval ({reason}).",
                "probability": 0.3,
            },
            {
                "title": "Inconclusive Multimodal Monitoring",
                "explanation": f"Recorded sensor streams lack sufficient interaction details to confirm criminal conduct ({reason}).",
                "probability": 0.2,
            },
        ]

    return {
        "hypotheses": hypotheses,
        "summary": summary,
    }


def generate_fusion_narrative(
    session_without_fusion: Dict[str, Any],
    fusion_core: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate forensic narrative summary (<=120 words) and exactly 3 hypotheses summing to 1.0."""
    if not is_configured():
        logger.info("GROQ_API_KEY not configured. Using rule-based fallback for fusion narrative.")
        return _rule_based_fusion_narrative(
            session_without_fusion, fusion_core, reason="rule-based fallback: LLM unconfigured"
        )

    timeline = fusion_core.get("timeline", [])
    conflicts = fusion_core.get("conflicts", [])
    overall_conf = fusion_core.get("overall_confidence", 0.0)

    event_lines = []
    for ev in timeline[:30]:
        eid = ev.get("id", "")
        mod = ev.get("modality", "")
        st = ev.get("start_raw", 0.0)
        et = ev.get("end_raw", st)
        cat = ev.get("category", "")
        desc = ev.get("description", "")
        conf = ev.get("confidence", 0.0)
        anchor = " [ANCHOR]" if ev.get("is_anchor") else ""
        event_lines.append(f"[{eid}] ({mod}, {cat}, conf={conf}){anchor} {st}s-{et}s: {desc}")

    conflict_lines = []
    for c in conflicts[:10]:
        conflict_lines.append(f"- {c.get('type')}: {c.get('description')}")

    evidence_payload = {
        "overall_confidence": overall_conf,
        "timeline_sample": event_lines,
        "conflicts": conflict_lines,
    }

    system_prompt = (
        "You are a forensic analyst. Use ONLY the provided timeline, conflicts and evidence. "
        "Do not invent facts. If evidence is insufficient say so. "
        'Output JSON: {"hypotheses": [{"title": str, "explanation": str, "probability": float}, '
        '...exactly 3], "summary": str}. '
        "The 3 hypotheses must have probabilities summing to 1.0. "
        "The summary must be <= 120 words in English."
    )
    user_prompt = f"Analyze the multimodal fusion evidence:\n{json.dumps(evidence_payload, indent=2)}"

    try:
        response = chat_json(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=900,
        )

        raw_hypotheses = response.get("hypotheses", [])
        if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) < 3:
            raise LLMError(f"LLM returned invalid hypotheses count: {len(raw_hypotheses) if isinstance(raw_hypotheses, list) else 0}")

        valid_items = []
        raw_probs = []
        for h in raw_hypotheses[:3]:
            title = str(h.get("title", "")).strip()
            explanation = str(h.get("explanation", "")).strip()
            if not title:
                raise LLMError("Hypothesis title cannot be empty.")
            prob = safe_float(h.get("probability", 0.0), 0.0)
            valid_items.append({"title": title, "explanation": explanation})
            raw_probs.append(prob)

        total_prob = sum(raw_probs)
        if total_prob <= 0.0:
            norm_probs = [0.5, 0.3, 0.2]
        else:
            norm_probs = [p / total_prob for p in raw_probs]

        p0 = round_conf(norm_probs[0])
        p1 = round_conf(norm_probs[1])
        p2 = round_conf(max(0.0, 1.0 - p0 - p1))
        final_probs = [p0, p1, round(1.0 - p0 - p1, 3)]

        hypotheses = [
            {
                "title": valid_items[i]["title"],
                "explanation": valid_items[i]["explanation"],
                "probability": float(final_probs[i]),
            }
            for i in range(3)
        ]

        summary = str(response.get("summary", "")).strip()
        words = summary.split()
        if len(words) > 120:
            summary = " ".join(words[:120])

        if not summary:
            fallback = _rule_based_fusion_narrative(session_without_fusion, fusion_core)
            summary = fallback["summary"]

        return {
            "hypotheses": hypotheses,
            "summary": summary,
        }
    except Exception as exc:
        logger.warning("Fusion narrative LLM failed (%s). Falling back to rule-based engine.", exc)
        return _rule_based_fusion_narrative(
            session_without_fusion, fusion_core, reason=f"rule-based fallback: LLM error ({exc})"
        )


# ============================================================================
# --- CHAT & GROUNDED FORENSIC QA ---
# ============================================================================

CHAT_SYSTEM_PROMPT = (
    "You are CrimeVision's forensic assistant. Answer ONLY from the EVIDENCE JSON below. "
    "If the answer is not in the evidence, reply exactly: 'The analysed evidence does not contain that information.' "
    "Do not speculate beyond the three hypotheses provided, and label them as hypotheses. "
    "Cite timestamps like [5.0s] when available. Be concise."
)


def phrase_time_facts(facts: List[str], t: float) -> str:
    """Phrase raw temporal facts into 2-3 concise sentences without inventing facts."""
    if not facts:
        return f"No significant forensic activity recorded around {t:.1f}s."

    fallback_text = f"At {t:.1f}s:\n" + "\n".join(f"- {f}" for f in facts)

    if not is_configured():
        return fallback_text

    system_prompt = (
        f"You are CrimeVision's forensic assistant. Phrase the following timestamped facts around {t:.1f}s "
        "into 2-3 concise sentences. Do NOT invent, assume, or add any new facts, people, or objects."
    )
    user_prompt = f"Facts around {t:.1f}s:\n" + "\n".join(facts)

    try:
        response = chat_text(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        return response.strip() or fallback_text
    except Exception as exc:
        logger.warning("LLM time facts phrasing failed: %s", exc)
        return fallback_text


def answer_forensic_question(
    evidence_json_str: str,
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Answer a forensic chat query strictly grounded in the provided evidence JSON."""
    if not is_configured():
        raise LLMError("LLM unconfigured")

    system_message = f"{CHAT_SYSTEM_PROMPT}\n\nEVIDENCE JSON:\n{evidence_json_str}"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_message}
    ]

    # Include the last 6 history turns
    if history and isinstance(history, list):
        for turn in history[-6:]:
            role = "user" if turn.get("role") == "user" else "assistant"
            content = str(turn.get("content") or turn.get("message") or "")
            if content:
                messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    response = chat_text(
        messages=messages,
        temperature=0.1,
        max_tokens=600,
    )
    return response.strip()





