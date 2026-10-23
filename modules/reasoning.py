from __future__ import annotations

"""Forensic reasoning engine generating scenario hypotheses, event timelines, and LLM inferences."""

import json
import logging
from typing import Any, Dict, List

import config
from modules.llm import LLMError, chat_json, is_configured
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

