from __future__ import annotations

"""Multimodal timeline fusion engine for CrimeVision AI."""

import logging
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import config
from modules.reasoning import generate_fusion_narrative
from modules.utils import round_conf, to_jsonable

logger = logging.getLogger("crimevision.fusion")


def _normalise_events(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten and canonicalize events across video, audio, and image modalities."""
    raw_events: List[Dict[str, Any]] = []

    # 1. Video events
    video_data = session.get("video")
    if isinstance(video_data, dict):
        for idx, ev in enumerate(video_data.get("events", []), start=1):
            ev_id = str(ev.get("id") or f"V{idx}")
            st = round(max(0.0, float(ev.get("start_raw", 0.0))), 3)
            et = round(max(st, float(ev.get("end_raw", st))), 3)
            conf = round_conf(float(ev.get("confidence", 0.5)))
            raw_events.append({
                "id": ev_id,
                "start_raw": st,
                "end_raw": et,
                "modality": "video",
                "category": str(ev.get("category", "Other")),
                "description": str(ev.get("description", "")),
                "confidence": conf,
                "sources": [ev_id],
                "is_anchor": False,
                "conflict": False,
            })

    # 2. Audio events
    audio_data = session.get("audio")
    if isinstance(audio_data, dict):
        for idx, ev in enumerate(audio_data.get("events", []), start=1):
            ev_id = str(ev.get("id") or f"A{idx}")
            st = round(max(0.0, float(ev.get("start_raw", 0.0))), 3)
            et = round(max(st, float(ev.get("end_raw", st))), 3)
            conf = round_conf(float(ev.get("confidence", 0.5)))
            raw_events.append({
                "id": ev_id,
                "start_raw": st,
                "end_raw": et,
                "modality": "audio",
                "category": str(ev.get("category", "Other")),
                "description": str(ev.get("description", "")),
                "confidence": conf,
                "sources": [ev_id],
                "is_anchor": False,
                "conflict": False,
            })

    # 3. Image event (at t=0 summarising scene, people_count, weapon_count)
    image_data = session.get("image")
    if isinstance(image_data, dict):
        scene_info = image_data.get("scene_classification")
        scene_label = "Unknown"
        scene_conf = 0.5
        if isinstance(scene_info, dict):
            scene_label = str(scene_info.get("label", "Unknown"))
            scene_conf = float(scene_info.get("confidence", 0.5))
        p_cnt = int(image_data.get("people_count", 0))
        w_cnt = int(image_data.get("weapon_count", 0))
        caption = str(image_data.get("caption") or "").strip()

        desc = f"Image scene summary: {scene_label}, {p_cnt} people, {w_cnt} weapons"
        if caption:
            desc += f" ('{caption}')"

        raw_events.append({
            "id": "I1",
            "start_raw": 0.0,
            "end_raw": 0.0,
            "modality": "image",
            "category": "Scene Summary",
            "description": desc,
            "confidence": round_conf(scene_conf),
            "sources": ["I1"],
            "is_anchor": False,
            "conflict": False,
        })

    return raw_events


def _calculate_max_time(session: Dict[str, Any], events: List[Dict[str, Any]]) -> float:
    """Determine maximum timestamp across all modalities and events."""
    times: List[float] = [0.0]
    for ev in events:
        times.append(float(ev.get("end_raw", 0.0)))
    if isinstance(session.get("video"), dict):
        times.append(float(session["video"].get("duration", 0.0)))
    if isinstance(session.get("audio"), dict):
        times.append(float(session["audio"].get("duration", 0.0)))
    return max(times)


def _are_compatible_events(a_ev: Dict[str, Any], v_ev: Dict[str, Any]) -> Tuple[bool, str]:
    """Check if audio and video events have compatible forensic categories."""
    a_cat = str(a_ev.get("category", "")).strip()
    v_cat = str(v_ev.get("category", "")).strip()
    a_desc = str(a_ev.get("description", "")).lower()
    v_desc = str(v_ev.get("description", "")).lower()

    # Threat / Aggression / Weapon-related audio <-> video Weapon appeared / Possible physical contact / Threat
    weapon_words = ("weapon", "knife", "gun", "firearm", "threat", "blade", "shoot", "kill")
    is_a_threat = a_cat in ("Threat", "Aggression") or any(w in a_desc for w in weapon_words)
    is_v_threat = (
        v_cat in ("Threat", "Aggression")
        or "weapon appeared" in v_desc
        or "possible physical contact" in v_desc
        or any(w in v_desc for w in weapon_words)
    )

    if is_a_threat and is_v_threat:
        cat = "Threat" if (a_cat == "Threat" or v_cat == "Threat" or "weapon" in a_desc or "weapon" in v_desc) else "Aggression"
        return True, cat

    # Arrival <-> Person entered
    is_a_arrival = a_cat == "Arrival" or "entered" in a_desc or "arrived" in a_desc
    is_v_arrival = v_cat == "Arrival" or "person entered" in v_desc
    if is_a_arrival and is_v_arrival:
        return True, "Arrival"

    # Departure <-> Person left / Person departed
    is_a_departure = a_cat == "Departure" or "left" in a_desc or "departed" in a_desc or "fled" in a_desc
    is_v_departure = v_cat == "Departure" or "person departed" in v_desc or "person left" in v_desc
    if is_a_departure and is_v_departure:
        return True, "Departure"

    return False, ""


def _are_within_time_window(ev1: Dict[str, Any], ev2: Dict[str, Any], window_s: float = 3.0) -> bool:
    """Check if two time intervals are within +- window_s seconds of each other."""
    s1, e1 = float(ev1["start_raw"]), float(ev1["end_raw"])
    s2, e2 = float(ev2["start_raw"]), float(ev2["end_raw"])
    return s1 <= e2 + window_s and s2 <= e1 + window_s


def _corroborate_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Greedily corroborate compatible audio and video events into fused events."""
    audio_events = [ev for ev in raw_events if ev.get("modality") == "audio"]
    video_events = [ev for ev in raw_events if ev.get("modality") == "video"]

    candidate_pairs: List[Dict[str, Any]] = []
    for a_ev in audio_events:
        for v_ev in video_events:
            compat, fused_cat = _are_compatible_events(a_ev, v_ev)
            if compat and _are_within_time_window(a_ev, v_ev, window_s=3.0):
                ca = float(a_ev["confidence"])
                cv = float(v_ev["confidence"])
                combined_conf = min(0.98, 1.0 - (1.0 - ca) * (1.0 - cv))
                candidate_pairs.append({
                    "audio_ev": a_ev,
                    "video_ev": v_ev,
                    "category": fused_cat,
                    "combined_conf": combined_conf,
                })

    # Greedy match by highest combined confidence
    candidate_pairs.sort(key=lambda p: float(p["combined_conf"]), reverse=True)

    used_audio_ids: Set[str] = set()
    used_video_ids: Set[str] = set()
    fused_events: List[Dict[str, Any]] = []

    for pair in candidate_pairs:
        a_ev = pair["audio_ev"]
        v_ev = pair["video_ev"]
        a_id = a_ev["id"]
        v_id = v_ev["id"]
        if a_id in used_audio_ids or v_id in used_video_ids:
            continue

        used_audio_ids.add(a_id)
        used_video_ids.add(v_id)

        fused_id = f"F{len(fused_events) + 1}"
        st = round(min(float(a_ev["start_raw"]), float(v_ev["start_raw"])), 3)
        et = round(max(float(a_ev["end_raw"]), float(v_ev["end_raw"])), 3)
        conf = round_conf(pair["combined_conf"])
        desc = f"Corroborated: {v_ev['description']} and {a_ev['description']}"

        fused_events.append({
            "id": fused_id,
            "start_raw": st,
            "end_raw": et,
            "modality": "fused",
            "category": pair["category"],
            "description": desc,
            "confidence": conf,
            "sources": [a_id, v_id],
            "is_anchor": True,
            "conflict": False,
        })

    # Uncorroborated events receive single-source penalty (* 0.85)
    uncorroborated: List[Dict[str, Any]] = []
    for ev in raw_events:
        if ev["id"] in used_audio_ids or ev["id"] in used_video_ids:
            continue
        penalized = dict(ev)
        penalized["confidence"] = round_conf(float(ev["confidence"]) * 0.85)
        uncorroborated.append(penalized)

    return fused_events + uncorroborated


def _mark_anchor_events(all_events: List[Dict[str, Any]]) -> None:
    """Flag anchor events (fused, 'Weapon appeared', or Threat/Aggression with conf >= 0.6)."""
    for ev in all_events:
        if ev.get("modality") == "fused":
            ev["is_anchor"] = True
        elif "weapon appeared" in str(ev.get("description", "")).lower():
            ev["is_anchor"] = True
        elif ev.get("category") in ("Threat", "Aggression") and float(ev.get("confidence", 0.0)) >= 0.6:
            ev["is_anchor"] = True


def _detect_conflicts(
    session: Dict[str, Any],
    all_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect cross-modal discrepancies across weapon presence, people counts, and scene labels."""
    conflicts: List[Dict[str, Any]] = []
    conflicted_event_ids: Set[str] = set()

    video_data = session.get("video") if isinstance(session.get("video"), dict) else None
    audio_data = session.get("audio") if isinstance(session.get("audio"), dict) else None
    image_data = session.get("image") if isinstance(session.get("image"), dict) else None

    # (a) Audio mentions weapon/knife/gun but video has zero weapon detections in entire video
    if video_data and audio_data and video_data.get("frames"):
        total_video_weapons = sum(int(f.get("weapon_count", 0)) for f in video_data.get("frames", []))
        if total_video_weapons == 0:
            audio_text = str(audio_data.get("transcript", "")).lower()
            weapon_terms = ("weapon", "knife", "gun", "firearm", "blade", "pistol", "revolver")
            weapon_mentioned_in_text = any(w in audio_text for w in weapon_terms)

            weapon_audio_events = [
                ev for ev in all_events
                if ev.get("modality") in ("audio", "fused")
                and (any(w in ev.get("description", "").lower() for w in weapon_terms) or ev.get("category") == "Threat")
            ]

            if weapon_mentioned_in_text or weapon_audio_events:
                involved_ids = [ev["id"] for ev in weapon_audio_events]
                if not involved_ids:
                    involved_ids = [ev["id"] for ev in all_events if ev.get("modality") == "audio"]
                conflicts.append({
                    "type": "weapon_presence_conflict",
                    "description": "Audio explicitly mentions weapons (knife/gun/weapon), but video visual analysis detected zero weapons.",
                    "involved_events": involved_ids,
                    "severity": "high",
                })
                for eid in involved_ids:
                    conflicted_event_ids.add(eid)

    # (b) Audio mentions N people / count that differs from max video people_count by >= 2
    if video_data and audio_data and video_data.get("frames"):
        max_video_people = max([int(f.get("people_count", 0)) for f in video_data.get("frames", [])] or [0])
        audio_full_text = str(audio_data.get("transcript", "")).lower()
        for ev in audio_data.get("events", []):
            audio_full_text += " " + str(ev.get("description", "")).lower()

        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        extracted_counts: List[int] = []
        for m in re.finditer(r"\b(\d+)\s+(?:people|persons|men|women|individuals|subjects|suspects|attackers)\b", audio_full_text):
            extracted_counts.append(int(m.group(1)))
        for m in re.finditer(r"\b(one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:people|persons|men|women|individuals|subjects|suspects|attackers)\b", audio_full_text):
            extracted_counts.append(word_to_num[m.group(1)])

        for count in extracted_counts:
            if abs(count - max_video_people) >= 2:
                involved_ids = [
                    ev["id"] for ev in all_events
                    if ev.get("modality") in ("audio", "video", "fused")
                    and ("person" in ev.get("description", "").lower() or "people" in ev.get("description", "").lower())
                ]
                if not involved_ids:
                    involved_ids = [ev["id"] for ev in all_events if ev.get("modality") in ("audio", "video")]
                conflicts.append({
                    "type": "people_count_mismatch",
                    "description": f"Audio mentions {count} people, but video maximum detected person count is {max_video_people}.",
                    "involved_events": involved_ids,
                    "severity": "medium",
                })
                for eid in involved_ids:
                    conflicted_event_ids.add(eid)
                break

    # (c) Image scene label differs from audio location_inference label when both != Unknown and conf >= 0.5
    if image_data and audio_data:
        img_scene = image_data.get("scene_classification")
        img_label = "Unknown"
        img_conf = 0.0
        if isinstance(img_scene, dict):
            img_label = str(img_scene.get("label", "Unknown")).strip()
            img_conf = float(img_scene.get("confidence", 0.0))

        aud_facts = audio_data.get("facts")
        aud_label = "Unknown"
        aud_conf = 0.0
        if isinstance(aud_facts, dict):
            loc_inf = aud_facts.get("location_inference")
            if isinstance(loc_inf, dict):
                aud_label = str(loc_inf.get("label", "Unknown")).strip()
                aud_conf = float(loc_inf.get("confidence", 0.0))

        if (
            img_label != "Unknown"
            and aud_label != "Unknown"
            and img_label.lower() != aud_label.lower()
            and img_conf >= 0.5
            and aud_conf >= 0.5
        ):
            involved_ids = [ev["id"] for ev in all_events if ev.get("modality") in ("image", "audio")]
            conflicts.append({
                "type": "scene_location_conflict",
                "description": f"Image classified scene as '{img_label}' (conf: {round_conf(img_conf)}) but audio inferred location as '{aud_label}' (conf: {round_conf(aud_conf)}).",
                "involved_events": involved_ids,
                "severity": "medium",
            })
            for eid in involved_ids:
                conflicted_event_ids.add(eid)

    # Each conflict reduces confidence of involved events by 0.15 (floor 0.05)
    for ev in all_events:
        if ev["id"] in conflicted_event_ids:
            ev["conflict"] = True
            ev["confidence"] = round_conf(max(0.05, float(ev["confidence"]) - 0.15))

    return conflicts


def _assemble_timeline(
    session: Dict[str, Any],
    all_events: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]], List[str], float]:
    """Assemble sorted timeline, integer second clock index, anchor IDs, and overall confidence."""
    timeline = sorted(all_events, key=lambda e: (float(e.get("start_raw", 0.0)), str(e.get("id", ""))))

    max_t = _calculate_max_time(session, timeline)
    max_sec = int(math.ceil(max_t))
    second_index: Dict[str, List[str]] = {}

    for s in range(0, max_sec + 1):
        s_f = float(s)
        active_ids: List[str] = []
        for ev in timeline:
            st = float(ev["start_raw"])
            et = float(ev["end_raw"])
            # Active if overlapping [s, s + 1.0] or within 0.5s point proximity
            if (max(st, s_f) <= min(et, s_f + 1.0)) or (st <= s_f <= et) or (abs(st - s_f) < 0.5):
                active_ids.append(str(ev["id"]))
        second_index[str(s)] = active_ids

    anchor_events = [str(ev["id"]) for ev in timeline if ev.get("is_anchor")]

    anchor_evs = [ev for ev in timeline if ev.get("is_anchor")]
    target_evs = anchor_evs if anchor_evs else timeline
    if target_evs:
        weights = [float(ev["confidence"]) for ev in target_evs]
        sum_w = sum(weights)
        if sum_w > 0.0:
            overall_confidence = round_conf(sum(w * w for w in weights) / sum_w)
        else:
            overall_confidence = 0.0
    else:
        overall_confidence = 0.0

    return timeline, second_index, anchor_events, overall_confidence


def fuse_session(session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Execute multimodal timeline fusion returning the canonical 'fusion' schema block.

    Deterministic Python pipeline with graceful fallback and per-step exception handling.
    """
    warnings: List[str] = []
    safe_session = session if isinstance(session, dict) else {}

    # Step 1: Normalise
    raw_events: List[Dict[str, Any]] = []
    try:
        raw_events = _normalise_events(safe_session)
    except Exception as exc:
        warnings.append(f"Normalisation warning: {exc}")
        logger.warning("Normalisation failed: %s", exc)

    # Step 3: Cross-modal corroboration
    all_events: List[Dict[str, Any]] = []
    try:
        all_events = _corroborate_events(raw_events)
    except Exception as exc:
        warnings.append(f"Corroboration warning: {exc}")
        logger.warning("Corroboration failed: %s", exc)
        all_events = raw_events

    # Step 4: Anchor event identification
    try:
        _mark_anchor_events(all_events)
    except Exception as exc:
        warnings.append(f"Anchor selection warning: {exc}")
        logger.warning("Anchor selection failed: %s", exc)

    # Step 5: Conflict detection & penalties
    conflicts: List[Dict[str, Any]] = []
    try:
        conflicts = _detect_conflicts(safe_session, all_events)
    except Exception as exc:
        warnings.append(f"Conflict detection warning: {exc}")
        logger.warning("Conflict detection failed: %s", exc)

    # Step 6: Output assembly
    timeline: List[Dict[str, Any]] = []
    second_index: Dict[str, List[str]] = {}
    anchor_events: List[str] = []
    overall_confidence: float = 0.0
    try:
        timeline, second_index, anchor_events, overall_confidence = _assemble_timeline(safe_session, all_events)
    except Exception as exc:
        warnings.append(f"Timeline assembly warning: {exc}")
        logger.warning("Timeline assembly failed: %s", exc)
        timeline = all_events
        second_index = {"0": [str(ev.get("id")) for ev in timeline if ev.get("id")]}
        anchor_events = [str(ev["id"]) for ev in timeline if ev.get("is_anchor")]
        overall_confidence = 0.0

    # Step 7: Narrative summary and hypotheses
    narrative: Dict[str, Any] = {}
    try:
        fusion_core = {
            "timeline": timeline,
            "second_index": second_index,
            "anchor_events": anchor_events,
            "conflicts": conflicts,
            "overall_confidence": overall_confidence,
        }
        session_without_fusion = {k: v for k, v in safe_session.items() if k != "fusion"}
        narrative = generate_fusion_narrative(session_without_fusion, fusion_core)
    except Exception as exc:
        warnings.append(f"Narrative generation warning: {exc}")
        logger.warning("Narrative generation failed: %s", exc)
        narrative = {
            "hypotheses": [
                {"title": "Forensic Hypothesis 1", "explanation": "Primary hypothesis based on available evidence.", "probability": 0.5},
                {"title": "Forensic Hypothesis 2", "explanation": "Alternative hypothesis based on available evidence.", "probability": 0.3},
                {"title": "Forensic Hypothesis 3", "explanation": "Uncertain scenario due to incomplete evidence.", "probability": 0.2},
            ],
            "summary": "Forensic analysis generated with warnings.",
        }

    return to_jsonable({
        "timeline": timeline,
        "second_index": second_index,
        "anchor_events": anchor_events,
        "conflicts": conflicts,
        "overall_confidence": overall_confidence,
        "hypotheses": narrative.get("hypotheses", []),
        "summary": narrative.get("summary", ""),
        "warnings": warnings,
    })
