from __future__ import annotations

"""Grounded conversational chat engine and evidence indexer for CrimeVision AI."""

import json
import logging
import math
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from modules import reasoning
from modules.session_store import load_session, update_session
from modules.utils import to_jsonable

logger = logging.getLogger("crimevision.chat")

_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other",
    "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
}


def _tokenize(text: str) -> List[str]:
    """Extract lowercased alphanumeric keywords excluding common stopwords."""
    if not text:
        return []
    words = re.findall(r"[a-zA-Z0-9_\-]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in _STOPWORDS]


def _overlaps(start: float, end: float, sec: int) -> bool:
    """Check if an interval [start, end] overlaps with integer second [sec, sec + 1.0]."""
    s_f = float(sec)
    e_f = s_f + 1.0
    return max(start, s_f) < min(end, e_f) or (start <= s_f <= end) or (abs(start - s_f) < 0.5)


def build_evidence_index(session: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a fast, searchable multimodal evidence index from existing session data.

    Returns:
        {
            "by_second": {
                "<int>": {
                    "video": {
                        "people_count": int,
                        "weapon_count": int,
                        "detections": list of labels,
                        "caption": str | None,
                    } | None,
                    "audio": [segments/events overlapping that second],
                    "events": [event_id_strings],
                }
            },
            "facts": [short forensic summary strings],
            "keywords": {keyword: [second_ints]},
        }
    """
    if not session or not isinstance(session, dict):
        return {"by_second": {}, "facts": [], "keywords": {}}

    video_data = session.get("video") if isinstance(session.get("video"), dict) else None
    audio_data = session.get("audio") if isinstance(session.get("audio"), dict) else None
    image_data = session.get("image") if isinstance(session.get("image"), dict) else None

    # Collect all relevant seconds across modalities
    seconds_set: Set[int] = set()

    # Video frames and events
    frame_map: Dict[int, Dict[str, Any]] = {}
    video_events: List[Dict[str, Any]] = []
    if video_data:
        for f in video_data.get("frames", []):
            t_sec = int(round(float(f.get("t", 0.0))))
            seconds_set.add(t_sec)
            frame_map[t_sec] = f

        for ev in video_data.get("events", []):
            video_events.append(ev)
            st = int(math.floor(float(ev.get("start_raw", 0.0))))
            et = int(math.ceil(float(ev.get("end_raw", st))))
            for s in range(st, et + 1):
                seconds_set.add(s)

    # Audio segments and events
    audio_segments: List[Dict[str, Any]] = []
    audio_events: List[Dict[str, Any]] = []
    if audio_data:
        for seg in audio_data.get("segments", []):
            audio_segments.append(seg)
            st = int(math.floor(float(seg.get("start_raw", 0.0))))
            et = int(math.ceil(float(seg.get("end_raw", st))))
            for s in range(st, et + 1):
                seconds_set.add(s)

        for ev in audio_data.get("events", []):
            audio_events.append(ev)
            st = int(math.floor(float(ev.get("start_raw", 0.0))))
            et = int(math.ceil(float(ev.get("end_raw", st))))
            for s in range(st, et + 1):
                seconds_set.add(s)

    # Keyword indexer
    keywords_map: Dict[str, Set[int]] = defaultdict(set)

    def _index_keyword(kw: str, sec: int) -> None:
        clean = kw.strip().lower()
        if len(clean) > 1 and clean not in _STOPWORDS:
            keywords_map[clean].add(sec)

    # Build by_second mapping
    by_second: Dict[str, Any] = {}
    sorted_seconds = sorted(seconds_set)

    for sec in sorted_seconds:
        # Video payload for second
        video_payload: Optional[Dict[str, Any]] = None
        if video_data:
            frame = frame_map.get(sec)
            if frame is not None:
                p_cnt = int(frame.get("people_count", 0))
                w_cnt = int(frame.get("weapon_count", 0))
                cap = frame.get("caption")
                dets = frame.get("detections", [])

                det_summary: List[str] = []
                for d in dets:
                    raw_lbl = str(d.get("raw_label") or d.get("label") or "").strip()
                    if raw_lbl:
                        det_summary.append(raw_lbl)
                        _index_keyword(raw_lbl, sec)

                if p_cnt > 0:
                    _index_keyword("person", sec)
                    _index_keyword("people", sec)
                    _index_keyword("human", sec)

                if w_cnt > 0:
                    _index_keyword("weapon", sec)

                if cap:
                    for tok in _tokenize(cap):
                        _index_keyword(tok, sec)

                video_payload = {
                    "people_count": p_cnt,
                    "weapon_count": w_cnt,
                    "detections": det_summary,
                    "caption": cap,
                }

        # Audio payload for second
        audio_payload: List[Dict[str, Any]] = []
        if audio_data:
            for seg in audio_segments:
                st = float(seg.get("start_raw", 0.0))
                et = float(seg.get("end_raw", st))
                if _overlaps(st, et, sec):
                    audio_payload.append({
                        "id": seg.get("id"),
                        "type": "segment",
                        "text": seg.get("text", ""),
                        "start_raw": round(st, 3),
                        "end_raw": round(et, 3),
                    })
                    for tok in _tokenize(seg.get("text", "")):
                        _index_keyword(tok, sec)

            for a_ev in audio_events:
                st = float(a_ev.get("start_raw", 0.0))
                et = float(a_ev.get("end_raw", st))
                if _overlaps(st, et, sec):
                    audio_payload.append({
                        "id": a_ev.get("id"),
                        "type": "event",
                        "category": a_ev.get("category", "Other"),
                        "description": a_ev.get("description", ""),
                        "who": a_ev.get("who", "Unknown"),
                        "where": a_ev.get("where", "Unknown"),
                        "confidence": a_ev.get("confidence", 0.5),
                        "start_raw": round(st, 3),
                        "end_raw": round(et, 3),
                    })
                    _index_keyword(a_ev.get("category", ""), sec)
                    for tok in _tokenize(a_ev.get("description", "")):
                        _index_keyword(tok, sec)

        # Overlapping visual and audio events IDs
        overlapping_event_ids: List[str] = []
        for v_ev in video_events:
            st = float(v_ev.get("start_raw", 0.0))
            et = float(v_ev.get("end_raw", st))
            if _overlaps(st, et, sec):
                overlapping_event_ids.append(str(v_ev.get("id")))
                _index_keyword(v_ev.get("category", ""), sec)
                for tok in _tokenize(v_ev.get("description", "")):
                    _index_keyword(tok, sec)

        for a_ev in audio_events:
            st = float(a_ev.get("start_raw", 0.0))
            et = float(a_ev.get("end_raw", st))
            if _overlaps(st, et, sec):
                ev_id = str(a_ev.get("id"))
                if ev_id not in overlapping_event_ids:
                    overlapping_event_ids.append(ev_id)

        by_second[str(sec)] = {
            "video": video_payload,
            "audio": audio_payload,
            "events": overlapping_event_ids,
        }

    # Index global crime type keywords
    if video_data and video_data.get("crime_type"):
        c_type = str(video_data.get("crime_type"))
        for tok in _tokenize(c_type):
            for sec in sorted_seconds:
                _index_keyword(tok, sec)

    # Convert keywords to sorted lists
    keywords_final: Dict[str, List[int]] = {
        kw: sorted(list(secs)) for kw, secs in keywords_map.items() if secs
    }

    # Build concise forensic facts
    facts: List[str] = []

    # 1. Video frame-level facts
    if video_data:
        frames = video_data.get("frames", [])
        for f in frames:
            sec = int(round(float(f.get("t", 0.0))))
            p = int(f.get("people_count", 0))
            w = int(f.get("weapon_count", 0))
            cap = f.get("caption")

            # Extract weapon specific labels
            weapon_names: List[str] = []
            for d in f.get("detections", []):
                if d.get("is_weapon") or str(d.get("label")).lower() == "weapon":
                    name = str(d.get("raw_label") or d.get("label") or "weapon")
                    if name.lower() not in [wn.lower() for wn in weapon_names]:
                        weapon_names.append(name)

            p_str = f"{p} people" if p != 1 else "1 person"
            w_str = f"{w} weapon" if w == 1 else f"{w} weapons"
            wn_str = f" ({', '.join(weapon_names)})" if weapon_names else ""

            if w > 0:
                facts.append(f"At {sec}s: {p_str}, {w_str}{wn_str}")
            elif cap:
                facts.append(f"At {sec}s: {p_str} - {cap}")
            elif p > 0:
                facts.append(f"At {sec}s: {p_str} detected")

        # 2. Visual anchor events
        for v_ev in video_data.get("events", []):
            eid = v_ev.get("id", "V")
            cat = v_ev.get("category", "Visual")
            st = round(float(v_ev.get("start_raw", 0.0)), 1)
            et = round(float(v_ev.get("end_raw", st)), 1)
            desc = v_ev.get("description", "")
            t_span = f"{int(st)}s" if st == et else f"{st}s-{et}s"
            facts.append(f"At {t_span} [{eid}]: {desc} ({cat})")

        # 3. Video crime classification
        if video_data.get("crime_type"):
            ct = video_data.get("crime_type")
            conf = video_data.get("crime_confidence", 0.0)
            facts.append(f"Video crime classification: {ct} ({round(float(conf) * 100, 1)}%)")

    # 4. Audio anchor events
    if audio_data:
        for a_ev in audio_data.get("events", []):
            eid = a_ev.get("id", "A")
            cat = a_ev.get("category", "Audio")
            st = round(float(a_ev.get("start_raw", 0.0)), 1)
            et = round(float(a_ev.get("end_raw", st)), 1)
            desc = a_ev.get("description", "")
            who = a_ev.get("who", "Unknown")
            where = a_ev.get("where", "Unknown")

            extra_parts: List[str] = []
            if who != "Unknown":
                extra_parts.append(f"who: {who}")
            if where != "Unknown":
                extra_parts.append(f"where: {where}")
            extra_str = f" [{', '.join(extra_parts)}]" if extra_parts else ""

            t_span = f"{int(st)}s" if st == et else f"{st}s-{et}s"
            facts.append(f"At {t_span} [{eid}]: {desc} ({cat}){extra_str}")

        # 5. Audio forensic facts
        a_facts = audio_data.get("facts")
        if isinstance(a_facts, dict):
            who = str(a_facts.get("who", "Unknown")).strip()
            where = str(a_facts.get("where", "Unknown")).strip()
            when = str(a_facts.get("when", "Unknown")).strip()
            suspect = str(a_facts.get("suspect_description", "Unknown")).strip()

            if who and who != "Unknown":
                facts.append(f"Audio subject: {who}")
            if where and where != "Unknown":
                facts.append(f"Audio location: {where}")
            if when and when != "Unknown":
                facts.append(f"Audio time reference: {when}")
            if suspect and suspect != "Unknown":
                facts.append(f"Audio suspect description: {suspect}")

            loc_inf = a_facts.get("location_inference")
            if isinstance(loc_inf, dict) and loc_inf.get("label") and loc_inf.get("label") != "Unknown":
                facts.append(f"Audio inferred location: {loc_inf.get('label')}")

        if audio_data.get("transcript"):
            tr = str(audio_data.get("transcript")).strip()
            if tr:
                short_tr = tr if len(tr) <= 100 else tr[:97] + "..."
                facts.append(f"Audio transcript: \"{short_tr}\"")

    # 6. Image evidence
    if image_data:
        scene = image_data.get("scene_classification", {}).get("label")
        if scene:
            facts.append(f"Image scene: {scene}")
        if image_data.get("caption"):
            facts.append(f"Image caption: {image_data.get('caption')}")
        img_p = int(image_data.get("people_count", 0))
        img_w = int(image_data.get("weapon_count", 0))
        if img_p > 0 or img_w > 0:
            facts.append(f"Image detections: {img_p} people, {img_w} weapons")

    # Deduplicate facts preserving order
    deduped_facts: List[str] = []
    seen_facts: Set[str] = set()
    for f in facts:
        if f not in seen_facts:
            seen_facts.add(f)
            deduped_facts.append(f)

    return {
        "by_second": by_second,
        "facts": deduped_facts,
        "keywords": keywords_final,
    }


def classify_intent(message: str) -> Dict[str, Any]:
    """Classify message intent into 'time' (with timestamp) or 'general'."""
    if not message or not isinstance(message, str):
        return {"type": "general", "t": None}

    text = message.strip()

    # Pattern 1: MM:SS or M:SS (e.g. 00:05, 1:30)
    mm_ss = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if mm_ss:
        mins = int(mm_ss.group(1))
        secs = int(mm_ss.group(2))
        return {"type": "time", "t": float(mins * 60 + secs)}

    # Pattern 2: at t=7, t=7.5
    t_eq = re.search(r"\b(?:at\s+)?t\s*=\s*(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
    if t_eq:
        return {"type": "time", "t": float(t_eq.group(1))}

    # Pattern 3: second 12, sec 5
    sec_prefix = re.search(r"\b(?:second|sec)\s+(\d+(?:\.\d+)?)\b", text, re.IGNORECASE)
    if sec_prefix:
        return {"type": "time", "t": float(sec_prefix.group(1))}

    # Pattern 4: at 5s, 5 sec, 12 seconds
    sec_suffix = re.search(r"\b(?:at\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\b", text, re.IGNORECASE)
    if sec_suffix:
        return {"type": "time", "t": float(sec_suffix.group(1))}

    return {"type": "general", "t": None}


def _get_session_duration(session: Dict[str, Any]) -> float:
    """Retrieve maximum duration across available modalities."""
    durations = [0.0]
    if isinstance(session.get("video"), dict):
        durations.append(float(session["video"].get("duration", 0.0)))
    if isinstance(session.get("audio"), dict):
        durations.append(float(session["audio"].get("duration", 0.0)))
    if isinstance(session.get("fusion"), dict):
        for ev in session["fusion"].get("timeline", []):
            durations.append(float(ev.get("end_raw", 0.0)))
    return max(durations)


def answer_time_query(session: Dict[str, Any], t: float) -> Dict[str, Any]:
    """Retrieve grounded forensic facts for time window [floor(t)-1 .. floor(t)+1]."""
    safe_session = session if isinstance(session, dict) else {}
    max_duration = _get_session_duration(safe_session)

    # Check if t is beyond the evidence duration
    if max_duration > 0 and (t < 0.0 or t > max_duration + 1.0):
        out_of_bounds_msg = (
            f"The requested timestamp ({t:.1f}s) is beyond the recorded evidence duration "
            f"(0.0s - {max_duration:.1f}s)."
        )
        return {
            "answer": out_of_bounds_msg,
            "sources": [],
            "facts": [],
        }

    index = build_evidence_index(safe_session)
    by_second = index.get("by_second", {})

    t_floor = int(math.floor(t))
    window_seconds = [s for s in range(max(0, t_floor - 1), t_floor + 2)]

    raw_facts: List[str] = []
    sources: Set[str] = set()

    for sec in window_seconds:
        sec_key = str(sec)
        entry = by_second.get(sec_key)
        if not entry:
            continue

        # Video facts
        v = entry.get("video")
        if v:
            p_cnt = int(v.get("people_count", 0))
            w_cnt = int(v.get("weapon_count", 0))
            dets = v.get("detections", [])
            cap = v.get("caption")

            p_str = f"{p_cnt} people" if p_cnt != 1 else "1 person"
            w_str = f"{w_cnt} weapons" if w_cnt != 1 else f"{w_cnt} weapon"
            det_str = f" ({', '.join(dets)})" if dets else ""

            if w_cnt > 0:
                raw_facts.append(f"At {sec}s (video): {p_str}, {w_str}{det_str}")
            elif cap:
                raw_facts.append(f"At {sec}s (video): {p_str} - {cap}")
            elif p_cnt > 0:
                raw_facts.append(f"At {sec}s (video): {p_str} detected")

        # Audio facts
        for a in entry.get("audio", []):
            if a.get("type") == "segment" and a.get("text"):
                raw_facts.append(f"At {sec}s (audio): \"{a.get('text')}\"")
            elif a.get("type") == "event" and a.get("description"):
                raw_facts.append(f"At {sec}s (audio event [{a.get('id')}]): {a.get('description')}")

        # Active events
        for eid in entry.get("events", []):
            sources.add(str(eid))

    if not raw_facts:
        raw_facts.append(f"At {t:.1f}s: No specific movements or events recorded in this time window.")

    sources.add(f"{t:.1f}s")

    # Phrasing: LLM if configured, else formatted string
    answer_text = reasoning.phrase_time_facts(raw_facts, t)

    return {
        "answer": answer_text,
        "sources": sorted(list(sources)),
        "facts": raw_facts,
    }


def _build_compact_context(session: Dict[str, Any]) -> str:
    """Build a compact forensic JSON context capped at ~12k characters, preferring fusion data."""
    compact: Dict[str, Any] = {}

    # 1. Fusion (highest priority)
    if isinstance(session.get("fusion"), dict):
        f = session["fusion"]
        compact["fusion"] = {
            "summary": f.get("summary"),
            "hypotheses": f.get("hypotheses"),
            "conflicts": f.get("conflicts"),
            "timeline": [
                {
                    "id": ev.get("id"),
                    "t": f"{ev.get('start_raw')}s-{ev.get('end_raw')}s",
                    "category": ev.get("category"),
                    "description": ev.get("description"),
                    "confidence": ev.get("confidence"),
                    "modality": ev.get("modality"),
                    "is_anchor": ev.get("is_anchor"),
                }
                for ev in f.get("timeline", [])[:25]
            ],
            "overall_confidence": f.get("overall_confidence"),
        }

    # 2. Image
    if isinstance(session.get("image"), dict):
        img = session["image"]
        compact["image"] = {
            "scene": img.get("scene_classification", {}).get("label") if isinstance(img.get("scene_classification"), dict) else None,
            "caption": img.get("caption"),
            "people_count": img.get("people_count"),
            "weapon_count": img.get("weapon_count"),
            "hypotheses": img.get("hypotheses"),
        }

    # 3. Audio
    if isinstance(session.get("audio"), dict):
        aud = session["audio"]
        tr = str(aud.get("transcript", ""))
        compact["audio"] = {
            "language": aud.get("language"),
            "duration": aud.get("duration"),
            "transcript": tr[:3000] if len(tr) > 3000 else tr,
            "facts": aud.get("facts"),
            "events": [
                {"id": ev.get("id"), "category": ev.get("category"), "description": ev.get("description"), "t": f"{ev.get('start_raw')}s-{ev.get('end_raw')}s"}
                for ev in aud.get("events", [])[:15]
            ],
        }

    # 4. Video
    if isinstance(session.get("video"), dict):
        vid = session["video"]
        frames_summary = []
        for f in vid.get("frames", []):
            frames_summary.append({
                "t": f.get("t"),
                "people": f.get("people_count"),
                "weapons": f.get("weapon_count"),
                "caption": f.get("caption"),
            })
        compact["video"] = {
            "duration": vid.get("duration"),
            "crime_type": vid.get("crime_type"),
            "crime_confidence": vid.get("crime_confidence"),
            "events": [
                {"id": ev.get("id"), "category": ev.get("category"), "description": ev.get("description"), "t": f"{ev.get('start_raw')}s-{ev.get('end_raw')}s"}
                for ev in vid.get("events", [])[:15]
            ],
            "frames_sample": frames_summary[:30],
        }

    json_str = json.dumps(to_jsonable(compact), indent=1)
    if len(json_str) > 12000:
        if "video" in compact and "frames_sample" in compact["video"]:
            compact["video"]["frames_sample"] = compact["video"]["frames_sample"][:10]
            json_str = json.dumps(to_jsonable(compact), indent=1)
        if len(json_str) > 12000:
            json_str = json_str[:12000] + "\n...[truncated]"

    return json_str


def answer_general(
    session: Dict[str, Any],
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Answer a general forensic inquiry with grounded LLM or deterministic keyword search."""
    safe_session = session if isinstance(session, dict) else {}
    context_str = _build_compact_context(safe_session)

    if reasoning.is_configured():
        try:
            answer = reasoning.answer_forensic_question(context_str, message, history)
            # Extract cited event IDs and timestamps from answer
            sources: List[str] = []
            for m in re.finditer(r"\b([VAF]\d+|I1)\b", answer):
                sources.append(m.group(1))
            for m in re.finditer(r"\[(\d+(?:\.\d+)?s)\]", answer):
                sources.append(m.group(1))
            return {
                "answer": answer,
                "sources": sorted(list(set(sources))),
            }
        except Exception as exc:
            logger.warning("General query LLM failed (%s), falling back to keyword search", exc)

    # Fallback without LLM: keyword search over index facts and fusion summary
    index = build_evidence_index(safe_session)
    facts = index.get("facts", [])
    query_tokens = _tokenize(message)

    matched_facts: List[Tuple[int, str]] = []
    for f in facts:
        f_lower = f.lower()
        score = sum(1 for tok in query_tokens if tok in f_lower)
        if score > 0:
            matched_facts.append((score, f))

    matched_facts.sort(key=lambda item: item[0], reverse=True)

    if matched_facts:
        top_facts = [f for _, f in matched_facts[:5]]
        answer_text = "Based on analysed forensic evidence:\n" + "\n".join(f"- {f}" for f in top_facts)
        sources = []
        for f in top_facts:
            for m in re.finditer(r"\[([VAF]\d+|I1)\]", f):
                sources.append(m.group(1))
        return {
            "answer": answer_text,
            "sources": sorted(list(set(sources))),
        }

    return {
        "answer": "The analysed evidence does not contain that information.",
        "sources": [],
    }


def chat(
    session_id: str,
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Process a conversational query against a persisted forensic session."""
    if not message or not isinstance(message, str) or not message.strip():
        raise ValueError("Message cannot be empty")

    session = load_session(session_id)
    if session is None:
        raise KeyError(f"Session '{session_id}' not found")

    intent = classify_intent(message)

    if intent["type"] == "time" and intent.get("t") is not None:
        res = answer_time_query(session, float(intent["t"]))
    else:
        chat_hist = history if history is not None else session.get("chat_history", [])
        res = answer_general(session, message, chat_hist)

    # Update session chat history
    session_history = list(session.get("chat_history", []))
    session_history.append({"role": "user", "content": message.strip()})
    session_history.append({
        "role": "assistant",
        "content": res["answer"],
        "intent": intent,
        "sources": res.get("sources", []),
    })

    update_session(session_id, chat_history=session_history)

    return {
        "answer": res["answer"],
        "intent": intent,
        "sources": res.get("sources", []),
    }

