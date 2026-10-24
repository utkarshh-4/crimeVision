from __future__ import annotations

"""Audio analysis module implementing Whisper ASR, language detection, and forensic fact extraction."""

import logging
import pathlib
import re
from typing import Any, Dict, List, Tuple

import librosa
import numpy as np

import config
from modules.model_registry import get_whisper
from modules.reasoning import extract_audio_events, extract_audio_facts
from modules.utils import to_jsonable

logger = logging.getLogger("crimevision.audio")


def load_audio(path: str | pathlib.Path) -> Tuple[np.ndarray, int]:
    """Load and validate an audio file at 16000 Hz mono float32, capped at 600 seconds."""
    audio_path = pathlib.Path(path)
    if not audio_path.is_file():
        raise ValueError(f"Audio file does not exist: {audio_path}")

    try:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    except Exception as exc:
        raise ValueError(f"Failed to read or decode audio file at '{audio_path}': {exc}") from exc

    if len(audio) == 0:
        raise ValueError(f"Audio file at '{audio_path}' is empty.")

    duration = float(len(audio)) / 16000.0
    if duration <= 0.2:
        raise ValueError(f"Audio file at '{audio_path}' is too short ({duration:.2f}s <= 0.2s).")

    max_samples = 600 * 16000
    if len(audio) > max_samples:
        logger.warning("Audio exceeds 600 seconds (%.2fs); capping at 600s.", duration)
        audio = audio[:max_samples]

    return audio.astype(np.float32), 16000


def _is_likely_speech(audio: np.ndarray, sr: int = 16000) -> bool:
    """Lightweight acoustic pre-check to distinguish vocal audio from pure tones or silence."""
    if len(audio) == 0:
        return False

    # Check 1: Silence threshold
    rms = librosa.feature.rms(y=audio)
    if float(np.mean(rms)) < 0.005:
        return False

    # Check 2: Pure tone check (very low spectral flatness and static spectral centroid)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=audio)))
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    std_centroid = float(np.std(centroid))

    # Continuous single-frequency tones lack speech dynamics
    if flatness < 1e-4 and std_centroid < 40.0:
        return False

    return True


def transcribe_audio(path: str | pathlib.Path, language: str = "auto") -> Dict[str, Any]:
    """Transcribe audio into text and timed segments using Whisper ASR with language routing."""
    audio, sr = load_audio(path)
    duration = float(len(audio)) / float(sr)

    # If the input is silence or a synthetic tone, return empty transcript cleanly
    if not _is_likely_speech(audio, sr):
        logger.info("Audio identified as non-speech / tone / silence; skipping ASR.")
        return {
            "language": language if language != "auto" else "en",
            "duration": round(duration, 3),
            "transcript": "",
            "segments": [],
        }

    language_map = {
        "te": "telugu",
        "hi": "hindi",
        "fr": "french",
        "en": "english",
        "auto": None,
    }
    lang_key = language.lower().strip()
    target_lang = language_map.get(lang_key, None)

    gen_kwargs: Dict[str, Any] = {"task": "transcribe"}
    if target_lang is not None:
        gen_kwargs["language"] = target_lang

    pipe = get_whisper()
    try:
        out = pipe(
            {"raw": audio, "sampling_rate": sr},
            return_timestamps=True,
            generate_kwargs=gen_kwargs,
        )
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        raise RuntimeError(f"Whisper transcription error: {exc}") from exc

    chunks = out.get("chunks", []) if isinstance(out, dict) else []
    segments: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        ts = chunk.get("timestamp")
        if not ts:
            start = 0.0
            end = duration
        else:
            start = ts[0] if ts[0] is not None else 0.0
            end = ts[1]
            if end is None:
                if (
                    i + 1 < len(chunks)
                    and chunks[i + 1].get("timestamp")
                    and chunks[i + 1]["timestamp"][0] is not None
                ):
                    end = chunks[i + 1]["timestamp"][0]
                else:
                    end = duration

        start_raw = max(0.0, float(start))
        end_raw = min(duration, max(start_raw, float(end)))
        text = str(chunk.get("text", "")).strip()

        if not text:
            continue

        if segments:
            start_raw = max(start_raw, segments[-1]["start_raw"])
            end_raw = max(start_raw, end_raw)

        segments.append({
            "start_raw": round(start_raw, 3),
            "end_raw": round(end_raw, 3),
            "text": text,
        })

    transcript = str(out.get("text", "")).strip() if isinstance(out, dict) else ""
    if not transcript and segments:
        transcript = " ".join(s["text"] for s in segments)

    # Detected language heuristic
    if language != "auto":
        detected_lang = language
    else:
        if any(0x0C00 <= ord(c) <= 0x0C7F for c in transcript):
            detected_lang = "te"
        elif any(0x0900 <= ord(c) <= 0x097F for c in transcript):
            detected_lang = "hi"
        else:
            french_stopwords = {
                "le", "la", "les", "des", "du", "un", "une", "est",
                "sont", "et", "dans", "pour", "qui", "que", "avec",
                "sur", "ce", "cette", "pas", "ne",
            }
            words = set(re.findall(r"\b\w+\b", transcript.lower()))
            if len(words & french_stopwords) >= 2:
                detected_lang = "fr"
            else:
                detected_lang = "en"

    return {
        "language": detected_lang,
        "duration": round(duration, 3),
        "transcript": transcript,
        "segments": segments,
    }


def analyze_audio(path: str | pathlib.Path, language: str = "auto") -> Dict[str, Any]:
    """Perform full forensic audio analysis conforming to the CANONICAL SESSION JSON SCHEMA."""
    transcription = transcribe_audio(path, language=language)

    facts = extract_audio_facts(
        transcript=transcription["transcript"],
        segments=transcription["segments"],
    )

    events = extract_audio_events(transcription["segments"])

    result: Dict[str, Any] = {
        "language": transcription["language"],
        "duration": transcription["duration"],
        "transcript": transcription["transcript"],
        "segments": transcription["segments"],
        "events": events,
        "location_inference": facts.get(
            "location_inference",
            {"label": "Unknown", "confidence": 0.0, "evidence": "None"},
        ),
        "suspect_description": facts.get("suspect_description", "Unknown"),
        "hypotheses": facts.get("hypotheses", []),
    }

    return to_jsonable(result)
