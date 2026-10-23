from __future__ import annotations

"""Smoke verification script for audio analysis, Whisper ASR, and fact extraction."""

import json
import pathlib
import sys

# Ensure repository root is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import requests
import soundfile as sf

import config
from modules.audio_analysis import analyze_audio


def main() -> int:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    jfk_path = config.UPLOAD_DIR / "jfk.wav"
    sine_path = config.UPLOAD_DIR / "sine_440hz.wav"

    # 1. Synthesize 4-second sine tone (440 Hz at 16000 Hz mono)
    print(f"Synthesizing 4-second sine tone to {sine_path}...")
    sr = 16000
    duration = 4.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine_wave = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(str(sine_path), sine_wave, sr)
    print("Sine tone generated.")

    # 2. Download JFK audio sample if missing
    if not jfk_path.exists():
        url = "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav"
        print(f"Downloading JFK speech sample from {url} to {jfk_path}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        jfk_path.write_bytes(resp.content)
        print("JFK audio sample downloaded successfully.")
    else:
        print(f"JFK sample already present at {jfk_path}.")

    # 3. Analyze JFK Speech Sample
    print("\n" + "=" * 60)
    print("ANALYZING JFK SPEECH SAMPLE")
    print("=" * 60)
    jfk_res = analyze_audio(str(jfk_path), language="auto")
    print(json.dumps(jfk_res, indent=2))

    # Verification checks on JFK sample
    jfk_transcript = jfk_res.get("transcript", "")
    jfk_segments = jfk_res.get("segments", [])
    jfk_lang = jfk_res.get("language", "")
    jfk_hypo = jfk_res.get("hypotheses", [])
    jfk_prob_sum = round(sum(float(h.get("probability", 0.0)) for h in jfk_hypo), 3)

    print("\nJFK Verification Checks:")
    print(f"  transcript non-empty: '{jfk_transcript}' ({bool(jfk_transcript)})")
    print(f"  language: '{jfk_lang}' (expected 'en': {jfk_lang == 'en'})")
    print(f"  segments >= 1: {len(jfk_segments)} ({len(jfk_segments) >= 1})")
    print(f"  hypotheses count == 3: {len(jfk_hypo)} ({len(jfk_hypo) == 3})")
    print(f"  hypotheses prob sum == 1.0: {jfk_prob_sum} ({jfk_prob_sum == 1.0})")

    assert bool(jfk_transcript), "Expected non-empty transcript for JFK sample"
    assert len(jfk_segments) >= 1, "Expected at least 1 segment for JFK sample"
    assert len(jfk_hypo) == 3, "Expected 3 hypotheses"
    assert jfk_prob_sum == 1.0, f"Expected probability sum 1.0, got {jfk_prob_sum}"

    # Verify monotonic increasing segment timestamps
    for i, s in enumerate(jfk_segments):
        start = s.get("start_raw", 0.0)
        end = s.get("end_raw", 0.0)
        assert start <= end, f"Segment {i} has start ({start}) > end ({end})"
        if i > 0:
            prev_start = jfk_segments[i - 1].get("start_raw", 0.0)
            assert start >= prev_start, f"Segment {i} start ({start}) < previous start ({prev_start})"

    # 4. Analyze Sine Tone Sample (No-Speech)
    print("\n" + "=" * 60)
    print("ANALYZING SINE TONE SAMPLE (NON-SPEECH)")
    print("=" * 60)
    sine_res = analyze_audio(str(sine_path), language="auto")
    print(json.dumps(sine_res, indent=2))

    sine_transcript = sine_res.get("transcript", "")
    sine_hypo = sine_res.get("hypotheses", [])
    sine_prob_sum = round(sum(float(h.get("probability", 0.0)) for h in sine_hypo), 3)

    print("\nSine Tone Verification Checks:")
    print(f"  transcript is empty or clean: '{sine_transcript}' ({sine_transcript == ''})")
    print(f"  hypotheses count == 3: {len(sine_hypo)} ({len(sine_hypo) == 3})")
    print(f"  hypotheses prob sum == 1.0: {sine_prob_sum} ({sine_prob_sum == 1.0})")

    assert sine_transcript == "", f"Expected empty transcript for pure sine tone, got '{sine_transcript}'"
    assert len(sine_hypo) == 3, "Expected 3 hypotheses for sine tone"
    assert sine_prob_sum == 1.0, f"Expected probability sum 1.0, got {sine_prob_sum}"

    print("\n" + "=" * 60)
    print("ALL AUDIO SMOKE CHECKS PASSED SUCCESSFULLY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
