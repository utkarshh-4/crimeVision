from __future__ import annotations

"""Prefetch and cache all neural AI models specified in config to ensure fast demo execution."""

import pathlib
import sys
import time

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    pipeline,
)


def prefetch():
    print("=" * 65)
    print("CRIMEVISION AI - MODEL CACHE PREFETCHER")
    print("=" * 65)
    start_total = time.time()

    models = [
        ("RT-DETR (Visual Detection)", config.RTDETR_MODEL, lambda: (
            RTDetrImageProcessor.from_pretrained(config.RTDETR_MODEL),
            RTDetrForObjectDetection.from_pretrained(config.RTDETR_MODEL),
        )),
        ("CLIP (Zero-Shot & Firearm Verification)", config.CLIP_MODEL, lambda: (
            CLIPProcessor.from_pretrained(config.CLIP_MODEL),
            CLIPModel.from_pretrained(config.CLIP_MODEL),
        )),
        ("BLIP (Forensic Captioning)", config.BLIP_MODEL, lambda: (
            BlipProcessor.from_pretrained(config.BLIP_MODEL),
            BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL),
        )),
        ("Whisper (Acoustic Transcription)", config.WHISPER_MODEL, lambda: pipeline(
            "automatic-speech-recognition",
            model=config.WHISPER_MODEL,
        )),
    ]

    for idx, (title, model_id, loader) in enumerate(models, start=1):
        print(f"\n[{idx}/4] Prefetching {title}...")
        print(f"      Model ID: {model_id}")
        t0 = time.time()
        try:
            loader()
            print(f"      -> Cached successfully in {time.time() - t0:.2f}s")
        except Exception as exc:
            print(f"      -> ERROR downloading {model_id}: {exc}")
            raise

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 65)
    print(f"ALL FORENSIC MODELS PREFETCHED & CACHED IN {total_elapsed:.1f}s")
    print("=" * 65)


if __name__ == "__main__":
    prefetch()
