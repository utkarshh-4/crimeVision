from __future__ import annotations

"""Smoke verification script for image analysis pipeline."""

import json
import pathlib
import sys

# Ensure project root is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests
import config
from modules.image_analysis import analyze_image


def main() -> int:
    sample_path = config.UPLOAD_DIR / "sample.jpg"

    if not sample_path.exists():
        url = "https://ultralytics.com/images/bus.jpg"
        print(f"Downloading sample image from {url} to {sample_path}...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        sample_path.write_bytes(resp.content)
        print("Sample image downloaded successfully.")
    else:
        print(f"Sample image already present at {sample_path}.")

    print("Running analyze_image on sample...")
    result = analyze_image(str(sample_path))

    print("\nAnalysis Result:")
    print(json.dumps(result, indent=2))

    # Assertions per task requirements
    people_count = result.get("people_count", 0)
    caption = result.get("caption", "")
    scene_label = result.get("scene", {}).get("label", "")
    hypotheses = result.get("hypotheses", [])
    sum_prob = round(sum(float(h.get("probability", 0.0)) for h in hypotheses), 3)

    print("\nVerification Checks:")
    print(f"  people_count >= 3: {people_count} ({people_count >= 3})")
    print(f"  caption non-empty: '{caption}' ({bool(caption)})")
    print(f"  scene.label non-empty: '{scene_label}' ({bool(scene_label)})")
    print(f"  hypotheses count == 3: {len(hypotheses)} ({len(hypotheses) == 3})")
    print(f"  hypotheses probability sum == 1.0: {sum_prob} ({sum_prob == 1.0})")
    for i, h in enumerate(hypotheses, 1):
        print(f"    [{i}] {h.get('title')} (p={h.get('probability')}): {h.get('explanation')}")

    assert people_count >= 3, f"Expected people_count >= 3, got {people_count}"
    assert bool(caption), "Expected non-empty caption"
    assert bool(scene_label), "Expected non-empty scene.label"
    assert len(hypotheses) == 3, f"Expected 3 hypotheses, got {len(hypotheses)}"
    assert sum_prob == 1.0, f"Expected probabilities to sum to 1.0, got {sum_prob}"
    assert all(bool(h.get("title")) for h in hypotheses), "All hypothesis titles must be non-empty"

    print("\nSMOKE IMAGE CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
