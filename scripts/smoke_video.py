from __future__ import annotations

"""Smoke verification script for video sampling, object detection timeline, and visual events."""

import json
import pathlib
import sys

# Ensure repository root is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import requests

import config
from modules.video_analysis import analyze_video, probe_video


def main() -> int:
    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = config.UPLOAD_DIR / "sample.mp4"

    # 1. Download sample video if missing
    if not sample_path.exists():
        url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
        print(f"Downloading sample video from {url} to {sample_path}...")
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(sample_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        print("Sample video downloaded successfully.")
    else:
        print(f"Sample video already present at {sample_path}.")

    # 2. Probe video metadata
    info = probe_video(sample_path)
    print("\nVideo Probe:")
    print(json.dumps(info, indent=2))

    # 3. Limit to 10 seconds for verification run
    config.MAX_VIDEO_SECONDS = 10
    print(f"\nRunning analyze_video (MAX_VIDEO_SECONDS={config.MAX_VIDEO_SECONDS})...")

    def on_progress(done: int, total: int) -> None:
        print(f"  Frame progress: {done}/{total}")

    result = analyze_video(sample_path, progress_cb=on_progress)

    frames = result.get("frames", [])
    events = result.get("events", [])
    first_3_people = [f.get("people_count", 0) for f in frames[:3]]

    print("\n" + "=" * 60)
    print("VIDEO ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Total sampled frames: {len(frames)}")
    print(f"First 3 frames' people_count: {first_3_people}")
    print(f"\nVisual Events ({len(events)}):")
    print(json.dumps(events, indent=2))

    # Verification Checks
    print("\nVerification Checks:")
    has_person = any(f.get("people_count", 0) >= 1 for f in frames)
    print(f"  Sampled frames >= 1: {len(frames)} ({len(frames) >= 1})")
    print(f"  >=1 frame with people_count >= 1: {has_person}")

    assert len(frames) >= 1, "Expected at least 1 sampled frame"
    assert has_person, "Expected at least one frame with people_count >= 1"

    print("\nSMOKE VIDEO CHECK PASSED SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
