from __future__ import annotations

"""End-to-end API smoke test verifying multimodal upload (image+audio+video), pipeline, chat, and PDF generation."""

import pathlib
import sys
import time
import requests

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

BASE_URL = "http://127.0.0.1:5000"


def main():
    print("=" * 65)
    print("CRIMEVISION AI - END-TO-END MULTIMODAL API AUDIT")
    print("=" * 65)
    t_start_total = time.time()

    # 1. Health check
    print("\n1. Checking /api/health...")
    try:
        r_health = requests.get(f"{BASE_URL}/api/health", timeout=10)
        print(f"Health Response [{r_health.status_code}]: {r_health.json()}")
        assert r_health.status_code == 200
        assert r_health.json().get("ok") is True
    except Exception as exc:
        print(f"Failed to connect to server at {BASE_URL}: {exc}")
        sys.exit(1)

    # 2. Upload sample.jpg, jfk.wav, and sample.mp4 to /api/analyze
    img_path = BASE_DIR / "data" / "uploads" / "sample.jpg"
    aud_path = BASE_DIR / "data" / "uploads" / "jfk.wav"
    vid_path = BASE_DIR / "data" / "uploads" / "sample.mp4"

    print(f"\n2. Uploading multimodal evidence:\n - Image: {img_path}\n - Audio: {aud_path}\n - Video: {vid_path}")
    for p in (img_path, aud_path, vid_path):
        if not p.exists():
            print(f"Required sample file missing: {p}")
            sys.exit(1)

    with open(img_path, "rb") as f_img, open(aud_path, "rb") as f_aud, open(vid_path, "rb") as f_vid:
        files = {
            "image": ("sample.jpg", f_img, "image/jpeg"),
            "audio": ("jfk.wav", f_aud, "audio/wav"),
            "video": ("sample.mp4", f_vid, "video/mp4"),
        }
        data = {"language": "auto"}
        t0 = time.time()
        r_analyze = requests.post(f"{BASE_URL}/api/analyze", files=files, data=data, timeout=60)

    print(f"Analyze Response [{r_analyze.status_code}] in {time.time() - t0:.2f}s: {r_analyze.text}")
    assert r_analyze.status_code == 202
    session_id = r_analyze.json().get("session_id")
    assert session_id is not None
    print(f"Forensic session initiated: {session_id}")

    # 3. Poll /api/status/<sid> every 2 seconds until done
    print(f"\n3. Polling /api/status/{session_id} every 2s...")
    start_poll = time.time()
    final_status = None
    last_progress = None

    while time.time() - start_poll < 360:
        r_status = requests.get(f"{BASE_URL}/api/status/{session_id}", timeout=10)
        assert r_status.status_code == 200
        status_data = r_status.json()
        curr_status = status_data.get("status")
        progress = status_data.get("progress", {})

        if progress != last_progress:
            print(f"[{int(time.time() - start_poll)}s] Status: {curr_status} | Progress: {progress}")
            last_progress = progress

        if curr_status in ("done", "error"):
            final_status = curr_status
            break

        time.sleep(2)

    pipeline_duration = time.time() - start_poll
    print(f"\nPipeline completed with status: '{final_status}' in {pipeline_duration:.1f}s")
    assert final_status == "done", f"Expected pipeline status 'done', got '{final_status}'"

    # 4. Fetch full session JSON
    print(f"\n4. Fetching session details from /api/session/{session_id}...")
    r_session = requests.get(f"{BASE_URL}/api/session/{session_id}", timeout=10)
    assert r_session.status_code == 200
    session_json = r_session.json()

    fusion = session_json.get("fusion", {})
    assert fusion is not None, "Session fusion block must not be None"
    timeline = fusion.get("timeline", [])
    print(f"Multimodal Timeline Event Count: {len(timeline)}")
    assert len(timeline) > 0, "Fusion timeline must not be empty"

    print(f"Fusion Summary: {fusion.get('summary')}")
    print(f"Overall Confidence: {fusion.get('overall_confidence')}")
    print(f"Anchor Events: {fusion.get('anchor_events')}")
    print(f"Evidentiary Conflicts: {fusion.get('conflicts')}")

    # 5. Test Chat endpoint
    print("\n5. Testing /api/chat with query 'What happened?'...")
    t0_chat = time.time()
    chat_payload = {
        "session_id": session_id,
        "message": "What happened?",
    }
    r_chat = requests.post(f"{BASE_URL}/api/chat", json=chat_payload, timeout=30)
    print(f"Chat Response [{r_chat.status_code}] in {time.time() - t0_chat:.2f}s:")
    assert r_chat.status_code == 200
    chat_resp = r_chat.json()
    answer = chat_resp.get("answer", "")
    print(f"Answer:\n{answer}")
    assert len(answer) > 0, "Chat answer must not be empty"

    # 6. Test PDF Report Download
    print(f"\n6. Testing /api/report/{session_id}...")
    t0_pdf = time.time()
    r_report = requests.get(f"{BASE_URL}/api/report/{session_id}", timeout=30)
    print(f"Report Response [{r_report.status_code}] in {time.time() - t0_pdf:.2f}s:")
    assert r_report.status_code == 200
    assert r_report.headers.get("Content-Type") == "application/pdf"
    assert len(r_report.content) > 5000, f"PDF too small: {len(r_report.content)} bytes"
    assert r_report.content.startswith(b"%PDF"), "Response does not start with %PDF header"
    print(f"Report downloaded successfully: {len(r_report.content)} bytes (starts with {r_report.content[:8]})")

    total_time = time.time() - t_start_total
    print("\n" + "=" * 65)
    print(f"ALL END-TO-END AUDIT CHECKS PASSED IN {total_time:.1f}s")
    print(f"Generated PDF Dossier: {BASE_DIR / 'data' / 'reports' / f'{session_id}.pdf'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
