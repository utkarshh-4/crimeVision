from __future__ import annotations

"""Orchestration pipeline for concurrent multimodal evidence analysis, fusion, and indexing."""

import concurrent.futures
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

from modules.audio_analysis import analyze_audio
from modules.chat import build_evidence_index
from modules.fusion import fuse_session
from modules.image_analysis import analyze_image
from modules.session_store import load_session, update_progress, update_session
from modules.video_analysis import analyze_video

logger = logging.getLogger("crimevision.pipeline")


def run_pipeline(session_id: str, language: str = "auto") -> None:
    """Execute asynchronous end-to-end multimodal forensic analysis pipeline for a session."""
    try:
        session = load_session(session_id)
        if session is None:
            logger.error("Pipeline cannot start: session %s not found", session_id)
            return

        update_session(session_id, status="running")

        media = session.get("media", {})
        if not isinstance(media, dict):
            media = {}

        has_image = bool(media.get("image"))
        has_video = bool(media.get("video"))
        has_audio = bool(media.get("audio"))

        if has_image:
            update_progress(session_id, "image", "running")
        if has_video:
            update_progress(session_id, "video", "running")
        if has_audio:
            update_progress(session_id, "audio", "running")

        def _process_image():
            img_path = media.get("image")
            try:
                res = analyze_image(img_path)
                update_session(session_id, image=res)
                update_progress(session_id, "image", "done")
                return "image", True, None
            except Exception as exc:
                logger.exception("Image analysis failed for session %s: %s", session_id, exc)
                update_progress(session_id, "image", "error")
                return "image", False, f"Image analysis error: {exc}"

        def _process_audio():
            aud_path = media.get("audio")
            try:
                res = analyze_audio(aud_path, language=language)
                update_session(session_id, audio=res)
                update_progress(session_id, "audio", "done")
                return "audio", True, None
            except Exception as exc:
                logger.exception("Audio analysis failed for session %s: %s", session_id, exc)
                update_progress(session_id, "audio", "error")
                return "audio", False, f"Audio analysis error: {exc}"

        def _process_video():
            vid_path = media.get("video")
            try:
                res = analyze_video(vid_path)
                update_session(session_id, video=res)
                update_progress(session_id, "video", "done")
                return "video", True, None
            except Exception as exc:
                logger.exception("Video analysis failed for session %s: %s", session_id, exc)
                update_progress(session_id, "video", "error")
                return "video", False, f"Video analysis error: {exc}"

        futures: List[concurrent.futures.Future[Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            if has_image:
                futures.append(executor.submit(_process_image))
            if has_audio:
                futures.append(executor.submit(_process_audio))
            if has_video:
                futures.append(executor.submit(_process_video))

            errors: List[str] = []
            success_count = 0
            for f in concurrent.futures.as_completed(futures):
                modality, ok, err_msg = f.result()
                if ok:
                    success_count += 1
                elif err_msg:
                    errors.append(err_msg)

        if errors:
            curr_session = load_session(session_id)
            if curr_session:
                curr_errors = list(curr_session.get("errors", [])) + errors
                update_session(session_id, errors=curr_errors)

        # Multimodal timeline fusion
        update_progress(session_id, "fusion", "running")
        curr_session = load_session(session_id)
        fusion_res = None
        if curr_session is not None:
            try:
                fusion_res = fuse_session(curr_session)
                update_session(session_id, fusion=fusion_res)
                update_progress(session_id, "fusion", "done")
            except Exception as exc:
                logger.exception("Fusion failed for session %s: %s", session_id, exc)
                update_progress(session_id, "fusion", "error")
                curr_errors = list(curr_session.get("errors", [])) + [f"Fusion error: {exc}"]
                update_session(session_id, errors=curr_errors)

        # Searchable evidence index for chat
        curr_session = load_session(session_id)
        if curr_session is not None:
            try:
                index_res = build_evidence_index(curr_session)
                update_session(session_id, index=index_res)
            except Exception as exc:
                logger.exception("Indexing failed for session %s: %s", session_id, exc)

        # Final status calculation
        if len(futures) > 0 and success_count == 0:
            final_status = "error"
        else:
            final_status = "done"

        update_session(session_id, status=final_status)
        update_progress(session_id, "report", "pending")
        logger.info("Pipeline completed for session %s with status %s", session_id, final_status)

    except Exception as exc:
        logger.exception("Unhandled fatal exception in pipeline for session %s: %s", session_id, exc)
        curr_session = load_session(session_id)
        if curr_session:
            errs = list(curr_session.get("errors", [])) + [f"Pipeline fatal error: {exc}"]
            update_session(session_id, status="error", errors=errs)


def start_pipeline(session_id: str, language: str = "auto") -> threading.Thread:
    """Launch asynchronous analysis pipeline in a background daemon thread."""
    thread = threading.Thread(
        target=run_pipeline,
        args=(session_id, language),
        daemon=True,
        name=f"pipeline-{session_id[:8]}",
    )
    thread.start()
    return thread
