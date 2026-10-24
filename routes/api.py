from __future__ import annotations

"""Flask API blueprint routes for CrimeVision AI."""

import logging
import re
from typing import Any, Dict

from flask import Blueprint, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

import config
from modules import chat, pipeline, session_store
from modules.device import get_device
from modules.llm import is_configured
from modules.utils import allowed_file

logger = logging.getLogger("crimevision.api")

api_bp = Blueprint("api", __name__, url_prefix="/api")
_SID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Service health status reporting active device and LLM readiness."""
    return jsonify({
        "ok": True,
        "device": str(get_device()),
        "groq_configured": bool(is_configured()),
    }), 200


@api_bp.route("/analyze", methods=["POST"])
def analyze_evidence():
    """Upload multimodal evidence files, initialize session, and launch background pipeline."""
    img_file = request.files.get("image")
    vid_file = request.files.get("video")
    aud_file = request.files.get("audio")

    # At least one media file required
    valid_img = img_file is not None and bool(img_file.filename)
    valid_vid = vid_file is not None and bool(vid_file.filename)
    valid_aud = aud_file is not None and bool(aud_file.filename)

    if not (valid_img or valid_vid or valid_aud):
        return jsonify({"error": "At least one media file (image, video, or audio) is required"}), 400

    # Language validation
    language = request.form.get("language", "auto").strip().lower()
    if language not in config.SUPPORTED_AUDIO_LANGS:
        return jsonify({
            "error": f"Unsupported language '{language}'. Supported: {sorted(list(config.SUPPORTED_AUDIO_LANGS))}"
        }), 400

    # Validate file extensions
    if valid_img and not allowed_file(img_file.filename, config.ALLOWED_IMAGE_EXT):
        return jsonify({"error": f"Invalid image extension. Allowed: {config.ALLOWED_IMAGE_EXT}"}), 400
    if valid_vid and not allowed_file(vid_file.filename, config.ALLOWED_VIDEO_EXT):
        return jsonify({"error": f"Invalid video extension. Allowed: {config.ALLOWED_VIDEO_EXT}"}), 400
    if valid_aud and not allowed_file(aud_file.filename, config.ALLOWED_AUDIO_EXT):
        return jsonify({"error": f"Invalid audio extension. Allowed: {config.ALLOWED_AUDIO_EXT}"}), 400

    # Create session
    session = session_store.new_session()
    sid = session["id"]

    upload_dir = config.UPLOAD_DIR / sid
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_media: Dict[str, str] = {}

    if valid_img and img_file is not None:
        raw_name = secure_filename(img_file.filename)
        ext = raw_name.rsplit(".", 1)[1].lower() if "." in raw_name else "jpg"
        target_path = upload_dir / f"image.{ext}"
        img_file.save(str(target_path))
        saved_media["image"] = str(target_path)

    if valid_vid and vid_file is not None:
        raw_name = secure_filename(vid_file.filename)
        ext = raw_name.rsplit(".", 1)[1].lower() if "." in raw_name else "mp4"
        target_path = upload_dir / f"video.{ext}"
        vid_file.save(str(target_path))
        saved_media["video"] = str(target_path)

    if valid_aud and aud_file is not None:
        raw_name = secure_filename(aud_file.filename)
        ext = raw_name.rsplit(".", 1)[1].lower() if "." in raw_name else "wav"
        target_path = upload_dir / f"audio.{ext}"
        aud_file.save(str(target_path))
        saved_media["audio"] = str(target_path)

    # Update session media paths
    session_store.update_session(sid, media=saved_media)

    # Launch asynchronous pipeline
    pipeline.start_pipeline(sid, language=language)

    return jsonify({"session_id": sid}), 202


@api_bp.route("/status/<sid>", methods=["GET"])
def get_session_status(sid: str):
    """Retrieve execution progress and status for an active session."""
    if not _SID_PATTERN.match(sid):
        return jsonify({"error": "Session not found"}), 404

    session = session_store.load_session(sid)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    return jsonify({
        "status": session.get("status", "unknown"),
        "progress": session.get("progress", {}),
        "errors": session.get("errors", []),
    }), 200


@api_bp.route("/session/<sid>", methods=["GET"])
def get_session_details(sid: str):
    """Retrieve complete persisted forensic analysis session payload."""
    if not _SID_PATTERN.match(sid):
        return jsonify({"error": "Session not found"}), 404

    session = session_store.load_session(sid)
    if session is None:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(session), 200


@api_bp.route("/chat", methods=["POST"])
def post_chat():
    """Process grounded conversational forensic query against an analyzed session."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON body"}), 400

    session_id = str(data.get("session_id", "")).strip()
    if not session_id or not _SID_PATTERN.match(session_id):
        return jsonify({"error": "Valid session_id is required"}), 400

    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required and cannot be empty"}), 400

    history = data.get("history")
    if history is not None and not isinstance(history, list):
        return jsonify({"error": "history must be a list of turns if provided"}), 400

    try:
        response = chat.chat(session_id, message, history)
        return jsonify(response), 200
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Error processing chat query: %s", exc)
        return jsonify({"error": f"Chat processing failed: {exc}"}), 500


@api_bp.route("/media/<sid>/<kind>", methods=["GET"])
def get_media_stream(sid: str, kind: str):
    """Stream uploaded media files with HTTP Range request support for seeking."""
    if not _SID_PATTERN.match(sid):
        return jsonify({"error": "Session not found"}), 404

    clean_kind = kind.strip().lower()
    if clean_kind not in ("image", "video", "audio"):
        return jsonify({"error": f"Invalid media kind '{kind}'. Allowed: image, video, audio"}), 400

    media_dir = config.UPLOAD_DIR / sid
    if not media_dir.exists():
        return jsonify({"error": "Media directory not found"}), 404

    matches = list(media_dir.glob(f"{clean_kind}.*"))
    if not matches:
        return jsonify({"error": f"No {clean_kind} media found for session {sid}"}), 404

    target_file = matches[0].name
    return send_from_directory(str(media_dir), target_file, conditional=True)


@api_bp.route("/report/<sid>", methods=["GET"])
def get_report(sid: str):
    """Download forensic PDF report."""
    if not _SID_PATTERN.match(sid):
        return jsonify({"error": "Session not found"}), 404

    try:
        from modules import reports
        pdf_path = reports.build_pdf(sid)
        return send_file(
            str(pdf_path),
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"CrimeVision_Report_{sid[:8]}.pdf",
        )
    except KeyError:
        return jsonify({"error": f"Session '{sid}' not found"}), 404
    except Exception as exc:
        logger.exception("Report generation failed for session %s: %s", sid, exc)
        return jsonify({"error": f"Failed to generate report: {exc}"}), 500
