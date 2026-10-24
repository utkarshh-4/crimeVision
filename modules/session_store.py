from __future__ import annotations

"""Session management and JSON persistence store for CrimeVision AI."""

import json
import logging
import os
import pathlib
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
from modules.utils import to_jsonable

logger = logging.getLogger("crimevision.session_store")

_SESSION_LOCK = threading.RLock()
_SESSION_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _is_valid_session_id(session_id: str) -> bool:
    """Validate that session_id is a 32-character hexadecimal string."""
    if not isinstance(session_id, str):
        return False
    return bool(_SESSION_ID_PATTERN.match(session_id))


def _get_session_path(session_id: str) -> pathlib.Path:
    """Return the absolute path for a session JSON file."""
    return config.SESSION_DIR / f"{session_id}.json"


def new_session(media: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new forensic analysis session and persist it to disk."""
    safe_media = media if isinstance(media, dict) else {}
    session_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    has_image = bool(safe_media.get("image"))
    has_video = bool(safe_media.get("video"))
    has_audio = bool(safe_media.get("audio"))

    progress = {
        "image": "pending" if has_image else "skipped",
        "video": "pending" if has_video else "skipped",
        "audio": "pending" if has_audio else "skipped",
        "fusion": "pending" if (has_image or has_video or has_audio) else "skipped",
        "report": "pending",
    }

    session: Dict[str, Any] = {
        "id": session_id,
        "session_id": session_id,
        "created_at": created_at,
        "status": "queued",
        "progress": progress,
        "media": safe_media,
        "errors": [],
        "image": None,
        "video": None,
        "audio": None,
        "fusion": None,
        "chat_history": [],
    }

    save_session(session)
    return session


def save_session(session: Dict[str, Any]) -> None:
    """Persist session data atomically to disk guarded by RLock."""
    if not isinstance(session, dict):
        raise ValueError("Session data must be a dictionary")

    session_id = session.get("id") or session.get("session_id")
    if not session_id or not _is_valid_session_id(str(session_id)):
        raise ValueError(f"Invalid session ID for saving: {session_id}")

    target_path = _get_session_path(str(session_id))
    jsonable_data = to_jsonable(session)
    payload = json.dumps(jsonable_data, indent=2)

    with _SESSION_LOCK:
        config.SESSION_DIR.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to temp file in the same directory, then os.replace
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f"{session_id}_",
            suffix=".tmp",
            dir=str(config.SESSION_DIR),
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, str(target_path))
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as rem_exc:
                    logger.debug("Failed to remove temp session file: %s", rem_exc)
            raise


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Load session from disk by ID, returning None if missing, invalid, or corrupt."""
    if not _is_valid_session_id(session_id):
        return None

    target_path = _get_session_path(session_id)

    with _SESSION_LOCK:
        if not target_path.exists():
            return None

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("Session file %s does not contain a JSON object", target_path)
                return None
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to decode session file %s: %s", target_path, exc)
            return None


def update_session(session_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    """Thread-safe read-modify-write updating arbitrary fields in a session."""
    if not _is_valid_session_id(session_id):
        return None

    with _SESSION_LOCK:
        session = load_session(session_id)
        if session is None:
            return None

        session.update(fields)
        save_session(session)
        return session


def update_progress(session_id: str, key: str, value: str) -> Optional[Dict[str, Any]]:
    """Thread-safe read-modify-write updating a specific progress key."""
    if not _is_valid_session_id(session_id):
        return None

    with _SESSION_LOCK:
        session = load_session(session_id)
        if session is None:
            return None

        if "progress" not in session or not isinstance(session["progress"], dict):
            session["progress"] = {}
        session["progress"][key] = value

        save_session(session)
        return session


def list_sessions() -> List[Dict[str, Any]]:
    """List metadata for all persisted sessions sorted by creation time descending."""
    sessions_meta: List[Dict[str, Any]] = []

    with _SESSION_LOCK:
        if not config.SESSION_DIR.exists():
            return []

        for p in config.SESSION_DIR.glob("*.json"):
            sid = p.stem
            if _is_valid_session_id(sid):
                session = load_session(sid)
                if session is not None:
                    sessions_meta.append({
                        "session_id": sid,
                        "created_at": session.get("created_at", ""),
                        "status": session.get("status", "unknown"),
                    })

    sessions_meta.sort(key=lambda s: str(s.get("created_at", "")), reverse=True)
    return sessions_meta
