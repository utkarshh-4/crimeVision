from __future__ import annotations

"""Pytest shared fixtures and test configuration for CrimeVision AI."""

import pathlib
import sys
from typing import Callable, Generator

import cv2
import numpy as np
from PIL import Image
import pytest
import soundfile as sf

# Ensure repository root is in sys.path
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from app import create_app


@pytest.fixture(autouse=True)
def tmp_data_dirs(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, pathlib.Path]:
    """Isolate runtime data directories into a temporary directory per test."""
    upload_dir = tmp_path / "uploads"
    session_dir = tmp_path / "sessions"
    report_dir = tmp_path / "reports"

    upload_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(config, "SESSION_DIR", session_dir)
    monkeypatch.setattr(config, "REPORT_DIR", report_dir)

    return {
        "upload_dir": upload_dir,
        "session_dir": session_dir,
        "report_dir": report_dir,
    }


@pytest.fixture(autouse=True)
def disable_groq_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure LLM client is unconfigured by default unless explicitly enabled in a test."""
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setattr(config, "GROQ_API_KEY", "")


@pytest.fixture
def client():
    """Create a Flask test client with testing mode enabled."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def make_synthetic_image(tmp_path: pathlib.Path) -> Callable[..., pathlib.Path]:
    """Factory fixture generating a tiny valid RGB image."""
    def _create(filename: str = "synthetic.jpg", size: tuple[int, int] = (64, 64), color: tuple[int, int, int] = (50, 100, 150)) -> pathlib.Path:
        target = tmp_path / filename
        img = Image.new("RGB", size, color=color)
        img.save(target, format="JPEG" if target.suffix.lower() in (".jpg", ".jpeg") else "PNG")
        return target
    return _create


@pytest.fixture
def make_synthetic_wav(tmp_path: pathlib.Path) -> Callable[..., pathlib.Path]:
    """Factory fixture generating a tiny valid mono WAV audio file with soundfile."""
    def _create(filename: str = "synthetic.wav", duration_s: float = 1.0, sr: int = 16000) -> pathlib.Path:
        target = tmp_path / filename
        num_samples = int(duration_s * sr)
        t = np.linspace(0, duration_s, num_samples, endpoint=False)
        # 440 Hz tone with float32 normalization
        audio_data = (0.3 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(str(target), audio_data, sr, subtype="FLOAT")
        return target
    return _create


@pytest.fixture
def make_synthetic_mp4(tmp_path: pathlib.Path) -> Callable[..., pathlib.Path]:
    """Factory fixture generating a tiny valid MP4 video file using cv2.VideoWriter."""
    def _create(filename: str = "synthetic.mp4", num_frames: int = 5, fps: float = 5.0, size: tuple[int, int] = (64, 64)) -> pathlib.Path:
        target = tmp_path / filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(target), fourcc, fps, size)
        try:
            for i in range(num_frames):
                frame = np.full((size[1], size[0], 3), fill_value=int((i * 40) % 255), dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()
        return target
    return _create


@pytest.fixture
def synthetic_image_path(make_synthetic_image: Callable[..., pathlib.Path]) -> pathlib.Path:
    """Fixture providing a ready-to-use tiny synthetic image path."""
    return make_synthetic_image()


@pytest.fixture
def synthetic_wav_path(make_synthetic_wav: Callable[..., pathlib.Path]) -> pathlib.Path:
    """Fixture providing a ready-to-use tiny synthetic WAV path."""
    return make_synthetic_wav()


@pytest.fixture
def synthetic_mp4_path(make_synthetic_mp4: Callable[..., pathlib.Path]) -> pathlib.Path:
    """Fixture providing a ready-to-use tiny synthetic MP4 path."""
    return make_synthetic_mp4()
