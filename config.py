from __future__ import annotations

"""Configuration module for CrimeVision AI platform."""

import os
import pathlib
from typing import List, Set
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
BASE_DIR: pathlib.Path = pathlib.Path(__file__).resolve().parent
DATA_DIR: pathlib.Path = BASE_DIR / "data"
UPLOAD_DIR: pathlib.Path = DATA_DIR / "uploads"
SESSION_DIR: pathlib.Path = DATA_DIR / "sessions"
REPORT_DIR: pathlib.Path = DATA_DIR / "reports"

# Ensure runtime directories exist
for directory in (DATA_DIR, UPLOAD_DIR, SESSION_DIR, REPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# AI Models
RTDETR_MODEL: str = "PekingU/rtdetr_r50vd"
CLIP_MODEL: str = "openai/clip-vit-base-patch32"
BLIP_MODEL: str = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "openai/whisper-base")

# Groq LLM Configuration
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL: str = "llama-3.1-8b-instant"
LLM_TEMPERATURE: float = 0.1

# Vision & Detection Thresholds
DET_THRESHOLD: float = 0.5
NMS_IOU: float = 0.7
FIREARM_THRESHOLD: float = 0.6

# Scene labels and crime taxonomy
SCENE_LABELS: List[str] = [
    "hall",
    "road",
    "bedroom",
    "kitchen",
    "living room",
    "bathroom",
    "office",
    "parking lot",
    "staircase",
    "garage",
    "shop",
    "park",
]

CRIME_TAXONOMY: List[str] = [
    "Assault",
    "Theft",
    "Robbery",
    "Armed Threat",
    "Vandalism",
    "Trespassing",
    "Kidnapping",
    "No Crime Detected",
]

WEAPON_COCO_LABELS: Set[str] = {"knife", "scissors", "baseball bat"}

# Video analysis settings
VIDEO_SAMPLE_HZ: int = 1
MAX_VIDEO_SECONDS: int = int(os.getenv("MAX_VIDEO_SECONDS", "90"))
CAPTION_EVERY_N_FRAMES: int = int(os.getenv("CAPTION_EVERY_N_FRAMES", "3"))

# Upload and media settings
MAX_UPLOAD_MB: int = 500
ALLOWED_IMAGE_EXT: Set[str] = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO_EXT: Set[str] = {"mp4", "avi", "mov", "mkv", "webm"}
ALLOWED_AUDIO_EXT: Set[str] = {"wav", "mp3", "m4a", "ogg", "flac"}
SUPPORTED_AUDIO_LANGS: Set[str] = {"auto", "te", "hi", "fr", "en"}

# Server settings
FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes")
PORT: int = int(os.getenv("PORT", "5000"))
