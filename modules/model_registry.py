from __future__ import annotations

"""Thread-safe lazy singleton model loaders for CrimeVision AI."""

import logging
import threading
from typing import Any, Optional, Tuple

from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    CLIPModel,
    CLIPProcessor,
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    pipeline,
)

import config
from modules.device import get_device, get_dtype

logger = logging.getLogger("crimevision.models")

_rtdetr_lock = threading.Lock()
_rtdetr_pair: Optional[Tuple[RTDetrImageProcessor, RTDetrForObjectDetection]] = None

_clip_lock = threading.Lock()
_clip_pair: Optional[Tuple[CLIPProcessor, CLIPModel]] = None

_blip_lock = threading.Lock()
_blip_pair: Optional[Tuple[BlipProcessor, BlipForConditionalGeneration]] = None

_whisper_lock = threading.Lock()
_whisper_pipe: Optional[Any] = None


def get_rtdetr() -> Tuple[RTDetrImageProcessor, RTDetrForObjectDetection]:
    """Retrieve or lazily initialize the RT-DETR object detection model and processor."""
    global _rtdetr_pair
    if _rtdetr_pair is None:
        with _rtdetr_lock:
            if _rtdetr_pair is None:
                device = get_device()
                logger.info("loading rtdetr on %s", device)
                try:
                    processor = RTDetrImageProcessor.from_pretrained(config.RTDETR_MODEL)
                    model = RTDetrForObjectDetection.from_pretrained(config.RTDETR_MODEL)
                    model.to(device).eval()
                    _rtdetr_pair = (processor, model)
                except Exception as exc:
                    err = (
                        f"Failed to load RT-DETR model '{config.RTDETR_MODEL}' on {device}: {exc}. "
                        "Please check internet connection for first download."
                    )
                    logger.error(err)
                    raise RuntimeError(err) from exc
    return _rtdetr_pair


def get_clip() -> Tuple[CLIPProcessor, CLIPModel]:
    """Retrieve or lazily initialize the CLIP zero-shot classification model and processor."""
    global _clip_pair
    if _clip_pair is None:
        with _clip_lock:
            if _clip_pair is None:
                device = get_device()
                logger.info("loading clip on %s", device)
                try:
                    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
                    model = CLIPModel.from_pretrained(config.CLIP_MODEL)
                    model.to(device).eval()
                    _clip_pair = (processor, model)
                except Exception as exc:
                    err = (
                        f"Failed to load CLIP model '{config.CLIP_MODEL}' on {device}: {exc}. "
                        "Please check internet connection for first download."
                    )
                    logger.error(err)
                    raise RuntimeError(err) from exc
    return _clip_pair


def get_blip() -> Tuple[BlipProcessor, BlipForConditionalGeneration]:
    """Retrieve or lazily initialize the BLIP captioning model and processor."""
    global _blip_pair
    if _blip_pair is None:
        with _blip_lock:
            if _blip_pair is None:
                device = get_device()
                logger.info("loading blip on %s", device)
                try:
                    processor = BlipProcessor.from_pretrained(config.BLIP_MODEL)
                    model = BlipForConditionalGeneration.from_pretrained(config.BLIP_MODEL)
                    model.to(device).eval()
                    _blip_pair = (processor, model)
                except Exception as exc:
                    err = (
                        f"Failed to load BLIP model '{config.BLIP_MODEL}' on {device}: {exc}. "
                        "Please check internet connection for first download."
                    )
                    logger.error(err)
                    raise RuntimeError(err) from exc
    return _blip_pair


def get_whisper() -> Any:
    """Retrieve or lazily initialize the Whisper ASR pipeline."""
    global _whisper_pipe
    if _whisper_pipe is None:
        with _whisper_lock:
            if _whisper_pipe is None:
                device = get_device()
                logger.info("loading whisper on %s", device)
                dev_id = 0 if device.type == "cuda" else -1
                torch_dtype = get_dtype(device)
                try:
                    pipe = pipeline(
                        "automatic-speech-recognition",
                        model=config.WHISPER_MODEL,
                        device=dev_id,
                        torch_dtype=torch_dtype,
                        chunk_length_s=30,
                        stride_length_s=5,
                    )
                    _whisper_pipe = pipe
                except Exception as exc:
                    err = (
                        f"Failed to load Whisper model '{config.WHISPER_MODEL}' on {device}: {exc}. "
                        "Please check internet connection for first download."
                    )
                    logger.error(err)
                    raise RuntimeError(err) from exc
    return _whisper_pipe
