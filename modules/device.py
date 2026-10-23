from __future__ import annotations

"""Device and precision management for CrimeVision AI models."""

import torch


def get_device() -> torch.device:
    """Return torch.device('cuda') if CUDA is available, otherwise torch.device('cpu')."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dtype(device: torch.device | str) -> torch.dtype:
    """Return torch.float16 for CUDA devices, otherwise torch.float32."""
    if isinstance(device, str):
        device = torch.device(device)
    return torch.float16 if device.type == "cuda" else torch.float32
