from __future__ import annotations

"""Utility functions for data serialization, type conversions, and file checks."""

import dataclasses
import datetime
import math
import pathlib
from typing import Any, Dict, List, Set, Union

import numpy as np
import torch

import config


def to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy, PyTorch, Path, and datetime objects into JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if torch.is_tensor(obj):
        detached = obj.detach().cpu()
        if detached.ndim == 0:
            return to_jsonable(detached.item())
        return to_jsonable(detached.tolist())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.generic, np.bool_)):
        return obj.item()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if dataclasses.is_dataclass(obj):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(item) for item in obj]
    return str(obj)


def round_conf(x: Any) -> float:
    """Round a confidence value to 3 decimal places clamped strictly within [0.0, 1.0]."""
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return max(0.0, min(1.0, round(val, 3)))
    except (TypeError, ValueError):
        return 0.0


def safe_float(x: Any, default: float = 0.0) -> float:
    """Convert value safely to a float, returning default if conversion fails or is NaN/inf."""
    try:
        val = float(x)
        if math.isnan(val) or math.isinf(val):
            return default
        return val
    except (TypeError, ValueError):
        return default


def ensure_dirs() -> None:
    """Ensure all required runtime data and media directories exist."""
    for path in (config.DATA_DIR, config.UPLOAD_DIR, config.SESSION_DIR, config.REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str, allowed_set: Set[str]) -> bool:
    """Check whether a filename possesses an allowed extension."""
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    normalized_allowed = {s.lstrip(".").lower() for s in allowed_set}
    return ext in normalized_allowed
