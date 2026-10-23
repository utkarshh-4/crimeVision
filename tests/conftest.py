from __future__ import annotations

"""Pytest shared fixtures and test configuration."""

import pathlib
import sys

# Ensure repository root is in sys.path for test imports
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
