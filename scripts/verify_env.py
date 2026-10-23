from __future__ import annotations

"""Environment verification script for CrimeVision AI dependencies and configuration."""

import importlib
import importlib.metadata
import os
import shutil
import sys
import warnings
from typing import List, Tuple

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")


def main() -> int:
    print("=" * 60)
    print("CrimeVision AI - Environment Verification")
    print("=" * 60)

    failures: List[str] = []

    # 1. Check Python version (>= 3.9 required)
    py_ver = sys.version_info
    py_ver_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    print(f"Python Version: {py_ver_str}")
    if py_ver < (3, 9):
        print(f"  [FAIL] Python >= 3.9 is required, found {py_ver_str}")
        failures.append(f"Python >= 3.9 (current: {py_ver_str})")
    else:
        print("  [OK] Python version is >= 3.9")

    # 2. Check required package imports and versions
    required_packages: List[Tuple[str, str]] = [
        ("flask", "flask"),
        ("werkzeug", "werkzeug"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("timm", "timm"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("opencv-python-headless", "cv2"),
        ("Pillow", "PIL"),
        ("librosa", "librosa"),
        ("soundfile", "soundfile"),
        ("groq", "groq"),
        ("reportlab", "reportlab"),
        ("requests", "requests"),
        ("pytest", "pytest"),
        ("python-dotenv", "dotenv"),
    ]

    print("\nChecking Package Imports:")
    imported_modules = {}
    for pkg_name, mod_name in required_packages:
        try:
            mod = importlib.import_module(mod_name)
            imported_modules[mod_name] = mod
            try:
                ver = importlib.metadata.version(pkg_name)
            except Exception:
                ver = getattr(mod, "__version__", "available")
            print(f"  [OK] {pkg_name} ({mod_name}) == {ver}")
        except Exception as exc:
            print(f"  [FAIL] {pkg_name} ({mod_name}): {exc}")
            failures.append(f"{pkg_name} ({mod_name})")

    # 3. Check PyTorch CUDA availability and chosen device
    print("\nChecking PyTorch Device:")
    if "torch" in imported_modules:
        torch_mod = imported_modules["torch"]
        cuda_avail = bool(torch_mod.cuda.is_available())
        chosen_device = "cuda" if cuda_avail else "cpu"
        print(f"  torch.cuda.is_available(): {cuda_avail}")
        print(f"  Chosen device: {chosen_device}")
        if cuda_avail:
            device_name = torch_mod.cuda.get_device_name(0)
            print(f"  CUDA Device Name: {device_name}")
    else:
        print("  [SKIP] torch not imported")

    # 4. Check NumPy major version == 1
    print("\nChecking NumPy Compatibility:")
    if "numpy" in imported_modules:
        np_mod = imported_modules["numpy"]
        np_ver = getattr(np_mod, "__version__", "0.0.0")
        major_ver = int(np_ver.split(".")[0])
        if major_ver != 1:
            print(
                f"  [WARNING] NumPy major version is {major_ver} ({np_ver})! "
                "PyTorch 2.4 wheels expect numpy<2. Please pin numpy==1.26.4."
            )
            failures.append(f"NumPy major version 1 required (found {np_ver})")
        else:
            print(f"  [OK] NumPy major version is 1 ({np_ver})")
    else:
        print("  [SKIP] numpy not imported")

    # 5. Check specific Transformers classes import
    print("\nChecking Transformers Model & Pipeline Classes:")
    try:
        from transformers import (  # noqa: F401
            RTDetrForObjectDetection,
            RTDetrImageProcessor,
            CLIPModel,
            BlipForConditionalGeneration,
            pipeline,
        )
        print("  [OK] RTDetrForObjectDetection, RTDetrImageProcessor, CLIPModel, "
              "BlipForConditionalGeneration, pipeline imported successfully")
    except Exception as exc:
        print(f"  [FAIL] Transformers class imports failed: {exc}")
        failures.append(f"Transformers classes: {exc}")

    # 6. Check GROQ_API_KEY environment variable (never log the actual key)
    print("\nChecking Environment Configuration:")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    key_status = "set" if bool(groq_key) else "missing"
    print(f"  GROQ_API_KEY: {key_status}")

    # 7. Check optional ffmpeg executable
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"  ffmpeg (OPTIONAL): Available at {ffmpeg_path}")
    else:
        print("  ffmpeg (OPTIONAL): Not found in PATH (librosa handles audio decoding directly)")

    # Summary
    print("\n" + "=" * 60)
    if failures:
        print(f"ENVIRONMENT FAILED: {', '.join(failures)}")
        print("=" * 60)
        return 1

    print("ENVIRONMENT READY")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
