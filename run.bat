@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Starting CrimeVision AI Forensic Command Center...
echo ===================================================

cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else (
    echo [WARNING] Virtual environment not found in venv\ or .venv\. Using system Python...
)

python app.py
pause
