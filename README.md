# CrimeVision AI – Intelligent Multimodal Forensic Analysis Platform

CrimeVision AI is a forensic analysis platform that processes crime-scene image, video, and audio evidence using local multimodal AI models, fuses the extracted information into a chronological timeline, supports grounded AI investigations via Groq LLM, reconstructs 3D crime scenes, and exports comprehensive forensic PDF reports.

## Setup

### Prerequisites
- Python 3.9 or higher (Python 3.11 recommended)

### 1. Environment Setup

#### Windows (PowerShell)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 2. Configuration & API Keys
Copy the example environment configuration file:
- **Windows (PowerShell)**:
  ```powershell
  Copy-Item .env.example .env
  ```
- **macOS / Linux**:
  ```bash
  cp .env.example .env
  ```

Open `.env` and set your Groq API key:
```ini
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Verify Environment
Run the environment verification script to ensure all packages and device capabilities are correctly configured:
```bash
python scripts/verify_env.py
```

## Optional Extras
- `segment-anything`: Optional for high-precision segmentation masks; omitted from core dependencies due to weight.
