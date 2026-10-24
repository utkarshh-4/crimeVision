# CrimeVision AI: Multimodal Forensic Analysis & Scene Reconstruction

> **Autonomous Digital Forensics & Cross-Modal Evidentiary Timeline Platform**  
> Corroborating surveillance video, acoustic wiretaps, and crime-scene imagery into grounded timelines, interactive 3D spatial reconstructions, and court-ready PDF dossiers.

---

## 1. Project Overview

**CrimeVision AI** is an advanced, multimodal forensic investigation platform designed for legal analysts, law enforcement investigators, and digital forensic examiners. When multiple disjoint pieces of evidence (CCTV footage, 911/witness audio recordings, and forensic scene photography) are submitted, CrimeVision AI orchestrates a concurrent neural processing pipeline that:

1. **Detects & Tracks Physical Evidence**: Identifies subjects (Humans), weapons (knives, firearms via zero-shot contrastive verification), and contextual objects using RT-DETR and zero-shot CLIP.
2. **Generates Acoustic Transcripts**: Transcribes multi-lingual audio statements (Telugu, Hindi, French, English, and auto-detection) using Whisper with fine-grained timestamp alignment.
3. **Cross-Modally Corroborates Events**: Maps video detections and acoustic events onto a unified, 1-second master timeline clock, identifying anchor events and flagging contradictions/conflicts.
4. **Formulates Forensic Hypotheses**: Uses LLaMA 3.3 via Groq (with deterministic rule-based fallbacks) to formulate probabilistic scenario explanations.
5. **Projects Interactive 3D Scenes**: Reconstructs physical subject coordinates into an interactive pseudo-3D scene powered by Three.js r128 with video scrub controls, motion trails, and raycasted metadata inspection.
6. **Produces Publication-Grade Reports**: Generates formal, downloadable forensic PDF dossiers with ReportLab Platypus, including annotated evidence imagery, keyframe strips, tracking line charts, and legal disclaimers.

---

## 2. System Architecture

```
                                  USER BROWSER
                     (Dark Command Center Dashboard / SPA)
                                       │
     ┌─────────────────────────────────┼─────────────────────────────────┐
     │ Upload Evidence                 │ 3D Spatial Scene                │ Forensic Chat
     │ (Image / Audio / Video)         │ (Three.js r128 OrbitControls)   │ (Grounded Querying)
     └─────────────────────────────────┼─────────────────────────────────┘
                                       │ HTTP / REST
                                       ▼
                       FLASK GATEWAY (routes/api.py)
   ┌────────────────────────────────────────────────────────────────────────┐
   │ /api/health   /api/analyze   /api/status   /api/session   /api/chat   /api/report │
   └───────────────────────────────────┬────────────────────────────────────┘
                                       │ Async Dispatch
                                       ▼
                     PIPELINE ORCHESTRATOR (modules/pipeline.py)
             ┌─────────────────────────┼─────────────────────────┐
             │                         │                         │
             ▼                         ▼                         ▼
      IMAGE ANALYSIS            AUDIO ANALYSIS            VIDEO ANALYSIS
 (modules/image_analysis.py) (modules/audio_analysis.py) (modules/video_analysis.py)
  • RT-DETR Detection       • Librosa Preprocessing   • 1 Hz Frame Sampling
  • CLIP Scene Classify     • Whisper ASR Pipeline    • RT-DETR Tracking
  • CLIP Firearm Verify     • Acoustic Fact Engine    • Visual Event Anchors
  • BLIP Image Caption      • Multilingual Support    • Taxonomy Classifier
             │                         │                         │
             └─────────────────────────┼─────────────────────────┘
                                       ▼
                       MULTIMODAL FUSION (modules/fusion.py)
     • Master 1-Second Unified Clock Alignment
     • Cross-Modal Temporal Corroboration (Aggression/Threat <-> Weapons)
     • Evidentiary Conflict Flagging (Dispute Detection)
     • Overall Confidence Scoring & Forensic Hypotheses
                                       │
                                       ├──────────────────────────┐
                                       ▼                          ▼
                          EVIDENCE INDEX (modules/chat.py)   PDF REPORT (modules/reports.py)
                           • Second-by-Second Inverted Index  • Platypus A4 Layout
                           • Keyword & Fact Extraction        • Pillow Box Annotations
                           • Grounded Chatbot Queries         • ReportLab LinePlot Chart
```

---

## 3. Forensic AI Models

| Task | Model Identifier | Architecture | Approx Download Size |
| :--- | :--- | :--- | :--- |
| **Object Detection & Tracking** | `PekingU/rtdetr_r50vd` | RT-DETR (ResNet-50) | ~168 MB |
| **Zero-Shot Scene & Firearm Check** | `openai/clip-vit-base-patch32` | CLIP Vision-Text | ~605 MB |
| **Forensic Captioning** | `Salesforce/blip-image-captioning-base` | BLIP Vision-Encoder | ~990 MB |
| **Acoustic Transcription** | `openai/whisper-base` | Whisper Encoder-Decoder | ~290 MB |
| **Grounded LLM Reasoning** | `llama-3.3-70b-versatile` *(Groq API)* | Auto-regressive Transformer | Cloud API (0 MB) |
| **Fallback LLM Reasoning** | `llama-3.1-8b-instant` *(Groq API)* | Auto-regressive Transformer | Cloud API (0 MB) |

---

## 4. Prerequisites

- **Python**: 3.9, 3.10, or 3.11 (tested on Python 3.11.3)
- **Node.js** *(Optional)*: v18+ for running JavaScript syntax audits (`node --check`)
- **System Memory**: Minimum 8 GB RAM (16 GB recommended for CPU inference)
- **Hardware Acceleration** *(Optional)*: NVIDIA CUDA GPU with $\ge 6$ GB VRAM. (Automatic CPU fallback is fully supported).

---

## 5. Setup & Installation

### Windows Setup

1. **Clone or Navigate to the Workspace**:
   ```cmd
   cd C:\PROJECTS\crimeVision
   ```

2. **Create and Activate a Virtual Environment**:
   ```cmd
   python -m venv venv
   call venv\Scripts\activate.bat
   ```

3. **Install Dependencies**:
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prefetch AI Models** *(Recommended for offline & fast demo runs)*:
   ```cmd
   python scripts\prefetch_models.py
   ```

### macOS & Linux Setup

1. **Navigate to the Repository**:
   ```bash
   cd crimeVision
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Prefetch AI Models**:
   ```bash
   python3 scripts/prefetch_models.py
   ```

---

## 6. Environment Configuration (.env)

Copy `.env.example` to `.env` in the repository root:
```ini
# Flask Server Configuration
PORT=5000
FLASK_DEBUG=0

# Groq LLM API Configuration
# Sign up at https://console.groq.com to acquire a free API key.
# If left blank, CrimeVision automatically executes deterministic, rule-based forensic fallbacks.
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FALLBACK_MODEL=llama-3.1-8b-instant

# Video Ingestion Settings
MAX_VIDEO_SECONDS=90
VIDEO_SAMPLE_HZ=1
CAPTION_EVERY_N_FRAMES=3
```

---

## 7. Running the Application

### Method 1: Using the Dedicated Launchers
- **Windows**: Double-click `run.bat` or run:
  ```cmd
  run.bat
  ```
- **macOS / Linux**: Execute:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

### Method 2: Direct CLI
```bash
python app.py
```
Open your browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 8. Running Automated Tests

CrimeVision AI includes a comprehensive test suite with 46 isolated tests utilizing mock singletons, synthetic media generators, and schema validation helpers:

```bash
# Run entire test suite
pytest

# Run with verbose output
pytest -v

# Run individual test modules
pytest tests/test_image.py
pytest tests/test_api.py
pytest tests/test_pipeline_end_to_end.py
pytest tests/test_reports.py
```

---

## 9. Usage Walkthrough

1. **Upload Evidence**:
   - In the left **Upload Evidence** card, select an image (JPG/PNG), an audio clip (WAV/MP3), and/or a video recording (MP4/AVI).
   - Choose transcription language (`Auto Detect`, `Telugu`, `Hindi`, `French`, or `English`).
   - Click **Analyse Evidence**.
2. **Observe Pipeline Progress**:
   - The **Pipeline Status** card tracks real-time stage execution (`Image Analysis` $\to$ `Audio Transcription` $\to$ `Video & Tracking` $\to$ `Multimodal Fusion`).
3. **Examine Evidence**:
   - **Evidence Tab**: View detection boxes overlaid on the image canvas (Green = Human, Red = Weapon, Blue = Object), scene classification, transcript segments, and 3 forensic hypotheses.
   - Click on any transcript timestamp to automatically seek the audio and video players.
4. **Inspect the Corroborated Timeline**:
   - **Timeline Tab**: Interactive horizontal floating-bar chart color-coded by modality with gold borders for anchor events and red borders for conflicts.
   - Per-second line chart tracks subject and weapon counts over time, synchronized with the video playhead.
5. **Query the Forensic Chatbot**:
   - In the right-hand panel, ask questions: *"What happened?"*, *"Who was present at 4s?"*, or *"Were weapons seen?"*.
   - Responses cite exact timestamps that act as clickable media seeks.
6. **3D Spatial Crime-Scene Projection**:
   - Open the **3D Scene** tab. Orbit, pan, and zoom around the ground grid.
   - Scrub the video timeline slider to watch subjects and weapons move across the ground plane with motion trails.
   - Hover over any entity to inspect its 3D coordinates, raw label, and bounding box.
7. **Download Court Dossier**:
   - In the **Report** tab, click **Download PDF Report** to generate a publication-grade ReportLab dossier.

---

## 10. API Reference Table

| Endpoint | Method | Payload / Params | Response | Description |
| :--- | :---: | :--- | :---: | :--- |
| `/api/health` | `GET` | None | `200 OK` | Reports server status, active compute device (`cuda`/`cpu`), and LLM availability. |
| `/api/analyze` | `POST` | `multipart/form-data`: `image`, `video`, `audio`, `language` | `202 Accepted` | Ingests media, creates a 32-hex `session_id`, and dispatches background pipeline. |
| `/api/status/<sid>` | `GET` | Session ID URL param | `200 OK` | Returns execution status (`queued`, `running`, `done`, `error`) and stage progress. |
| `/api/session/<sid>` | `GET` | Session ID URL param | `200 OK` | Retrieves complete persisted forensic JSON dossier (all modalities and fusion). |
| `/api/chat` | `POST` | `{"session_id": str, "message": str, "history": list}` | `200 OK` | Processes grounded conversational forensic query against index. |
| `/api/media/<sid>/<kind>`| `GET` | `kind`: `image`, `video`, or `audio` | `200 OK` | Streams uploaded media with HTTP Range support for seeking. |
| `/api/report/<sid>` | `GET` | Session ID URL param | `200 OK` | Generates and streams downloadable forensic PDF report (`application/pdf`). |

---

## 11. Troubleshooting

- **CUDA Out-Of-Memory (OOM)**:
  - If processing high-resolution video causes GPU memory exhaustion, lower `MAX_VIDEO_SECONDS` in `.env` (e.g. `MAX_VIDEO_SECONDS=30`), or force CPU execution by setting environment variable `CUDA_VISIBLE_DEVICES=""`.
- **First Run Initial Model Download Delay**:
  - The first time CrimeVision AI analyzes evidence without prefetching, it will download ~2.0 GB of models from HuggingFace. Run `python scripts/prefetch_models.py` once to cache all weights ahead of time.
- **Groq API Rate Limits / Authentication (401 / 429)**:
  - If your Groq API key is expired or rate-limited, CrimeVision AI logs a warning and automatically falls back to deterministic rule-based hypotheses without crashing.
- **NumPy 2.x Compatibility Issues**:
  - PyTorch 2.4 and torchvision are pinned to `numpy==1.26.4` in `requirements.txt`. If you encounter C-extension attribute errors, run `pip install "numpy<2"`.
- **Port 5000 Already In Use**:
  - On macOS, AirPlay Receiver uses port 5000 by default (disable AirPlay Receiver in System Settings) or change `PORT=5050` in `.env`.

---

## 12. Known Limitations & Forensic Disclaimers

1. **COCO Taxonomy Weapon Boundaries**:
   - The standard COCO detection taxonomy used by RT-DETR includes `knife`, `scissors`, and `baseball bat`, but does not have a native `handgun` or `rifle` class. CrimeVision AI compensates by combining RT-DETR with zero-shot contrastive CLIP firearm verification (`check_firearm`).
2. **Whisper-Base Transcription on Low-Resource Languages**:
   - `whisper-base` offers high-speed transcription for English and French, but may exhibit lower phoneme accuracy on regional Telugu or noisy Hindi audio. For critical operations, evaluate larger checkpoints.
3. **Forensic Expert Verification Requirement**:
   - All AI-generated scenario hypotheses, acoustic transcripts, and cross-modal timeline alignments are decision-support heuristics designed to aid investigators and do **not** constitute sworn legal evidence without independent human validation.

---

## 13. Optional Extras & Extensibility

- **Segment Anything (SAM) Instance Segmentation**: Replace rectangular bounding boxes with pixel-accurate polygon segmentation masks.
- **Whisper-Small / Medium**: Upgrade `WHISPER_MODEL=openai/whisper-small` in `.env` for elevated multilingual accuracy.
- **BLIP-2 / LLaVA Vision-Language Reasoning**: Incorporate open-source visual instruction models for conversational query answering directly against video frames.
