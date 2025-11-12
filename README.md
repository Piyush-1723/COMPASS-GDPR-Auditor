# COMPASS - GDPR Privacy Policy Auditor

COMPASS (Compliance Oriented Mapping & Privacy Assessment System) is an AI-assisted GDPR auditor that ingests privacy policies (PDF, DOCX, TXT, or URL) and produces a color-coded compliance report. It combines a fine‑tuned Legal‑BERT classifier with light rule indicators and exports a full PDF report.

## Features

- Hybrid chunking for robust clause segmentation.
- Fine‑tuned Legal‑BERT classifier (primary decision source). Rule indicators are shown as context but do not override AI decisions.
- HTML→PDF export powered by WeasyPrint for reliable Unicode wrapping and layout.
- FastAPI endpoints for files, raw text, and URLs, returning structured JSON.
- Simple frontend to upload, review results, and download the PDF report.

## Prerequisites

- Docker Desktop (recommended) with NVIDIA GPU support for acceleration. The app automatically falls back to CPU.
- If running outside Docker: Python 3.10+ and system libraries for WeasyPrint (Cairo/Pango/GDK‑PixBuf).

## Model location (Option A – active)

The fine‑tuned model is stored at `training/models/legalbert_3way/` and mounted into the backend container at `/app/model_store/legalbert_3way` via `docker-compose.yml`. You can replace this directory with your own model artifacts (config.json, model.safetensors, tokenizer files) without changing code.

Optional: You can set `MODEL_DIR` env var to point to a custom path inside the container; by default the mounted path is used when present.

## Run the stack

1) Clone the repository

```powershell
git clone <your-repo-url>
cd COMPASS-GDPR-Auditor
```

2) Start the backend (FastAPI)

```powershell
docker compose up --build
```

Backend will listen on http://localhost:8000

3) Open API docs (optional)

Visit http://localhost:8000/docs and try:

- POST /analyze/file
- POST /analyze/text
- POST /analyze/url
- POST /report/pdf (export the full PDF report)
- GET /report/pdf/test (tiny smoke test for PDF renderer)
- GET /system/info (reports GPU availability, torch version, WeasyPrint/pydyf versions)

4) Run the frontend (static)

```powershell
cd frontend
python -m http.server 5173
```

Open http://localhost:5173 and point it at http://localhost:8000 for the API.

## Project structure

```
COMPASS-GDPR-Auditor/
├─ docker-compose.yml            # Backend service and model mount
├─ backend/
│  ├─ Dockerfile                 # CUDA-enabled PyTorch base image + WeasyPrint libs
│  ├─ main.py                    # FastAPI app + WeasyPrint PDF export
│  ├─ model_loader.py            # Loads fine‑tuned classifier or prototype fallback
│  ├─ data/
│  │  └─ fine_tuned_samples.json # Prototype text (kept for fallback)
│  ├─ requirements.txt
│  └─ .dockerignore
├─ frontend/
│  ├─ index.html
│  ├─ javascript.js
│  └─ style.css
├─ training/
│  ├─ train.py, infer.py, *.csv  # Training code and datasets
│  └─ models/
│     └─ legalbert_3way/         # Final model artifacts mounted by compose
└─ test_docs/                    # Sample inputs for quick testing
```

## Environment variables

- MODEL_DIR (optional): override the model path inside the container. By default, the model from `training/models/legalbert_3way` is mounted to `/app/model_store/legalbert_3way`.

## File types supported

- PDF (text extraction via pypdf)
- DOCX (python-docx)
- TXT (UTF‑8)
- URL (HTML fetched and cleaned via BeautifulSoup)

## PDF generation

The app uses WeasyPrint for HTML→PDF export. System fonts (DejaVu) are installed in the Docker image, and styles ensure robust wrapping for long tokens and mixed Unicode. If you deploy outside Docker, install Cairo, Pango, and GDK‑PixBuf.

## Troubleshooting

- PDF engine smoke test: GET `/report/pdf/test` should return a small PDF.
- Versions: GET `/system/info` shows `weasyprint_version` and `pydyf_version`.
- GPU availability is reported under `gpu_available`; the app still works on CPU.

## Git and large files

Model weights can be large. Consider using Git LFS to store artifacts like `*.safetensors`:

```powershell
git lfs install
git lfs track "training/models/**/model.safetensors"
git add .gitattributes
```

This repository includes a `.gitignore` to exclude common caches (`__pycache__`, `.venv`, etc.) and a backend `.dockerignore` to keep Docker builds fast.

## Development notes

- The backend no longer uses fpdf2. WeasyPrint is the only PDF engine.
- Rule indicators are kept for transparency; the final status is always AI‑driven.
- Duplicate model directories were removed in favor of the mount at `training/models/legalbert_3way`.

## Troubleshooting

- If `/report/pdf` fails, check `/system/info` for WeasyPrint/pydyf versions. They should be compatible (this repo pins them).
- If running with GPU, ensure NVIDIA runtime is enabled in Docker Desktop. The app still works on CPU.

## License

TBD.