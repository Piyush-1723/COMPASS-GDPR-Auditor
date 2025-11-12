# COMPASS – GDPR Privacy Policy Auditor

Professional, minimal overview of the system.

## 1. Overview
COMPASS (Compliance Oriented Mapping & Privacy Assessment System) analyzes privacy policies (PDF, DOCX, TXT, or URL) and produces a structured GDPR compliance report. A fine‑tuned Legal‑BERT classifier determines status (compliant / ambiguous / non‑compliant). Rule indicators are shown for context only. Reports can be exported as PDF via WeasyPrint.

## 2. Core Features
1. Fine‑tuned Legal‑BERT classifier (primary decision logic).
2. Hybrid text chunking (paragraph+sentence) to stay within model limits.
3. Rule keyword indicators (do not override AI decision).
4. HTML → PDF export with WeasyPrint (Unicode-safe, robust wrapping).
5. FastAPI backend + simple static frontend.

## 3. Requirements
| Component | Recommended |
|-----------|------------|
| Runtime   | Docker Desktop (GPU optional) |
| Python (non-Docker) | 3.10+ with Cairo, Pango, GDK-PixBuf |
| GPU (optional) | NVIDIA with CUDA (auto fallback to CPU) |

## 4. Model Location (Option A Active)
Fine‑tuned model lives at `training/models/legalbert_3way/` and is mounted into the container as `/app/model_store/legalbert_3way` by `docker-compose.yml`. Override with env var `MODEL_DIR` if needed.

Required model files:
- config.json, model.safetensors, tokenizer.json, tokenizer_config.json, special_tokens_map.json, vocab.txt

## 5. Quick Start (Windows PowerShell)
```powershell
# 1. Clone
git clone <your-repo-url>
cd COMPASS-GDPR-Auditor

# 2. Start backend (build first time)
docker compose up --build

# 3. (Optional) Serve frontend
cd frontend
python -m http.server 5173
```
Backend: http://localhost:8000  |  Frontend: http://localhost:5173

## 6. Key API Endpoints
| Method | Path               | Purpose |
|--------|--------------------|---------|
| GET    | /system/info       | Runtime + versions (GPU, WeasyPrint, pydyf) |
| POST   | /analyze/file      | Analyze uploaded PDF/DOCX/TXT |
| POST   | /analyze/text      | Analyze raw pasted text |
| POST   | /analyze/url       | Fetch + analyze web page |
| POST   | /report/pdf        | Generate full PDF compliance report |
| GET    | /report/pdf/test   | Smoke test for PDF engine |

## 7. Standard Workflow
1. Start backend (`docker compose up --build`).
2. (Optional) Serve frontend static files.
3. Upload or paste policy text via UI (or call API directly).
4. Review JSON status counts and per‑section details.
5. Export PDF (`POST /report/pdf`).
6. Monitor `/system/info` for environment diagnostics.

## 8. Project Structure

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

## 9. Environment Variable

- MODEL_DIR (optional): override the model path inside the container. By default, the model from `training/models/legalbert_3way` is mounted to `/app/model_store/legalbert_3way`.

## 10. Supported Input Types

- PDF (text extraction via pypdf)
- DOCX (python-docx)
- TXT (UTF‑8)
- URL (HTML fetched and cleaned via BeautifulSoup)

## 11. PDF Export

The app uses WeasyPrint for HTML→PDF export. System fonts (DejaVu) are installed in the Docker image, and styles ensure robust wrapping for long tokens and mixed Unicode. If you deploy outside Docker, install Cairo, Pango, and GDK‑PixBuf.

## 12. Troubleshooting

- PDF engine smoke test: GET `/report/pdf/test` should return a small PDF.
- Versions: GET `/system/info` shows `weasyprint_version` and `pydyf_version`.
- GPU availability is reported under `gpu_available`; the app still works on CPU.

## 13. Git & Large Files

Model weights can be large. Consider using Git LFS to store artifacts like `*.safetensors`:

```powershell
git lfs install
git lfs track "training/models/**/model.safetensors"
git add .gitattributes
```

This repository includes a `.gitignore` to exclude common caches (`__pycache__`, `.venv`, etc.) and a backend `.dockerignore` to keep Docker builds fast.

## 14. Development Notes

- The backend no longer uses fpdf2. WeasyPrint is the only PDF engine.
- Rule indicators are kept for transparency; the final status is always AI‑driven.
- Duplicate model directories were removed in favor of the mount at `training/models/legalbert_3way`.

## 15. Additional Notes

- If `/report/pdf` fails, check `/system/info` for WeasyPrint/pydyf versions. They should be compatible (this repo pins them).
- If running with GPU, ensure NVIDIA runtime is enabled in Docker Desktop. The app still works on CPU.

## 16. License
TBD