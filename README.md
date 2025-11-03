# COMPASS - GDPR Privacy Policy Auditor

COMPASS (Compliance Oriented Mapping & Privacy Assessment System) is an AI-assisted GDPR auditor that ingests privacy policies (PDF, DOCX, TXT, or URL) and produces a color-coded compliance report. The MVP demonstrates automated document segmentation, Legal-BERT powered analysis, and actionable summaries that distinguish compliant (green), ambiguous (yellow), and non-compliant (red) sections.

## Features

- **Hybrid chunking** that preserves the existing paragraph/sentence logic for robust clause segmentation.
- **Legal-BERT prototype classifier** that scores each chunk as compliant, ambiguous, or non-compliant and surfaces the closest reference rationale.
- **Rule-based GDPR detectors** that highlight high-risk keywords (e.g., "without consent") and adjust the final status when legal cues are present.
- **API endpoints** for file uploads, raw text, and URL-based analysis, returning a structured JSON report with counts, confidence, and supporting evidence.
- **Interactive frontend dashboard** that visualizes sections with green/yellow/red highlights, displays keyword chips, and lets users download the JSON report.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) with optional NVIDIA GPU support for acceleration (COMPASS automatically falls back to CPU).
- Python 3.10+ is required only if you plan to run the backend without Docker.

## Quick Start

1. **Clone the repository**

    ```powershell
    git clone <your-repo-url>
    cd COMPASS-GDPR-Auditor
    ```

2. **Launch the backend (FastAPI + Legal-BERT)**

    ```powershell
    docker-compose up --build
    ```

    The first build downloads PyTorch, transformers, and the Legal-BERT weights; subsequent runs are much faster thanks to cached volumes.

3. **Open the API docs (optional)**

    Visit [http://localhost:8000/docs](http://localhost:8000/docs) to experiment with:

    - `POST /analyze/file`
    - `POST /analyze/text`
    - `POST /analyze/url`

    Each endpoint returns the structured report consumed by the frontend.

4. **Use the frontend dashboard**

    Serve the static files (any HTTP server works). For example:

    ```powershell
    cd frontend
    python -m http.server 5173
    ```

    Open [http://localhost:5173](http://localhost:5173) in your browser, submit a privacy policy, and view the interactive compliance report. Ensure the backend is running at `http://localhost:8000` (the default API base URL).

## Output Overview

Each response bundles:

- **Metadata** – source name, chunk count, runtime device, timestamps.
- **Summary** – compliant/ambiguous/non-compliant totals, compliance ratio, and the most notable GDPR signals.
- **Chunks** – per-section details including the final status, AI confidence, contributing rule matches, and the prototype snippet Legal-BERT used for similarity.

The frontend maps these statuses to the green/red/yellow visual scheme and provides a one-click JSON download for downstream reporting.

## Future Enhancements

- Extend training data and prototype sets with domain-specific corpora or fine-tuned Legal-BERT checkpoints.
- Generate full PDF/CSV exports with clause-level commentary.
- Layer in additional jurisdictions (e.g., India’s DPDP Act) and rule+ML hybrid scoring.