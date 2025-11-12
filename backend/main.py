import datetime as dt
import io
import re
import unicodedata
from typing import Any, Dict, List, Tuple

import docx
import nltk
import requests
import torch
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from weasyprint import HTML, __version__ as weasy_version
import importlib
import inspect
from pydantic import BaseModel, HttpUrl
from pypdf import PdfReader
from html import escape as html_escape

from model_loader import (
    CATEGORY_COLORS,
    ComplianceResult,
    get_compliance_classifier,
    get_legal_tokenizer,
    get_active_ai_source,
)

# --- NLTK resource management ---
REQUIRED_NLTK_PACKAGES = ("punkt", "punkt_tab")
nltk_ready = False


def ensure_nltk_resources() -> None:
    global nltk_ready
    if nltk_ready:
        return

    for resource in REQUIRED_NLTK_PACKAGES:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    nltk_ready = True


# --- GDPR Clause Labels and Keywords for Rule-Based Scoring ---
CLAUSE_KEYWORDS = {
    "data_collection": ["collect", "gather", "information from users"],
    "data_sharing": ["share", "third-party", "without consent"],
    "user_consent": ["consent", "opt-in", "permission"],
    "data_retention": ["retain", "storage period", "retain for"],
    "rights_of_user": ["access", "delete", "portability", "withdraw consent"],
    "security_measures": ["encrypt", "secure", "protected", "safeguard"],
    "non_compliance_warning": ["without consent", "unlawful", "sell data", "no opt-out"],
    "cookies_tracking": ["cookies", "tracking", "analytics"]
}

CLAUSE_LABEL_TITLES = {
    "data_collection": "Data Collection",
    "data_sharing": "Data Sharing",
    "user_consent": "User Consent",
    "data_retention": "Data Retention",
    "rights_of_user": "Rights of Users",
    "security_measures": "Security Measures",
    "non_compliance_warning": "Potential Non-Compliance",
    "cookies_tracking": "Cookies & Tracking",
}

# Mapping labels to risk color
LABEL_RISK_MAPPING = {
    "data_collection": "green",
    "data_sharing": "red",
    "user_consent": "green",
    "data_retention": "yellow",
    "rights_of_user": "green",
    "security_measures": "green",
    "non_compliance_warning": "red",
    "cookies_tracking": "yellow"
}

LABEL_GDPR_ARTICLES = {
    "data_collection": ["Art. 5(1)(a)", "Art. 6"],
    "data_sharing": ["Art. 13", "Art. 14", "Art. 28"],
    "user_consent": ["Art. 6(1)(a)", "Art. 7"],
    "data_retention": ["Art. 5(1)(e)", "Art. 30"],
    "rights_of_user": ["Art. 12", "Art. 15-18", "Art. 21"],
    "security_measures": ["Art. 32"],
    "non_compliance_warning": ["Art. 5", "Art. 83"],
    "cookies_tracking": ["Art. 6(1)(a)", "Art. 7", "Art. 5(3) ePrivacy"],
}

LABEL_RECOMMENDATIONS = {
    "data_collection": [
        "Document lawful bases for all data collection activities.",
        "Update the privacy notice to describe what data is collected and why.",
    ],
    "data_sharing": [
        "List all third parties receiving data and specify their roles.",
        "Obtain explicit consent before sharing personal data for secondary purposes.",
        "Execute controller-processor contracts with each processor (Art. 28).",
    ],
    "user_consent": [
        "Ensure consent is freely given, specific, informed, and unambiguous.",
        "Provide clear opt-in mechanisms with the ability to withdraw consent easily.",
    ],
    "data_retention": [
        "Define concrete retention periods or criteria for each data category.",
        "Implement deletion or anonymisation routines after retention periods expire.",
    ],
    "rights_of_user": [
        "Document procedures to respond to data subject requests within one month.",
        "Provide self-service tools or contact channels for exercising data subject rights.",
    ],
    "security_measures": [
        "Review technical and organisational measures regularly for effectiveness.",
        "Implement encryption, access controls, and incident response plans.",
    ],
    "non_compliance_warning": [
        "Perform a DPIA to identify and mitigate high-risk processing activities.",
        "Engage legal counsel to align processing activities with GDPR principles.",
    ],
    "cookies_tracking": [
        "Deploy a consent banner that records affirmative choices before setting non-essential cookies.",
        "Offer granular controls for analytics, advertising, and functional cookies.",
    ],
}

STATUS_RISK_LEVEL = {
    "non_compliant": "High",
    "ambiguous": "Medium",
    "compliant": "Low",
}

STATUS_TITLES = {
    "non_compliant": "Non-Compliant",
    "ambiguous": "Ambiguous",
    "compliant": "Compliant",
}

STATUS_DEFAULT_ARTICLES = {
    "non_compliant": ["Art. 5", "Art. 6"],
    "ambiguous": ["Art. 5(1)(a)", "Art. 24"],
    "compliant": ["Art. 24", "Art. 32"],
}

STATUS_DEFAULT_RECOMMENDATIONS = {
    "non_compliant": [
        "Address identified legal gaps before continuing the processing activity.",
        "Update governance documentation to evidence compliance with GDPR principles.",
    ],
    "ambiguous": [
        "Clarify ambiguous language in the privacy policy with specific legal bases.",
        "Add detail about user rights, retention periods, or safeguards to remove uncertainty.",
    ],
    "compliant": [
        "Maintain records of processing and monitor changes to ensure continued compliance.",
    ],
}

# --- Constants ---
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_TOKENS = 512

# --- Lazy-loaded AI Model and Tokenizer ---
compliance_classifier = None
tokenizer = None

# --- Create FastAPI App ---
app = FastAPI(title="COMPASS API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextAnalysisRequest(BaseModel):
    content: str
    document_name: str | None = None


class UrlAnalysisRequest(BaseModel):
    url: HttpUrl


class PdfReportRequest(BaseModel):
    metadata: Dict[str, Any]
    summary: Dict[str, Any]
    chunks: List[Dict[str, Any]]


# --- Helper function to load the model on first use ---
def load_model() -> None:
    global compliance_classifier, tokenizer
    if compliance_classifier is None:
        print("--- Loading Legal-BERT compliance classifier ---")
        compliance_classifier = get_compliance_classifier()
        print("--- Legal-BERT classifier ready. ---")
    if tokenizer is None:
        tokenizer = get_legal_tokenizer()
    ensure_nltk_resources()


# --- Helper functions for text extraction ---
def extract_text_from_docx(file_stream: io.BytesIO) -> str:
    document = docx.Document(file_stream)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])

def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    reader = PdfReader(file_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_txt(file_stream: io.BytesIO) -> str:
    return file_stream.read().decode("utf-8")

# --- Hybrid chunking function ---
def hybrid_chunking(text: str) -> list[str]:
    """Split text into manageable chunks within model token limits.

    Uses tokenizer.encode to count tokens (tokenizer.tokenize may be missing in some builds).
    """
    load_model()
    chunks: List[str] = []
    normalized_text = text.replace("\r\n", "\n")
    paragraphs = re.split(r"\n\s*\n", normalized_text)

    for paragraph in paragraphs:
        cleaned = paragraph.strip()
        if not cleaned:
            continue
        # Rough token estimation using word count to avoid strict tokenizer dependency
        word_count = len(cleaned.split())
        if word_count <= 300:
            chunks.append(cleaned)
        else:
            sentences = nltk.sent_tokenize(cleaned)
            chunks.extend(s.strip() for s in sentences if s.strip())

    return chunks

# --- Rule-based labeling ---
RULE_LABEL_STATUS = {
    "non_compliance_warning": "non_compliant",
    "data_sharing": "non_compliant",
    "data_collection": "ambiguous",
    "data_retention": "ambiguous",
    "cookies_tracking": "ambiguous",
    "user_consent": "compliant",
    "rights_of_user": "compliant",
    "security_measures": "compliant",
}

STATUS_PRIORITY = ["non_compliant", "ambiguous", "compliant"]


def collect_rule_matches(clause: str) -> List[Dict[str, str]]:
    lower_clause = clause.lower()
    matches: List[Dict[str, str]] = []
    for label, keywords in CLAUSE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower_clause:
                matches.append(
                    {
                        "label": label,
                        "keyword": keyword,
                        "risk": LABEL_RISK_MAPPING[label],
                        "status": RULE_LABEL_STATUS[label],
                    }
                )
    return matches


def determine_status(ai_status: str, _rule_matches: List[Dict[str, str]]) -> Tuple[str, str]:
    """Always use the AI-driven decision; ignore rule overrides while retaining indicators."""
    ai_src = get_active_ai_source()
    return ai_status, f"ai:{ai_src}"


def describe_status_source(source: str | None) -> str:
    if not source:
        return "AI classifier"
    if source.startswith("ai:"):
        _, _, mode = source.partition(":")
        return "Fine-tuned AI classifier" if mode == "classifier" else "Prototype similarity model"
    return "AI classifier"


def safe_text(text: str) -> str:
    if not text:
        return ""

    # Normalize common punctuation and whitespace variants
    replacements = {
        "\u2022": "-",  # • bullet to hyphen
        "\u2013": "-",  # – en dash
        "\u2014": "-",  # — em dash
        "\u201C": '"',   # “
        "\u201D": '"',   # ”
        "\u2019": "'",  # ’
        "\u00A0": " ",  # NBSP to space
        "\t": " ",      # tabs to space
        "\u200B": "",   # zero-width space
        "\u200C": "",   # zero-width non-joiner
        "\u200D": "",   # zero-width joiner
        "\u00AD": "-",  # soft hyphen to hyphen
    }
    sanitized = text
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)

    # Remove other control/format characters except newlines
    sanitized = "".join(
        ch if (ch in "\n\r" or not unicodedata.category(ch).startswith("C")) else ""
        for ch in sanitized
    )

    # Collapse excessive whitespace
    sanitized = re.sub(r"[ \f\v]+", " ", sanitized)

    # Hard-break very long unbroken tokens to avoid width errors in fpdf2
    def breaker(match: re.Match[str]) -> str:
        token = match.group(0)
        chunks = [token[i:i + 20] for i in range(0, len(token), 20)]
        return " ".join(chunks)

    sanitized = re.sub(r"\S{40,}", breaker, sanitized)

    return sanitized


def summarize_text(text: str, limit: int = 700) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    if len(collapsed) <= limit:
        return collapsed
    truncated = collapsed[:limit]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return f"{truncated}..."


# (Removed unused hex_to_rgb helper after migrating fully to HTML/CSS PDF generation)




def generate_pdf_report_weasy(report: Dict[str, Any]) -> bytes:
    """Generate the full PDF report using WeasyPrint (HTML + CSS to PDF).

    Designed to be robust against long words, mixed Unicode, and complex wrapping cases.
    """
    metadata = report.get("metadata", {})
    summary = report.get("summary", {})
    status_counts = summary.get("status_counts", {})
    ratio = summary.get("compliance_ratio", 0)
    flagged_keywords = summary.get("flagged_keywords", [])
    chunks = report.get("chunks", [])

    generated_at = (metadata.get("generated_at") or dt.datetime.utcnow().isoformat()).replace("T", " ")

    def esc(s: Any) -> str:
        return html_escape(str(s) if s is not None else "")

    # Build HTML
    parts: List[str] = []
    parts.append("""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>COMPASS GDPR Compliance Report</title>
    <style>
        @page { size: A4; margin: 14mm 12mm; }
        html, body { font-family: 'DejaVu Sans', 'Segoe UI', Arial, sans-serif; color: #2d3748; }
        body { font-size: 11pt; line-height: 1.35; }
        h1 { font-size: 18pt; margin: 0 0 6pt 0; }
        h2 { font-size: 13pt; color: #2c5282; margin: 10pt 0 6pt 0; }
        h3 { font-size: 11.5pt; margin: 8pt 0 4pt 0; }
        p { margin: 4pt 0; }
        small, .muted { color: #475569; }
        .meta { color: #475569; font-size: 10pt; }
        .chip { display: inline-block; padding: 1px 6px; border-radius: 6px; background: #f1f5f9; margin-right: 4px; }
        .chip.red { color: #e11d48; }
        .chip.yellow { color: #ca8a04; }
        .chip.green { color: #16a34a; }
        .section { break-inside: avoid; margin-bottom: 10pt; }
        .block { break-inside: avoid; padding: 6pt 8pt; background: #fafafa; border: 1px solid #e5e7eb; border-radius: 6px; margin: 6pt 0; }
        .kv { margin: 2pt 0; }
        .list { margin: 2pt 0 2pt 10pt; padding: 0; }
        .list li { margin: 2pt 0; }
        .status-title { font-weight: bold; }
        .byline { font-size: 9pt; color: #475569; }
        .code { font-family: 'DejaVu Sans Mono', 'Consolas', monospace; font-size: 9.5pt; }
        /* Robust wrapping for long tokens */
        * { word-break: break-word; overflow-wrap: anywhere; }
        pre, .pre { white-space: pre-wrap; }
    </style>
</head>
<body>
    <header class=\"section\">
        <h1>COMPASS GDPR Compliance Report</h1>
        <div class=\"meta\">
""")
    parts.append(f"      <div>Source: {esc(metadata.get('source', 'Unknown'))} ({esc(metadata.get('source_type', 'n/a'))})</div>")
    parts.append(f"      <div>Generated: {esc(generated_at)} • Sections analyzed: {esc(metadata.get('chunk_count', 0))} • Device: {esc(metadata.get('device', 'CPU'))}</div>")
    parts.append(f"      <div>Word count: {esc(metadata.get('word_count', 0))} • Character count: {esc(metadata.get('character_count', 0))}</div>")
    parts.append("    </div>\n  </header>\n")

    # Snapshot
    parts.append("<section class=\"section\">")
    parts.append("<h2>Overall compliance snapshot</h2>")
    parts.append("<p>Compliance ratio: {pct:.1f}% of sections are compliant</p>".format(pct=ratio * 100))
    parts.append(
        "<p>Breakdown — "
        "<span class=\"chip red\">Non-Compliant: {non}</span> "
        "<span class=\"chip yellow\">Ambiguous: {amb}</span> "
        "<span class=\"chip green\">Compliant: {comp}</span>".format(
            non=int(status_counts.get("non_compliant", 0)),
            amb=int(status_counts.get("ambiguous", 0)),
            comp=int(status_counts.get("compliant", 0)),
        )
    )
    parts.append("</section>")

    # Flagged indicators
    parts.append("<section class=\"section\">")
    parts.append("<h2>Flagged GDPR indicators</h2>")
    if flagged_keywords:
        parts.append("<ul class=\"list\">")
        for item in flagged_keywords:
            label_key = str(item.get("label", "") or "")
            label = (CLAUSE_LABEL_TITLES.get(label_key, label_key) or "").replace("_", " ").title()
            keywords = ", ".join(sorted(item.get("keywords", [])))
            parts.append(
                f"<li><span class=\"status-title\">{esc(label)}:</span> {esc(keywords)} ("
                f"{int(item.get('count', 0))} reference(s))</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p class=\"muted\">No GDPR risk keywords were flagged in this report.</p>")
    parts.append("</section>")

    # Clause-by-clause
    parts.append("<section class=\"section\">")
    parts.append("<h2>Clause-by-clause analysis</h2>")
    if not chunks:
        parts.append("<p class=\"muted\">No clause segments available.</p>")
    else:
        for status in STATUS_PRIORITY:
            status_chunks = [c for c in chunks if c.get("status") == status]
            if not status_chunks:
                continue
            status_title = f"{STATUS_TITLES.get(status, status.title())} sections ({STATUS_RISK_LEVEL.get(status, 'Medium')} risk)"
            color = CATEGORY_COLORS.get(status, "#2d3748")
            parts.append(f"<h3 style=\"color:{esc(color)}\">{esc(status_title)}</h3>")
            for chunk in status_chunks:
                parts.append("<div class=\"block\">")
                parts.append(f"<div class=\"status-title\">Section {esc(chunk.get('id','?'))}</div>")
                parts.append(f"<p class=\"pre\">{esc(summarize_text(chunk.get('text','')))}</p>")
                parts.append(f"<p class=\"byline\">Decision basis: {esc(describe_status_source(chunk.get('status_source')))}</p>")
                articles = ", ".join(chunk.get("gdpr_articles", []) ) or "Not specified"
                parts.append(f"<p class=\"byline\">GDPR articles: {esc(articles)}</p>")
                recs = chunk.get("recommendations", [])
                if recs:
                    parts.append("<div><span class=\"status-title\">Recommended actions:</span><ul class=\"list\">")
                    for rec in recs:
                        parts.append(f"<li>{esc(rec)}</li>")
                    parts.append("</ul></div>")
                rule_matches = chunk.get("rule_matches", [])
                if rule_matches:
                    seen = set()
                    items_html: List[str] = []
                    for match in rule_matches:
                        key = (match.get("label"), match.get("keyword"))
                        if key in seen:
                            continue
                        seen.add(key)
                        label_key = str(match.get("label", "") or "")
                        label = (CLAUSE_LABEL_TITLES.get(label_key, label_key) or "").replace("_", " ").title()
                        items_html.append(f"<li>{esc(label)}: '{esc(match.get('keyword',''))}'</li>")
                    if items_html:
                        parts.append("<div><span class=\"status-title\">Rule indicators:</span><ul class=\"list\">")
                        parts.extend(items_html)
                        parts.append("</ul></div>")
                parts.append("</div>")
    parts.append("</section>")

    # Footer
    parts.append("<p class=\"byline\">Report generated by COMPASS GDPR Auditor</p>")
    parts.append("</body></html>")

    html_str = "".join(parts)
    try:
        pdf_bytes = HTML(string=html_str).write_pdf()
        return pdf_bytes
    except Exception as exc:
        # Provide clearer error context
        raise RuntimeError(f"WeasyPrint PDF generation failed: {type(exc).__name__}: {exc}") from exc
def build_analysis_response(raw_text: str, source_name: str, source_type: str) -> Dict[str, Any]:
    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Document is empty after text extraction.")

    chunks = hybrid_chunking(raw_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not segment the document into chunks.")

    classifier = compliance_classifier
    if classifier is None:
        raise HTTPException(status_code=500, detail="Classifier not initialized.")

    status_counts = {label: 0 for label in CATEGORY_COLORS}
    keyword_tracker: Dict[str, Dict[str, Any]] = {}
    chunk_entries: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks, start=1):
        analysis: ComplianceResult = classifier.classify(chunk)
        rule_matches = collect_rule_matches(chunk)
        status, status_reason = determine_status(analysis.status, rule_matches)
        status_counts[status] += 1

        for match in rule_matches:
            entry = keyword_tracker.setdefault(
                match["label"],
                {
                    "count": 0,
                    "keywords": set(),
                    "risk": match["risk"],
                    "status": match["status"],
                },
            )
            entry["count"] += 1
            entry["keywords"].add(match["keyword"])

        scores_percentage = {label: round(score * 100, 2) for label, score in analysis.scores.items()}

        articles = set()
        for match in rule_matches:
            articles.update(LABEL_GDPR_ARTICLES.get(match["label"], []))
        if not articles:
            articles.update(STATUS_DEFAULT_ARTICLES.get(status, []))

        recommendations = set()
        for match in rule_matches:
            recommendations.update(LABEL_RECOMMENDATIONS.get(match["label"], []))
        recommendations.update(STATUS_DEFAULT_RECOMMENDATIONS.get(status, []))

        chunk_entries.append(
            {
                "id": idx,
                "text": chunk.strip(),
                "status": status,
                "color": CATEGORY_COLORS[status],
                "risk_level": STATUS_RISK_LEVEL[status],
                "status_source": status_reason,
                "scores": scores_percentage,
                "probability": round(analysis.probability * 100, 2),
                "ai_status": analysis.status,
                "ai_probability": round(analysis.probability * 100, 2),
                "raw_similarity": {label: round(value, 3) for label, value in analysis.raw_similarity.items()},
                "calibrated_similarity": {
                    label: round(value, 3) for label, value in analysis.adjusted_similarity.items()
                },
                "rule_matches": rule_matches,
                "gdpr_articles": sorted(articles),
                "recommendations": sorted(recommendations),
                "top_prototype": {
                    "text": analysis.top_prototype["text"],
                    "similarity": round(float(analysis.top_prototype["similarity"]), 3),
                },
            }
        )

    flagged_keywords = [
        {
            "label": label,
            "count": data["count"],
            "keywords": sorted(data["keywords"]),
            "risk": data["risk"],
            "status": data["status"],
        }
        for label, data in keyword_tracker.items()
    ]
    flagged_keywords.sort(key=lambda item: item["count"], reverse=True)

    total_chunks = len(chunks)
    compliance_ratio = round(status_counts["compliant"] / total_chunks, 3)

    status_order = ["non_compliant", "ambiguous", "compliant"]
    chunk_entries.sort(key=lambda item: status_order.index(item["status"]))

    summary = {
        "status_counts": status_counts,
        "compliance_ratio": compliance_ratio,
        "flagged_keywords": flagged_keywords,
        "top_sections": [
            {
                "id": entry["id"],
                "status": entry["status"],
                "color": entry["color"],
                "risk_level": entry["risk_level"],
                "articles": entry["gdpr_articles"],
                "recommendations": entry["recommendations"][:2],
                "snippet": entry["text"][:200],
            }
            for entry in chunk_entries[:5]
        ],
    }

    metadata = {
        "source": source_name,
        "source_type": source_type,
        "chunk_count": total_chunks,
        "generated_at": dt.datetime.utcnow().isoformat() + "Z",
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "character_count": len(raw_text),
        "word_count": len(raw_text.split()),
    }

    return {
        "metadata": metadata,
        "summary": summary,
        "chunks": chunk_entries,
    }


def extract_text_from_url(url: str) -> str:
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "COMPASS-GDPR-Analyzer/1.0"})
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network errors
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}") from exc

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()
    text_content = soup.get_text("\n")
    cleaned_lines = [line.strip() for line in text_content.splitlines() if line.strip()]
    return "\n".join(cleaned_lines)

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "COMPASS Backend is running"}

@app.get("/system/info")
def system_info():
    gpu_available = torch.cuda.is_available()
    # Attempt to introspect pydyf version for debugging PDF backend mismatch
    try:
        pydyf_mod = importlib.import_module("pydyf")
        pydyf_version = getattr(pydyf_mod, "__version__", "unknown")
        try:
            pdf_cls = getattr(pydyf_mod, "PDF", None)
            if pdf_cls is not None:
                sig = inspect.signature(pdf_cls.__init__)
                pydyf_pdf_init_params = len(sig.parameters)
            else:
                pydyf_pdf_init_params = "missing-PDF-class"
        except Exception:
            pydyf_pdf_init_params = "inspect-failed"
    except Exception:
        pydyf_version = "not-installed"
        pydyf_pdf_init_params = "n/a"
    try:
        cssselect2_mod = importlib.import_module("cssselect2")
        cssselect2_version = getattr(cssselect2_mod, "__version__", "unknown")
    except Exception:
        cssselect2_version = "not-installed"
    try:
        tinycss2_mod = importlib.import_module("tinycss2")
        tinycss2_version = getattr(tinycss2_mod, "__version__", "unknown")
    except Exception:
        tinycss2_version = "not-installed"
    return {
        "gpu_available": gpu_available,
        "device": "GPU" if gpu_available else "CPU",
        "torch_version": torch.__version__,
        "weasyprint_version": weasy_version,
        "pydyf_version": pydyf_version,
        "pydyf_pdf_init_params": pydyf_pdf_init_params,
        "cssselect2_version": cssselect2_version,
        "tinycss2_version": tinycss2_version,
    }


@app.get("/report/pdf/test")
def pdf_smoke_test():
    """Generate a tiny PDF using WeasyPrint to validate the rendering pipeline."""
    html_str = """
    <!DOCTYPE html>
    <html><head><meta charset=\"utf-8\" />
    <style>@page { size: A4; margin: 15mm; } body { font-family: 'DejaVu Sans', Arial, sans-serif; }</style>
    </head>
    <body>
      <h1>PDF smoke test</h1>
      <p>This is a minimal WeasyPrint render test.</p>
    </body></html>
    """
    try:
        pdf_bytes = HTML(string=html_str).write_pdf()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"WeasyPrint smoke test failed: {type(exc).__name__}: {exc}")
    filename = f"compass-smoke-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )

@app.post("/analyze/file")
async def analyze_policy_file(file: UploadFile = File(...)):
    load_model()
    allowed_content_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Received '{file.content_type}', expected one of {allowed_content_types}",
        )

    try:
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        if file.content_type == "application/pdf":
            raw_text = extract_text_from_pdf(file_stream)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raw_text = extract_text_from_docx(file_stream)
        else:
            raw_text = extract_text_from_txt(file_stream)

        return build_analysis_response(raw_text, file.filename or "uploaded document", file.content_type)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - broad catch for runtime issues
        raise HTTPException(status_code=500, detail=f"Error processing file: {exc}") from exc


@app.post("/analyze/text")
def analyze_policy_text(payload: TextAnalysisRequest):
    load_model()
    document_name = payload.document_name or "pasted text"
    return build_analysis_response(payload.content, document_name, "text/plain")


@app.post("/analyze/url")
def analyze_policy_url(payload: UrlAnalysisRequest):
    load_model()
    raw_text = extract_text_from_url(str(payload.url))
    return build_analysis_response(raw_text, str(payload.url), "text/html")


@app.post("/report/pdf")
def export_pdf_report(report: PdfReportRequest):
    try:
        # Prefer robust HTML->PDF engine to avoid width errors
        pdf_bytes = generate_pdf_report_weasy(report.model_dump())
    except Exception as exc:  # pragma: no cover - PDF generation issues are logged at runtime
        raise HTTPException(status_code=500, detail=f"Failed to render PDF report: {exc}") from exc

    filename = f"compass-gdpr-report-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )
