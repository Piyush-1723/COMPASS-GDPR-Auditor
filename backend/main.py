import datetime as dt
import io
import re
from typing import Any, Dict, List, Tuple

import docx
import nltk
import requests
import torch
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fpdf import FPDF
from pydantic import BaseModel, HttpUrl
from pypdf import PdfReader

from model_loader import (
    CATEGORY_COLORS,
    ComplianceResult,
    get_compliance_classifier,
    get_legal_tokenizer,
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
    load_model()
    chunks = []
    normalized_text = text.replace("\r\n", "\n")
    paragraphs = re.split(r"\n\s*\n", normalized_text)

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue
        token_count = len(tokenizer.tokenize(paragraph))
        if token_count <= MAX_TOKENS:
            chunks.append(paragraph.strip())
        else:
            sentences = nltk.sent_tokenize(paragraph)
            chunks.extend(s.strip() for s in sentences)

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


def determine_status(ai_status: str, rule_matches: List[Dict[str, str]]) -> Tuple[str, str]:
    if rule_matches:
        for priority in STATUS_PRIORITY:
            for match in rule_matches:
                if match["status"] == priority:
                    reason = f"rule:{match['label']}:{match['keyword']}"
                    return priority, reason
    return ai_status, "ai:prototype"


def describe_status_source(source: str | None) -> str:
    if not source:
        return "AI-driven classification"
    if source.startswith("rule:"):
        try:
            _, label, keyword = source.split(":", 2)
        except ValueError:
            return "Rule-based decision"
        readable = CLAUSE_LABEL_TITLES.get(label, label.replace("_", " ").title())
        return f"Rule match • {readable} • '{keyword}'"
    if source.startswith("ai:"):
        return "Legal-BERT prototype similarity"
    return "AI-driven classification"


def safe_text(text: str) -> str:
    replacements = {
        "•": "-",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    sanitized = text
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)

    def breaker(match: re.Match[str]) -> str:
        token = match.group(0)
        chunks = [token[i:i + 35] for i in range(0, len(token), 35)]
        return " ".join(chunks)

    sanitized = re.sub(r"\S{60,}", breaker, sanitized)
    return sanitized.encode("latin-1", "ignore").decode("latin-1")


def summarize_text(text: str, limit: int = 700) -> str:
    collapsed = re.sub(r"\s+", " ", text or "").strip()
    if len(collapsed) <= limit:
        return collapsed
    truncated = collapsed[:limit]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return f"{truncated}..."


def hex_to_rgb(color: str) -> Tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        return 45, 55, 72
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def generate_pdf_report(report: Dict[str, Any]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.add_page()

    base_text_color = (45, 55, 72)
    accent_color = (44, 82, 130)

    metadata = report.get("metadata", {})
    summary = report.get("summary", {})
    status_counts = summary.get("status_counts", {})
    ratio = summary.get("compliance_ratio", 0)
    flagged_keywords = summary.get("flagged_keywords", [])
    chunks = report.get("chunks", [])

    generated_at = metadata.get("generated_at", dt.datetime.utcnow().isoformat())
    generated_display = generated_at.replace("T", " ").replace("Z", " UTC")

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*base_text_color)
    pdf.multi_cell(0, 8, safe_text("COMPASS GDPR Compliance Report"))

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(71, 85, 105)
    pdf.multi_cell(
        0,
        6,
        safe_text(
            f"Source: {metadata.get('source', 'Unknown source')} ({metadata.get('source_type', 'n/a')})"
        ),
    )
    pdf.multi_cell(
        0,
        6,
        safe_text(
            f"Generated: {generated_display} • Sections analyzed: {metadata.get('chunk_count', 0)} • Device: {metadata.get('device', 'CPU')}"
        ),
    )
    pdf.multi_cell(
        0,
        6,
        safe_text(
            f"Word count: {metadata.get('word_count', 0)} • Character count: {metadata.get('character_count', 0)}"
        ),
    )
    pdf.ln(4)

    def heading(title: str) -> None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*accent_color)
        pdf.multi_cell(0, 7, safe_text(title))
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(*base_text_color)

    heading("Overall compliance snapshot")
    pdf.multi_cell(
        0,
        6,
        safe_text(
            f"Compliance ratio: {ratio * 100:.1f}% of sections are compliant"
        ),
    )
    pdf.multi_cell(
        0,
        6,
        safe_text(
            "Breakdown — Non-Compliant: {non} • Ambiguous: {amb} • Compliant: {comp}".format(
                non=status_counts.get("non_compliant", 0),
                amb=status_counts.get("ambiguous", 0),
                comp=status_counts.get("compliant", 0),
            )
        ),
    )
    pdf.ln(3)

    heading("Flagged GDPR indicators")
    if flagged_keywords:
        pdf.set_font("Helvetica", "", 10)
        for item in flagged_keywords:
            label = CLAUSE_LABEL_TITLES.get(item.get("label", ""), item.get("label", "")).replace("_", " ").title()
            keywords = ", ".join(sorted(item.get("keywords", [])))
            line = f"- {label}: {keywords} ({item.get('count', 0)} reference(s))"
            pdf.multi_cell(0, 5, safe_text(line))
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, safe_text("No GDPR risk keywords were flagged in this report."))
    pdf.ln(4)

    heading("Clause-by-clause analysis")
    if not chunks:
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, safe_text("No clause segments available."))
    else:
        for status in STATUS_PRIORITY:
            status_chunks = [chunk for chunk in chunks if chunk.get("status") == status]
            if not status_chunks:
                continue

            status_title = f"{STATUS_TITLES.get(status, status.title())} sections ({STATUS_RISK_LEVEL.get(status, 'Medium')} risk)"
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(*hex_to_rgb(CATEGORY_COLORS.get(status, "#2d3748")))
            pdf.multi_cell(0, 6, safe_text(status_title))
            pdf.set_text_color(*base_text_color)

            for chunk in status_chunks:
                pdf.set_font("Helvetica", "B", 11)
                header_text = f"Section {chunk.get('id', '?')}"
                pdf.multi_cell(0, 6, safe_text(header_text))

                pdf.set_font("Helvetica", "", 10)
                pdf.multi_cell(0, 5, safe_text(summarize_text(chunk.get("text", ""))))

                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(0, 5, safe_text(f"Decision basis: {describe_status_source(chunk.get('status_source'))}"))

                articles = ", ".join(chunk.get("gdpr_articles", [])) or "Not specified"
                pdf.set_font("Helvetica", "", 9)
                pdf.multi_cell(0, 5, safe_text(f"GDPR articles: {articles}"))

                recommendations = chunk.get("recommendations", [])
                if recommendations:
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.multi_cell(0, 5, safe_text("Recommended actions:"))
                    pdf.set_font("Helvetica", "", 9)
                    for rec in recommendations:
                        pdf.multi_cell(0, 5, safe_text(f"- {rec}"))

                rule_matches = chunk.get("rule_matches", [])
                if rule_matches:
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.multi_cell(0, 5, safe_text("Rule indicators:"))
                    pdf.set_font("Helvetica", "", 9)
                    seen = set()
                    for match in rule_matches:
                        key = (match.get("label"), match.get("keyword"))
                        if key in seen:
                            continue
                        seen.add(key)
                        label = CLAUSE_LABEL_TITLES.get(match.get("label", ""), match.get("label", "")).replace("_", " ").title()
                        pdf.multi_cell(0, 5, safe_text(f"- {label}: '{match.get('keyword', '')}'"))

                pdf.ln(2)

            pdf.ln(1)

    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.multi_cell(0, 5, safe_text("Report generated by COMPASS GDPR Auditor"))

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    return pdf_bytes
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
    return {
        "gpu_available": gpu_available,
        "device": "GPU" if gpu_available else "CPU",
        "torch_version": torch.__version__,
    }

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
        pdf_bytes = generate_pdf_report(report.model_dump())
    except Exception as exc:  # pragma: no cover - PDF generation issues are logged at runtime
        raise HTTPException(status_code=500, detail=f"Failed to render PDF report: {exc}") from exc

    filename = f"compass-gdpr-report-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
    )
