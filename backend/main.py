import io
import docx
import nltk
import re
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pypdf import PdfReader
from transformers import pipeline, AutoTokenizer

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

# --- Constants ---
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_TOKENS = 512

# --- Lazy-loaded AI Model and Tokenizer ---
analyzer_pipeline = None
tokenizer = None

# --- Create FastAPI App ---
app = FastAPI(title="COMPASS API")


# --- Helper function to load the model on first use ---
def load_model():
    global analyzer_pipeline, tokenizer
    if analyzer_pipeline is None:
        print("--- Loading Legal-BERT model for the first time... ---")
        device = 0 if torch.cuda.is_available() else -1
        device_type = "GPU" if device == 0 else "CPU"
        print(f"--- Using {device_type} for model inference ---")

        # Use text-classification pipeline for Legal-BERT
        analyzer_pipeline = pipeline(
            "text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=device
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("--- Legal-BERT loaded successfully. ---")


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
def assign_rule_based_label(clause: str):
    lower_clause = clause.lower()
    for label, keywords in CLAUSE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower_clause:
                return label, LABEL_RISK_MAPPING[label], 1.0
    return None, None, None

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
        "text/plain"
    ]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Received '{file.content_type}', expected one of {allowed_content_types}"
        )

    try:
        file_content = await file.read()
        file_stream = io.BytesIO(file_content)

        if file.content_type == "application/pdf":
            raw_text = extract_text_from_pdf(file_stream)
        elif file.content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raw_text = extract_text_from_docx(file_stream)
        elif file.content_type == "text/plain":
            raw_text = extract_text_from_txt(file_stream)

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the document.")

        chunks = hybrid_chunking(raw_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not segment the document into chunks.")

        # --- Perform Rule-based + Legal-BERT AI Analysis ---
        results = []
        for chunk in chunks:
            # Rule-based label
            rule_label, rule_risk, rule_score = assign_rule_based_label(chunk)

            # AI-based prediction
            ai_result = analyzer_pipeline(chunk)[0]  # {"label": "LABEL_0", "score": 0.987}
            ai_label = ai_result["label"]
            ai_score = ai_result["score"]

            results.append({
                "text": chunk[:200],  # snippet
                "rule_label": rule_label,
                "rule_risk": rule_risk,
                "rule_score": rule_score,
                "ai_label": ai_label,
                "ai_score": ai_score
            })

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "chunk_count": len(chunks),
            "analysis_results": results[:5]  # return first 5 for testing
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
