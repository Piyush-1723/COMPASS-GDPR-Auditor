import io
import docx
import nltk
import re
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pypdf import PdfReader
from transformers import pipeline, AutoTokenizer

# --- Constants ---
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # placeholder
MAX_TOKENS = 512

# --- Lazy-loaded AI Model and Tokenizer ---
analyzer_pipeline = None
tokenizer = None
device = 0 if torch.cuda.is_available() else -1

# --- Create FastAPI App ---
app = FastAPI(title="COMPASS API")

# --- Helper function to load the model on first use ---
def load_model():
    """Loads the model and tokenizer into the global variables."""
    global analyzer_pipeline, tokenizer
    if analyzer_pipeline is None:
        print("--- Loading AI model for the first time... ---")
        analyzer_pipeline = pipeline(
            "sentiment-analysis", model=MODEL_NAME, device=device
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"--- Model loaded successfully on {'GPU' if device == 0 else 'CPU'} ---")

# --- Text extraction ---
def extract_text_from_docx(file_stream: io.BytesIO) -> str:
    document = docx.Document(file_stream)
    return "\n".join([paragraph.text for paragraph in document.paragraphs])

def extract_text_from_pdf(file_stream: io.BytesIO) -> str:
    reader = PdfReader(file_stream)
    return "".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_txt(file_stream: io.BytesIO) -> str:
    return file_stream.read().decode("utf-8")

# --- Hybrid Chunking ---
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

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {
        "message": "COMPASS Backend is running",
        "device": "GPU" if device == 0 else "CPU",
    }

@app.post("/analyze/file")
async def analyze_policy_file(file: UploadFile = File(...)):
    load_model()
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Got '{file.content_type}', expected one of {allowed_types}",
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

        if not raw_text.strip():
            raise HTTPException(status_code=400, detail="No text extracted.")

        chunks = hybrid_chunking(raw_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No valid chunks found.")

        analysis_result = analyzer_pipeline(chunks[0])

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "chunk_count": len(chunks),
            "analysis_of_first_chunk": analysis_result[0],
            "chunks_snippet": chunks[:5],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
