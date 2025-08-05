import io
import docx
import nltk
import re  # Import the regular expression library
from fastapi import FastAPI, File, UploadFile, HTTPException
from pypdf import PdfReader
from transformers import pipeline, AutoTokenizer

# --- Constants ---
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_TOKENS = 512

# --- Lazy-loaded AI Model and Tokenizer ---
analyzer_pipeline = None
tokenizer = None

# --- Create FastAPI App ---
app = FastAPI(title="COMPASS API")


# --- Helper function to load the model on first use ---
def load_model():
    """Loads the model and tokenizer into the global variables."""
    global analyzer_pipeline, tokenizer
    if analyzer_pipeline is None:
        print("--- Loading AI model for the first time... ---")
        analyzer_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print("--- Model loaded successfully. ---")


# --- Helper functions for text extraction and processing ---
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

# --- UPDATED hybrid_chunking function ---
def hybrid_chunking(text: str) -> list[str]:
    """
    Splits text into chunks using a robust method. Prioritizes paragraphs, 
    but if a paragraph is too long, it splits that paragraph into sentences.
    """
    load_model() # Ensure model is loaded before tokenizing
    chunks = []
    
    # Normalize line endings and split into paragraphs using regular expressions
    # This robustly handles different kinds of blank lines
    normalized_text = text.replace('\r\n', '\n')
    paragraphs = re.split(r'\n\s*\n', normalized_text)
    
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
    return {"message": "COMPASS Backend is running"}


@app.post("/analyze/file")
async def analyze_policy_file(file: UploadFile = File(...)):
    """
    Receives a policy file, extracts text, uses hybrid chunking,
    and performs a basic analysis.
    """
    load_model() # Ensure model is loaded before analysis
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
        
        raw_text = ""
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

        # --- Perform AI Analysis on each chunk ---
        # For now, we still only analyze the first chunk to test the pipeline
        analysis_result = analyzer_pipeline(chunks[0])

        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "chunk_count": len(chunks),
            "analysis_of_first_chunk": analysis_result[0],
            "chunks_snippet": chunks[:5]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {e}")