from transformers import pipeline, AutoTokenizer
import torch

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"

# Globals
_analyzer_pipeline = pipeline("text-classification", model=MODEL_NAME, device=get_device())
_tokenizer = None

def get_device():
    """Detects GPU if available, else falls back to CPU."""
    return 0 if torch.cuda.is_available() else -1

def get_pipeline():
    """Lazy-loads the pipeline with GPU/CPU fallback."""
    global _analyzer_pipeline
    if _analyzer_pipeline is None:
        print("--- Loading model pipeline (GPU if available) ---")
        _analyzer_pipeline = pipeline("sentiment-analysis", model=MODEL_NAME, device=get_device())
        print("--- Model ready ---")
    return _analyzer_pipeline

def get_tokenizer():
    """Lazy-loads the tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer
