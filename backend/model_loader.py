"""Utility module for loading Legal-BERT.

This module supports two classification modes:
1) Fine-tuned classifier (preferred): If a local model directory is present
    at backend/model_store/legalbert_3way (or MODEL_DIR env var), we load
    AutoModelForSequenceClassification and return probability-based decisions.
2) Prototype similarity (fallback): If no fine-tuned model is found, we use
    an embedding-based prototype similarity approach.

Both modes return the same ComplianceResult structure to keep the FastAPI
endpoints stable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 512
FINETUNED_PROTOTYPE_PATH = Path(__file__).resolve().parent / "data" / "fine_tuned_samples.json"
# Local path where the trained classifier is expected to be mounted/copied
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model_store" / "legalbert_3way"

CATEGORY_COLORS: Dict[str, str] = {
    "compliant": "#2F855A",
    "non_compliant": "#C53030",
    "ambiguous": "#D69E2E",
}

DEFAULT_CATEGORY_PROTOTYPES: Dict[str, List[str]] = {
    "compliant": [
        "The organization processes personal data lawfully, fairly, and transparently with documented consent.",
        "Users can access, rectify, and erase their personal information at any time through established procedures.",
        "Personal data is collected for specified purposes and retained only for as long as necessary to meet those purposes.",
    ],
    "non_compliant": [
        "Personal data may be sold or shared with partners without providing an opt-out mechanism to users.",
        "We collect information without informing users or obtaining lawful consent beforehand.",
        "Your data can be retained indefinitely even after the original purpose has been fulfilled.",
    ],
    "ambiguous": [
        "The company may share information with trusted partners when necessary to provide services.",
        "Data might be stored for a reasonable time period as determined by internal policies.",
        "We strive to protect your information but cannot guarantee absolute security in all circumstances.",
    ],
}


def _load_category_prototypes() -> Dict[str, List[str]]:
    if FINETUNED_PROTOTYPE_PATH.exists():
        try:
            with FINETUNED_PROTOTYPE_PATH.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            normalized: Dict[str, List[str]] = {}
            for label, samples in loaded.items():
                normalized[label] = [sample.strip() for sample in samples if sample and sample.strip()]
            expected_labels = set(DEFAULT_CATEGORY_PROTOTYPES.keys())
            if expected_labels.issubset(normalized.keys()):
                print("--- Loaded fine-tuned prototypes from disk ---")
                return normalized
        except Exception as exc:  # pragma: no cover - safeguard for malformed files
            print(f"[WARN] Failed to load fine-tuned prototypes: {exc}. Using defaults.")
    return DEFAULT_CATEGORY_PROTOTYPES


CATEGORY_PROTOTYPES: Dict[str, List[str]] = _load_category_prototypes()

CLASS_WEIGHTS = {
    "non_compliant": 1.15,
    "ambiguous": 1.05,
    "compliant": 1.0,
}

CLASS_BIASES = {
    "non_compliant": 0.05,
    "ambiguous": 0.02,
    "compliant": 0.0,
}


@dataclass
class ComplianceResult:
    """Structured representation of a classifier decision."""

    status: str
    probability: float
    scores: Dict[str, float]
    raw_similarity: Dict[str, float]
    adjusted_similarity: Dict[str, float]
    top_prototype: Dict[str, float | str]


class LegalBertService:
    """Handles loading Legal-BERT and generating sentence embeddings."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Initializing Legal-BERT backbone on {self.device.type.upper()} ---")
        self.model = AutoModel.from_pretrained(MODEL_NAME).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(self, texts: Iterable[str]) -> torch.Tensor:
        """Returns L2-normalized embeddings for the provided texts."""

        if isinstance(texts, str):  # type: ignore[arg-type]
            texts = [texts]  # pragma: no cover - defensive path

        text_list = list(texts)
        if not text_list:
            raise ValueError("No texts provided for encoding.")

        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        with torch.no_grad():
            outputs = self.model(**tokens)

        embeddings = self._mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
        normalized = F.normalize(embeddings, p=2, dim=1)
        return normalized.cpu()

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        masked_state = last_hidden_state * attention_mask.unsqueeze(-1)
        token_sum = masked_state.sum(dim=1)
        mask_sum = attention_mask.sum(dim=1).clamp(min=1)
        return token_sum / mask_sum.unsqueeze(-1)


class LegalComplianceClassifier:
    """Implements prototype-based compliance classification on top of Legal-BERT embeddings."""

    def __init__(self, service: LegalBertService) -> None:
        self.service = service
        self.prototype_embeddings: Dict[str, torch.Tensor] = {}
        self.prototype_texts: Dict[str, List[str]] = {}
        self._prepare_prototypes()

    def _prepare_prototypes(self) -> None:
        for label, examples in CATEGORY_PROTOTYPES.items():
            self.prototype_embeddings[label] = self.service.encode(examples)
            self.prototype_texts[label] = examples

    def classify(self, text: str) -> ComplianceResult:
        embedding = self.service.encode([text])[0]

        raw_similarity: Dict[str, float] = {}
        adjusted_similarity: Dict[str, float] = {}
        top_prototypes: Dict[str, Dict[str, float | str]] = {}
        similarity_values: List[float] = []
        labels: List[str] = []

        for label, prototypes in self.prototype_embeddings.items():
            sims = torch.matmul(prototypes, embedding)
            mean_score = float(sims.mean())
            weight = CLASS_WEIGHTS.get(label, 1.0)
            bias = CLASS_BIASES.get(label, 0.0)
            calibrated_score = mean_score * weight + bias
            raw_similarity[label] = mean_score
            adjusted_similarity[label] = calibrated_score
            best_idx = int(torch.argmax(sims))
            top_prototypes[label] = {
                "text": self.prototype_texts[label][best_idx],
                "similarity": float(sims[best_idx]),
            }
            similarity_values.append(calibrated_score)
            labels.append(label)

        score_tensor = torch.tensor(similarity_values)
        probabilities_tensor = torch.softmax(score_tensor, dim=0)
        scores: Dict[str, float] = {labels[idx]: float(probabilities_tensor[idx]) for idx in range(len(labels))}

        status = max(scores.items(), key=lambda kv: kv[1])[0]

        return ComplianceResult(
            status=status,
            probability=scores[status],
            scores=scores,
            raw_similarity=raw_similarity,
            adjusted_similarity=adjusted_similarity,
            top_prototype=top_prototypes[status],
        )


# --- Fine-tuned classifier support ---

class ClassifierService:
    """Loads a fine-tuned sequence classifier and provides prediction utilities."""

    def __init__(self, model_dir: Path) -> None:
        if not model_dir.exists() or not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Model directory not found or missing config.json: {model_dir}")
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- Loading fine-tuned classifier from {self.model_dir} on {self.device.type.upper()} ---")
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir)).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        # Build label mappings from config (ensures alignment with training)
        try:
            cfg = self.model.config
            self.id2label = {int(k): v for k, v in getattr(cfg, "id2label", {}).items()}
            self.label2id = {k: int(v) for k, v in getattr(cfg, "label2id", {}).items()}
        except Exception:
            # Fallback to common ordering
            labels = ["compliant", "ambiguous", "non_compliant"]
            self.id2label = {i: l for i, l in enumerate(labels)}
            self.label2id = {l: i for i, l in self.id2label.items()}

    def predict_proba(self, texts: Iterable[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        text_list = list(texts)
        if not text_list:
            raise ValueError("No texts provided for prediction.")

        tokens = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        with torch.no_grad():
            logits = self.model(**tokens).logits
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu()


class ClassifierBasedCompliance:
    """Classifier-backed compliance classification to match ComplianceResult shape."""

    def __init__(self, service: ClassifierService) -> None:
        self.service = service

    def classify(self, text: str) -> ComplianceResult:
        probs = self.service.predict_proba([text])[0]
        scores: Dict[str, float] = {self.service.id2label[i]: float(probs[i]) for i in range(len(probs))}
        status = max(scores.items(), key=lambda kv: kv[1])[0]
        probability = scores[status]

        # We don't have prototype info; surface the chosen label and prob
        top_prototype = {"text": "N/A (fine-tuned classifier)", "similarity": probability}

        # raw/adjusted similarity placeholders: use probabilities for both
        return ComplianceResult(
            status=status,
            probability=probability,
            scores=scores,
            raw_similarity=scores,
            adjusted_similarity=scores,
            top_prototype=top_prototype,
        )


_service: LegalBertService | None = None
_proto_classifier: LegalComplianceClassifier | None = None
_clf_service: ClassifierService | None = None
_clf_classifier: ClassifierBasedCompliance | None = None


def get_service() -> LegalBertService:
    global _service
    if _service is None:
        _service = LegalBertService()
    return _service


def _resolve_model_dir() -> Path | None:
    # Allow override via env var, else use default path under backend/model_store
    env_path = os.getenv("MODEL_DIR")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        return p if p.exists() else None
    return DEFAULT_MODEL_DIR if DEFAULT_MODEL_DIR.exists() else None


def get_compliance_classifier():
    """Return the best available classifier (fine-tuned if present, else prototype-based)."""
    global _clf_classifier, _clf_service, _proto_classifier
    # Prefer fine-tuned model when available
    if _clf_classifier is None:
        model_dir = _resolve_model_dir()
        if model_dir is not None:
            try:
                _clf_service = ClassifierService(model_dir)
                _clf_classifier = ClassifierBasedCompliance(_clf_service)
                return _clf_classifier
            except Exception as exc:
                print(f"[WARN] Failed to initialize fine-tuned classifier: {exc}. Falling back to prototype mode.")

    # Fallback to prototype classifier
    if _proto_classifier is None:
        _proto_classifier = LegalComplianceClassifier(get_service())
    return _proto_classifier


def get_legal_tokenizer() -> AutoTokenizer:
    # Use classifier tokenizer if the fine-tuned model is active
    if _clf_service is not None:
        return _clf_service.tokenizer
    return get_service().tokenizer


def get_active_ai_source() -> str:
    """Return a short tag for the active AI decision source.

    - "classifier": fine-tuned sequence classifier is active
    - "prototype": prototype-similarity fallback is active
    """
    return "classifier" if _clf_classifier is not None else "prototype"
