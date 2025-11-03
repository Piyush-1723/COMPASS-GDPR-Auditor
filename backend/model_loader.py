"""Utility module for loading Legal-BERT and running lightweight compliance analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
MAX_LENGTH = 512
FINETUNED_PROTOTYPE_PATH = Path(__file__).resolve().parent / "data" / "fine_tuned_samples.json"

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
        top_prototype: Dict[str, float | str] = {}
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
            top_prototype[label] = {
                "text": self.prototype_texts[label][best_idx],
                "similarity": float(sims[best_idx]),
            }
            similarity_values.append(calibrated_score)
            labels.append(label)

        score_tensor = torch.tensor(similarity_values)
        probabilities_tensor = torch.softmax(score_tensor, dim=0)
        scores = {labels[idx]: float(probabilities_tensor[idx]) for idx in range(len(labels))}

        status = max(scores, key=scores.get)

        return ComplianceResult(
            status=status,
            probability=scores[status],
            scores=scores,
            raw_similarity=raw_similarity,
            adjusted_similarity=adjusted_similarity,
            top_prototype=top_prototype[status],
        )


_service: LegalBertService | None = None
_classifier: LegalComplianceClassifier | None = None


def get_service() -> LegalBertService:
    global _service
    if _service is None:
        _service = LegalBertService()
    return _service


def get_compliance_classifier() -> LegalComplianceClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LegalComplianceClassifier(get_service())
    return _classifier


def get_legal_tokenizer() -> AutoTokenizer:
    return get_service().tokenizer
