#!/usr/bin/env python
"""
Train a 3-way GDPR compliance classifier on nlpaueb/legal-bert-base-uncased.
- Validates dataset columns and labels
- Creates stratified train/dev/test splits (80/10/10)
- Fine-tunes with HF Trainer
- Saves best model and tokenizer to training/models/legalbert_3way/
- Saves evaluation metrics, confusion matrix, top misclassifications, and a summary JSON

Run (Windows PowerShell):
    # Create and activate venv, then install torch (CUDA 12.8) and deps
    #   pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
    #   pip install -r training/requirements.txt
    # Train
    #   python training/train.py --epochs 4 --batch_size 8 --lr 2e-5 --max_length 256
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from transformers import EarlyStoppingCallback
from transformers.trainer_utils import set_seed

CANONICAL_LABELS = ["compliant", "ambiguous", "non_compliant"]
LABEL2ID = {l: i for i, l in enumerate(CANONICAL_LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def load_and_validate_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: text, label")

    def norm_label(x: str) -> str:
        if not isinstance(x, str):
            return "ambiguous"
        y = x.strip().lower().replace("-", "_").replace(" ", "_")
        if y in CANONICAL_LABELS:
            return y
        mapper = {
            "compliance": "compliant",
            "noncompliant": "non_compliant",
            "non_compliance": "non_compliant",
            "unclear": "ambiguous",
            "unknown": "ambiguous",
        }
        return mapper.get(y, "ambiguous")

    df["label"] = df["label"].map(norm_label)

    counts = df["label"].value_counts().to_dict()
    synthetic_count = int(df["synthetic"].sum()) if "synthetic" in df.columns and df["synthetic"].dtype != object else 0
    stats = {
        "total_rows": int(len(df)),
        "label_counts": counts,
        "synthetic_rows": synthetic_count,
        "columns": list(df.columns),
    }
    print("Dataset summary:", json.dumps(stats, indent=2))
    return df


def stratified_split(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    y = df["label"].map(LABEL2ID)
    train_idx, temp_idx = next(splitter.split(df, y))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    y_temp = temp_df["label"].map(LABEL2ID)
    dev_idx, test_idx = next(splitter2.split(temp_df, y_temp))
    dev_df = temp_df.iloc[dev_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)
    return train_df, dev_df, test_df


@dataclass
class EncodedDataset(torch.utils.data.Dataset):
    # Hold raw tokenizer outputs (lists), let the DataCollator handle padding/tensorization per batch
    encodings: Dict[str, List[List[int]]]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return python lists; DataCollatorWithPadding will pad and convert to tensors
        item = {k: self.encodings[k][idx] for k in self.encodings}
        item["labels"] = self.labels[idx]
        return item


def prepare_encodings(tokenizer, texts: List[str], max_length: int) -> Dict[str, List[List[int]]]:
    # Do not pad here; let DataCollatorWithPadding handle dynamic padding at batch time
    return tokenizer(texts, truncation=True, max_length=max_length, padding=False)


def compute_metrics_builder(id2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        report = classification_report(
            labels,
            preds,
            target_names=[id2label[i] for i in sorted(id2label)],
            output_dict=True,
            zero_division=0,
        )
        out = {
            "accuracy": float(report.get("accuracy", 0.0)),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"]),
        }
        for i, name in id2label.items():
            out[f"f1_{name}"] = float(report.get(name, {}).get("f1-score", 0.0))
        return out

    return compute_metrics


class WeightedTrainer(Trainer):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def detect_imbalance(train_labels: List[int]) -> Tuple[bool, torch.Tensor]:
    counts = np.bincount(train_labels, minlength=len(LABEL2ID))
    max_min_ratio = counts.max() / max(1, counts.min())
    use_weights = max_min_ratio >= 1.5
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()
    return use_weights, torch.tensor(weights, dtype=torch.float32)


def save_reproducible_splits(train_df, dev_df, test_df, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    dev_df.to_csv(os.path.join(out_dir, "dev.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test.csv"), index=False)


def save_eval_artifacts(y_true, y_pred, texts, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=CANONICAL_LABELS, output_dict=True, zero_division=0)
    with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CANONICAL_LABELS))))
    pd.DataFrame(cm, index=CANONICAL_LABELS, columns=CANONICAL_LABELS).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv")
    )
    wrong = [(i, yt, yp) for i, (yt, yp) in enumerate(zip(y_true, y_pred)) if yt != yp]
    rows = []
    for idx, yt, yp in wrong[:20]:
        rows.append({"true": ID2LABEL[yt], "pred": ID2LABEL[yp], "text": texts[idx][:500]})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "top_misclassified.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=os.path.join("training", "gdpr_final_training_dataset.csv"))
    parser.add_argument("--model_name", default="nlpaueb/legal-bert-base-uncased")
    parser.add_argument("--output_dir", default=os.path.join("training", "models", "legalbert_3way"))
    parser.add_argument("--splits_dir", default="training")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    # Simplified configuration (rolling back experimental flags)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_fp16", action="store_true", help="Force fp16; default: auto if CUDA")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16 even if CUDA")
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = load_and_validate_dataset(args.data_path)
    train_df, dev_df, test_df = stratified_split(df, seed=args.seed)
    save_reproducible_splits(train_df, dev_df, test_df, out_dir=args.splits_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    train_enc = prepare_encodings(tokenizer, train_df["text"].tolist(), args.max_length)
    dev_enc = prepare_encodings(tokenizer, dev_df["text"].tolist(), args.max_length)
    test_enc = prepare_encodings(tokenizer, test_df["text"].tolist(), args.max_length)

    y_train = train_df["label"].map(LABEL2ID).tolist()
    y_dev = dev_df["label"].map(LABEL2ID).tolist()
    y_test = test_df["label"].map(LABEL2ID).tolist()

    train_ds = EncodedDataset(train_enc, y_train)
    dev_ds = EncodedDataset(dev_enc, y_dev)
    test_ds = EncodedDataset(test_enc, y_test)

    torch_dtype = None
    if torch.cuda.is_available() and not args.no_fp16:
        torch_dtype = torch.float16
    if args.use_fp16:
        torch_dtype = torch.float16

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(CANONICAL_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=torch_dtype,
    )

    use_weights, weights = detect_imbalance(y_train)
    class_weights = weights if use_weights else None
    if use_weights:
        print(f"Detected class imbalance; using class weights: {weights.tolist()}")

    fp16_auto = torch.cuda.is_available() and not args.no_fp16
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(8, args.batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        # HF renamed this to eval_strategy in newer releases (and backported in some builds)
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=fp16_auto,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=50,
        report_to=[],
        save_total_limit=2,
        seed=args.seed,
        group_by_length=False,
    )

    # Simplify trainer without early stopping if patience=0
    callbacks = []
    if getattr(args, "early_stopping_patience", 0) > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    if class_weights is not None:
        trainer = WeightedTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics_builder(ID2LABEL),
            callbacks=callbacks,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics_builder(ID2LABEL),
            callbacks=callbacks,
        )

    train_out = trainer.train()
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    preds = trainer.predict(test_ds)
    y_pred = np.argmax(preds.predictions, axis=-1)
    save_eval_artifacts(y_true=y_test, y_pred=y_pred, texts=test_df["text"].tolist(), out_dir=args.output_dir)

    summary = {
        "train_samples": len(train_df),
        "dev_samples": len(dev_df),
        "test_samples": len(test_df),
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "args": vars(args),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    # Save summary alongside model artifacts for easier tracking
    with open(os.path.join(args.output_dir, "train_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done. Model saved to:", args.output_dir)
    print("Evaluation report:", os.path.join(args.output_dir, "eval_report.json"))
    print("Confusion matrix:", os.path.join(args.output_dir, "confusion_matrix.csv"))
    print("Train results summary:", os.path.join(args.output_dir, "train_results.json"))


if __name__ == "__main__":
    main()
