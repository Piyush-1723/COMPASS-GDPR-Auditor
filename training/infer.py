#!/usr/bin/env python
import argparse
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="training/models/legalbert_3way")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch = tok([args.text], truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model(**batch).logits.detach().cpu().numpy()
    probs = softmax(out)[0]
    label_id = int(probs.argmax())
    id2label = model.config.id2label
    result = {
        "label": id2label[label_id],
        "confidence": float(probs[label_id]),
        "scores": {id2label[i]: float(probs[i]) for i in range(len(probs))},
        "device": device,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
