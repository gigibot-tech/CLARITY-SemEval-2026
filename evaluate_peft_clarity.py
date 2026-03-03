#!/usr/bin/env python3
"""
Evaluate a PEFT (LoRA) adapter + base model on QEvasion clarity task.

Usage:
  python evaluate_peft_clarity.py \
    --base-model roberta-base \
    --adapter-path /path/to/peft/adapter \
    --split test

Requirements:
  pip install transformers datasets torch scikit-learn peft
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


LABEL2ID = {"Clear Reply": 0, "Clear Non-Reply": 1, "Ambivalent": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_qevasion_data(split: str):
    dataset = load_dataset("ailsntua/QEvasion")
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")
    ds = dataset[split]
    texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(ds["question"], ds["interview_answer"])
    ]
    labels = np.array([LABEL2ID.get(lbl, 2) for lbl in ds["clarity_label"]])
    print(f"Loaded {len(texts)} samples")
    print(f"Label distribution: {Counter(labels)}")
    return texts, labels


def predict(model, tokenizer, texts: list[str], device: torch.device, batch_size: int):
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            all_logits.append(outputs.logits.cpu().numpy())
    logits = np.concatenate(all_logits, axis=0)
    preds = np.argmax(logits, axis=-1)
    return preds


def evaluate(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\nAccuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} (macro)")
    print(f"Recall:    {recall:.4f} (macro)")
    print(f"F1:        {f1:.4f} (macro)")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[ID2LABEL[i] for i in range(3)], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEFT adapter + base model on QEvasion")
    parser.add_argument("--base-model", required=True, help="Base HF model name (e.g., roberta-base)")
    parser.add_argument("--adapter-path", required=True, help="Path to PEFT adapter directory")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Dataset split")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if PeftModel is None:
        raise ImportError("PEFT is not installed. Run: pip install peft")

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    texts, labels = load_qevasion_data(args.split)
    preds = predict(model, tokenizer, texts, device, args.batch_size)
    metrics = evaluate(labels, preds)

    if args.output:
        args.output.write_text(json.dumps(metrics, indent=2))
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
