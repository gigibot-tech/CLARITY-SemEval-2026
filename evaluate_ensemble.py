#!/usr/bin/env python3
"""
Evaluate RoBERTa ensemble models on QEvasion dataset (train/test splits).

Usage:
  python evaluate_ensemble.py [--split train|test|both] [--threshold 0.5]

This script:
1. Loads all 3 ensemble models
2. Evaluates each model individually
3. Evaluates the ensemble (majority vote / averaged logits)
4. Reports metrics: accuracy, precision, recall, F1, per-class metrics

Common to include with HuggingFace model repos for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    SCRIPT_DIR / "roberta-ensemble-model-1" / "final",
    SCRIPT_DIR / "roberta-ensemble-model-2" / "final",
    SCRIPT_DIR / "roberta-ensemble-model-3" / "final",
]

# Label mapping (from model config)
ID2LABEL = {0: "Clear Reply", 1: "Clear Non-Reply", 2: "Ambivalent"}
LABEL2ID = {"Clear Reply": 0, "Clear Non-Reply": 1, "Ambivalent": 2}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(model_paths: list[Path], device: torch.device):
    """Load all ensemble models."""
    models = []
    tokenizer = None
    for path in model_paths:
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        print(f"  Loading {path.parent.name}...")
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        model.to(device)
        model.eval()
        models.append(model)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(str(path))
    return models, tokenizer


def map_label_to_id(label_str: str) -> int:
    """Map clarity_label string to numeric ID."""
    label_str = str(label_str).strip()
    # Handle various label formats
    if label_str in LABEL2ID:
        return LABEL2ID[label_str]
    # Single character abbreviations
    if label_str == "C" or "Clear Reply" in label_str:
        return 0
    if label_str == "N" or "Non-Reply" in label_str:
        return 1
    if label_str == "A" or "Ambivalent" in label_str:
        return 2
    print(f"  Warning: Unknown label '{label_str}', mapping to 2 (Ambivalent)")
    return 2


def predict_single(model, tokenizer, texts: list[str], device: torch.device, batch_size: int = 16):
    """Get predictions from a single model."""
    all_logits = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting", leave=False):
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
    return np.concatenate(all_logits, axis=0)


def ensemble_predict(models, tokenizer, texts: list[str], device: torch.device, method: str = "average"):
    """
    Ensemble prediction using all models.
    
    method:
      - 'average': average logits, then argmax
      - 'vote': majority vote on predictions
    """
    all_model_logits = []
    for i, model in enumerate(models):
        print(f"  Model {i+1}/{len(models)}...")
        logits = predict_single(model, tokenizer, texts, device)
        all_model_logits.append(logits)
    
    if method == "average":
        avg_logits = np.mean(all_model_logits, axis=0)
        preds = np.argmax(avg_logits, axis=-1)
        probs = torch.softmax(torch.tensor(avg_logits), dim=-1).numpy()
    else:  # vote
        all_preds = [np.argmax(logits, axis=-1) for logits in all_model_logits]
        preds = []
        for i in range(len(texts)):
            votes = [p[i] for p in all_preds]
            preds.append(Counter(votes).most_common(1)[0][0])
        preds = np.array(preds)
        probs = None
    
    return preds, probs, all_model_logits


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]):
    """Compute and print metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f} (macro)")
    print(f"  Recall:    {recall:.4f} (macro)")
    print(f"  F1:        {f1:.4f} (macro)")
    
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    
    print("  Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"  {cm}")
    
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def load_qevasion_data(split: str):
    """Load QEvasion dataset and extract texts + labels."""
    print(f"\nLoading QEvasion dataset ({split} split)...")
    dataset = load_dataset("ailsntua/QEvasion")
    
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")
    
    ds = dataset[split]
    
    # Build input texts (question + answer)
    texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(ds["question"], ds["interview_answer"])
    ]
    
    # Map labels
    labels = np.array([map_label_to_id(lbl) for lbl in ds["clarity_label"]])
    
    print(f"  Loaded {len(texts)} samples")
    print(f"  Label distribution: {Counter(labels)}")
    
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa ensemble on QEvasion")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Which split to evaluate")
    parser.add_argument("--method", choices=["average", "vote"], default="average",
                        help="Ensemble method: average logits or majority vote")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load models
    print("\nLoading ensemble models...")
    models, tokenizer = load_models(MODEL_PATHS, device)
    if not models:
        print("No models found!")
        return
    print(f"Loaded {len(models)} models")
    
    results = {}
    splits = ["train", "test"] if args.split == "both" else [args.split]
    
    for split in splits:
        texts, labels = load_qevasion_data(split)
        label_names = [ID2LABEL[i] for i in range(3)]
        
        # Individual model evaluation
        print(f"\n{'='*60}")
        print(f"EVALUATING ON {split.upper()} SPLIT")
        print(f"{'='*60}")
        
        for i, model in enumerate(models):
            print(f"\n--- Model {i+1} ---")
            logits = predict_single(model, tokenizer, texts, device)
            preds = np.argmax(logits, axis=-1)
            metrics = evaluate(labels, preds, label_names)
            results[f"model_{i+1}_{split}"] = metrics
        
        # Ensemble evaluation
        print(f"\n--- ENSEMBLE ({args.method}) ---")
        preds, probs, _ = ensemble_predict(models, tokenizer, texts, device, method=args.method)
        metrics = evaluate(labels, preds, label_names)
        results[f"ensemble_{args.method}_{split}"] = metrics
    
    # Save results
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for key, metrics in results.items():
        print(f"{key}: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
