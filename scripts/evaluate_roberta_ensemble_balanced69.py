#!/usr/bin/env python3
"""
Evaluate a 3-model RoBERTa ensemble on a balanced QEvasion subset (69 samples by default).

Resolution order per model:
1) local directory path (if model weights exist)
2) Hugging Face repo + subfolder fallback

Outputs:
- metrics JSON
- per-example predictions CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_IMPORT_ERROR = None
try:
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:
    _IMPORT_ERROR = exc
    np = None
    torch = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    f1_score = None
    precision_score = None
    recall_score = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from balanced_loader import load_balanced_test_data  # noqa: E402


LABELS = ["Direct Reply", "Direct Non-Reply", "Indirect"]
LABEL2ID = {"Direct Reply": 0, "Direct Non-Reply": 1, "Indirect": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def has_weights(model_dir: Path) -> bool:
    required = ["model.safetensors", "pytorch_model.bin", "pytorch_model.bin.index.json"]
    return any((model_dir / f).exists() for f in required)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_sources(
    local_dirs: List[Path],
    hf_repo: str,
    hf_subfolders: List[str],
) -> List[Tuple[str, str, str | None]]:
    """
    Returns list of tuples:
      ("local", "/abs/path", None) or ("hf", "repo-id", "subfolder")
    """
    sources: List[Tuple[str, str, str | None]] = []
    for i, model_dir in enumerate(local_dirs):
        if model_dir.exists() and has_weights(model_dir):
            sources.append(("local", str(model_dir.resolve()), None))
        else:
            sources.append(("hf", hf_repo, hf_subfolders[i]))
    return sources


def load_tokenizer_from_source(source: Tuple[str, str, str | None]):
    kind, base, sub = source
    if kind == "local":
        return AutoTokenizer.from_pretrained(base)
    return AutoTokenizer.from_pretrained(base, subfolder=sub)


def load_model_from_source(source: Tuple[str, str, str | None], device: torch.device):
    kind, base, sub = source
    if kind == "local":
        model = AutoModelForSequenceClassification.from_pretrained(base)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(base, subfolder=sub)
    model.to(device)
    model.eval()
    return model


def predict_logits(
    model,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    all_logits = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits.detach().cpu().numpy()
        all_logits.append(logits)
    return np.concatenate(all_logits, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": {
            ID2LABEL[i]: float(f1_score(y_true, y_pred, labels=[i], average="macro", zero_division=0))
            for i in range(3)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=[ID2LABEL[i] for i in range(3)],
            zero_division=0,
            output_dict=True,
        ),
    }


def parse_active_models(active: str) -> List[int]:
    vals: List[int] = []
    for part in str(active).split(","):
        p = part.strip()
        if not p:
            continue
        try:
            v = int(p)
        except ValueError as exc:
            raise ValueError(f"Invalid model index '{p}' in --active-models") from exc
        if v not in (1, 2, 3):
            raise ValueError(f"Model index must be 1,2,3; got {v}")
        vals.append(v)
    vals = sorted(set(vals))
    if not vals:
        raise ValueError("--active-models resolved to empty set")
    return vals


def main():
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa ensemble on balanced QEvasion subset")
    parser.add_argument(
        "--local-model-1",
        type=Path,
        default=REPO_ROOT / "roberta/roberta-ensemble-model-1/final",
        help="Local path for ensemble model 1",
    )
    parser.add_argument(
        "--local-model-2",
        type=Path,
        default=REPO_ROOT / "roberta/roberta-ensemble-model-2/final",
        help="Local path for ensemble model 2",
    )
    parser.add_argument(
        "--local-model-3",
        type=Path,
        default=REPO_ROOT / "roberta/roberta-ensemble-model-3/final",
        help="Local path for ensemble model 3",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default="gigibot/ensemble-qeval",
        help="HF repo id used as fallback if local weights are missing",
    )
    parser.add_argument("--hf-subfolder-1", type=str, default="model-1")
    parser.add_argument("--hf-subfolder-2", type=str, default="model-2")
    parser.add_argument("--hf-subfolder-3", type=str, default="model-3")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=23,
        help="Balanced examples per class (23 => total 69)",
    )
    parser.add_argument("--dataset-name", type=str, default="ailsntua/QEvasion")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--active-models",
        type=str,
        default="1,2,3",
        help="Comma-separated model indices to use (e.g. 1,2 or 3)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory to save metrics and predictions",
    )
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Missing dependencies. Install required packages, e.g. "
            "`pip install torch transformers datasets scikit-learn`."
        ) from _IMPORT_ERROR

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    active_models = parse_active_models(args.active_models)

    device = get_device()
    local_dirs = [args.local_model_1, args.local_model_2, args.local_model_3]
    hf_subfolders = [args.hf_subfolder_1, args.hf_subfolder_2, args.hf_subfolder_3]
    sources = build_model_sources(local_dirs, args.hf_repo, hf_subfolders)

    print(f"Device: {device}")
    print(f"Active model set: {active_models}")
    print("Model source resolution:")
    for i, s in enumerate(sources, start=1):
        enabled = "yes" if i in active_models else "no"
        if s[0] == "local":
            print(f"  model-{i}: local -> {s[1]} (active={enabled})")
        else:
            print(f"  model-{i}: hf -> {s[1]} / {s[2]} (active={enabled})")

    tokenizer = None
    tokenizer_source = None
    for i, source in enumerate(sources, start=1):
        if i not in active_models:
            continue
        try:
            tokenizer = load_tokenizer_from_source(source)
            tokenizer_source = i
            print(f"Tokenizer loaded from model-{i} source")
            break
        except Exception:
            continue
    if tokenizer is None:
        raise RuntimeError(
            "Failed to load tokenizer from all model sources. "
            "Check local paths and/or HF repo access."
        )

    examples, gold_labels = load_balanced_test_data(
        split=args.split,
        samples_per_label=args.samples_per_label,
        dataset_name=args.dataset_name,
    )
    texts = [f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in examples]
    y_true = np.array([LABEL2ID[x] for x in gold_labels], dtype=np.int64)

    ensemble_logits = None
    per_model_preds: List[List[int]] = []  # per loaded model
    per_model_metrics: Dict[str, Dict] = {}
    loaded_indices: List[int] = []  # 1-based model indices that loaded (e.g. [1, 2] or [1, 2, 3])
    for i, source in enumerate(sources, start=1):
        if i not in active_models:
            continue
        print(f"Loading model {i}/{len(sources)} ...")
        try:
            model = load_model_from_source(source, device)
        except Exception as exc:
            print(
                f"  Warning: Failed to load model-{i} from {source}. "
                "Skipping; will evaluate with the other model(s) only."
            )
            continue

        logits = predict_logits(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            batch_size=args.batch_size,
        )
        preds = np.argmax(logits, axis=-1)
        per_model_preds.append(preds.tolist())
        per_model_metrics[f"model_{i}"] = compute_metrics(y_true, preds)
        pred_dist = Counter(preds.tolist())
        if len(pred_dist) == 1:
            print(
                f"  Warning: model-{i} predicted only one class on this eval subset: {dict(pred_dist)}"
            )
        loaded_indices.append(i)
        ensemble_logits = logits if ensemble_logits is None else (ensemble_logits + logits)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    if not loaded_indices:
        raise RuntimeError(
            "Could not load any ensemble model. Check local paths and HF repo access."
        )
    print(f"Using {len(loaded_indices)} model(s): {loaded_indices}")

    y_pred = np.argmax(ensemble_logits, axis=-1)
    metrics = compute_metrics(y_true, y_pred)
    metrics["gold_distribution"] = dict(Counter(gold_labels))
    metrics["pred_distribution"] = dict(Counter([ID2LABEL[i] for i in y_pred]))
    metrics["per_model_metrics"] = per_model_metrics

    # Recommend a conservative subset based on individual macro-F1 on this labeled eval subset.
    ranked = sorted(
        loaded_indices,
        key=lambda m: per_model_metrics[f"model_{m}"]["f1_macro"],
        reverse=True,
    )
    recommended_models = ranked[:1]
    chance_macro_f1 = 1.0 / len(LABELS)  # balanced 3-class chance baseline ~= 0.3333
    if len(ranked) > 1:
        second = ranked[1]
        if per_model_metrics[f"model_{second}"]["f1_macro"] > chance_macro_f1:
            recommended_models.append(second)

    run = {
        "split": args.split,
        "samples_per_label": args.samples_per_label,
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "active_models": active_models,
        "device": str(device),
        "tokenizer_source_model": tokenizer_source,
        "loaded_models": loaded_indices,
        "recommended_active_models": recommended_models,
        "sources": [
            {"kind": s[0], "base": s[1], "subfolder": s[2]} for s in sources
        ],
    }

    out_dir = args.output_dir / f"ensemble_balanced69_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(run, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "per_model_metrics.json").write_text(json.dumps(per_model_metrics, indent=2))

    # Map model index (1,2,3) to per_model_preds column; use -1 if that model was not loaded
    def pred_id(model_num: int, row_i: int) -> int:
        if model_num not in loaded_indices:
            return -1
        idx = loaded_indices.index(model_num)
        return per_model_preds[idx][row_i]

    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "question",
                "answer",
                "gold_label",
                "pred_label",
                "correct",
                "model1_pred_id",
                "model2_pred_id",
                "model3_pred_id",
            ],
        )
        writer.writeheader()
        for i, ex in enumerate(examples):
            writer.writerow(
                {
                    "index": i,
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "gold_label": gold_labels[i],
                    "pred_label": ID2LABEL[int(y_pred[i])],
                    "correct": int(int(y_pred[i]) == int(y_true[i])),
                    "model1_pred_id": pred_id(1, i),
                    "model2_pred_id": pred_id(2, i),
                    "model3_pred_id": pred_id(3, i),
                }
            )

    print("\n=== Ensemble metrics (balanced subset) ===")
    print(f"n_samples:       {metrics['n_samples']}")
    print(f"accuracy:        {metrics['accuracy']:.6f}")
    print(f"precision_macro: {metrics['precision_macro']:.6f}")
    print(f"recall_macro:    {metrics['recall_macro']:.6f}")
    print(f"f1_macro:        {metrics['f1_macro']:.6f}")
    print("\nPer-model macro F1:")
    for mi in loaded_indices:
        key = f"model_{mi}"
        print(f"  {key}: {per_model_metrics[key]['f1_macro']:.6f}")
    rec = ",".join(str(x) for x in recommended_models)
    rerun_cmd = (
        f"{shlex.quote(sys.executable)} {shlex.quote(str(Path(__file__).resolve()))} "
        f"--split {args.split} --samples-per-label {args.samples_per_label} "
        f"--batch-size {args.batch_size} --active-models {rec}"
    )
    print(f"\nRecommended active models from this run: {recommended_models}")
    print("Recommended rerun command:")
    print(rerun_cmd)
    print(f"\nSaved to: {out_dir}")
    print(f"- {out_dir / 'run_config.json'}")
    print(f"- {out_dir / 'metrics.json'}")
    print(f"- {out_dir / 'per_model_metrics.json'}")
    print(f"- {pred_csv}")


if __name__ == "__main__":
    main()
