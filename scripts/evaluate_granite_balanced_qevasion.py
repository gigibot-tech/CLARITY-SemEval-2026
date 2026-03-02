#!/usr/bin/env python3
"""
Evaluate a Granite LoRA checkpoint on a balanced QEvasion subset.

Default behavior evaluates all 69 balanced test samples (23 per class) by:
1) loading QEvasion test split
2) stratified balancing via balanced_loader
3) running Granite generation-based label prediction
4) reporting accuracy/macro-F1/confusion matrix and saving predictions
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

_IMPORT_ERROR = None
try:
    from datasets import load_dataset
    import torch
    from peft import PeftModel
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    _IMPORT_ERROR = exc
    torch = None
    PeftModel = None
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    f1_score = None
    precision_score = None
    recall_score = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    load_dataset = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from balanced_loader import load_balanced_test_data  # noqa: E402


LABELS = ["Direct Reply", "Direct Non-Reply", "Indirect"]
QE_LABEL_TO_CLARITY = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_label(raw_label: str) -> str:
    label = (raw_label or "").strip()
    if label in LABELS:
        return label
    if label in QE_LABEL_TO_CLARITY:
        return QE_LABEL_TO_CLARITY[label]

    label_l = label.lower()
    if "direct non" in label_l or "non-reply" in label_l or "decline" in label_l:
        return "Direct Non-Reply"
    if "direct reply" in label_l or "clear reply" in label_l:
        return "Direct Reply"
    if "indirect" in label_l or "ambivalent" in label_l or "evas" in label_l:
        return "Indirect"
    return "Indirect"


def build_prompt(question: str, answer: str) -> str:
    return f"""You are analyzing political interview answers for clarity classification.

Question: {question}
Answer: {answer}

Analyze the answer step-by-step:
1. Does it directly address the question?
2. Is it evasive or indirect?
3. Does it decline to answer?

Provide your reasoning and then classify as one of:
- "Direct Reply": Directly answers the question
- "Direct Non-Reply": Explicitly declines or claims inability to answer
- "Indirect": Evasive, indirect, or partially answers

Respond in JSON format:
{{
  "reasoning": "Your step-by-step analysis...",
  "label": "Direct Reply|Direct Non-Reply|Indirect"
}}"""


def extract_label_from_response(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "label" in obj:
            return normalize_label(str(obj["label"]))
    except Exception:
        pass

    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, dict) and "label" in obj:
                return normalize_label(str(obj["label"]))
        except Exception:
            pass

    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
    if label_match:
        return normalize_label(label_match.group(1))

    return normalize_label(cleaned)


def resolve_adapter_path(adapter_path: Path) -> Path:
    if adapter_path.exists():
        return adapter_path.resolve()

    candidate = REPO_ROOT / "granite_clarity_finetuned" / "checkpoint-60"
    if candidate.exists():
        return candidate.resolve()

    candidate = REPO_ROOT / "granite_clarity_finetuned"
    if candidate.exists():
        return candidate.resolve()

    raise FileNotFoundError(f"Adapter path not found: {adapter_path}")


def load_model_and_tokenizer(
    adapter_path: Path,
    device: torch.device,
    local_files_only: bool,
    use_device_map_auto: bool,
):
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")

    config = json.loads(config_path.read_text())
    base_model_name = config.get("base_model_name_or_path", "ibm-granite/granite-3.2-2b-instruct")
    print(f"Base model: {base_model_name}")
    print(f"Adapter path: {adapter_path}")

    # Prefer base model tokenizer; some checkpoint folders can resolve to a tokenizer
    # artifact that tokenizes everything to empty input IDs.
    tokenizer_sources: List[object] = [base_model_name]
    parent = adapter_path.parent
    if parent != adapter_path:
        tokenizer_sources.append(parent)
    tokenizer_sources.append(adapter_path)

    tokenizer = None
    for source in tokenizer_sources:
        try:
            candidate = AutoTokenizer.from_pretrained(source, local_files_only=local_files_only)
            sanity = candidate("tokenizer_sanity_check", return_tensors="pt").get("input_ids")
            if sanity is None or sanity.shape[-1] == 0:
                print(f"Tokenizer source rejected (empty tokenization): {source}")
                continue
            tokenizer = candidate
            print(f"Tokenizer loaded from: {source}")
            break
        except Exception:
            continue
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer from adapter path or base model")

    model_load_kwargs = {
        "torch_dtype": "auto",
        "local_files_only": local_files_only,
    }
    # device_map='auto' can create meta/offload structures that break PEFT adapter loading
    # in some environments; keep it opt-in.
    if use_device_map_auto:
        model_load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_load_kwargs)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()

    # When not using CUDA, explicitly place model on selected device if possible.
    if not torch.cuda.is_available():
        try:
            model.to(device)
        except Exception:
            pass

    return model, tokenizer


def load_full_split_data(split: str, dataset_name: str) -> Tuple[List[Dict], List[str]]:
    dataset = load_dataset(dataset_name)
    if split not in dataset:
        raise ValueError(f"Split '{split}' not found. Available: {list(dataset.keys())}")

    label_mapping = {
        "Clear Reply": "Direct Reply",
        "Clear Non-Reply": "Direct Non-Reply",
        "Ambivalent Reply": "Indirect",
        "Ambivalent": "Indirect",
    }

    examples: List[Dict] = []
    labels: List[str] = []
    for item in dataset[split]:
        raw = item.get("clarity_label", "")
        mapped = label_mapping.get(raw, raw)
        if mapped not in LABELS:
            continue
        examples.append(
            {
                "question": str(item.get("interview_question", item.get("question", ""))),
                "answer": str(item.get("interview_answer", "")),
            }
        )
        labels.append(mapped)
    return examples, labels


def generate_response(
    model,
    tokenizer,
    device: torch.device,
    question: str,
    answer: str,
    temperature: float,
    max_new_tokens: int,
    use_cache: bool,
) -> str:
    messages = [{"role": "user", "content": build_prompt(question, answer)}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        formatted = messages[0]["content"]

    inputs = tokenizer(formatted, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    do_sample = temperature > 0.0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-5) if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=use_cache,
        )
    text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip()


def predict_with_voting(
    model,
    tokenizer,
    device: torch.device,
    question: str,
    answer: str,
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
    use_cache: bool,
) -> Tuple[str, Dict[str, int], str]:
    votes: List[str] = []
    last_response = ""

    for _ in range(num_samples):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=question,
            answer=answer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )
        last_response = response
        votes.append(extract_label_from_response(response))

    vote_counter = Counter(votes)
    pred_label = vote_counter.most_common(1)[0][0] if vote_counter else "Indirect"
    return pred_label, dict(vote_counter), last_response


def evaluate(
    model,
    tokenizer,
    device: torch.device,
    examples: List[Dict],
    true_labels: List[str],
    num_samples: int,
    temperature: float,
    max_new_tokens: int,
    log_every: int,
    use_cache: bool,
    intermediate_metrics_path: Path | None = None,
    progress_log_path: Path | None = None,
) -> Tuple[Dict, List[Dict]]:
    predictions: List[str] = []
    rows: List[Dict] = []

    total = len(examples)
    correct_so_far = 0
    for idx, (example, gold) in enumerate(zip(examples, true_labels), start=1):
        pred, votes, response = predict_with_voting(
            model=model,
            tokenizer=tokenizer,
            device=device,
            question=example["question"],
            answer=example["answer"],
            num_samples=num_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
        )
        predictions.append(pred)
        if pred == gold:
            correct_so_far += 1
        rows.append(
            {
                "index": idx - 1,
                "question": example["question"],
                "answer": example["answer"],
                "gold_label": gold,
                "pred_label": pred,
                "correct": int(pred == gold),
                "votes": json.dumps(votes, ensure_ascii=False),
                "raw_response": response,
            }
        )
        if idx == 1 or idx == total or (log_every > 0 and idx % log_every == 0):
            running_acc = correct_so_far / idx
            seen_gold = true_labels[:idx]
            seen_pred = predictions[:]
            running_precision = float(
                precision_score(seen_gold, seen_pred, average="macro", zero_division=0)
            )
            running_recall = float(
                recall_score(seen_gold, seen_pred, average="macro", zero_division=0)
            )
            running_f1 = float(
                f1_score(seen_gold, seen_pred, average="macro", zero_division=0)
            )
            running_gold_dist = dict(Counter(seen_gold))
            running_pred_dist = dict(Counter(seen_pred))
            print(
                f"[{idx}/{total}] running_acc={running_acc:.4f} "
                f"running_macro_f1={running_f1:.4f} "
                f"running_precision={running_precision:.4f} "
                f"running_recall={running_recall:.4f} "
                f"last_gold={gold} last_pred={pred} "
                f"pred_dist={running_pred_dist}"
            )
            if intermediate_metrics_path is not None:
                intermediate = {
                    "seen_examples": idx,
                    "total_examples": total,
                    "running_accuracy": running_acc,
                    "running_precision_macro": running_precision,
                    "running_recall_macro": running_recall,
                    "running_macro_f1": running_f1,
                    "running_gold_distribution": running_gold_dist,
                    "running_pred_distribution": running_pred_dist,
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                }
                intermediate_metrics_path.write_text(json.dumps(intermediate, indent=2))
            if progress_log_path is not None:
                progress = {
                    "seen_examples": idx,
                    "total_examples": total,
                    "running_accuracy": running_acc,
                    "running_precision_macro": running_precision,
                    "running_recall_macro": running_recall,
                    "running_macro_f1": running_f1,
                    "last_gold": gold,
                    "last_pred": pred,
                    "running_pred_distribution": running_pred_dist,
                    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                }
                with progress_log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(progress, ensure_ascii=False) + "\n")

    metrics = {
        "n_examples": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "precision_macro": float(precision_score(true_labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
        "per_class_f1": {
            label: float(f1_score(true_labels, predictions, labels=[label], average="macro", zero_division=0))
            for label in LABELS
        },
        "gold_distribution": dict(Counter(true_labels)),
        "pred_distribution": dict(Counter(predictions)),
        "classification_report": classification_report(
            true_labels,
            predictions,
            labels=LABELS,
            target_names=LABELS,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=LABELS).tolist(),
    }
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate Granite checkpoint on QEvasion subset")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=REPO_ROOT / "granite_clarity_finetuned" / "checkpoint-60",
        help="Path to LoRA adapter/checkpoint folder",
    )
    parser.add_argument("--dataset-name", type=str, default="ailsntua/QEvasion", help="Hugging Face QEvasion dataset")
    parser.add_argument("--split", choices=["test", "train"], default="test", help="QEvasion split")
    parser.add_argument(
        "--full-split",
        action="store_true",
        help="Evaluate full split (~308 test samples) instead of balanced subset",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=None,
        help="Balanced samples per class (default: min class count, typically 23 on test => 69 total)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling")
    parser.add_argument("--num-samples", type=int, default=1, help="Votes per example (self-consistency)")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max generated tokens")
    parser.add_argument("--log-every", type=int, default=10, help="Progress logging interval (examples)")
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        default=True,
        help="Write running metrics during evaluation (default: True)",
    )
    parser.add_argument(
        "--no-save-intermediate",
        action="store_false",
        dest="save_intermediate",
        help="Disable writing intermediate metrics/progress logs",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Enable generation KV-cache (off by default for Granite+PEFT compatibility)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=True,
        help="Use only local Hugging Face cache for base model/tokenizer (default: True)",
    )
    parser.add_argument(
        "--allow-online",
        action="store_false",
        dest="local_files_only",
        help="Allow online fetching from Hugging Face for missing files",
    )
    parser.add_argument(
        "--use-device-map-auto",
        action="store_true",
        help="Enable device_map='auto' when loading base Granite model (off by default)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory where evaluation artifacts are saved",
    )
    args = parser.parse_args()

    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Missing dependencies. Install required packages before running this script, e.g. "
            "`pip install torch transformers peft datasets scikit-learn`."
        ) from _IMPORT_ERROR

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    adapter_path = resolve_adapter_path(args.adapter_path)
    run_name = "granite_full_qevasion" if args.full_split else "granite_balanced_qevasion"
    output_dir = args.output_dir / f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device}")
    print(f"Sampling seed: {args.seed}")

    model, tokenizer = load_model_and_tokenizer(
        adapter_path,
        device,
        local_files_only=args.local_files_only,
        use_device_map_auto=args.use_device_map_auto,
    )

    if args.full_split:
        examples, true_labels = load_full_split_data(split=args.split, dataset_name=args.dataset_name)
        print(f"Evaluating on full {args.split} split: {len(examples)} examples")
    else:
        examples, true_labels = load_balanced_test_data(
            split=args.split,
            samples_per_label=args.samples_per_label,
            dataset_name=args.dataset_name,
        )
        print(f"Evaluating on {len(examples)} balanced examples")

    intermediate_metrics_path = output_dir / "intermediate_metrics.json" if args.save_intermediate else None
    progress_log_path = output_dir / "progress.jsonl" if args.save_intermediate else None

    metrics, rows = evaluate(
        model=model,
        tokenizer=tokenizer,
        device=device,
        examples=examples,
        true_labels=true_labels,
        num_samples=max(1, args.num_samples),
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        log_every=args.log_every,
        use_cache=args.use_cache,
        intermediate_metrics_path=intermediate_metrics_path,
        progress_log_path=progress_log_path,
    )

    run_config = {
        "adapter_path": str(adapter_path),
        "dataset_name": args.dataset_name,
        "split": args.split,
        "full_split": args.full_split,
        "samples_per_label": args.samples_per_label,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": args.use_cache,
        "save_intermediate": args.save_intermediate,
        "local_files_only": args.local_files_only,
        "use_device_map_auto": args.use_device_map_auto,
        "device": str(device),
    }

    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    pred_path = output_dir / "predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "question",
                "answer",
                "gold_label",
                "pred_label",
                "correct",
                "votes",
                "raw_response",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Final Metrics ===")
    print(f"Samples:    {metrics['n_examples']}")
    print(f"Accuracy:   {metrics['accuracy']:.4f}")
    print(f"Macro-F1:   {metrics['macro_f1']:.4f}")
    print(f"Precision:  {metrics['precision_macro']:.4f}")
    print(f"Recall:     {metrics['recall_macro']:.4f}")
    print(f"Gold dist:  {metrics['gold_distribution']}")
    print(f"Pred dist:  {metrics['pred_distribution']}")
    print(f"\nSaved to: {output_dir}")
    print(f"- {output_dir / 'run_config.json'}")
    print(f"- {output_dir / 'metrics.json'}")
    print(f"- {pred_path}")


if __name__ == "__main__":
    main()
