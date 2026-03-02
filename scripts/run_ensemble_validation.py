#!/usr/bin/env python3
"""
Ensemble validation: RoBERTa (binary DNR) + Granite (3-class) on QEvasion test set (308 samples).
Shows Granite standalone, RoBERTa standalone, and Ensemble in one run.

Primary data source: clear-non-reply-predictions-roberta.csv (saved from the direct-non-reply notebook).
This file contains both the test data AND RoBERTa predictions (try_1, try_2, try_3, majority_wins).

If pre-computed Granite log exists, uses that; otherwise runs inference from checkpoint.

Run from repo root:
  /usr/bin/python3 scripts/run_ensemble_validation.py
"""

from __future__ import annotations

import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Fixed paths — QEvasion test set (308)
# ---------------------------------------------------------------------------
# Search paths for RoBERTa predictions CSV (Kaggle, Colab, local)
ROBERTA_CSV_CANDIDATES = [
    REPO_ROOT / "clear-non-reply-predictions-roberta.csv",
    Path("/kaggle/input/clear-non-reply-preds/clear-non-reply-predictions-roberta.csv"),
    Path("/kaggle/input/roberta-predictions/clear-non-reply-predictions-roberta.csv"),
    Path("/kaggle/working/clear-non-reply-predictions-roberta.csv"),
    Path("/content/clear-non-reply-predictions-roberta.csv"),
]

def find_roberta_csv() -> Path:
    """Find RoBERTa predictions CSV from candidate paths."""
    for p in ROBERTA_CSV_CANDIDATES:
        if p.exists():
            return p
    return ROBERTA_CSV_CANDIDATES[0]  # fallback for error message

ROBERTA_PREDICTIONS_CSV = find_roberta_csv()

# Fallback eval CSV (SemEval 237)
SEMEVAL_CSV = REPO_ROOT / "dataset" / "clarity_task_evaluation_dataset.csv"

ROBERTA_MODEL_DIR = REPO_ROOT / "evasion_binary_roberta_2"

# Granite predictions for QEvasion (separate from SemEval granite_answers.txt)
GRANITE_QEVASION_LOG = REPO_ROOT / "granite_qevasion_predictions.txt"
GRANITE_BASE_MODEL = "ibm-granite/granite-3.2-8b-instruct"

# Search paths for Granite checkpoint (Kaggle, Colab, local)
GRANITE_ADAPTER_CANDIDATES = [
    REPO_ROOT / "checkpoint64",
    Path("/kaggle/input/granite-checkpoint64"),
    Path("/kaggle/input/checkpoint64"),
    Path("/kaggle/input/granite-lora-checkpoint"),
    Path("/kaggle/working/checkpoint64"),
    Path("/content/checkpoint64"),
]

def find_granite_adapter() -> Path:
    """Find Granite adapter directory from candidate paths."""
    for p in GRANITE_ADAPTER_CANDIDATES:
        if p.exists() and (p / "adapter_model.safetensors").exists():
            return p
    return GRANITE_ADAPTER_CANDIDATES[0]  # fallback for error message

GRANITE_ADAPTER_DIR = find_granite_adapter()

OUT_JSON = REPO_ROOT / "results" / "ensemble_qevasion_metrics.json"

LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent": "Indirect",
    "Ambivalent Reply": "Indirect",
}
LABELS = ["Direct Reply", "Direct Non-Reply", "Indirect"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def map_label(raw: str) -> str:
    v = str(raw).strip()
    return LABEL_MAP.get(v, v if v in LABELS else "Indirect")


def has_weights(d: Path) -> bool:
    return any((d / f).exists() for f in [
        "model.safetensors", "pytorch_model.bin", "adapter_model.safetensors",
    ])

# ---------------------------------------------------------------------------
# Data loading — uses RoBERTa predictions CSV as primary source
# ---------------------------------------------------------------------------

def load_eval_data() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Load eval data, true labels, and RoBERTa predictions from clear-non-reply-predictions-roberta.csv.
    
    Returns:
        eval_df: DataFrame with index, question, answer
        labels: true labels (mapped to Direct Reply / Direct Non-Reply / Indirect)
        roberta_df: DataFrame with RoBERTa predictions (try_1..try_3, majority)
    """
    if not ROBERTA_PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"Primary eval data not found: {ROBERTA_PREDICTIONS_CSV}\n"
            "Run the direct-non-reply notebook first to generate it."
        )
    
    print(f"[Data] Loading from: {ROBERTA_PREDICTIONS_CSV.name}")
    df = pd.read_csv(ROBERTA_PREDICTIONS_CSV)
    
    # Build eval DataFrame
    idx_col = "index" if "index" in df.columns else None
    q_col = "question" if "question" in df.columns else "interview_question"
    a_col = "interview_answer" if "interview_answer" in df.columns else "answer"
    
    eval_df = pd.DataFrame({
        "index": df[idx_col] if idx_col else range(len(df)),
        "question": df[q_col].fillna(""),
        "answer": df[a_col].fillna(""),
    }).reset_index(drop=True)
    
    # Extract labels
    if "clarity_label" not in df.columns:
        raise ValueError("clarity_label column missing from predictions CSV")
    labels = [map_label(str(v)) for v in df["clarity_label"]]
    
    if len(set(labels)) <= 1:
        raise ValueError(f"Labels are single-class: {set(labels)}")
    
    # Build RoBERTa predictions DataFrame
    roberta_df = df[["index", "try_1", "try_2", "try_3"]].copy() if "index" in df.columns else df[["try_1", "try_2", "try_3"]].copy()
    if "index" not in roberta_df.columns:
        roberta_df["index"] = range(len(roberta_df))
    
    if "majority_wins" in df.columns:
        roberta_df["roberta_nonreply_majority"] = df["majority_wins"]
    else:
        roberta_df["roberta_nonreply_majority"] = (
            roberta_df[["try_1", "try_2", "try_3"]].sum(axis=1) >= 2
        ).astype(int)
    
    print(f"[Data] Loaded {len(eval_df)} samples, label dist: {dict(Counter(labels))}")
    return eval_df, labels, roberta_df


# ---------------------------------------------------------------------------
# Granite: load pre-computed log or run inference
# ---------------------------------------------------------------------------

def parse_granite_log(log_path: Path) -> pd.DataFrame:
    pat = re.compile(
        r"^\[(\d+)/(\d+)\]\s+True=([^|]+)\s+\|\s+Pred=([^|]+)\s+\|\s+[^|]*\|\s+votes=(\{.*?\})"
    )
    rows = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            idx1 = int(m.group(1))
            pred = map_label(m.group(4).strip())
            try:
                vd = ast.literal_eval(m.group(5).strip())
                if not isinstance(vd, dict):
                    vd = {}
            except Exception:
                vd = {}
            vote_labels = []
            for k, n in vd.items():
                try:
                    vote_labels.extend([map_label(k)] * int(n))
                except Exception:
                    pass
            rows.append({"sample_idx": idx1 - 1, "pred": pred, "vote_labels": vote_labels})
    if not rows:
        raise ValueError(f"No parsable lines in {log_path}")
    return pd.DataFrame(rows).drop_duplicates(subset=["sample_idx"], keep="last").sort_values("sample_idx").reset_index(drop=True)


def load_or_run_granite(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with sample_idx, pred, vote_labels."""
    expected_n = len(eval_df)
    
    # Check if QEvasion-specific log exists with correct sample count
    if GRANITE_QEVASION_LOG.exists():
        try:
            gdf = parse_granite_log(GRANITE_QEVASION_LOG)
            if len(gdf) >= expected_n * 0.9:  # at least 90% coverage
                print(f"[Granite] Using QEvasion log: {GRANITE_QEVASION_LOG.name} ({len(gdf)} samples)")
                return gdf
            else:
                print(f"[Granite] QEvasion log has only {len(gdf)}/{expected_n} samples, will re-run inference")
        except Exception as e:
            print(f"[Granite] Failed to parse {GRANITE_QEVASION_LOG.name}: {e}")

    print(f"[Granite] Running inference on {expected_n} samples from {GRANITE_ADAPTER_DIR.name}/ ...")
    if not has_weights(GRANITE_ADAPTER_DIR):
        raise FileNotFoundError(
            f"Granite adapter weights not found in {GRANITE_ADAPTER_DIR}. "
            "Either place adapter_model.safetensors there (from Colab) or provide granite_answers.txt."
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"[Granite] Loading base model: {GRANITE_BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(GRANITE_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(GRANITE_BASE_MODEL, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base_model, str(GRANITE_ADAPTER_DIR))
    model.eval()

    from granite_clarity_strategy import GraniteClarityStrategy
    strategy = GraniteClarityStrategy()

    rows = []
    for _, row in eval_df.iterrows():
        q, a, idx = row["question"], row["answer"], int(row["index"])
        prompt = strategy.build_prompt(q, a)
        messages = [{"role": "user", "content": prompt}]

        vote_labels = []
        for _ in range(3):
            try:
                formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                formatted = prompt
            inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=300, temperature=0.7, do_sample=True,
                                     pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            try:
                parsed = strategy.extract_json(text)
                vote_labels.append(map_label(parsed.get("label", "Indirect")))
            except Exception:
                vote_labels.append("Indirect")

        vote_counts = Counter(vote_labels)
        pred = vote_counts.most_common(1)[0][0]
        rows.append({"sample_idx": idx, "pred": pred, "vote_labels": vote_labels})
        print(f"  [{idx+1:03d}/{len(eval_df)}] Pred={pred} | votes={dict(vote_counts)}")

    gdf = pd.DataFrame(rows)
    with GRANITE_QEVASION_LOG.open("w", encoding="utf-8") as f:
        for _, r in gdf.iterrows():
            f.write(f"[{r['sample_idx']+1:03d}/{len(gdf)}] Pred={r['pred']} | votes={dict(Counter(r['vote_labels']))}\n")
    print(f"[Granite] Saved log → {GRANITE_QEVASION_LOG.name}")
    return gdf

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(true: list[str], pred: list[str]) -> dict:
    label_to_idx = {l: i for i, l in enumerate(LABELS)}
    cm = [[0] * 3 for _ in range(3)]
    correct = 0
    for t, p in zip(true, pred):
        cm[label_to_idx[t]][label_to_idx[p]] += 1
        if t == p:
            correct += 1
    n = max(1, len(true))
    per_class = {}
    f1s = []
    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(3)) - tp
        fn = sum(cm[i][c] for c in range(3)) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    return {"accuracy": round(correct / n, 4), "macro_f1": round(sum(f1s) / 3, 4),
            "confusion_matrix": cm, "per_class": per_class}


def print_metrics(title: str, metrics: dict, n: int) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}  (N={n})")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Macro F1:  {metrics['macro_f1']:.4f}")
    for label in LABELS:
        pc = metrics["per_class"][label]
        print(f"  {label:20s}  P={pc['precision']:.4f}  R={pc['recall']:.4f}  F1={pc['f1']:.4f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    eval_df, labels, roberta_df = load_eval_data()
    granite_df = load_or_run_granite(eval_df)

    rmap = roberta_df.set_index("index")
    gmap = granite_df.set_index("sample_idx")

    # Use all indices from eval (RoBERTa is guaranteed same size since it comes from same CSV)
    all_idx = list(range(len(eval_df)))
    
    # Check Granite coverage
    granite_idx = set(gmap.index)
    common_idx = [i for i in all_idx if i in granite_idx]
    
    if not common_idx:
        print("[Warning] Granite log has no matching indices — using all eval indices, Granite preds = Indirect")
        common_idx = all_idx
    elif len(common_idx) < len(all_idx):
        print(f"Note: Granite covers {len(common_idx)}/{len(all_idx)} samples")

    true_used = [labels[i] for i in common_idx]
    n = len(common_idx)

    # --- Granite standalone ---
    granite_pred = []
    for i in common_idx:
        if i in gmap.index:
            granite_pred.append(map_label(gmap.loc[i, "pred"]))
        else:
            granite_pred.append("Indirect")
    granite_metrics = compute_metrics(true_used, granite_pred)
    print_metrics("Granite standalone (3-class)", granite_metrics, n)

    # --- RoBERTa standalone (binary → 3-class) ---
    roberta_3class = ["Direct Non-Reply" if int(rmap.loc[i, "roberta_nonreply_majority"]) == 1 else "Indirect" for i in common_idx]
    roberta_metrics = compute_metrics(true_used, roberta_3class)
    print_metrics("RoBERTa standalone (binary DNR, rest=Indirect)", roberta_metrics, n)

    true_bin = [1 if t == "Direct Non-Reply" else 0 for t in true_used]
    pred_bin = [int(rmap.loc[i, "roberta_nonreply_majority"]) for i in common_idx]
    tp = sum(g == 1 and p == 1 for g, p in zip(true_bin, pred_bin))
    fp = sum(g == 0 and p == 1 for g, p in zip(true_bin, pred_bin))
    fn = sum(g == 1 and p == 0 for g, p in zip(true_bin, pred_bin))
    bp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    br = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    bf1 = (2 * bp * br / (bp + br)) if (bp + br) > 0 else 0.0
    print(f"  --- Binary DNR:  P={bp:.4f}  R={br:.4f}  F1={bf1:.4f}")

    # --- Ensemble ---
    ensemble_pred = []
    for i in common_idx:
        r = rmap.loc[i]
        r_votes = int(r.get("try_1", 0)) + int(r.get("try_2", 0)) + int(r.get("try_3", 0))
        
        if i in gmap.index:
            g = gmap.loc[i]
            g_vlabels = g["vote_labels"] if isinstance(g["vote_labels"], list) else []
            g_dnr = sum(1 for v in g_vlabels if v == "Direct Non-Reply")
            g_dr = sum(1 for v in g_vlabels if v == "Direct Reply")
            g_ind = sum(1 for v in g_vlabels if v == "Indirect")
            if not g_vlabels:
                gp = map_label(g.get("pred", "Indirect"))
                g_dnr, g_dr, g_ind = (3, 0, 0) if gp == "Direct Non-Reply" else (0, 3, 0) if gp == "Direct Reply" else (0, 0, 3)
        else:
            g_dnr, g_dr, g_ind = 0, 0, 3

        score_dnr = (r_votes + g_dnr) / 2.0
        score_dr = float(g_dr)
        score_ind = float(g_ind) + 1.0
        if score_dnr >= 1.5 and score_dnr >= score_dr and score_dnr >= score_ind:
            ensemble_pred.append("Direct Non-Reply")
        elif score_dr >= score_ind:
            ensemble_pred.append("Direct Reply")
        else:
            ensemble_pred.append("Indirect")

    ensemble_metrics = compute_metrics(true_used, ensemble_pred)
    print_metrics("Ensemble: RoBERTa + Granite (sum_both_balanced_v4)", ensemble_metrics, n)

    # --- Save ---
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "n": n,
        "granite_standalone": granite_metrics,
        "roberta_standalone_3class": roberta_metrics,
        "roberta_binary_dnr": {"precision": round(bp, 4), "recall": round(br, 4), "f1": round(bf1, 4)},
        "ensemble": ensemble_metrics,
        "fusion_rule": "sum_both_balanced_v4",
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
