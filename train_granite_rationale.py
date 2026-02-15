#!/usr/bin/env python3
"""
Train Granite 3.2 on rationale dataset for CLARITY.

- Loads rationale CSV (qevasion_rationale_dataset), uses reasoning + label columns.
- Builds SFT data in Granite eval format (reasoning + JSON label).
- Optional few-shot: prepend K examples to prompt; track which few-shot sets help most on eval.
- Monitors/samples during training; evaluates on (question, answer) pairs with voting.

Device support (single backend per run):
- GPU (CUDA): default; supports --load-8bit and device_map.
- TPU (Colab/Cloud): auto-detected when torch_xla is installed and COLAB_TPU_ADDR is set
  or XLA world size >= 1. Uses bfloat16 + LoRA; 8-bit and device_map disabled.
- MPS/CPU: fallback; 8-bit disabled.

If you hit CUDA OOM on a small GPU: reduce --batch-size (e.g. 1), --max-length (e.g. 512),
or set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:1. Or run on TPU (Colab: Runtime > Change runtime type > TPU, then pip install torch_xla).
"""

import gc
import json
import os
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# Suppress known resource_tracker semaphore warning at shutdown (we use num_workers=0)
warnings.filterwarnings("ignore", message="resource_tracker: There appear to be", category=UserWarning, module="multiprocessing.resource_tracker")

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

# Paths - Always use Google Drive as source of truth
DRIVE_BASE = Path.home() / "Library" / "CloudStorage" / "GoogleDrive-andrearachetta@gmail.com" / "My Drive"
RATIONALE_FOLDER = DRIVE_BASE / "granite_clarity" / "qevasion_rationale"
RATIONALE_CSV = RATIONALE_FOLDER / "qevasion_rationale_dataset_20260204_163024.csv"
OUTPUT_DIR = DRIVE_BASE / "granite_clarity" / "granite_clarity_finetuned"
EVAL_SPLIT_NAME = "test"  # QEvasion split for eval

REQUIRED_CSV_COLUMNS = [
    "interview_question", "interview_answer", "clarity_label",
    "verdict_match", "initial_reasoning",
]

# Label mapping: QEvasion -> CLARITY submission format
CLARITY_LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}

REVERSE_LABEL_MAP = {v: k for k, v in CLARITY_LABEL_MAP.items()}


def resolve_rationale_csv(folder: Path, default_path: Path) -> Path:
    """
    If folder exists and contains a CSV with strictly more rows than default_path, return that path;
    else return default_path. Uses required columns to validate CSVs.
    """
    if not folder.exists() or not folder.is_dir():
        return default_path
    default_rows = 0
    if default_path.exists():
        try:
            default_rows = len(pd.read_csv(default_path))
        except Exception:
            pass
    best_path = default_path
    best_rows = default_rows
    for csv_path in folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_path)
            if not all(c in df.columns for c in REQUIRED_CSV_COLUMNS):
                continue
            n = len(df)
            if n > best_rows:
                best_rows = n
                best_path = csv_path
        except Exception:
            continue
    # Use the longer file only if it has strictly more rows than the default
    return best_path if best_rows > default_rows else default_path


def load_rationale_csv(csv_path: Path) -> pd.DataFrame:
    """Load rationale CSV and filter to rows with valid reasoning and verdict_match."""
    df = pd.read_csv(csv_path)
    # verdict_match can be string "True"/"False" or bool
    df["verdict_match"] = df["verdict_match"].astype(str).str.lower().eq("true")
    df = df[df["verdict_match"]]
    df = df[df["initial_reasoning"].notna() & (df["initial_reasoning"].astype(str).str.len() > 0)]
    df["final_verdict"] = df["final_verdict"].fillna(df["initial_verdict"]).fillna(df["clarity_label"])
    return df


def drop_conflicting_qa_rows(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, int]:
    """
    Drop rows where identical (question, answer) pairs have conflicting clarity_label values.
    Returns (clean_df, dropped_rows, conflicting_pair_count).
    """
    key = (
        df["interview_question"].astype(str).str.strip()
        + " ||| "
        + df["interview_answer"].astype(str).str.strip()
    )
    per_key = df.assign(_k=key).groupby("_k")["clarity_label"].nunique()
    bad_keys = set(per_key[per_key > 1].index.tolist())
    if not bad_keys:
        return df, 0, 0
    mask = key.isin(bad_keys)
    dropped_rows = int(mask.sum())
    conflict_pairs = int(len(bad_keys))
    return df[~mask].copy(), dropped_rows, conflict_pairs


def balance_rationale_df(df: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    """
    Balance rationale rows by mapped CLARITY label.
    mode: none | downsample | upsample
    """
    if mode == "none":
        return df

    tmp = df.copy()
    tmp["_mapped_label"] = tmp.apply(row_to_clarity_label, axis=1)
    groups = {label: g for label, g in tmp.groupby("_mapped_label")}
    if not groups:
        return df

    counts = {k: len(v) for k, v in groups.items()}
    if mode == "downsample":
        target = min(counts.values())
        parts = [g.sample(n=target, random_state=seed, replace=False) for g in groups.values()]
    elif mode == "upsample":
        target = max(counts.values())
        parts = [
            g.sample(n=target, random_state=seed, replace=(len(g) < target))
            for g in groups.values()
        ]
    else:
        raise ValueError(f"Unknown balance mode: {mode}")

    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out.drop(columns=["_mapped_label"], errors="ignore")


def row_to_clarity_label(row: pd.Series) -> str:
    """Map a row's label to CLARITY format."""
    label = row.get("final_verdict") or row.get("clarity_label") or row.get("initial_verdict")
    if pd.isna(label):
        return "Indirect"
    label = str(label).strip()
    return CLARITY_LABEL_MAP.get(label, label if label in ("Direct Reply", "Direct Non-Reply", "Indirect") else "Indirect")


def build_assistant_output(row: pd.Series) -> str:
    """Build target assistant output: JSON with reasoning and label."""
    reasoning = str(row.get("initial_reasoning") or "").strip()
    correction = row.get("correction_applied")
    if correction in (True, "True", "true", "1", 1):
        corrective = str(row.get("corrective_reasoning") or "").strip()
        if corrective:
            reasoning = reasoning + "\n\n[Correction]\n" + corrective
    label = row_to_clarity_label(row)
    return json.dumps({"reasoning": reasoning, "label": label}, ensure_ascii=False)


def build_user_prompt(question: str, answer: str, few_shot_examples: Optional[List[Dict]] = None) -> str:
    """Same instruction as GraniteClarityEvaluation / granite_clarity_strategy."""
    prompt = f"""You are analyzing political interview answers for clarity classification.

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
    if few_shot_examples:
        examples_text = []
        for ex in few_shot_examples:
            examples_text.append(
                f"Example:\nQuestion: {ex['question'][:200]}...\nAnswer: {ex['answer'][:200]}...\n"
                f"Response: {ex['output']}"
            )
        prompt = "Here are some examples.\n\n" + "\n\n".join(examples_text) + "\n\nNow do the following.\n\n" + prompt
    return prompt


def build_training_examples(
    df: pd.DataFrame,
    tokenizer,
    max_length: int = 1024,
    few_shot_pool: Optional[pd.DataFrame] = None,
    num_few_shot: int = 0,
    drop_zero_supervision: bool = True,
    min_supervised_tokens: int = 1,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Build list of {input_ids, attention_mask, labels} for SFT (labels = -100 on user part).
    Also returns stats, including how many rows were skipped due to low/no supervised tokens.
    """
    model_name = "ibm-granite/granite-3.2-2b-instruct"
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    few_shot_list = None
    if num_few_shot > 0 and few_shot_pool is not None and len(few_shot_pool) >= num_few_shot:
        few_shot_list = few_shot_pool.sample(n=num_few_shot).to_dict("records")
        few_shot_list = [
            {
                "question": str(r.get("interview_question", "")),
                "answer": str(r.get("interview_answer", "")),
                "output": build_assistant_output(r),
            }
            for r in few_shot_list
        ]

    examples = []
    stats: Dict[str, Any] = {
        "input_rows": int(len(df)),
        "kept_rows": 0,
        "dropped_missing_qa": 0,
        "dropped_low_supervision": 0,
        "rows_truncated_to_maxlen": 0,
        "mean_supervised_tokens": 0.0,
        "mean_prompt_len_tokens": 0.0,
        "mean_full_len_tokens": 0.0,
        "kept_label_distribution": {},
        "dropped_low_supervision_by_label": {},
    }
    supervised_tokens_all: List[int] = []
    prompt_lens: List[int] = []
    full_lens: List[int] = []
    kept_labels: List[str] = []
    dropped_labels: List[str] = []

    for _, row in df.iterrows():
        question = str(row.get("interview_question", ""))
        answer = str(row.get("interview_answer", ""))
        if not question or not answer:
            stats["dropped_missing_qa"] += 1
            continue
        user_prompt = build_user_prompt(question, answer, few_shot_list)
        assistant_output = build_assistant_output(row)
        mapped_label = row_to_clarity_label(row)

        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ]
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            formatted = user_prompt + "\n\n" + assistant_output

        tokenized = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        full_len = int(sum(attention_mask))

        # Find where assistant starts: after the last user turn in the template
        # Granite template typically has special tokens; we mask loss on user part
        try:
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = tokenizer(prompt_only, return_tensors=None, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)
        except Exception:
            prompt_len = len(tokenizer(user_prompt, return_tensors=None, add_special_tokens=False)["input_ids"])

        supervised_tokens = max(0, min(full_len, max_length) - min(prompt_len, max_length))
        if full_len >= max_length:
            stats["rows_truncated_to_maxlen"] += 1
        if drop_zero_supervision and supervised_tokens < max(0, min_supervised_tokens):
            stats["dropped_low_supervision"] += 1
            dropped_labels.append(mapped_label)
            continue

        labels = [-100] * len(input_ids)
        for i in range(prompt_len, min(len(input_ids), max_length)):
            labels[i] = input_ids[i]

        examples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
        supervised_tokens_all.append(int(supervised_tokens))
        prompt_lens.append(int(prompt_len))
        full_lens.append(int(full_len))
        kept_labels.append(mapped_label)

    stats["kept_rows"] = int(len(examples))
    if supervised_tokens_all:
        stats["mean_supervised_tokens"] = float(sum(supervised_tokens_all) / len(supervised_tokens_all))
        stats["mean_prompt_len_tokens"] = float(sum(prompt_lens) / len(prompt_lens))
        stats["mean_full_len_tokens"] = float(sum(full_lens) / len(full_lens))
    stats["kept_label_distribution"] = dict(Counter(kept_labels))
    stats["dropped_low_supervision_by_label"] = dict(Counter(dropped_labels))

    return examples, stats


# Optional TPU (PyTorch/XLA); used when running on Colab TPU or Cloud TPU
_TPU_AVAILABLE = False
_xla_device = None
try:
    import torch_xla.core.xla_model as xm
    # Colab sets COLAB_TPU_ADDR when TPU runtime is selected; else check world size
    if os.environ.get("COLAB_TPU_ADDR"):
        _xla_device = xm.xla_device()
        _TPU_AVAILABLE = True
    elif getattr(xm, "xrt_world_size", None) is not None and xm.xrt_world_size() >= 1:
        _xla_device = xm.xla_device()
        _TPU_AVAILABLE = True
except Exception:
    pass


def get_device():
    """Return the best available device: TPU > CUDA > MPS > CPU."""
    if _TPU_AVAILABLE and _xla_device is not None:
        return _xla_device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_tpu():
    return _TPU_AVAILABLE


def is_cuda():
    return torch.cuda.is_available() and not _TPU_AVAILABLE


def evaluate_with_voting(
    model,
    tokenizer,
    eval_examples: List[Dict],
    num_samples: int = 3,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    few_shot_examples: Optional[List[Dict]] = None,
) -> Tuple[float, float, List[str], List[str]]:
    """Run self-consistency (multiple samples, majority vote) on eval_examples. Returns accuracy, macro_f1, preds, gold.
    few_shot_examples: optional list of {"question", "answer", "output"} for prompt prepending at eval time."""
    from collections import Counter
    import re

    device = next(model.parameters()).device
    all_preds = []
    all_gold = []
    n_total = len(eval_examples)

    for idx, ex in enumerate(eval_examples):
        if (idx + 1) % 10 == 0 or idx == 0 or idx == n_total - 1:
            print(f"  Eval {idx + 1}/{n_total}...", flush=True)
        question = ex["question"]
        answer = ex["answer"]
        gold_label = ex["clarity_label"]
        if isinstance(gold_label, str) and gold_label not in ("Direct Reply", "Direct Non-Reply", "Indirect"):
            gold_label = CLARITY_LABEL_MAP.get(gold_label, "Indirect")
        all_gold.append(gold_label)

        user_prompt = build_user_prompt(question, answer, few_shot_examples=few_shot_examples)
        messages = [{"role": "user", "content": user_prompt}]
        votes = []

        for _ in range(num_samples):
            try:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(formatted, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                # Parse label from JSON
                m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
                if m:
                    label = m.group(1)
                    if label not in ("Direct Reply", "Direct Non-Reply", "Indirect"):
                        label = CLARITY_LABEL_MAP.get(label, "Indirect")
                    votes.append(label)
                else:
                    votes.append("Indirect")
            except Exception:
                votes.append("Indirect")

        majority = Counter(votes).most_common(1)[0][0]
        all_preds.append(majority)

    acc = accuracy_score(all_gold, all_preds)
    macro_f1 = f1_score(all_gold, all_preds, average="macro", labels=["Direct Reply", "Direct Non-Reply", "Indirect"])
    return acc, macro_f1, all_preds, all_gold


class EvalVotingCallback(TrainerCallback):
    """Run evaluation with 3-sample voting at save_steps and log accuracy/F1."""

    def __init__(self, eval_examples: List[Dict], tokenizer, num_samples: int = 3):
        self.eval_examples = eval_examples
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and args.save_steps > 0 and state.global_step % args.save_steps == 0:
            if model is None:
                return
            acc, macro_f1, _, _ = evaluate_with_voting(
                model, self.tokenizer, self.eval_examples, num_samples=self.num_samples
            )
            print(f"[Eval @ step {state.global_step}] accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")


def build_few_shot_pool_from_rationale_df(df: pd.DataFrame, max_pool: int = 50) -> List[Tuple[Dict[str, Any], int]]:
    """Build list of (few_shot_example_dict, original_row_index) for analysis. Each dict has question, answer, output."""
    pool = []
    for idx, row in df.head(max_pool).iterrows():
        q = str(row.get("interview_question", ""))
        a = str(row.get("interview_answer", ""))
        if not q or not a:
            continue
        pool.append(({"question": q, "answer": a, "output": build_assistant_output(row)}, int(idx)))
    return pool


def analyze_which_few_shot_pairs_help_most(
    model,
    tokenizer,
    eval_examples: List[Dict],
    few_shot_pool_with_indices: List[Tuple[Dict, int]],
    num_trials: int = 15,
    k_shot: int = 2,
    num_vote_samples: int = 3,
    top_n_runs: int = 5,
    output_dir: Optional[Path] = None,
    save_csv: bool = True,
) -> None:
    """
    Run eval multiple times with different random K-shot subsets; report which pool indices
    appeared most often in the best-performing runs (so we know which few-shot pairs were most right).
    Print the best few-shot examples and save them to best_few_shot_examples.csv when save_csv is True.
    """
    from collections import Counter
    import random

    if len(few_shot_pool_with_indices) < k_shot:
        print("Few-shot pool smaller than k_shot; skipping analysis.")
        return

    results = []  # (accuracy, list of indices used)
    pool_list = [x[0] for x in few_shot_pool_with_indices]
    index_list = [x[1] for x in few_shot_pool_with_indices]
    index_to_example = {idx: d for d, idx in few_shot_pool_with_indices}

    for trial in range(num_trials):
        indices = random.sample(range(len(pool_list)), k_shot)
        few_shot = [pool_list[i] for i in indices]
        indices_used = [index_list[i] for i in indices]
        acc, macro_f1, _, _ = evaluate_with_voting(
            model, tokenizer, eval_examples,
            num_samples=num_vote_samples,
            few_shot_examples=few_shot,
        )
        results.append((acc, indices_used))

    results.sort(key=lambda x: -x[0])
    print(f"\n--- Few-shot analysis ({num_trials} trials, K={k_shot}-shot) ---")
    print(f"Top-{top_n_runs} runs by accuracy:")
    for i, (acc, inds) in enumerate(results[:top_n_runs]):
        print(f"  {i+1}. accuracy={acc:.4f}  pool_indices={inds}")

    # Count how often each pool index appeared in top-N runs
    top_indices_flat = []
    for acc, inds in results[:top_n_runs]:
        top_indices_flat.extend(inds)
    counts = Counter(top_indices_flat)
    print(f"\nFew-shot pool indices that appeared most often in top-{top_n_runs} runs (most helpful):")
    for idx, cnt in counts.most_common(10):
        print(f"  index {idx}: {cnt} times")

    # Build records for best few-shot examples (top 10 by count)
    records = []
    for idx, cnt in counts.most_common(10):
        ex = index_to_example.get(idx)
        if ex is None:
            continue
        records.append({
            "pool_index": idx,
            "count_in_top_runs": cnt,
            "question": ex.get("question", ""),
            "answer": ex.get("answer", ""),
            "output": ex.get("output", ""),
        })

    # Print best few-shot examples (truncate long text)
    if records:
        print("\nBest few-shot examples (content):")
        truncate = 120
        for i, rec in enumerate(records, 1):
            q = (rec["question"] or "")[:truncate] + ("..." if len(rec["question"] or "") > truncate else "")
            a = (rec["answer"] or "")[:truncate] + ("..." if len(rec["answer"] or "") > truncate else "")
            out = (rec["output"] or "")[:truncate] + ("..." if len(rec["output"] or "") > truncate else "")
            print(f"  Best #{i} (index {rec['pool_index']}, appeared {rec['count_in_top_runs']} times):")
            print(f"    Question: {q}")
            print(f"    Answer: {a}")
            print(f"    Output: {out}")

    # Save to CSV
    if save_csv and records:
        csv_path = (Path(output_dir) / "best_few_shot_examples.csv") if output_dir else Path("best_few_shot_examples.csv")
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"\nSaved best few-shot examples to: {csv_path}")


def load_eval_data_from_qevasion(split: str = "test", max_per_label: Optional[int] = 30) -> List[Dict]:
    """Load balanced eval from QEvasion (same as balanced_loader)."""
    from collections import Counter
    ds = load_dataset("ailsntua/QEvasion", split=split)
    examples = []
    for i in range(len(ds)):
        item = ds[i]
        clarity = item.get("clarity_label")
        mapped = CLARITY_LABEL_MAP.get(clarity, clarity)
        if mapped not in ("Direct Reply", "Direct Non-Reply", "Indirect"):
            continue
        examples.append({
            "question": str(item.get("interview_question", item.get("question", ""))),
            "answer": str(item.get("interview_answer", "")),
            "clarity_label": mapped,
        })
    if max_per_label is not None:
        by_label = {}
        for ex in examples:
            L = ex["clarity_label"]
            by_label.setdefault(L, []).append(ex)
        n = min(len(v) for v in by_label.values()) if by_label else 0
        n = min(n, max_per_label)
        examples = []
        for L, lst in by_label.items():
            examples.extend(random.sample(lst, min(n, len(lst))))
        random.shuffle(examples)
    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rationale-csv", type=str, default=str(RATIONALE_CSV))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument(
        "--length-profile",
        choices=["custom", "1024", "1536", "2048"],
        default="custom",
        help="Preset for max_length. If not custom, overrides --max-length.",
    )
    parser.add_argument(
        "--drop-zero-supervision",
        action="store_true",
        default=True,
        help="Drop rows where prompt consumes full context and no assistant tokens are supervised (default: True)",
    )
    parser.add_argument(
        "--keep-zero-supervision",
        action="store_false",
        dest="drop_zero_supervision",
        help="Keep rows even when supervised token count is zero",
    )
    parser.add_argument(
        "--min-supervised-tokens",
        type=int,
        default=1,
        help="Minimum supervised assistant tokens required to keep a row (used when --drop-zero-supervision is enabled)",
    )
    parser.add_argument(
        "--drop-conflicting-qa",
        action="store_true",
        default=False,
        help="Drop rows where identical question+answer text appears with conflicting clarity labels",
    )
    parser.add_argument(
        "--balance-rationale",
        choices=["none", "downsample", "upsample"],
        default="none",
        help="Optional class balancing on mapped CLARITY labels before tokenization",
    )
    parser.add_argument("--few-shot", type=int, default=0, help="Number of few-shot examples in prompt (0 = no few-shot)")
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--compare-few-shot-eval", action="store_true", help="After training, run eval with 0/1/2/3-shot and analyze which few-shot pairs help most")
    parser.add_argument("--few-shot-analysis-trials", type=int, default=12, help="Number of random K-shot trials for 'which few-shot pairs' analysis")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True, help="Enable gradient checkpointing to save memory (default: True)")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing", help="Disable gradient checkpointing")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2, help="Gradient accumulation steps for memory-efficient training")
    parser.add_argument("--load-8bit", action="store_true", help="Load model in 8-bit (bitsandbytes) to save memory")
    parser.add_argument("--eval", action="store_true", help="Run pre- and post-training evaluation (default: fast, 12 examples 1 sample)")
    parser.add_argument("--full-eval", action="store_true", help="Use full eval (60 examples, 3-sample voting); only with --eval")
    args = parser.parse_args()

    if args.length_profile != "custom":
        args.max_length = int(args.length_profile)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(
        f"Config: seed={args.seed}, max_length={args.max_length}, "
        f"drop_zero_supervision={args.drop_zero_supervision}, "
        f"min_supervised_tokens={args.min_supervised_tokens}, "
        f"drop_conflicting_qa={args.drop_conflicting_qa}, "
        f"balance_rationale={args.balance_rationale}",
        flush=True,
    )

    # Clear caches to free memory before loading model (reduces chance of OOM kill)
    gc.collect()
    if not is_tpu():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    print("Cleared GPU/MPS/CPU caches.", flush=True)
    if is_tpu():
        print("Using TPU (PyTorch/XLA). 8-bit and device_map disabled.", flush=True)

    # Prefer longest rationale CSV in folder over default when user did not pass --rationale-csv
    if args.rationale_csv == str(RATIONALE_CSV):
        resolved = resolve_rationale_csv(RATIONALE_FOLDER, Path(RATIONALE_CSV))
        args.rationale_csv = str(resolved)
        if resolved != RATIONALE_CSV:
            print(f"Using longer rationale CSV: {resolved}")

    model_name = "ibm-granite/granite-3.2-2b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = get_device()

    if is_tpu():
        # TPU: no device_map (multi-GPU/CPU offload), no 8-bit (bitsandbytes is CUDA-only)
        load_kwargs = {"torch_dtype": torch.bfloat16}
        use_8bit = False
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        model = model.to(device)
    else:
        load_kwargs = {
            "torch_dtype": "auto",
            "device_map": "auto",
        }
        use_8bit = getattr(args, "load_8bit", False)
        # 8-bit on Mac (MPS/CPU) causes bitsandbytes shape/index errors; use 8-bit only on CUDA
        if use_8bit and not torch.cuda.is_available():
            print("--load-8bit skipped on non-CUDA (Mac/CPU); using full precision + LoRA to avoid bitsandbytes errors.", flush=True)
            use_8bit = False
        if use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                load_kwargs["device_map"] = "auto"
            except Exception:
                use_8bit = False
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    # Apply LoRA when user asked for --load-8bit (saves memory; on TPU we still use LoRA for efficiency)
    use_lora = getattr(args, "load_8bit", False) or is_tpu()
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except Exception as e:
            raise RuntimeError(
                "PEFT/LoRA required for --load-8bit (pip install peft). "
                "Install it or run without --load-8bit for full fine-tuning."
            ) from e
    # 8-bit + gradient checkpointing triggers bitsandbytes errors on Mac/CPU; disable GC when using 8-bit
    enable_gradient_checkpointing = getattr(args, "gradient_checkpointing", True) and not use_8bit
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    elif use_8bit:
        print("Gradient checkpointing disabled with 8-bit to avoid bitsandbytes errors on this device.", flush=True)

    # Load rationale CSV
    df = load_rationale_csv(Path(args.rationale_csv))
    preprocessing_summary: Dict[str, Any] = {
        "source_csv": str(args.rationale_csv),
        "rows_after_base_filters": int(len(df)),
    }
    print(f"Loaded {len(df)} rationale rows with verdict_match=True")
    print(f"Mapped-label distribution (raw filtered): {dict(Counter(df.apply(row_to_clarity_label, axis=1).tolist()))}")
    preprocessing_summary["mapped_distribution_after_base_filters"] = dict(
        Counter(df.apply(row_to_clarity_label, axis=1).tolist())
    )

    if args.drop_conflicting_qa:
        df, dropped_rows, conflict_pairs = drop_conflicting_qa_rows(df)
        print(
            f"Dropped conflicting QA rows: {dropped_rows} "
            f"across {conflict_pairs} conflicting QA pairs. Remaining={len(df)}"
        )
        preprocessing_summary["dropped_conflicting_rows"] = int(dropped_rows)
        preprocessing_summary["conflicting_pair_count"] = int(conflict_pairs)
        preprocessing_summary["rows_after_conflict_drop"] = int(len(df))

    if args.balance_rationale != "none":
        before_dist = dict(Counter(df.apply(row_to_clarity_label, axis=1).tolist()))
        df = balance_rationale_df(df, mode=args.balance_rationale, seed=args.seed)
        after_dist = dict(Counter(df.apply(row_to_clarity_label, axis=1).tolist()))
        print(f"Applied balance mode={args.balance_rationale}.")
        print(f"  Before: {before_dist}")
        print(f"  After:  {after_dist}")
        preprocessing_summary["balance_mode"] = args.balance_rationale
        preprocessing_summary["mapped_distribution_before_balance"] = before_dist
        preprocessing_summary["mapped_distribution_after_balance"] = after_dist

    if args.max_train:
        df = df.sample(n=min(args.max_train, len(df)), random_state=args.seed)
    preprocessing_summary["rows_after_max_train"] = int(len(df))
    # For few-shot, use a separate pool (exclude from train if small dataset)
    few_shot_pool = df.sample(n=min(len(df), max(20, args.few_shot * 4)), random_state=args.seed + 1) if args.few_shot else None
    train_examples, train_stats = build_training_examples(
        df,
        tokenizer,
        max_length=args.max_length,
        few_shot_pool=few_shot_pool,
        num_few_shot=args.few_shot,
        drop_zero_supervision=args.drop_zero_supervision,
        min_supervised_tokens=args.min_supervised_tokens,
    )
    print("Training example build stats:")
    print(
        f"  input_rows={train_stats['input_rows']}, kept_rows={train_stats['kept_rows']}, "
        f"dropped_missing_qa={train_stats['dropped_missing_qa']}, "
        f"dropped_low_supervision={train_stats['dropped_low_supervision']}, "
        f"truncated_rows={train_stats['rows_truncated_to_maxlen']}"
    )
    print(
        f"  mean_supervised_tokens={train_stats['mean_supervised_tokens']:.2f}, "
        f"mean_prompt_len={train_stats['mean_prompt_len_tokens']:.2f}, "
        f"mean_full_len={train_stats['mean_full_len_tokens']:.2f}"
    )
    print(f"  kept_label_distribution={train_stats['kept_label_distribution']}")
    if train_stats["dropped_low_supervision_by_label"]:
        print(f"  dropped_low_supervision_by_label={train_stats['dropped_low_supervision_by_label']}")

    if len(train_examples) == 0:
        raise RuntimeError(
            "No training examples left after filtering/tokenization. "
            "Try increasing --max-length or using --keep-zero-supervision."
        )

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "training_data_stats.json").write_text(
        json.dumps(
            {
                "config": {
                    "seed": args.seed,
                    "max_length": args.max_length,
                    "drop_zero_supervision": args.drop_zero_supervision,
                    "min_supervised_tokens": args.min_supervised_tokens,
                    "drop_conflicting_qa": args.drop_conflicting_qa,
                    "balance_rationale": args.balance_rationale,
                    "few_shot": args.few_shot,
                },
                "preprocessing_summary": preprocessing_summary,
                "tokenization_summary": train_stats,
            },
            indent=2,
        )
    )
    print(f"Saved training data stats to: {out_path / 'training_data_stats.json'}")

    train_dataset = Dataset.from_list(train_examples)

    # Eval set from QEvasion
    eval_examples = load_eval_data_from_qevasion(split=EVAL_SPLIT_NAME, max_per_label=args.max_eval // 3)
    print(f"Eval examples: {len(eval_examples)}")

    # Custom collator: pad batch and preserve our labels (loss only on assistant tokens)
    def _collate_fn(examples):
        batch = {
            "input_ids": torch.tensor([e["input_ids"] for e in examples], dtype=torch.long),
            "attention_mask": torch.tensor([e["attention_mask"] for e in examples], dtype=torch.long),
            "labels": torch.tensor([e["labels"] for e in examples], dtype=torch.long),
        }
        return batch

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 2),
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="no",
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=is_cuda(),
        bf16=is_tpu(),
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    do_eval = getattr(args, "eval", False) or getattr(args, "full_eval", False)
    fast_eval = do_eval and not getattr(args, "full_eval", False)
    eval_examples_to_use = eval_examples[:12] if fast_eval else eval_examples
    eval_num_samples = 1 if fast_eval else 3
    if do_eval and fast_eval:
        print(f"Eval: fast mode (12 examples, 1 sample)", flush=True)

    callbacks = []
    if do_eval:
        callbacks.append(EvalVotingCallback(eval_examples_to_use, tokenizer, num_samples=eval_num_samples))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_collate_fn,
        callbacks=callbacks,
    )

    if do_eval:
        print("Pre-training evaluation...", flush=True)
        acc_before, f1_before, _, _ = evaluate_with_voting(model, tokenizer, eval_examples_to_use, num_samples=eval_num_samples)
        print(f"Before training: accuracy={acc_before:.4f}, macro_f1={f1_before:.4f}")

    trainer.train()

    if do_eval:
        print("Post-training evaluation...", flush=True)
        acc_after, f1_after, preds, gold = evaluate_with_voting(model, tokenizer, eval_examples_to_use, num_samples=eval_num_samples)
        print(f"After training: accuracy={acc_after:.4f}, macro_f1={f1_after:.4f}")

    try:
        trainer.save_model(args.output_dir, safe_serialization=True)
    except TypeError:
        trainer.save_model(args.output_dir)
    try:
        tokenizer.save_pretrained(args.output_dir, safe_serialization=True)
    except TypeError:
        tokenizer.save_pretrained(args.output_dir)

    # Write a short summary so notebook/CI can read it even if subprocess stdout was lost
    summary = {
        "output_dir": str(args.output_dir),
        "checkpoints": "Last 2 checkpoints kept in output_dir (checkpoint-XXX); final model in output_dir.",
        "training_data_stats": str(Path(args.output_dir) / "training_data_stats.json"),
    }
    if do_eval:
        summary["accuracy_before"] = float(acc_before)
        summary["accuracy_after"] = float(acc_after)
        summary["macro_f1_before"] = float(f1_before)
        summary["macro_f1_after"] = float(f1_after)
    (Path(args.output_dir) / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Saved training_summary.json to {args.output_dir}", flush=True)

    if getattr(args, "compare_few_shot_eval", False):
        # Compare 0-shot vs 1/2/3-shot at inference (same eval set)
        print("\n--- Comparing 0-shot vs few-shot at inference ---")
        for k in (0, 1, 2, 3):
            if k == 0:
                acc, f1, _, _ = evaluate_with_voting(model, tokenizer, eval_examples, num_samples=3, few_shot_examples=None)
            else:
                df_full = load_rationale_csv(Path(args.rationale_csv))
                pool_with_idx = build_few_shot_pool_from_rationale_df(df_full, max_pool=30)
                if len(pool_with_idx) < k:
                    acc, f1 = 0.0, 0.0
                else:
                    few_shot = [pool_with_idx[i][0] for i in random.sample(range(len(pool_with_idx)), k)]
                    acc, f1, _, _ = evaluate_with_voting(model, tokenizer, eval_examples, num_samples=3, few_shot_examples=few_shot)
            print(f"  {k}-shot: accuracy={acc:.4f}, macro_f1={f1:.4f}")

        # Which few-shot pairs were most right: run many trials with random K-shot, see which pool indices appear in top runs
        df_full = load_rationale_csv(Path(args.rationale_csv))
        pool_with_idx = build_few_shot_pool_from_rationale_df(df_full, max_pool=40)
        if len(pool_with_idx) >= 2:
            analyze_which_few_shot_pairs_help_most(
                model, tokenizer, eval_examples,
                few_shot_pool_with_indices=pool_with_idx,
                num_trials=args.few_shot_analysis_trials,
                k_shot=2,
                num_vote_samples=3,
                top_n_runs=5,
                output_dir=Path(args.output_dir),
                save_csv=True,
            )
        else:
            print("Few-shot pool too small for 'which pairs help most' analysis.")


if __name__ == "__main__":
    main()
