#!/usr/bin/env python3
"""
Prepare a cleaner, balanced rationale dataset for Granite training.

Pipeline:
1) Load rationale CSV
2) Apply training filters (verdict_match=True, non-empty reasoning)
3) Optional: drop QA pairs with conflicting labels
4) Optional: drop rows with zero supervised tokens at given max_length
5) Downsample to balanced class counts
6) Save CSV + JSONL + summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


DEFAULT_INPUT = Path("/Users/andrearachetta/Desktop/qevasion_rationale/qevasion_rationale_dataset_20260204_163024.csv")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports_acl_submission" / "balanced_rationale_dataset"
DEFAULT_TOKENIZER = REPO_ROOT / "granite_clarity_finetuned"

MAP_TO_CLARITY = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}


def map_label(raw: object) -> str:
    v = "" if pd.isna(raw) else str(raw).strip()
    if v in MAP_TO_CLARITY:
        return MAP_TO_CLARITY[v]
    if v in {"Direct Reply", "Direct Non-Reply", "Indirect"}:
        return v
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


def add_supervision_columns(
    df: pd.DataFrame,
    tokenizer_path: str,
    max_length: int,
) -> pd.DataFrame:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    sanity = tok("sanity", return_tensors="pt")["input_ids"]
    if int(sanity.shape[-1]) == 0:
        raise RuntimeError(f"Tokenizer from {tokenizer_path} tokenizes to empty inputs.")

    full_lens = []
    prompt_lens = []
    sup_tokens = []
    truncated = []
    zero_sup = []

    for _, row in df.iterrows():
        q = str(row["interview_question"])
        a = str(row["interview_answer"])
        reasoning = str(row["initial_reasoning"]).strip()
        label = str(row["mapped_label"])

        prompt = build_prompt(q, a)
        assistant = json.dumps({"reasoning": reasoning, "label": label}, ensure_ascii=False)
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ]

        try:
            formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_only = tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = prompt + "\n\n" + assistant
            prompt_only = prompt

        full = tok(formatted, truncation=True, max_length=max_length, padding="max_length", return_tensors=None)
        p_ids = tok(prompt_only, return_tensors=None, add_special_tokens=False)["input_ids"]

        f_len = int(sum(full["attention_mask"]))
        p_len = int(len(p_ids))
        sup = max(0, min(f_len, max_length) - min(p_len, max_length))

        full_lens.append(f_len)
        prompt_lens.append(p_len)
        sup_tokens.append(int(sup))
        truncated.append(bool(f_len >= max_length))
        zero_sup.append(bool(sup == 0))

    out = df.copy()
    out["full_len_tokens"] = full_lens
    out["prompt_len_tokens"] = prompt_lens
    out["supervised_tokens"] = sup_tokens
    out["truncated_to_maxlen"] = truncated
    out["zero_supervised"] = zero_sup
    return out


def to_jsonl_records(df: pd.DataFrame) -> list[Dict]:
    records = []
    for _, row in df.iterrows():
        q = str(row["interview_question"])
        a = str(row["interview_answer"])
        reasoning = str(row["initial_reasoning"]).strip()
        label = str(row["mapped_label"])
        output = json.dumps({"reasoning": reasoning, "label": label}, ensure_ascii=False)
        records.append(
            {
                "instruction": "Analyze the interview answer and classify clarity.",
                "input": f"Question: {q}\nAnswer: {a}",
                "output": output,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare balanced rationale dataset for Granite.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--tokenizer-path", type=str, default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--drop-conflicting-qa", action="store_true", default=True)
    parser.add_argument("--keep-conflicting-qa", action="store_false", dest="drop_conflicting_qa")
    parser.add_argument("--drop-zero-supervised", action="store_true", default=True)
    parser.add_argument("--keep-zero-supervised", action="store_false", dest="drop_zero_supervised")
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input_csv)
    raw["verdict_match"] = raw["verdict_match"].astype(str).str.lower().eq("true")

    # Same filtering behavior as training script.
    df = raw[raw["verdict_match"]].copy()
    df = df[df["initial_reasoning"].notna() & (df["initial_reasoning"].astype(str).str.len() > 0)]
    df["final_verdict"] = df["final_verdict"].fillna(df["initial_verdict"]).fillna(df["clarity_label"])
    df["mapped_label"] = df["final_verdict"].map(map_label)

    before = len(df)
    dropped_conflict = 0
    if args.drop_conflicting_qa:
        key = df["interview_question"].astype(str).str.strip() + " ||| " + df["interview_answer"].astype(str).str.strip()
        per_key_nlabels = df.assign(_k=key).groupby("_k")["clarity_label"].nunique()
        bad = set(per_key_nlabels[per_key_nlabels > 1].index.tolist())
        mask_bad = key.isin(bad)
        dropped_conflict = int(mask_bad.sum())
        df = df[~mask_bad].copy()

    # Supervision audit and optional zero-supervision removal.
    df = add_supervision_columns(df, tokenizer_path=args.tokenizer_path, max_length=args.max_length)
    dropped_zero = 0
    if args.drop_zero_supervised:
        dropped_zero = int(df["zero_supervised"].sum())
        df = df[~df["zero_supervised"]].copy()

    # Balanced downsample.
    label_counts = df["mapped_label"].value_counts().to_dict()
    min_count = min(label_counts.values()) if label_counts else 0
    balanced = (
        df.groupby("mapped_label", group_keys=False)
        .sample(n=min_count, random_state=args.seed)
        .reset_index(drop=True)
    ) if min_count > 0 else df.iloc[0:0].copy()

    # Save outputs.
    csv_out = out_dir / "rationale_balanced.csv"
    balanced.to_csv(csv_out, index=False)

    jsonl_out = out_dir / "RationaleTraining_balanced.jsonl"
    records = to_jsonl_records(balanced)
    with jsonl_out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    token_audit_out = out_dir / "token_audit_before_balancing.csv"
    df.to_csv(token_audit_out, index=False)

    summary = {
        "input_csv": str(args.input_csv),
        "output_csv": str(csv_out),
        "output_jsonl": str(jsonl_out),
        "settings": {
            "drop_conflicting_qa": args.drop_conflicting_qa,
            "drop_zero_supervised": args.drop_zero_supervised,
            "max_length": args.max_length,
            "tokenizer_path": args.tokenizer_path,
            "seed": args.seed,
        },
        "counts": {
            "after_training_filters": before,
            "dropped_conflicting_rows": dropped_conflict,
            "dropped_zero_supervised_rows": dropped_zero,
            "before_balancing_after_cleaning": int(len(df)),
            "balanced_per_label": int(min_count),
            "balanced_total_rows": int(len(balanced)),
        },
        "distribution_before_balancing": label_counts,
        "distribution_after_balancing": balanced["mapped_label"].value_counts().to_dict(),
        "supervision_before_balancing": {
            "truncated_rows": int(df["truncated_to_maxlen"].sum()),
            "zero_supervised_rows": int(df["zero_supervised"].sum()),
            "mean_supervised_tokens_by_label": df.groupby("mapped_label")["supervised_tokens"].mean().round(2).to_dict(),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Saved: {csv_out}")
    print(f"Saved: {jsonl_out}")
    print(f"Saved: {out_dir / 'summary.json'}")
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
