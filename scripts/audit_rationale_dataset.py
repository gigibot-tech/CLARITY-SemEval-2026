#!/usr/bin/env python3
"""
Audit the Granite rationale-training CSV and export quality diagnostics.

Default output directory:
  reports_acl_submission/rationale_dataset_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


DEFAULT_INPUT = Path("/Users/andrearachetta/Desktop/qevasion_rationale/qevasion_rationale_dataset_20260204_163024.csv")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "reports_acl_submission" / "rationale_dataset_audit"
DEFAULT_TOKENIZER = REPO_ROOT / "granite_clarity_finetuned"

LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}


def map_label(raw: object) -> str:
    value = "" if pd.isna(raw) else str(raw).strip()
    if value in LABEL_MAP:
        return LABEL_MAP[value]
    if value in {"Direct Reply", "Direct Non-Reply", "Indirect"}:
        return value
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


def add_token_supervision_audit(
    train_df: pd.DataFrame,
    tokenizer_path: Path | str,
    max_length: int,
) -> pd.DataFrame:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    sanity = tokenizer("sanity", return_tensors="pt")["input_ids"]
    if int(sanity.shape[-1]) == 0:
        raise RuntimeError(f"Tokenizer from {tokenizer_path} tokenizes to empty inputs.")

    rows = []
    for _, row in train_df.iterrows():
        question = str(row["interview_question"])
        answer = str(row["interview_answer"])
        reasoning = str(row["initial_reasoning"]).strip()
        label = str(row["train_label"])
        prompt = build_prompt(question, answer)
        assistant = json.dumps({"reasoning": reasoning, "label": label}, ensure_ascii=False)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = prompt + "\n\n" + assistant
            prompt_only = prompt

        full_ids = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        prompt_ids = tokenizer(prompt_only, return_tensors=None, add_special_tokens=False)["input_ids"]

        full_len = int(sum(full_ids["attention_mask"]))
        prompt_len = int(len(prompt_ids))
        supervised = max(0, min(full_len, max_length) - min(prompt_len, max_length))

        rows.append(
            {
                "index": row.get("index"),
                "clarity_label": row.get("clarity_label"),
                "initial_verdict": row.get("initial_verdict"),
                "final_verdict": row.get("final_verdict"),
                "train_label": row.get("train_label"),
                "full_len_tokens": full_len,
                "prompt_len_tokens": prompt_len,
                "supervised_tokens": int(supervised),
                "zero_supervised": bool(supervised == 0),
                "truncated_to_maxlen": bool(full_len >= max_length),
            }
        )
    return pd.DataFrame(rows)


def build_summary(raw_df: pd.DataFrame, train_df: pd.DataFrame, token_df: pd.DataFrame) -> Dict:
    raw_false = int((~raw_df["verdict_match"]).sum())
    initial_mismatch = int(
        (
            train_df["initial_verdict"].astype(str).str.strip()
            != train_df["clarity_label"].astype(str).str.strip()
        ).sum()
    )
    final_mismatch = int(
        (
            train_df["final_verdict"].astype(str).str.strip()
            != train_df["clarity_label"].astype(str).str.strip()
        ).sum()
    )
    qa_key = (
        train_df["interview_question"].astype(str).str.strip()
        + " ||| "
        + train_df["interview_answer"].astype(str).str.strip()
    )
    conflicting_same_qa = int(
        (train_df.assign(_k=qa_key).groupby("_k")["clarity_label"].nunique() > 1).sum()
    )

    return {
        "raw_rows": int(len(raw_df)),
        "train_rows_after_filters": int(len(train_df)),
        "verdict_match_false_rows_dropped": raw_false,
        "label_distribution_raw_clarity": raw_df["clarity_label"].value_counts(dropna=False).to_dict(),
        "label_distribution_train_clarity": train_df["clarity_label"].value_counts(dropna=False).to_dict(),
        "label_distribution_train_mapped": train_df["train_label"].value_counts(dropna=False).to_dict(),
        "initial_vs_clarity_mismatch_rows": initial_mismatch,
        "final_vs_clarity_mismatch_rows": final_mismatch,
        "conflicting_same_qa_pairs": conflicting_same_qa,
        "token_audit": {
            "rows": int(len(token_df)),
            "rows_truncated_to_maxlen": int(token_df["truncated_to_maxlen"].sum()),
            "rows_with_zero_supervised_tokens": int(token_df["zero_supervised"].sum()),
            "zero_supervised_by_mapped_label": token_df.groupby("train_label")["zero_supervised"].sum().to_dict(),
            "mean_supervised_tokens_by_mapped_label": token_df.groupby("train_label")[
                "supervised_tokens"
            ].mean().round(2).to_dict(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rationale dataset quality for Granite training.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-path", type=str, default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--max-length", type=int, default=1024)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv)
    raw_df["verdict_match"] = raw_df["verdict_match"].astype(str).str.lower().eq("true")

    train_df = raw_df[raw_df["verdict_match"]].copy()
    train_df = train_df[train_df["initial_reasoning"].notna() & (train_df["initial_reasoning"].astype(str).str.len() > 0)]
    train_df["final_verdict"] = train_df["final_verdict"].fillna(train_df["initial_verdict"]).fillna(train_df["clarity_label"])
    train_df["train_label"] = train_df["final_verdict"].map(map_label)

    qa_key = (
        train_df["interview_question"].astype(str).str.strip()
        + " ||| "
        + train_df["interview_answer"].astype(str).str.strip()
    )
    conflict_keys = (
        train_df.assign(_k=qa_key).groupby("_k")["clarity_label"].nunique()
    )
    conflict_keys = set(conflict_keys[conflict_keys > 1].index.tolist())

    conflicts = train_df.assign(_k=qa_key)
    conflicts = conflicts[conflicts["_k"].isin(conflict_keys)].copy()
    conflicts.sort_values(["_k", "index"], inplace=True)
    conflicts.to_csv(output_dir / "conflicting_labels_same_qa.csv", index=False)

    initial_mismatch_mask = (
        train_df["initial_verdict"].astype(str).str.strip()
        != train_df["clarity_label"].astype(str).str.strip()
    )
    train_df[initial_mismatch_mask].to_csv(output_dir / "initial_vs_clarity_mismatches.csv", index=False)
    raw_df[~raw_df["verdict_match"]].to_csv(output_dir / "dropped_verdict_match_false.csv", index=False)

    token_df = add_token_supervision_audit(
        train_df=train_df,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
    )
    token_df.to_csv(output_dir / "token_supervision_audit.csv", index=False)

    summary = build_summary(raw_df=raw_df, train_df=train_df, token_df=token_df)
    summary["source_csv"] = str(args.input_csv)
    summary["max_length"] = int(args.max_length)
    summary["tokenizer_path"] = str(args.tokenizer_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    readme = [
        "# Rationale Dataset Audit",
        "",
        f"- Source CSV: `{args.input_csv}`",
        f"- Output dir: `{output_dir}`",
        "",
        "## Re-run",
        f"`{Path('/Users/andrearachetta/Desktop/.venv/bin/python')} {Path(__file__).resolve()} --output-dir {output_dir}`",
        "",
        "## Core Counts",
        f"- Raw rows: **{summary['raw_rows']}**",
        f"- Train rows after filters: **{summary['train_rows_after_filters']}**",
        f"- Dropped (`verdict_match=False`): **{summary['verdict_match_false_rows_dropped']}**",
        "",
        "## Mapped Label Distribution",
    ]
    for label, count in train_df["train_label"].value_counts().to_dict().items():
        readme.append(f"- {label}: **{int(count)}**")
    readme.extend(
        [
            "",
            "## Quality Flags",
            f"- Same QA text with conflicting labels: **{summary['conflicting_same_qa_pairs']}**",
            f"- Initial-vs-clarity mismatches: **{summary['initial_vs_clarity_mismatch_rows']}**",
            f"- Final-vs-clarity mismatches: **{summary['final_vs_clarity_mismatch_rows']}**",
            "",
            "## Token Supervision (for current max length)",
            f"- Truncated rows: **{summary['token_audit']['rows_truncated_to_maxlen']} / {summary['token_audit']['rows']}**",
            f"- Zero-supervision rows: **{summary['token_audit']['rows_with_zero_supervised_tokens']} / {summary['token_audit']['rows']}**",
            "",
            "## Outputs",
            "- `summary.json`",
            "- `conflicting_labels_same_qa.csv`",
            "- `initial_vs_clarity_mismatches.csv`",
            "- `dropped_verdict_match_false.csv`",
            "- `token_supervision_audit.csv`",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(readme))

    print(f"Audit written to: {output_dir}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
