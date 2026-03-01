#!/usr/bin/env python3
"""
### ============================================================================
### Granite CLARITY Training from JSONL (Kaggle/Colab Ready)
### ============================================================================
### Directly trains from:
### - RationaleTraining_raw.jsonl
### - HardCases_corrections.jsonl (optional)
### ============================================================================
"""

### ============================================================================
### Cell 1: Imports & Runtime Detection
### ============================================================================

import gc
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

IN_KAGGLE = os.path.exists("/kaggle/working")
IN_COLAB = bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))
BASE_DIR = Path("/kaggle/working") if IN_KAGGLE else (Path("/content/drive/MyDrive/granite_clarity") if IN_COLAB else Path.cwd())

### ============================================================================
### Cell 2: Config
### ============================================================================

TRAINING_JSONL = BASE_DIR / "qevasion_rationale" / "train_cap250_seed42_granite_8b_local" / "RationaleTraining_raw.jsonl"
HARD_CASES_JSONL = BASE_DIR / "qevasion_rationale" / "train_cap250_seed42_granite_8b_local" / "HardCases_corrections.jsonl"
OUTPUT_DIR = BASE_DIR / "granite_2b_finetuned_jsonl"

MODEL_ID = "ibm-granite/granite-3.2-2b-instruct"
SEED = 42
EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACC = 8
LR = 2e-5
MAX_LENGTH = 1024
VALIDATION_SPLIT = 0.15
USE_8BIT = torch.cuda.is_available()
USE_LORA = True
DO_EVAL = True
EVAL_SAMPLES = 12
EVAL_VOTE_SAMPLES = 1
SAVE_EVAL_ERRORS = True

LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}


### ============================================================================
### Cell 3: Data & Prompt Helpers
### ============================================================================

def map_label(label: str) -> str:
    label = str(label).strip()
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    if label in ("Direct Reply", "Direct Non-Reply", "Indirect"):
        return label
    return "Indirect"


def parse_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qa = re.match(r"Q:\s*(.*?)\nA:\s*(.*?)(?:\n|$)", rec.get("input", ""), re.DOTALL)
    if not qa:
        return None
    think = re.search(r"<think>(.*?)</think>", rec.get("output", ""), re.DOTALL | re.IGNORECASE)
    verdict = re.search(r"Verdict:\s*([^\n]+)", rec.get("output", ""), re.IGNORECASE)
    if not verdict:
        return None
    return {
        "question": qa.group(1).strip(),
        "answer": qa.group(2).strip(),
        "reasoning": think.group(1).strip() if think else "",
        "label": map_label(rec.get("ground_truth", verdict.group(1).strip())),
        "is_hard_case": bool(rec.get("is_hard_case", False)),
    }


def load_rows(main_path: Path, hard_path: Optional[Path]) -> List[Dict[str, Any]]:
    if not main_path.exists():
        raise FileNotFoundError(f"Missing main JSONL: {main_path}")
    rows: List[Dict[str, Any]] = []
    for path, force_hard in [(main_path, False), (hard_path, True)]:
        if not path or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if force_hard:
                    rec["is_hard_case"] = True
                parsed = parse_record(rec)
                if parsed:
                    rows.append(parsed)
    return rows


def upsample(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    target = max(len(v) for v in by_label.values())
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for group in by_label.values():
        out.extend(group if len(group) >= target else rng.choices(group, k=target))
    rng.shuffle(out)
    return out


def user_prompt(q: str, a: str) -> str:
    return f"""You are analyzing political interview answers for clarity classification.

Question: {q}
Answer: {a}

Respond in JSON:
{{"reasoning":"...","label":"Direct Reply|Direct Non-Reply|Indirect"}}"""


def assistant_output(reasoning: str, label: str, is_hard: bool) -> str:
    if is_hard:
        reasoning = f"[Self-corrected analysis]\\n{reasoning}"
    return json.dumps({"reasoning": reasoning, "label": label}, ensure_ascii=False)


### ============================================================================
### Cell 4: Build Tokenized Dataset
### ============================================================================

def build_examples(rows: List[Dict[str, Any]], tokenizer, max_length: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        up = user_prompt(r["question"], r["answer"])
        ao = assistant_output(r["reasoning"], r["label"], r["is_hard_case"])
        msgs = [{"role": "user", "content": up}, {"role": "assistant", "content": ao}]
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = up + "\n\n" + ao
        tok = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors=None)
        try:
            up_text = tokenizer.apply_chat_template([{"role": "user", "content": up}], tokenize=False, add_generation_prompt=True)
            up_len = len(tokenizer(up_text, add_special_tokens=False)["input_ids"])
        except Exception:
            up_len = len(tokenizer(up, add_special_tokens=False)["input_ids"])
        labels = [-100] * len(tok["input_ids"])
        for i in range(up_len, len(tok["input_ids"])):
            labels[i] = tok["input_ids"][i]
        if sum(1 for x in labels if x != -100) < 5:
            continue
        out.append({"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"], "labels": labels})
    return out


### ============================================================================
### Cell 5: Model + Train/Eval
### ============================================================================

def load_model_tokenizer():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32, "device_map": "auto"}
    if USE_8BIT and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            pass
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    if USE_LORA:
        try:
            import importlib
            peft = importlib.import_module("peft")
            cfg = peft.LoraConfig(
                r=8, lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05, bias="none", task_type=peft.TaskType.CAUSAL_LM,
            )
            model = peft.get_peft_model(model, cfg)
            model.print_trainable_parameters()
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def eval_rows(max_per_label: int) -> List[Dict[str, str]]:
    ds = load_dataset("ailsntua/QEvasion", split="test")
    rows = [{"question": str(d.get("interview_question", d.get("question", ""))), "answer": str(d.get("interview_answer", "")), "label": map_label(str(d.get("clarity_label", "Indirect")))} for d in ds]
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    n = min(max_per_label, min(len(v) for v in by.values()))
    out: List[Dict[str, str]] = []
    for g in by.values():
        out.extend(random.sample(g, n))
    random.shuffle(out)
    return out


def evaluate(model, tokenizer, rows: List[Dict[str, str]], vote_samples: int) -> Tuple[float, float, List[str], List[str]]:
    device = next(model.parameters()).device
    preds, gold = [], []
    for r in rows:
        gold.append(r["label"])
        votes = []
        p = user_prompt(r["question"], r["answer"])
        for _ in range(vote_samples):
            try:
                text = tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                inp = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inp, max_new_tokens=256, do_sample=True, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
                dec = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
                m = re.search(r'"label"\s*:\s*"([^"]+)"', dec)
                votes.append(map_label(m.group(1)) if m else "Indirect")
            except Exception:
                votes.append("Indirect")
        preds.append(Counter(votes).most_common(1)[0][0])
    return accuracy_score(gold, preds), f1_score(gold, preds, average="macro", labels=["Direct Reply", "Direct Non-Reply", "Indirect"]), preds, gold


### ============================================================================
### Cell 6: Main
### ============================================================================

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    rows = upsample(load_rows(TRAINING_JSONL, HARD_CASES_JSONL if HARD_CASES_JSONL.exists() else None), SEED)
    model, tokenizer = load_model_tokenizer()
    examples = build_examples(rows, tokenizer, MAX_LENGTH)
    if not examples:
        raise RuntimeError("No examples built from JSONL.")

    random.shuffle(examples)
    n_val = max(1, int(len(examples) * VALIDATION_SPLIT)) if VALIDATION_SPLIT > 0 else 0
    val_ex = examples[:n_val] if n_val else None
    train_ex = examples[n_val:] if n_val else examples

    summary: Dict[str, Any] = {"training_rows": len(rows), "tokenized_examples": len(examples), "train_examples": len(train_ex), "val_examples": len(val_ex) if val_ex else 0}

    ev = []
    if DO_EVAL:
        ev = eval_rows(max(1, EVAL_SAMPLES // 3))
        a0, f0, _, _ = evaluate(model, tokenizer, ev, EVAL_VOTE_SAMPLES)
        summary["accuracy_before"] = float(a0)
        summary["macro_f1_before"] = float(f0)
        print(f"Before training: accuracy={a0:.4f}, macro_f1={f0:.4f}")

    train_ds = Dataset.from_list(train_ex)
    val_ds = Dataset.from_list(val_ex) if val_ex else None

    def collate(exs):
        return {
            "input_ids": torch.tensor([e["input_ids"] for e in exs], dtype=torch.long),
            "attention_mask": torch.tensor([e["attention_mask"] for e in exs], dtype=torch.long),
            "labels": torch.tensor([e["labels"] for e in exs], dtype=torch.long),
        }

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=50 if val_ds else None,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False if val_ds else None,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate)
    trainer.train()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    if DO_EVAL:
        a1, f1, preds, gold = evaluate(model, tokenizer, ev, EVAL_VOTE_SAMPLES)
        summary["accuracy_after"] = float(a1)
        summary["macro_f1_after"] = float(f1)
        print(f"After training: accuracy={a1:.4f}, macro_f1={f1:.4f}")
        if SAVE_EVAL_ERRORS:
            errs = []
            for ex, p, g in zip(ev, preds, gold):
                if p != g:
                    errs.append({"interview_question": ex["question"], "interview_answer": ex["answer"], "clarity_label": g, "model_prediction": p})
            (OUTPUT_DIR / "eval_errors_for_hard_mining.json").write_text(json.dumps(errs, indent=2, ensure_ascii=False), encoding="utf-8")
            summary["eval_error_count"] = len(errs)

    (OUTPUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {OUTPUT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default=None)
    parser.add_argument("--hard-cases", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--validation-split", type=float, default=VALIDATION_SPLIT)
    parser.add_argument("--no-eval", action="store_true")
    a = parser.parse_args()

    if a.jsonl:
        TRAINING_JSONL = Path(a.jsonl)  # type: ignore[misc]
    if a.hard_cases:
        HARD_CASES_JSONL = Path(a.hard_cases)  # type: ignore[misc]
    if a.output_dir:
        OUTPUT_DIR = Path(a.output_dir)  # type: ignore[misc]
    EPOCHS = a.epochs  # type: ignore[misc]
    BATCH_SIZE = a.batch_size  # type: ignore[misc]
    MAX_LENGTH = a.max_length  # type: ignore[misc]
    VALIDATION_SPLIT = a.validation_split  # type: ignore[misc]
    DO_EVAL = not a.no_eval  # type: ignore[misc]

    main()
