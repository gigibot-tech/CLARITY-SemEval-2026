#!/usr/bin/env python3
"""
### ============================================================================
### Granite CLARITY Training from JSONL (Kaggle/Colab Ready)
### ============================================================================
### Uses JSONL outputs from rationale generation directly:
###   - RationaleTraining_raw.jsonl
###   - HardCases_corrections.jsonl (optional)
###
### Run:
###   python train_granite_from_jsonl.py --jsonl /path/to/RationaleTraining_raw.jsonl
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

if IN_KAGGLE:
    BASE_DIR = Path("/kaggle/working")
elif IN_COLAB:
    BASE_DIR = Path("/content/drive/MyDrive/granite_clarity")
else:
    BASE_DIR = Path.cwd()

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

CLARITY_LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}


### ============================================================================
### Cell 3: JSONL Parsing
### ============================================================================

def map_label(label: str) -> str:
    label = str(label).strip()
    if label in CLARITY_LABEL_MAP:
        return CLARITY_LABEL_MAP[label]
    if label in ("Direct Reply", "Direct Non-Reply", "Indirect"):
        return label
    return "Indirect"


def parse_jsonl_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    input_text = rec.get("input", "")
    output_text = rec.get("output", "")

    qa = re.match(r"Q:\s*(.*?)\nA:\s*(.*?)(?:\n|$)", input_text, re.DOTALL)
    if not qa:
        return None
    question = qa.group(1).strip()
    answer = qa.group(2).strip()

    reasoning = ""
    verdict = ""
    think = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL | re.IGNORECASE)
    if think:
        reasoning = think.group(1).strip()
    vm = re.search(r"Verdict:\s*([^\n]+)", output_text, re.IGNORECASE)
    if vm:
        verdict = vm.group(1).strip()

    if not question or not answer or not verdict:
        return None

    return {
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "label": map_label(rec.get("ground_truth", verdict)),
        "is_hard_case": bool(rec.get("is_hard_case", False)),
    }


def load_jsonl_rows(main_jsonl: Path, hard_jsonl: Optional[Path] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not main_jsonl.exists():
        raise FileNotFoundError(f"Missing main JSONL: {main_jsonl}")

    with main_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = parse_jsonl_record(json.loads(line))
            if parsed:
                rows.append(parsed)

    if hard_jsonl and hard_jsonl.exists():
        with hard_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["is_hard_case"] = True
                parsed = parse_jsonl_record(rec)
                if parsed:
                    rows.append(parsed)

    return rows


def upsample_balance(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    counts = {k: len(v) for k, v in by_label.items()}
    target = max(counts.values())
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for _, group in by_label.items():
        out.extend(group if len(group) >= target else rng.choices(group, k=target))
    rng.shuffle(out)
    return out


### ============================================================================
### Cell 4: Prompt/Tokenization
### ============================================================================

def build_user_prompt(question: str, answer: str) -> str:
    return f"""You are analyzing political interview answers for clarity classification.

Question: {question}
Answer: {answer}

Analyze the answer step-by-step and output JSON:
{{
  "reasoning": "Your analysis...",
  "label": "Direct Reply|Direct Non-Reply|Indirect"
}}"""


def build_assistant_output(reasoning: str, label: str, is_hard_case: bool) -> str:
    if is_hard_case:
        reasoning = f"[Self-corrected analysis]\n{reasoning}"
    return json.dumps({"reasoning": reasoning, "label": label}, ensure_ascii=False)


def build_examples(rows: List[Dict[str, Any]], tokenizer, max_length: int) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    for row in rows:
        user_prompt = build_user_prompt(row["question"], row["answer"])
        assistant_output = build_assistant_output(row["reasoning"], row["label"], row["is_hard_case"])
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            formatted = user_prompt + "\n\n" + assistant_output

        tok = tokenizer(
            formatted,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        input_ids = tok["input_ids"]
        attention_mask = tok["attention_mask"]

        try:
            prompt_only = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_len = len(tokenizer(prompt_only, return_tensors=None, add_special_tokens=False)["input_ids"])
        except Exception:
            prompt_len = len(tokenizer(user_prompt, return_tensors=None, add_special_tokens=False)["input_ids"])

        labels = [-100] * len(input_ids)
        for i in range(prompt_len, len(input_ids)):
            labels[i] = input_ids[i]

        if sum(1 for x in labels if x != -100) < 5:
            continue
        examples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
    return examples


### ============================================================================
### Cell 5: Eval Helpers
### ============================================================================

def load_eval_rows(max_per_label: int) -> List[Dict[str, str]]:
    ds = load_dataset("ailsntua/QEvasion", split="test")
    rows: List[Dict[str, str]] = []
    for item in ds:
        rows.append(
            {
                "question": str(item.get("interview_question", item.get("question", ""))),
                "answer": str(item.get("interview_answer", "")),
                "label": map_label(str(item.get("clarity_label", "Indirect"))),
            }
        )

    by_label: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    n = min(max_per_label, min(len(v) for v in by_label.values()))
    out: List[Dict[str, str]] = []
    for _, group in by_label.items():
        out.extend(random.sample(group, n))
    random.shuffle(out)
    return out


def evaluate(model, tokenizer, rows: List[Dict[str, str]], vote_samples: int) -> Tuple[float, float, List[str], List[str]]:
    device = next(model.parameters()).device
    preds: List[str] = []
    gold: List[str] = []
    for row in rows:
        gold.append(row["label"])
        prompt = build_user_prompt(row["question"], row["answer"])
        votes: List[str] = []
        for _ in range(vote_samples):
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inp = tokenizer(formatted, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inp,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
                m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
                votes.append(map_label(m.group(1)) if m else "Indirect")
            except Exception:
                votes.append("Indirect")
        preds.append(Counter(votes).most_common(1)[0][0])

    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro", labels=["Direct Reply", "Direct Non-Reply", "Indirect"])
    return acc, macro_f1, preds, gold


### ============================================================================
### Cell 6: Model & Training
### ============================================================================

def load_model_tokenizer():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    load_kwargs: Dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
    }
    if USE_8BIT and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            pass

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)

    if USE_LORA:
        try:
            import importlib

            peft = importlib.import_module("peft")
            lora_cfg = peft.LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=peft.TaskType.CAUSAL_LM,
            )
            model = peft.get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
        except Exception:
            pass

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def train(model, tokenizer, train_ex: List[Dict[str, Any]], val_ex: Optional[List[Dict[str, Any]]]):
    train_ds = Dataset.from_list(train_ex)
    val_ds = Dataset.from_list(val_ex) if val_ex else None

    def collate_fn(examples):
        return {
            "input_ids": torch.tensor([e["input_ids"] for e in examples], dtype=torch.long),
            "attention_mask": torch.tensor([e["attention_mask"] for e in examples], dtype=torch.long),
            "labels": torch.tensor([e["labels"] for e in examples], dtype=torch.long),
        }

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        logging_steps=10,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=50 if val_ds else None,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="eval_loss" if val_ds else None,
        greater_is_better=False if val_ds else None,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


### ============================================================================
### Cell 7: Main
### ============================================================================

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    rows = load_jsonl_rows(TRAINING_JSONL, HARD_CASES_JSONL if HARD_CASES_JSONL.exists() else None)
    rows = upsample_balance(rows, SEED)

    model, tokenizer = load_model_tokenizer()

    examples = build_examples(rows, tokenizer, MAX_LENGTH)
    if not examples:
        raise RuntimeError("No examples built. Check JSONL format and MAX_LENGTH.")

    random.shuffle(examples)
    n_val = max(1, int(len(examples) * VALIDATION_SPLIT)) if VALIDATION_SPLIT > 0 else 0
    val_examples = examples[:n_val] if n_val > 0 else None
    train_examples = examples[n_val:] if n_val > 0 else examples

    summary: Dict[str, Any] = {
        "training_rows": len(rows),
        "tokenized_examples": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples) if val_examples else 0,
    }

    eval_rows: List[Dict[str, str]] = []
    if DO_EVAL:
        eval_rows = load_eval_rows(max_per_label=max(1, EVAL_SAMPLES // 3))
        acc_before, f1_before, _, _ = evaluate(model, tokenizer, eval_rows, EVAL_VOTE_SAMPLES)
        summary["accuracy_before"] = float(acc_before)
        summary["macro_f1_before"] = float(f1_before)
        print(f"Before training: accuracy={acc_before:.4f}, macro_f1={f1_before:.4f}")

    train(model, tokenizer, train_examples, val_examples)

    if DO_EVAL:
        acc_after, f1_after, preds, gold = evaluate(model, tokenizer, eval_rows, EVAL_VOTE_SAMPLES)
        summary["accuracy_after"] = float(acc_after)
        summary["macro_f1_after"] = float(f1_after)
        print(f"After training: accuracy={acc_after:.4f}, macro_f1={f1_after:.4f}")

        if SAVE_EVAL_ERRORS:
            errors = []
            for ex, p, g in zip(eval_rows, preds, gold):
                if p != g:
                    errors.append(
                        {
                            "interview_question": ex["question"],
                            "interview_answer": ex["answer"],
                            "clarity_label": g,
                            "model_prediction": p,
                        }
                    )
            (OUTPUT_DIR / "eval_errors_for_hard_mining.json").write_text(
                json.dumps(errors, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            summary["eval_error_count"] = len(errors)

    (OUTPUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {OUTPUT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default=None, help="Path to RationaleTraining_raw.jsonl")
    parser.add_argument("--hard-cases", type=str, default=None, help="Path to HardCases_corrections.jsonl")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--validation-split", type=float, default=VALIDATION_SPLIT)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    if args.jsonl:
        TRAINING_JSONL = Path(args.jsonl)  # type: ignore[misc]
    if args.hard_cases:
        HARD_CASES_JSONL = Path(args.hard_cases)  # type: ignore[misc]
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)  # type: ignore[misc]
    EPOCHS = args.epochs  # type: ignore[misc]
    BATCH_SIZE = args.batch_size  # type: ignore[misc]
    MAX_LENGTH = args.max_length  # type: ignore[misc]
    VALIDATION_SPLIT = args.validation_split  # type: ignore[misc]
    DO_EVAL = not args.no_eval  # type: ignore[misc]

    main()
#!/usr/bin/env python3
"""
### ============================================================================
### Granite CLARITY Training from JSONL - Kaggle/Colab Ready
### ============================================================================
###
### Reads JSONL files directly from rationale generation:
### - RationaleTraining_raw.jsonl
### - HardCases_corrections.jsonl (optional)
###
### Usage:
###   python train_granite_from_jsonl.py --jsonl /path/to/RationaleTraining_raw.jsonl
###   python train_granite_from_jsonl.py --jsonl main.jsonl --hard-cases hard.jsonl
### ============================================================================
"""

### ============================================================================
### Cell 1: Imports and Environment Detection
### ============================================================================

import gc
import json
import os
import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

warnings.filterwarnings("ignore", message="resource_tracker")

import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

IN_KAGGLE = os.path.exists("/kaggle/working")
try:
    import google.colab  # type: ignore
    IN_COLAB = True
except Exception:
    IN_COLAB = False

if IN_KAGGLE:
    BASE_DIR = Path("/kaggle/working")
elif IN_COLAB:
    BASE_DIR = Path("/content/drive/MyDrive/granite_clarity")
else:
    BASE_DIR = Path.cwd()

### ============================================================================
### Cell 2: Configuration
### ============================================================================

TRAINING_JSONL = BASE_DIR / "qevasion_rationale" / "train_cap250_seed42_granite_8b_local" / "RationaleTraining_raw.jsonl"
HARD_CASES_JSONL = BASE_DIR / "qevasion_rationale" / "train_cap250_seed42_granite_8b_local" / "HardCases_corrections.jsonl"
OUTPUT_DIR = BASE_DIR / "granite_2b_finetuned"

MODEL_ID = "ibm-granite/granite-3.2-2b-instruct"
SEED = 42
EPOCHS = 2
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 1024
VALIDATION_SPLIT = 0.15
USE_8BIT = torch.cuda.is_available()
USE_LORA = True

DO_EVAL = True
EVAL_SAMPLES = 12
EVAL_VOTING_SAMPLES = 1
SAVE_EVAL_ERRORS = True

CLARITY_LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}


### ============================================================================
### Cell 3: Data Loading Helpers
### ============================================================================

def map_to_clarity_label(label: str) -> str:
    label = str(label).strip()
    if label in CLARITY_LABEL_MAP:
        return CLARITY_LABEL_MAP[label]
    if label in ("Direct Reply", "Direct Non-Reply", "Indirect"):
        return label
    return "Indirect"


def parse_jsonl_record(rec: dict) -> Optional[Dict]:
    input_text = rec.get("input", "")
    output_text = rec.get("output", "")

    qa_match = re.match(r"Q:\s*(.*?)\nA:\s*(.*?)(?:\n|$)", input_text, re.DOTALL)
    if not qa_match:
        return None
    question = qa_match.group(1).strip()
    answer = qa_match.group(2).strip()

    reasoning = ""
    verdict = ""
    think_match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        reasoning = think_match.group(1).strip()
    verdict_match = re.search(r"Verdict:\s*([^\n]+)", output_text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).strip()

    if not question or not answer or not verdict:
        return None

    ground_truth = rec.get("ground_truth", verdict)
    is_hard_case = bool(rec.get("is_hard_case", False))

    return {
        "question": question,
        "answer": answer,
        "reasoning": reasoning,
        "label": map_to_clarity_label(ground_truth),
        "is_hard_case": is_hard_case,
    }


def load_jsonl_data(main_jsonl: Path, hard_cases_jsonl: Optional[Path] = None) -> List[Dict]:
    rows: List[Dict] = []

    if main_jsonl.exists():
        with main_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parsed = parse_jsonl_record(json.loads(line))
                if parsed:
                    rows.append(parsed)
    else:
        raise FileNotFoundError(f"Main JSONL not found: {main_jsonl}")

    if hard_cases_jsonl and hard_cases_jsonl.exists():
        with hard_cases_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["is_hard_case"] = True
                parsed = parse_jsonl_record(rec)
                if parsed:
                    rows.append(parsed)

    return rows


def balance_rows(rows: List[Dict], seed: int = 42) -> List[Dict]:
    by_label: Dict[str, List[Dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)
    counts = {k: len(v) for k, v in by_label.items()}
    target = max(counts.values())
    rng = random.Random(seed)
    out: List[Dict] = []
    for _, group in by_label.items():
        if len(group) < target:
            out.extend(rng.choices(group, k=target))
        else:
            out.extend(group)
    rng.shuffle(out)
    return out


### ============================================================================
### Cell 4: Prompt/Tokenization Helpers
### ============================================================================

def build_user_prompt(question: str, answer: str) -> str:
    return f"""You are analyzing political interview answers for clarity classification.

Question: {question}
Answer: {answer}

Analyze the answer step-by-step:
1. Does it directly address the question?
2. Is it evasive or indirect?
3. Does it decline to answer?

Respond in JSON format:
{{
  "reasoning": "Your step-by-step analysis...",
  "label": "Direct Reply|Direct Non-Reply|Indirect"
}}"""


def build_assistant_output(reasoning: str, label: str, is_hard_case: bool = False) -> str:
    if is_hard_case:
        reasoning = f"[Self-corrected analysis]\n{reasoning}"
    return json.dumps({"reasoning": reasoning, "label": label}, ensure_ascii=False)


def build_training_examples(rows: List[Dict], tokenizer, max_length: int = 1024) -> List[Dict]:
    examples: List[Dict] = []
    for row in rows:
        user_prompt = build_user_prompt(row["question"], row["answer"])
        assistant_output = build_assistant_output(row["reasoning"], row["label"], row.get("is_hard_case", False))
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ]
        try:
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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

        labels = [-100] * len(input_ids)
        for i in range(prompt_len, min(len(input_ids), max_length)):
            labels[i] = input_ids[i]

        if sum(1 for x in labels if x != -100) < 5:
            continue

        examples.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

    return examples


### ============================================================================
### Cell 5: Evaluation
### ============================================================================

def load_eval_data(split: str = "test", max_per_label: int = 4) -> List[Dict]:
    ds = load_dataset("ailsntua/QEvasion", split=split)
    rows: List[Dict] = []
    for item in ds:
        label = map_to_clarity_label(item.get("clarity_label", "Indirect"))
        rows.append(
            {
                "question": str(item.get("interview_question", item.get("question", ""))),
                "answer": str(item.get("interview_answer", "")),
                "clarity_label": label,
            }
        )
    by_label: Dict[str, List[Dict]] = {}
    for r in rows:
        by_label.setdefault(r["clarity_label"], []).append(r)
    n = min(max_per_label, min(len(v) for v in by_label.values()))
    out: List[Dict] = []
    for label, group in by_label.items():
        out.extend(random.sample(group, n))
    random.shuffle(out)
    return out


def evaluate_with_voting(model, tokenizer, eval_rows: List[Dict], num_samples: int = 1):
    device = next(model.parameters()).device
    preds: List[str] = []
    gold: List[str] = []
    for ex in eval_rows:
        gold.append(ex["clarity_label"])
        prompt = build_user_prompt(ex["question"], ex["answer"])
        votes: List[str] = []
        for _ in range(num_samples):
            try:
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(formatted, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
                votes.append(map_to_clarity_label(m.group(1)) if m else "Indirect")
            except Exception:
                votes.append("Indirect")
        preds.append(Counter(votes).most_common(1)[0][0])
    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, average="macro", labels=["Direct Reply", "Direct Non-Reply", "Indirect"])
    return acc, macro_f1, preds, gold


### ============================================================================
### Cell 6: Training
### ============================================================================

def load_model_and_tokenizer(model_id: str, use_8bit: bool, use_lora: bool):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    load_kwargs: Dict[str, Any] = {"torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32, "device_map": "auto"}
    if use_8bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        except Exception:
            pass
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            lora = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora)
            model.print_trainable_parameters()
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model, tokenizer


def train_model(model, tokenizer, train_examples: List[Dict], val_examples: Optional[List[Dict]], output_dir: Path):
    train_ds = Dataset.from_list(train_examples)
    val_ds = Dataset.from_list(val_examples) if val_examples else None

    def collate_fn(examples):
        return {
            "input_ids": torch.tensor([e["input_ids"] for e in examples], dtype=torch.long),
            "attention_mask": torch.tensor([e["attention_mask"] for e in examples], dtype=torch.long),
            "labels": torch.tensor([e["labels"] for e in examples], dtype=torch.long),
        }

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=50 if val_ds is not None else None,
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=val_ds is not None,
        metric_for_best_model="eval_loss" if val_ds is not None else None,
        greater_is_better=False if val_ds is not None else None,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )
    trainer.train()
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return trainer


### ============================================================================
### Cell 7: Main
### ============================================================================

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    rows = load_jsonl_data(TRAINING_JSONL, HARD_CASES_JSONL if HARD_CASES_JSONL.exists() else None)
    rows = balance_rows(rows, SEED)
    model, tokenizer = load_model_and_tokenizer(MODEL_ID, USE_8BIT, USE_LORA)
    examples = build_training_examples(rows, tokenizer, MAX_LENGTH)

    if not examples:
        raise RuntimeError("No tokenized examples built. Check JSONL format and MAX_LENGTH.")

    random.shuffle(examples)
    n_val = max(1, int(len(examples) * VALIDATION_SPLIT)) if VALIDATION_SPLIT > 0 else 0
    val_examples = examples[:n_val] if n_val > 0 else None
    train_examples = examples[n_val:] if n_val > 0 else examples

    if DO_EVAL:
        eval_rows = load_eval_data("test", max_per_label=max(1, EVAL_SAMPLES // 3))
        acc_before, f1_before, _, _ = evaluate_with_voting(model, tokenizer, eval_rows, EVAL_VOTING_SAMPLES)
        print(f"Before training: accuracy={acc_before:.4f}, macro_f1={f1_before:.4f}")

    train_model(model, tokenizer, train_examples, val_examples, OUTPUT_DIR)

    summary: Dict[str, Any] = {
        "training_rows": len(rows),
        "tokenized_examples": len(examples),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples) if val_examples else 0,
    }

    if DO_EVAL:
        acc_after, f1_after, preds, gold = evaluate_with_voting(model, tokenizer, eval_rows, EVAL_VOTING_SAMPLES)
        print(f"After training: accuracy={acc_after:.4f}, macro_f1={f1_after:.4f}")
        summary.update(
            {
                "accuracy_before": float(acc_before),
                "accuracy_after": float(acc_after),
                "macro_f1_before": float(f1_before),
                "macro_f1_after": float(f1_after),
            }
        )

        if SAVE_EVAL_ERRORS:
            errors = []
            for ex, pred, g in zip(eval_rows, preds, gold):
                if pred != g:
                    errors.append(
                        {
                            "interview_question": ex["question"],
                            "interview_answer": ex["answer"],
                            "clarity_label": g,
                            "model_prediction": pred,
                        }
                    )
            (OUTPUT_DIR / "eval_errors_for_hard_mining.json").write_text(
                json.dumps(errors, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            summary["eval_error_count"] = len(errors)

    (OUTPUT_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary to: {OUTPUT_DIR / 'training_summary.json'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, default=None, help="Path to RationaleTraining_raw.jsonl")
    parser.add_argument("--hard-cases", type=str, default=None, help="Path to HardCases_corrections.jsonl")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--validation-split", type=float, default=VALIDATION_SPLIT)
    parser.add_argument("--no-eval", action="store_true")
    args = parser.parse_args()

    if args.jsonl:
        TRAINING_JSONL = Path(args.jsonl)  # type: ignore[misc]
    if args.hard_cases:
        HARD_CASES_JSONL = Path(args.hard_cases)  # type: ignore[misc]
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)  # type: ignore[misc]
    EPOCHS = args.epochs  # type: ignore[misc]
    BATCH_SIZE = args.batch_size  # type: ignore[misc]
    MAX_LENGTH = args.max_length  # type: ignore[misc]
    VALIDATION_SPLIT = args.validation_split  # type: ignore[misc]
    DO_EVAL = not args.no_eval  # type: ignore[misc]

    main()
#!/usr/bin/env python3
"""
### ============================================================================
### Granite CLARITY Training from JSONL - Kaggle/Colab Ready
### ============================================================================
###
### Copy-paste this entire file into a Kaggle/Colab notebook cell, or run as script.
### Reads JSONL files directly from rationale generation (RationaleTraining_raw.jsonl,
### HardCases_corrections.jsonl) without intermediate CSV conversion.
###
### Usage:
###   python train_granite_from_jsonl.py --jsonl path/to/RationaleTraining_raw.jsonl
###   python train_granite_from_jsonl.py --jsonl main.jsonl --hard-cases hard.jsonl
###
### In notebook: just run the cells below.
### ============================================================================
"""

### ============================================================================
### Cell 1: Install Dependencies (run once)
### ============================================================================
# !pip install -q transformers datasets accelerate bitsandbytes peft scikit-learn

### ============================================================================
### Cell 2: Imports and Environment Detection
### ============================================================================

import gc
import json
import os
import random
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

warnings.filterwarnings("ignore", message="resource_tracker")

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

# Environment detection
IN_KAGGLE = os.path.exists('/kaggle/working')
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_KAGGLE:
    BASE_DIR = Path('/kaggle/working')
    INPUT_DIR = Path('/kaggle/input')
    print('Running on Kaggle')
elif IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    BASE_DIR = Path('/content/drive/MyDrive/granite_clarity')
    INPUT_DIR = BASE_DIR
    print('Running on Colab')
else:
    BASE_DIR = Path.cwd()
    INPUT_DIR = BASE_DIR
    print('Running locally')

print(f'BASE_DIR: {BASE_DIR}')
print(f'GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

### ============================================================================
### Cell 3: Configuration
### ============================================================================

# --- EDIT THESE PATHS ---
# Main training JSONL (from rationale generation Cell 5)
TRAINING_JSONL = BASE_DIR / 'qevasion_rationale' / 'train_cap250_seed42_granite_8b_local' / 'RationaleTraining_raw.jsonl'
# Hard cases JSONL (optional, from rationale generation Cell 5)
HARD_CASES_JSONL = BASE_DIR / 'qevasion_rationale' / 'train_cap250_seed42_granite_8b_local' / 'HardCases_corrections.jsonl'
# Output directory for trained model
OUTPUT_DIR = BASE_DIR / 'granite_2b_finetuned'

# --- TRAINING CONFIG ---
MODEL_ID = 'ibm-granite/granite-3.2-2b-instruct'
SEED = 42
EPOCHS = 2
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-5
MAX_LENGTH = 1024
VALIDATION_SPLIT = 0.15  # Hold out 15% for validation
USE_8BIT = torch.cuda.is_available()  # Auto-enable on GPU
USE_LORA = True  # Always use LoRA for memory efficiency
SAVE_EVAL_ERRORS = True  # Save misclassified examples for hard mining

# --- EVAL CONFIG ---
DO_EVAL = True
EVAL_SAMPLES = 12  # Number of eval examples (fast mode)
EVAL_VOTING_SAMPLES = 1  # Samples per example for voting (1=fast, 3=accurate)

print(f'TRAINING_JSONL: {TRAINING_JSONL}')
print(f'HARD_CASES_JSONL: {HARD_CASES_JSONL}')
print(f'OUTPUT_DIR: {OUTPUT_DIR}')
print(f'USE_8BIT: {USE_8BIT}, USE_LORA: {USE_LORA}')

### ============================================================================
### Cell 4: Label Mapping and Data Loading
### ============================================================================

CLARITY_LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}

def map_to_clarity_label(label: str) -> str:
    """Map various label formats to CLARITY format."""
    label = str(label).strip()
    if label in CLARITY_LABEL_MAP:
        return CLARITY_LABEL_MAP[label]
    if label in ("Direct Reply", "Direct Non-Reply", "Indirect"):
        return label
    return "Indirect"  # Default fallback


def parse_jsonl_record(rec: dict) -> Optional[Dict]:
    """Parse a single JSONL record into training format."""
    input_text = rec.get('input', '')
    output_text = rec.get('output', '')
    
    # Parse Q and A from input
    qa_match = re.match(r'Q:\s*(.*?)\nA:\s*(.*?)(?:\n|$)', input_text, re.DOTALL)
    if not qa_match:
        return None
    question = qa_match.group(1).strip()
    answer = qa_match.group(2).strip()
    
    # Parse reasoning and verdict from output
    reasoning = ''
    verdict = ''
    think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL | re.IGNORECASE)
    if think_match:
        reasoning = think_match.group(1).strip()
    verdict_match = re.search(r'Verdict:\s*([^\n]+)', output_text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).strip()
    
    if not question or not answer or not verdict:
        return None
    
    # Check if this is a hard case
    is_hard_case = rec.get('is_hard_case', False)
    ground_truth = rec.get('ground_truth', verdict)
    
    return {
        'question': question,
        'answer': answer,
        'reasoning': reasoning,
        'verdict': map_to_clarity_label(verdict),
        'ground_truth': map_to_clarity_label(ground_truth),
        'is_hard_case': is_hard_case,
    }


def load_jsonl_data(jsonl_path: Path, hard_cases_path: Optional[Path] = None) -> List[Dict]:
    """Load training data from JSONL files."""
    records = []
    
    # Load main JSONL
    if jsonl_path.exists():
        with jsonl_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                parsed = parse_jsonl_record(rec)
                if parsed:
                    records.append(parsed)
        print(f'Loaded {len(records)} records from {jsonl_path.name}')
    else:
        print(f'Warning: {jsonl_path} not found')
    
    # Load hard cases JSONL
    hard_count = 0
    if hard_cases_path and hard_cases_path.exists():
        with hard_cases_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec['is_hard_case'] = True  # Mark as hard case
                parsed = parse_jsonl_record(rec)
                if parsed:
                    records.append(parsed)
                    hard_count += 1
        print(f'Loaded {hard_count} hard cases from {hard_cases_path.name}')
    
    return records


def balance_data(records: List[Dict], mode: str = 'upsample', seed: int = 42) -> List[Dict]:
    """Balance data by label (upsample minority or downsample majority)."""
    by_label = {}
    for r in records:
        label = r['ground_truth']
        by_label.setdefault(label, []).append(r)
    
    counts = {k: len(v) for k, v in by_label.items()}
    print(f'Label distribution before balance: {counts}')
    
    rng = random.Random(seed)
    
    if mode == 'upsample':
        target = max(counts.values())
        balanced = []
        for label, items in by_label.items():
            if len(items) < target:
                # Upsample with replacement
                balanced.extend(rng.choices(items, k=target))
            else:
                balanced.extend(items)
    elif mode == 'downsample':
        target = min(counts.values())
        balanced = []
        for label, items in by_label.items():
            balanced.extend(rng.sample(items, min(len(items), target)))
    else:
        balanced = records
    
    rng.shuffle(balanced)
    new_counts = Counter(r['ground_truth'] for r in balanced)
    print(f'Label distribution after balance: {dict(new_counts)}')
    return balanced

### ============================================================================
### Cell 5: Tokenization and Dataset Building
### ============================================================================

def build_user_prompt(question: str, answer: str) -> str:
    """Build the user prompt for Granite."""
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


def build_assistant_output(reasoning: str, label: str, is_hard_case: bool = False) -> str:
    """Build the target assistant output."""
    if is_hard_case:
        reasoning = f"[Self-corrected analysis]\n{reasoning}"
    return json.dumps({"reasoning": reasoning, "label": label}, ensure_ascii=False)


def build_training_examples(
    records: List[Dict],
    tokenizer,
    max_length: int = 1024,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Build tokenized training examples."""
    examples = []
    stats = {
        'input_records': len(records),
        'kept': 0,
        'dropped_missing': 0,
        'dropped_truncated': 0,
        'label_distribution': Counter(),
    }
    
    for rec in records:
        question = rec['question']
        answer = rec['answer']
        reasoning = rec['reasoning']
        label = rec['ground_truth']
        is_hard = rec.get('is_hard_case', False)
        
        if not question or not answer:
            stats['dropped_missing'] += 1
            continue
        
        user_prompt = build_user_prompt(question, answer)
        assistant_output = build_assistant_output(reasoning, label, is_hard)
        
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_output},
        ]
        
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
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
        
        # Find where assistant starts for label masking
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
        
        # Labels: -100 for prompt tokens, actual ids for assistant tokens
        labels = [-100] * len(input_ids)
        for i in range(prompt_len, min(len(input_ids), max_length)):
            labels[i] = input_ids[i]
        
        supervised_tokens = sum(1 for l in labels if l != -100)
        if supervised_tokens < 5:
            stats['dropped_truncated'] += 1
            continue
        
        examples.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })
        stats['kept'] += 1
        stats['label_distribution'][label] += 1
    
    stats['label_distribution'] = dict(stats['label_distribution'])
    return examples, stats

### ============================================================================
### Cell 6: Evaluation Functions
### ============================================================================

def load_eval_data(split: str = "test", max_per_label: int = 20) -> List[Dict]:
    """Load balanced evaluation data from QEvasion."""
    ds = load_dataset("ailsntua/QEvasion", split=split)
    examples = []
    
    for item in ds:
        clarity = item.get("clarity_label")
        mapped = map_to_clarity_label(clarity)
        if mapped not in ("Direct Reply", "Direct Non-Reply", "Indirect"):
            continue
        examples.append({
            "question": str(item.get("interview_question", item.get("question", ""))),
            "answer": str(item.get("interview_answer", "")),
            "clarity_label": mapped,
        })
    
    # Balance by label
    by_label = {}
    for ex in examples:
        by_label.setdefault(ex["clarity_label"], []).append(ex)
    
    n = min(len(v) for v in by_label.values()) if by_label else 0
    n = min(n, max_per_label)
    
    balanced = []
    for label, items in by_label.items():
        balanced.extend(random.sample(items, min(n, len(items))))
    random.shuffle(balanced)
    
    return balanced


def evaluate_with_voting(
    model,
    tokenizer,
    eval_examples: List[Dict],
    num_samples: int = 1,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> Tuple[float, float, List[str], List[str]]:
    """Evaluate model with optional self-consistency voting."""
    device = next(model.parameters()).device
    all_preds = []
    all_gold = []
    
    for idx, ex in enumerate(eval_examples):
        if (idx + 1) % 5 == 0:
            print(f'  Eval {idx + 1}/{len(eval_examples)}...', flush=True)
        
        question = ex["question"]
        answer = ex["answer"]
        gold_label = ex["clarity_label"]
        all_gold.append(gold_label)
        
        user_prompt = build_user_prompt(question, answer)
        messages = [{"role": "user", "content": user_prompt}]
        votes = []
        
        for _ in range(num_samples):
            try:
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(formatted, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                
                text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Parse label from JSON
                m = re.search(r'"label"\s*:\s*"([^"]+)"', text)
                if m:
                    label = map_to_clarity_label(m.group(1))
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

### ============================================================================
### Cell 7: Model Loading
### ============================================================================

def load_model_and_tokenizer(
    model_id: str,
    use_8bit: bool = False,
    use_lora: bool = True,
):
    """Load model with optional 8-bit quantization and LoRA."""
    print(f'Loading model: {model_id}')
    
    # Clear memory
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "device_map": "auto",
    }
    
    if use_8bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print('Using 8-bit quantization')
        except ImportError:
            print('bitsandbytes not available, using full precision')
            use_8bit = False
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    
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
        except ImportError:
            print('peft not available, using full fine-tuning')
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()
    
    return model, tokenizer

### ============================================================================
### Cell 8: Training
### ============================================================================

def train_model(
    model,
    tokenizer,
    train_examples: List[Dict],
    val_examples: Optional[List[Dict]],
    output_dir: Path,
    epochs: int = 2,
    batch_size: int = 1,
    gradient_accumulation: int = 8,
    learning_rate: float = 2e-5,
    save_steps: int = 100,
    eval_steps: int = 50,
):
    """Train the model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dataset = Dataset.from_list(train_examples)
    val_dataset = Dataset.from_list(val_examples) if val_examples else None
    
    def collate_fn(examples):
        return {
            "input_ids": torch.tensor([e["input_ids"] for e in examples], dtype=torch.long),
            "attention_mask": torch.tensor([e["attention_mask"] for e in examples], dtype=torch.long),
            "labels": torch.tensor([e["labels"] for e in examples], dtype=torch.long),
        }
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        max_grad_norm=1.0,
        logging_steps=10,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=eval_steps if val_dataset else None,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False if val_dataset else None,
        fp16=torch.cuda.is_available(),
        report_to="none",
        dataloader_num_workers=0,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    print(f'Starting training: {len(train_examples)} train, {len(val_examples) if val_examples else 0} val')
    trainer.train()
    
    # Save model
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f'Model saved to: {output_dir}')
    
    return trainer

### ============================================================================
### Cell 9: Main Execution
### ============================================================================

def main():
    """Main training function."""
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Load data
    print('\n=== Loading Data ===')
    records = load_jsonl_data(TRAINING_JSONL, HARD_CASES_JSONL)
    
    if not records:
        raise ValueError(f'No training data found. Check paths:\n  {TRAINING_JSONL}\n  {HARD_CASES_JSONL}')
    
    # Balance data
    records = balance_data(records, mode='upsample', seed=SEED)
    
    # Load model
    print('\n=== Loading Model ===')
    model, tokenizer = load_model_and_tokenizer(
        MODEL_ID,
        use_8bit=USE_8BIT,
        use_lora=USE_LORA,
    )
    
    # Build training examples
    print('\n=== Building Training Examples ===')
    all_examples, stats = build_training_examples(records, tokenizer, MAX_LENGTH)
    print(f'Training stats: {stats}')
    
    # Train/val split
    if VALIDATION_SPLIT > 0:
        n_val = max(1, int(len(all_examples) * VALIDATION_SPLIT))
        random.shuffle(all_examples)
        val_examples = all_examples[:n_val]
        train_examples = all_examples[n_val:]
        print(f'Split: {len(train_examples)} train, {len(val_examples)} validation')
    else:
        train_examples = all_examples
        val_examples = None
    
    # Pre-training eval
    if DO_EVAL:
        print('\n=== Pre-Training Evaluation ===')
        eval_data = load_eval_data(split="test", max_per_label=EVAL_SAMPLES // 3)
        acc_before, f1_before, _, _ = evaluate_with_voting(
            model, tokenizer, eval_data, num_samples=EVAL_VOTING_SAMPLES
        )
        print(f'Before training: accuracy={acc_before:.4f}, macro_f1={f1_before:.4f}')
    
    # Train
    print('\n=== Training ===')
    trainer = train_model(
        model, tokenizer,
        train_examples, val_examples,
        OUTPUT_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
    )
    
    # Post-training eval
    if DO_EVAL:
        print('\n=== Post-Training Evaluation ===')
        acc_after, f1_after, preds, gold = evaluate_with_voting(
            model, tokenizer, eval_data, num_samples=EVAL_VOTING_SAMPLES
        )
        print(f'After training: accuracy={acc_after:.4f}, macro_f1={f1_after:.4f}')
        
        # Save evaluation errors for hard mining
        if SAVE_EVAL_ERRORS:
            errors = []
            for ex, pred, true_label in zip(eval_data, preds, gold):
                if pred != true_label:
                    errors.append({
                        "interview_question": ex["question"],
                        "interview_answer": ex["answer"],
                        "clarity_label": true_label,
                        "model_prediction": pred,
                    })
            errors_path = OUTPUT_DIR / 'eval_errors_for_hard_mining.json'
            errors_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False))
            print(f'Saved {len(errors)} evaluation errors to: {errors_path}')
    
    # Save training summary
    summary = {
        "model_id": MODEL_ID,
        "training_records": len(records),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples) if val_examples else 0,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
    }
    if DO_EVAL:
        summary["accuracy_before"] = float(acc_before)
        summary["accuracy_after"] = float(acc_after)
        summary["macro_f1_before"] = float(f1_before)
        summary["macro_f1_after"] = float(f1_after)
        summary["improvement"] = float(acc_after - acc_before)
    
    (OUTPUT_DIR / 'training_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'\nTraining complete! Summary saved to: {OUTPUT_DIR / "training_summary.json"}')
    
    return model, tokenizer

### ============================================================================
### Cell 10: Run Training
### ============================================================================

if __name__ == "__main__":
    # For script mode, allow CLI args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--jsonl', type=str, help='Path to main training JSONL')
    parser.add_argument('--hard-cases', type=str, help='Path to hard cases JSONL')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--max-length', type=int, default=MAX_LENGTH)
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()
    
    # Override config with CLI args
    if args.jsonl:
        TRAINING_JSONL = Path(args.jsonl)
    if args.hard_cases:
        HARD_CASES_JSONL = Path(args.hard_cases)
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    if args.epochs:
        EPOCHS = args.epochs
    if args.batch_size:
        BATCH_SIZE = args.batch_size
    if args.max_length:
        MAX_LENGTH = args.max_length
    if args.no_eval:
        DO_EVAL = False
    
    main()

# For notebook mode, just run:
# model, tokenizer = main()
