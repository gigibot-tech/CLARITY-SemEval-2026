#!/usr/bin/env python3
"""
Push roberta-ensemble models to Hugging Face Hub.

Usage:
  1. pip install huggingface_hub
  2. hf auth login  (or set HF_TOKEN env var)
  3. python push_ensemble_to_huggingface.py
     # or: python push_ensemble_to_huggingface.py --repo-id gigibot/ensemble-qeval

Options:
  --repo-id       HF repo (default: gigibot/ensemble-qeval)
  --mode single   Upload all 3 models to ONE repo (recommended)
  --mode separate Upload each model to its own repo (3 separate repos)
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIRS = {
    "model-1": SCRIPT_DIR / "roberta-ensemble-model-1" / "final",
    "model-2": SCRIPT_DIR / "roberta-ensemble-model-2" / "final",
    "model-3": SCRIPT_DIR / "roberta-ensemble-model-3" / "final",
}

README_TEMPLATE = """---
license: mit
language:
- en
tags:
- roberta
- text-classification
- ensemble
- clarity
- qevasion
pipeline_tag: text-classification
---

# RoBERTa Clarity Ensemble

This repository contains **3 RoBERTa-large models** fine-tuned for clarity classification (Clear Reply / Clear Non-Reply / Ambivalent).

## Models

| Model | Description |
|-------|-------------|
| `model-1/` | RoBERTa-large fine-tuned on clarity task |
| `model-2/` | RoBERTa-large fine-tuned on clarity task (different seed/split) |
| `model-3/` | RoBERTa-large fine-tuned on clarity task (different seed/split) |

## Usage

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load one model
model = AutoModelForSequenceClassification.from_pretrained("gigibot/ensemble-qeval", subfolder="model-1")
tokenizer = AutoTokenizer.from_pretrained("gigibot/ensemble-qeval", subfolder="model-1")

# Or load all 3 for ensemble voting
models = []
for i in [1, 2, 3]:
    m = AutoModelForSequenceClassification.from_pretrained(
        "gigibot/ensemble-qeval",
        subfolder=f"model-{{i}}"
    )
    models.append(m)

# Ensemble inference
def ensemble_predict(text, models, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    logits_sum = None
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(**inputs)
            if logits_sum is None:
                logits_sum = out.logits
            else:
                logits_sum += out.logits
    return torch.argmax(logits_sum, dim=-1).item()
```

## Labels

- 0: Clear Reply
- 1: Clear Non-Reply  
- 2: Ambivalent

## Training

Each model was fine-tuned from `roberta-large` on the QEvasion clarity dataset.
"""


def push_single_repo(repo_id: str, private: bool = False):
    """Push all 3 models into one HF repo with subfolders."""
    api = HfApi()
    
    # Create repo
    print(f"Creating repo: {repo_id}")
    create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
    
    # Upload each model to a subfolder
    for name, path in MODEL_DIRS.items():
        if not path.exists():
            print(f"  Skip {name}: {path} not found")
            continue
        print(f"  Uploading {name} from {path}...")
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            path_in_repo=name,
            commit_message=f"Add {name}",
        )
    
    # Upload README
    readme_path = SCRIPT_DIR / "README_ensemble_hf.md"
    readme_content = README_TEMPLATE.replace("gigibot/ensemble-qeval", repo_id)
    readme_path.write_text(readme_content)
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        commit_message="Add README",
    )
    readme_path.unlink()
    
    print(f"\n✅ Done! View at: https://huggingface.co/{repo_id}")


def push_separate_repos(base_repo_id: str, private: bool = False):
    """Push each model to its own repo."""
    api = HfApi()
    
    for name, path in MODEL_DIRS.items():
        if not path.exists():
            print(f"Skip {name}: {path} not found")
            continue
        repo_id = f"{base_repo_id}-{name}"
        print(f"Creating repo: {repo_id}")
        create_repo(repo_id, repo_type="model", exist_ok=True, private=private)
        print(f"  Uploading {path}...")
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            commit_message=f"Upload {name}",
        )
        print(f"  ✅ https://huggingface.co/{repo_id}")


def main():
    ap = argparse.ArgumentParser(description="Push RoBERTa ensemble to Hugging Face")
    ap.add_argument("--repo-id", default="gigibot/ensemble-qeval", help="HF repo ID (default: gigibot/ensemble-qeval)")
    ap.add_argument("--mode", choices=["single", "separate"], default="single",
                    help="single=all models in one repo; separate=one repo per model")
    ap.add_argument("--private", action="store_true", help="Make repo(s) private")
    args = ap.parse_args()
    
    if args.mode == "single":
        push_single_repo(args.repo_id, private=args.private)
    else:
        push_separate_repos(args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
