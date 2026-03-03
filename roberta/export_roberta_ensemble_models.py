#!/usr/bin/env python3
"""
Convert roberta-ensemble-model-1 and roberta-ensemble-model-2 to ONNX and/or PyTorch formats.
Uses the 'final' checkpoint in each folder (config + model.safetensors).

Usage:
  python export_roberta_ensemble_models.py [--format onnx|pt|both] [--out-dir DIR]

Outputs:
  - ONNX:  model.onnx (fixed seq_len 256, dynamic batch)
  - PyTorch:  model.pt (state_dict), model_full.pt (full model + config refs)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Paths relative to script or Desktop
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIRS = [
    SCRIPT_DIR / "roberta-ensemble-model-1" / "final",
    SCRIPT_DIR / "roberta-ensemble-model-2" / "final",
]
DEFAULT_OUT_DIR = SCRIPT_DIR / "roberta_ensemble_exported"
MAX_LENGTH = 256


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_path: Path):
    """Load RoBERTa from HuggingFace checkpoint (config + safetensors)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    return model, tokenizer


class _LogitsOnlyWrapper(torch.nn.Module):
    """Wraps HF model so forward returns logits tensor (required for ONNX)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits


def export_onnx(model, tokenizer, out_path: Path, max_length: int = MAX_LENGTH):
    """Export model to ONNX (fixed seq_len, dynamic batch)."""
    model.eval()
    device = get_device()
    model = model.to(device)
    wrapper = _LogitsOnlyWrapper(model)

    # Dummy inputs for export (batch=1, seq_len=max_length)
    batch_size = 1
    dummy_input_ids = torch.randint(0, 1000, (batch_size, max_length), device=device, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, max_length, device=device, dtype=torch.long)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size"},
                "attention_mask": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    print(f"  ONNX saved: {out_path}")


def export_pt_state_dict(model, out_path: Path):
    """Save only state_dict (smaller, load with model.load_state_dict)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"  State dict saved: {out_path}")


def export_pt_full(model, tokenizer, model_path: Path, out_path: Path):
    """Save full model + path to config so it can be loaded without config in same dir."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save model state_dict + tokenizer path; loading script will use from_pretrained(model_path) and load_state_dict
    payload = {
        "state_dict": model.state_dict(),
        "config_path": str(model_path),
    }
    torch.save(payload, out_path)
    print(f"  Full .pt saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Export RoBERTa ensemble models to ONNX / PyTorch")
    ap.add_argument("--format", choices=["onnx", "pt", "both"], default="both", help="Export format")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Base output directory")
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH, help="Sequence length for ONNX export")
    args = ap.parse_args()

    out_dir = args.out_dir
    export_onnx_fmt = args.format in ("onnx", "both")
    export_pt_fmt = args.format in ("pt", "both")

    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            print(f"Skip (not found): {model_dir}")
            continue
        name = model_dir.parent.name  # e.g. roberta-ensemble-model-1
        sub_out = out_dir / name
        sub_out.mkdir(parents=True, exist_ok=True)
        print(f"Loading {name} from {model_dir}")
        model, tokenizer = load_model_and_tokenizer(model_dir)
        model.eval()

        if export_onnx_fmt:
            export_onnx(model, tokenizer, sub_out / "model.onnx", max_length=args.max_length)
        if export_pt_fmt:
            export_pt_state_dict(model, sub_out / "model.pt")
            export_pt_full(model, tokenizer, model_dir, sub_out / "model_full.pt")

        # Always copy tokenizer files so ONNX/PT can be used with same tokenizer
        tokenizer.save_pretrained(str(sub_out))
        print(f"  Tokenizer saved under {sub_out}")

    print("Done.")


if __name__ == "__main__":
    main()
