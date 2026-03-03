#!/usr/bin/env python3
"""
Clarity Classification Ensemble Training with PyTorch Lightning.

Fixes for overfitting:
- Strong regularization (dropout, weight decay, label smoothing)
- Early stopping on validation F1
- Lower learning rate with warmup
- Stratified K-fold for ensemble diversity
- Class-weighted loss for imbalance

Usage:
  python train_clarity_ensemble.py --model roberta-base --epochs 5 --num_folds 3
  python train_clarity_ensemble.py --model roberta-large --epochs 3 --lr 1e-5

Requirements:
  pip install pytorch-lightning transformers datasets scikit-learn torchmetrics peft
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Accuracy, F1Score, Precision, Recall
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
try:
    from peft import LoraConfig, TaskType, get_peft_model
    from peft import PeftModel
except ImportError:  # Optional dependency
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    PeftModel = None


# ============================================================================
# Configuration
# ============================================================================

LABEL2ID = {"Clear Reply": 0, "Clear Non-Reply": 1, "Ambivalent": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 3


# ============================================================================
# Dataset
# ============================================================================

class ClarityDataset(Dataset):
    """PyTorch Dataset for QEvasion clarity classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: AutoTokenizer,
        max_length: int = 256,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================================
# Lightning Module
# ============================================================================

class ClarityClassifier(pl.LightningModule):
    """PyTorch Lightning module for clarity classification."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes: int = NUM_CLASSES,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.1,
        warmup_ratio: float = 0.1,
        dropout: float = 0.2,
        label_smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
        total_steps: int = 1000,
        use_peft: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        # Load pretrained model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            classifier_dropout=dropout,
        )

        # Optional PEFT (LoRA)
        if use_peft:
            if get_peft_model is None:
                raise ImportError(
                    "PEFT is not installed. Run: pip install peft"
                )
            target_modules = lora_target_modules or ["query", "key", "value"]
            lora_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, lora_config)

        # Class weights for imbalanced data
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

        # Per-class F1 for debugging
        self.val_f1_per_class = F1Score(task="multiclass", num_classes=num_classes, average=None)

        # Test metrics
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.test_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _compute_loss(self, logits, labels):
        """Compute weighted cross-entropy loss with label smoothing."""
        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
        else:
            weights = None

        return F.cross_entropy(
            logits,
            labels,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )

    def training_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        labels = batch["label"]

        loss = self._compute_loss(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        # Metrics
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        self.log("train/f1_macro", self.train_f1, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        labels = batch["label"]

        loss = self._compute_loss(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        # Metrics
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1_per_class(preds, labels)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/f1_macro", self.val_f1, prog_bar=True)
        self.log("val/precision", self.val_precision)
        self.log("val/recall", self.val_recall)

        return loss

    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        logits = outputs.logits
        labels = batch["label"]

        loss = self._compute_loss(logits, labels)
        preds = torch.argmax(logits, dim=-1)

        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)

        self.log("test/loss", loss, prog_bar=True)
        self.log("test/acc", self.test_acc, prog_bar=True)
        self.log("test/f1_macro", self.test_f1, prog_bar=True)
        self.log("test/precision", self.test_precision)
        self.log("test/recall", self.test_recall)

        return loss

    def on_validation_epoch_end(self):
        # Log per-class F1
        f1_per_class = self.val_f1_per_class.compute()
        for i, f1 in enumerate(f1_per_class):
            self.log(f"val/f1_{ID2LABEL[i]}", f1)
        self.val_f1_per_class.reset()

    def configure_optimizers(self):
        # Separate weight decay for different parameter groups
        no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=1e-8,
        )

        # Linear warmup + decay
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.total_steps * self.hparams.warmup_ratio),
            num_training_steps=self.hparams.total_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ============================================================================
# Data Module
# ============================================================================

class ClarityDataModule(pl.LightningDataModule):
    """Lightning DataModule for QEvasion clarity dataset."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        batch_size: int = 16,
        max_length: int = 256,
        num_workers: int = 0,
        train_texts: list[str] = None,
        train_labels: list[int] = None,
        val_texts: list[str] = None,
        val_labels: list[int] = None,
        test_texts: list[str] = None,
        test_labels: list[int] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers

        self.train_texts = train_texts
        self.train_labels = train_labels
        self.val_texts = val_texts
        self.val_labels = val_labels
        self.test_texts = test_texts
        self.test_labels = test_labels

        self.tokenizer = None

    def setup(self, stage=None):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if stage == "fit" or stage is None:
            self.train_dataset = ClarityDataset(
                self.train_texts, self.train_labels,
                self.tokenizer, self.max_length,
            )
            self.val_dataset = ClarityDataset(
                self.val_texts, self.val_labels,
                self.tokenizer, self.max_length,
            )

        if stage == "test" or stage is None:
            if self.test_texts is not None:
                self.test_dataset = ClarityDataset(
                    self.test_texts, self.test_labels,
                    self.tokenizer, self.max_length,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


# ============================================================================
# Data Loading
# ============================================================================

def _prepare_texts_labels(df) -> Tuple[List[str], List[int]]:
    """Build (texts, labels) from a DataFrame with question/answer and clarity_label."""
    q_col = "question" if "question" in df.columns else "interview_question"
    a_col = "interview_answer" if "interview_answer" in df.columns else "answer"
    lbl_col = "clarity_label"
    if q_col not in df.columns or a_col not in df.columns or lbl_col not in df.columns:
        raise ValueError(f"CSV must have {q_col}/{a_col}/{lbl_col} (or question/interview_answer/clarity_label)")
    texts = [
        f"Question: {q}\nAnswer: {a}"
        for q, a in zip(df[q_col].astype(str), df[a_col].astype(str))
    ]
    labels = [LABEL2ID.get(str(lbl).strip(), 2) for lbl in df[lbl_col]]
    return texts, labels


def load_split_from_csv(csv_path: str) -> Tuple[List[str], List[int]]:
    """Load a single split (train or test) from a QEvasion-style CSV. Not used in training when used as test input."""
    import pandas as pd
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(path)
    texts, labels = _prepare_texts_labels(df)
    print(f"Loaded {len(texts)} samples from {csv_path}")
    return texts, labels


def load_qevasion_data(train_csv: Optional[str] = None, test_csv: Optional[str] = None):
    """Load and prepare QEvasion dataset. If train_csv or test_csv are set, load that split from CSV instead of HF."""
    if train_csv is not None:
        train_texts, train_labels = load_split_from_csv(train_csv)
    else:
        train_texts, train_labels = None, None
    if test_csv is not None:
        test_texts, test_labels = load_split_from_csv(test_csv)
    else:
        test_texts, test_labels = None, None

    if train_texts is None or test_texts is None:
        print("Loading QEvasion dataset from HuggingFace...")
        dataset = load_dataset("ailsntua/QEvasion")

        def prepare_split(split):
            ds = dataset[split]
            texts = [
                f"Question: {q}\nAnswer: {a}"
                for q, a in zip(ds["question"], ds["interview_answer"])
            ]
            labels = [LABEL2ID.get(lbl, 2) for lbl in ds["clarity_label"]]
            return texts, labels

        if train_texts is None:
            train_texts, train_labels = prepare_split("train")
        if test_texts is None:
            test_texts, test_labels = prepare_split("test")

    print(f"Train: {len(train_texts)} samples")
    print(f"Test: {len(test_texts)} samples")
    print(f"Train label distribution: {Counter(train_labels)}")
    print(f"Test label distribution: {Counter(test_labels)}")

    return train_texts, train_labels, test_texts, test_labels


def compute_class_weights_tensor(labels: list[int]) -> torch.Tensor:
    """Compute balanced class weights."""
    classes = np.array(sorted(set(labels)))
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# Training Functions
# ============================================================================

def train_single_fold(
    fold_idx: int,
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    """Train a single fold of the ensemble."""

    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}/{args.num_folds}")
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")
    print(f"{'='*60}")

    # Compute class weights from training data
    class_weights = compute_class_weights_tensor(train_labels)
    print(f"Class weights: {class_weights.tolist()}")

    # Data module
    dm = ClarityDataModule(
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        test_texts=test_texts,
        test_labels=test_labels,
    )
    dm.setup()

    # Calculate total steps
    num_batches = math.ceil(len(dm.train_dataset) / args.batch_size)
    steps_per_epoch = math.ceil(num_batches / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)

    # Model
    model = ClarityClassifier(
        model_name=args.model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
        class_weights=class_weights,
        total_steps=total_steps,
        use_peft=args.use_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    # Callbacks
    fold_dir = output_dir / f"fold-{fold_idx + 1}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=fold_dir,
            filename="best-{epoch:02d}-{val/f1_macro:.4f}",
            monitor="val/f1_macro",
            mode="max",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/f1_macro",
            patience=args.patience,
            mode="max",
            min_delta=0.001,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = CSVLogger(save_dir=fold_dir, name="logs")

    # Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.grad_accum,
        precision=args.precision,
        deterministic=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, dm)

    # Load best checkpoint
    best_model_path = callbacks[0].best_model_path
    print(f"Best model: {best_model_path}")

    # Test evaluation
    if test_texts:
        trainer.test(model, dm, ckpt_path="best")

    # Save final model in HuggingFace format
    final_dir = fold_dir / "final"
    final_dir.mkdir(exist_ok=True)

    best_model = ClarityClassifier.load_from_checkpoint(best_model_path)
    best_model.model.save_pretrained(final_dir)
    dm.tokenizer.save_pretrained(final_dir)
    if args.use_peft:
        (final_dir / "peft_base_model.txt").write_text(args.model)
        if args.merge_lora:
            if PeftModel is None:
                raise ImportError("PEFT is not installed. Run: pip install peft")
            merge_dir = Path(args.merge_lora_output) if args.merge_lora_output else (fold_dir / "merged")
            merge_dir.mkdir(exist_ok=True)
            merged_model = best_model.model.merge_and_unload()
            merged_model.save_pretrained(merge_dir)
            dm.tokenizer.save_pretrained(merge_dir)
            (merge_dir / "merged_from_peft.txt").write_text(str(final_dir))

    # Optional: overwrite fixed ensemble folders (roberta-ensemble-model-1..N)
    if args.overwrite_ensemble_dirs:
        target_root = Path(args.overwrite_ensemble_root)
        target_dir = target_root / f"roberta-ensemble-model-{fold_idx + 1}" / "final"
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(final_dir, target_dir)
        if args.use_peft and args.merge_lora:
            merged_source = Path(args.merge_lora_output) if args.merge_lora_output else (fold_dir / "merged")
            merged_target = target_root / f"roberta-ensemble-model-{fold_idx + 1}" / "merged"
            if merged_target.exists():
                shutil.rmtree(merged_target)
            shutil.copytree(merged_source, merged_target)

    print(f"Model saved to: {final_dir}")

    # Return metrics
    return {
        "fold": fold_idx + 1,
        "best_val_f1": float(callbacks[0].best_model_score),
        "best_model_path": str(final_dir),
    }


def train_ensemble(args: argparse.Namespace):
    """Train ensemble using stratified K-fold."""

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Load data (optionally from saved CSV, e.g. dataset/qevasion_test_308.csv from clear-non-reply-evaluation-roberta.ipynb)
    train_texts, train_labels, test_texts, test_labels = load_qevasion_data(
        train_csv=getattr(args, "train_csv", None),
        test_csv=getattr(args, "test_csv", None),
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"ensemble_{args.model.replace('/', '_')}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = vars(args)
    config["timestamp"] = timestamp
    config["train_size"] = len(train_texts)
    config["test_size"] = len(test_texts)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_texts, train_labels)):
        fold_train_texts = [train_texts[i] for i in train_idx]
        fold_train_labels = [train_labels[i] for i in train_idx]
        fold_val_texts = [train_texts[i] for i in val_idx]
        fold_val_labels = [train_labels[i] for i in val_idx]

        fold_result = train_single_fold(
            fold_idx=fold_idx,
            train_texts=fold_train_texts,
            train_labels=fold_train_labels,
            val_texts=fold_val_texts,
            val_labels=fold_val_labels,
            test_texts=test_texts,
            test_labels=test_labels,
            args=args,
            output_dir=output_dir,
        )
        results.append(fold_result)

    # Save results
    results_path = output_dir / "ensemble_results.json"
    results_path.write_text(json.dumps(results, indent=2))

    # Summary
    print("\n" + "=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)
    avg_f1 = np.mean([r["best_val_f1"] for r in results])
    print(f"Average validation F1: {avg_f1:.4f}")
    for r in results:
        print(f"  Fold {r['fold']}: F1={r['best_val_f1']:.4f}")
    print(f"\nModels saved to: {output_dir}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train clarity classification ensemble with PyTorch Lightning"
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="roberta-base",
        help="HuggingFace model name (default: roberta-base, recommended over roberta-large to reduce overfitting)"
    )
    parser.add_argument(
        "--use_peft", action="store_true",
        help="Enable PEFT (LoRA) adapters instead of full fine-tuning"
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05)")
    parser.add_argument(
        "--lora_target_modules",
        type=lambda s: [m.strip() for m in s.split(",") if m.strip()],
        default=None,
        help="Comma-separated target modules for LoRA (default: query,key,value)",
    )
    parser.add_argument(
        "--merge_lora", action="store_true",
        help="After training with PEFT, merge LoRA adapters into full model"
    )
    parser.add_argument(
        "--merge_lora_output", type=str, default="",
        help="Optional output directory for merged model (default: fold-N/merged)"
    )

    # Training
    parser.add_argument("--epochs", type=int, default=5, help="Max epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay (default: 0.1)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience (default: 2)")

    # Regularization
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (default: 0.2)")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")

    # Data
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length (default: 256)")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument(
        "--test-csv",
        type=str,
        default=None,
        help="Optional path to test split CSV (e.g. dataset/qevasion_test_308.csv). If set, use this instead of QEvasion test from HF.",
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help="Optional path to train split CSV. If set, use this instead of QEvasion train from HF.",
    )

    # Ensemble
    parser.add_argument("--num_folds", type=int, default=3, help="Number of K-fold splits (default: 3)")

    # Output
    parser.add_argument("--output_dir", type=str, default="./clarity_ensemble", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--overwrite_ensemble_dirs",
        action="store_true",
        help="Overwrite ./roberta-ensemble-model-1..N/final after each fold",
    )
    parser.add_argument(
        "--overwrite_ensemble_root",
        type=str,
        default=".",
        help="Root path for roberta-ensemble-model-* folders (default: .)",
    )

    # Hardware
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Lightning accelerator (auto/cpu/gpu/mps)",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision (default: 16-mixed)")

    args = parser.parse_args()

    print("Configuration:")
    print(json.dumps(vars(args), indent=2))

    train_ensemble(args)


if __name__ == "__main__":
    main()
