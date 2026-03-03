"""
PyTorch Lightning training for Qwen3Classifier (classification_head or generative).
Use from notebook or CLI. Data: no reasoning columns; uses interview_question, interview_answer, clarity_label.
"""
import os
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from qwen_classifier import Qwen3Classifier
from preprocessing import preprocess_function_qwen_generative, preprocess_function_qwen_classification_head
from transformers import AutoTokenizer


# Default 3-class clarity labels (same as QEvasion / rationale dataset)
DEFAULT_ID_TO_LABEL = {0: "Ambivalent", 1: "Clear Non-Reply", 2: "Clear Reply"}
DEFAULT_LABEL_TO_ID = {v: k for k, v in DEFAULT_ID_TO_LABEL.items()}


def build_tokenized_dataset(dataset, tokenizer, label_to_id, mode, max_length=512):
    """Build tokenized dataset with input_ids, attention_mask, labels. No reasoning columns."""
    if mode == "classification_head":
        def fn(examples):
            return preprocess_function_qwen_classification_head(
                examples, tokenizer=tokenizer, label_to_id=label_to_id, max_length=max_length
            )
    else:
        def fn(examples):
            return preprocess_function_qwen_generative(
                examples, tokenizer=tokenizer, label_to_id=label_to_id,
                enable_thinking=True, max_length=max_length
            )
    tokenized = dataset.map(fn, batched=True, batch_size=32)
    cols = [c for c in tokenized["train"].column_names if c not in ("input_ids", "attention_mask", "labels")]
    tokenized = tokenized.remove_columns(cols)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


def train(
    model_name,
    mode="classification_head",
    train_dataset=None,
    val_dataset=None,
    id_to_label=None,
    label_to_id=None,
    max_epochs=3,
    batch_size=4,
    lr=1e-4,
    checkpoint_dir=None,
    max_length=512,
):
    """
    Run PyTorch Lightning training for Qwen3Classifier.

    train_dataset / val_dataset: HuggingFace Dataset with keys interview_question, interview_answer, clarity_label.
    If you have a single dataset, split it first (e.g. train_test_split(0.1)).
    """
    id_to_label = id_to_label or DEFAULT_ID_TO_LABEL
    label_to_id = label_to_id or DEFAULT_LABEL_TO_ID
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if train_dataset is None or val_dataset is None:
        raise ValueError("Provide train_dataset and val_dataset (e.g. from load_dataset or from CSV)")

    # Wrap in DatasetDict-like for map
    from datasets import DatasetDict
    ds = DatasetDict({"train": train_dataset, "validation": val_dataset})
    tokenized = build_tokenized_dataset(ds, tokenizer, label_to_id, mode, max_length=max_length)
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        num_workers=0,
    )

    model = Qwen3Classifier(
        model_name=model_name,
        lr=lr,
        use_lora=True,
        num_labels=3,
        mode=mode,
        id_to_label=id_to_label,
        label_to_id=label_to_id,
    )
    if mode == "generative":
        model.tokenizer = tokenizer

    checkpoint_dir = checkpoint_dir or f"./checkpoints_qwen_{mode}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val/f1", patience=2, mode="max", verbose=True),
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{{epoch:02d}}-{{val_f1:.4f}}",
            monitor="val/f1",
            mode="max",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)
    return model, trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-8B-bnb-4bit")
    parser.add_argument("--mode", type=str, choices=("classification_head", "generative"), default="classification_head")
    parser.add_argument("--data_path", type=str, default=None, help="CSV path (optional); else use QEvasion from HuggingFace")
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    if args.data_path and os.path.isfile(args.data_path):
        from datasets import load_dataset
        dataset = load_dataset("csv", data_files=args.data_path, split="train")
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds, val_ds = split["train"], split["test"]
    else:
        from datasets import load_dataset
        dataset = load_dataset("ailsntua/QEvasion")
        train_ds = dataset["train"]
        val_ds = dataset["test"].train_test_split(test_size=0.5, seed=42)["train"]  # small val

    train(
        model_name=args.model_name,
        mode=args.mode,
        train_dataset=train_ds,
        val_dataset=val_ds,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        max_length=args.max_length,
    )
