"""
Qwen3-8B Classifier with LoRA support for both classification head and generative modes.
"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
from torchmetrics import Accuracy, F1Score
import torch.nn as nn
import json
import re


class Qwen3Classifier(pl.LightningModule):
    """Qwen3-8B with LoRA - supports both classification head and generative modes."""

    def __init__(self, model_name: str, lr: float = 1e-4, 
                 use_lora: bool = True, num_labels: int = 3, 
                 mode: str = "generative",
                 id_to_label: dict = None, label_to_id: dict = None):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = None  # set before fit for generative val: model.tokenizer = tokenizer

        # GGUF repos only contain .gguf files; transformers needs safetensors/pytorch. Use a HF model name instead.
        if "GGUF" in (model_name or "").upper():
            raise ValueError(
                "GGUF model repos cannot be loaded with HuggingFace transformers (they have no pytorch_model.bin or model.safetensors). "
                "Use a HuggingFace model for this pipeline, e.g. 'Qwen/Qwen3-8B' (full precision) or 'unsloth/Qwen3-8B-bnb-4bit' (4-bit, requires bitsandbytes). "
                "To use GGUF, load the .gguf file with llama-cpp-python or Ollama instead."
            )

        # Load model as CausalLM
        # Note: Don't use device_map="auto" with PyTorch Lightning - it creates meta tensors
        # Lightning will handle device placement automatically
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use float16 to save memory for large models
        )

        config = base_model.config
        hidden_size = config.hidden_size
        model_dtype = next(base_model.model.parameters()).dtype

        self.mode = mode
        self.num_labels = num_labels
        self.id2label = id_to_label if id_to_label is not None else {}
        self.label2id = label_to_id if label_to_id is not None else {}
        
        if mode == "classification_head":
            # Classification head approach
            self.backbone = base_model.model
            self.classifier = nn.Linear(hidden_size, num_labels).to(dtype=model_dtype)
            nn.init.normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)
            
            if use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION,
                    r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                )
                self.backbone = get_peft_model(self.backbone, lora_config)
                self.backbone.print_trainable_parameters()
            if getattr(self.backbone, "gradient_checkpointing_enable", None) is not None:
                self.backbone.gradient_checkpointing_enable()
        else:
            # Generative approach - use full model
            self.model = base_model
            if getattr(self.model, "gradient_checkpointing_enable", None) is not None:
                self.model.gradient_checkpointing_enable()
            if use_lora:
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                )
                self.model = get_peft_model(self.model, lora_config)
                self.model.print_trainable_parameters()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.train_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_labels)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_labels, average="macro")

    def forward(self, input_ids, attention_mask, labels=None):
        if self.mode == "classification_head":
            # Classification head approach
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            attention_mask_expanded = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
            pooled_output = (hidden_states * attention_mask_expanded).sum(1) / attention_mask.sum(1, keepdim=True).to(dtype=hidden_states.dtype)
            logits = self.classifier(pooled_output)
            
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            
            from transformers.modeling_outputs import SequenceClassifierOutput
            return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=hidden_states)
        else:
            # Generative approach - return logits for next token prediction
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            return outputs

    def generate_classification(self, input_ids, attention_mask, tokenizer, max_new_tokens=2000):
        """Generate text and parse classification from JSON response."""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode generated text
        generated_texts = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse JSON or extract label
        predictions = []
        for text in generated_texts:
            label_id = self._parse_generated_text(text)
            if label_id is None:
                label_id = 0  # Default to Ambivalent if parsing fails
            predictions.append(label_id)
        
        return torch.tensor(predictions, device=input_ids.device)
    
    def _parse_generated_text(self, text):
        """Parse generated text to extract class label."""
        text = text.strip().lower()
        
        # Try to parse JSON first
        try:
            # Look for JSON object
            json_match = re.search(r'\{[^}]+\}', text)
            if json_match:
                parsed = json.loads(json_match.group())
                if 'label' in parsed:
                    label_str = parsed['label'].strip()
                    return self._label_to_id(label_str)
                if 'classification' in parsed:
                    label_str = parsed['classification'].strip()
                    return self._label_to_id(label_str)
        except Exception as e:
            pass  # Silently continue to fallback parsing
        
        # Try direct label matching (case-insensitive)
        text_lower = text.lower()
        
        # Check for "Clear Reply" first (before "Clear Non-Reply")
        if 'clear reply' in text_lower and 'non-reply' not in text_lower:
            return self.label2id.get('Clear Reply', 2)
        
        # Check for "Clear Non-Reply"
        if 'clear non-reply' in text_lower or ('clear' in text_lower and 'non-reply' in text_lower):
            return self.label2id.get('Clear Non-Reply', 1)
        
        # Check for "Ambivalent"
        if 'ambivalent' in text_lower:
            return self.label2id.get('Ambivalent', 0)
        
        # If nothing matches, return None (caller should handle)
        return None

    def _label_to_id(self, label_str):
        """Convert label string to ID."""
        label_str = label_str.lower().strip()
        for label_name, label_id in self.label2id.items():
            if label_name.lower() == label_str:
                return label_id
        return 0  # Default to Ambivalent

    def training_step(self, batch, batch_idx):
        if self.mode == "classification_head":
            outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=-1)
        else:
            # Generative: use language modeling loss
            outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss
            
            # For training, we still need predictions - use logits at prompt end
            # This is approximate - generative training is more complex
            logits = outputs.logits[:, -1, :]  # Last token logits
            # Map to classification (this is a simplification)
            preds = torch.argmax(logits, dim=-1) % self.num_labels  # Rough approximation

        self.train_acc(preds, batch["labels"])
        self.train_f1(preds, batch["labels"])
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        self.log("train/f1", self.train_f1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.mode == "classification_head":
            outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=-1)
        else:
            outputs = self(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss
            # For validation, use generation + parsing if tokenizer was set (e.g. model.tokenizer = tokenizer)
            if getattr(self, "tokenizer", None) is not None:
                preds = self.generate_classification(
                    batch["input_ids"],
                    batch["attention_mask"],
                    self.tokenizer,
                )
            else:
                preds = torch.argmax(outputs.logits[:, -1, :], dim=-1) % self.num_labels

        self.val_acc(preds, batch["labels"])
        self.val_f1(preds, batch["labels"])
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/f1", self.val_f1, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)