# Experiment Assets Map (Non-SemEval Baseline)

This file documents artifacts that are **not part of the original tracked SemEval clone** in this workspace.

Baseline definition used here:
- "SemEval baseline" = files currently tracked by `git ls-files`.
- "Non-baseline" = currently untracked experiment files/folders you added locally.

## Git Delta (What Changed vs Baseline)

Tracked modifications (visible in `git diff --name-status`):
- `.gitignore` (expanded ignores for local experiment outputs)
- `Granite_Rationale_Generation_Colab.ipynb` (Drive-first + safer resume)
- `clear-non-reply-roberta.ipynb` (local RoBERTa voting + exports)
- `granite_training_only.ipynb` (training-only Granite workflow changes)

Local additions are intentionally mostly **untracked and/or gitignored** (e.g. `results/`, `reports_acl_submission/`, exported model folders) to keep the SemEval baseline clean.

## Key Results Snapshot (Logged Artifacts)

These are the best numbers currently saved in this workspace (paths below are local and typically gitignored):

| Task | Eval Set | N | Metric | Value | Evidence |
|---|---|---:|---|---:|---|
| 3-class (DR/DNR/Indirect), fused ensemble | SemEval gold eval (local) | 234 | macro F1 | 0.4864 | `results/roberta_gold_eval_local/ensemble_validation_metrics.json` |
| 3-class per-label F1 (same run) | SemEval gold eval (local) | 234 | F1(Direct Reply) | 0.6369 | `results/roberta_gold_eval_local/ensemble_validation_metrics.json` |
| Binary legacy (Clear Reply vs Not Clear) | legacy eval slice | 200 | macro F1 | 0.5967 | `reports_acl_submission/deberta_legacy_binary_breakdown.json` |
| Binary legacy per-class F1 (same run) | legacy eval slice | 200 | F1(Clear Reply) | 0.3830 | `reports_acl_submission/deberta_legacy_binary_breakdown.json` |
| Balanced-69 sanity eval (RoBERTa ensemble) | QEvasion balanced test subset | 69 | macro F1 | 0.1667 | `results/ensemble_balanced69_20260214_151415/metrics.json` |
| Balanced-69 sanity eval (single model-3) | QEvasion balanced test subset | 69 | macro F1 | 0.3140 | `results/ensemble_balanced69_20260214_151415/model3_only_metrics.json` |

## Ensemble Validation on SemEval Gold (One Run)

Single command, no flags: `python scripts/run_ensemble_validation.py`. Evaluates RoBERTa (binary DNR) + Granite (3-class) on SemEval gold (N=234). Fusion rule: `sum_both_balanced_v4`.

| Model | Macro F1 | F1 (DR) | F1 (DNR) | F1 (Indirect) |
|-------|----------|---------|----------|---------------|
| Granite standalone | 0.3929 | 0.5714 | 0.3882 | 0.2190 |
| RoBERTa standalone | 0.2492 | 0.0000 | 0.1111 | 0.6364 |
| **Ensemble (sum_both_balanced_v4)** | **0.4864** | **0.6369** | **0.3929** | **0.4294** |

*Source: `results/roberta_gold_eval_local/ensemble_validation_metrics.json`*

### Fusion-rule variants (same gold set)

From `scripts/evaluate_roberta_gold_and_merge.py` with different `--fusion-rule`:

| Rule | Description | Macro F1 | F1 (DR) | F1 (DNR) | F1 (Ind.) |
|------|-------------|----------|---------|----------|-----------|
| trust_specialist | RoBERTa for DNR only; Granite for DR/Indirect | 0.44 | 0.62 | 0.11 | 0.59 |
| sum_both_balanced_v4 | Combined votes, DNR bar 1.5 (default) | 0.49 | 0.64 | 0.39 | 0.43 |
| granite_only | Granite baseline (no RoBERTa) | 0.39 | 0.57 | 0.39 | 0.22 |
| dnr_guardrail | DNR guardrail variant | 0.46 | 0.65 | 0.13 | 0.60 |

### Results folder map (experiment vs output)

| Folder | Role | Main output |
|--------|------|-------------|
| `results/roberta_gold_eval_local/` | **Output** | `ensemble_validation_metrics.json` — canonical run (Granite + RoBERTa + Ensemble) |
| `results/roberta_gold_eval_local/trust_specialist/` | Experiment | `merged_ensemble_metrics.json` (trust_specialist fusion) |
| `results/roberta_gold_eval_local/roberta_dnr_then_granite/` | Experiment | Same script, roberta_dnr_then_granite rule |
| `results/roberta_gold_eval_local_dnr_guardrail/` | Experiment | dnr_guardrail fusion |
| `results/roberta_gold_eval_local_aggressive_or/` | Experiment | aggressive_or fusion |
| `results/roberta_gold_eval_local_max_compare/` | Experiment | max_compare fusion |
| `results/roberta_gold_eval_local_conservative/` | Experiment | conservative fusion |
| `results/roberta_gold_eval_local_granite_only/` | Baseline | Granite only (no RoBERTa) |
| `results/ensemble_qevasion_metrics.json` | Output | Same three-way comparison on QEvasion test (308/234 samples) |

## Quick Run Commands

Run RoBERTa ensemble on balanced 69-sample QEvasion subset:

```bash
/Users/andrearachetta/Desktop/.venv/bin/python /Users/andrearachetta/Desktop/CLARITY-SemEval-2026/scripts/evaluate_roberta_ensemble_balanced69.py \
  --split test \
  --samples-per-label 23 \
  --batch-size 16
```

Run Granite LoRA checkpoint on balanced 69-sample QEvasion subset:

```bash
/Users/andrearachetta/Desktop/.venv/bin/python /Users/andrearachetta/Desktop/CLARITY-SemEval-2026/scripts/evaluate_granite_balanced_qevasion.py \
  --adapter-path /Users/andrearachetta/Desktop/CLARITY-SemEval-2026/granite_clarity_finetuned/checkpoint-60 \
  --split test \
  --samples-per-label 23 \
  --num-samples 1 \
  --temperature 0.0
```

**Ensemble validation (SemEval gold or QEvasion test, no flags):**

```bash
cd /Users/andrearachetta/Desktop/CLARITY-SemEval-2026
python scripts/run_ensemble_validation.py
```

Uses `clear-non-reply-predictions-roberta.csv` as data source; if Granite predictions missing, runs inference from `checkpoint64/`.

## Non-Baseline Top-Level Folders

| Path | What It Is | Main Use | Key Results / Notes |
|---|---|---|---|
| `.venv_ensemble/` | Local virtualenv created during tooling attempts | Isolated Python env for package installs/tests | Optional. Safe to delete if unused |
| `__pycache__/` | Python bytecode cache | Runtime cache only | Safe to delete |
| `clarity/` | Classic CLARITY submission artifacts | Minimal classifier scripts, `.pkl` model, submission pickle/CSV | Keep if you still use legacy submission path |
| `clarity_ensemble/` | Timestamped ensemble run logs | `config.json`, fold log CSV/hparams for run provenance | Keep for reporting/reproducibility |
| `deberta/` | DeBERTa and binary-eval result snapshots | JSON metric dumps from different eval attempts | Binary QEvasion snapshot: `deberta/model_evaluation_results_qevasion.json` (F1=0.3878). Legacy binary table: `reports_acl_submission/deberta_legacy_binary_breakdown.json` (macro F1=0.5967) |
| `granite_clarity_finetuned/` | Granite LoRA adapter outputs | Adapter weights, tokenizer files, checkpoint state | Keep (core Granite artifact) |
| `kaggle_pulled/` | Pulled Kaggle notebook assets | Backup of notebook runs + kernel metadata | Reference only; safe to delete if unused |
| `qevasion_binary_model/` | DistilBERT binary model artifact folder | Tokenizer/config/training args for binary clarity path | Keep if binary experiments remain relevant |
| `evasion_binary_roberta_2/` | RoBERTa binary model folder (DNR vs rest) | Lightweight binary non-reply detector used in fusion rules | Used by gold-eval merge scripts (binary votes) |
| `reports_acl_submission/` | ACL/SemEval writeups + dataset audits | LaTeX body (`acl_body_submission_v2.tex`), rationale dataset audits, evaluation breakdown JSONs | Best 3-class logged: macro F1=0.4864 at `results/roberta_gold_eval_local/ensemble_validation_metrics.json` |
| `reports/` | Paper/report outputs | LaTeX source and compiled PDF | Keep (deliverables) |
| `roberta/` | RoBERTa ensemble workspace | Train scripts, checkpoint metadata, ONNX exports | Keep (core ensemble workspace) |
| `scripts/__pycache__/` | Script bytecode cache | Runtime cache only | Safe to delete |
| `results/` | Generated evaluation outputs | Metrics/prediction CSVs for different runs | Best 3-class logged: macro F1=0.4864 at `results/roberta_gold_eval_local/ensemble_validation_metrics.json` |

## Detailed Folder Contents

### `clarity/`

| Path | Use |
|---|---|
| `clarity/clarity_classifier_demo.py` | Demo wrapper around the simple classifier path |
| `clarity/clarity_classifier_minimal.py` | Minimal classifier implementation used for quick inference/submission |
| `clarity/clarity_classifier_simple.pkl` | Serialized trained classic classifier |
| `clarity/clarity_submission.pickle` | Pickle-format submission artifact |
| `clarity/clarity_codebench_submission.csv` | CSV-format submission/export copy |

### `clarity_ensemble/`

| Path | Use |
|---|---|
| `clarity_ensemble/ensemble_roberta-base_20260201_132359/` | Run folder with config + fold logs |
| `clarity_ensemble/ensemble_roberta-base_20260201_134652/` | Run folder used for `valbest` provenance (`val/f1_macro` argmax) |
| `clarity_ensemble/ensemble_roberta-base_20260201_135710/` | Additional run config/log folder |
| `clarity_ensemble/ensemble_roberta-base_20260201_185917/` | Additional run config/log folder |
| `clarity_ensemble/ensemble_roberta-base_20260202_135909/` | Additional run config/log folder |

Notes:
- These folders are mostly **metrics/config logs**, not packaged model weights.
- `ensemble_roberta-base_20260201_134652` has the clearest valbest trace in `fold-1/logs/version_0/metrics.csv`.

### `deberta/`

| Path | Use |
|---|---|
| `deberta/model_evaluation_results.json` | DeBERTa eval snapshot summary |
| `deberta/model_evaluation_results_qevasion.json` | QEvasion-specific metric dump |
| `deberta/model_evaluation_results copy.json` | Earlier binary-eval result dump with raw preds/labels |

### `granite_clarity_finetuned/`

| Path | Use |
|---|---|
| `granite_clarity_finetuned/adapter_model.safetensors` | Main LoRA adapter weights |
| `granite_clarity_finetuned/adapter_config.json` | LoRA/base-model config |
| `granite_clarity_finetuned/checkpoint-60/` | Training checkpoint with optimizer/scheduler/trainer state |
| `granite_clarity_finetuned/tokenizer*.json` | Tokenizer artifacts for loading adapter workflow |

### `qevasion_binary_model/`

| Path | Use |
|---|---|
| `qevasion_binary_model/config.json` | DistilBERT config for binary classification setup |
| `qevasion_binary_model/tokenizer*.json`, `vocab.txt` | Tokenizer assets |
| `qevasion_binary_model/training_args.bin` | Stored training args snapshot |

### `evasion_binary_roberta_2/`

| Path | Use |
|---|---|
| `evasion_binary_roberta_2/config.json` | RoBERTa binary classifier config (Direct Non-Reply vs rest) |
| `evasion_binary_roberta_2/tokenizer*.json` | Tokenizer assets used by the binary classifier |

### `reports/`

| Path | Use |
|---|---|
| `reports/SEMEVAL_2026_experiment_report.tex` | Main LaTeX report source |
| `reports/SEMEVAL_2026_experiment_report.pdf` | Compiled report |
| `reports/SEMEVAL_2026_experiment_report.log` | XeLaTeX log for troubleshooting |

### `roberta/`

| Path | Use |
|---|---|
| `roberta/train_clarity_ensemble.py` | Main PyTorch Lightning ensemble training pipeline |
| `roberta/train_clarity_ensemble.ipynb` | Notebook variant of ensemble training |
| `roberta/export_roberta_ensemble_models.py` | ONNX/PT export helper |
| `roberta/roberta-ensemble-model-1/` | Per-model checkpoint/final/tokenizer/log folders |
| `roberta/roberta-ensemble-model-2/` | Per-model checkpoint/final/tokenizer/log folders |
| `roberta/roberta-ensemble-model-3/` | Per-model checkpoint/final/tokenizer/log folders |
| `roberta/roberta_ensemble_exported/` | Exported ONNX artifacts |

Important:
- Current `roberta-ensemble-model-*/final/` folders in this workspace contain tokenizer/config metadata but may not include `model.safetensors` or `pytorch_model.bin`.
- If local weights are missing, use HF fallback (`gigibot/ensemble-qeval`) via `scripts/evaluate_roberta_ensemble_balanced69.py`.

### `reports_acl_submission/`

Local SemEval/ACL submission assets and audits (typically gitignored).

| Path | Use |
|---|---|
| `reports_acl_submission/acl_body_submission_v2.tex` | ACL-formatted paper body section (methods/experiments/results) |
| `reports_acl_submission/deberta_legacy_binary_breakdown.json` | Legacy binary DeBERTa breakdown (N=200, macro F1=0.5967) |
| `reports_acl_submission/rationale_dataset_audit*/` | Rationale dataset audits (label distribution, conflicts, supervision coverage) |
| `reports_acl_submission/balanced_rationale_dataset*/` | Prepared balanced rationale datasets (CSV/JSONL + summaries) |

### `results/` (Local Outputs)

Generated evaluation outputs and metrics (gitignored). Key runs:

| Path | What It Contains |
|------|------------------|
| `results/roberta_gold_eval_local/ensemble_validation_metrics.json` | **Canonical run**: Granite + RoBERTa + Ensemble (macro F1=0.4864 on N=234, fusion `sum_both_balanced_v4`) |
| `results/roberta_gold_eval_local/trust_specialist/merged_ensemble_metrics.json` | trust_specialist fusion (Macro F1 0.44, Acc 0.55) |
| `results/roberta_gold_eval_local_dnr_guardrail/merged_ensemble_metrics.json` | DNR guardrail variant (macro F1=0.46 on N=234) |
| `results/roberta_gold_eval_local_granite_only/merged_ensemble_metrics.json` | Granite-only baseline (macro F1=0.39) |
| `results/roberta_gold_eval_local_aggressive_or/`, `max_compare/`, `conservative/`, `roberta_dnr_then_granite/` | Other fusion experiments (each has `merged_ensemble_metrics.json`) |
| `results/ensemble_qevasion_metrics.json` | Same three-way comparison on QEvasion test set (308/234 samples) |
| `results/ensemble_balanced69_20260214_151415/metrics.json` | Balanced-69 ensemble sanity run (macro F1=0.1667 on N=69) |
| `results/ensemble_balanced69_20260214_151415/model3_only_metrics.json` | Balanced-69 model-3-only run (macro F1=0.3140 on N=69) |
| `results/granite_balanced_qevasion_20260214_181605/metrics.json` | Granite adapter quick check (N=3; `samples_per_label=1`) |

## Non-Baseline Root Files

| Path | Use |
|---|---|
| `GRANITE_CLARITY_PIPELINE_DESIGN.md` | Design document for integrated Granite training/eval/submission pipeline |
| `GraniteClarityEvaluation.ipynb` | Main Granite experimentation notebook |
| `GraniteClarityEvaluation copy.ipynb` | Variant/backup notebook snapshot |
| `GraniteClarityEvaluation copy 2.ipynb` | Variant/backup notebook snapshot |
| `QwenTuning.ipynb` | Qwen tuning notebook workflow |
| `Untitled3.ipynb` | Scratch notebook |
| `RationaleTraining_raw.jsonl` | Rationale SFT/KD JSONL output (generated; should live under an output dir) |
| `balanced_loader.py` | Balanced QEvasion sampling and CLARITY eval-data loader |
| `debug_utils.py` | Generative output debugging helpers (Qwen path) |
| `ensemble_train_results.json` | Saved ensemble metric summary snapshot |
| `evaluate_ensemble.py` | Legacy ensemble evaluator (train/test splits) |
| `evaluate_peft_clarity.py` | PEFT adapter evaluator for QEvasion |
| `generate_clarity_submission.py` | Legacy submission generation script from simple classifier |
| `generate_rationale_dataset.py` | Rationale generation pipeline (external API-based) |
| `granite_clarity_model.py` | Granite model loading/inference utility |
| `granite_clarity_strategy.py` | Granite prompt + JSON label strategy |
| `granite_self_consistency.py` | Granite voting/self-consistency wrapper |
| `preprocessing.py` | Qwen preprocessing functions |
| `push_ensemble_to_huggingface.py` | Push 3-model ensemble package to HF Hub |
| `qwen_classifier.py` | Qwen classifier Lightning module |
| `train_granite_rationale.py` | Granite rationale SFT training/eval script |
| `train_qwen.py` | Qwen training entrypoint |

## Non-Baseline Additions Under `scripts/`

| Path | Use |
|---|---|
| `scripts/evaluate_granite_balanced_qevasion.py` | New balanced-69 Granite checkpoint evaluation runner |
| `scripts/evaluate_roberta_ensemble_balanced69.py` | New balanced-69 3-model RoBERTa ensemble evaluation runner (local-or-HF source resolution) |
| `scripts/evaluate_roberta_gold_and_merge.py` | SemEval gold-eval runner (RoBERTa voting + optional Granite fusion); produces `results/roberta_gold_eval_local/ensemble_validation_metrics.json` style metrics |
| `scripts/run_ensemble_qevasion_validation.py` | QEvasion test fusion runner (RoBERTa binary votes + Granite 3-class predictions on same indices); writes `results/ensemble_qevasion_metrics.json` |
