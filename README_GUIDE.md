# Experiment Assets Map (Quick Guide and Core submissions)

This file documents artifacts that are **not part of the original tracked SemEval clone** in this workspace.

## Git Delta (What Changed vs Baseline)

- `.gitignore` (expanded ignores for local experiment outputs)
- `Granite_Rationale_Generation_Colab.ipynb` (Drive-first + rationale data augmentation and generation)
- `clear-non-reply-roberta.ipynb` (local RoBERTa voting + exports)
- `granite_training_only.ipynb` (training-only Granite workflow changes - ALSO on Kaggle - see README.md)

## Ensemble Validation on SemEval Gold (One Run)

Single command, no flags: `python scripts/run_ensemble_validation.py`. Evaluates RoBERTa (binary DNR) + Granite (3-class) on SemEval gold (N=234). Fusion rule: `sum_both_balanced_v4`.

| Model | Macro F1 | F1 (DR) | F1 (DNR) | F1 (Indirect) |
|-------|----------|---------|----------|---------------|
| Granite standalone | 0.3929 | 0.5714 | 0.3882 | 0.2190 |
| RoBERTa DNR standalone | 0.2492 | 0.0000 | 0.1111 | 0.6364 |
| **Ensemble (sum_both_balanced_v4)** | **0.4864** | **0.6369** | **0.3929** | **0.4294** |

*Source: `results/roberta_gold_eval_local/ensemble_validation_metrics.json`*
