# Evaluation results (paper-ready)

SemEval gold set: N=234. Single run: `python scripts/run_ensemble_validation.py` (no flags).

## Main run: Granite, RoBERTa, Ensemble

| Model | Macro F1 | F1 (DR) | F1 (DNR) | F1 (Indirect) |
|-------|----------|---------|----------|---------------|
| Granite standalone | 0.3929 | 0.5714 | 0.3882 | 0.2190 |
| RoBERTa standalone | 0.2492 | 0.0000 | 0.1111 | 0.6364 |
| **Ensemble (sum_both_balanced_v4)** | **0.4864** | **0.6369** | **0.3929** | **0.4294** |

*Source: `results/roberta_gold_eval_local/ensemble_validation_metrics.json`*

## Fusion-rule variants

| Rule | Macro F1 | Acc | F1 (DR) | F1 (DNR) | F1 (Indirect) |
|------|----------|-----|---------|----------|---------------|
| trust_specialist | 0.44 | 0.55 | 0.62 | 0.11 | 0.59 |
| sum_both_balanced_v4 | 0.49 | 0.50 | 0.64 | 0.39 | 0.43 |
| granite_only | 0.39 | 0.40 | 0.57 | 0.39 | 0.22 |
| dnr_guardrail | 0.46 | 0.57 | 0.65 | 0.13 | 0.60 |

*From `scripts/evaluate_roberta_gold_and_merge.py` with different `--fusion-rule`; evidence in `results/roberta_gold_eval_local*/` and `README_EXPERIMENTS.md`.*
