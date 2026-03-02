# CLARITY - Unmasking Political Question Evasions

[![Paper Status](https://img.shields.io/badge/EMNLP%202024-Accepted-brightgreen)](https://example.com/link-to-paper)
[![arXiv](https://img.shields.io/badge/arXiv-2409.13879-b31b1b)](https://arxiv.org/abs/2409.13879)

![alt text](https://github.com/konstantinosftw/CLARITY-SemEval-2026/blob/main/logo.jpg?raw=true)

This repository provides resources for **detecting and classifying response clarity in political interviews** (taxonomy, dataset, baselines). The [paper](https://arxiv.org/abs/2409.13879) and [dataset (QEvasion)](https://huggingface.co/datasets/ailsntua/QEvasion) are from the original CLARITY work.

---

## SemEval-2026 submission (our workflow)

We address the **SemEval 2026 CLARITY task** with a different execution path than the paper baselines:

- **Task:** 3-class clarity (Direct Reply, Direct Non-Reply, Indirect) on QEvasion / SemEval gold.
- **Our pipeline:** RoBERTa binary (DNR vs rest) + Granite 3.2 8B (3-class with LoRA), combined via a fusion rule (`sum_both_balanced_v4`).
- **Reference Kaggle notebook (Granite training only):** [`granite-training-only`](https://www.kaggle.com/code/gigibot/granite-training-only/notebook)
- **Single-command eval (no flags):** run from repo root:
  ```bash
  pip install -r requirements.txt  # or: transformers datasets peft pandas scikit-learn
  python scripts/run_ensemble_validation.py
  ```
  Uses `clear-non-reply-predictions-roberta.csv` as input; if Granite predictions are missing, runs inference from the checkpoint.
- **Kaggle:** Use `kaggle_ensemble_evaluation.ipynb` — clone repo, attach datasets (RoBERTa predictions CSV + Granite checkpoint), run all cells. See the notebook for required input dataset names.

**Results and details:** `README_EXPERIMENTS.md` (folder map, fusion variants, metrics) and `RESULTS.md` (paper-ready tables).

---

## More on our experiments (local / untracked)

Artifacts and fusion variants (RoBERTa/Granite/DeBERTa, rationale generation, etc.) are documented in `README_EXPERIMENTS.md`: folder map, run commands, and where metrics are saved. Key numbers are in `RESULTS.md`. Many result files live under `results/` and are gitignored.

---

## Original CLARITY paper: dataset and baselines

The following describes the **upstream** dataset and the **paper’s** training/inference (Falcon, LoRA, encoders). Our SemEval submission uses the pipeline above instead.

**Dataset:** [Hugging Face QEvasion](https://huggingface.co/datasets/ailsntua/QEvasion) — annotated QA pairs. The repo’s dataset folder may contain raw format, Inter-Annotator Agreement, and Counterfactual Summaries; we use the Hugging Face version for our runs.

**Installation:** `pip install -r requirements.txt`

### 1. Dataset Analysis

#### 1.1 Statistics of the Dataset
To obtain statistics of the dataset, run the following command:
```
>>> cd scripts
>>> python datasetAnalysis.py
```

#### 1.2 Analysis of Counterfactual Summaries
To analyze counterfactual summaries, execute the following command:
```
>>> python counterfactual_summaries_analysis.py
```

### 2. Zero-Shot Inference
#### 2.1 Zero-Shot Inference on Open-source Models
For the Falcon-40b model (similarly with any other hugging face model):
```
>>> python zero_shot_.py --model_name "tiiuae/falcon-40b" --output_file "falcon_40b_zero_shot_clarity.pickle"
```
```
>>> python zero_shot_.py --model_name "tiiuae/falcon-40b" --output_file "falcon_40b_zero_shot_evasion.pickle" --add_specific_labels
```
#### 2.2 Zero-Shot Inference on GPT3.5_turbo
For direct clarity problem:
```
>>> python scripts/chatgpt_zero_shot_.py --token ... --output_file "falcon_40b_zero_shot_clarity.pickle" 
```
For evasion based clarity problem:
```
>>> python chatgpt_zero_shot_.py --token ... --output_file "falcon_40b_zero_shot_evasion.pickle" --add_specific_labels
```

#### 3. Training your own model
Using lora.py, you can train the model with the following arguments:

- model_name
- train_size (default: 2700 samples)
- annotators_ids (Ids of annotators used during training; default: None, using all instances regardless of annotator)
- output_model_dir (Directory to save the trained model)
- add_specific_labels (Include this flag to specify whether evasion labels, e.g., General, Partia, etc., should be added or not.)
Example commands:
```
>>> python lora.py --model_name "tiiuae/falcon-40b" --output_model_dir "falcon_40b_clarity"
>>> python lora.py --model_name "tiiuae/falcon-40b" --output_model_dir "falcon_40b_clarity"
```

or 

```
>>> python lora.py --model_name "tiiuae/falcon-40b" --output_model_dir "falcon_40b_evasion" --add_specific_labels
```
The second command will train a models on the evasion based clarity problem (all the labels) instead of the 3 classes of evasion problem only.

Similarly, for training the encoders: 
```
>>> python encoder_train.py --model_name "roberta-base" --experiment "direct_clarity"
>>> python encoder_train.py --model_name "roberta-base" --experiment "evasion_based_clarity"
```

and inference: 
```
>>> python encoder_inference.py --model_name "roberta-base" --experiment "direct_clarity"
>>> python encoder_inference.py --model_name "roberta-base" --experiment "evasion_based_clarity"
```


### 4. Results Presented in the Paper
In order to export the results presented in the paper, run the following command:

```
>>> python results.py
```


## Abstract

*Equivocation and ambiguity in public speech are well-studied discourse phenomena, especially in political science and analysis of political interviews. Inspired by the well-grounded theory on equivocation, we aim to resolve the closely related problem of response clarity in questions extracted from political interviews, leveraging the capabilities of Large Language Models (LLMs) and human expertise. To this end, we introduce a **novel taxonomy** that frames the task of detecting and classifying response clarity and a corresponding **clarity classification dataset** which consists of question-answer (QA) pairs drawn from political interviews and annotated accordingly. Our proposed two-level taxonomy addresses the clarity of a response in terms of the information provided for a given question (high-level) and also provides a fine-grained taxonomy of evasion techniques that relate to unclear, ambiguous responses (lower-level).*

*We combine ChatGPT and human annotators to collect, validate, and annotate discrete QA pairs from political interviews, to be used for our newly introduced response clarity task.*

*We provide a detailed analysis and conduct several experiments with different model architectures, sizes, and adaptation methods to gain insights and establish new baselines over the proposed dataset and task.*


## Contact

For questions or collaborations, please contact [kthomas@islab.ntua.gr](mailto:kthomas@islab.ntua.gr) or [geofila@islab.ntua.gr](mailto:geofila@islab.ntua.gr).

---

*Note: This repository is under active development. Please check back for updates.*
