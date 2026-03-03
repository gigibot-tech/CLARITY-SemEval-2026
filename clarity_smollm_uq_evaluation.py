# =============================================================================
# CLARITY SemEval 2026 — SmolLM + Laplace vs MC Dropout vs Temp Scaling (UQ)
# =============================================================================
#
# Same pipeline as kaggle_notebooks/smollm_laplace_evaluation.py but on QEvasion/CLARITY:
# 3-way classification (Direct Reply, Direct Non-Reply, Indirect), HF dataset ailsntua/QEvasion.
# UQ: Laplace (last-layer), MC Dropout, Temperature Scaling; metrics: Accuracy, Brier, ECE, AURC.
# Optionally NUM_RUNS > 1 for mean ± std over seeds.
#
# Run: python clarity_smollm_uq_evaluation.py
# Requires: torch, transformers, datasets, tqdm; optional laplace-torch + curvlinops-for-pytorch
#
# =============================================================================

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from collections import Counter, defaultdict

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOLLM_MODEL = "HuggingFaceTB/SmolLM2-360M"
EMBEDDING_DIM = 960
HIDDEN_DIM = 512
NUM_CLASSES = 3
CLARITY_LABELS = ["Direct Reply", "Direct Non-Reply", "Indirect"]
LABEL_TO_IDX = {l: i for i, l in enumerate(CLARITY_LABELS)}
# QEvasion -> CLARITY (from balanced_loader / train_granite_rationale)
QEVASION_TO_CLARITY = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent Reply": "Indirect",
    "Ambivalent": "Indirect",
}

BATCH_SIZE = 32
EPOCHS = 4
LR = 1e-3
NUM_TRAIN = 900   # 300 per class; cap for balance (QEvasion train ~3.45k)
NUM_TEST = 300    # 100 per class; QEvasion test has 308, we cap for balance
NUM_RUNS = 1      # set to 3 or 5 for mean ± std
MC_DROPOUT_SAMPLES = 30
ENSEMBLE_SIZE = 0
DATASET_NAME = "ailsntua/QEvasion"
MAX_LENGTH = 256

print(f"Device: {DEVICE} | CLARITY classes: {NUM_CLASSES} | Train: {NUM_TRAIN} Test: {NUM_TEST} | Runs: {NUM_RUNS}")


# -----------------------------------------------------------------------------
# Data: QEvasion -> CLARITY (text + label_id)
# -----------------------------------------------------------------------------
@dataclass
class Example:
    prompt: str
    label: int


def load_qevasion_clarity(
    split: str,
    max_examples: Optional[int] = None,
    samples_per_label: Optional[int] = None,
) -> List[Example]:
    """Load QEvasion split, map to CLARITY labels, return list of Example(prompt, label_id)."""
    from datasets import load_dataset
    ds = load_dataset(DATASET_NAME, split=split)
    by_label = defaultdict(list)
    for item in ds:
        clarity_raw = item.get("clarity_label") or ""
        mapped = QEVASION_TO_CLARITY.get(clarity_raw, clarity_raw)
        if mapped not in CLARITY_LABELS:
            continue
        q = str(item.get("interview_question", item.get("question", "")) or "").strip()
        a = str(item.get("interview_answer", "") or "").strip()
        if not a:
            continue
        text = f"question: {q} answer: {a}" if q else a
        label_id = LABEL_TO_IDX[mapped]
        by_label[label_id].append(Example(prompt=text, label=label_id))
    # Balance: same count per class (min count or samples_per_label)
    if samples_per_label is not None:
        n_per = samples_per_label
    else:
        n_per = min(len(v) for v in by_label.values()) if by_label else 0
    out = []
    for label_id in range(NUM_CLASSES):
        pool = by_label.get(label_id, [])
        if n_per and pool:
            chosen = random.sample(pool, min(n_per, len(pool)))
            out.extend(chosen)
    random.shuffle(out)
    if max_examples is not None and len(out) > max_examples:
        out = random.sample(out, max_examples)
    return out


# -----------------------------------------------------------------------------
# SmolLM client (frozen, mean-pooled embeddings)
# -----------------------------------------------------------------------------
class SmolLMClient:
    def __init__(self):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(SMOLLM_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(SMOLLM_MODEL, torch_dtype=torch.float32).to(DEVICE)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def embed(self, texts: List[str]) -> torch.Tensor:
        inp = self.tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        h = self.model(**inp).last_hidden_state
        mask = inp["attention_mask"].unsqueeze(-1)
        return (h * mask).sum(1) / mask.sum(1).clamp(1e-9)


def embed_all(client, data: List[Example], batch_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    embs, labels = [], []
    for i in tqdm(range(0, len(data), batch_size), desc="Embed"):
        batch = data[i : i + batch_size]
        e = client.embed([x.prompt for x in batch]).cpu()
        embs.append(e)
        labels.extend([x.label for x in batch])
    return torch.cat(embs), torch.tensor(labels, dtype=torch.long)


class EmbDataset(Dataset):
    def __init__(self, embs, labels):
        self.embs, self.labels = embs, labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return self.embs[i], self.labels[i]


# -----------------------------------------------------------------------------
# Model: projection + last_layer (Laplace on last_layer only)
# -----------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, HIDDEN_DIM), nn.ReLU(), nn.Dropout(0.1), nn.LayerNorm(HIDDEN_DIM)
        )
        self.last_layer = nn.Linear(HIDDEN_DIM, NUM_CLASSES)

    def forward(self, x):
        return self.last_layer(self.features(x))


# -----------------------------------------------------------------------------
# MC Dropout, Temperature Scaling
# -----------------------------------------------------------------------------
def mc_dropout_predict(model, loader, device, n_samples: int = 30):
    model.train()
    all_probs = []
    with torch.no_grad():
        for embs, _ in loader:
            embs = embs.to(device)
            logits_stack = torch.stack([model(embs) for _ in range(n_samples)], dim=0)
            probs_stack = torch.softmax(logits_stack, dim=-1)
            all_probs.append(probs_stack)
    probs_stack = torch.cat(all_probs, dim=1)
    mean_probs = probs_stack.mean(0)
    var_probs = probs_stack.var(0).sum(-1)
    return mean_probs, var_probs


def fit_temperature(model, loader, device, lr=0.01, max_iter=100):
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for embs, labels in loader:
            logits_list.append(model(embs.to(device)).cpu())
            labels_list.append(labels)
    logits_cal = torch.cat(logits_list)
    labels_cal = torch.cat(labels_list)
    T = nn.Parameter(torch.ones(1) * 1.5)
    opt = optim.LBFGS([T], lr=lr, max_iter=max_iter)
    def closure():
        opt.zero_grad()
        scaled = logits_cal.to(device) / T.clamp(min=1e-2)
        nll = nn.functional.cross_entropy(scaled, labels_cal.to(device))
        nll.backward()
        return nll
    opt.step(closure)
    return T.detach().item()


def predict_temperature_scaled(model, loader, device, T: float):
    model.eval()
    probs_list = []
    with torch.no_grad():
        for embs, _ in loader:
            embs = embs.to(device)
            logits = model(embs) / max(T, 1e-2)
            probs_list.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(probs_list)


# -----------------------------------------------------------------------------
# Metrics: AURC, ECE, Brier, etc.
# -----------------------------------------------------------------------------
def aurc_from_probs(all_true: torch.Tensor, all_probs: torch.Tensor) -> float:
    all_probs = all_probs.cpu() if all_probs.device.type != "cpu" else all_probs
    all_true = all_true.cpu() if all_true.device.type != "cpu" else all_true
    conf = all_probs.max(1).values
    pred = all_probs.argmax(1)
    err = (pred != all_true).float()
    order = torch.argsort(conf, descending=True)
    err_ord = err[order]
    n = len(err_ord)
    cum, aurc = 0.0, 0.0
    for k in range(1, n + 1):
        cum += err_ord[k - 1].item()
        aurc += cum / k
    return aurc / n


def compute_metrics(all_true: torch.Tensor, all_probs: torch.Tensor, n_bins: int = 10) -> dict:
    all_probs = all_probs.cpu() if all_probs.device.type != "cpu" else all_probs
    all_true = all_true.cpu() if all_true.device.type != "cpu" else all_true
    pred = all_probs.argmax(1)
    acc = pred.eq(all_true).float().mean().item()
    one_hot = torch.zeros_like(all_probs).scatter_(1, all_true.unsqueeze(1), 1.0)
    brier = (all_probs - one_hot).pow(2).sum(1).mean().item()
    confidences = all_probs.max(1).values
    correct_bin = (pred == all_true).float()
    ece = 0.0
    for i in range(n_bins):
        low, high = i / n_bins, (i + 1) / n_bins
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.sum() > 0:
            acc_bin = correct_bin[in_bin].mean().item()
            conf_bin = confidences[in_bin].mean().item()
            ece += in_bin.sum().item() * abs(acc_bin - conf_bin)
    ece = ece / len(all_true)
    aurc = aurc_from_probs(all_true, all_probs)
    probs_np = all_probs.numpy()
    entropy = -(probs_np * np.log(probs_np + 1e-9)).sum(1)
    confidence_np = confidences.numpy()
    correct_mask = (pred == all_true).numpy()
    n_correct, n_incorrect = correct_mask.sum(), (~correct_mask).sum()
    return {
        "accuracy": acc,
        "brier": brier,
        "ece": ece,
        "aurc": aurc,
        "mean_conf_correct": float(confidence_np[correct_mask].mean()) if n_correct > 0 else float("nan"),
        "mean_conf_incorrect": float(confidence_np[~correct_mask].mean()) if n_incorrect > 0 else float("nan"),
        "mean_ent_correct": float(entropy[correct_mask].mean()) if n_correct > 0 else float("nan"),
        "mean_ent_incorrect": float(entropy[~correct_mask].mean()) if n_incorrect > 0 else float("nan"),
    }


def run_one(seed: int, client, train_data: List[Example], test_data: List[Example], verbose: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if verbose:
        print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    train_embs, train_labels = embed_all(client, train_data)
    test_embs, test_labels = embed_all(client, test_data)
    if verbose:
        print(f"Train embs: {train_embs.shape}")
    train_loader = DataLoader(EmbDataset(train_embs, train_labels), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(EmbDataset(test_embs, test_labels), batch_size=BATCH_SIZE, shuffle=False)
    calib_loader = DataLoader(EmbDataset(train_embs, train_labels), batch_size=BATCH_SIZE, shuffle=False)
    model = Head().to(DEVICE)
    if verbose:
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for embs, labels in train_loader:
            embs, labels = embs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(embs), labels).backward()
            opt.step()
            _, pred = model(embs).max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
        if verbose:
            print(f"Epoch {epoch+1}/{EPOCHS} train acc: {100.*correct/total:.1f}%")
    la = None
    try:
        from laplace import Laplace
        la = Laplace(model, likelihood="classification", subset_of_weights="last_layer", hessian_structure="diag")
        la.fit(calib_loader)
        if verbose:
            print("Laplace fitted (last_layer, diag)")
    except (ImportError, ModuleNotFoundError) as e:
        if verbose and "curvlinops" in str(e):
            print("Laplace skipped: curvlinops backend missing. Install: uv pip install curvlinops-for-pytorch")
        elif verbose:
            print("Laplace skipped:", e)
    model.eval()
    all_true_list, all_point_probs, all_bayes_probs, all_bayes_var = [], [], [], []
    for embs, labels in test_loader:
        embs = embs.to(DEVICE)
        with torch.no_grad():
            logits = model(embs)
            all_point_probs.append(torch.softmax(logits, dim=-1).cpu())
        if la is not None:
            out = la(embs, pred_type="glm", link_approx="probit")
            if isinstance(out, tuple):
                probs, var = out[0].cpu(), out[1].cpu()
                all_bayes_var.append(var)
            else:
                probs = out.cpu()
            all_bayes_probs.append(probs)
        all_true_list.append(labels)
    all_true = torch.cat(all_true_list)
    all_point_probs = torch.cat(all_point_probs)
    if la is not None and all_bayes_probs:
        all_bayes_probs = torch.cat(all_bayes_probs)
        all_bayes_var_t = torch.cat(all_bayes_var) if all_bayes_var else None
        laplace_var_mean = (all_bayes_var_t.sum(1) if all_bayes_var_t.dim() > 1 else all_bayes_var_t).mean().item()
    else:
        all_bayes_probs = None
        laplace_var_mean = None
    if verbose:
        print("Running MC Dropout...")
    all_mcd_probs, all_mcd_var = mc_dropout_predict(model, test_loader, DEVICE, n_samples=MC_DROPOUT_SAMPLES)
    model.eval()
    mcd_var_mean = all_mcd_var.mean().item()
    T_opt = fit_temperature(model, calib_loader, DEVICE)
    all_temp_probs = predict_temperature_scaled(model, test_loader, DEVICE, T_opt)
    if verbose:
        print(f"Temperature Scaling fitted: T = {T_opt:.4f}")
    all_ensemble_probs = None
    if ENSEMBLE_SIZE > 0:
        ensemble_probs_list = []
        for s in range(ENSEMBLE_SIZE):
            torch.manual_seed(seed + 100 + s)
            head = Head().to(DEVICE)
            opt_e = optim.Adam(head.parameters(), lr=LR)
            for _ in range(EPOCHS):
                head.train()
                for embs, labels in train_loader:
                    embs, labels = embs.to(DEVICE), labels.to(DEVICE)
                    opt_e.zero_grad()
                    nn.functional.cross_entropy(head(embs), labels).backward()
                    opt_e.step()
            head.eval()
            probs_e = []
            with torch.no_grad():
                for embs, _ in test_loader:
                    probs_e.append(torch.softmax(head(embs.to(DEVICE)), dim=-1).cpu())
            ensemble_probs_list.append(torch.cat(probs_e))
        all_ensemble_probs = torch.stack(ensemble_probs_list, dim=0).mean(0)
        torch.manual_seed(seed)
    methods = [
        ("Point", compute_metrics(all_true, all_point_probs)),
        ("MC Dropout", compute_metrics(all_true, all_mcd_probs)),
        ("Temp Scaling", compute_metrics(all_true, all_temp_probs)),
    ]
    if all_bayes_probs is not None:
        methods.insert(1, ("Laplace", compute_metrics(all_true, all_bayes_probs)))
    if all_ensemble_probs is not None:
        methods.append(("Ensemble", compute_metrics(all_true, all_ensemble_probs)))
    return methods, laplace_var_mean, mcd_var_mean


def aggregate_runs(all_runs: List[List[Tuple[str, dict]]]) -> List[Tuple[str, dict]]:
    name_to_values = defaultdict(lambda: defaultdict(list))
    for run in all_runs:
        for name, metrics in run:
            for k, v in metrics.items():
                if isinstance(v, float) and (v == v):
                    name_to_values[name][k].append(v)
    out = []
    for name in ["Point", "Laplace", "MC Dropout", "Temp Scaling", "Ensemble"]:
        if name not in name_to_values:
            continue
        agg = {}
        for k in ["accuracy", "brier", "ece", "aurc", "mean_conf_correct", "mean_conf_incorrect", "mean_ent_correct", "mean_ent_incorrect"]:
            if k not in name_to_values[name]:
                continue
            vals = name_to_values[name][k]
            agg[k] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))
        out.append((name, agg))
    return out


# -----------------------------------------------------------------------------
# Main: load data once, run one or multiple seeds, print table
# -----------------------------------------------------------------------------
def main():
    print("Loading SmolLM2-360M...")
    client = SmolLMClient()
    print("Loading QEvasion (CLARITY) train/test...")
    train_data = load_qevasion_clarity("train", max_examples=NUM_TRAIN, samples_per_label=NUM_TRAIN // NUM_CLASSES)
    test_data = load_qevasion_clarity("test", max_examples=NUM_TEST, samples_per_label=NUM_TEST // NUM_CLASSES)
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    c_train = Counter(ex.label for ex in train_data)
    c_test = Counter(ex.label for ex in test_data)
    print(f"Train per class: {dict(c_train)} | Test per class: {dict(c_test)}")

    if NUM_RUNS == 1:
        methods, laplace_var_mean, mcd_var_mean = run_one(42, client, train_data, test_data, verbose=True)
        aggregated = False
    else:
        all_runs_methods = []
        laplace_vars, mcd_vars = [], []
        for r in range(NUM_RUNS):
            print(f"\n--- Run {r+1}/{NUM_RUNS} (seed={42+r}) ---")
            run_methods, lv, mv = run_one(42 + r, client, train_data, test_data, verbose=True)
            all_runs_methods.append(run_methods)
            if lv is not None:
                laplace_vars.append(lv)
            mcd_vars.append(mv)
        methods = aggregate_runs(all_runs_methods)
        aggregated = True
        laplace_var_mean = np.mean(laplace_vars) if laplace_vars else None
        mcd_var_mean = np.mean(mcd_vars)

    def _fmt(v, agg=False):
        if agg and isinstance(v, tuple):
            return f"{v[0]:.4f} ± {v[1]:.4f}"
        if isinstance(v, (int, float)) and (v == v):
            return f"{v:.4f}"
        return "n/a"

    col_w = 18 if aggregated else 14
    n_cols = len(methods)
    header = "  " + "Metric".ljust(26) + "".join(m[0].ljust(col_w) for m in methods)
    print("\n" + "=" * (26 + n_cols * col_w))
    print("CLARITY SemEval UQ: Laplace vs MC Dropout vs Temp Scaling" + (f" (mean ± std over {NUM_RUNS} runs)" if aggregated else ""))
    print("=" * (26 + n_cols * col_w))
    print(header)
    print("  " + "-" * (26 + n_cols * col_w))
    for key, label in [
        ("accuracy", "Accuracy (%)"),
        ("brier", "Brier (lower=better)"),
        ("ece", "ECE (lower=better)"),
        ("aurc", "AURC/AULC (lower=better)"),
        ("mean_conf_correct", "Mean conf (correct)"),
        ("mean_conf_incorrect", "Mean conf (incorrect)"),
        ("mean_ent_correct", "Mean entropy (correct)"),
        ("mean_ent_incorrect", "Mean entropy (incorrect)"),
    ]:
        def cell(m):
            v = m[1].get(key)
            if v is None:
                return "n/a"
            if key == "accuracy":
                return f"{100*v:.2f}%" if not aggregated else f"{100*v[0]:.2f}% ± {100*v[1]:.2f}%"
            return _fmt(v, aggregated)
        row = "  " + label.ljust(26) + "".join(cell(m).ljust(col_w) for m in methods)
        print(row)
    if laplace_var_mean is not None:
        print(f"\n  Laplace predictive var:  mean = {laplace_var_mean:.6f}" + (f" (over {NUM_RUNS} runs)" if aggregated else ""))
    print(f"  MC Dropout predictive var: mean = {mcd_var_mean:.6f}" + (f" (over {NUM_RUNS} runs)" if aggregated else ""))
    print("=" * (26 + n_cols * col_w))
    print("Done. CLARITY 3-way (Direct Reply, Direct Non-Reply, Indirect). Metrics: AURC/AULC, ECE, Brier.")


if __name__ == "__main__":
    main()
