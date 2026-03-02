#!/usr/bin/env python3
"""
QEvasion validation ensemble: RoBERTa (binary DNR) + Granite (3-class) on the SAME QEvasion test set.

No CLI flags; uses fixed paths.
Run from repo root:
  /usr/bin/python3 scripts/run_ensemble_qevasion_validation.py

Expected inputs:
- dataset/qevasion_test_308.csv
    (from clear-non-reply-evaluation-roberta.ipynb save cell; should include index, try_1..try_3)
- results/granite_qevasion_predictions.csv
    (Granite predictions on the same 308 rows; must include index/sample_idx and pred)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]

ROBERTA_QEVASION_CSV = REPO_ROOT / "dataset" / "qevasion_test_308.csv"
GRANITE_QEVASION_CSV = REPO_ROOT / "results" / "granite_qevasion_predictions.csv"
OUT_JSON = REPO_ROOT / "results" / "ensemble_qevasion_metrics.json"

LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent": "Indirect",
    "Ambivalent Reply": "Indirect",
    "Direct Reply": "Direct Reply",
    "Direct Non-Reply": "Direct Non-Reply",
    "Indirect": "Indirect",
}
LABELS = ["Direct Reply", "Direct Non-Reply", "Indirect"]


def map_label(raw: object) -> str:
    v = str(raw).strip()
    return LABEL_MAP.get(v, "Indirect")


def parse_vote_labels(cell: object) -> list[str]:
    if isinstance(cell, list):
        return [map_label(v) for v in cell]
    if isinstance(cell, str):
        s = cell.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [map_label(v) for v in parsed]
            if isinstance(parsed, dict):
                return [map_label(k) for k, n in parsed.items() for _ in range(int(n))]
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [map_label(v) for v in parsed]
            if isinstance(parsed, dict):
                return [map_label(k) for k, n in parsed.items() for _ in range(int(n))]
        except Exception:
            pass
    return []


def compute_metrics(true: list[str], pred: list[str]) -> dict:
    label_to_idx = {l: i for i, l in enumerate(LABELS)}
    cm = [[0 for _ in LABELS] for _ in LABELS]  # rows=true, cols=pred
    correct = 0

    for t, p in zip(true, pred):
        ti, pi = label_to_idx[t], label_to_idx[p]
        cm[ti][pi] += 1
        if t == p:
            correct += 1

    n = max(1, len(true))
    acc = correct / n

    per_class = {}
    f1s = []
    for i, label in enumerate(LABELS):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(LABELS))) - tp
        fn = sum(cm[i][c] for c in range(len(LABELS))) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)

    macro_f1 = sum(f1s) / len(f1s)
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": cm, "per_class": per_class}


def load_roberta_qevasion(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run clear-non-reply-evaluation-roberta.ipynb save cell first to create dataset/qevasion_test_308.csv"
        )
    df = pd.read_csv(path)

    if "index" not in df.columns:
        df = df.reset_index(drop=True)
        df["index"] = df.index

    required_votes = ["try_1", "try_2", "try_3"]
    missing_votes = [c for c in required_votes if c not in df.columns]
    if missing_votes:
        raise ValueError(f"{path} missing vote columns: {missing_votes}")

    if "roberta_nonreply_majority" not in df.columns:
        df["roberta_nonreply_majority"] = (df[required_votes].sum(axis=1) >= 2).astype(int)

    if "clarity_label" not in df.columns:
        raise ValueError(f"{path} must include clarity_label for 3-class evaluation")

    gold = [map_label(v) for v in df["clarity_label"].tolist()]
    return df, gold


def load_granite_qevasion(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Export Granite predictions on QEvasion test to this file first (columns: index/sample_idx, pred)."
        )
    gdf = pd.read_csv(path)

    if "sample_idx" not in gdf.columns:
        if "index" in gdf.columns:
            gdf = gdf.rename(columns={"index": "sample_idx"})
        else:
            raise ValueError(f"{path} must include 'sample_idx' or 'index'")

    if "pred" not in gdf.columns:
        for alt in ["prediction", "majority_wins", "label"]:
            if alt in gdf.columns:
                gdf = gdf.rename(columns={alt: "pred"})
                break
    if "pred" not in gdf.columns:
        raise ValueError(f"{path} must include prediction column: pred/prediction/majority_wins/label")

    if "vote_labels" not in gdf.columns:
        if "votes" in gdf.columns:
            gdf["vote_labels"] = gdf["votes"]
        else:
            gdf["vote_labels"] = ""

    gdf["sample_idx"] = pd.to_numeric(gdf["sample_idx"], errors="coerce")
    gdf = gdf[gdf["sample_idx"].notna()].copy()
    gdf["sample_idx"] = gdf["sample_idx"].astype(int)
    gdf["pred"] = gdf["pred"].map(map_label)

    return gdf.drop_duplicates(subset=["sample_idx"], keep="last").sort_values("sample_idx").reset_index(drop=True)


def fuse_sum_both_balanced_v4(r_row: pd.Series, g_row: pd.Series) -> str:
    g_votes = parse_vote_labels(g_row.get("vote_labels", ""))
    g_dnr_votes = int(sum(1 for v in g_votes if v == "Direct Non-Reply"))
    g_dr_votes = int(sum(1 for v in g_votes if v == "Direct Reply"))
    g_ind_votes = int(sum(1 for v in g_votes if v == "Indirect"))

    # Fallback when no Granite vote list is provided: treat Granite pred as 3/3 votes.
    if not g_votes:
        g_major = map_label(g_row.get("pred", "Indirect"))
        if g_major == "Direct Reply":
            g_dr_votes = 3
        elif g_major == "Direct Non-Reply":
            g_dnr_votes = 3
        else:
            g_ind_votes = 3

    r_vote_count = int(r_row.get("try_1", 0)) + int(r_row.get("try_2", 0)) + int(r_row.get("try_3", 0))

    score_dnr = (r_vote_count + g_dnr_votes) / 2.0
    score_dr = float(g_dr_votes)
    score_ind = float(g_ind_votes) + 1.0

    if score_dnr >= 1.5 and score_dnr >= score_dr and score_dnr >= score_ind:
        return "Direct Non-Reply"
    if score_dr >= score_ind:
        return "Direct Reply"
    return "Indirect"


def main() -> None:
    roberta_df, gold = load_roberta_qevasion(ROBERTA_QEVASION_CSV)
    granite_df = load_granite_qevasion(GRANITE_QEVASION_CSV)

    rmap = roberta_df.set_index("index")
    gmap = granite_df.set_index("sample_idx")

    common_idx = sorted(set(rmap.index).intersection(set(gmap.index)))
    if not common_idx:
        raise ValueError("No overlapping indices between RoBERTa QEvasion CSV and Granite QEvasion CSV")

    if len(common_idx) < len(roberta_df):
        print(f"Note: evaluating on {len(common_idx)}/{len(roberta_df)} rows due to overlap")

    pred = []
    true_used = []
    for i in common_idx:
        r = rmap.loc[i]
        g = gmap.loc[i]
        pred.append(fuse_sum_both_balanced_v4(r, g))
        true_used.append(gold[i])

    metrics = compute_metrics(true_used, pred)
    metrics["n_evaluated"] = len(common_idx)
    metrics["n_total_roberta_rows"] = len(roberta_df)
    metrics["fusion_rule"] = "sum_both_balanced_v4"

    print("QEvasion Ensemble (RoBERTa + Granite)")
    print(f"  N: {metrics['n_evaluated']}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Macro F1: {metrics['macro_f1']:.4f}")
    for label in LABELS:
        pc = metrics["per_class"][label]
        print(f"  {label}: P={pc['precision']:.4f} R={pc['recall']:.4f} F1={pc['f1']:.4f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
