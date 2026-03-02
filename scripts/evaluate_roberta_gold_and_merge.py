#!/usr/bin/env python3
"""
Evaluate RoBERTa on SemEval gold eval and optionally merge with Granite predictions.

Both models vote 3 times. RoBERTa: 0-3 votes for Direct Non-Reply (binary). Granite: 0-3 for each of
DR, DNR, Indirect (sum=3). Best balance: sum_both_balanced_v4 (macro F1 ~0.49, DR F1 >0.5, Indirect >0.4;
DNR stays ~0.39 with this ensemble).
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LABEL_MAP = {
    "Clear Reply": "Direct Reply",
    "Clear Non-Reply": "Direct Non-Reply",
    "Ambivalent": "Indirect",
    "Ambivalent Reply": "Indirect",
}
LABELS = ["Direct Reply", "Direct Non-Reply", "Indirect"]


def map_label(raw: str) -> str:
    v = str(raw).strip()
    return LABEL_MAP.get(v, v if v in LABELS else "Indirect")


def first_existing(paths: List[str]) -> Optional[Path]:
    for p in paths:
        path = Path(p)
        if path.exists():
            return path
    return None


def compute_binary_dnr_metrics(gold_labels: List[str], pred_binary: List[int]) -> Dict:
    """RoBERTa is binary (DNR vs rest). Compute DNR precision, recall, F1.

    If gold_labels are ints/bools (0/1), use them directly (as in notebook-style binary labels).
    Otherwise, map CLARITY labels to DNR vs rest.
    """
    assert len(gold_labels) == len(pred_binary)
    gold_bin: List[int] = []
    for g in gold_labels:
        if isinstance(g, (int, bool)):
            gold_bin.append(int(g))
        else:
            gold_bin.append(1 if map_label(g) == "Direct Non-Reply" else 0)
    tp = sum(1 for g, p in zip(gold_bin, pred_binary) if g == 1 and p == 1)
    fp = sum(1 for g, p in zip(gold_bin, pred_binary) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold_bin, pred_binary) if g == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_metrics(true: List[str], pred: List[str]) -> Dict:
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


def load_gold(eval_csv: Path, labels_txt: Path) -> Tuple[pd.DataFrame, List[str]]:
    with labels_txt.open("r", encoding="utf-8") as f:
        gold = [map_label(line.strip()) for line in f if line.strip()]

    with eval_csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    records = []
    for i, r in enumerate(rows):
        idx_raw = str(r.get("index", "")).strip()
        idx = int(idx_raw) if idx_raw.isdigit() else i
        q = str(r.get("question") or r.get("interview_question") or "").strip()
        a = str(r.get("interview_answer") or r.get("answer") or "").strip()
        records.append({"index": idx, "question": q, "answer": a})
    df = pd.DataFrame(records).sort_values("index").reset_index(drop=True)

    if len(df) != len(gold):
        raise ValueError(f"Mismatch: eval rows={len(df)} vs gold labels={len(gold)}")
    if int(df["index"].min()) != 0 or int(df["index"].max()) != (len(df) - 1):
        raise ValueError("Expected contiguous index range 0..N-1 in eval CSV")
    return df, gold


def infer_positive_class_id(model) -> int:
    id2label = getattr(model.config, "id2label", None) or {}
    for k, v in id2label.items():
        if "non" in str(v).lower() and "reply" in str(v).lower():
            return int(k)
    # fallback for binary classifier
    if getattr(model.config, "num_labels", 2) == 2:
        return 1
    return 0


def roberta_predict_votes(
    eval_df: pd.DataFrame,
    model_path: str,
    max_length: int,
    batch_size: int,
    num_votes: int,
    positive_class_id: Optional[int],
) -> pd.DataFrame:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    pos_id = positive_class_id if positive_class_id is not None else infer_positive_class_id(model)

    def predict_once() -> List[int]:
        preds: List[int] = []
        for s in range(0, len(eval_df), batch_size):
            batch = eval_df.iloc[s : s + batch_size]
            enc = tokenizer(
                batch["question"].tolist(),
                batch["answer"].tolist(),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = model(**enc).logits
                cls = torch.argmax(out, dim=-1).detach().cpu().tolist()
            preds.extend([1 if int(c) == int(pos_id) else 0 for c in cls])
        return preds

    vote_cols = {}
    for i in range(num_votes):
        vote_cols[f"try_{i+1}"] = predict_once()

    out = eval_df[["index"]].copy()
    for k, v in vote_cols.items():
        out[k] = v
    out["roberta_nonreply_majority"] = (
        out[[f"try_{i+1}" for i in range(num_votes)]].sum(axis=1) >= ((num_votes // 2) + 1)
    ).astype(int)
    return out


def parse_granite_votes(votes_cell) -> List[str]:
    if isinstance(votes_cell, list):
        return [map_label(v) for v in votes_cell]
    if isinstance(votes_cell, str):
        s = votes_cell.strip()
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


def merge_with_granite(
    gold: List[str],
    roberta_votes_df: pd.DataFrame,
    granite_df: pd.DataFrame,
    fusion_rule: str,
) -> Tuple[pd.DataFrame, Dict]:
    gdf = granite_df.copy()
    if "sample_idx" not in gdf.columns:
        raise ValueError("Granite dataframe must include sample_idx")
    if "index" not in roberta_votes_df.columns:
        roberta_votes_df = roberta_votes_df.copy()
        roberta_votes_df["index"] = list(range(len(roberta_votes_df)))

    gdf["sample_idx"] = gdf["sample_idx"].astype(int)
    rdf = roberta_votes_df.copy()
    rdf["index"] = rdf["index"].astype(int)

    gmap = gdf.drop_duplicates(subset=["sample_idx"], keep="last").set_index("sample_idx")
    rmap = rdf.drop_duplicates(subset=["index"], keep="last").set_index("index")

    common_idx = sorted(set(rmap.index).intersection(set(gmap.index)))
    if not common_idx:
        raise ValueError("No overlapping indices between RoBERTa votes and Granite predictions.")
    if len(common_idx) < len(gold):
        missing = sorted(set(range(len(gold))) - set(common_idx))
        print(
            f"⚠️ Partial overlap: evaluating on {len(common_idx)}/{len(gold)} samples. "
            f"Missing indices: {missing[:20]}{'...' if len(missing) > 20 else ''}"
        )

    pred = []
    true_used = []
    rows = []
    for i in common_idx:
        r = rmap.loc[i]
        g = gmap.loc[i]
        true = gold[i]

        g_votes = parse_granite_votes(g.get("vote_labels", g.get("votes", "")))
        g_major = map_label(g.get("pred", "Indirect"))
        r_nonreply = int(r["roberta_nonreply_majority"])
        r_vote_count = int(r.get("try_1", 0)) + int(r.get("try_2", 0)) + int(r.get("try_3", 0))
        g_dnr_votes = int(sum(1 for v in g_votes if v == "Direct Non-Reply"))
        g_dr_votes = int(sum(1 for v in g_votes if v == "Direct Reply"))
        g_ind_votes = int(sum(1 for v in g_votes if v == "Indirect"))

        # fusion rules
        if fusion_rule == "aggressive_or":
            # Original aggressive behavior (often over-predicts Direct Non-Reply)
            final = "Direct Non-Reply" if (r_nonreply == 1 or ("Direct Non-Reply" in g_votes)) else g_major
        elif fusion_rule == "max_compare":
            # User-requested style: compare non-reply vote strengths.
            # If RoBERTa non-reply votes exceed Granite non-reply votes and are majority, choose non-reply.
            # Otherwise keep Granite majority.
            if r_vote_count > g_dnr_votes and r_vote_count >= 2:
                final = "Direct Non-Reply"
            elif r_vote_count < g_dnr_votes and g_dnr_votes >= 2:
                final = "Direct Non-Reply"
            else:
                final = g_major
        elif fusion_rule == "conservative":
            # Safer: only force non-reply on strong non-reply signal
            final = "Direct Non-Reply" if (r_vote_count == 3 or g_dnr_votes >= 2) else g_major
        elif fusion_rule == "dnr_guardrail":
            # Keep Granite generally, but guard against Granite's over-prediction of Direct Non-Reply.
            # If Granite predicts DNR while RoBERTa strongly says NOT non-reply, switch to Granite's
            # non-DNR preference (Direct Reply vs Indirect) based on Granite vote counts.
            if g_major == "Direct Non-Reply" and r_nonreply == 0:
                g_dr_votes = int(sum(1 for v in g_votes if v == "Direct Reply"))
                g_ind_votes = int(sum(1 for v in g_votes if v == "Indirect"))
                final = "Direct Reply" if g_dr_votes > g_ind_votes else "Indirect"
            else:
                final = g_major
        elif fusion_rule == "sum_both_votes":
            # Both vote 3 times. RoBERTa: 0-3 votes for DNR. Granite: 0-3 for DR, DNR, Ind (sum=3).
            # Combined DNR = RoBERTa_DNR + Granite_DNR (0-6). Normalize so each class on 0-3 scale, then argmax.
            score_dnr = (r_vote_count + g_dnr_votes) / 2.0  # 0 to 3
            score_dr = float(g_dr_votes)   # 0 to 3
            score_ind = float(g_ind_votes)  # 0 to 3
            if score_dnr >= score_dr and score_dnr >= score_ind:
                final = "Direct Non-Reply"
            elif score_dr >= score_ind:
                final = "Direct Reply"
            else:
                final = "Indirect"
        elif fusion_rule == "sum_both_votes_indirect_plus1":
            # Same as sum_both_votes but give Indirect +1 to balance under-prediction.
            score_dnr = (r_vote_count + g_dnr_votes) / 2.0
            score_dr = float(g_dr_votes)
            score_ind = float(g_ind_votes) + 1.0
            if score_dnr >= score_dr and score_dnr >= score_ind:
                final = "Direct Non-Reply"
            elif score_dr >= score_ind:
                final = "Direct Reply"
            else:
                final = "Indirect"
        elif fusion_rule == "sum_both_balanced":
            # DNR and DR only when both models agree (so F1 > 0.5 on clear classes). Indirect gets the rest.
            # DNR: only when BOTH RoBERTa and Granite clearly say DNR (>=2 each) → high precision DNR.
            # DR vs Ind: summed votes with Indirect +1 so ambiguous cases lean Indirect.
            if r_vote_count >= 2 and g_dnr_votes >= 2:
                final = "Direct Non-Reply"
            else:
                score_dr = float(g_dr_votes)
                score_ind = float(g_ind_votes) + 1.0
                final = "Direct Reply" if score_dr >= score_ind else "Indirect"
        elif fusion_rule == "sum_both_balanced_v2":
            # DNR when combined evidence strong (r+g_dnr >= 4) so we get both precision and recall. Else DR vs Ind with Indirect +1.
            combined_dnr = r_vote_count + g_dnr_votes
            if combined_dnr >= 4:
                final = "Direct Non-Reply"
            else:
                score_dr = float(g_dr_votes)
                score_ind = float(g_ind_votes) + 1.0
                final = "Direct Reply" if score_dr >= score_ind else "Indirect"
        elif fusion_rule == "sum_both_balanced_v3":
            # DNR when RoBERTa says DNR (2+) AND Granite doesn't strongly say DR (g_dr < 2) → fewer false DNRs on clear DR.
            # Else: DR vs Ind with Indirect +1. Goal: DR and DNR both F1 > 0.5.
            if r_vote_count >= 2 and g_dr_votes < 2:
                final = "Direct Non-Reply"
            else:
                score_dr = float(g_dr_votes)
                score_ind = float(g_ind_votes) + 1.0
                final = "Direct Reply" if score_dr >= score_ind else "Indirect"
        elif fusion_rule == "sum_both_balanced_v4":
            # All three from summed votes: normalize DNR to 0-3, DR and Ind+1. DNR only when it wins AND has clear support (>= 1.5).
            # So we need 3+ combined DNR votes and DNR must beat both DR and (Ind+1). Balances all three.
            score_dnr = (r_vote_count + g_dnr_votes) / 2.0
            score_dr = float(g_dr_votes)
            score_ind = float(g_ind_votes) + 1.0
            if score_dnr >= 1.5 and score_dnr >= score_dr and score_dnr >= score_ind:
                final = "Direct Non-Reply"
            elif score_dr >= score_ind:
                final = "Direct Reply"
            else:
                final = "Indirect"
        elif fusion_rule == "sum_both_balanced_v5":
            # Slightly higher DNR bar (1.75 = 3.5 combined) to boost DNR precision; DR gets +0.5 when RoBERTa says no DNR.
            score_dnr = (r_vote_count + g_dnr_votes) / 2.0
            score_dr = float(g_dr_votes) + (0.5 if r_vote_count == 0 else 0.0)  # bonus when RoBERTa says "reply"
            score_ind = float(g_ind_votes) + 1.0
            if score_dnr >= 1.75 and score_dnr >= score_dr and score_dnr >= score_ind:
                final = "Direct Non-Reply"
            elif score_dr >= score_ind:
                final = "Direct Reply"
            else:
                final = "Indirect"
        elif fusion_rule == "trust_specialist":
            # Granite has DR F1 > 0.5, RoBERTa has NDR F1 > 0.5. Trust each for its strong class.
            # DNR only when RoBERTa says DNR (majority). Else use Granite (DR/Ind); when Granite says DNR resolve by votes + Ind+1.
            if r_nonreply == 1:
                final = "Direct Non-Reply"
            else:
                if g_major == "Direct Non-Reply":
                    score_dr = float(g_dr_votes)
                    score_ind = float(g_ind_votes) + 1.0
                    final = "Direct Reply" if score_dr >= score_ind else "Indirect"
                else:
                    final = g_major
        elif fusion_rule == "roberta_dnr_then_granite":
            # RoBERTa "owns" Direct Non-Reply: only predict DNR when RoBERTa majority says so.
            # Otherwise use Granite (no DNR from Granite alone).
            if r_nonreply == 1:
                final = "Direct Non-Reply"
            else:
                # Granite decides between Direct Reply and Indirect only (ignore Granite DNR when RoBERTa says no).
                if g_major == "Direct Non-Reply":
                    g_dr_votes = int(sum(1 for v in g_votes if v == "Direct Reply"))
                    g_ind_votes = int(sum(1 for v in g_votes if v == "Indirect"))
                    final = "Direct Reply" if g_dr_votes >= g_ind_votes else "Indirect"
                else:
                    final = g_major
        elif fusion_rule == "roberta_dnr_then_granite_indirect_plus1":
            # Same as roberta_dnr_then_granite, but when choosing DR vs Indirect give Indirect +1 weight.
            if r_nonreply == 1:
                final = "Direct Non-Reply"
            else:
                if g_major == "Direct Non-Reply":
                    g_dr_votes = int(sum(1 for v in g_votes if v == "Direct Reply"))
                    g_ind_votes = int(sum(1 for v in g_votes if v == "Indirect"))
                    # Indirect gets +1 to balance under-prediction
                    final = "Direct Reply" if g_dr_votes > (g_ind_votes + 1) else "Indirect"
                else:
                    g_dr_votes = int(sum(1 for v in g_votes if v == "Direct Reply"))
                    g_ind_votes = int(sum(1 for v in g_votes if v == "Indirect"))
                    # Weighted: equal weight or Indirect +1
                    final = "Direct Reply" if g_dr_votes > (g_ind_votes + 1) else "Indirect"
        else:
            # granite_only
            final = g_major

        pred.append(final)
        true_used.append(true)
        rows.append(
            {
                "index": i,
                "true": true,
                "pred": final,
                "granite_pred": g_major,
                "granite_votes": g_votes,
                "roberta_nonreply_majority": r_nonreply,
                "roberta_nonreply_vote_count": r_vote_count,
                "granite_nonreply_vote_count": g_dnr_votes,
                "fusion_rule": fusion_rule,
            }
        )

    metrics = compute_metrics(true_used, pred)
    metrics["n_evaluated"] = len(common_idx)
    metrics["n_gold_total"] = len(gold)
    return pd.DataFrame(rows), metrics


def load_granite_from_log(granite_log_txt: Path) -> pd.DataFrame:
    pat = re.compile(
        r"^\[(\d{3})/(\d+)\]\s+True=([^|]+)\s+\|\s+Pred=([^|]+)\s+\|\s+[^|]*\|\s+votes=(\{.*\})\s+\|\s+max_toks=(\d+)\s*$"
    )
    rows = []
    with granite_log_txt.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line.strip())
            if not m:
                continue
            idx1 = int(m.group(1))
            pred = map_label(m.group(4).strip())
            votes_raw = m.group(5).strip()
            votes_dict = {}
            try:
                votes_dict = ast.literal_eval(votes_raw)
                if not isinstance(votes_dict, dict):
                    votes_dict = {}
            except Exception:
                votes_dict = {}
            vote_labels: List[str] = []
            for k, n in votes_dict.items():
                try:
                    vote_labels.extend([map_label(k)] * int(n))
                except Exception:
                    pass
            rows.append({"sample_idx": idx1 - 1, "pred": pred, "vote_labels": vote_labels})

    if not rows:
        raise ValueError(f"No parsable prediction lines found in granite log: {granite_log_txt}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["sample_idx"], keep="last")
    return df.sort_values("sample_idx").reset_index(drop=True)


def load_granite_predictions(granite_pred_csv: Optional[str], granite_log_txt: Optional[str]) -> pd.DataFrame:
    if granite_pred_csv:
        gdf = pd.read_csv(granite_pred_csv)
        if "sample_idx" not in gdf.columns:
            raise ValueError("Granite predictions CSV must include 'sample_idx'")
        if "pred" not in gdf.columns:
            raise ValueError("Granite predictions CSV must include 'pred'")
        if "vote_labels" not in gdf.columns and "votes" in gdf.columns:
            gdf["vote_labels"] = gdf["votes"]
        if "vote_labels" not in gdf.columns:
            gdf["vote_labels"] = ""
        return gdf
    if granite_log_txt:
        return load_granite_from_log(Path(granite_log_txt))
    raise ValueError("Provide either --granite-predictions-csv or --granite-log-txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roberta-model-path", default=".", help="Path to RoBERTa classifier")
    parser.add_argument("--positive-class-id", type=int, default=None, help="Optional explicit non-reply class id")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-votes", type=int, default=3)

    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--labels-txt", default=None)
    parser.add_argument("--granite-predictions-csv", default=None, help="gold_eval_predictions.csv from Granite run")
    parser.add_argument("--granite-log-txt", default=None, help="granite_answers.txt style log from Gold eval")
    parser.add_argument(
        "--fusion-rule",
        default="sum_both_balanced_v4",
        choices=["granite_only", "conservative", "max_compare", "aggressive_or", "dnr_guardrail", "trust_specialist", "sum_both_votes", "sum_both_votes_indirect_plus1", "sum_both_balanced", "sum_both_balanced_v2", "sum_both_balanced_v3", "sum_both_balanced_v4", "sum_both_balanced_v5", "roberta_dnr_then_granite", "roberta_dnr_then_granite_indirect_plus1"],
        help="Fusion strategy for combining RoBERTa non-reply and Granite 3-class outputs",
    )
    parser.add_argument(
        "--roberta-votes-csv",
        default=None,
        help="Use existing RoBERTa votes CSV (must include index, try_1..try_3 or roberta_nonreply_majority)",
    )
    parser.add_argument("--check-only", action="store_true", help="Only validate input alignment and exit")
    parser.add_argument(
        "--eval-roberta-on-balanced",
        action="store_true",
        help="Run RoBERTa on the balanced QEvasion test set (69 samples, like evaluate_roberta_ensemble) and report binary DNR F1",
    )
    parser.add_argument("--output-dir", default="results/roberta_gold_eval")
    args = parser.parse_args()

    if args.eval_roberta_on_balanced:
        eval_csv = Path(args.eval_csv) if args.eval_csv else first_existing([
            "dataset/clarity_task_evaluation_dataset.csv",
            str(REPO_ROOT / "dataset/clarity_task_evaluation_dataset.csv"),
        ])
        labels_txt = Path(args.labels_txt) if args.labels_txt else first_existing([
            "/Users/andrearachetta/Downloads/task1_eval_labels.txt",
            "task1_eval_labels.txt",
            "dataset/task1_eval_labels.txt",
            str(REPO_ROOT / "dataset/task1_eval_labels.txt"),
        ])
        if eval_csv is None or not eval_csv.exists():
            raise FileNotFoundError("Need --eval-csv for --eval-roberta-on-balanced.")
        if labels_txt and labels_txt.exists():
            eval_df_full, gold_full = load_gold(eval_csv, labels_txt)
        else:
            eval_df_full = pd.read_csv(eval_csv)
            eval_df_full["question"] = eval_df_full.get("interview_question", eval_df_full.get("question", ""))
            eval_df_full["answer"] = eval_df_full.get("interview_answer", eval_df_full.get("answer", ""))
            eval_df_full["index"] = range(len(eval_df_full))
            gold_full = [map_label(str(r.get("clarity_label", "Indirect"))) for _, r in eval_df_full.iterrows()]
        # Stratified sample: 23 per class (69 total)
        rng = random.Random(42)
        by_label = {}
        for i, g in enumerate(gold_full):
            by_label.setdefault(g, []).append(i)
        n_per = 23
        idx = []
        for g in LABELS:
            pool = by_label.get(g, [])
            idx.extend(rng.sample(pool, min(n_per, len(pool))))
        rng.shuffle(idx)
        gold = [gold_full[i] for i in idx]
        eval_set_name = "SemEval_gold_balanced69"

        if args.roberta_votes_csv and Path(args.roberta_votes_csv).exists():
            # Use existing votes: subset to the 69 indices
            roberta_votes = pd.read_csv(args.roberta_votes_csv)
            if "index" not in roberta_votes.columns:
                raise ValueError("roberta_votes_csv must have 'index' column")
            if "roberta_nonreply_majority" not in roberta_votes.columns:
                nv = max(1, int(args.num_votes))
                req = [f"try_{i+1}" for i in range(nv)]
                roberta_votes["roberta_nonreply_majority"] = (roberta_votes[req].sum(axis=1) >= (nv // 2 + 1)).astype(int)
            rmap = roberta_votes.set_index("index")
            pred_bin = [int(rmap.loc[i, "roberta_nonreply_majority"]) for i in idx if i in rmap.index]
            # Prefer notebook-style binary gold if available; otherwise fall back to CLARITY labels
            if "is_clear_non_reply_label" in rmap.columns:
                gold_aligned = [int(rmap.loc[i, "is_clear_non_reply_label"]) for i in idx if i in rmap.index]
            else:
                gold_aligned = [gold[j] for j, i in enumerate(idx) if i in rmap.index]
            if len(pred_bin) != len(gold_aligned) or len(pred_bin) < 60:
                print(f"Warning: only {len(pred_bin)} of {len(idx)} indices found in votes; using all available.")
            if pred_bin and gold_aligned:
                roberta_bin = compute_binary_dnr_metrics(gold_aligned, pred_bin)
                print(f"RoBERTa binary DNR on balanced set (n={len(gold_aligned)}, {eval_set_name}, from existing votes): F1={roberta_bin['f1']:.3f} P={roberta_bin['precision']:.3f} R={roberta_bin['recall']:.3f}")
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                with (out_dir / "roberta_balanced69_binary_dnr_metrics.json").open("w", encoding="utf-8") as f:
                    json.dump({"roberta_binary_dnr": roberta_bin, "n": len(gold_aligned), "eval_set": eval_set_name}, f, indent=2)
                print(f"Saved: {out_dir / 'roberta_balanced69_binary_dnr_metrics.json'}")
            return
        # Run RoBERTa model on the 69 samples
        eval_df = eval_df_full.iloc[idx].copy().reset_index(drop=True)
        eval_df["index"] = range(len(eval_df))
        roberta_votes = roberta_predict_votes(
            eval_df=eval_df,
            model_path=args.roberta_model_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_votes=max(1, int(args.num_votes)),
            positive_class_id=args.positive_class_id,
        )
        pred_bin = roberta_votes["roberta_nonreply_majority"].astype(int).tolist()
        roberta_bin = compute_binary_dnr_metrics(gold, pred_bin)
        print(f"RoBERTa binary DNR on balanced set (n={len(gold)}, {eval_set_name}): F1={roberta_bin['f1']:.3f} P={roberta_bin['precision']:.3f} R={roberta_bin['recall']:.3f}")
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "roberta_balanced69_binary_dnr_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"roberta_binary_dnr": roberta_bin, "n": len(gold), "eval_set": eval_set_name}, f, indent=2)
        print(f"Saved: {out_dir / 'roberta_balanced69_binary_dnr_metrics.json'}")
        return

    eval_csv = Path(args.eval_csv) if args.eval_csv else first_existing(
        [
            "/kaggle/input/datasets/gigibot/claritysemevalevaldataset/clarity_task_evaluation_dataset.csv",
            "/kaggle/input/claritysemevalevaldataset/clarity_task_evaluation_dataset.csv",
            "dataset/clarity_task_evaluation_dataset.csv",
        ]
    )
    labels_txt = Path(args.labels_txt) if args.labels_txt else first_existing(
        [
            "/kaggle/input/datasets/gigibot/claritysemevalevaldataset/task1_eval_labels.txt",
            "/kaggle/input/claritysemevalevaldataset/task1_eval_labels.txt",
            "/Users/andrearachetta/Downloads/task1_eval_labels.txt",
            "task1_eval_labels.txt",
            "dataset/task1_eval_labels.txt",
        ]
    )
    if eval_csv is None or labels_txt is None:
        raise FileNotFoundError("Could not resolve eval CSV / labels TXT.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_df, gold = load_gold(eval_csv, labels_txt)
    print(f"Alignment OK: {len(eval_df)} samples")

    if args.check_only:
        print("Check-only mode complete.")
        return

    if args.roberta_votes_csv:
        roberta_votes = pd.read_csv(args.roberta_votes_csv).sort_values("index").reset_index(drop=True)
        if len(roberta_votes) != len(eval_df):
            # Try alignment by index (local 308-row roberta vs 237-row gold)
            if "index" not in roberta_votes.columns:
                raise ValueError(
                    f"Provided --roberta-votes-csv rows ({len(roberta_votes)}) do not match eval rows ({len(eval_df)}), "
                    "and no 'index' column is available for alignment."
                )
            keep_idx = set(eval_df["index"].tolist())
            roberta_votes = roberta_votes[roberta_votes["index"].isin(keep_idx)].copy()
            roberta_votes = roberta_votes.sort_values("index").reset_index(drop=True)
            if len(roberta_votes) != len(eval_df):
                raise ValueError(
                    f"After index alignment, RoBERTa rows ({len(roberta_votes)}) still do not match eval rows ({len(eval_df)})."
                )
        if "roberta_nonreply_majority" not in roberta_votes.columns:
            req = [f"try_{i+1}" for i in range(max(1, int(args.num_votes)))]
            missing = [c for c in req if c not in roberta_votes.columns]
            if missing:
                raise ValueError(
                    "roberta_nonreply_majority missing and vote cols missing: "
                    f"{missing}. Provide a compatible votes CSV."
                )
            roberta_votes["roberta_nonreply_majority"] = (
                roberta_votes[req].sum(axis=1) >= ((len(req) // 2) + 1)
            ).astype(int)
        print(f"Loaded existing RoBERTa votes: {args.roberta_votes_csv}")
    else:
        roberta_votes = roberta_predict_votes(
            eval_df=eval_df,
            model_path=args.roberta_model_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_votes=max(1, int(args.num_votes)),
            positive_class_id=args.positive_class_id,
        )
    roberta_votes.to_csv(out_dir / "roberta_gold_votes.csv", index=False)
    print(f"Saved: {out_dir / 'roberta_gold_votes.csv'}")

    # RoBERTa is binary (DNR vs rest). Report its DNR F1 on this eval set so we see why 3-way DNR can drop.
    roberta_bin = None
    rmap = roberta_votes.set_index("index")
    pred_bin = [int(rmap.loc[idx, "roberta_nonreply_majority"]) for idx in eval_df["index"] if idx in rmap.index]
    gold_aligned = [gold[i] for i, idx in enumerate(eval_df["index"]) if idx in rmap.index]
    if pred_bin and len(pred_bin) == len(gold_aligned):
        roberta_bin = compute_binary_dnr_metrics(gold_aligned, pred_bin)
        print(
            f"RoBERTa binary DNR on this set (n={len(pred_bin)}): "
            f"F1={roberta_bin['f1']:.3f} P={roberta_bin['precision']:.3f} R={roberta_bin['recall']:.3f}"
        )
    if args.granite_predictions_csv:
        granite_df = load_granite_predictions(args.granite_predictions_csv, args.granite_log_txt)
        merged_df, metrics = merge_with_granite(
            gold=gold,
            roberta_votes_df=roberta_votes,
            granite_df=granite_df,
            fusion_rule=args.fusion_rule,
        )
        if roberta_bin is not None:
            metrics["roberta_binary_dnr"] = roberta_bin
        merged_df.to_csv(out_dir / "merged_ensemble_predictions.csv", index=False)
        with (out_dir / "merged_ensemble_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("\nMerged ensemble metrics:")
        print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))
        print(f"Saved: {out_dir / 'merged_ensemble_predictions.csv'}")
        print(f"Saved: {out_dir / 'merged_ensemble_metrics.json'}")
    elif args.granite_log_txt:
        granite_df = load_granite_predictions(args.granite_predictions_csv, args.granite_log_txt)
        merged_df, metrics = merge_with_granite(
            gold=gold,
            roberta_votes_df=roberta_votes,
            granite_df=granite_df,
            fusion_rule=args.fusion_rule,
        )
        if roberta_bin is not None:
            metrics["roberta_binary_dnr"] = roberta_bin
        merged_df.to_csv(out_dir / "merged_ensemble_predictions.csv", index=False)
        with (out_dir / "merged_ensemble_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("\nMerged ensemble metrics:")
        print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))
        print(f"Saved: {out_dir / 'merged_ensemble_predictions.csv'}")
        print(f"Saved: {out_dir / 'merged_ensemble_metrics.json'}")
    else:
        print("\nNo --granite-predictions-csv provided; only RoBERTa gold votes were generated.")


if __name__ == "__main__":
    main()

