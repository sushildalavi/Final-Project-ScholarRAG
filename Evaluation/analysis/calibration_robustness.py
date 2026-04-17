"""Leakage-aware calibration analysis with bootstrap CIs, reliability diagrams, PR curves.

Addresses three flaws in the original ablation report:

1. The `msa_A` column in claim_scores_scored.csv equals the binary support label
   (mean=1.0 std=0.0 for supported, mean=0.0 std=0.0 for unsupported). Any model
   that uses A trivially hits 100% accuracy — A leaks the label.
2. Reporting scalar Brier/ECE on a 243-row test set with a 9.4:1 class ratio
   gives no sense of variance. We add non-parametric bootstrap CIs.
3. Points are split by query-id, but claims from the same paper can still appear
   in both train and test. We add a paper-grouped split.

Run:
    python Evaluation/analysis/calibration_robustness.py \
        --claims Evaluation/data/human_labels/claim_scores_scored.csv \
        --out-dir Evaluation/data/robustness \
        --fig-dir Evaluation/figures/robustness
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


FEATURE_SETS: dict[str, list[str]] = {
    "M-only": ["msa_M"],
    "S-only": ["msa_S"],
    "M+S": ["msa_M", "msa_S"],
    "M+S+A(leak)": ["msa_M", "msa_S", "msa_A"],
}

DOC_ID_RE = re.compile(r"uploaded:(\d+):")


@dataclass
class FoldResult:
    model: str
    fold: int
    n_test: int
    n_pos: int
    n_neg: int
    accuracy: float
    f1_macro: float
    brier: float
    ece: float
    roc_auc: float
    pr_auc: float
    y_true: np.ndarray
    y_prob: np.ndarray


def extract_doc_id(evidence_id: str) -> str:
    if not isinstance(evidence_id, str):
        return "unknown"
    m = DOC_ID_RE.search(evidence_id)
    return m.group(1) if m else "unknown"


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(conf - acc)
    return float(ece)


def fit_predict(
    df: pd.DataFrame,
    features: list[str],
    splits: list[tuple[np.ndarray, np.ndarray]],
) -> list[FoldResult]:
    X = df[features].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=int)
    results: list[FoldResult] = []

    for fold, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(X[tr], y[tr])
        y_prob = model.predict_proba(X[te])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results.append(
            FoldResult(
                model="__tmp__",
                fold=fold,
                n_test=len(te),
                n_pos=int(y[te].sum()),
                n_neg=int((1 - y[te]).sum()),
                accuracy=float((y_pred == y[te]).mean()),
                f1_macro=float(f1_score(y[te], y_pred, average="macro", zero_division=0)),
                brier=float(brier_score_loss(y[te], y_prob)),
                ece=expected_calibration_error(y[te], y_prob),
                roc_auc=float(roc_auc_score(y[te], y_prob)),
                pr_auc=float(average_precision_score(y[te], y_prob)),
                y_true=y[te].copy(),
                y_prob=y_prob.copy(),
            )
        )
    return results


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> dict:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return {"mean": float(arr.mean()), "lo": lo, "hi": hi, "n": int(arr.size)}


def aggregate(model_name: str, folds: list[FoldResult]) -> dict:
    if not folds:
        return {"model": model_name, "folds": 0}
    metrics = ["accuracy", "f1_macro", "brier", "ece", "roc_auc", "pr_auc"]
    out: dict = {"model": model_name, "folds": len(folds)}
    for m in metrics:
        out[m] = bootstrap_ci([getattr(f, m) for f in folds])
    return out


def reliability_plot(y_true: np.ndarray, y_prob: np.ndarray, title: str, path: Path, n_bins: int = 10) -> None:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    xs, ys, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        xs.append(y_prob[mask].mean())
        ys.append(y_true[mask].mean())
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1)
    sizes = np.array(counts, dtype=float)
    sizes = 30 + 120 * (sizes / sizes.max()) if sizes.size else sizes
    ax.scatter(xs, ys, s=sizes, alpha=0.75)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical fraction supported")
    ax.set_title(title)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pr_plot(folds_by_model: dict[str, list[FoldResult]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    for name, folds in folds_by_model.items():
        if not folds:
            continue
        y_true = np.concatenate([f.y_true for f in folds])
        y_prob = np.concatenate([f.y_prob for f in folds])
        p, r, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall (pooled across folds)")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fig-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.claims)
    labeled_df = raw_df[raw_df["support_label"].isin(["supported", "unsupported"])].copy()
    missing_feature_mask = labeled_df[["msa_M", "msa_S", "msa_A"]].isna().any(axis=1)
    skipped_incomplete = int(missing_feature_mask.sum())
    df = labeled_df.loc[~missing_feature_mask].copy()
    df["y"] = (df["support_label"] == "supported").astype(int)
    df["doc_id"] = df["evidence_ids"].apply(extract_doc_id)

    if df.empty:
        raise SystemExit("No labeled claims with complete M/S/A features were found.")

    # Label-leakage check: is msa_A constant within each class?
    leakage = (
        df.groupby("support_label")["msa_A"]
        .agg(["mean", "std", "min", "max", "count"])
        .round(4)
        .to_dict(orient="index")
    )

    # Two splits: by query, by paper (doc_id)
    splits = {}
    for grp_col, name in [("query_id", "group_by_query"), ("doc_id", "group_by_paper")]:
        gkf = GroupKFold(n_splits=min(args.n_splits, df[grp_col].nunique()))
        splits[name] = list(gkf.split(df, df["y"], groups=df[grp_col]))

    report: dict = {
        "source": args.claims,
        "n_claims_raw": int(len(raw_df)),
        "n_claims_labeled": int(len(labeled_df)),
        "n_claims_complete": int(len(df)),
        "n_claims_skipped_incomplete_features": skipped_incomplete,
        "n_supported": int(df["y"].sum()),
        "n_unsupported": int(len(df) - df["y"].sum()),
        "n_queries": int(df["query_id"].nunique()),
        "n_papers": int(df["doc_id"].nunique()),
        "label_leakage_msa_A": leakage,
        "warning": (
            "msa_A is constant within each class (1.0 supported / 0.0 unsupported). "
            "Any model using A trivially reaches 100% accuracy. The leakage-free "
            "benchmark below excludes A. The M+S+A row is retained for transparency."
        ),
        "splits": {},
    }

    fold_cache: dict[str, dict[str, list[FoldResult]]] = {}
    for split_name, split_list in splits.items():
        per_model: dict[str, list[FoldResult]] = {}
        agg: list[dict] = []
        for model_name, features in FEATURE_SETS.items():
            fr = fit_predict(df, features, split_list)
            for x in fr:
                x.model = model_name
            per_model[model_name] = fr
            agg.append(aggregate(model_name, fr))
        fold_cache[split_name] = per_model
        report["splits"][split_name] = {
            "n_folds": len(split_list),
            "models": agg,
        }

    # Figures per split
    for split_name, per_model in fold_cache.items():
        pr_plot(per_model, fig_dir / f"pr_{split_name}.png")
        for model_name, folds in per_model.items():
            if not folds:
                continue
            y_true = np.concatenate([f.y_true for f in folds])
            y_prob = np.concatenate([f.y_prob for f in folds])
            safe = model_name.replace("/", "_").replace("+", "p").replace("(", "_").replace(")", "")
            reliability_plot(
                y_true,
                y_prob,
                title=f"Reliability — {model_name} ({split_name})",
                path=fig_dir / f"reliability_{safe}_{split_name}.png",
            )

    out_path = out_dir / "calibration_robustness.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
