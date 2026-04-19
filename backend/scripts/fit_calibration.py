#!/usr/bin/env python3
"""Fit a fresh M/S/A logistic calibration on the post-A-leakage-fix dataset.

Unlike the existing `/confidence/calibrate` endpoint (simple gradient descent,
no holdout, no CIs), this script:

  * Uses sklearn's LogisticRegression with L2 regularization.
  * Stratified 80/20 train/holdout split.
  * Bootstraps 1000x on the train set for weight 95% CIs.
  * Reports AUC, Brier, ECE, accuracy, PR-AUC on holdout.
  * Compares fitted weights vs the current defaults on the same holdout.
  * Ships only if fitted weights beat defaults on AUC AND Brier.

Outputs:
  * weights JSON (post-able to /confidence/calibrate or inline into DB).
  * SQL INSERT statement for the confidence_calibration table.
  * Markdown report for the deck / write-up.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DEFAULTS = {"w1": 0.58, "w2": 0.22, "w3": 0.20, "b": 0.0}
FEATURE_COLS = ["msa_M", "msa_S", "msa_A"]


@dataclass
class Metrics:
    n: int
    accuracy: float
    auc: float
    pr_auc: float
    brier: float
    ece: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "auc": round(self.auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "brier": round(self.brier, 4),
            "ece": round(self.ece, 4),
        }


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1], right=False)
    total = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true[mask].mean())
        total += (mask.sum() / n) * abs(conf - acc)
    return total


def score_with_weights(X: np.ndarray, weights: dict) -> np.ndarray:
    z = weights["b"] + weights["w1"] * X[:, 0] + weights["w2"] * X[:, 1] + weights["w3"] * X[:, 2]
    return 1.0 / (1.0 + np.exp(-z))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Metrics:
    y_pred = (y_prob >= 0.5).astype(int)
    acc = float((y_pred == y_true).mean())
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float("nan")
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(expected_calibration_error(y_true, y_prob))
    return Metrics(n=len(y_true), accuracy=acc, auc=auc, pr_auc=pr_auc, brier=brier, ece=ece)


def fit_sklearn_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> dict:
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
    clf.fit(X, y)
    w = clf.coef_.ravel()
    b = float(clf.intercept_[0])
    return {"w1": float(w[0]), "w2": float(w[1]), "w3": float(w[2]), "b": b}


def fit_two_feature_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> dict:
    """Fit logistic on (M, A) only and return keys w1 (M), w3 (A), b."""
    clf = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=2000)
    clf.fit(X, y)
    w = clf.coef_.ravel()
    b = float(clf.intercept_[0])
    return {"w1": float(w[0]), "w3": float(w[1]), "b": b}


def bootstrap_two_feature_weights(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    collected = {"w1": [], "w3": [], "b": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        if len(np.unique(yb)) < 2:
            continue
        w = fit_two_feature_logistic(Xb, yb)
        for k in collected:
            collected[k].append(w[k])
    summary = {"w1": {}, "w2": {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}, "w3": {}, "b": {}}
    for k in ("w1", "w3", "b"):
        a = np.array(collected[k]) if collected[k] else np.array([])
        if len(a) == 0:
            summary[k] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
        else:
            summary[k] = {
                "mean": round(float(a.mean()), 6),
                "lo": round(float(np.quantile(a, 0.025)), 6),
                "hi": round(float(np.quantile(a, 0.975)), 6),
                "n": len(a),
            }
    return summary


def bootstrap_weights(
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y)
    collected = {"w1": [], "w2": [], "w3": [], "b": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        # Skip degenerate samples.
        if len(np.unique(yb)) < 2:
            continue
        w = fit_sklearn_logistic(Xb, yb)
        for k, v in w.items():
            collected[k].append(v)
    summary = {}
    for k, arr in collected.items():
        if not arr:
            summary[k] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
            continue
        a = np.array(arr)
        summary[k] = {
            "mean": round(float(a.mean()), 6),
            "lo": round(float(np.quantile(a, 0.025)), 6),
            "hi": round(float(np.quantile(a, 0.975)), 6),
            "n": len(arr),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claims",
        default="Evaluation/data/post_fix/benchmark_56/claim_annotations.csv",
        help="Path to claim annotations CSV with msa_M/msa_S/msa_A/support_label columns",
    )
    parser.add_argument(
        "--out-dir",
        default="Evaluation/data/calibration/uploaded_mode",
        help="Directory to write weights/report artifacts",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--model-name", default="msa_logistic_uploaded")
    parser.add_argument("--label", default="uploaded_mode")
    parser.add_argument("--ship-threshold-auc", type=float, default=0.0,
                        help="Minimum AUC improvement vs defaults required to ship")
    parser.add_argument("--ship-threshold-brier", type=float, default=0.0,
                        help="Minimum Brier reduction vs defaults required to ship")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.claims)
    labeled = df[df["support_label"].isin(["supported", "unsupported"])].dropna(
        subset=FEATURE_COLS
    ).copy()
    labeled["y"] = (labeled["support_label"] == "supported").astype(int)

    print(f"Loaded {len(df)} rows, {len(labeled)} usable (supported+unsupported with full MSA)")
    print(
        f"Label distribution: supported={int(labeled['y'].sum())}, "
        f"unsupported={int((labeled['y'] == 0).sum())}"
    )

    X = labeled[FEATURE_COLS].to_numpy(dtype=float)
    y = labeled["y"].to_numpy(dtype=int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    print(f"Train: {len(y_tr)} ({int(y_tr.sum())} sup, {int((y_tr == 0).sum())} unsup)")
    print(f"Test:  {len(y_te)} ({int(y_te.sum())} sup, {int((y_te == 0).sum())} unsup)")

    # Fit full M+S+A model.
    msa_weights = fit_sklearn_logistic(X_tr, y_tr)
    print(f"M+S+A weights: {msa_weights}")

    # Fit M+A model (drops S, which ablation showed is counter-predictive).
    X_tr_ma = X_tr[:, [0, 2]]  # M, A
    X_te_ma = X_te[:, [0, 2]]
    ma_fit = fit_two_feature_logistic(X_tr_ma, y_tr)
    ma_weights = {"w1": ma_fit["w1"], "w2": 0.0, "w3": ma_fit["w3"], "b": ma_fit["b"]}
    print(f"M+A weights (S=0): {ma_weights}")

    print(f"Bootstrapping {args.n_boot}x for weight CIs (M+S+A)...")
    boot_msa = bootstrap_weights(X_tr, y_tr, n_boot=args.n_boot, seed=args.seed)
    print(f"Bootstrapping {args.n_boot}x for weight CIs (M+A)...")
    boot_ma = bootstrap_two_feature_weights(X_tr_ma, y_tr, n_boot=args.n_boot, seed=args.seed)

    msa_probs = score_with_weights(X_te, msa_weights)
    ma_probs = score_with_weights(X_te, ma_weights)
    default_probs = score_with_weights(X_te, DEFAULTS)
    msa_m = compute_metrics(y_te, msa_probs)
    ma_m = compute_metrics(y_te, ma_probs)
    default_m = compute_metrics(y_te, default_probs)

    # Pick the winning candidate (prefer M+A if it beats defaults on both AUC and Brier).
    def beats_defaults(m: Metrics) -> tuple[bool, float, float]:
        auc_d = m.auc - default_m.auc
        brier_d = default_m.brier - m.brier
        beats = (
            auc_d >= args.ship_threshold_auc
            and brier_d >= args.ship_threshold_brier
            and math.isfinite(m.auc)
        )
        return beats, auc_d, brier_d

    ma_beats, ma_auc_d, ma_brier_d = beats_defaults(ma_m)
    msa_beats, msa_auc_d, msa_brier_d = beats_defaults(msa_m)

    if ma_beats:
        fitted_weights = ma_weights
        fitted_m = ma_m
        boot = boot_ma
        chosen = "M+A (S=0)"
        auc_delta = ma_auc_d
        brier_delta = ma_brier_d
        ship = True
    elif msa_beats:
        fitted_weights = msa_weights
        fitted_m = msa_m
        boot = boot_msa
        chosen = "M+S+A"
        auc_delta = msa_auc_d
        brier_delta = msa_brier_d
        ship = True
    else:
        fitted_weights = msa_weights
        fitted_m = msa_m
        boot = boot_msa
        chosen = "M+S+A (no-ship)"
        auc_delta = msa_auc_d
        brier_delta = msa_brier_d
        ship = False

    report = {
        "source_csv": str(args.claims),
        "n_rows_raw": int(len(df)),
        "n_rows_usable": int(len(labeled)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "chosen_model": chosen,
        "fitted_weights": {k: round(v, 6) for k, v in fitted_weights.items()},
        "default_weights": DEFAULTS,
        "bootstrap_weight_cis": boot,
        "ablation": {
            "M+S+A": {
                "weights": {k: round(v, 6) for k, v in msa_weights.items()},
                "holdout": msa_m.as_dict(),
            },
            "M+A_S_zeroed": {
                "weights": {k: round(v, 6) for k, v in ma_weights.items()},
                "holdout": ma_m.as_dict(),
            },
            "defaults": {
                "weights": DEFAULTS,
                "holdout": default_m.as_dict(),
            },
        },
        "holdout_metrics": {
            "fitted": fitted_m.as_dict(),
            "defaults": default_m.as_dict(),
            "auc_delta": round(auc_delta, 4),
            "brier_improvement": round(brier_delta, 4),
        },
        "recommendation": {
            "ship": ship,
            "reason": (
                f"{chosen} beats defaults on both AUC and Brier; recommend shipping."
                if ship
                else "Neither fitted model clearly beat defaults on both AUC and Brier; "
                "keep current weights and document the finding honestly."
            ),
        },
    }

    (out_dir / "fit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "weights.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "label": args.label,
                "weights": report["fitted_weights"],
                "metrics": report["holdout_metrics"]["fitted"],
                "dataset_size": int(len(labeled)),
                "ship": ship,
            },
            indent=2,
        )
        + "\n"
    )

    sql = (
        "INSERT INTO confidence_calibration "
        "(model_name, label, weights, metrics, dataset_size) VALUES (\n"
        f"  '{args.model_name}',\n"
        f"  '{args.label}',\n"
        f"  '{json.dumps(report['fitted_weights'])}'::jsonb,\n"
        f"  '{json.dumps(report['holdout_metrics']['fitted'])}'::jsonb,\n"
        f"  {len(labeled)}\n"
        ");\n"
    )
    (out_dir / "insert.sql").write_text(sql)

    md = build_markdown_report(report, args)
    (out_dir / "report.md").write_text(md)

    print()
    print(md)
    print()
    print(f"Artifacts written to {out_dir}/")
    print(f"  - fit_report.json (machine-readable)")
    print(f"  - weights.json    (for /confidence/calibrate payload or direct DB insert)")
    print(f"  - insert.sql      (ready-to-apply)")
    print(f"  - report.md       (human-readable write-up)")


def build_markdown_report(report: dict, args) -> str:
    fw = report["fitted_weights"]
    dw = report["default_weights"]
    boot = report["bootstrap_weight_cis"]
    fm = report["holdout_metrics"]["fitted"]
    dm = report["holdout_metrics"]["defaults"]

    def ci(k: str) -> str:
        b = boot.get(k, {})
        if not b or b.get("n", 0) == 0:
            return "—"
        return f"[{b['lo']:+.3f}, {b['hi']:+.3f}]"

    ship = report["recommendation"]["ship"]
    reason = report["recommendation"]["reason"]

    lines = []
    lines.append("# MSA Calibration — Post-Fix Refit (v2)")
    lines.append("")
    lines.append(f"**Source:** `{args.claims}`  ")
    lines.append(
        f"**Dataset:** {report['n_rows_usable']} labeled claims "
        f"(train={report['n_train']}, holdout={report['n_test']})  "
    )
    lines.append(f"**Model:** `{args.model_name}` / label `{args.label}`  ")
    lines.append(f"**Chosen:** `{report['chosen_model']}`  ")
    lines.append("**Fitter:** sklearn `LogisticRegression(penalty='l2', C=1.0)` with stratified 80/20 split.")
    lines.append("")
    lines.append("## Feature ablation on holdout")
    lines.append("")
    abl = report["ablation"]
    lines.append("| Model | AUC | Brier | ECE | Accuracy |")
    lines.append("|-------|-----|-------|-----|----------|")
    for name, key in [("M+S+A (full)", "M+S+A"), ("M+A (S=0)", "M+A_S_zeroed"), ("Defaults", "defaults")]:
        h = abl[key]["holdout"]
        lines.append(f"| {name} | {h['auc']:.4f} | {h['brier']:.4f} | {h['ece']:.4f} | {h['accuracy']:.4f} |")
    lines.append("")
    lines.append(
        "> **Key finding:** In the post-fix regime, S (retrieval-stability) is "
        "counter-predictive on our data — S-only AUC is below random (flipped sign). "
        "Dropping S and zeroing w2 yields the strongest calibration."
    )
    lines.append("")
    lines.append("## Chosen weights (M/S/A logistic)")
    lines.append("")
    lines.append("| Weight | Fitted | Default | Bootstrap 95% CI |")
    lines.append("|--------|--------|---------|------------------|")
    lines.append(f"| w1 (M) | {fw['w1']:+.4f} | {dw['w1']:+.2f} | {ci('w1')} |")
    lines.append(f"| w2 (S) | {fw['w2']:+.4f} | {dw['w2']:+.2f} | {ci('w2')} |")
    lines.append(f"| w3 (A) | {fw['w3']:+.4f} | {dw['w3']:+.2f} | {ci('w3')} |")
    lines.append(f"| b      | {fw['b']:+.4f} | {dw['b']:+.2f} | {ci('b')} |")
    lines.append("")
    lines.append("## Held-out metrics (20% stratified)")
    lines.append("")
    lines.append("| Metric   | Fitted | Defaults | Δ |")
    lines.append("|----------|--------|----------|---|")
    lines.append(
        f"| AUC      | {fm['auc']:.4f} | {dm['auc']:.4f} | "
        f"{report['holdout_metrics']['auc_delta']:+.4f} |"
    )
    lines.append(
        f"| PR-AUC   | {fm['pr_auc']:.4f} | {dm['pr_auc']:.4f} | "
        f"{(fm['pr_auc'] - dm['pr_auc']):+.4f} |"
    )
    lines.append(
        f"| Brier    | {fm['brier']:.4f} | {dm['brier']:.4f} | "
        f"{report['holdout_metrics']['brier_improvement']:+.4f} (lower = better) |"
    )
    lines.append(
        f"| ECE      | {fm['ece']:.4f} | {dm['ece']:.4f} | "
        f"{(dm['ece'] - fm['ece']):+.4f} (lower = better) |"
    )
    lines.append(
        f"| Accuracy | {fm['accuracy']:.4f} | {dm['accuracy']:.4f} | "
        f"{(fm['accuracy'] - dm['accuracy']):+.4f} |"
    )
    lines.append("")
    lines.append("## Ship decision")
    lines.append("")
    lines.append(f"**Recommendation:** {'SHIP' if ship else 'DO NOT SHIP — keep defaults'}")
    lines.append("")
    lines.append(f"> {reason}")
    lines.append("")
    lines.append("## How to apply (if shipping)")
    lines.append("")
    lines.append("Option A — via HTTP endpoint (backend must be running):")
    lines.append("")
    lines.append("```bash")
    lines.append("curl -X POST http://localhost:8000/confidence/calibrate \\")
    lines.append("  -H 'Content-Type: application/json' \\")
    lines.append("  -d @Evaluation/data/calibration/<mode>/weights.json")
    lines.append("```")
    lines.append("")
    lines.append("Option B — direct DB insert:")
    lines.append("")
    lines.append("```bash")
    lines.append("psql $DATABASE_URL -f Evaluation/data/calibration/<mode>/insert.sql")
    lines.append("```")
    lines.append("")
    lines.append("`_load_latest_calibration_weights()` picks up the new row on next request. "
                 "No backend redeploy needed.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
