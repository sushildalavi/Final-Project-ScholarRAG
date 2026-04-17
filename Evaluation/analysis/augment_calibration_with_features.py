"""Compute the 5 new per-claim features on the existing labeled dataset
and re-run the leakage-free ablation to see if S-only F1 actually lifts.

The features are re-derived from the stored claim + evidence text, so we
can retro-compute them without running the backend. They are INDEPENDENT
of the support label by construction, so adding them to the logistic does
not reintroduce leakage.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


_STOP = {
    "about", "after", "among", "being", "have", "having", "into", "makes",
    "make", "many", "most", "much", "only", "other", "paper", "some",
    "such", "than", "that", "then", "their", "there", "these", "this",
    "those", "very", "what", "when", "where", "which", "while", "with",
    "would", "could", "should", "these", "those", "very", "study",
    "model", "method",
}


def _content(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", (text or "").lower())
        if t not in _STOP
    }


def _entailment_margin_feature(claim: str, evidence: str) -> float:
    c = _content(claim)
    e = _content(evidence)
    if not c:
        return 0.0
    pos_overlap = len(c & e) / max(1, len(c))
    # Neg evidence = presence of negation in evidence without matching
    # negation in claim → contradiction signal.
    claim_has_neg = bool(re.search(r"\b(not|never|no|cannot|without|n't)\b", claim.lower()))
    evid_has_neg = bool(re.search(r"\b(not|never|no|cannot|without|n't)\b", evidence.lower()))
    polarity_mismatch = (claim_has_neg != evid_has_neg)
    margin = pos_overlap - (0.5 if polarity_mismatch else 0.0)
    return max(0.0, min(1.0, margin))


def _specificity_feature(evidence: str) -> float:
    length = len(evidence or "")
    if length == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - abs(length - 260.0) / 600.0))


def _cross_sentence_feature(claim: str, neighbors: list[str]) -> float:
    if not neighbors:
        return 0.5
    c = _content(claim)
    if not c:
        return 0.5
    scores = []
    for n in neighbors:
        n_tokens = _content(n)
        if not n_tokens:
            continue
        scores.append(len(c & n_tokens) / max(1, len(c)))
    return float(sum(scores) / max(1, len(scores))) if scores else 0.5


def _retrieval_diversity_feature(evidence_id: str) -> float:
    # Without the full retrieval context we approximate diversity via the
    # evidence_id string: "uploaded:<doc>:<chunk>:<page>". Count distinct
    # doc-id prefixes observed in the label file, per-query. We fall back
    # to 0.5 when nothing is known.
    # Computed in aggregate in main() where we have the full dataframe.
    return 0.5  # placeholder; patched in main()


def _stability_margin_feature(claim: str, evidence: str) -> float:
    # Proxy: relative length of matched content to claim content. A tight
    # match is a stable signal.
    c = _content(claim)
    e = _content(evidence)
    if not c:
        return 0.0
    ratio = len(c & e) / max(1, len(c))
    return max(0.0, min(1.0, ratio * 1.25))  # cap at 1


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / len(y_true)) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=13) -> dict:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(arr.size),
    }


def _doc_id_from_evidence(evidence_id: str) -> str:
    m = re.search(r"uploaded:(\d+):", str(evidence_id or ""))
    return m.group(1) if m else "unknown"


def augment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    neighbor_by_query: dict[str, list[str]] = {}
    for qid, group in df.groupby("query_id"):
        neighbor_by_query[qid] = group["claim_text"].fillna("").tolist()

    # Doc-diversity per query: fraction of distinct doc_ids / rows.
    diversity_by_query = (
        df.assign(doc_id=df["evidence_ids"].apply(_doc_id_from_evidence))
        .groupby("query_id")["doc_id"]
        .apply(lambda s: len(set(s)) / max(1, len(s)))
        .to_dict()
    )

    new_cols = {
        "feat_entailment_margin": [],
        "feat_citation_specificity": [],
        "feat_cross_sentence": [],
        "feat_retrieval_diversity": [],
        "feat_stability_margin": [],
    }

    for _, row in df.iterrows():
        claim = str(row.get("claim_text", ""))
        evidence = str(row.get("evidence_text", ""))
        qid = row.get("query_id")
        neighbors = [n for n in (neighbor_by_query.get(qid) or []) if n and n != claim]
        diversity = float(diversity_by_query.get(qid, 0.5))

        new_cols["feat_entailment_margin"].append(_entailment_margin_feature(claim, evidence))
        new_cols["feat_citation_specificity"].append(_specificity_feature(evidence))
        new_cols["feat_cross_sentence"].append(_cross_sentence_feature(claim, neighbors[:4]))
        new_cols["feat_retrieval_diversity"].append(diversity)
        new_cols["feat_stability_margin"].append(_stability_margin_feature(claim, evidence))

    for col, vals in new_cols.items():
        df[col] = vals
    return df


def evaluate_features(df: pd.DataFrame, features: list[str], label_col: str = "support_label", n_splits: int = 5) -> dict:
    y = (df[label_col] == "supported").astype(int).to_numpy()
    X = df[features].fillna(0.5).to_numpy(dtype=float)
    groups_query = df["query_id"].astype(str).to_numpy()
    groups_paper = df["evidence_ids"].apply(_doc_id_from_evidence).to_numpy()

    out = {}
    for split_name, groups in (("by_query", groups_query), ("by_paper", groups_paper)):
        gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
        fold_acc, fold_f1, fold_brier, fold_ece, fold_roc, fold_pr = [], [], [], [], [], []
        for tr, te in gkf.split(X, y, groups=groups):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            pred = (p >= 0.5).astype(int)
            fold_acc.append(float((pred == y[te]).mean()))
            fold_f1.append(float(f1_score(y[te], pred, average="macro", zero_division=0)))
            fold_brier.append(float(brier_score_loss(y[te], p)))
            fold_ece.append(expected_calibration_error(y[te], p))
            fold_roc.append(float(roc_auc_score(y[te], p)))
            fold_pr.append(float(average_precision_score(y[te], p)))
        out[split_name] = {
            "accuracy": bootstrap_ci(fold_acc),
            "f1_macro": bootstrap_ci(fold_f1),
            "brier": bootstrap_ci(fold_brier),
            "ece": bootstrap_ci(fold_ece),
            "roc_auc": bootstrap_ci(fold_roc),
            "pr_auc": bootstrap_ci(fold_pr),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--augmented-csv", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.claims)
    df = df[df["support_label"].isin(["supported", "unsupported"])].copy()

    df_aug = augment(df)
    if args.augmented_csv:
        Path(args.augmented_csv).parent.mkdir(parents=True, exist_ok=True)
        df_aug.to_csv(args.augmented_csv, index=False)

    NEW_FEATURES = [
        "feat_entailment_margin",
        "feat_citation_specificity",
        "feat_cross_sentence",
        "feat_retrieval_diversity",
        "feat_stability_margin",
    ]

    report = {
        "source": args.claims,
        "n_claims": int(len(df_aug)),
        "feature_definitions": {
            "feat_entailment_margin": "content-token overlap minus 0.5 when polarity mismatches",
            "feat_citation_specificity": "peaks at 260-char evidence spans",
            "feat_cross_sentence": "avg content-overlap with neighbouring claims in same query",
            "feat_retrieval_diversity": "fraction of distinct doc-ids observed per query",
            "feat_stability_margin": "clipped 1.25 * claim-content-overlap-ratio",
        },
        "feature_stats_by_label": {
            lab: {
                feat: {
                    "mean": round(float(df_aug[df_aug["support_label"] == lab][feat].mean()), 4),
                    "std": round(float(df_aug[df_aug["support_label"] == lab][feat].std()), 4),
                }
                for feat in NEW_FEATURES
            }
            for lab in ("supported", "unsupported")
        },
        "models": {
            "S_only (baseline)": evaluate_features(df_aug, ["msa_S"]),
            "S + 5_new_features": evaluate_features(df_aug, ["msa_S"] + NEW_FEATURES),
            "5_new_features_only": evaluate_features(df_aug, NEW_FEATURES),
            "M_S_plus_features": evaluate_features(df_aug, ["msa_M", "msa_S"] + NEW_FEATURES),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
