#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import NamedTuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json, utc_now_iso
from backend.open_eval_spreadsheet import build_calibration_records_from_claim_csv


class Sample(NamedTuple):
    query_id: str
    claim_id: str
    m: float
    s: float
    a: float
    y: int


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sigmoid(x: float) -> float:
    x = max(-60.0, min(60.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _safe_div(num: float, den: float) -> float:
    if not den:
        return 0.0
    return num / den


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _extract_samples(claim_csv: str | Path) -> list[Sample]:
    records = build_calibration_records_from_claim_csv(claim_csv)
    samples: list[Sample] = []
    for idx, record in enumerate(records, start=1):
        msa = record.get("msa")
        if not isinstance(msa, dict):
            continue
        try:
            m = _clamp01(float(msa.get("M")))
            s = _clamp01(float(msa.get("S")))
            a = _clamp01(float(msa.get("A")))
        except Exception:
            continue
        label = str(record.get("label") or "").strip().lower()
        if label not in {"supported", "unsupported"}:
            continue
        y = 1 if label == "supported" else 0
        query_id = str(record.get("query_id") or "").strip() or "__unknown_query__"
        claim_id = str(record.get("claim_id") or "").strip() or f"row_{idx}"
        samples.append(Sample(query_id=query_id, claim_id=claim_id, m=m, s=s, a=a, y=y))
    return samples


def _split_samples(samples: list[Sample], train_ratio: float, seed: int, group_by_query: bool) -> tuple[list[Sample], list[Sample], dict]:
    if len(samples) < 4:
        raise SystemExit("Need at least 4 labeled rows with M/S/A for train/test split.")

    if group_by_query:
        grouped: dict[str, list[Sample]] = {}
        for sample in samples:
            grouped.setdefault(sample.query_id, []).append(sample)
        query_ids = list(grouped.keys())
        if len(query_ids) >= 2:
            rng = random.Random(seed)
            rng.shuffle(query_ids)
            n_train_groups = int(round(train_ratio * len(query_ids)))
            n_train_groups = max(1, min(len(query_ids) - 1, n_train_groups))
            train_qids = set(query_ids[:n_train_groups])
            train_rows: list[Sample] = []
            test_rows: list[Sample] = []
            for qid in query_ids:
                target = train_rows if qid in train_qids else test_rows
                target.extend(grouped[qid])
            if train_rows and test_rows:
                return train_rows, test_rows, {
                    "strategy": "query_grouped",
                    "groups_total": len(query_ids),
                    "groups_train": n_train_groups,
                    "groups_test": len(query_ids) - n_train_groups,
                    "train_query_ids": sorted(train_qids),
                }

    rows = list(samples)
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_train = int(round(train_ratio * len(rows)))
    n_train = max(1, min(len(rows) - 1, n_train))
    train_rows = rows[:n_train]
    test_rows = rows[n_train:]
    return train_rows, test_rows, {
        "strategy": "row_random",
        "rows_total": len(rows),
        "rows_train": len(train_rows),
        "rows_test": len(test_rows),
    }


def _sample_feature(sample: Sample, feature: str) -> float:
    if feature == "m":
        return sample.m
    if feature == "s":
        return sample.s
    if feature == "a":
        return sample.a
    raise ValueError(f"Unknown feature name: {feature}")


def _predict(sample: Sample, weights: dict[str, float], bias: float, features: list[str]) -> float:
    z = bias
    for feature in features:
        z += float(weights.get(feature, 0.0)) * _sample_feature(sample, feature)
    return _sigmoid(z)


def _fit_logistic(
    samples: list[Sample],
    features: list[str],
    *,
    iters: int = 2200,
    lr: float = 0.38,
    l2: float = 0.001,
    initial_weights: dict[str, float] | None = None,
    initial_bias: float = 0.0,
) -> tuple[dict[str, float], float]:
    if not samples:
        return ({feature: 0.0 for feature in features}, float(initial_bias))

    weights = {feature: float((initial_weights or {}).get(feature, 0.0)) for feature in features}
    bias = float(initial_bias)
    n = float(len(samples))

    for _ in range(max(1, iters)):
        grad_w = {feature: 0.0 for feature in features}
        grad_b = 0.0
        for sample in samples:
            p = _predict(sample, weights, bias, features)
            diff = p - float(sample.y)
            for feature in features:
                grad_w[feature] += diff * _sample_feature(sample, feature)
            grad_b += diff
        for feature in features:
            grad = (grad_w[feature] / n) + (l2 * weights[feature])
            weights[feature] -= lr * grad
        grad_b = (grad_b / n) + (l2 * bias)
        bias -= lr * grad_b

    return weights, bias


def _roc_auc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = [score for score, y in zip(y_score, y_true) if y == 1]
    negatives = [score for score, y in zip(y_score, y_true) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for ps in positives:
        for ns in negatives:
            if ps > ns:
                wins += 1.0
            elif ps == ns:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _pr_auc(y_true: list[int], y_score: list[float]) -> float | None:
    positives = sum(1 for y in y_true if y == 1)
    if positives == 0:
        return None
    ranked = sorted(zip(y_score, y_true), key=lambda item: item[0], reverse=True)
    tp = 0
    fp = 0
    prev_recall = 0.0
    area = 0.0
    for _, label in ranked:
        if label == 1:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = _safe_div(tp, tp + fp)
        area += (recall - prev_recall) * precision
        prev_recall = recall
    return area


def _calibration(y_true: list[int], y_score: list[float], n_bins: int = 10) -> tuple[float, float, list[dict[str, float | int]]]:
    if not y_true:
        return 0.0, 0.0, []

    bins: list[dict[str, float | int]] = []
    total = len(y_true)
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        indices: list[int] = []
        for idx, score in enumerate(y_score):
            in_bin = (lo <= score < hi) or (i == n_bins - 1 and lo <= score <= hi)
            if in_bin:
                indices.append(idx)
        if not indices:
            bins.append(
                {
                    "bin": i,
                    "lower": lo,
                    "upper": hi,
                    "count": 0,
                    "mean_confidence": 0.0,
                    "empirical_accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        conf = sum(y_score[idx] for idx in indices) / len(indices)
        acc = sum(y_true[idx] for idx in indices) / len(indices)
        gap = abs(acc - conf)
        ece += gap * (len(indices) / total)
        mce = max(mce, gap)
        bins.append(
            {
                "bin": i,
                "lower": lo,
                "upper": hi,
                "count": len(indices),
                "mean_confidence": conf,
                "empirical_accuracy": acc,
                "gap": gap,
            }
        )
    return ece, mce, bins


def _evaluate(samples: list[Sample], weights: dict[str, float], bias: float, features: list[str]) -> dict:
    if not samples:
        return {
            "n": 0,
            "positives": 0,
            "negatives": 0,
        }
    y_true = [sample.y for sample in samples]
    y_score = [_predict(sample, weights, bias, features) for sample in samples]
    y_pred = [1 if score >= 0.5 else 0 for score in y_score]

    tp = sum(1 for yp, yt in zip(y_pred, y_true) if yp == 1 and yt == 1)
    tn = sum(1 for yp, yt in zip(y_pred, y_true) if yp == 0 and yt == 0)
    fp = sum(1 for yp, yt in zip(y_pred, y_true) if yp == 1 and yt == 0)
    fn = sum(1 for yp, yt in zip(y_pred, y_true) if yp == 0 and yt == 1)

    accuracy = _safe_div(tp + tn, len(y_true))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    brier = sum((score - label) ** 2 for score, label in zip(y_score, y_true)) / len(y_true)
    roc_auc = _roc_auc(y_true, y_score)
    pr_auc = _pr_auc(y_true, y_score)
    ece, mce, bins = _calibration(y_true, y_score, n_bins=10)

    return {
        "n": len(y_true),
        "positives": sum(y_true),
        "negatives": len(y_true) - sum(y_true),
        "accuracy": _round(accuracy),
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "brier": _round(brier),
        "roc_auc": _round(roc_auc),
        "pr_auc": _round(pr_auc),
        "ece": _round(ece),
        "mce": _round(mce),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "calibration_bins": [
            {
                "bin": int(item["bin"]),
                "lower": _round(float(item["lower"])),
                "upper": _round(float(item["upper"])),
                "count": int(item["count"]),
                "mean_confidence": _round(float(item["mean_confidence"])),
                "empirical_accuracy": _round(float(item["empirical_accuracy"])),
                "gap": _round(float(item["gap"])),
            }
            for item in bins
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate M/S/A confidence calibration on held-out data with leakage-safe split and ablations."
    )
    parser.add_argument("--claims", required=True, help="Annotated claim CSV with msa_M, msa_S, msa_A, support_label.")
    parser.add_argument("-o", "--output", required=True, help="Output JSON path.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of groups reserved for training (default: 0.8).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/test split.")
    parser.add_argument(
        "--group-by",
        choices=("query_id", "row"),
        default="query_id",
        help="Split by query_id (recommended) or by independent rows.",
    )
    parser.add_argument("--iters", type=int, default=2200, help="Number of gradient steps for logistic fitting.")
    args = parser.parse_args()

    if not (0.1 <= args.train_ratio <= 0.9):
        raise SystemExit("--train-ratio must be between 0.1 and 0.9")

    samples = _extract_samples(args.claims)
    if len(samples) < 4:
        raise SystemExit("Need at least 4 labeled rows with valid msa_M/msa_S/msa_A values.")

    train_rows, test_rows, split_info = _split_samples(
        samples,
        train_ratio=float(args.train_ratio),
        seed=int(args.seed),
        group_by_query=(args.group_by == "query_id"),
    )

    ablations: list[tuple[str, list[str]]] = [
        ("M+S+A", ["m", "s", "a"]),
        ("M-only", ["m"]),
        ("S-only", ["s"]),
        ("A-only", ["a"]),
    ]

    default_weights = {"m": 0.58, "s": 0.22, "a": 0.20}

    models: list[dict] = []
    for model_name, features in ablations:
        init = {feature: default_weights.get(feature, 0.0) for feature in features}
        weights, bias = _fit_logistic(
            train_rows,
            features,
            iters=max(1, int(args.iters)),
            lr=0.38,
            l2=0.001,
            initial_weights=init,
            initial_bias=0.0,
        )
        models.append(
            {
                "model": model_name,
                "kind": "trained_logistic",
                "features": features,
                "weights": {feature: _round(weights.get(feature, 0.0), 6) for feature in features},
                "bias": _round(bias, 6),
                "train": _evaluate(train_rows, weights, bias, features),
                "test": _evaluate(test_rows, weights, bias, features),
            }
        )

    models.append(
        {
            "model": "Heuristic baseline (fixed M+S+A)",
            "kind": "fixed_weights",
            "features": ["m", "s", "a"],
            "weights": default_weights,
            "bias": 0.0,
            "train": _evaluate(train_rows, default_weights, 0.0, ["m", "s", "a"]),
            "test": _evaluate(test_rows, default_weights, 0.0, ["m", "s", "a"]),
        }
    )

    output = {
        "mode": "open_corpus_calibration_eval",
        "created_at": utc_now_iso(),
        "source_file": str(args.claims),
        "total_samples": len(samples),
        "split": {
            **split_info,
            "seed": int(args.seed),
            "train_ratio": float(args.train_ratio),
        },
        "models": models,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output, output_path)
    print(f"Wrote calibration evaluation report to {output_path}")


if __name__ == "__main__":
    main()
