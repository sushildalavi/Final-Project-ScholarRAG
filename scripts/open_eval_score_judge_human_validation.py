#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def _normalize_label(label: str | None) -> str:
    v = str(label or "").strip().lower()
    if v in {"supported", "support", "yes", "true", "1"}:
        return "supported"
    if v in {"unsupported", "not_supported", "no", "false", "0"}:
        return "unsupported"
    return ""


def _cohen_kappa(a: list[str], b: list[str]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    classes = ["supported", "unsupported"]
    expected = 0.0
    for cls in classes:
        pa = sum(1 for x in a if x == cls) / n
        pb = sum(1 for y in b if y == cls) / n
        expected += pa * pb
    if math.isclose(1.0 - expected, 0.0):
        return None
    return (observed - expected) / (1.0 - expected)


def _bootstrap_ci(values: list[float], *, n_boot: int = 2000, seed: int = 19) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(max(200, n_boot)):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * len(means))]
    hi = means[min(len(means) - 1, int(0.975 * len(means)))]
    return {"mean": sum(values) / n, "lo": lo, "hi": hi}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score judge-vs-human agreement on a manually labeled claim sample.")
    parser.add_argument(
        "--sample-csv",
        default="Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv",
        help="Annotated CSV with `judge_label` and `human_label` columns.",
    )
    parser.add_argument(
        "--out",
        default="Evaluation/data/post_fix/benchmark_56/judge_human_validation_report.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    rows = []
    with Path(args.sample_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            judge = _normalize_label(row.get("judge_label"))
            human = _normalize_label(row.get("human_label"))
            if not judge:
                continue
            rows.append({"judge_label": judge, "human_label": human, **row})

    scored = [r for r in rows if r["human_label"] in {"supported", "unsupported"}]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not scored:
        report = {
            "mode": "judge_human_validation_score",
            "sample_csv": args.sample_csv,
            "status": "pending_human_labels",
            "n_rows_total": len(rows),
            "n_rows_scored": 0,
            "n_rows_missing_human_label": len(rows),
            "message": "Fill `human_label` with supported/unsupported, then rerun this command.",
        }
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {out_path}")
        return

    y_true = [1 if r["human_label"] == "supported" else 0 for r in scored]
    y_pred = [1 if r["judge_label"] == "supported" else 0 for r in scored]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n = len(scored)

    accuracy = (tp + tn) / max(1, n)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)
    kapp = _cohen_kappa([r["judge_label"] for r in scored], [r["human_label"] for r in scored])

    correctness = [1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)]
    accuracy_ci = _bootstrap_ci(correctness)

    report = {
        "mode": "judge_human_validation_score",
        "sample_csv": args.sample_csv,
        "n_rows_total": len(rows),
        "n_rows_scored": n,
        "n_rows_missing_human_label": len(rows) - n,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "metrics": {
            "accuracy": accuracy,
            "accuracy_ci_95": accuracy_ci,
            "precision_supported": precision,
            "recall_supported": recall,
            "f1_supported": f1,
            "cohen_kappa": kapp,
        },
    }

    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
