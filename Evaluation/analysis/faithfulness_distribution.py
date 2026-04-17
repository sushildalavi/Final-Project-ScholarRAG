"""Per-query faithfulness distribution + query-type stratification + bootstrap CIs.

Addresses the headline-vs-distribution problem:
    README says "90.9% supported" but per-query scores range from 0.0 to 1.0.
    A mean is not enough; decision-makers need the full distribution and
    per-subgroup stratification.

This script:
- Reads judge_eval_final.json and joins per-claim supported flags back to queries.
- Computes per-query support rate.
- Stratifies by query type (factual / synthesis / methodology / comparison)
  inferred from query text heuristics.
- Reports median, IQR, worst-decile, bootstrap CI on the mean.
- Writes a histogram + per-stratum bar chart.

Run:
    python Evaluation/analysis/faithfulness_distribution.py \
        --judge Evaluation/data/llm_judge/judge_eval_final.json \
        --out-dir Evaluation/data/robustness \
        --fig-dir Evaluation/figures/robustness
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SYNTHESIS_HINTS = ("compare", "contrast", "differ", "versus", " vs ", "trade-off", "tradeoff")
METHODOLOGY_HINTS = ("how does", "how is", "architecture", "mechanism", "train", "pretrain", "method", "algorithm")
FACTUAL_HINTS = ("what is", "what are", "definition", "define", "which", "what does")


def classify(query: str) -> str:
    q = (query or "").lower()
    if any(h in q for h in SYNTHESIS_HINTS):
        return "synthesis/comparison"
    if any(h in q for h in METHODOLOGY_HINTS):
        return "methodology"
    if any(h in q for h in FACTUAL_HINTS):
        return "factual"
    return "other"


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> dict:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(values.size),
    }


def quantiles(values: np.ndarray) -> dict:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {"p10": None, "p25": None, "median": None, "p75": None, "p90": None, "min": None}
    return {
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def extract_claim_rows(data: dict) -> list[dict]:
    details = data.get("details") or []
    rows: list[dict] = []
    for detail in details:
        query = detail.get("query")
        query_id = detail.get("query_id")
        faithfulness = detail.get("faithfulness")
        if not query or not isinstance(faithfulness, dict):
            continue
        for claim in faithfulness.get("claims") or []:
            supported = claim.get("supported")
            if supported is None:
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "supported": bool(supported),
                }
            )
    if rows:
        return rows

    fallback = []
    for item in data.get("claims") or []:
        if item.get("query") and item.get("supported") is not None:
            fallback.append(
                {
                    "query_id": item.get("query_id"),
                    "query": item.get("query", ""),
                    "supported": bool(item.get("supported", False)),
                }
            )
    return fallback


def extract_query_rows(data: dict) -> list[dict]:
    rows: list[dict] = []
    for detail in data.get("details") or []:
        query = detail.get("query")
        query_id = detail.get("query_id")
        faithfulness = detail.get("faithfulness")
        if not query or not isinstance(faithfulness, dict):
            continue
        score = faithfulness.get("overall_score")
        if score is None:
            continue
        try:
            score_value = float(score)
        except Exception:
            continue
        rows.append(
            {
                "query_id": query_id,
                "query": query,
                "support_rate": score_value,
                "n_claims": len(faithfulness.get("claims") or []),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fig-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(args.judge).read_text())
    query_rows = extract_query_rows(data)
    if not query_rows:
        raise SystemExit("No query-level faithfulness rows found in judge file.")
    per_query = pd.DataFrame(query_rows)
    per_query["stratum"] = per_query["query"].apply(classify)

    worst = per_query.nsmallest(10, "support_rate")[["query_id", "query", "support_rate", "n_claims"]]

    claim_rows = extract_claim_rows(data)
    claims = pd.DataFrame(claim_rows)
    rates = per_query["support_rate"].to_numpy()
    overall = {
        "n_queries": int(len(per_query)),
        "n_queries_with_sentence_claims": int((per_query["n_claims"] > 0).sum()),
        "n_claims": int(len(claims)),
        "claim_support_rate": float(claims["supported"].mean()) if not claims.empty else None,
        "query_level_bootstrap_ci_95": bootstrap_ci(rates),
        "query_level_quantiles": quantiles(rates),
    }

    strata = {}
    for stratum, sub in per_query.groupby("stratum"):
        arr = sub["support_rate"].to_numpy()
        strata[stratum] = {
            "n_queries": int(len(sub)),
            "bootstrap_ci_95": bootstrap_ci(arr),
            "quantiles": quantiles(arr),
        }

    report = {
        "source": args.judge,
        "overall": overall,
        "by_stratum": strata,
        "worst_decile_queries": worst.to_dict(orient="records"),
    }

    out_path = out_dir / "faithfulness_distribution.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")

    # Histogram
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(rates, bins=np.linspace(0, 1, 21), edgecolor="white")
    ax.axvline(rates.mean(), color="red", ls="--", lw=1.5, label=f"mean={rates.mean():.2f}")
    ax.axvline(np.median(rates), color="orange", ls="--", lw=1.5, label=f"median={np.median(rates):.2f}")
    ax.set_xlabel("Per-query support rate")
    ax.set_ylabel("Number of queries")
    ax.set_title("Faithfulness — per-query distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "faithfulness_hist.png", dpi=140)
    plt.close(fig)

    # Stratum bar chart
    fig, ax = plt.subplots(figsize=(5.5, 4))
    names = list(strata.keys())
    means = [strata[n]["bootstrap_ci_95"]["mean"] or 0 for n in names]
    los = [strata[n]["bootstrap_ci_95"]["lo"] or 0 for n in names]
    his = [strata[n]["bootstrap_ci_95"]["hi"] or 0 for n in names]
    err_lo = [max(0, m - lo) for m, lo in zip(means, los)]
    err_hi = [max(0, hi - m) for m, hi in zip(means, his)]
    x = np.arange(len(names))
    counts = [strata[n]["n_queries"] for n in names]
    ax.bar(x, means, yerr=[err_lo, err_hi], capsize=5, color="#4C72B0", alpha=0.85)
    for xi, n in zip(x, counts):
        ax.text(xi, 0.02, f"n={n}", ha="center", color="white", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Mean per-query support rate (95% CI)")
    ax.set_title("Faithfulness by query type")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "faithfulness_by_stratum.png", dpi=140)
    plt.close(fig)
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
