#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _suite_pass_map(payload: dict, suite: str) -> dict[str, int]:
    out = {}
    for row in (payload.get(suite, {}) or {}).get("results", []):
        out[str(row.get("id"))] = 1 if bool(row.get("passed")) else 0
    return out


def _mcnemar_exact(before: dict[str, int], after: dict[str, int]) -> dict:
    keys = sorted(set(before.keys()) & set(after.keys()))
    n01 = sum(1 for k in keys if before[k] == 0 and after[k] == 1)  # improved
    n10 = sum(1 for k in keys if before[k] == 1 and after[k] == 0)  # regressed
    discordant = n01 + n10
    if discordant == 0:
        p_two = 1.0
        p_one = 1.0
    else:
        # Two-sided and directional ("after > before") exact sign tests on discordant pairs.
        p_two = float(binomtest(min(n01, n10), n=discordant, p=0.5, alternative="two-sided").pvalue)
        p_one = float(binomtest(n01, n=discordant, p=0.5, alternative="greater").pvalue)
    return {
        "n_items": len(keys),
        "n_improved": int(n01),
        "n_regressed": int(n10),
        "discordant": int(discordant),
        "p_value_two_sided": p_two,
        "p_value_one_sided_improvement": p_one,
    }


def _query_support_rates_from_judge(judge_payload: dict) -> dict[str, float]:
    by_q = {}
    for c in judge_payload.get("claims", []):
        qid = str(c.get("query_id") or "").strip()
        if not qid:
            continue
        bucket = by_q.setdefault(qid, {"total": 0, "supported": 0})
        bucket["total"] += 1
        bucket["supported"] += 1 if bool(c.get("supported")) else 0
    out = {}
    for qid, b in by_q.items():
        if b["total"] > 0:
            out[qid] = b["supported"] / b["total"]
    return out


def _paired_wilcoxon(before: dict[str, float], after: dict[str, float]) -> dict:
    keys = sorted(set(before.keys()) & set(after.keys()))
    x = np.array([before[k] for k in keys], dtype=float)
    y = np.array([after[k] for k in keys], dtype=float)
    delta = y - x
    if len(keys) == 0 or np.allclose(delta, 0):
        p_two = 1.0
        p_one = 1.0
        stat = 0.0
    else:
        stat, p_two = wilcoxon(y, x, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        _, p_one = wilcoxon(y, x, zero_method="wilcox", correction=False, alternative="greater", mode="auto")
        stat = float(stat)
        p_two = float(p_two)
        p_one = float(p_one)
    return {
        "n_pairs": int(len(keys)),
        "before_mean": float(x.mean()) if len(keys) else 0.0,
        "after_mean": float(y.mean()) if len(keys) else 0.0,
        "delta_mean": float(delta.mean()) if len(keys) else 0.0,
        "median_delta": float(np.median(delta)) if len(keys) else 0.0,
        "wilcoxon_stat": stat,
        "p_value_two_sided": p_two,
        "p_value_one_sided_improvement": p_one,
    }


def _pairwise_mode_significance(cross_payload: dict, left: str, right: str, metric: str) -> dict:
    a_rows = {r["query_id"]: float(r[metric]) for r in cross_payload.get("per_query_metrics", {}).get(left, [])}
    b_rows = {r["query_id"]: float(r[metric]) for r in cross_payload.get("per_query_metrics", {}).get(right, [])}
    keys = sorted(set(a_rows.keys()) & set(b_rows.keys()))
    if not keys:
        return {"n_pairs": 0, "p_value": None}
    a = np.array([a_rows[k] for k in keys], dtype=float)
    b = np.array([b_rows[k] for k in keys], dtype=float)
    d = a - b
    if np.allclose(d, 0):
        p = 1.0
        stat = 0.0
    else:
        stat, p = wilcoxon(a, b, alternative="two-sided", mode="auto")
        stat = float(stat)
        p = float(p)
    return {
        "n_pairs": len(keys),
        "left_mean": float(a.mean()),
        "right_mean": float(b.mean()),
        "delta_mean": float((a - b).mean()),
        "wilcoxon_stat": stat,
        "p_value": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal significance tests for post-fix A/B evaluation.")
    parser.add_argument("--baseline-validation", required=True)
    parser.add_argument("--post-validation", required=True)
    parser.add_argument("--baseline-judge", required=True)
    parser.add_argument("--post-judge", required=True)
    parser.add_argument("--cross-corpus", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_val = _load(args.baseline_validation)
    post_val = _load(args.post_validation)
    baseline_judge = _load(args.baseline_judge)
    post_judge = _load(args.post_judge)

    adv_before = _suite_pass_map(baseline_val, "adversarial")
    adv_after = _suite_pass_map(post_val, "adversarial")
    abs_before = _suite_pass_map(baseline_val, "abstention")
    abs_after = _suite_pass_map(post_val, "abstention")

    faith_before = _query_support_rates_from_judge(baseline_judge)
    faith_after = _query_support_rates_from_judge(post_judge)

    report = {
        "mode": "ab_significance_tests",
        "validation_mcnemar": {
            "adversarial": _mcnemar_exact(adv_before, adv_after),
            "abstention": _mcnemar_exact(abs_before, abs_after),
            "combined": _mcnemar_exact({**adv_before, **abs_before}, {**adv_after, **abs_after}),
        },
        "faithfulness_wilcoxon": _paired_wilcoxon(faith_before, faith_after),
    }

    if args.cross_corpus:
        cross = _load(args.cross_corpus)
        has_weighted = "hybrid_weighted" in (cross.get("per_query_metrics") or {})

        mrr_section = {
            "hybrid_rrf_vs_bm25": _pairwise_mode_significance(cross, "hybrid_rrf", "bm25", "mrr"),
            "dense_vs_bm25": _pairwise_mode_significance(cross, "dense_lsi", "bm25", "mrr"),
            "hybrid_rrf_vs_dense": _pairwise_mode_significance(cross, "hybrid_rrf", "dense_lsi", "mrr"),
        }
        r10_section = {
            "hybrid_rrf_vs_bm25": _pairwise_mode_significance(cross, "hybrid_rrf", "bm25", "recall@10"),
            "dense_vs_bm25": _pairwise_mode_significance(cross, "dense_lsi", "bm25", "recall@10"),
            "hybrid_rrf_vs_dense": _pairwise_mode_significance(cross, "hybrid_rrf", "dense_lsi", "recall@10"),
        }
        if has_weighted:
            mrr_section["hybrid_weighted_vs_bm25"] = _pairwise_mode_significance(
                cross, "hybrid_weighted", "bm25", "mrr"
            )
            mrr_section["hybrid_weighted_vs_hybrid_rrf"] = _pairwise_mode_significance(
                cross, "hybrid_weighted", "hybrid_rrf", "mrr"
            )
            r10_section["hybrid_weighted_vs_bm25"] = _pairwise_mode_significance(
                cross, "hybrid_weighted", "bm25", "recall@10"
            )
            r10_section["hybrid_weighted_vs_hybrid_rrf"] = _pairwise_mode_significance(
                cross, "hybrid_weighted", "hybrid_rrf", "recall@10"
            )

        report["cross_corpus_pairwise"] = {
            "mrr": mrr_section,
            "recall@10": r10_section,
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
