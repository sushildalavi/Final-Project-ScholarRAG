"""Post-audit metrics: LLM-ensemble agreement, strict-grounding coverage tradeoff,
   and bootstrap CIs on the small-n tracks.

This script deliberately replaces the single-pair "IAA" narrative with an
LLM-ensemble kappa, because both annotator B and annotator C are disclosed
as synthetic and are NOT independent human raters. We also:

- Quantify the strict-grounding faithfulness gain against the coverage loss
  (strict answers fewer claims; partial gains are selection bias).
- Attach bootstrap 95% CIs to retrieval metrics on 120q, faithfulness
  on the 56-query rerun, and the pairwise ensemble kappas.

Run:
    python Evaluation/analysis/ensemble_and_coverage.py \\
        --out-json Evaluation/data/post_fix/ensemble_and_coverage.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score


ROOT = Path(__file__).resolve().parents[2]


def _bootstrap_ci_of_mean(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> dict:
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
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


def _bootstrap_ci_of_kappa(y1, y2, n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> dict:
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    n = len(y1)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    kappas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            k = cohen_kappa_score(y1[idx], y2[idx])
            if not math.isnan(k):
                kappas.append(k)
        except Exception:
            pass
    if not kappas:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    kappas = np.asarray(kappas)
    return {
        "mean": float(cohen_kappa_score(y1, y2)),
        "lo": float(np.quantile(kappas, alpha / 2)),
        "hi": float(np.quantile(kappas, 1 - alpha / 2)),
        "n": int(n),
    }


def _load_label_col(path: Path, col: str = "human_label") -> list[str] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[col].astype(str).str.strip().str.lower().tolist()


# ─────────────────────────────────────────────────────────────────────────────
# 1) LLM-ensemble agreement on the 50-claim judge sample
# ─────────────────────────────────────────────────────────────────────────────
def ensemble_agreement_claims() -> dict:
    bench = ROOT / "Evaluation" / "data" / "post_fix" / "benchmark_56"
    a = _load_label_col(bench / "judge_human_validation_sample.csv")
    b = _load_label_col(bench / "judge_human_validation_sample_annotator_b.csv")
    c = _load_label_col(bench / "judge_human_validation_sample_annotator_c.csv")
    judge = _load_label_col(bench / "judge_human_validation_sample.csv", col="judge_label")

    if a is None or b is None or c is None:
        return {"error": "missing annotator files"}

    def _kappa(x, y):
        return _bootstrap_ci_of_kappa(x, y)

    # Ensemble majority vote
    votes = list(zip(a, b, c))
    majority = [
        "supported" if sum(1 for v in trio if v == "supported") >= 2 else "unsupported"
        for trio in votes
    ]

    return {
        "n_claims": len(a),
        "label_balance": {
            "A_supported": int(sum(1 for x in a if x == "supported")),
            "B_supported": int(sum(1 for x in b if x == "supported")),
            "C_supported": int(sum(1 for x in c if x == "supported")),
            "ensemble_majority_supported": int(sum(1 for x in majority if x == "supported")),
        },
        "disclosure": (
            "Annotator A is the rubric-based baseline over the existing human-labeled set. "
            "Annotator B is deterministic-lexical (flip heuristic). "
            "Annotator C is NLI+content-overlap blended. "
            "All three are LLM/algorithmic, not independent human raters. "
            "Numbers below are LLM-ensemble agreement as a PROXY FOR IAA, not IAA itself."
        ),
        "pairwise_kappa_95ci": {
            "A_vs_B": _kappa(a, b),
            "A_vs_C": _kappa(a, c),
            "B_vs_C": _kappa(b, c),
        },
        "ensemble_vs_judge_kappa_95ci": _kappa(judge, majority) if judge is not None else None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2) Retrieval-sheet ensemble agreement (3-class)
# ─────────────────────────────────────────────────────────────────────────────
def ensemble_agreement_retrieval() -> dict:
    d = ROOT / "Evaluation" / "data" / "post_fix" / "retrieval_120q_human_template"
    a = _load_label_col(d / "retrieval_annotations.csv", col="relevance_label")
    b = _load_label_col(d / "retrieval_annotations_annotator_b.csv", col="relevance_label")
    c = _load_label_col(d / "retrieval_annotations_annotator_c.csv", col="relevance_label")
    if a is None or b is None or c is None:
        return {"error": "missing retrieval annotation files"}

    return {
        "n_rows": len(a),
        "pairwise_kappa_95ci": {
            "A_vs_B": _bootstrap_ci_of_kappa(a, b),
            "A_vs_C": _bootstrap_ci_of_kappa(a, c),
            "B_vs_C": _bootstrap_ci_of_kappa(b, c),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3) Strict-grounding: support-rate × coverage tradeoff
# ─────────────────────────────────────────────────────────────────────────────
def strict_grounding_tradeoff() -> dict:
    base_fd = ROOT / "Evaluation" / "data" / "post_fix" / "benchmark_56_baseline_before_strict" / "faithfulness_distribution.json"
    strict_fd = ROOT / "Evaluation" / "data" / "post_fix" / "benchmark_56_strict" / "faithfulness_distribution.json"

    if not base_fd.exists() or not strict_fd.exists():
        return {"error": f"missing FD files: base={base_fd.exists()} strict={strict_fd.exists()}"}

    base = json.loads(base_fd.read_text())
    strict = json.loads(strict_fd.read_text())

    base_overall = base.get("overall", {})
    strict_overall = strict.get("overall", {})
    base_claims = int(base_overall.get("n_claims", 0))
    strict_claims = int(strict_overall.get("n_claims", 0))
    base_support = float(base_overall.get("claim_support_rate", 0.0) or 0.0)
    strict_support = float(strict_overall.get("claim_support_rate", 0.0) or 0.0)

    coverage = strict_claims / max(1, base_claims)
    effective_support_base = base_support
    effective_support_strict = strict_support * coverage

    return {
        "baseline": {
            "n_claims": base_claims,
            "claim_support_rate": round(base_support, 4),
            "query_level_mean": base_overall.get("query_level_bootstrap_ci_95", {}).get("mean"),
            "query_level_ci_95": base_overall.get("query_level_bootstrap_ci_95"),
        },
        "strict": {
            "n_claims": strict_claims,
            "claim_support_rate": round(strict_support, 4),
            "query_level_mean": strict_overall.get("query_level_bootstrap_ci_95", {}).get("mean"),
            "query_level_ci_95": strict_overall.get("query_level_bootstrap_ci_95"),
        },
        "coverage_ratio_strict_over_baseline": round(coverage, 4),
        "effective_support_rate_baseline": round(effective_support_base, 4),
        "effective_support_rate_strict": round(effective_support_strict, 4),
        "interpretation": (
            f"Strict grounding reduced claims from {base_claims} to {strict_claims} "
            f"(coverage {coverage:.0%}). Per-claim support rate went from "
            f"{base_support:.3f} to {strict_support:.3f}. Correcting for the lost "
            f"coverage, the EFFECTIVE support rate (support_rate × coverage) "
            f"moves from {effective_support_base:.3f} to "
            f"{effective_support_strict:.3f}. Report both columns."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4) Bootstrap CIs on retrieval 120q metrics (per-query)
# ─────────────────────────────────────────────────────────────────────────────
def retrieval_120q_cis() -> dict:
    path = ROOT / "Evaluation" / "data" / "retrieval" / "retrieval_eval_120q_final.json"
    if not path.exists():
        return {"error": f"missing {path}"}
    data = json.loads(path.read_text())
    details = data.get("details") or []
    if not details:
        return {"error": "no per-query details"}

    # Each detail has `retrieval_only_top` + gold doc_id. Compute per-query
    # Recall@{1,3,5,10} and reciprocal rank, then bootstrap CI of the mean.
    ranks_reciprocal: list[float] = []
    recall_at: dict[int, list[int]] = {1: [], 3: [], 5: [], 10: []}
    rerank_recall_at: dict[int, list[int]] = {1: [], 3: [], 5: [], 10: []}
    rerank_rr: list[float] = []

    for row in details:
        gold = row.get("gold_doc_id")
        for label, rr_list, kr_map in (
            ("retrieval_only_top", ranks_reciprocal, recall_at),
            ("rerank_top", rerank_rr, rerank_recall_at),
        ):
            top = row.get(label) or []
            found_rank = None
            for i, hit in enumerate(top, start=1):
                if hit.get("doc_id") == gold:
                    found_rank = i
                    break
            rr_list.append(1.0 / found_rank if found_rank else 0.0)
            for k in (1, 3, 5, 10):
                kr_map[k].append(1 if (found_rank is not None and found_rank <= k) else 0)

    result = {
        "n_queries": len(details),
        "retrieval_only": {
            "mrr_95ci": _bootstrap_ci_of_mean(ranks_reciprocal),
            **{f"recall_at_{k}_95ci": _bootstrap_ci_of_mean(vs) for k, vs in recall_at.items()},
        },
        "retrieval_rerank": {
            "mrr_95ci": _bootstrap_ci_of_mean(rerank_rr),
            **{f"recall_at_{k}_95ci": _bootstrap_ci_of_mean(vs) for k, vs in rerank_recall_at.items()},
        },
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5) Bootstrap CI on 56-claim IAA kappa (single point previously)
#    Wraps the fresh_iaa_report numbers with a resampling CI on the kappa stat.
# ─────────────────────────────────────────────────────────────────────────────
def iaa_bootstrap_ci() -> dict:
    bench = ROOT / "Evaluation" / "data" / "post_fix" / "benchmark_56"
    a = _load_label_col(bench / "judge_human_validation_sample.csv")
    b = _load_label_col(bench / "judge_human_validation_sample_annotator_b.csv")
    c = _load_label_col(bench / "judge_human_validation_sample_annotator_c.csv")
    if a is None or b is None or c is None:
        return {"error": "missing sample files"}
    return {
        "note": "Bootstrap CI on pairwise kappa between LLM-ensemble annotators (proxy for IAA).",
        "A_vs_B_kappa_95ci": _bootstrap_ci_of_kappa(a, b),
        "A_vs_C_kappa_95ci": _bootstrap_ci_of_kappa(a, c),
        "B_vs_C_kappa_95ci": _bootstrap_ci_of_kappa(b, c),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    report = {
        "generated_for": "research-grade audit follow-up",
        "disclosure_required_in_any_publication": (
            "Annotators B and C are LLM/algorithmic passes, not independent humans. "
            "The pairwise-kappa numbers below are LLM-ENSEMBLE AGREEMENT used as "
            "a proxy for IAA. Do not report them as human inter-annotator agreement."
        ),
        "ensemble_agreement_claims": ensemble_agreement_claims(),
        "ensemble_agreement_retrieval": ensemble_agreement_retrieval(),
        "strict_grounding_tradeoff": strict_grounding_tradeoff(),
        "retrieval_120q_bootstrap_cis": retrieval_120q_cis(),
        "iaa_bootstrap_cis": iaa_bootstrap_ci(),
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
