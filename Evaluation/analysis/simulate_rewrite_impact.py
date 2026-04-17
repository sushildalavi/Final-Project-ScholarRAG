"""Simulate the effective-support impact of the post-generation claim
rewrite pass on the existing 546-claim baseline, without re-running the
backend.

The rewrite pass (backend/services/assistant_utils._rewrite_ungrounded_claims)
hedges claims that have no citation OR whose cited evidence does not
entail them. Post-hoc rule for this simulation:

- A claim's sentence is considered "hedgable" if:
  - It has no evidence_ids, OR
  - content-token overlap with evidence_text < 0.15

- If a claim is labeled unsupported by the judge AND we would have
  hedged it, we assume the hedged version would be judged supported
  (because hedging reduces the factual strength to the level the
  evidence actually supports). Rate of this successful rescue is
  bounded to 0.65 to reflect that not every rewrite saves the claim.

- If a claim is labeled supported AND hedged, we keep it supported.
- If a claim is labeled supported AND not hedged, we keep it supported.
- Unsupported AND not hedged stays unsupported.

Reports:
- Baseline per-claim and per-query support.
- Post-rewrite (simulated) per-claim and per-query support.
- Coverage and effective-support comparison.
- Bootstrap 95% CI on the gain.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


_STOP = {
    "about", "these", "those", "while", "where", "which",
    "paper", "study", "model", "method", "there", "their",
    "makes", "make", "also", "very", "most", "with", "from",
}


def _content(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{3,}", (text or "").lower())
        if t not in _STOP
    }


def _is_hedgable(claim: dict) -> bool:
    ev_ids = claim.get("evidence_ids")
    if not ev_ids or ev_ids in ("[]", "", "null"):
        return True
    claim_tokens = _content(claim.get("claim_text", ""))
    ev_text = claim.get("evidence_text", "") or claim.get("evidence", "")
    if not ev_text:
        return True
    overlap = len(claim_tokens & _content(ev_text)) / max(1, len(claim_tokens))
    return overlap < 0.15


def _bootstrap_ci_of_mean(values, n_boot=2000, seed=13):
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, 0.025)),
        "hi": float(np.quantile(means, 0.975)),
        "n": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-judge", required=True)
    parser.add_argument("--baseline-claims-csv", required=True, help="Needed for evidence_text")
    parser.add_argument("--rescue-rate", type=float, default=0.65)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import pandas as pd
    judge = json.loads(Path(args.baseline_judge).read_text())
    claims_df = pd.read_csv(args.baseline_claims_csv)

    # Index claim_text + evidence_text by (query_id, claim_id) for lookup.
    evidence_map = {
        (str(row["query_id"]), str(row["claim_id"])): str(row.get("evidence_text") or "")
        for _, row in claims_df.iterrows()
        if "query_id" in claims_df.columns and "claim_id" in claims_df.columns
    }

    rng = np.random.default_rng(13)
    baseline_supported = []
    rewrite_supported = []
    per_query_baseline: dict[str, list[int]] = {}
    per_query_rewrite: dict[str, list[int]] = {}
    hedged_count = 0
    rescued_count = 0

    for claim in judge.get("claims", []):
        sup = bool(claim.get("supported", False))
        claim_text = claim.get("claim_text", "")
        qid = str(claim.get("query_id", ""))
        cid = str(claim.get("claim_id", ""))
        claim_with_ev = dict(claim)
        claim_with_ev["evidence_text"] = evidence_map.get((qid, cid), "")

        hedgable = _is_hedgable(claim_with_ev)
        baseline_supported.append(1 if sup else 0)
        per_query_baseline.setdefault(qid, []).append(1 if sup else 0)

        if sup:
            rewrite_supported.append(1)
            per_query_rewrite.setdefault(qid, []).append(1)
            continue
        if hedgable:
            hedged_count += 1
            # Stochastic rescue
            if rng.random() < args.rescue_rate:
                rescued_count += 1
                rewrite_supported.append(1)
                per_query_rewrite.setdefault(qid, []).append(1)
                continue
        rewrite_supported.append(0)
        per_query_rewrite.setdefault(qid, []).append(0)

    per_query_baseline_rates = [float(np.mean(v)) for v in per_query_baseline.values()]
    per_query_rewrite_rates = [float(np.mean(v)) for v in per_query_rewrite.values()]
    deltas = [b - a for a, b in zip(per_query_baseline_rates, per_query_rewrite_rates)]

    report = {
        "mode": "simulate_rewrite_impact",
        "source": args.baseline_judge,
        "n_claims_total": int(len(baseline_supported)),
        "n_claims_hedgable": int(hedged_count),
        "n_claims_rescued": int(rescued_count),
        "rescue_rate_assumed": args.rescue_rate,
        "baseline": {
            "claim_support_rate": round(float(np.mean(baseline_supported)), 4),
            "query_level_mean_95ci": _bootstrap_ci_of_mean(per_query_baseline_rates),
        },
        "rewrite_simulated": {
            "claim_support_rate": round(float(np.mean(rewrite_supported)), 4),
            "query_level_mean_95ci": _bootstrap_ci_of_mean(per_query_rewrite_rates),
        },
        "delta": {
            "claim_level_absolute": round(float(np.mean(rewrite_supported)) - float(np.mean(baseline_supported)), 4),
            "query_level_absolute_mean_95ci": _bootstrap_ci_of_mean(deltas),
        },
        "coverage_vs_strict_grounding": {
            "strict_n_claims_kept": 202,
            "rewrite_n_claims_kept": int(len(baseline_supported)),
            "coverage_ratio_vs_baseline": 1.0,
            "coverage_ratio_vs_strict": round(len(baseline_supported) / 202.0, 3),
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
