#!/usr/bin/env python3
"""Recompute M/S/A features on a public-mode claim CSV that has labels but no features.

This is the public-mode counterpart to the feature pass in benchmark_56. For each
labeled (claim_text, evidence_text) row:

  * M  = entailment_prob(claim, evidence)        via backend.services.nli
  * A  = _compute_agreement_score(...)           lexical cap-4 across the distinct
                                                  evidence snippets for the same query
  * S  = 0.0                                     (no perturbation context post-hoc)

S is left at 0 because public-mode calibration zeroes its weight anyway (based on
uploaded-mode ablation which showed S was counter-predictive). Post-hoc S would
require re-running retrieval with query perturbations — not worth it for the fit.

Run:
    python backend/scripts/recompute_public_msa.py \
        --claims Evaluation/data/calibration/public_mode/claim_annotations_with_consensus.csv \
        --out    Evaluation/data/calibration/public_mode/claim_annotations_with_msa.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.nli import entailment_prob
from backend.services.assistant_utils import _compute_agreement_score


def build_context_map_per_query(df: pd.DataFrame) -> dict[str, dict[int, dict]]:
    """For each query_id, build the context_map expected by _compute_agreement_score.

    Shape: {query_id: {citation_idx: {"snippet": str, "source": str, ...}}}
    `source` is used as the distinct-doc key by the lexical cap-4 agreement calc.
    """
    out: dict[str, dict[int, dict]] = {}
    for _, row in df.iterrows():
        qid = str(row.get("query_id") or "")
        if not qid:
            continue
        eid = str(row.get("evidence_ids") or "").split(",")[0].strip() or f"row{row.name}"
        snippet = str(row.get("evidence_text") or "")
        ctx = out.setdefault(qid, {})
        idx = len(ctx)
        ctx[idx] = {
            "snippet": snippet,
            "source": eid,
            "doc_id": eid,
            "title": str(row.get("claim_text") or "")[:60],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.claims)
    context_by_query = build_context_map_per_query(df)

    t0 = time.time()
    m_vals, s_vals, a_vals = [], [], []
    for idx, row in df.iterrows():
        claim = str(row.get("claim_text") or "").strip()
        evidence = str(row.get("evidence_text") or "").strip()
        qid = str(row.get("query_id") or "")
        eid = str(row.get("evidence_ids") or "").split(",")[0].strip() or f"row{idx}"

        if not claim or not evidence:
            m_vals.append(0.0)
            s_vals.append(0.0)
            a_vals.append(0.0)
            continue

        try:
            m = float(entailment_prob(claim, evidence))
        except Exception as exc:
            print(f"  [warn] M failed on row {idx}: {exc}")
            m = 0.0

        try:
            context_map = context_by_query.get(qid, {})
            a = float(_compute_agreement_score(claim, context_map, eid))
        except Exception as exc:
            print(f"  [warn] A failed on row {idx}: {exc}")
            a = 0.0

        m_vals.append(round(m, 4))
        s_vals.append(0.0)
        a_vals.append(round(a, 4))

        if (idx + 1) % 25 == 0:
            print(f"  {idx + 1}/{len(df)} rows  elapsed={time.time()-t0:.0f}s")

    df["msa_M"] = m_vals
    df["msa_S"] = s_vals
    df["msa_A"] = a_vals

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print()
    print(f"Wrote {out_path} ({len(df)} rows, elapsed {time.time()-t0:.0f}s)")
    if "support_label" in df.columns:
        print("\nFeature stats by label:")
        print(df.groupby("support_label")[["msa_M", "msa_S", "msa_A"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
