#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.assistant_utils import _compute_agreement_score


def _parse_evidence_ids(raw: object) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            arr = json.loads(text)
            return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            try:
                arr = ast.literal_eval(text)
                return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass
    return [x.strip() for x in re.split(r"[,\s;]+", text) if x.strip()]


def _doc_id_from_evidence_id(eid: str) -> int | None:
    # Handles uploaded:<doc_id>:<chunk_id>:<page_no> and similar variants.
    parts = str(eid or "").split(":")
    for token in parts:
        if token.isdigit():
            try:
                return int(token)
            except Exception:
                return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate leakage-free msa_A values on the legacy 634-claim labeled CSV."
    )
    parser.add_argument("--in-csv", default="Evaluation/data/human_labels/claim_scores_scored.csv")
    parser.add_argument("--out-csv", default="Evaluation/data/human_labels/claim_scores_scored_clean.csv")
    parser.add_argument("--summary-out", default="Evaluation/data/human_labels/claim_scores_scored_clean_summary.json")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    required = {"claim_text", "evidence_text", "evidence_ids", "support_label", "msa_A"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    old_a = df["msa_A"].astype(float)
    clean_a = []

    for row in df.itertuples(index=False):
        claim_text = str(getattr(row, "claim_text", "") or "")
        evidence_text = str(getattr(row, "evidence_text", "") or "")
        evidence_ids = _parse_evidence_ids(getattr(row, "evidence_ids", ""))
        if not evidence_ids:
            evidence_ids = ["evidence:unknown"]

        context_map = {}
        for idx, eid in enumerate(evidence_ids, start=1):
            context_map[idx] = {
                "snippet": evidence_text,
                "doc_id": _doc_id_from_evidence_id(eid),
                "chunk_id": None,
                "source": "uploaded",
            }
        a = _compute_agreement_score(claim_text, context_map, evidence_ids[0])
        clean_a.append(float(a))

    df_out = df.copy()
    df_out["msa_A_old"] = old_a
    df_out["msa_A"] = clean_a
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    summary = {
        "mode": "regenerate_human_claim_scores_clean",
        "source_csv": args.in_csv,
        "output_csv": str(out_csv),
        "n_rows": int(len(df_out)),
        "msa_A_old_by_label": (
            df_out.groupby("support_label")["msa_A_old"].agg(["mean", "std", "min", "max"]).to_dict()
        ),
        "msa_A_clean_by_label": (
            df_out.groupby("support_label")["msa_A"].agg(["mean", "std", "min", "max"]).to_dict()
        ),
        "delta_abs_mean": float((df_out["msa_A"] - df_out["msa_A_old"]).abs().mean()),
    }
    summary_out = Path(args.summary_out)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {summary_out}")


if __name__ == "__main__":
    main()
