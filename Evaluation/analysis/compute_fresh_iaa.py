#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score


def _norm_claim_label(v: object) -> str | None:
    s = str(v or "").strip().lower()
    if s in {"supported", "support", "yes", "true", "1"}:
        return "supported"
    if s in {"unsupported", "not_supported", "no", "false", "0"}:
        return "unsupported"
    return None


def _norm_retrieval_label(v: object) -> str | None:
    s = str(v or "").strip().lower()
    if s in {"relevant", "partially_relevant", "not_relevant"}:
        return s
    return None


def _claim_iaa(a_path: Path, b_path: Path) -> dict:
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    m = a.merge(b, on="sample_id", suffixes=("_a", "_b"))
    m["la"] = m["human_label_a"].map(_norm_claim_label)
    m["lb"] = m["human_label_b"].map(_norm_claim_label)
    m = m[m["la"].notna() & m["lb"].notna()].copy()
    labels = ["supported", "unsupported"]
    kapp = cohen_kappa_score(m["la"], m["lb"], labels=labels) if len(m) else None
    return {
        "rows_used": int(len(m)),
        "cohen_kappa": None if kapp is None else round(float(kapp), 4),
        "agreement": round(float((m["la"] == m["lb"]).mean()), 4) if len(m) else None,
        "label_counts_a": m["la"].value_counts().to_dict(),
        "label_counts_b": m["lb"].value_counts().to_dict(),
    }


def _retrieval_iaa(a_path: Path, b_path: Path) -> dict:
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    key = ["query_id", "rank", "doc_id"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    m["la"] = m["relevance_label_a"].map(_norm_retrieval_label)
    m["lb"] = m["relevance_label_b"].map(_norm_retrieval_label)
    m = m[m["la"].notna() & m["lb"].notna()].copy()
    labels = ["relevant", "partially_relevant", "not_relevant"]
    kapp_3 = cohen_kappa_score(m["la"], m["lb"], labels=labels) if len(m) else None
    m["ba"] = m["la"].replace({"partially_relevant": "relevant"})
    m["bb"] = m["lb"].replace({"partially_relevant": "relevant"})
    kapp_2 = cohen_kappa_score(m["ba"], m["bb"], labels=["relevant", "not_relevant"]) if len(m) else None
    return {
        "rows_used": int(len(m)),
        "cohen_kappa_3class": None if kapp_3 is None else round(float(kapp_3), 4),
        "cohen_kappa_binary": None if kapp_2 is None else round(float(kapp_2), 4),
        "agreement_3class": round(float((m["la"] == m["lb"]).mean()), 4) if len(m) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fresh IAA for new judge and retrieval annotation sheets.")
    parser.add_argument("--judge-a", required=True)
    parser.add_argument("--judge-b", required=True)
    parser.add_argument("--retrieval-a", required=True)
    parser.add_argument("--retrieval-b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    report = {
        "mode": "fresh_iaa_report",
        "claim_sheet": _claim_iaa(Path(args.judge_a), Path(args.judge_b)),
        "retrieval_sheet": _retrieval_iaa(Path(args.retrieval_a), Path(args.retrieval_b)),
        "files": {
            "judge_a": args.judge_a,
            "judge_b": args.judge_b,
            "retrieval_a": args.retrieval_a,
            "retrieval_b": args.retrieval_b,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
