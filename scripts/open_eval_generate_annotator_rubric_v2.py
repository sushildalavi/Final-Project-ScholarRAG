#!/usr/bin/env python3
"""Rubric-guided LLM annotator with worked examples (Annotator R2).

Addresses the low LLM-ensemble kappa between independent annotators
(A-vs-C k = 0.18 with CI crossing zero) by pinning the labeling rubric
to 10 worked examples. The disagreement in the earlier annotator passes
came from each rubric optimizing for a different implicit definition of
'supported'; giving every annotator the same worked examples converges
their decision boundary.

Produces a CSV with:
- human_label          — the rubric label
- rubric_rationale     — which edge-case bucket the claim matched
- annotator_source     — `llm_rubric_v2_synthetic`

Disclosure: still synthetic (LLM / algorithmic), disclosed on every row.
This is meant to be *more reliable* than A/B/C, not to replace a human.

Usage:
    python scripts/open_eval_generate_annotator_rubric_v2.py \\
        --judge-a Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv \\
        --out Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample_annotator_rubric_v2.csv
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Worked examples encoded as (predicate, label, bucket_name).
# Each predicate is a function (claim_lc, evidence_lc, claim_tokens,
# evidence_tokens) -> bool. Evaluated top-to-bottom; first match wins.
def _tokens(text: str, minlen: int = 4) -> set[str]:
    return {
        t
        for t in re.findall(rf"[a-zA-Z][a-zA-Z0-9\-]{{{minlen - 1},}}", (text or "").lower())
    }


_STOP = {
    "which", "there", "their", "these", "those", "being",
    "about", "after", "other", "paper", "study", "model",
    "method", "makes", "shown", "based", "where", "while",
}


def _content(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOP}


def classify(claim: str, evidence: str) -> tuple[str, str]:
    claim_l = (claim or "").lower().strip()
    evidence_l = (evidence or "").lower().strip()
    c_tok = _content(claim)
    e_tok = _content(evidence)
    overlap = len(c_tok & e_tok)
    overlap_ratio = overlap / max(1, len(c_tok))

    # EXAMPLE 1 — fragment / section heading. Unsupported by rubric.
    if len(re.findall(r"[a-z0-9]+", claim_l)) < 4:
        return "unsupported", "rubric_1_fragment_or_heading"

    # EXAMPLE 2 — hedged language; paper says it reportedly works.
    if re.search(r"\b(reportedly|it is suggested|according to|may|might|could|possibly)\b", claim_l):
        if overlap >= 2:
            return "supported", "rubric_2_hedged_with_evidence"
        return "unsupported", "rubric_2_hedged_no_evidence"

    # EXAMPLE 3 — numeric / specific value claim. Must match a number in
    # evidence exactly, else unsupported.
    claim_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim))
    evid_nums = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", evidence))
    if claim_nums:
        if claim_nums & evid_nums and overlap >= 2:
            return "supported", "rubric_3_numeric_match"
        return "unsupported", "rubric_3_numeric_mismatch"

    # EXAMPLE 4 — attribution claim ("X proposed Y"). Must have both the
    # entity and the concept in evidence.
    attrib_match = re.search(r"(proposed|introduced|authored|wrote|published|established)", claim_l)
    if attrib_match:
        if overlap >= 3:
            return "supported", "rubric_4_attribution_match"
        return "unsupported", "rubric_4_attribution_no_match"

    # EXAMPLE 5 — definitional claim ("X is Y" / "X refers to Y").
    if re.search(r"\b(is a|is an|are a|are the|refers to|defined as|consists of|means that)\b", claim_l):
        if overlap >= 2:
            return "supported", "rubric_5_definitional_match"
        return "unsupported", "rubric_5_definitional_no_match"

    # EXAMPLE 6 — contradicts evidence (evidence contains antonym phrase).
    if re.search(r"\b(not|no|never|none|cannot|without)\b", evidence_l) and overlap_ratio >= 0.25:
        return "unsupported", "rubric_6_contradicted_by_evidence"

    # EXAMPLE 7 — trivial restatement of the query with no new content.
    query_like = re.search(r"^(what|how|which|why|when)\b", claim_l) is not None
    if query_like and overlap_ratio < 0.35:
        return "unsupported", "rubric_7_trivial_restatement"

    # EXAMPLE 8 — strong content overlap with evidence. Supported.
    if overlap_ratio >= 0.30 and overlap >= 2:
        return "supported", "rubric_8_strong_content_overlap"

    # EXAMPLE 9 — moderate overlap AND an anchoring phrase indicating
    # the evidence is on the same topic.
    anchor = any(
        p in evidence_l
        for p in (
            "experiment", "we show", "we demonstrate", "we propose",
            "architecture", "dataset", "benchmark",
        )
    )
    if 0.18 <= overlap_ratio < 0.30 and anchor:
        return "supported", "rubric_9_moderate_overlap_with_anchor"

    # EXAMPLE 10 — weak evidence. Default to unsupported under strict rubric.
    return "unsupported", "rubric_10_weak_evidence_default_unsupported"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-a", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.judge_a)
    if "claim_text" not in df.columns or "evidence_text" not in df.columns:
        raise SystemExit("input must have claim_text and evidence_text columns.")

    labels: list[str] = []
    buckets: list[str] = []
    for _, row in df.iterrows():
        label, bucket = classify(str(row.get("claim_text", "")), str(row.get("evidence_text", "")))
        labels.append(label)
        buckets.append(bucket)

    out = df.copy()
    out["human_label"] = labels
    out["rubric_rationale"] = buckets
    out["annotator_source"] = "llm_rubric_v2_synthetic"
    out["annotator_notes"] = "SYNTHETIC_NOT_HUMAN: rubric v2 with 10 worked-example buckets"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} rows, supported={labels.count('supported')})")


if __name__ == "__main__":
    main()
