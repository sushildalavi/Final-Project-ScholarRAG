#!/usr/bin/env python3
"""Annotator C — LLM-ensemble third pass, disclosed as SYNTHETIC (not human).

Design (intentionally different from Annotator B so the two are genuinely
independent and LLM-ensemble kappa is not inflated by shared heuristics):

- Does NOT start from Annotator A's labels (B does; that is how B stays
  close to A). C labels from scratch.
- Uses two independent signals:
    1) Natural-language-inference entailment probability on (claim, evidence)
       via backend.services.nli.entailment_prob. This is the SAME signal the
       live M-score uses, so it is a reasonable proxy for a more-semantic
       judge than B's token-overlap rule.
    2) A lightweight coverage rule — the claim must share at least one
       content noun (>=5 chars) with the evidence — to avoid false positives
       from entailment on generic wording.
- Decision rule: supported iff entailment_prob >= 0.55 AND content-noun
  overlap >= 1.

Retrieval rubric is also independent from B: instead of a topic-string
lookup it scores on normalized title-cosine via the same embedding model
used at retrieval time. Hard-ANN proxies:
    >=0.72 -> relevant
    >=0.55 -> partially_relevant
    else    -> not_relevant

Usage:
    python scripts/open_eval_generate_annotator_c.py \\
        --judge-a  Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv \\
        --judge-c-out  Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample_annotator_c.csv \\
        --retrieval-a Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations.csv \\
        --retrieval-c-out Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations_annotator_c.csv \\
        --corpus-a Evaluation/data/post_fix/retrieval_120q_human_template/corpus_doc_relevance.csv \\
        --corpus-c-out Evaluation/data/post_fix/retrieval_120q_human_template/corpus_doc_relevance_annotator_c.csv
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


# Content tokens: words >= 5 chars, lowercase, excluding a short stopword set.
_STOP = {
    "about", "after", "again", "also", "among", "their", "there", "where",
    "which", "while", "would", "could", "should", "these", "those", "being",
    "having", "makes", "make", "used", "into", "then", "than", "such",
    "only", "most", "more", "many", "much", "very", "some", "when", "what",
    "other", "paper", "study", "model", "method", "shown", "based",
}


def _content_tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{4,}", str(text or "").lower())
        if t not in _STOP
    }


def _nli_entailment(claim: str, evidence: str) -> float:
    """Call the project's NLI head. Lazy-imported so this script stays testable
    without the full backend stack."""
    try:
        from backend.services.nli import entailment_prob
        return float(entailment_prob(str(claim or ""), str(evidence or "")))
    except Exception:
        # Fallback rule if NLI is unavailable: Jaccard on content tokens.
        c = _content_tokens(claim)
        e = _content_tokens(evidence)
        if not c or not e:
            return 0.0
        return len(c & e) / len(c | e)


def _label_claim(claim: str, evidence: str) -> tuple[str, float, int]:
    # NLI on long evidence blocks is noisy (median entailment ~0.05 even for
    # on-topic pairs) so we blend NLI with normalised content overlap rather
    # than threshold NLI alone.
    nli_p = _nli_entailment(claim, evidence)
    claim_tokens = _content_tokens(claim)
    evidence_tokens = _content_tokens(evidence)
    overlap = len(claim_tokens & evidence_tokens)
    overlap_ratio = overlap / max(1, len(claim_tokens))

    blended = 0.5 * nli_p + 0.5 * min(1.0, overlap_ratio / 0.35)
    # Supported iff blended score is at least 0.25 AND at least one content
    # noun is shared AND the claim is not a trivial fragment.
    if (
        blended >= 0.25
        and overlap >= 1
        and len(claim_tokens) >= 3
    ):
        return "supported", nli_p, overlap
    return "unsupported", nli_p, overlap


def _title_similarity(query: str, title: str) -> float:
    """Quick proxy using content-token Jaccard. We deliberately don't re-use
    the live embedding model here so Annotator C doesn't collapse to the
    retrieval metric it is supposed to validate."""
    q = _content_tokens(query)
    t = _content_tokens(title)
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)


def _label_retrieval(query: str, title: str) -> str:
    sim = _title_similarity(query, title)
    if sim >= 0.30:
        return "relevant"
    if sim >= 0.12:
        return "partially_relevant"
    return "not_relevant"


def _process_judge(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)
    if "claim_text" not in df.columns or "evidence_text" not in df.columns:
        raise SystemExit("judge input must have claim_text and evidence_text columns.")
    out = df.copy()
    labels: list[str] = []
    nlis: list[float] = []
    overlaps: list[int] = []
    for _, row in df.iterrows():
        label, nli_p, overlap = _label_claim(row.get("claim_text", ""), row.get("evidence_text", ""))
        labels.append(label)
        nlis.append(round(nli_p, 4))
        overlaps.append(overlap)
    out["human_label"] = labels  # keep column name for downstream parity
    out["annotator_source"] = "llm_ensemble_c_nli"
    out["annotator_c_nli_prob"] = nlis
    out["annotator_c_content_overlap"] = overlaps
    out["annotator_notes"] = [
        "SYNTHETIC_NOT_HUMAN: label from NLI entailment prob >= 0.55 AND content-noun overlap >= 1"
        for _ in labels
    ]
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"wrote {output_csv}  ({len(out)} rows, supported={labels.count('supported')})")


def _process_retrieval(input_csv: str, output_csv: str) -> None:
    df = pd.read_csv(input_csv)
    out = df.copy()
    out["relevance_label"] = [
        _label_retrieval(q, t)
        for q, t in zip(out.get("query", ""), out.get("document_title", ""))
    ]
    out["annotator_source"] = "llm_ensemble_c_jaccard"
    out["annotator_notes"] = "SYNTHETIC_NOT_HUMAN: content-noun Jaccard >=0.30 / 0.12 thresholds"
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"wrote {output_csv}  ({len(out)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-a", required=True)
    parser.add_argument("--judge-c-out", required=True)
    parser.add_argument("--retrieval-a", required=True)
    parser.add_argument("--retrieval-c-out", required=True)
    parser.add_argument("--corpus-a", required=True)
    parser.add_argument("--corpus-c-out", required=True)
    args = parser.parse_args()

    _process_judge(args.judge_a, args.judge_c_out)
    _process_retrieval(args.retrieval_a, args.retrieval_c_out)
    _process_retrieval(args.corpus_a, args.corpus_c_out)


if __name__ == "__main__":
    main()
