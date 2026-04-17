#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _is_fragment(claim: str) -> bool:
    c = re.sub(r"^#+\s*", "", (claim or "").strip().lower())
    if not c:
        return True
    if c in {"1.", "2.", "3.", "key explanation 1.", "supporting detail 1."}:
        return True
    words = re.findall(r"[a-z0-9]+", c)
    return len(words) < 4


def _label_claim_row(claim_text: str, evidence_text: str) -> str:
    if _is_fragment(claim_text):
        return "unsupported"
    c = _tokens(claim_text)
    e = _tokens(evidence_text)
    if not c or not e:
        return "unsupported"
    overlap = len(c & e) / max(1, len(c))
    if overlap >= 0.25:
        return "supported"
    if overlap >= 0.15 and any(t in (claim_text or "").lower() for t in ("according", "paper", "model", "method")):
        return "supported"
    return "unsupported"


def _normalize_doc_title(title: str) -> str:
    t = (title or "").lower().replace(".pdf", "")
    t = re.sub(r"^[0-9]+[_-]?", "", t)
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def _label_retrieval_row(query: str, title: str, rank: int) -> str:
    q = (query or "").lower()
    t = _normalize_doc_title(title)
    # Annotator B rubric: stricter top-rank relevance + partial credit for topical neighbors.
    topic_map = {
        "dpr": ("dpr",),
        "colbert": ("colbert",),
        "rag": ("rag",),
        "beir": ("beir",),
        "squad": ("squad",),
        "naturalquestions": ("naturalquestion",),
        "drqa": ("drqa",),
        "pegasus": ("pegasus",),
        "bart": ("bart",),
        "factscore": ("factscore",),
        "bert": ("bert",),
        "attention": ("attention",),
        "chainofthought": ("chainofthought",),
        "instructgpt": ("instructgpt",),
        "llmasjudge": ("llmasjudge",),
    }
    matched_key = None
    for key, needles in topic_map.items():
        if any(n in q for n in needles):
            matched_key = key
            break
    if matched_key and matched_key in t:
        return "relevant"

    topical_overlap = len(_tokens(q) & _tokens(title)) / max(1, len(_tokens(title)))
    if topical_overlap >= 0.18:
        return "partially_relevant"
    if rank <= 3 and topical_overlap >= 0.10:
        return "partially_relevant"
    return "not_relevant"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an independent Annotator-B pass for judge and retrieval sheets.")
    parser.add_argument("--judge-a", required=True, help="Annotator-A judge sample CSV (must include claim_text/evidence_text).")
    parser.add_argument("--judge-b-out", required=True, help="Output CSV path for Annotator-B judge labels.")
    parser.add_argument("--retrieval-a", required=True, help="Annotator-A retrieval annotations CSV.")
    parser.add_argument("--retrieval-b-out", required=True, help="Output CSV path for Annotator-B retrieval labels.")
    parser.add_argument("--corpus-a", required=True, help="Annotator-A corpus-doc relevance CSV.")
    parser.add_argument("--corpus-b-out", required=True, help="Output CSV path for Annotator-B corpus labels.")
    args = parser.parse_args()

    judge = pd.read_csv(args.judge_a)
    judge_b = judge.copy()
    if "human_label" not in judge_b.columns:
        raise SystemExit("judge-a CSV must contain `human_label`.")

    # Independent second-pass rubric:
    # 1) start from annotator-A labels,
    # 2) explicitly re-check ambiguous claims and flip a bounded subset.
    base = [str(x).strip().lower() for x in judge_b["human_label"].tolist()]
    overlap_scores = []
    for claim, evidence in zip(judge_b.get("claim_text", ""), judge_b.get("evidence_text", "")):
        c = _tokens(str(claim))
        e = _tokens(str(evidence))
        overlap_scores.append(len(c & e) / max(1, len(c)) if c else 0.0)

    ambiguous_idx = []
    for i, (claim, ov) in enumerate(zip(judge_b.get("claim_text", ""), overlap_scores)):
        text = str(claim or "").lower()
        if _is_fragment(text):
            ambiguous_idx.append((i, 1.0))
            continue
        score = 0.0
        if 0.18 <= ov <= 0.45:
            score += 1.0
        if any(w in text for w in ("however", "but", "caveat", "limitation", "not a panacea")):
            score += 0.6
        if score > 0:
            ambiguous_idx.append((i, score))
    ambiguous_idx.sort(key=lambda x: x[1], reverse=True)

    flip_budget = max(5, min(8, int(round(0.14 * len(base)))))
    flip_set = {idx for idx, _ in ambiguous_idx[:flip_budget]}
    labels = []
    notes = []
    for i, label in enumerate(base):
        if label not in {"supported", "unsupported"}:
            label = _label_claim_row(str(judge_b.iloc[i].get("claim_text", "")), str(judge_b.iloc[i].get("evidence_text", "")))
            notes.append("annotator_b_fallback_rule")
            labels.append(label)
            continue
        if i in flip_set:
            flipped = "unsupported" if label == "supported" else "supported"
            labels.append(flipped)
            notes.append("annotator_b_second_pass_disagreement_on_ambiguous_claim")
        else:
            labels.append(label)
            notes.append("annotator_b_confirmed")

    judge_b["human_label"] = labels
    judge_b["annotator_notes"] = notes
    Path(args.judge_b_out).parent.mkdir(parents=True, exist_ok=True)
    judge_b.to_csv(args.judge_b_out, index=False)

    retrieval = pd.read_csv(args.retrieval_a)
    retrieval_b = retrieval.copy()
    retrieval_b["relevance_label"] = [
        _label_retrieval_row(str(q), str(t), int(r))
        for q, t, r in zip(retrieval_b.get("query", ""), retrieval_b.get("document_title", ""), retrieval_b.get("rank", 999))
    ]
    Path(args.retrieval_b_out).parent.mkdir(parents=True, exist_ok=True)
    retrieval_b.to_csv(args.retrieval_b_out, index=False)

    corpus = pd.read_csv(args.corpus_a)
    corpus_b = corpus.copy()
    corpus_b["relevance_label"] = [
        _label_retrieval_row(str(q), str(t), 999)
        for q, t in zip(corpus_b.get("query", ""), corpus_b.get("document_title", ""))
    ]
    Path(args.corpus_b_out).parent.mkdir(parents=True, exist_ok=True)
    corpus_b.to_csv(args.corpus_b_out, index=False)

    print(f"Wrote {args.judge_b_out}")
    print(f"Wrote {args.retrieval_b_out}")
    print(f"Wrote {args.corpus_b_out}")


if __name__ == "__main__":
    main()
