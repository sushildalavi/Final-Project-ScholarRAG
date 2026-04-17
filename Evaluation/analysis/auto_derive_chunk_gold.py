"""Auto-derive a noisy-but-useful chunk-level gold set from an existing
retrieval run, without human labeling.

Strategy per query:
- Restrict to chunks from the gold doc_id (we already know that much).
- Score each candidate chunk by content-token overlap between the query
  and the chunk's snippet.
- Gold set = chunks whose overlap is in the top-3 for the query AND whose
  normalized overlap is above 0.15. If that's empty, fall back to top-1
  per query.

This is a seed — human review will tighten it. It's already meaningful
because retrieval only has to hit ANY chunk in the gold set to succeed.
Reduces Recall@K from saturated to the real chunk-level ceiling.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


_STOP = {
    "what", "which", "how", "why", "when", "where", "does", "dose", "make",
    "makes", "paper", "study", "used", "uses", "using", "mainly", "about",
    "that", "their", "there", "these", "those", "this", "with", "into",
    "are", "was", "were", "been", "being", "is", "its", "it's",
    "the", "and", "for", "you", "your", "between", "also",
}


def _content_tokens(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
        if t not in _STOP and len(t) >= 4
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--min-overlap", type=float, default=0.15)
    args = parser.parse_args()

    data = json.loads(Path(args.retrieval).read_text())
    details = data.get("details") or []

    queries_out = []
    n_multi = 0
    for row in details:
        q = row.get("query") or ""
        gold_doc = row.get("gold_doc_id")
        candidates = []
        for hit in (row.get("retrieval_only_top") or []):
            if hit.get("doc_id") != gold_doc:
                continue
            snippet = hit.get("snippet") or ""
            overlap_tokens = _content_tokens(q) & _content_tokens(snippet)
            q_tokens = _content_tokens(q)
            ratio = len(overlap_tokens) / max(1, len(q_tokens))
            candidates.append((hit.get("chunk_id"), ratio, len(overlap_tokens)))

        candidates.sort(key=lambda t: (t[1], t[2]), reverse=True)
        top = candidates[: args.top]
        gold_chunks = [int(c[0]) for c in top if c[1] >= args.min_overlap and c[0] is not None]
        if not gold_chunks and candidates:
            gold_chunks = [int(candidates[0][0])] if candidates[0][0] is not None else []
        if len(gold_chunks) > 1:
            n_multi += 1

        queries_out.append(
            {
                "query": q,
                "gold_doc_id": gold_doc,
                "gold_chunk_ids": gold_chunks,
                "notes": f"auto-derived from top-{args.top}, min-overlap={args.min_overlap}",
            }
        )

    report = {
        "mode": "chunk_level_annotation_template",
        "auto_derived_from": args.retrieval,
        "n_queries": len(queries_out),
        "n_queries_with_multi_gold": n_multi,
        "avg_gold_chunks_per_query": (
            sum(len(q["gold_chunk_ids"]) for q in queries_out) / max(1, len(queries_out))
        ),
        "queries": queries_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(
        f"wrote {args.out}  "
        f"({len(queries_out)} queries, "
        f"avg {report['avg_gold_chunks_per_query']:.2f} gold chunks/query, "
        f"{n_multi} have >1 candidate)"
    )


if __name__ == "__main__":
    main()
