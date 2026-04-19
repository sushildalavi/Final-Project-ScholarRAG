#!/usr/bin/env python3
"""Run public-mode eval queries and emit a claim_annotations.csv ready for human labeling.

This is the public-mode counterpart to the benchmark_56 pipeline. For each query
in the input JSON:

  * Calls assistant_answer(scope="public", compute_msa=True, run_judge_llm=True)
  * Extracts per-sentence claims, cited evidence snippets, and per-citation M/S/A
  * Writes rows in the same schema as Evaluation/data/post_fix/benchmark_56/claim_annotations.csv
    so the existing fit pipeline (backend/scripts/fit_calibration.py) can consume it.

Prerequisites
    * Backend running (DB connected), API keys configured for public providers
      (Semantic Scholar, arXiv, CrossRef, OpenAlex, etc.)
    * The public retrieval providers should be reachable.

Run:
    python backend/scripts/extract_public_mode_eval.py \
        --queries Evaluation/queries/queries_public_mode.json \
        --out Evaluation/data/calibration/public_mode/claim_annotations.csv

The script is slow (~10-20 minutes for 60 queries) because public APIs are
rate-limited and the LLM judge runs per-claim. That's expected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import build_claim_rows
from backend.open_eval_spreadsheet import (
    CLAIM_ANNOTATION_FIELDS,
    build_claim_annotation_rows,
    dump_csv_rows,
)


def _public_answer_payload(query_entry: dict, k: int = 8) -> dict:
    """Build the request payload for assistant_answer in public scope."""
    return {
        "query": query_entry["query"],
        "scope": "public",
        "k": k,
        "answer_mode": "research_synthesis",
        "compute_msa": True,
        "run_judge": True,
        "allow_general_background": False,
    }


def _call_assistant(query_entry: dict, k: int) -> dict:
    """Run public-mode assistant_answer and return the full response dict."""
    from backend.app import assistant_answer

    return assistant_answer(_public_answer_payload(query_entry, k=k))


def _export_citations(citations: list[dict]) -> list[dict]:
    """Shape citations into the rows expected by build_claim_annotation_rows."""
    exported: list[dict] = []
    for idx, c in enumerate(citations or [], start=1):
        cid = c.get("id") or idx
        msa = c.get("msa") or {}
        exported.append(
            {
                "citation_id": f"S{cid}",
                "citation_index": cid,
                "title": c.get("title"),
                "source": c.get("source"),
                "url": c.get("url"),
                "snippet": c.get("snippet") or c.get("abstract") or "",
                "evidence_id": _evidence_id_for(c),
                "confidence": c.get("confidence"),
                "msa_M": msa.get("M"),
                "msa_S": msa.get("S"),
                "msa_A": msa.get("A"),
            }
        )
    return exported


def _evidence_id_for(citation: dict) -> str:
    """Construct a stable evidence_id for a public citation."""
    source = (citation.get("source") or citation.get("venue") or "public").lower().replace(" ", "")
    url = citation.get("url") or citation.get("doi") or ""
    if url:
        slug = url.split("/")[-1][:40] or "unknown"
    else:
        slug = f"idx{citation.get('id', 'x')}"
    return f"{source}:{slug}"


def run_query(entry: dict, k: int) -> dict | None:
    """Run one query and return a row structured for build_claim_annotation_rows.

    Returns None on failure (logs error, caller can record).
    """
    try:
        result = _call_assistant(entry, k=k)
    except Exception as exc:
        print(f"  ERROR query_id={entry.get('query_id')}: {exc}")
        return None

    citations = _export_citations(result.get("citations") or [])
    claims = build_claim_rows(entry["query_id"], result.get("answer") or "", citations)
    return {
        "query_id": entry["query_id"],
        "query": entry["query"],
        "citations": citations,
        "claims": claims,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, help="Path to queries JSON (queries_public_mode.json)")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--k", type=int, default=8, help="Top-k citations per answer")
    parser.add_argument("--limit", type=int, default=None, help="Optional: only run first N queries (for smoke testing)")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Seconds to sleep between queries (rate limit public APIs)",
    )
    args = parser.parse_args()

    query_set = json.loads(Path(args.queries).read_text())
    queries = query_set["queries"]
    if args.limit:
        queries = queries[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(queries)} queries in public scope -> {out_path}")
    answer_rows: list[dict] = []
    errors: list[dict] = []
    t0 = time.time()

    for i, entry in enumerate(queries, start=1):
        print(f"[{i}/{len(queries)}] {entry.get('query_id')} — {entry.get('query')[:60]}...")
        row = run_query(entry, k=args.k)
        if row is None:
            errors.append({"query_id": entry.get("query_id"), "query": entry.get("query")})
        else:
            answer_rows.append(row)
        if args.sleep and i < len(queries):
            time.sleep(args.sleep)

    claim_rows = build_claim_annotation_rows(answer_rows)
    dump_csv_rows(out_path, CLAIM_ANNOTATION_FIELDS, claim_rows)

    manifest = {
        "mode": "public_mode_calibration_extraction",
        "queries_run": len(queries),
        "queries_ok": len(answer_rows),
        "queries_error": len(errors),
        "claim_rows": len(claim_rows),
        "elapsed_sec": round(time.time() - t0, 1),
        "output_csv": str(out_path),
        "errors": errors,
    }
    manifest_path = out_path.parent / "extraction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print()
    print(f"Wrote {len(claim_rows)} claim rows to {out_path}")
    print(f"Manifest: {manifest_path}")
    if errors:
        print(f"Failed queries: {len(errors)} (see manifest)")


if __name__ == "__main__":
    main()
