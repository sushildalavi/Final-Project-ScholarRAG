#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import (
    dump_json,
    export_answer_for_query,
    load_query_set,
    ready_documents,
    utc_now_iso,
)


def _load_existing(path: Path) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return [], []
    queries = payload.get("queries")
    errors = payload.get("errors")
    return (queries if isinstance(queries, list) else [], errors if isinstance(errors, list) else [])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ScholarRAG answers for a query set with per-query checkpointing."
    )
    parser.add_argument("--queries", required=True, help="Path to query JSON")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--compute-msa", action="store_true")
    parser.add_argument("--run-judge-llm", action="store_true")
    parser.add_argument("--strict-grounding", action="store_true")
    args = parser.parse_args()

    query_set = load_query_set(args.queries)
    docs = ready_documents()
    if not docs:
        raise SystemExit("No ready uploaded documents found.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    exported_queries, errors = _load_existing(out_path)
    done = {str(row.get("query_id") or "") for row in exported_queries}
    failed = {str(row.get("query_id") or "") for row in errors}

    total = len(query_set["queries"])
    for idx, query_entry in enumerate(query_set["queries"], start=1):
        query_id = str(query_entry.get("query_id") or "")
        if query_id in done:
            print(f"[skip {idx}/{total}] {query_id} already exported", flush=True)
            continue
        if query_id in failed:
            print(f"[retry {idx}/{total}] {query_id}", flush=True)
        else:
            print(f"[run {idx}/{total}] {query_id}: {query_entry.get('query')}", flush=True)
        try:
            exported_queries.append(
                export_answer_for_query(
                    query_entry,
                    k=max(1, args.k),
                    compute_msa=bool(args.compute_msa),
                    run_judge_llm=bool(args.run_judge_llm),
                    strict_grounding=bool(args.strict_grounding),
                    all_docs=docs,
                )
            )
            errors = [row for row in errors if str(row.get("query_id") or "") != query_id]
        except Exception as exc:
            errors = [row for row in errors if str(row.get("query_id") or "") != query_id]
            errors.append(
                {
                    "query_id": query_id,
                    "query": query_entry.get("query"),
                    "error": str(exc),
                }
            )
            print(f"  error: {exc}", flush=True)

        payload = {
            "mode": "open_corpus_answer_export",
            "created_at": utc_now_iso(),
            "k": max(1, args.k),
            "compute_msa": bool(args.compute_msa),
            "run_judge_llm": bool(args.run_judge_llm),
            "strict_grounding": bool(args.strict_grounding),
            "queries": exported_queries,
            "errors": errors,
        }
        dump_json(payload, out_path)

    print(f"Wrote answer export to {out_path}")
    if errors:
        print(f"Completed with {len(errors)} errors; rerun the same command to retry unfinished queries.")


if __name__ == "__main__":
    main()
