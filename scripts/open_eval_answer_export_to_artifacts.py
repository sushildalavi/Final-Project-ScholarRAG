#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json
from backend.open_eval_spreadsheet import (
    CLAIM_ANNOTATION_FIELDS,
    QUERY_SUMMARY_FIELDS,
    build_claim_annotation_rows,
    build_query_summary_rows,
    dump_csv_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an answer export JSON into claim CSV, query summary CSV, and judge-style JSON."
    )
    parser.add_argument("--answers", required=True, help="Path to answer export JSON")
    parser.add_argument("--out-dir", required=True, help="Directory for derived artifacts")
    args = parser.parse_args()

    payload = json.loads(Path(args.answers).read_text())
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise SystemExit("Answer export must contain a non-empty `queries` list.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_summary_path = out_dir / "query_summary.csv"
    claim_csv_path = out_dir / "claim_annotations.csv"
    judge_json_path = out_dir / "judge_eval_post_fix.json"

    dump_csv_rows(query_summary_path, QUERY_SUMMARY_FIELDS, build_query_summary_rows(queries))
    dump_csv_rows(claim_csv_path, CLAIM_ANNOTATION_FIELDS, build_claim_annotation_rows(queries))

    details = []
    claims = []
    for query_row in queries:
        faithfulness = query_row.get("faithfulness")
        details.append(
            {
                "query_id": query_row.get("query_id"),
                "query": query_row.get("query"),
                "answer": query_row.get("answer"),
                "citations": query_row.get("citations") or [],
                "faithfulness": faithfulness if isinstance(faithfulness, dict) else {},
                "doc_id": query_row.get("doc_id"),
                "doc_ids": query_row.get("doc_ids"),
                "scope": query_row.get("doc_scope"),
            }
        )
        for claim in query_row.get("claims") or []:
            label = str(claim.get("label") or "").strip().lower()
            if label not in {"supported", "unsupported"}:
                continue
            claims.append(
                {
                    "query_id": query_row.get("query_id"),
                    "query": query_row.get("query"),
                    "doc_id": query_row.get("doc_id"),
                    "claim_id": claim.get("claim_id"),
                    "claim_text": claim.get("text"),
                    "evidence_ids": json.dumps(claim.get("evidence_ids") or []),
                    "supported": label == "supported",
                    "reason": "",
                }
            )

    batch_metrics = []
    for row in details:
        faithfulness = row.get("faithfulness") if isinstance(row.get("faithfulness"), dict) else {}
        batch_metrics.append(
            {
                "mean_overall_score": float(faithfulness.get("overall_score", 0.0) or 0.0),
                "mean_coverage": float(faithfulness.get("citation_coverage", 0.0) or 0.0),
                "unsupported_total": int(faithfulness.get("unsupported_count", 0) or 0),
                "count": int(faithfulness.get("sentence_count", 0) or 0),
            }
        )

    dump_json(
        {
            "details": details,
            "claims": claims,
            "batch_metrics": batch_metrics,
        },
        judge_json_path,
    )

    print(f"Wrote {query_summary_path}")
    print(f"Wrote {claim_csv_path}")
    print(f"Wrote {judge_json_path}")


if __name__ == "__main__":
    main()
