#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import _apply_faithfulness_labels_to_claims
from backend.services.judge import evaluate_faithfulness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing sentence-level faithfulness claims in an answer export JSON."
    )
    parser.add_argument("--answers", required=True, help="Path to answers_post_fix.json")
    args = parser.parse_args()

    path = Path(args.answers)
    payload = json.loads(path.read_text())
    queries = payload.get("queries")
    if not isinstance(queries, list):
        raise SystemExit("Expected `queries` list in answers export.")

    updated = 0
    for row in queries:
        faith = row.get("faithfulness") if isinstance(row.get("faithfulness"), dict) else {}
        claims = faith.get("claims") if isinstance(faith.get("claims"), list) else []
        if claims:
            continue

        fb = evaluate_faithfulness(
            query=row.get("query") or "",
            answer=row.get("answer") or "",
            citations=row.get("citations") or [],
            use_llm=False,
        )
        merged = dict(faith)
        merged["claims"] = fb.get("claims") or []
        merged["unsupported"] = fb.get("unsupported") or []
        merged["supported_count"] = int(fb.get("supported_count", 0) or 0)
        merged["unsupported_count"] = int(fb.get("unsupported_count", 0) or 0)
        merged["sentence_count"] = int(fb.get("sentence_count", 0) or 0)
        if merged.get("overall_score") is None:
            merged["overall_score"] = float(fb.get("overall_score", 0.0) or 0.0)
        if merged.get("citation_coverage") is None:
            merged["citation_coverage"] = float(fb.get("citation_coverage", 0.0) or 0.0)
        if not merged.get("method"):
            merged["method"] = "heuristic_backfill"
        row["faithfulness"] = merged
        row["claims"] = _apply_faithfulness_labels_to_claims(row.get("claims") or [], merged)
        updated += 1

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Backfilled faithfulness claims for {updated} queries")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
