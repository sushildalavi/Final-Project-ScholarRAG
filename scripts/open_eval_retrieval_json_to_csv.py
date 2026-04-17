#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval_spreadsheet import (
    CORPUS_DOC_FIELDS,
    RETRIEVAL_ANNOTATION_FIELDS,
    build_corpus_doc_rows,
    build_retrieval_annotation_rows,
    dump_csv_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert retrieval export JSON to annotation-friendly CSV files."
    )
    parser.add_argument("--in-json", required=True, help="Path to retrieval export JSON")
    parser.add_argument("--out-dir", required=True, help="Directory for CSV outputs")
    args = parser.parse_args()

    payload = json.loads(Path(args.in_json).read_text())
    rows = payload.get("queries")
    if not isinstance(rows, list) or not rows:
        raise SystemExit("Input JSON must contain a non-empty `queries` list.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    retrieval_csv = out_dir / "retrieval_annotations.csv"
    corpus_csv = out_dir / "corpus_doc_relevance.csv"

    dump_csv_rows(retrieval_csv, RETRIEVAL_ANNOTATION_FIELDS, build_retrieval_annotation_rows(rows))
    dump_csv_rows(corpus_csv, CORPUS_DOC_FIELDS, build_corpus_doc_rows(rows))

    print(f"Wrote {retrieval_csv}")
    print(f"Wrote {corpus_csv}")


if __name__ == "__main__":
    main()
