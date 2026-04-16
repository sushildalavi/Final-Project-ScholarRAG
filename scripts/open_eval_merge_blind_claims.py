#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval_spreadsheet import CLAIM_ANNOTATION_FIELDS, dump_csv_rows, load_csv_rows

MERGED_FIELDS = [*CLAIM_ANNOTATION_FIELDS, "blind_id"]


def _index_by_blind_id(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        blind_id = str(row.get("blind_id") or "").strip()
        if not blind_id:
            raise SystemExit(f"{label} is missing `blind_id` values.")
        if blind_id in out:
            raise SystemExit(f"{label} contains duplicate blind_id: {blind_id}")
        out[blind_id] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge manually annotated blinded claims with hidden M/S/A key into claim_annotations format."
    )
    parser.add_argument("--blind-annotations", required=True, help="Annotated blinded claims CSV.")
    parser.add_argument("--blind-key", required=True, help="Hidden blind key CSV generated earlier.")
    parser.add_argument("-o", "--output", required=True, help="Merged claim annotations CSV output path.")
    args = parser.parse_args()

    annotation_rows = load_csv_rows(args.blind_annotations)
    key_rows = load_csv_rows(args.blind_key)
    if not annotation_rows:
        raise SystemExit("Annotated blinded claims CSV is empty.")
    if not key_rows:
        raise SystemExit("Blind key CSV is empty.")

    annotation_by_id = _index_by_blind_id(annotation_rows, "Annotated blinded claims CSV")

    merged_rows: list[dict[str, str]] = []
    missing_annotation_ids: list[str] = []
    for key_row in key_rows:
        blind_id = str(key_row.get("blind_id") or "").strip()
        annotation_row = annotation_by_id.get(blind_id)
        if not annotation_row:
            missing_annotation_ids.append(blind_id)
            continue

        merged_rows.append(
            {
                "query_id": annotation_row.get("query_id", key_row.get("query_id", "")),
                "query": annotation_row.get("query", ""),
                "claim_id": annotation_row.get("claim_id", key_row.get("claim_id", "")),
                "claim_text": annotation_row.get("claim_text", ""),
                "evidence_ids": annotation_row.get("evidence_ids", ""),
                "evidence_text": annotation_row.get("evidence_text", ""),
                "msa_M": key_row.get("msa_M", ""),
                "msa_S": key_row.get("msa_S", ""),
                "msa_A": key_row.get("msa_A", ""),
                "support_label": annotation_row.get("support_label", ""),
                "citation_correct": annotation_row.get("citation_correct", ""),
                "annotator_notes": annotation_row.get("annotator_notes", ""),
                "blind_id": blind_id,
            }
        )

    if missing_annotation_ids:
        preview = ", ".join(missing_annotation_ids[:5])
        suffix = "" if len(missing_annotation_ids) <= 5 else ", ..."
        raise SystemExit(
            "Annotated blinded claims CSV is missing rows for blind_id(s): "
            f"{preview}{suffix} (total missing: {len(missing_annotation_ids)})"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_csv_rows(output_path, MERGED_FIELDS, merged_rows)
    print(f"Wrote {len(merged_rows)} merged claim rows to {output_path}")


if __name__ == "__main__":
    main()
