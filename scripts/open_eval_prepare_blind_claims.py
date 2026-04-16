#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json, utc_now_iso
from backend.open_eval_spreadsheet import dump_csv_rows, load_csv_rows

BLIND_FIELDS = [
    "blind_id",
    "query_id",
    "query",
    "claim_id",
    "claim_text",
    "evidence_ids",
    "evidence_text",
    "support_label",
    "citation_correct",
    "annotator_notes",
]

KEY_FIELDS = [
    "blind_id",
    "source_row",
    "query_id",
    "claim_id",
    "msa_M",
    "msa_S",
    "msa_A",
    "original_support_label",
    "original_citation_correct",
    "original_annotator_notes",
]


def _required_columns_present(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise SystemExit("Claim annotations CSV is empty.")
    required = {"query_id", "claim_id", "claim_text", "evidence_text"}
    missing = [name for name in sorted(required) if name not in rows[0]]
    if missing:
        raise SystemExit(f"Claim annotations CSV is missing required columns: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create blinded claim-annotation CSV and hidden M/S/A key for unbiased manual labeling."
    )
    parser.add_argument("--claims", required=True, help="Input claim annotations CSV with msa_M/msa_S/msa_A columns.")
    parser.add_argument("--out-dir", required=True, help="Output directory for blinded files.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle row order before assigning blind IDs (recommended).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --shuffle is enabled.",
    )
    parser.add_argument(
        "--id-prefix",
        default="blind",
        help="Prefix for generated blind IDs (default: blind).",
    )
    parser.add_argument(
        "--preserve-existing-labels",
        action="store_true",
        help="Keep existing support_label/citation_correct values in blinded CSV.",
    )
    args = parser.parse_args()

    rows = load_csv_rows(args.claims)
    _required_columns_present(rows)

    ordered_rows = list(enumerate(rows, start=1))
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(ordered_rows)

    blinded_rows: list[dict[str, str]] = []
    key_rows: list[dict[str, str]] = []
    for blind_index, (source_row, row) in enumerate(ordered_rows, start=1):
        blind_id = f"{args.id_prefix}_{blind_index:05d}"
        blinded_rows.append(
            {
                "blind_id": blind_id,
                "query_id": row.get("query_id", ""),
                "query": row.get("query", ""),
                "claim_id": row.get("claim_id", ""),
                "claim_text": row.get("claim_text", ""),
                "evidence_ids": row.get("evidence_ids", ""),
                "evidence_text": row.get("evidence_text", ""),
                "support_label": row.get("support_label", "") if args.preserve_existing_labels else "",
                "citation_correct": row.get("citation_correct", "") if args.preserve_existing_labels else "",
                "annotator_notes": row.get("annotator_notes", "") if args.preserve_existing_labels else "",
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "source_row": str(source_row),
                "query_id": row.get("query_id", ""),
                "claim_id": row.get("claim_id", ""),
                "msa_M": row.get("msa_M", ""),
                "msa_S": row.get("msa_S", ""),
                "msa_A": row.get("msa_A", ""),
                "original_support_label": row.get("support_label", ""),
                "original_citation_correct": row.get("citation_correct", ""),
                "original_annotator_notes": row.get("annotator_notes", ""),
            }
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    blinded_path = out_dir / "claim_annotations_blind.csv"
    key_path = out_dir / "claim_annotations_blind_key.csv"
    manifest_path = out_dir / "claim_annotations_blind_manifest.json"

    dump_csv_rows(blinded_path, BLIND_FIELDS, blinded_rows)
    dump_csv_rows(key_path, KEY_FIELDS, key_rows)
    dump_json(
        {
            "mode": "open_corpus_claim_blinding",
            "created_at": utc_now_iso(),
            "source_file": str(args.claims),
            "row_count": len(blinded_rows),
            "shuffle": bool(args.shuffle),
            "seed": args.seed if args.shuffle else None,
            "id_prefix": args.id_prefix,
            "preserve_existing_labels": bool(args.preserve_existing_labels),
            "files": {
                "blinded_claims": str(blinded_path),
                "blind_key": str(key_path),
            },
        },
        manifest_path,
    )

    print(f"Wrote blinded claims CSV to {blinded_path}")
    print(f"Wrote hidden key CSV to {key_path}")
    print(f"Wrote manifest JSON to {manifest_path}")


if __name__ == "__main__":
    main()
