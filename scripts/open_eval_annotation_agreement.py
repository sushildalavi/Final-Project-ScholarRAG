#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import utc_now_iso
from backend.open_eval_spreadsheet import load_csv_rows


def _normalize_support(value: object) -> str | None:
    v = str(value or "").strip().lower()
    if v in {"supported", "support", "yes", "true", "1"}:
        return "supported"
    if v in {"unsupported", "not_supported", "no", "false", "0"}:
        return "unsupported"
    return None


def _normalize_citation(value: object) -> str | None:
    v = str(value or "").strip().lower()
    if v in {"true", "yes", "1"}:
        return "true"
    if v in {"false", "no", "0"}:
        return "false"
    return None


def _cohen_kappa(a_labels: list[str], b_labels: list[str], classes: list[str]) -> float | None:
    if not a_labels or not b_labels or len(a_labels) != len(b_labels):
        return None
    n = len(a_labels)
    observed = sum(1 for a, b in zip(a_labels, b_labels) if a == b) / n

    pe = 0.0
    for cls in classes:
        pa = sum(1 for label in a_labels if label == cls) / n
        pb = sum(1 for label in b_labels if label == cls) / n
        pe += pa * pb
    if pe >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return (observed - pe) / (1.0 - pe)


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _build_lookup(path: str | Path) -> dict[str, dict[str, str]]:
    rows = load_csv_rows(path)
    lookup: dict[str, dict[str, str]] = {}
    for row in rows:
        blind_id = str(row.get("blind_id") or "").strip()
        if not blind_id:
            continue
        lookup[blind_id] = row
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement (Cohen's kappa) for blinded claim annotations."
    )
    parser.add_argument("--annotator-a", required=True, help="Annotated blinded claims CSV from annotator A.")
    parser.add_argument("--annotator-b", required=True, help="Annotated blinded claims CSV from annotator B.")
    parser.add_argument("-o", "--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    lookup_a = _build_lookup(args.annotator_a)
    lookup_b = _build_lookup(args.annotator_b)
    overlap_ids = sorted(set(lookup_a.keys()) & set(lookup_b.keys()))
    if not overlap_ids:
        raise SystemExit("No overlapping blind_id rows found between annotator files.")

    support_a: list[str] = []
    support_b: list[str] = []
    citation_a: list[str] = []
    citation_b: list[str] = []

    support_confusion = {
        "supported_supported": 0,
        "supported_unsupported": 0,
        "unsupported_supported": 0,
        "unsupported_unsupported": 0,
    }

    for blind_id in overlap_ids:
        row_a = lookup_a[blind_id]
        row_b = lookup_b[blind_id]
        sa = _normalize_support(row_a.get("support_label"))
        sb = _normalize_support(row_b.get("support_label"))
        if sa is not None and sb is not None:
            support_a.append(sa)
            support_b.append(sb)
            support_confusion[f"{sa}_{sb}"] += 1

        ca = _normalize_citation(row_a.get("citation_correct"))
        cb = _normalize_citation(row_b.get("citation_correct"))
        if ca is not None and cb is not None:
            citation_a.append(ca)
            citation_b.append(cb)

    support_kappa = _cohen_kappa(support_a, support_b, ["supported", "unsupported"])
    citation_kappa = _cohen_kappa(citation_a, citation_b, ["true", "false"])

    report = {
        "mode": "open_corpus_annotation_agreement",
        "created_at": utc_now_iso(),
        "files": {
            "annotator_a": str(args.annotator_a),
            "annotator_b": str(args.annotator_b),
        },
        "overlap_rows": len(overlap_ids),
        "support_label": {
            "rows_used": len(support_a),
            "cohen_kappa": _round(support_kappa),
            "confusion": support_confusion,
        },
        "citation_correct": {
            "rows_used": len(citation_a),
            "cohen_kappa": _round(citation_kappa),
        },
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote agreement report to {out_path}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
