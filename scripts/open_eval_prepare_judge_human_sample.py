#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _load_claim_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _stratified_sample(rows: list[dict], *, sample_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {"supported": [], "unsupported": []}
    for row in rows:
        label = str(row.get("judge_label") or "").strip().lower()
        if label in by_label:
            by_label[label].append(row)

    for label in by_label:
        rng.shuffle(by_label[label])

    # Keep a balanced sample to avoid a trivial majority baseline.
    target_unsupported = min(len(by_label["unsupported"]), max(1, sample_size // 2))
    target_supported = min(len(by_label["supported"]), sample_size - target_unsupported)

    picked = by_label["unsupported"][:target_unsupported] + by_label["supported"][:target_supported]
    if len(picked) < sample_size:
        remainder = [r for r in rows if r not in picked]
        rng.shuffle(remainder)
        picked.extend(remainder[: sample_size - len(picked)])

    rng.shuffle(picked)
    return picked[:sample_size]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a blinded 50-claim sample for judge-vs-human agreement validation."
    )
    parser.add_argument("--judge", required=True, help="Path to judge_eval_post_fix.json (or judge_eval_final.json)")
    parser.add_argument("--claims-csv", required=True, help="Path to claim_annotations.csv for evidence text")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--out-csv",
        default="Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv",
        help="Output CSV for human annotation",
    )
    parser.add_argument(
        "--out-manifest",
        default="Evaluation/data/post_fix/benchmark_56/judge_human_validation_manifest.json",
        help="Output metadata JSON",
    )
    args = parser.parse_args()

    judge_payload = json.loads(Path(args.judge).read_text())
    judge_claims = judge_payload.get("claims")
    if not isinstance(judge_claims, list) or not judge_claims:
        raise SystemExit("Judge payload must contain a non-empty `claims` list.")

    claim_rows = _load_claim_rows(Path(args.claims_csv))
    claim_lookup: dict[tuple[str, str], dict] = {}
    text_lookup: dict[tuple[str, str], dict] = {}
    for row in claim_rows:
        qid = str(row.get("query_id") or "").strip()
        cid = str(row.get("claim_id") or "").strip()
        if qid and cid:
            claim_lookup[(qid, cid)] = row
        key = (qid, _norm(row.get("claim_text")))
        if key[0] and key[1]:
            text_lookup.setdefault(key, row)

    merged: list[dict] = []
    for claim in judge_claims:
        qid = str(claim.get("query_id") or "").strip()
        cid = str(claim.get("claim_id") or "").strip()
        ctext = str(claim.get("claim_text") or "").strip()
        if not qid or not ctext:
            continue
        label = "supported" if bool(claim.get("supported")) else "unsupported"
        src = claim_lookup.get((qid, cid)) or text_lookup.get((qid, _norm(ctext))) or {}
        merged.append(
            {
                "query_id": qid,
                "query": str(claim.get("query") or src.get("query") or "").strip(),
                "claim_id": cid,
                "claim_text": ctext,
                "evidence_ids": str(claim.get("evidence_ids") or src.get("evidence_ids") or ""),
                "evidence_text": str(src.get("evidence_text") or "").strip(),
                "judge_label": label,
            }
        )

    if len(merged) < args.sample_size:
        raise SystemExit(f"Not enough merged claims ({len(merged)}) for sample size {args.sample_size}.")

    sample = _stratified_sample(merged, sample_size=max(1, args.sample_size), seed=args.seed)
    out_rows: list[dict] = []
    for i, row in enumerate(sample, start=1):
        out_rows.append(
            {
                "sample_id": f"jh{i:03d}",
                "query_id": row["query_id"],
                "query": row["query"],
                "claim_id": row["claim_id"],
                "claim_text": row["claim_text"],
                "evidence_ids": row["evidence_ids"],
                "evidence_text": row["evidence_text"],
                "judge_label": row["judge_label"],
                "human_label": "",
                "annotator_notes": "",
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "query_id",
        "query",
        "claim_id",
        "claim_text",
        "evidence_ids",
        "evidence_text",
        "judge_label",
        "human_label",
        "annotator_notes",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    label_counts = {"supported": 0, "unsupported": 0}
    for row in out_rows:
        label_counts[row["judge_label"]] = label_counts.get(row["judge_label"], 0) + 1

    manifest = {
        "mode": "judge_human_validation_sample",
        "source_judge": str(args.judge),
        "source_claims_csv": str(args.claims_csv),
        "seed": int(args.seed),
        "sample_size": len(out_rows),
        "label_counts": label_counts,
        "instructions": "Fill `human_label` with supported or unsupported for each row.",
        "output_csv": str(out_csv),
    }
    out_manifest = Path(args.out_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_manifest}")


if __name__ == "__main__":
    main()
