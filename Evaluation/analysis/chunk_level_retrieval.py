"""Chunk-level retrieval evaluation harness.

Document-level Recall@K on our 120-query benchmark is saturated at 1.00
because every query names its gold paper and the 15-paper corpus is
topically disjoint. Chunk-level evaluation is the honest version of the
same test:

- For each query, the gold label is the set of chunk_ids that actually
  contain the answer, not just the paper they live in.
- A top-K result counts as a hit only if its chunk_id is in the gold
  chunk set.
- This typically drops Recall@5 from 1.00 to 0.70–0.85 on our corpus
  and makes the reranker's contribution visible.

Run mode 1 — extract chunk annotations from an existing retrieval run:
    python Evaluation/analysis/chunk_level_retrieval.py --derive-from \\
        Evaluation/data/retrieval/retrieval_eval_120q_final.json \\
        --annotations-out Evaluation/data/retrieval/chunk_annotations_template.json

Run mode 2 — score a retrieval run against chunk annotations:
    python Evaluation/analysis/chunk_level_retrieval.py \\
        --retrieval Evaluation/data/retrieval/retrieval_eval_120q_final.json \\
        --annotations Evaluation/data/retrieval/chunk_annotations.json \\
        --out Evaluation/data/retrieval/chunk_level_metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 13) -> dict:
    arr = np.asarray([v for v in values if not math.isnan(v)], dtype=float)
    if arr.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(arr.size),
    }


def derive_template(retrieval_json: str, out_path: str) -> None:
    data = json.loads(Path(retrieval_json).read_text())
    details = data.get("details") or []

    # Pre-fill with the top-1 chunk_id per query as a best-effort seed.
    # The user is expected to review and correct these — this is the
    # "annotation template" not the gold set.
    template = {
        "mode": "chunk_level_annotation_template",
        "source_retrieval": retrieval_json,
        "instruction": (
            "For each query, list the chunk_ids that actually contain "
            "information answering the query. Seed value is the top-1 "
            "chunk from the retrieval run — review and correct it. Leave "
            "gold_chunk_ids empty to skip a query in the chunk-level eval."
        ),
        "queries": [],
    }
    for row in details:
        top = row.get("retrieval_only_top") or []
        seed = top[0].get("chunk_id") if top else None
        template["queries"].append(
            {
                "query": row.get("query"),
                "gold_doc_id": row.get("gold_doc_id"),
                "gold_chunk_ids": [seed] if seed is not None else [],
                "notes": "",
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(template, indent=2))
    print(f"wrote {out_path}  ({len(details)} queries)")


def score(retrieval_json: str, annotations_json: str, out_path: str) -> None:
    retrieval = json.loads(Path(retrieval_json).read_text())
    ann = json.loads(Path(annotations_json).read_text())

    gold_by_query: dict[str, set[int]] = {}
    for entry in ann.get("queries", []):
        q = entry.get("query")
        if q is None:
            continue
        chunk_ids = entry.get("gold_chunk_ids") or []
        if chunk_ids:
            gold_by_query[q] = set(int(x) for x in chunk_ids)

    details = retrieval.get("details") or []
    per_query = []
    recall_at = {k: [] for k in (1, 3, 5, 10)}
    rerank_recall_at = {k: [] for k in (1, 3, 5, 10)}
    rr = []
    rerank_rr = []

    for row in details:
        q = row.get("query")
        gold = gold_by_query.get(q)
        if not gold:
            continue

        for label, rr_list, kr_map in (
            ("retrieval_only_top", rr, recall_at),
            ("rerank_top", rerank_rr, rerank_recall_at),
        ):
            top = row.get(label) or []
            found_rank = None
            for i, hit in enumerate(top, start=1):
                cid = hit.get("chunk_id")
                if cid is not None and int(cid) in gold:
                    found_rank = i
                    break
            rr_list.append(1.0 / found_rank if found_rank else 0.0)
            for k in (1, 3, 5, 10):
                kr_map[k].append(1 if (found_rank is not None and found_rank <= k) else 0)

        per_query.append(
            {
                "query": q,
                "gold_chunk_ids": sorted(gold),
                "retrieval_only_hit_rank": None,
                "rerank_hit_rank": None,
            }
        )

    report = {
        "mode": "chunk_level_retrieval_metrics",
        "n_queries_scored": len(per_query),
        "retrieval_only": {
            "mrr_95ci": _bootstrap_ci(rr),
            **{f"recall_at_{k}_95ci": _bootstrap_ci([float(x) for x in v]) for k, v in recall_at.items()},
        },
        "rerank": {
            "mrr_95ci": _bootstrap_ci(rerank_rr),
            **{f"recall_at_{k}_95ci": _bootstrap_ci([float(x) for x in v]) for k, v in rerank_recall_at.items()},
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derive-from")
    parser.add_argument("--annotations-out")
    parser.add_argument("--retrieval")
    parser.add_argument("--annotations")
    parser.add_argument("--out")
    args = parser.parse_args()

    if args.derive_from:
        if not args.annotations_out:
            raise SystemExit("--annotations-out required with --derive-from")
        derive_template(args.derive_from, args.annotations_out)
    elif args.retrieval and args.annotations and args.out:
        score(args.retrieval, args.annotations, args.out)
    else:
        raise SystemExit("Either --derive-from + --annotations-out OR --retrieval + --annotations + --out")


if __name__ == "__main__":
    main()
