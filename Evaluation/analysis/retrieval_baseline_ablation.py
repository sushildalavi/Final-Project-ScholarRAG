#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json, ready_documents, utc_now_iso
from backend.pdf_ingest import search_chunks
from backend.services.db import fetchall

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _tokenize(text: str | None) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1]


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _bootstrap_ci(values: list[float], *, n_boot: int = 2000, seed: int = 13) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(max(200, int(n_boot))):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return {"mean": _safe_mean(values), "lo": float(lo), "hi": float(hi)}


def _dcg(binary_relevance: list[int], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(binary_relevance[:k], start=1):
        score += rel / math.log2(i + 1)
    return score


def _ndcg_for_gold(pred_doc_ids: list[int], gold_doc_id: int, k: int = 10) -> float:
    rel = [1 if int(doc_id) == int(gold_doc_id) else 0 for doc_id in pred_doc_ids[:k]]
    ideal = [1] + [0] * (k - 1)
    denom = _dcg(ideal, k)
    if denom <= 0:
        return 0.0
    return _dcg(rel, k) / denom


def _paper_key_from_name(name: str | None) -> str | None:
    p = (name or "").lower()
    checks = (
        ("dense passage retrieval", "dpr"),
        ("colbert", "colbert"),
        ("retrieval-augmented generation", "rag"),
        ("retrieval augmented generation", "rag"),
        ("beir", "beir"),
        ("squad", "squad"),
        ("natural questions", "naturalquestions"),
        ("drqa", "drqa"),
        ("pegasus", "pegasus"),
        ("bart", "bart"),
        ("factscore", "factscore"),
        ("bert", "bert"),
        ("attention is all you need", "attention"),
        ("chain-of-thought", "chainofthought"),
        ("chain of thought", "chainofthought"),
        ("instructgpt", "instructgpt"),
        ("llm as a judge", "llmasjudge"),
        ("llmasjudge", "llmasjudge"),
    )
    for needle, key in checks:
        if needle in p:
            return key
    return None


def _doc_key_from_title(title: str | None) -> str:
    raw = (title or "").strip().lower()
    base = raw.replace(".pdf", "")
    base = re.sub(r"^[0-9]+[_-]?", "", base)
    return _norm(base)


@dataclass
class EvalCase:
    query: str
    gold_doc_id: int
    paper: str
    case_id: str


def _resolve_cases(eval_set: Path, docs: list[dict[str, Any]]) -> tuple[list[EvalCase], list[dict[str, Any]], list[int]]:
    raw = json.loads(eval_set.read_text())
    cases = raw if isinstance(raw, list) else (raw.get("cases") or raw.get("queries") or [])

    docs_by_id = {int(row["doc_id"]): row for row in docs if row.get("doc_id") is not None}
    docs_by_key = {_doc_key_from_title(row.get("title")): int(row["doc_id"]) for row in docs}

    usable: list[EvalCase] = []
    skipped: list[dict[str, Any]] = []
    resolved_doc_ids: set[int] = set()

    for idx, row in enumerate(cases, start=1):
        query = str(row.get("query") or "").strip()
        if not query:
            skipped.append({"index": idx, "reason": "empty_query"})
            continue

        paper_name = str(row.get("paper") or row.get("paper_title") or "")
        gold_doc_id: int | None = None
        if paper_name:
            key = _paper_key_from_name(paper_name)
            if key:
                for doc_key, doc_id in docs_by_key.items():
                    if key in doc_key:
                        gold_doc_id = doc_id
                        break

        expected_doc = row.get("expected_doc_id")
        if gold_doc_id is None and expected_doc is not None:
            try:
                candidate = int(expected_doc)
                if candidate in docs_by_id:
                    gold_doc_id = candidate
            except Exception:
                gold_doc_id = None

        if gold_doc_id is None:
            skipped.append(
                {
                    "index": idx,
                    "query": query,
                    "paper": paper_name,
                    "expected_doc_id": expected_doc,
                    "reason": "cannot_resolve_gold_doc_id",
                }
            )
            continue

        case_id = str(row.get("id") or f"case_{idx}")
        usable.append(EvalCase(query=query, gold_doc_id=int(gold_doc_id), paper=paper_name, case_id=case_id))
        resolved_doc_ids.add(int(gold_doc_id))

    return usable, skipped, sorted(resolved_doc_ids)


def _dense_rank(query: str, *, k: int, doc_ids: list[int]) -> tuple[list[int], dict[int, float]]:
    rows = search_chunks({"q": query, "k": max(60, k * 6), "doc_ids": doc_ids})["results"]
    by_doc_best: dict[int, float] = {}
    order: list[int] = []
    for row in rows:
        try:
            doc_id = int(row.get("document_id"))
            dist = float(row.get("distance", 1.0) or 1.0)
        except Exception:
            continue
        sim = max(0.0, 1.0 - min(1.0, dist))
        if doc_id not in by_doc_best:
            by_doc_best[doc_id] = sim
            order.append(doc_id)
        else:
            by_doc_best[doc_id] = max(by_doc_best[doc_id], sim)
    order = sorted(order, key=lambda d: by_doc_best.get(d, 0.0), reverse=True)
    return order[:k], by_doc_best


def _build_doc_bm25_index(doc_ids: list[int]) -> dict[str, Any]:
    rows = fetchall(
        """
        SELECT chunks.document_id, chunks.text
        FROM chunks
        JOIN documents ON documents.id = chunks.document_id
        WHERE documents.status='ready' AND chunks.document_id = ANY(%s)
        """,
        [doc_ids],
    )

    docs_tokens: dict[int, list[str]] = defaultdict(list)
    for row in rows:
        try:
            doc_id = int(row.get("document_id"))
        except Exception:
            continue
        docs_tokens[doc_id].extend(_tokenize(row.get("text")))

    N = len(docs_tokens)
    df: Counter[str] = Counter()
    lengths = {}
    tf_map: dict[int, Counter[str]] = {}
    for doc_id, tokens in docs_tokens.items():
        tf = Counter(tokens)
        tf_map[doc_id] = tf
        lengths[doc_id] = len(tokens)
        for tok in tf.keys():
            df[tok] += 1
    avgdl = (sum(lengths.values()) / max(1, len(lengths))) if lengths else 1.0
    return {
        "N": N,
        "df": df,
        "tf_map": tf_map,
        "doc_len": lengths,
        "avgdl": avgdl,
    }


def _bm25_score(query_tokens: list[str], tf: Counter[str], *, N: int, df: Counter[str], dl: int, avgdl: float) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0
    if not query_tokens:
        return 0.0
    for term in query_tokens:
        if term not in tf:
            continue
        n_qi = df.get(term, 0)
        idf = math.log(1.0 + ((N - n_qi + 0.5) / (n_qi + 0.5)))
        f = tf[term]
        denom = f + k1 * (1.0 - b + b * (dl / max(1e-9, avgdl)))
        score += idf * ((f * (k1 + 1.0)) / max(1e-9, denom))
    return score


def _sparse_rank(query: str, *, k: int, index: dict[str, Any]) -> tuple[list[int], dict[int, float]]:
    q_tokens = _tokenize(query)
    scores: dict[int, float] = {}
    for doc_id, tf in index["tf_map"].items():
        scores[doc_id] = _bm25_score(
            q_tokens,
            tf,
            N=index["N"],
            df=index["df"],
            dl=index["doc_len"].get(doc_id, 0),
            avgdl=index["avgdl"],
        )
    ranked = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
    return ranked[:k], scores


def _rrf_merge(rank_a: list[int], rank_b: list[int], *, k: int, c: int = 60) -> list[int]:
    pos_a = {doc_id: i + 1 for i, doc_id in enumerate(rank_a)}
    pos_b = {doc_id: i + 1 for i, doc_id in enumerate(rank_b)}
    all_docs = set(rank_a) | set(rank_b)
    scored = []
    for doc in all_docs:
        s = 0.0
        if doc in pos_a:
            s += 1.0 / (c + pos_a[doc])
        if doc in pos_b:
            s += 1.0 / (c + pos_b[doc])
        scored.append((doc, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k]]


def _lexical_rerank(query: str, candidates: list[int], *, dense_scores: dict[int, float], sparse_scores: dict[int, float], k: int) -> list[int]:
    q_terms = set(_tokenize(query))
    scored = []
    sparse_max = max([sparse_scores.get(doc, 0.0) for doc in candidates] + [1.0])
    for rank, doc_id in enumerate(candidates, start=1):
        sparse_norm = sparse_scores.get(doc_id, 0.0) / max(1e-9, sparse_max)
        dense = dense_scores.get(doc_id, 0.0)
        overlap_bonus = 0.0
        if q_terms:
            # Small bonus for candidate docs already strongly represented in sparse terms.
            overlap_bonus = min(1.0, sparse_norm) * 0.15
        score = (0.55 * sparse_norm) + (0.35 * dense) + overlap_bonus + (0.10 / rank)
        scored.append((doc_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored[:k]]


def _per_case_metrics(pred_doc_ids: list[int], gold_doc_id: int) -> dict[str, float]:
    hit1 = 1.0 if pred_doc_ids[:1] and int(pred_doc_ids[0]) == int(gold_doc_id) else 0.0
    hit5 = 1.0 if int(gold_doc_id) in [int(d) for d in pred_doc_ids[:5]] else 0.0
    hit10 = 1.0 if int(gold_doc_id) in [int(d) for d in pred_doc_ids[:10]] else 0.0
    rr = 0.0
    for i, doc_id in enumerate(pred_doc_ids[:10], start=1):
        if int(doc_id) == int(gold_doc_id):
            rr = 1.0 / i
            break
    ndcg = _ndcg_for_gold(pred_doc_ids, gold_doc_id, k=10)
    return {"recall@1": hit1, "recall@5": hit5, "recall@10": hit10, "mrr": rr, "ndcg@10": ndcg}


def _aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]
    out = {}
    for metric in metrics:
        vals = [float(r["metrics"][metric]) for r in rows]
        out[metric] = _bootstrap_ci(vals)
    misses = [r for r in rows if not r["metrics"]["recall@1"]]
    return {
        "n_cases": len(rows),
        "metrics": out,
        "top1_errors": len(misses),
        "top1_error_examples": misses[:15],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval baseline ablation for uploaded corpus: dense, BM25, hybrid (RRF), hybrid+rerank."
    )
    parser.add_argument("--eval-set", default="Evaluation/data/retrieval/golden_set.json")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default="Evaluation/data/post_fix/retrieval_ablation_120q.json")
    parser.add_argument("--fig", default="Evaluation/figures/post_fix/retrieval_ablation_120q.png")
    args = parser.parse_args()

    docs = ready_documents()
    if not docs:
        raise SystemExit("No ready uploaded documents found.")

    eval_set = Path(args.eval_set)
    if not eval_set.exists():
        raise SystemExit(f"Missing eval set: {eval_set}")

    cases, skipped, benchmark_doc_ids = _resolve_cases(eval_set, docs)
    if not cases:
        raise SystemExit("No usable eval cases after resolving gold doc IDs.")

    sparse_index = _build_doc_bm25_index(benchmark_doc_ids)
    if sparse_index["N"] == 0:
        raise SystemExit("No indexed docs found for BM25 baseline.")

    mode_rows: dict[str, list[dict[str, Any]]] = {
        "dense": [],
        "bm25": [],
        "hybrid_rrf": [],
        "hybrid_rrf_rerank": [],
    }

    for idx, case in enumerate(cases, start=1):
        if idx % 20 == 0:
            print(f"[{idx}/{len(cases)}] {case.case_id}", flush=True)
        dense_rank, dense_scores = _dense_rank(case.query, k=max(20, args.k), doc_ids=benchmark_doc_ids)
        sparse_rank, sparse_scores = _sparse_rank(case.query, k=max(20, args.k), index=sparse_index)
        hybrid_rank = _rrf_merge(dense_rank, sparse_rank, k=max(20, args.k))
        rerank = _lexical_rerank(case.query, hybrid_rank[:20], dense_scores=dense_scores, sparse_scores=sparse_scores, k=max(20, args.k))

        mode_preds = {
            "dense": dense_rank[: args.k],
            "bm25": sparse_rank[: args.k],
            "hybrid_rrf": hybrid_rank[: args.k],
            "hybrid_rrf_rerank": rerank[: args.k],
        }
        for mode, pred in mode_preds.items():
            row = {
                "case_id": case.case_id,
                "query": case.query,
                "paper": case.paper,
                "gold_doc_id": case.gold_doc_id,
                "pred_doc_ids": pred,
                "metrics": _per_case_metrics(pred, case.gold_doc_id),
            }
            mode_rows[mode].append(row)

    summary = {
        "mode": "retrieval_baseline_ablation",
        "created_at": utc_now_iso(),
        "eval_set": str(eval_set),
        "k": int(args.k),
        "n_ready_docs": len(docs),
        "n_cases_total": len(json.loads(eval_set.read_text())),
        "n_cases_used": len(cases),
        "n_cases_skipped": len(skipped),
        "skipped_cases": skipped[:40],
        "modes": {mode: _aggregate_mode(rows) for mode, rows in mode_rows.items()},
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(summary, out_path)

    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    mode_names = list(summary["modes"].keys())
    r1 = [summary["modes"][m]["metrics"]["recall@1"]["mean"] for m in mode_names]
    r10 = [summary["modes"][m]["metrics"]["recall@10"]["mean"] for m in mode_names]
    mrr = [summary["modes"][m]["metrics"]["mrr"]["mean"] for m in mode_names]

    x = np.arange(len(mode_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, r1, w, label="Recall@1")
    ax.bar(x, r10, w, label="Recall@10")
    ax.bar(x + w, mrr, w, label="MRR")
    ax.set_xticks(x)
    ax.set_xticklabels(mode_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Retrieval Ablation (120q, doc-level)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {out_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
