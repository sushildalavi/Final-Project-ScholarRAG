#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if len(t) > 1]


def _ensure_scifact(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "scifact.zip"
    extract_dir = root / "scifact"
    corpus_file = extract_dir / "corpus.jsonl"
    if corpus_file.exists():
        return extract_dir
    if not zip_path.exists():
        urllib.request.urlretrieve(SCIFACT_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    if not corpus_file.exists():
        raise SystemExit(f"SciFact corpus missing after extraction: {corpus_file}")
    return extract_dir


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(row["score"])
            if score > 0:
                qrels[qid].add(did)
    return qrels


def _dcg(vals: list[int], k: int) -> float:
    out = 0.0
    for i, v in enumerate(vals[:k], start=1):
        out += v / math.log2(i + 1)
    return out


def _metrics_for_query(ranked_doc_ids: list[str], gold: set[str], k: int = 10) -> dict[str, float]:
    recall = 1.0 if any(d in gold for d in ranked_doc_ids[:k]) else 0.0
    rr = 0.0
    for i, did in enumerate(ranked_doc_ids[:k], start=1):
        if did in gold:
            rr = 1.0 / i
            break
    rel = [1 if d in gold else 0 for d in ranked_doc_ids[:k]]
    ideal = [1] * min(len(gold), k)
    ndcg = 0.0
    denom = _dcg(ideal, k)
    if denom > 0:
        ndcg = _dcg(rel, k) / denom
    return {"recall@10": recall, "mrr": rr, "ndcg@10": ndcg}


def _bootstrap_ci(vals: list[float], n_boot: int = 2000, seed: int = 23) -> dict[str, float]:
    rng = random.Random(seed)
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    boots = []
    for _ in range(max(200, n_boot)):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))]
    return {"mean": sum(vals) / n, "lo": lo, "hi": hi}


def _bm25_rank(
    query: str, doc_ids: list[str], tf_map: dict[str, Counter], df: Counter, avgdl: float
) -> tuple[list[str], dict[str, float]]:
    q_tokens = _tokens(query)
    N = len(doc_ids)
    k1 = 1.5
    b = 0.75
    scores = {}
    for did in doc_ids:
        tf = tf_map[did]
        dl = sum(tf.values())
        score = 0.0
        for term in q_tokens:
            if term not in tf:
                continue
            n_qi = df.get(term, 0)
            idf = math.log(1.0 + ((N - n_qi + 0.5) / (n_qi + 0.5)))
            f = tf[term]
            denom = f + k1 * (1 - b + b * (dl / max(1e-9, avgdl)))
            score += idf * ((f * (k1 + 1.0)) / max(1e-9, denom))
        scores[did] = score
    ranked = sorted(doc_ids, key=lambda d: scores[d], reverse=True)
    return ranked, scores


def _rrf(a: list[str], b: list[str], k: int, c: int = 60) -> list[str]:
    pa = {d: i + 1 for i, d in enumerate(a)}
    pb = {d: i + 1 for i, d in enumerate(b)}
    all_ids = set(a) | set(b)
    scored = []
    for did in all_ids:
        s = 0.0
        if did in pa:
            s += 1 / (c + pa[did])
        if did in pb:
            s += 1 / (c + pb[did])
        scored.append((did, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:k]]


def _rank_score_map(ranked: list[str]) -> dict[str, float]:
    return {d: 1.0 / (i + 1) for i, d in enumerate(ranked)}


def _weighted_hybrid_rank(dense_rank: list[str], bm25_rank: list[str], k: int, w_bm25: float = 0.82) -> list[str]:
    dense_s = _rank_score_map(dense_rank)
    bm25_s = _rank_score_map(bm25_rank)
    all_ids = set(dense_rank[:k * 4]) | set(bm25_rank[:k * 4])
    scored = []
    w_dense = 1.0 - w_bm25
    for did in all_ids:
        score = (w_bm25 * bm25_s.get(did, 0.0)) + (w_dense * dense_s.get(did, 0.0))
        scored.append((did, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored[:k]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BEIR SciFact-lite cross-corpus generalization evaluation.")
    parser.add_argument("--out", default="Evaluation/data/post_fix/cross_corpus_scifact_lite.json")
    parser.add_argument("--fig", default="Evaluation/figures/post_fix/cross_corpus_scifact_lite.png")
    parser.add_argument("--cache-dir", default="Evaluation/data/cross_corpus_cache")
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    scifact_dir = _ensure_scifact(Path(args.cache_dir))
    corpus_rows = _load_jsonl(scifact_dir / "corpus.jsonl")
    query_rows = _load_jsonl(scifact_dir / "queries.jsonl")
    qrels = _load_qrels(scifact_dir / "qrels" / "test.tsv")

    corpus = {str(r["_id"]): f"{r.get('title','')} {r.get('text','')}" for r in corpus_rows}
    queries = {str(r["_id"]): str(r.get("text") or "") for r in query_rows}

    # keep only test queries with relevance labels
    eval_qids = [qid for qid in queries.keys() if qid in qrels and qrels[qid]]
    rng = random.Random(args.seed)
    rng.shuffle(eval_qids)
    eval_qids = eval_qids[: max(1, int(args.n_queries))]

    doc_ids = list(corpus.keys())
    doc_texts = [corpus[d] for d in doc_ids]

    # Sparse + dense (LSI over TF-IDF) retrieval.
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(doc_texts)
    svd_dim = min(256, max(16, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=svd_dim, random_state=args.seed)
    X_dense = normalize(svd.fit_transform(X))

    tf_map = {}
    df = Counter()
    for did, text in corpus.items():
        tf = Counter(_tokens(text))
        tf_map[did] = tf
        for t in tf:
            df[t] += 1
    avgdl = sum(sum(tf.values()) for tf in tf_map.values()) / max(1, len(tf_map))

    per_mode = {"dense_lsi": [], "bm25": [], "hybrid_rrf": [], "hybrid_weighted": []}
    details = []
    per_query_metrics = {"dense_lsi": [], "bm25": [], "hybrid_rrf": [], "hybrid_weighted": []}

    for idx, qid in enumerate(eval_qids, start=1):
        if idx % 20 == 0:
            print(f"[{idx}/{len(eval_qids)}] qid={qid}", flush=True)
        q = queries[qid]
        gold = qrels[qid]

        q_sparse = vec.transform([q])
        q_dense = normalize(svd.transform(q_sparse))
        sim = (X_dense @ q_dense.T).reshape(-1)
        dense_rank = [doc_ids[i] for i in np.argsort(-sim)]
        bm25_rank, _ = _bm25_rank(q, doc_ids, tf_map, df, avgdl)
        hybrid_rank = _rrf(dense_rank[:200], bm25_rank[:200], k=200)
        hybrid_weighted = _weighted_hybrid_rank(dense_rank[:300], bm25_rank[:300], k=200, w_bm25=0.82)

        for mode, rank in (
            ("dense_lsi", dense_rank),
            ("bm25", bm25_rank),
            ("hybrid_rrf", hybrid_rank),
            ("hybrid_weighted", hybrid_weighted),
        ):
            m = _metrics_for_query(rank, gold, k=10)
            per_mode[mode].append(m)
            per_query_metrics[mode].append(
                {"query_id": qid, "recall@10": m["recall@10"], "mrr": m["mrr"], "ndcg@10": m["ndcg@10"]}
            )

        details.append(
            {
                "query_id": qid,
                "query": q,
                "gold_doc_ids": sorted(gold),
                "top_dense": dense_rank[:10],
                "top_bm25": bm25_rank[:10],
                "top_hybrid": hybrid_rank[:10],
                "top_hybrid_weighted": hybrid_weighted[:10],
            }
        )

    summary = {}
    for mode, rows in per_mode.items():
        recall = [r["recall@10"] for r in rows]
        mrr = [r["mrr"] for r in rows]
        ndcg = [r["ndcg@10"] for r in rows]
        summary[mode] = {
            "n_queries": len(rows),
            "recall@10": _bootstrap_ci(recall, seed=args.seed),
            "mrr": _bootstrap_ci(mrr, seed=args.seed + 1),
            "ndcg@10": _bootstrap_ci(ndcg, seed=args.seed + 2),
        }

    report = {
        "mode": "cross_corpus_scifact_lite",
        "dataset": "BEIR/SciFact test",
        "n_docs": len(doc_ids),
        "n_queries_total_with_qrels": len([q for q in queries if q in qrels and qrels[q]]),
        "n_queries_evaluated": len(eval_qids),
        "seed": int(args.seed),
        "metrics": summary,
        "per_query_metrics": per_query_metrics,
        "details_sample": details[:30],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    modes = list(summary.keys())
    x = np.arange(len(modes))
    r10 = [summary[m]["recall@10"]["mean"] for m in modes]
    mrr = [summary[m]["mrr"]["mean"] for m in modes]
    ndcg = [summary[m]["ndcg@10"]["mean"] for m in modes]
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w, r10, w, label="Recall@10")
    ax.bar(x, mrr, w, label="MRR")
    ax.bar(x + w, ndcg, w, label="nDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Corpus Generalization: SciFact-lite")
    ax.legend()
    fig.tight_layout()
    fig_path = Path(args.fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    print(f"Wrote {out}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
