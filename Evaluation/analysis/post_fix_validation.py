#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.open_eval import dump_json, ready_documents, utc_now_iso
import backend.app as app_module


class _DummyCompletions:
    def create(self, **kwargs):
        content = "Validation answer [S1]."
        return type(
            "Completion",
            (),
            {"choices": [type("Choice", (), {"message": type("Msg", (), {"content": content})()})()]},
        )()


class _DummyClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _DummyCompletions()})()


@contextmanager
def _patched_client(use_live_llm: bool):
    if use_live_llm:
        yield
        return
    old_client = app_module.client
    app_module.client = _DummyClient()
    try:
        yield
    finally:
        app_module.client = old_client


def _load_queries(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must be a JSON object with a `queries` list.")
    queries = payload.get("queries")
    if not isinstance(queries, list) or not queries:
        raise SystemExit(f"{path} must contain a non-empty `queries` list.")
    return payload, queries


def _norm(text: str | None) -> str:
    return (text or "").strip().lower()


def _title_matches_key(title: str | None, paper_key: str | None) -> bool:
    title_norm = _norm(title)
    key_norm = _norm(paper_key).replace(".pdf", "")
    if not title_norm or not key_norm:
        return False
    return key_norm in title_norm or title_norm.startswith(key_norm)


def _retrieval_policy_mode(response: dict[str, Any]) -> str:
    policy = response.get("retrieval_policy")
    if isinstance(policy, dict):
        return str(policy.get("mode") or "")
    return ""


def _is_abstention_response(response: dict[str, Any]) -> bool:
    label = _norm(((response.get("confidence") or {}) if isinstance(response.get("confidence"), dict) else {}).get("label"))
    answer = _norm(response.get("answer"))
    mode = _norm(_retrieval_policy_mode(response))
    return (
        "abstain" in label
        or mode == "abstention"
        or "insufficient evidence" in answer
        or "couldn't find reliable matching evidence" in answer
    )


def _run_query(query: str, *, k: int, run_judge_llm: bool, use_live_llm: bool) -> dict[str, Any]:
    payload = {
        "query": query,
        "scope": "uploaded",
        "k": int(k),
        "run_judge": False,
        "run_judge_llm": bool(run_judge_llm),
    }
    with _patched_client(use_live_llm):
        return app_module.assistant_answer(payload)


def _evaluate_adversarial(rows: list[dict[str, Any]], *, k: int, run_judge_llm: bool, use_live_llm: bool) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    success_count = 0

    for idx, row in enumerate(rows, start=1):
        print(f"[adversarial {idx}/{len(rows)}] {row.get('id')}: {row.get('query')}", flush=True)
        response = _run_query(str(row.get("query") or ""), k=k, run_judge_llm=run_judge_llm, use_live_llm=use_live_llm)
        citations = response.get("citations") or []
        top_title = citations[0].get("title") if citations else None
        gold_key = row.get("gold_paper_key")
        blocked_rank1 = row.get("expected_topk_must_exclude_from_rank_1") or []
        rank1_ok = _title_matches_key(top_title, gold_key)
        excluded_ok = not any(_title_matches_key(top_title, bad_key) for bad_key in blocked_rank1)
        passed = bool(rank1_ok and excluded_ok and not _is_abstention_response(response))
        if passed:
            success_count += 1

        results.append(
            {
                "id": row.get("id"),
                "query": row.get("query"),
                "gold_paper_key": gold_key,
                "rank1_title": top_title,
                "top_titles": [c.get("title") for c in citations[:5]],
                "answer_preview": (response.get("answer") or "")[:300],
                "retrieval_mode": _retrieval_policy_mode(response),
                "passed": passed,
                "checks": {
                    "rank1_matches_gold": rank1_ok,
                    "rank1_excludes_blocked_keys": excluded_ok,
                    "abstained": _is_abstention_response(response),
                },
            }
        )

    return {
        "count": len(rows),
        "passed": success_count,
        "pass_rate": round(success_count / max(1, len(rows)), 4),
        "results": results,
    }


def _evaluate_abstention(rows: list[dict[str, Any]], *, k: int, run_judge_llm: bool, use_live_llm: bool) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    success_count = 0

    for idx, row in enumerate(rows, start=1):
        print(f"[abstention {idx}/{len(rows)}] {row.get('id')}: {row.get('query')}", flush=True)
        response = _run_query(str(row.get("query") or ""), k=k, run_judge_llm=run_judge_llm, use_live_llm=use_live_llm)
        citations = response.get("citations") or []
        abstained = _is_abstention_response(response)
        passed = bool(abstained and not citations)
        if passed:
            success_count += 1

        results.append(
            {
                "id": row.get("id"),
                "query": row.get("query"),
                "trap_type": row.get("trap_type"),
                "answer_preview": (response.get("answer") or "")[:300],
                "retrieval_mode": _retrieval_policy_mode(response),
                "citations_count": len(citations),
                "top_titles": [c.get("title") for c in citations[:5]],
                "passed": passed,
                "checks": {
                    "abstained": abstained,
                    "returned_no_citations": len(citations) == 0,
                },
            }
        )

    return {
        "count": len(rows),
        "passed": success_count,
        "pass_rate": round(success_count / max(1, len(rows)), 4),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run targeted post-fix validation on adversarial and abstention query sets.")
    parser.add_argument("--adversarial", default="Evaluation/queries/queries_adversarial.json")
    parser.add_argument("--abstention", default="Evaluation/queries/queries_abstention.json")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--out", default="Evaluation/data/post_fix/post_fix_validation.json")
    parser.add_argument("--run-judge-llm", action="store_true", help="Kept for parity with other eval entry points; not needed for this validation.")
    parser.add_argument("--use-live-llm", action="store_true", help="Use the real text generation call instead of the stubbed validator response.")
    args = parser.parse_args()

    docs = ready_documents()
    if not docs:
        raise SystemExit("No ready uploaded documents found.")

    adv_meta, adv_queries = _load_queries(args.adversarial)
    abs_meta, abs_queries = _load_queries(args.abstention)

    report = {
        "mode": "post_fix_validation",
        "created_at": utc_now_iso(),
        "k": int(args.k),
        "document_count": len(docs),
        "documents": [{"doc_id": d["doc_id"], "title": d["title"]} for d in docs],
        "adversarial": {
            "description": adv_meta.get("description"),
            "notes": adv_meta.get("notes"),
            **_evaluate_adversarial(
                adv_queries,
                k=args.k,
                run_judge_llm=bool(args.run_judge_llm),
                use_live_llm=bool(args.use_live_llm),
            ),
        },
        "abstention": {
            "description": abs_meta.get("description"),
            "notes": abs_meta.get("notes"),
            **_evaluate_abstention(
                abs_queries,
                k=args.k,
                run_judge_llm=bool(args.run_judge_llm),
                use_live_llm=bool(args.use_live_llm),
            ),
        },
    }
    dump_json(report, args.out)
    print(f"Wrote post-fix validation report to {args.out}")


if __name__ == "__main__":
    main()
