from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _metric(model_rows: list[dict], model_name: str, key: str) -> float | None:
    for row in model_rows:
        if row.get("model") == model_name:
            metric = row.get(key) or {}
            return metric.get("mean")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--fig-dir", required=True)
    parser.add_argument("--retrieval-ablation", default=None)
    parser.add_argument("--human-manifest", default=None)
    parser.add_argument("--human-report", default=None)
    parser.add_argument("--legacy-clean-calib", default=None)
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir)
    fig_dir = Path(args.fig_dir)
    validation = _load(Path(args.validation))
    faith = _load(benchmark_dir / "faithfulness_distribution.json")
    calib = _load(benchmark_dir / "calibration_robustness.json")
    retrieval_ablation = _load(Path(args.retrieval_ablation)) if args.retrieval_ablation else None
    human_manifest = _load(Path(args.human_manifest)) if args.human_manifest else None
    human_report = _load(Path(args.human_report)) if args.human_report and Path(args.human_report).exists() else None
    legacy_clean_calib = _load(Path(args.legacy_clean_calib)) if args.legacy_clean_calib else None
    remaining_failures = []
    for suite_name in ("adversarial", "abstention"):
        for item in validation.get(suite_name, {}).get("results") or []:
            if not item.get("passed", False):
                remaining_failures.append(
                    {
                        "suite": suite_name,
                        "id": item.get("id"),
                        "query": item.get("query"),
                    }
                )

    group_by_paper = (calib.get("splits") or {}).get("group_by_paper", {})
    group_by_query = (calib.get("splits") or {}).get("group_by_query", {})
    paper_models = group_by_paper.get("models") or []
    query_models = group_by_query.get("models") or []

    summary = {
        "targeted_validation": {
            "adversarial_pass_rate": validation["adversarial"]["pass_rate"],
            "adversarial_passed": validation["adversarial"]["passed"],
            "adversarial_total": validation["adversarial"]["count"],
            "abstention_pass_rate": validation["abstention"]["pass_rate"],
            "abstention_passed": validation["abstention"]["passed"],
            "abstention_total": validation["abstention"]["count"],
            "remaining_failures": remaining_failures,
        },
        "post_fix_faithfulness": {
            "queries_scored": faith["overall"]["n_queries"],
            "queries_with_sentence_claims": faith["overall"]["n_queries_with_sentence_claims"],
            "n_claims": faith["overall"]["n_claims"],
            "claim_support_rate": faith["overall"]["claim_support_rate"],
            "query_level_mean": faith["overall"]["query_level_bootstrap_ci_95"]["mean"],
            "query_level_ci_95": {
                "lo": faith["overall"]["query_level_bootstrap_ci_95"]["lo"],
                "hi": faith["overall"]["query_level_bootstrap_ci_95"]["hi"],
            },
            "query_level_median": faith["overall"]["query_level_quantiles"]["median"],
            "worst_decile_queries": faith["worst_decile_queries"],
        },
        "post_fix_calibration": {
            "claims_raw": calib["n_claims_raw"],
            "claims_labeled": calib["n_claims_labeled"],
            "claims_complete": calib["n_claims_complete"],
            "claims_skipped_incomplete_features": calib["n_claims_skipped_incomplete_features"],
            "supported_complete": calib["n_supported"],
            "unsupported_complete": calib["n_unsupported"],
            "group_by_paper_M+S": {
                "accuracy": _metric(paper_models, "M+S", "accuracy"),
                "f1_macro": _metric(paper_models, "M+S", "f1_macro"),
                "brier": _metric(paper_models, "M+S", "brier"),
                "ece": _metric(paper_models, "M+S", "ece"),
                "roc_auc": _metric(paper_models, "M+S", "roc_auc"),
                "pr_auc": _metric(paper_models, "M+S", "pr_auc"),
            },
            "group_by_query_M+S": {
                "accuracy": _metric(query_models, "M+S", "accuracy"),
                "f1_macro": _metric(query_models, "M+S", "f1_macro"),
                "brier": _metric(query_models, "M+S", "brier"),
                "ece": _metric(query_models, "M+S", "ece"),
                "roc_auc": _metric(query_models, "M+S", "roc_auc"),
                "pr_auc": _metric(query_models, "M+S", "pr_auc"),
            },
        },
        "retrieval_ablation_120q": (
            {
                mode: {
                    "recall@1": payload["metrics"]["recall@1"]["mean"],
                    "recall@10": payload["metrics"]["recall@10"]["mean"],
                    "mrr": payload["metrics"]["mrr"]["mean"],
                    "top1_errors": payload["top1_errors"],
                    "n_cases": payload["n_cases"],
                }
                for mode, payload in (retrieval_ablation.get("modes") or {}).items()
            }
            if isinstance(retrieval_ablation, dict)
            else None
        ),
        "legacy_634_clean_A_calibration": (
            {
                "n_claims_complete": legacy_clean_calib.get("n_claims_complete"),
                "label_leakage_msa_A": legacy_clean_calib.get("label_leakage_msa_A"),
                "group_by_query_M+S": {
                    "accuracy": _metric((legacy_clean_calib.get("splits") or {}).get("group_by_query", {}).get("models") or [], "M+S", "accuracy"),
                    "f1_macro": _metric((legacy_clean_calib.get("splits") or {}).get("group_by_query", {}).get("models") or [], "M+S", "f1_macro"),
                    "brier": _metric((legacy_clean_calib.get("splits") or {}).get("group_by_query", {}).get("models") or [], "M+S", "brier"),
                    "ece": _metric((legacy_clean_calib.get("splits") or {}).get("group_by_query", {}).get("models") or [], "M+S", "ece"),
                },
            }
            if isinstance(legacy_clean_calib, dict)
            else None
        ),
        "judge_human_validation": {
            "sample_manifest": human_manifest,
            "report": human_report,
            "status": (
                "complete"
                if isinstance(human_report, dict) and human_report.get("status") != "pending_human_labels"
                else "pending_human_labels"
            ),
        },
        "figure_paths": {
            "faithfulness_hist": str(fig_dir / "faithfulness_hist.png"),
            "faithfulness_by_stratum": str(fig_dir / "faithfulness_by_stratum.png"),
            "pr_group_by_query": str(fig_dir / "pr_group_by_query.png"),
            "pr_group_by_paper": str(fig_dir / "pr_group_by_paper.png"),
            "reliability_MpS_group_by_query": str(fig_dir / "reliability_MpS_group_by_query.png"),
            "reliability_MpS_group_by_paper": str(fig_dir / "reliability_MpS_group_by_paper.png"),
            "retrieval_ablation_120q": "Evaluation/figures/post_fix/retrieval_ablation_120q.png",
        },
    }

    md = f"""# ScholarRAG Post-Fix Evaluation Summary

## Headline

- Targeted adversarial retrieval validation: {validation["adversarial"]["passed"]}/{validation["adversarial"]["count"]} ({_pct(validation["adversarial"]["pass_rate"])})
- Targeted abstention validation: {validation["abstention"]["passed"]}/{validation["abstention"]["count"]} ({_pct(validation["abstention"]["pass_rate"])})
- Post-fix faithfulness query mean: {_pct(faith["overall"]["query_level_bootstrap_ci_95"]["mean"])} with 95% CI [{_pct(faith["overall"]["query_level_bootstrap_ci_95"]["lo"])}, {_pct(faith["overall"]["query_level_bootstrap_ci_95"]["hi"])}]
- Post-fix claim support rate: {_pct(faith["overall"]["claim_support_rate"])} over {faith["overall"]["n_claims"]} judged sentence-level claims
- Post-fix calibration sample: {calib["n_claims_complete"]}/{calib["n_claims_labeled"]} labeled claims had complete live M/S/A values

## Key Caveats

- The targeted validation improved the exact failure mode, but it is not perfect. Remaining misses are still present in the adversarial and abstention sets.
- The post-fix faithfulness query distribution is harsh: median {faith["overall"]["query_level_quantiles"]["median"]:.2f}, with many zero-score queries in the worst decile.
- Sentence-level judge claims were available for {faith["overall"]["n_queries_with_sentence_claims"]}/{faith["overall"]["n_queries"]} scored queries. Query-level `overall_score` covers the full scored set and is the more reliable headline.
- Calibration results are now leakage-free, but they are based on a small complete-feature subset: {calib["n_claims_complete"]} claims spanning {calib["n_queries"]} queries and {calib["n_papers"]} papers.

## Leakage-Free Calibration Snapshot

- Group-by-paper M+S accuracy: {_pct(_metric(paper_models, "M+S", "accuracy"))}
- Group-by-paper M+S macro-F1: {_metric(paper_models, "M+S", "f1_macro"):.3f}
- Group-by-paper M+S Brier: {_metric(paper_models, "M+S", "brier"):.3f}
- Group-by-paper M+S ECE: {_metric(paper_models, "M+S", "ece"):.3f}
- Group-by-paper M+S ROC-AUC: {_metric(paper_models, "M+S", "roc_auc"):.3f}
- Group-by-paper M+S PR-AUC: {_metric(paper_models, "M+S", "pr_auc"):.3f}

## Remaining Targeted Failures

"""
    for item in remaining_failures:
        md += f"- {item['suite']} `{item['id']}`: {item['query']}\n"

    md += "\n## Worst-Decile Faithfulness Queries\n\n"
    for item in faith["worst_decile_queries"]:
        md += f"- `{item['query_id']}` ({item['support_rate']:.2f}): {item['query']}\n"

    if isinstance(retrieval_ablation, dict):
        md += "\n## Retrieval Ablation (120q)\n\n"
        for mode, payload in (retrieval_ablation.get("modes") or {}).items():
            metric = payload.get("metrics") or {}
            md += (
                f"- `{mode}`: Recall@1={metric.get('recall@1', {}).get('mean', 0.0):.3f}, "
                f"Recall@10={metric.get('recall@10', {}).get('mean', 0.0):.3f}, "
                f"MRR={metric.get('mrr', {}).get('mean', 0.0):.3f}\n"
            )

    if isinstance(human_manifest, dict):
        md += "\n## Judge-vs-Human Validation\n\n"
        if isinstance(human_report, dict):
            metrics = human_report.get("metrics") or {}
            md += (
                f"- Completed sample size: {human_report.get('n_rows_scored')}\n"
                f"- Accuracy: {metrics.get('accuracy', 0.0):.3f}\n"
                f"- Cohen's kappa: {metrics.get('cohen_kappa')}\n"
            )
        else:
            md += (
                f"- Prepared sample: {human_manifest.get('sample_size')} claims.\n"
                "- Human labels are still required in `judge_human_validation_sample.csv`.\n"
            )

    (benchmark_dir / "headline_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (benchmark_dir / "presentation_summary.md").write_text(md)
    print(f"Wrote {benchmark_dir / 'headline_metrics.json'}")
    print(f"Wrote {benchmark_dir / 'presentation_summary.md'}")


if __name__ == "__main__":
    main()
