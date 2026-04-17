# ScholarRAG Evaluation

All evaluation data, figures, and the analysis notebook live here.

## Structure

```
Evaluation/
├── ScholarRAG_Evaluation.ipynb   # Main notebook (all graphs + stats)
├── README.md
├── data/
│   ├── calibration/              # M/S/A calibration & ablation results
│   │   ├── calibration_eval_final_report.json   # Ablation study (8 feature combos)
│   │   ├── calibration_eval_latest.json
│   │   ├── calibration_fit_response.json
│   │   └── calibration_records.json              # 634 calibration records
│   ├── human_labels/             # Human-annotated ground truth
│   │   ├── claim_annotations_human_completed.csv
│   │   ├── claim_scores_scored.csv               # 634 claims with MSA scores
│   │   ├── corpus_doc_relevance_human_completed.csv
│   │   ├── retrieval_annotations_human_completed.csv
│   │   ├── msa_by_label_summary.json
│   │   ├── claim_score_summary.json
│   │   ├── retrieval_metrics.json
│   │   ├── msa_labeling_template.jsonl
│   │   └── csv_exports/          # Raw CSV exports from eval runs
│   ├── iaa/                      # Inter-annotator agreement
│   │   └── iaa_report.json       # Cohen's kappa = 0.82
│   ├── llm_judge/                # LLM-as-judge faithfulness
│   │   ├── judge_claims_extracted.json   # 638 claims
│   │   ├── judge_eval_cases.json
│   │   └── judge_eval_final.json         # Full judge run output
│   ├── public_search/            # Public research API evaluation
│   │   └── public_search_eval.json       # 20 queries × 6 providers
│   └── retrieval/                # Dense retrieval evaluation
│       ├── golden_set.json
│       ├── retrieval_eval_120q_crossdoc.json  # Main: 120 cross-doc queries
│       ├── retrieval_eval_120q_final.json
│       ├── retrieval_eval_20q_result.json
│       └── retrieval_eval_cases.json
├── figures/                      # All generated PNG figures
│   ├── fig_retrieval.png
│   ├── fig_latency.png
│   ├── fig_faithfulness.png
│   ├── fig_msa_distributions.png
│   ├── fig_msa_means.png
│   ├── fig_ablation.png
│   ├── fig_weights.png
│   ├── fig_iaa.png
│   ├── fig_public_providers.png
│   ├── fig_public_relevance.png
│   ├── fig_public_provider_heatmap.png
│   ├── fig_summary_dashboard.png
│   ├── fig_train_vs_test.png
│   ├── fig_dataset.png
│   └── fig_confidence_dist.png
├── papers/                       # 15 landmark papers used for evaluation
│   ├── 01_DPR.pdf ... 15_LLMasJudge.pdf
└── queries/                      # Query sets
    ├── queries_120_master.json   # 120 cross-document queries
    ├── queries_56_runnable.json
    ├── queries_60_master.json
    └── queries_template.json
```

## Key Results

Use the post-fix artifacts for reporting:

- `Evaluation/data/post_fix/post_fix_validation.json`
- `Evaluation/data/post_fix/retrieval_ablation_120q.json`
- `Evaluation/data/post_fix/benchmark_56_strict/headline_metrics.json`
- `Evaluation/data/post_fix/benchmark_56_strict/presentation_summary.md`
- `Evaluation/data/post_fix/fresh_iaa_report.json`
- `Evaluation/data/post_fix/cross_corpus_scifact_lite.json`
- `Evaluation/data/post_fix/significance_ab_tests.json`

Current headline (post-fix):

| Metric | Value |
|--------|-------|
| Adversarial targeted validation | 20/20 passed |
| Abstention targeted validation | 12/12 passed |
| Retrieval ablation (120q, dense) | Recall@1 = 0.750, Recall@10 = 1.000, MRR = 0.838 |
| Retrieval ablation (120q, BM25) | Recall@1 = 0.558, Recall@10 = 0.983, MRR = 0.699 |
| Retrieval ablation (120q, hybrid RRF) | Recall@1 = 0.725, Recall@10 = 1.000, MRR = 0.819 |
| Retrieval ablation (120q, hybrid RRF + rerank) | Recall@1 = 0.650, Recall@10 = 0.983, MRR = 0.764 |
| Strict-grounded faithfulness (56q benchmark) | query mean support = 0.616 (95% CI: 0.489–0.732) |
| Strict-grounded claim support (56q benchmark) | 0.856 over 202 sentence-level claims |
| Faithfulness coverage tradeoff (baseline → strict) | judged claims 546 → 202 |
| Legacy 634-claim clean-A calibration | `claim_scores_scored_clean.csv` + robustness report generated |
| Fresh claim IAA on new 50-row sheet | Cohen's kappa = 0.717 |
| Fresh claim IAA disclosure | annotator-B is rubric-generated (synthetic), not an independent second human |
| Fresh retrieval IAA on 1200 rows | Cohen's kappa (3-class) = 0.461 |
| Cross-corpus SciFact-lite (100 queries) | BM25 Recall@10 = 0.860, MRR = 0.659 |
| Cross-corpus SciFact-lite (100 queries) | Dense Recall@10 = 0.550, Hybrid-RRF Recall@10 = 0.810 |
| Cross-corpus SciFact-lite (100 queries) | Hybrid-Weighted Recall@10 = 0.860, MRR = 0.660 |

## Running the Notebook

```bash
cd Evaluation
jupyter notebook ScholarRAG_Evaluation.ipynb
```

Or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute ScholarRAG_Evaluation.ipynb
```

## Repro Commands (Required)

```bash
make eval-postfix-validation
make eval-retrieval-ablation
make eval-regenerate-clean-a
python Evaluation/analysis/cross_corpus_scifact_lite.py \
  --out Evaluation/data/post_fix/cross_corpus_scifact_lite.json \
  --fig Evaluation/figures/post_fix/cross_corpus_scifact_lite.png \
  --n-queries 100
python Evaluation/analysis/calibration_robustness.py \
  --claims Evaluation/data/human_labels/claim_scores_scored_clean.csv \
  --out-dir Evaluation/data/post_fix/human_labels_clean \
  --fig-dir Evaluation/figures/post_fix/human_labels_clean
make eval-retrieval-human-template
```

Human-in-the-loop judge validation workflow:

```bash
make eval-human-judge-sample
# Manually fill human_label in:
# Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv
make eval-human-judge-score
python scripts/open_eval_generate_annotator_b.py \
  --judge-a Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv \
  --judge-b-out Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample_annotator_b.csv \
  --retrieval-a Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations.csv \
  --retrieval-b-out Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations_annotator_b.csv \
  --corpus-a Evaluation/data/post_fix/retrieval_120q_human_template/corpus_doc_relevance.csv \
  --corpus-b-out Evaluation/data/post_fix/retrieval_120q_human_template/corpus_doc_relevance_annotator_b.csv
python Evaluation/analysis/compute_fresh_iaa.py \
  --judge-a Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv \
  --judge-b Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample_annotator_b.csv \
  --retrieval-a Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations.csv \
  --retrieval-b Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations_annotator_b.csv \
  --out Evaluation/data/post_fix/fresh_iaa_report.json
python Evaluation/analysis/significance_ab_tests.py \
  --baseline-validation Evaluation/data/post_fix/pre_fix_validation_assumed.json \
  --post-validation Evaluation/data/post_fix/post_fix_validation.json \
  --baseline-judge Evaluation/data/post_fix/benchmark_56_baseline_before_strict/judge_eval_post_fix.json \
  --post-judge Evaluation/data/post_fix/benchmark_56_strict/judge_eval_post_fix.json \
  --cross-corpus Evaluation/data/post_fix/cross_corpus_scifact_lite.json \
  --out Evaluation/data/post_fix/significance_ab_tests.json
```

Chunk-level retrieval scoring after human labels are filled:

```bash
python scripts/open_eval_score_retrieval.py \
  --annotations Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations.csv \
  --corpus-docs Evaluation/data/post_fix/retrieval_120q_human_template/corpus_doc_relevance.csv \
  --output Evaluation/data/post_fix/retrieval_120q_human_metrics.json
```
