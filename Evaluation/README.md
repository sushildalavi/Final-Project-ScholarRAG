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

| Metric | Value |
|--------|-------|
| Recall@10 (uploaded, cross-doc) | 0.942 |
| MRR (uploaded) | 0.908 |
| Faithfulness (LLM judge) | 90.9% supported |
| Calibration accuracy (M+S+A) | 1.0000 |
| Brier score (M+S+A) | 0.0013 |
| ECE (M+S+A) | 0.0193 |
| Cohen's kappa (IAA) | 0.8203 |
| Public search providers | 6/7 active |

## Running the Notebook

```bash
cd Evaluation
jupyter notebook ScholarRAG_Evaluation.ipynb
```

Or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute ScholarRAG_Evaluation.ipynb
```
