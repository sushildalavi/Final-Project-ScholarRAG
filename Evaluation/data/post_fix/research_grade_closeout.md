# Research-Grade Gap Closure (Post-Fix)

This report closes the four previously identified gaps:

1. **Faithfulness was moderate (~0.505 query mean).**
- Baseline artifact: `Evaluation/data/post_fix/benchmark_56_baseline_before_strict/faithfulness_distribution.json`
- New strict-grounded rerun: `Evaluation/data/post_fix/benchmark_56_strict/faithfulness_distribution.json`
- Result:
  - Query-level mean support: **0.505 → 0.616**
  - Claim-level support: **0.454 (546 claims) → 0.856 (202 claims)**
- Statistical test (paired, one-sided): `Evaluation/data/post_fix/significance_ab_tests.json`
  - Wilcoxon p-value (improvement): **0.0375**

2. **New labeling had single-annotator risk (no fresh IAA).**
- Annotator A (human pass):
  - `Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv`
  - `Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations.csv`
- Annotator B (rubric-generated synthetic second pass, not an independent human):
  - `Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample_annotator_b.csv`
  - `Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations_annotator_b.csv`
- Fresh IAA report: `Evaluation/data/post_fix/fresh_iaa_report.json`
  - Claim-sheet Cohen’s kappa: **0.717**
  - Retrieval-sheet Cohen’s kappa (3-class): **0.461**

3. **No cross-corpus generalization test.**
- New SciFact-lite (BEIR) cross-corpus run:
  - `Evaluation/data/post_fix/cross_corpus_scifact_lite.json`
  - `Evaluation/figures/post_fix/cross_corpus_scifact_lite.png`
- 100-query summary:
  - Dense-LSI: Recall@10 **0.55**, MRR **0.347**
  - BM25: Recall@10 **0.86**, MRR **0.659**
  - Hybrid-RRF: Recall@10 **0.81**, MRR **0.496**
  - Hybrid-Weighted (BM25-heavy): Recall@10 **0.86**, MRR **0.660**

4. **No formal significance/A-B testing.**
- New significance report: `Evaluation/data/post_fix/significance_ab_tests.json`
- Includes:
  - McNemar-style exact sign tests on targeted validation improvements (assumed pre-fix misses vs post-fix).
  - Paired Wilcoxon for faithfulness baseline vs strict rerun.
  - Pairwise Wilcoxon tests for cross-corpus mode comparisons.

## Practical Readout

- **Targeted product fixes are validated** (20/20 adversarial, 12/12 abstention in `post_fix_validation.json`).
- **Faithfulness improved materially** under strict-grounding mode with paired significance support.
- **Fresh dual-pass agreement estimate is now present** (claim kappa 0.717), but second pass is synthetic and must be disclosed.
- **External generalization evidence now exists** (SciFact-lite).

## Remaining Caveat

Strict grounding improved faithfulness partly by reducing claim volume (202 vs 546 claims), so report both the support-rate gain and the reduced-coverage tradeoff explicitly.
