# Research-Grade Gap Closure (Post-Fix v2)

This report closes the four gaps previously identified and the three
follow-up items flagged in the audit (synthetic annotator disclosure,
strict-grounding coverage tradeoff, and bootstrap CIs on small-n tracks).

All numeric tables below come from scripts and JSON files that live in
`Evaluation/analysis/` and `Evaluation/data/post_fix/`. Re-run
`python Evaluation/analysis/ensemble_and_coverage.py` to regenerate.

---

## 1. Faithfulness — strict-grounding is a coverage tradeoff, not a free win

Artifact: `Evaluation/data/post_fix/ensemble_and_coverage.json → strict_grounding_tradeoff`

| Metric | Baseline (before strict) | Strict grounding |
|---|---|---|
| Claims evaluated | 546 | **202** |
| Raw per-claim support rate | 0.454 | 0.856 |
| Query-level mean support | 0.505 [0.397, 0.618] | 0.616 [0.489, 0.732] |
| **Coverage** (strict / baseline claims) | 1.00 | **0.37** |
| **Effective support rate** (support × coverage) | **0.454** | **0.317** |

**Honest reading.** Strict grounding does increase the *per-claim* support
rate (0.454 → 0.856) because it filters out ungrounded claims upstream,
but the system answers only 37% as many claims. Corrected for the lost
coverage, the **effective** support rate *drops* from 0.454 to 0.317. Report
both columns. The paired Wilcoxon p=0.0375 on the query-level mean
(`significance_ab_tests.json`) is valid as long as the coverage tradeoff
is disclosed alongside it.

---

## 2. Labeling quality — LLM-ensemble agreement, NOT human IAA

**Disclosure required in every downstream publication.** Annotators A, B,
and C are LLM / algorithmic passes, not independent human raters. The
pairwise kappa numbers below are *LLM-ensemble agreement used as a proxy
for IAA, not IAA itself.* The original κ=0.82 from
`Evaluation/data/iaa/iaa_report.json` is still a genuine human IAA over a
different 150-claim sample and remains the canonical human-IAA citation.

Annotator design (`Evaluation/data/post_fix/benchmark_56/`):
- **Annotator A** — rubric-based pass over the existing labeled set.
- **Annotator B** — deterministic token-overlap rule that starts from A's
  labels and applies a bounded flip heuristic. Correlates with A by
  construction.
- **Annotator C** — NLI-entailment + content-noun-overlap blend,
  independent of A/B (see `scripts/open_eval_generate_annotator_c.py`).

### Pairwise ensemble kappa on the 50-claim judge sample (bootstrap 95% CI)

| Pair | κ | 95% CI |
|---|---|---|
| A vs B | 0.717 | [0.508, 0.880] |
| A vs C | **0.180** | **[−0.101, 0.443]** |
| B vs C | 0.065 | [−0.210, 0.333] |

**Honest reading.** B-vs-A looks high because B starts from A's labels.
A-vs-C and B-vs-C are at or below chance once CI is computed — i.e., the
labeling task is genuinely hard, and different LLM rubrics disagree. Do
not describe the 0.717 as "IAA". The ensemble majority vote (2-of-3)
agrees with the GPT-4o-mini judge at κ = 0.64 [0.42, 0.84].

### Retrieval-sheet ensemble (3-class)

See `ensemble_and_coverage.json → ensemble_agreement_retrieval` for the
full pairwise numbers.

---

## 3. Adversarial + abstention — validated, with significance

- 20/20 adversarial queries, 12/12 abstention queries in `post_fix_validation.json`.
- Combined McNemar one-sided p = **0.0156**
  (`significance_ab_tests.json → validation_mcnemar → combined`).

---

## 4. Cross-corpus generalization — BEIR SciFact-lite

Artifact: `cross_corpus_scifact_lite.json`

| Mode | Recall@10 | MRR |
|---|---|---|
| Dense-LSI | 0.55 | 0.347 |
| BM25 | 0.86 | 0.659 |
| Hybrid-RRF | 0.81 | 0.496 |
| Hybrid-weighted (BM25-heavy) | 0.86 | 0.660 |

Honest finding: BM25 beats dense on out-of-distribution data, consistent
with the BEIR paper's headline finding. The project's hybrid pipeline is
justified *because* of this, not despite it.

---

## 5. Bootstrap CIs on the small-n tracks

Artifact: `ensemble_and_coverage.json → retrieval_120q_bootstrap_cis, iaa_bootstrap_cis`

- **Retrieval 120q** — all Recall@K and MRR CIs = [1.00, 1.00]. The CI
  itself is the honest reading: the benchmark is saturated by construction
  (every query names its gold paper; 15 topically-disjoint papers). Do
  not cite these numbers as evidence of retrieval quality.
- **Faithfulness 56q** — query-level means with 95% CIs:
  - Baseline: 0.505 [0.397, 0.618]
  - Strict: 0.616 [0.489, 0.732]
  - Paired Wilcoxon p = 0.0375 (one-sided)
- **IAA 50-claim ensemble** — bootstrap CIs reported per pair above,
  full JSON in `iaa_bootstrap_cis`.

---

## Practical readout for the deck / paper

### Can claim (with citations to JSONs in this repo)
- Adversarial 20/20 and abstention 12/12 pass; McNemar p = 0.016.
- Faithfulness per-claim support rate improved under strict grounding
  (paired Wilcoxon p = 0.0375) — **but effective support rate dropped
  from 0.454 to 0.317 due to coverage loss**. Quote both numbers.
- BM25 outperforms dense on BEIR SciFact-lite (0.86 vs 0.55 Recall@10);
  hybrid matches BM25 (0.86) with better MRR (0.660).
- Retrieval on the 120q internal set saturates at Recall@10 = 1.00 with
  zero-width CI. Benchmark is saturated; it is not a performance claim.

### Must disclose
- Annotators B and C are LLM / algorithmic, not human. Ensemble kappa is
  a proxy for IAA, not IAA itself. The 0.82 human-IAA number is valid
  only for the original 150-claim sample.
- Strict-grounding faithfulness gain comes with a coverage loss;
  effective support rate is the fairer comparison.
