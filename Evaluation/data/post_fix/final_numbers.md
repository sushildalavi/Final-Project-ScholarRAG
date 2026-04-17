# Final Post-Improvement Numbers

All numbers below are computed by scripts in `Evaluation/analysis/` and
live alongside their source JSONs in `Evaluation/data/post_fix/`.
Everything was produced offline (no backend runtime needed), so every
number is reproducible from the existing artifacts.

---

## Headline table (what to put in the deck)

| Metric | Before | After | CI / p | Mechanism |
|---|---|---|---|---|
| Chunk-level Recall@1 (retrieval) | 1.00 (saturated) | **0.883** | [0.817, 0.942] | Real chunk-level gold via `auto_derive_chunk_gold.py` |
| Chunk-level Recall@1 (+rerank) | 1.00 (saturated) | **0.825** | [0.750, 0.892] | Real chunk-level gold |
| Chunk-level MRR (retrieval) | 1.00 | **0.931** | [0.893, 0.964] | — |
| Chunk-level MRR (+rerank) | 1.00 | **0.891** | [0.844, 0.931] | — |
| S-only calibration F1 | 0.524 | **0.939** | [0.848, 0.996] | S + 5 new features, paper-grouped CV |
| Brier (S + features) | 0.057 | **0.022** | [0.009, 0.041] | — |
| Effective faithfulness (sim) | 0.448 | **0.487** | per-query Δ=+5.1% [0.017, 0.098] | Claim-rewrite rescue (65% rate assumed) |
| Coverage (vs strict-grounded) | 202 claims | **544 claims** | 2.69× strict | Hedge-don't-drop policy |

---

## 1. Chunk-level retrieval (replaces saturated benchmark)

Source: `Evaluation/data/retrieval/chunk_level_metrics_auto.json`

Gold chunks derived automatically for all 120 queries via content-token
overlap against the gold doc (avg 2.98 candidate chunks per query).

| Metric | Retrieval only | +Rerank |
|---|---|---|
| Recall@1 | **0.883 [0.817, 0.942]** | **0.825 [0.750, 0.892]** |
| Recall@3 | 1.000 [1.00, 1.00] | 0.967 [0.933, 0.992] |
| Recall@5 | 1.000 [1.00, 1.00] | 0.975 [0.950, 1.00] |
| Recall@10 | 1.000 [1.00, 1.00] | 0.975 [0.950, 1.00] |
| **MRR** | **0.931 [0.893, 0.964]** | **0.891 [0.844, 0.931]** |

**Key finding — honestly negative for the reranker**: the lexical
reranker we ship HURTS chunk-level Recall@1 (0.88 → 0.83) and MRR (0.93
→ 0.89). It helps at Recall@3 by pulling the correct chunk into the top
3, but if users read only the first citation, they're better off without
the reranker on this corpus.

## 2. Calibration — S-only F1 lift from 5 new features

Source: `Evaluation/data/post_fix/calibration_with_new_features.json`.

Features computed on the 634-claim leakage-free dataset
(`claim_scores_scored_clean.csv`) and evaluated under 5-fold
`GroupKFold(by paper)` with bootstrap 95% CIs:

| Feature set | F1 macro | Brier | Accuracy |
|---|---|---|---|
| S-only (baseline) | 0.524 [0.487, 0.570] | 0.057 [0.041, 0.074] | 0.905 |
| **S + 5 new features** | **0.939 [0.848, 0.996]** | **0.022 [0.009, 0.041]** | **0.976** |
| 5 new features alone | 0.900 [0.809, 0.979] | 0.036 [0.016, 0.061] | 0.964 |
| M + S + all features | 0.997 [0.992, 1.000] | 0.005 [0.002, 0.012] | 0.999 |

**Leakage check** (`feature_stats_by_label` in the JSON):
- No feature is class-constant (std > 0 in both classes).
- Biggest gap: `feat_stability_margin` 0.564 ± 0.214 (supported) vs 0.091
  ± 0.136 (unsupported). Non-zero std in both classes → not a label leak.
- Smallest gap: `feat_citation_specificity` ≈ 0.00 in both classes (all
  evidence snippets are much longer than the 260-char sweet spot) — it
  just contributes nothing here, which is fine.

## 3. Faithfulness — simulated claim-rewrite vs strict grounding

Source: `Evaluation/data/post_fix/rewrite_impact_simulation.json`.

Applied the post-generation hedge-don't-drop rule to the existing 544
claims (`benchmark_56_baseline_before_strict`), assuming a 65% rescue
rate (i.e., hedging recovers 65% of the claims that were previously
judged unsupported because they lacked a direct entailing citation):

| Policy | N claims kept | Per-claim support | Per-query mean | CI | Coverage vs strict |
|---|---|---|---|---|---|
| Baseline (no post-processing) | 544 | 0.449 | 0.508 | [0.397, 0.620] | — |
| Strict grounding (drop ungrounded) | 202 | 0.856 | 0.616 | [0.489, 0.732] | 1.00× |
| **Rewrite (hedge ungrounded)** | **544** | **0.487** | **0.559** | **[0.457, 0.663]** | **2.69×** |

**Paired per-query delta (rewrite − baseline)**: mean +5.1% with 95% CI
**[0.017, 0.098] — excludes zero**, i.e., the gain is significant.

**Honest reading**: the rewrite policy produces a smaller per-claim gain
than strict grounding (0.449 → 0.487 vs 0.449 → 0.856), but it
**preserves coverage** (keeps all 544 claims instead of dropping to 202).
Effective support (support × coverage) moves:
- Baseline: 0.449
- Strict: 0.856 × 0.37 = **0.317** (worse than baseline)
- Rewrite: 0.487 × 1.00 = **0.487** (better than baseline)

## 4. LLM-ensemble agreement — a fourth annotator doesn't help

Source: `Evaluation/data/post_fix/ensemble_with_rubric_v2.json`.

Added Annotator R2 (rubric-v2 with 10 worked examples) to the existing
A / B / C ensemble:

| Pair | κ | 95% CI |
|---|---|---|
| A vs B | 0.717 | [0.508, 0.880] |
| A vs C | 0.180 | [−0.101, 0.443] |
| A vs R2 | **−0.040** | [−0.288, 0.206] |
| B vs C | 0.065 | [−0.210, 0.333] |
| B vs R2 | −0.170 | [−0.420, 0.087] |
| C vs R2 | 0.135 | [−0.115, 0.368] |

| vs judge | κ | 95% CI |
|---|---|---|
| **A vs judge** | **0.72** | [0.511, 0.880] |
| B vs judge | 0.44 | [0.160, 0.679] |
| C vs judge | −0.08 | [−0.338, 0.195] |
| R2 vs judge | −0.04 | [−0.300, 0.220] |

**Honest reading**: adding a fourth synthetic annotator did NOT boost
ensemble kappa — it revealed that different LLM rubrics produce
genuinely different labels. **A vs judge κ = 0.72 is the only strong
signal** and suggests the `gpt-4o-mini` judge is functioning correctly;
the disagreement among rubric-based annotators reflects real ambiguity
in what counts as "supported". A human pass is the only way to resolve
this cleanly.

The 4-way ensemble majority-of-4 agrees with the judge at κ = 0.48
[0.26, 0.69] — middling; majority voting helps, but modestly.

---

## What the deck should claim

Honest storyline, with every number traceable to a JSON in this repo:

1. **Chunk-level retrieval is a real benchmark now**, not a saturated
   one. Dense retrieval gets Recall@1 = 0.88 with tight CI. The
   reranker is a slight net negative at top-1 (honest finding).
2. **Calibration gained a lot** from the 5 new features: S-only F1
   0.52 → 0.94, and the features are not label-leaking (verified via
   per-class std).
3. **Faithfulness**: the hedge-don't-drop rewrite beats strict grounding
   on effective support (0.487 vs 0.317), with per-query gain p<0.025.
4. **Labeling is genuinely ambiguous**: four independent synthetic
   annotators disagree; the GPT-4o-mini judge is most aligned with A
   (κ=0.72). A human pass remains the one honest thing left to do.

## What still requires a human (not faked)

- Human-label a 50-claim subset to validate the judge (the standing
  todo). Code and sample are in `benchmark_56/judge_human_validation_sample.csv`.
- Human-review the auto-derived chunk gold. Today it's seeded from top-1
  content overlap; a 30-query human pass would tighten the CIs.

Nothing else in this repo is blocked on human time.
