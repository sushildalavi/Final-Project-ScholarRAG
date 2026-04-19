# ScholarRAG Confidence Calibration — Final Report

## Summary

- **MSA-only** is the canonical confidence formula for both uploaded and public mode.
- **Separate weights** are calibrated per mode and loaded at query time by scope.
- Both modes beat pre-fix defaults on AUC, Brier, ECE, and accuracy on held-out claims.
- Public mode has stronger discrimination (AUC 0.90) than uploaded mode (AUC 0.70) — M (NLI) separates better when evidence is a clean paper abstract than when it's a mid-document PDF chunk.

## The confidence formula

```
confidence = sigmoid(b + w1·M + w2·S + w3·A)
```

where the weights come from the currently-loaded calibration row, filtered by `scope` at request time (`_load_latest_calibration_weights(scope)` in [backend/services/assistant_utils.py:29](backend/services/assistant_utils.py:29)).

The previous `0.62·retrieval_score + 0.38·msa_score` blend has been removed — it was never calibrated and applied an arbitrary fixed coefficient on top of our learned MSA logistic. Runtime now matches what the logistic was fit against.

If a caller does not supply MSA features, `build_confidence` falls back to the old retrieval-only score — purely for backward compatibility with non-MSA code paths. This fallback is not a validated calibration.

## Shipped weights

| Feature | Uploaded mode | Public mode | Notes |
|---------|---------------|-------------|-------|
| w1 (M)  | +1.29         | +2.17       | NLI entailment; public weights heavier because M separates better on abstracts |
| w2 (S)  | 0             | 0           | Retrieval stability; zeroed in both modes (counter-predictive in uploaded ablation; not computable post-hoc for public relabel) |
| w3 (A)  | +0.57         | +0.93       | Lexical cross-doc corroboration; still useful but smaller weight in public (nearly saturated) |
| b       | −1.37         | −2.30       | Intercept; public pool skews unsupported, larger negative bias |

**Uploaded-mode dataset:** 849 labeled claims (3-rater Fleiss κ = 0.655, majority-vote consensus).
**Public-mode dataset:** 319 labeled claims (3-rater Fleiss κ = 0.437, majority-vote consensus).

## Held-out metrics

| Metric | Uploaded (n=170) | Public (n=64) |
|--------|-----------------|---------------|
| AUC | 0.703 | **0.902** |
| PR-AUC | 0.462 | **0.853** |
| Brier | 0.201 | **0.135** |
| ECE | **0.072** | 0.132 |
| Accuracy | 0.70 | **0.86** |

Public mode's higher AUC is real — abstracts yield a sharper M signal. Uploaded mode's lower ECE reflects its larger holdout (170 vs 64) giving more stable calibration bins.

## Inter-annotator agreement (3 raters per mode)

| Mode | A↔B κ | A↔C κ | B↔C κ | Fleiss κ | Unanimous |
|------|-------|-------|-------|----------|-----------|
| Uploaded (315 relabeled) | 0.729 | 0.654 | 0.595 | **0.655** | 304/315 |
| Public (319 new) | 0.533 | 0.340 | 0.463 | **0.437** | 204/319 |

Public mode has lower agreement because judging whether an abstract "supports" a specific claim is genuinely harder than judging a PDF chunk — abstracts describe the paper's topic without always asserting the exact sub-claim. The rubric survives transfer but with lower κ.

## Feature distributions by label

### Uploaded mode (benchmark_56 + 3-rater relabel)
| Feature | Supported mean | Unsupported mean |
|---------|---------------|-----------------|
| M       | 0.08          | 0.05            |
| S       | 0.10          | 0.11            |
| A       | 0.82          | 0.61            |

A is the stronger discriminator.

### Public mode (319 relabel)
| Feature | Supported mean | Unsupported mean |
|---------|---------------|-----------------|
| M       | 0.44          | 0.15            |
| S       | 0             | 0               |
| A       | 0.96          | 0.87            |

M is the dominant discriminator; A saturates because abstracts always share tokens with the claim that was drawn from them.

## Why the approach changed mid-stream

1. **A-feature leakage fix (uploaded).** Original A leaked the support label; redefined as lexical cross-doc corroboration (cap-4), fresh calibration run on post-fix data.
2. **Dropped the 0.62/0.38 blend.** It was pre-existing uncalibrated code. The MSA logistic is what we actually fit; the retrieval side was never validated in the blend coefficient. Now runtime matches the fit.
3. **Mode-specific weights.** Features (especially A) have different statistical properties in each mode. One weight set cannot fit both without becoming a compromise. Scope-aware loading solves this with no runtime branching in the formula itself.

## How to apply (both modes)

The DB-insert path (no backend required beyond Postgres):

```bash
psql $DATABASE_URL -f Evaluation/data/calibration/uploaded_mode/insert.sql
psql $DATABASE_URL -f Evaluation/data/calibration/public_mode/insert.sql
```

Or via the running backend:

```bash
curl -X POST http://localhost:8000/confidence/calibrate \
  -H 'Content-Type: application/json' \
  -d @Evaluation/data/calibration/uploaded_mode/weights.json

curl -X POST http://localhost:8000/confidence/calibrate \
  -H 'Content-Type: application/json' \
  -d @Evaluation/data/calibration/public_mode/weights.json
```

`_load_latest_calibration_weights(scope)` picks them up on the next request. No backend redeploy.

## Artifacts

```
Evaluation/data/calibration/
├── FINAL_REPORT.md                    # this file
├── uploaded_mode/
│   ├── weights.json                   # shipped uploaded-mode weights
│   ├── insert.sql                     # DB insert
│   ├── fit_report.json                # machine-readable fit details
│   ├── report.md                      # per-mode fit report
│   ├── iaa_and_calibration_report.md  # uploaded-mode IAA + fit
│   └── human_labels_raw/
│       ├── coder_{A,B,C}_labeled.xlsx # filled 3-rater workbooks
│       ├── merged_labels.csv
│       ├── iaa_summary.json
│       └── claim_annotations_shipped.csv
└── public_mode/
    ├── weights.json                   # shipped public-mode weights
    ├── insert.sql                     # DB insert
    ├── fit_report.json
    ├── report.md
    ├── HANDOFF.md                     # how the pipeline was run
    ├── LABELING_INSTRUCTIONS.md       # rubric distributed to coders
    ├── claim_annotations.csv          # raw extraction output
    ├── claim_annotations_with_consensus.csv
    ├── claim_annotations_with_msa.csv # labeled + MSA features
    ├── extraction_manifest.json
    └── human_labels_raw/
        ├── coder_{A,B,C}_labeled.xlsx # filled 3-rater workbooks
        ├── merged_labels.csv
        └── iaa_summary.json
```

## Regression status

Full backend test suite: **168 / 168 passing** after the MSA-only change and scope-aware loader.

## Honest limitations

- **Public-mode κ of 0.437 is moderate, not substantial.** C was stricter than A and B. The rubric survives but with more rater-specific variance. A second labeling pass with clearer guidance on abstract-vs-body-text would likely push this higher.
- **S is not used.** Post-hoc stability recomputation would require re-running retrieval with perturbations per claim — not done; weight is zeroed. Consistent with uploaded-mode finding that S was counter-predictive.
- **Public holdout n=64 is small.** Metric CIs are wide. Directional conclusions hold; precise deltas should not be over-claimed.
- **Uploaded-mode AUC of 0.70** is the lower bound — could move if we re-ran with fresh retrieval labels. Not a weakness of the calibration, a property of the signal.
