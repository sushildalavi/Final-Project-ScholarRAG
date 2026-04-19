# MSA Calibration — Uploaded Mode

**Source:** `Evaluation/data/calibration/uploaded_mode/human_labels_raw/claim_annotations_shipped.csv`
**Dataset:** 849 labeled claims (train=679, holdout=170)
**Model:** `msa_logistic_uploaded` / label `uploaded_mode`  
**Chosen:** `M+A (S=0)`  
**Fitter:** sklearn `LogisticRegression(penalty='l2', C=1.0)` with stratified 80/20 split.

## Feature ablation on holdout

| Model | AUC | Brier | ECE | Accuracy |
|-------|-----|-------|-----|----------|
| M+S+A (full) | 0.6975 | 0.2003 | 0.0988 | 0.7000 |
| M+A (S=0) | 0.7025 | 0.2008 | 0.0718 | 0.7000 |
| Defaults | 0.6680 | 0.2701 | 0.2549 | 0.3000 |

> **Key finding:** In the post-fix regime, S (retrieval-stability) is counter-predictive on our data — S-only AUC is below random (flipped sign). Dropping S and zeroing w2 yields the strongest calibration.

## Chosen weights (M/S/A logistic)

| Weight | Fitted | Default | Bootstrap 95% CI |
|--------|--------|---------|------------------|
| w1 (M) | +1.2903 | +0.58 | [+0.287, +2.086] |
| w2 (S) | +0.0000 | +0.22 | — |
| w3 (A) | +0.5719 | +0.20 | [+0.129, +1.056] |
| b      | -1.3716 | +0.00 | [-1.812, -0.981] |

## Held-out metrics (20% stratified)

| Metric   | Fitted | Defaults | Δ |
|----------|--------|----------|---|
| AUC      | 0.7025 | 0.6680 | +0.0345 |
| PR-AUC   | 0.4622 | 0.4218 | +0.0404 |
| Brier    | 0.2008 | 0.2701 | +0.0693 (lower = better) |
| ECE      | 0.0718 | 0.2549 | +0.1831 (lower = better) |
| Accuracy | 0.7000 | 0.3000 | +0.4000 |

## Ship decision

**Recommendation:** SHIP

> M+A (S=0) beats defaults on both AUC and Brier; recommend shipping.

## How to apply (if shipping)

Option A — via HTTP endpoint (backend must be running):

```bash
curl -X POST http://localhost:8000/confidence/calibrate \
  -H 'Content-Type: application/json' \
  -d @Evaluation/data/calibration/uploaded_mode/weights.json
```

Option B — direct DB insert:

```bash
psql $DATABASE_URL -f Evaluation/data/calibration/uploaded_mode/insert.sql
```

`_load_latest_calibration_weights()` picks up the new row on next request. No backend redeploy needed.
