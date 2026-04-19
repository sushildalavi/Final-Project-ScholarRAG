# MSA Calibration — Public Mode

**Source:** `Evaluation/data/calibration/public_mode/claim_annotations_with_msa.csv`
**Dataset:** 319 labeled claims (train=255, holdout=64)
**Model:** `msa_logistic_public` / label `public_mode`
**Chosen:** `M+A (S=0)`
**Fitter:** sklearn `LogisticRegression(penalty='l2', C=1.0)` with stratified 80/20 split.

## Feature ablation on holdout

| Model | AUC | Brier | ECE | Accuracy |
|-------|-----|-------|-----|----------|
| M+S+A (full) | 0.9018 | 0.1354 | 0.1317 | 0.8594 |
| M+A (S=0) | 0.9018 | 0.1354 | 0.1317 | 0.8594 |
| Defaults | 0.9018 | 0.2613 | 0.3743 | 0.2969 |

> **Key finding:** S (retrieval-stability) is not usable in public mode. The post-hoc S column is all-zero because recomputing S requires re-running retrieval with query perturbations per claim, which we did not do. The M+A fit matches M+S+A exactly because the S column carries no information.

## Chosen weights (M/S/A logistic)

| Weight | Fitted | Default | Bootstrap 95% CI |
|--------|--------|---------|------------------|
| w1 (M) | +2.1684 | +0.58 | [+1.393, +2.886] |
| w2 (S) | +0.0000 | +0.22 | — |
| w3 (A) | +0.9289 | +0.20 | [+0.221, +1.650] |
| b      | -2.3002 | +0.00 | [-3.098, -1.612] |

## Held-out metrics (20% stratified)

| Metric   | Fitted | Defaults | Δ |
|----------|--------|----------|---|
| AUC      | 0.9018 | 0.9018 | +0.0000 |
| PR-AUC   | 0.8532 | 0.8532 | +0.0000 |
| Brier    | 0.1354 | 0.2613 | +0.1259 (lower = better) |
| ECE      | 0.1317 | 0.3743 | +0.2426 (lower = better) |
| Accuracy | 0.8594 | 0.2969 | +0.5625 |

AUC is identical because it's rank-order only and both weight sets preserve the same ordering on M+A. Brier, ECE, and accuracy are what actually change — the fitted weights produce probabilities that are calibrated rather than arbitrary.

## Ship decision

**SHIP.** Fitted weights beat defaults on Brier, ECE, and accuracy; AUC is tied.

## How to apply

Option A — via HTTP endpoint (backend must be running):

```bash
curl -X POST http://localhost:8000/confidence/calibrate \
  -H 'Content-Type: application/json' \
  -d @Evaluation/data/calibration/public_mode/weights.json
```

Option B — direct DB insert:

```bash
psql $DATABASE_URL -f Evaluation/data/calibration/public_mode/insert.sql
```

`_load_latest_calibration_weights("public")` picks up the new row on next request. No backend redeploy needed.
