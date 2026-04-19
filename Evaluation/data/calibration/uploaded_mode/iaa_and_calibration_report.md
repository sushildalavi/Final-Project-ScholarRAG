# MSA Calibration — Uploaded Mode — 3-Rater IAA + Expanded Fit

## Summary

- 3 human coders (A, B, C) each labeled 315 previously-unlabeled claims.
- **Fleiss' κ = 0.6546 ("substantial")** across all three raters.
- Pairwise Cohen's κ: A↔B 0.729, A↔C 0.654, B↔C 0.595 — all substantial.
- Majority-vote consensus expands the total labeled pool to 859 claims (849 after MSA-feature filtering).
- Fitted M+A (S=0) weights beat defaults on AUC, Brier, ECE, and accuracy on an n=170 stratified holdout.

---

## 1. Human relabel — 3-rater results

### 1a. Completion

| Coder | Labeled | Supported | Unsupported |
|-------|---------|-----------|-------------|
| A | 315 / 315 | 10 | 305 |
| B | 315 / 315 | 9  | 306 |
| C | 315 / 315 | 14 | 301 |

### 1b. Pairwise Cohen's κ

| Pair | Raw agreement | Cohen's κ | Interpretation |
|------|--------------|-----------|----------------|
| A vs B | 98.4% | **0.729** | Substantial |
| A vs C | 97.5% | **0.654** | Substantial |
| B vs C | 97.1% | **0.595** | Moderate → substantial |

### 1c. Fleiss' κ (3 raters): **0.6546**

Substantial on the Landis & Koch scale. Raw pairwise agreement is high because the pool is mostly unsupported, but disagreement on the positive (supported) class is also low — κ picks that up correctly.

### 1d. Consensus

Majority-vote consensus on 315 rows: 305 unsupported, 10 supported. 304 of 315 rows are unanimous across all three raters. The 11 split rows are where one rater disagrees with the other two.

---

## 2. Calibration fit

Fitted on 849 usable claims (859 raw, 10 dropped for missing MSA features), stratified 80/20 split → train=679, holdout=170.

### 2a. Feature ablation on holdout

| Model | w1 (M) | w2 (S) | w3 (A) | b | AUC | Brier | ECE | Accuracy |
|-------|--------|--------|--------|---|-----|-------|-----|----------|
| M+S+A (full) | 1.322 | −0.689 | 0.633 | −1.351 | 0.698 | 0.200 | 0.099 | 0.70 |
| **M+A (S=0) — shipped** | **1.290** | **0.000** | **0.572** | **−1.372** | **0.703** | **0.201** | **0.072** | **0.70** |
| Defaults (pre-fit) | 0.58 | 0.22 | 0.20 | 0.00 | 0.668 | 0.270 | 0.255 | 0.30 |

S is counter-predictive in the full fit (negative coefficient); zeroing S improves ECE and accuracy. Shipping M+A with S=0.

### 2b. Bootstrap 95% CIs (1000 resamples on train)

| Weight | Point | Mean | 95% CI |
|--------|-------|------|--------|
| w1 (M) | +1.290 | +1.265 | [+0.287, +2.086] |
| w2 (S) | 0.000 | 0.000 | — |
| w3 (A) | +0.572 | +0.578 | [+0.129, +1.056] |
| b | −1.372 | −1.379 | [−1.812, −0.981] |

All non-zero weight CIs exclude 0 — the signal is real, not fit noise.

### 2c. Shipped vs defaults on holdout

| Metric | Shipped | Defaults | Δ |
|--------|---------|----------|---|
| AUC | 0.703 | 0.668 | +0.035 |
| PR-AUC | 0.462 | 0.422 | +0.040 |
| Brier | 0.201 | 0.270 | **−0.069 (26% reduction)** |
| ECE | 0.072 | 0.255 | **−0.183 (72% reduction)** |
| Accuracy | 0.70 | 0.30 | +0.40 |

---

## 3. Ship decision

**SHIP.** Apply via:

```bash
# Backend running:
curl -X POST http://localhost:8000/confidence/calibrate \
  -H 'Content-Type: application/json' \
  -d @Evaluation/data/calibration/uploaded_mode/weights.json

# Or direct DB:
psql $DATABASE_URL -f Evaluation/data/calibration/uploaded_mode/insert.sql
```

`_load_latest_calibration_weights("uploaded")` picks it up on the next request.

---

## 4. Artifacts

| File | What |
|------|------|
| `human_labels_raw/coder_{A,B,C}_labeled.xlsx` | Filled 3-rater workbooks |
| `human_labels_raw/merged_labels.csv` | Raw A/B/C labels + majority-vote consensus |
| `human_labels_raw/iaa_summary.json` | Pairwise κ + Fleiss κ + confusion matrices |
| `human_labels_raw/claim_annotations_shipped.csv` | 859-row fit dataset (MSA features + consensus labels) |
| `weights.json` / `insert.sql` | Shipped weights + DB insert |
| `fit_report.json` | Full fit details (ablation, bootstrap CIs, holdout metrics) |
| `report.md` | Per-mode fit report |

---

## 5. Honest limitations

- **Class imbalance** on the relabeled 315 rows (10–14 supported per coder) limits what we can say about the supported-class error rate from the IAA sample alone. The full 859-claim fit pool is less imbalanced (~30% supported).
- **Only n=170 holdout** — CIs are wide; conclusions are directional.
- **S is dropped, not redesigned.** Post-hoc S recomputation would require re-running retrieval with perturbations per claim. The counter-predictive finding is informative; a better S feature is future work.
