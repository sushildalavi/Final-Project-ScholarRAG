# Public-Mode Calibration — Handoff (completed)

This is how the public-mode calibration pipeline was run end-to-end. All steps
have been executed; artifacts are shipped in this directory.

---

## Step 1 — Extract public-mode claims (needs backend running)

```bash
cd /Users/sushildalavi/Desktop/USC/Final-Project-ScholarRAG
source .venv/bin/activate

# Start backend in another terminal if not already running:
#   uvicorn backend.app:app --reload --port 8000

python backend/scripts/extract_public_mode_eval.py \
  --queries Evaluation/queries/queries_public_mode.json \
  --out Evaluation/data/calibration/public_mode/claim_annotations.csv
```

**What this does:**
- Runs all 60 queries in `queries_public_mode.json` through `assistant_answer(scope="public")`.
- Hits Semantic Scholar, arXiv, CrossRef, OpenAlex, Springer, Elsevier, IEEE in parallel.
- Extracts every generated answer sentence as a claim, paired with its cited evidence snippet.
- Output: 327 claim rows (319 usable after label filtering).

**Runtime:** ~15 minutes (public APIs are rate-limited; LLM judge runs per-claim).

**Tuning applied to public retrieval:**
- Per-provider fetch limits raised 2x (15→30 for arXiv/OpenAlex; 10→20–25 for others).
- Sparse-vs-dense reranking weight raised from 0.25 to 0.35.
- Corroboration bonus raised from max +0.03 to max +0.10 (papers appearing in multiple providers rank higher).
- Query variants expanded from 2 to 3 (keyword-only variant added for BM25 coverage).
- Pool cap for reranking raised from `k*2` to `k*3` in public scope.
- Provider errors logged (previously swallowed silently).

---

## Step 2 — Generate the three labeling workbooks

```bash
python backend/scripts/generate_labeling_workbooks.py \
  --claims Evaluation/data/calibration/public_mode/claim_annotations.csv \
  --out-dir Evaluation/data/calibration/public_mode
```

Blank templates were handed to three coders. Filled versions are now archived under
`human_labels_raw/coder_{A,B,C}_labeled.xlsx`.

---

## Step 3 — Compute IAA and merge labels

After the three filled workbooks came back:

1. Pairwise Cohen's κ and Fleiss' κ computed → `human_labels_raw/iaa_summary.json`
2. Majority-vote consensus merged → `human_labels_raw/merged_labels.csv`
3. Consensus joined onto extraction output → `claim_annotations_with_consensus.csv`

**IAA result:** Fleiss' κ = 0.437 ("moderate"). Pairwise A↔B 0.533, A↔C 0.340, B↔C 0.463. Lower than uploaded mode's 0.655 — genuinely harder to judge whether an abstract "supports" a specific sub-claim.

---

## Step 4 — Recompute MSA features post-hoc

Because extraction didn't populate per-claim M/S/A, they were recomputed:

```bash
python backend/scripts/recompute_public_msa.py \
  --claims Evaluation/data/calibration/public_mode/claim_annotations_with_consensus.csv \
  --out    Evaluation/data/calibration/public_mode/claim_annotations_with_msa.csv
```

- M = `entailment_prob(claim, evidence)` via `backend.services.nli`
- A = `_compute_agreement_score(...)` — lexical cap-4 across distinct evidence snippets per query
- S = 0.0 (perturbation context not available post-hoc; weight is zeroed anyway)

---

## Step 5 — Fit the public-mode logistic

```bash
python backend/scripts/fit_calibration.py \
  --claims Evaluation/data/calibration/public_mode/claim_annotations_with_msa.csv \
  --out-dir Evaluation/data/calibration/public_mode \
  --model-name msa_logistic_public \
  --label public_mode
```

**Shipped weights:** w1=+2.17, w2=0, w3=+0.93, b=−2.30.
**Holdout (n=64):** AUC 0.90, PR-AUC 0.85, Brier 0.14, ECE 0.13, Accuracy 0.86.

---

## Step 6 — Apply

```bash
# Via backend:
curl -X POST http://localhost:8000/confidence/calibrate \
  -H 'Content-Type: application/json' \
  -d @Evaluation/data/calibration/public_mode/weights.json

# Or direct DB:
psql $DATABASE_URL -f Evaluation/data/calibration/public_mode/insert.sql
```

`_load_latest_calibration_weights("public")` picks it up on the next request. Uploaded-mode queries are unaffected — the loader is scope-aware.

---

## Files created in this round

**Code:**
- `backend/scripts/extract_public_mode_eval.py`
- `backend/scripts/generate_labeling_workbooks.py`
- `backend/scripts/recompute_public_msa.py`
- `backend/scripts/fit_calibration.py`
- `backend/public_search.py` — retrieval tuning
- `backend/services/assistant_utils.py` — `_load_latest_calibration_weights(scope)`
- `backend/app.py` — callers pass `scope`; pool cap raised to `k*3`
- `backend/confidence.py` — MSA-only formula (no retrieval blend)

**Data:**
- `Evaluation/queries/queries_public_mode.json` — 60 curated NLP/ML queries
- `Evaluation/data/calibration/uploaded_mode/` — canonical uploaded-mode artifacts
- `Evaluation/data/calibration/public_mode/` — canonical public-mode artifacts
- `Evaluation/data/calibration/FINAL_REPORT.md` — consolidated summary

**Regression:** 168/168 backend tests pass after all changes.
