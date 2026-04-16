# ScholarRAG Open-Corpus Evaluation Runbook

This workflow adds a manual, annotation-first evaluation mode for arbitrary uploaded papers without changing the existing benchmark pipeline in:

- `POST /eval/run`
- `POST /eval/judge`
- `POST /confidence/calibrate`
- `scripts/eval_retrieval.py`

The open-corpus workflow is file-based and is intended for a single researcher or small team running evaluations manually.

## Scope

Current assumptions based on the repository:

- Open-corpus evaluation targets **uploaded documents** only.
- Query files use `doc_scope: "uploaded"`.
- Retrieval export reuses `backend.pdf_ingest.search_chunks()`.
- Answer export reuses `backend.app.assistant_answer()`.
- Claim splitting reuses the existing sentence/citation parsing logic.
- Calibration reuses the existing `POST /confidence/calibrate` flow.

Example post-annotation files are included here:

- `Evaluation/open_corpus/query_summary_example.csv`
- `Evaluation/open_corpus/retrieval_annotations_example.csv`
- `Evaluation/open_corpus/corpus_doc_relevance_example.csv`
- `Evaluation/open_corpus/claim_annotations_example.csv`
- `Evaluation/open_corpus/retrieval_annotations_example.json`
- `Evaluation/open_corpus/retrieval_metrics_example.json`
- `Evaluation/open_corpus/claim_annotations_example.json`
- `Evaluation/open_corpus/calibration_records_example.json`
- `Evaluation/open_corpus/calibration_records_example.jsonl`

## 1. Upload Arbitrary Papers

Upload papers through the app until their status is `ready`.

You can confirm the backend is seeing ready docs from the UI or by calling:

```bash
curl -s http://127.0.0.1:8000/documents/latest | jq
```

## 2. Prepare Queries

You have two options.

### Option A: Manual query file

Start from:

- `Evaluation/open_corpus/queries_template.json`

Expected schema:

```json
{
  "queries": [
    {
      "query_id": "q1",
      "query": "What retrieval method does this paper propose?",
      "doc_scope": "uploaded",
      "doc_id": 12
    },
    {
      "query_id": "q2",
      "query": "Compare the methods proposed in these selected papers.",
      "doc_scope": "uploaded",
      "doc_ids": [12, 14]
    }
  ]
}
```

Supported fields:

- `query_id`: required stable identifier
- `query`: required user-style query text
- `doc_scope`: must be `uploaded`
- `doc_id`: optional single selected doc
- `doc_ids`: optional multi-doc selected scope
- `allow_general_background`: optional, defaults to `false`

If both `doc_id` and `doc_ids` are omitted, the open-corpus export scripts treat the query as a whole-corpus uploaded-doc query and run it over all ready uploaded documents.

### Option B: Generate a starter query set from uploaded docs

```bash
python scripts/open_eval_generate_queries.py \
  --doc-ids 12 14 16 \
  --per-doc 4 \
  --cross-doc 6 \
  -o Evaluation/open_corpus/generated_queries.json
```

If `--doc-ids` is omitted, the script uses all ready uploaded docs.

The generator is template-based, not LLM-based. It produces realistic starter prompts and is intended for manual review before evaluation.

## 3. Export Spreadsheet-Friendly Annotation Files

Preferred workflow for human annotators:

```bash
python scripts/open_eval_export_csv.py \
  --queries Evaluation/open_corpus/generated_queries.json \
  --out-dir Evaluation/open_corpus/csv_run \
  --retrieval-k 10 \
  --answer-k 8 \
  --compute-msa
```

This writes:

- `query_summary.csv`
- `retrieval_annotations.csv`
- `claim_annotations.csv`
- `corpus_doc_relevance.csv`
- `export_manifest.json`

These CSVs open directly in Excel.

### Blind claims before annotation (recommended)

To prevent label leakage during claim support annotation, create a blinded claim sheet that hides `msa_M`, `msa_S`, and `msa_A`:

```bash
python scripts/open_eval_prepare_blind_claims.py \
  --claims Evaluation/open_corpus/csv_run/claim_annotations.csv \
  --out-dir Evaluation/open_corpus/csv_run \
  --shuffle \
  --seed 42
```

This writes:

- `claim_annotations_blind.csv` (share with annotators)
- `claim_annotations_blind_key.csv` (keep private)
- `claim_annotations_blind_manifest.json`

### CSV file shapes

`query_summary.csv`

- `query_id`
- `query`
- `generated_answer`
- `answer_level_notes`

`retrieval_annotations.csv`

- `query_id`
- `query`
- `rank`
- `doc_id`
- `document_title`
- `chunk_id`
- `page`
- `retrieval_score`
- `chunk_text`
- `relevance_label`

`corpus_doc_relevance.csv`

- `query_id`
- `query`
- `doc_id`
- `document_title`
- `relevance_label`

`claim_annotations.csv`

- `query_id`
- `query`
- `claim_id`
- `claim_text`
- `evidence_ids`
- `evidence_text`
- `msa_M`
- `msa_S`
- `msa_A`
- `support_label`
- `citation_correct`
- `annotator_notes`

`corpus_doc_relevance.csv` is optional for annotation, but recommended if you want stricter document-level Recall/MRR denominators. If you only annotate `retrieval_annotations.csv`, the scorer still works and will derive doc relevance from the ranked rows.

## 4. Export Raw JSON Results

If you want raw JSON artifacts in addition to the CSVs, use the existing export scripts below.

Run retrieval over the query set and produce an annotation-ready file:

```bash
python scripts/open_eval_export_retrieval.py \
  --queries Evaluation/open_corpus/generated_queries.json \
  --k 10 \
  --out Evaluation/open_corpus/retrieval_run.json \
  --annotation-out Evaluation/open_corpus/retrieval_annotations.json
```

Each query export includes:

- `retrieved`: top-k chunk-level hits
- `retrieved_docs`: unique docs in retrieval order
- `corpus_docs`: docs within the query scope that annotators should judge for relevance

Relevant fields in `retrieved`:

- `rank`
- `doc_id`
- `title`
- `chunk_id`
- `score`
- `distance`
- `page`
- `chunk_text`
- `relevance_label`

## 5. Annotate Retrieval Relevance

If you are using the spreadsheet workflow, annotate:

- `Evaluation/open_corpus/csv_run/retrieval_annotations.csv`
- optionally `Evaluation/open_corpus/csv_run/corpus_doc_relevance.csv`

If you are using the JSON workflow, annotate:

- `Evaluation/open_corpus/retrieval_annotations.json`

Recommended label scheme:

- `relevant`
- `partially_relevant`
- `not_relevant`

Where to annotate:

- CSV:
  - `corpus_doc_relevance.csv -> relevance_label`
  - `retrieval_annotations.csv -> relevance_label`
- JSON:
  - `corpus_docs[*].relevance_label`
  - `retrieved_docs[*].relevance_label`
  - `retrieved[*].relevance_label`

Preferred source of truth:

- `corpus_doc_relevance.csv` or `corpus_docs[*]` for strict document-level Recall/MRR
- `retrieval_annotations.csv` or `retrieved[*]` for chunk-level annotation and error analysis

Metric conversion used by the scoring script:

- Binary relevance for Recall/MRR:
  - `relevant` -> 1
  - `partially_relevant` -> 1
  - `not_relevant` -> 0
- Graded relevance for nDCG:
  - `relevant` -> 2
  - `partially_relevant` -> 1
  - `not_relevant` -> 0

## 6. Export Answers and Claim Annotations

Generate ScholarRAG answers and a claim annotation template:

```bash
python scripts/open_eval_export_answers.py \
  --queries Evaluation/open_corpus/generated_queries.json \
  --k 8 \
  --out Evaluation/open_corpus/answers_run.json \
  --claim-annotation-out Evaluation/open_corpus/claim_annotations.json
```

If you want per-claim M/S/A features exported for later calibration, enable:

```bash
python scripts/open_eval_export_answers.py \
  --queries Evaluation/open_corpus/generated_queries.json \
  --k 8 \
  --compute-msa \
  --out Evaluation/open_corpus/answers_run.json \
  --claim-annotation-out Evaluation/open_corpus/claim_annotations.json
```

Notes:

- `--compute-msa` reuses the current evidence scoring path inside `assistant_answer()`.
- This is more expensive because it computes entailment-based support features.
- `--run-judge-llm` is optional and only needed if you explicitly want LLM judge output during export.

Each answer export contains:

- `answer`
- `citations`
- `claims`

Each claim contains:

- `claim_id`
- `text`
- `citation_ids`
- `evidence_ids`
- `evidence_text`
- `label`
- `citation_correct`
- `msa` when `--compute-msa` is enabled

## 7. Annotate Claim Support

If you are using the spreadsheet workflow, annotate one of:

- `Evaluation/open_corpus/csv_run/claim_annotations.csv`
- recommended: `Evaluation/open_corpus/csv_run/claim_annotations_blind.csv`

If you are using the JSON workflow, annotate:

- `Evaluation/open_corpus/claim_annotations.json`

Required claim labels:

- `supported`
- `unsupported`

Optional:

- `citation_correct`: `true`, `false`, or `null`

The current calibration path is binary, so keep labels binary even if you personally note edge cases.

If you have two independent annotators on an overlap subset, compute agreement:

```bash
python scripts/open_eval_annotation_agreement.py \
  --annotator-a Evaluation/open_corpus/csv_run/claim_annotations_blind_annotator_a.csv \
  --annotator-b Evaluation/open_corpus/csv_run/claim_annotations_blind_annotator_b.csv \
  -o Evaluation/open_corpus/annotation_agreement.json
```

Use this report for Cohen's kappa before adjudication.

## 8. Merge Blinded Claims Back With M/S/A

After adjudication, merge the final blinded annotations with the hidden M/S/A key:

```bash
python scripts/open_eval_merge_blind_claims.py \
  --blind-annotations Evaluation/open_corpus/csv_run/claim_annotations_blind_final.csv \
  --blind-key Evaluation/open_corpus/csv_run/claim_annotations_blind_key.csv \
  -o Evaluation/open_corpus/csv_run/claim_annotations_merged.csv
```

Use `claim_annotations_merged.csv` for calibration record generation.

## 9. Compute Retrieval Metrics

Once retrieval annotations are complete:

```bash
python scripts/open_eval_score_retrieval.py \
  --annotations Evaluation/open_corpus/csv_run/retrieval_annotations.csv \
  --corpus-docs Evaluation/open_corpus/csv_run/corpus_doc_relevance.csv \
  -o Evaluation/open_corpus/retrieval_metrics.json
```

The same script also accepts the earlier JSON format:

```bash
python scripts/open_eval_score_retrieval.py \
  --annotations Evaluation/open_corpus/retrieval_annotations.json \
  -o Evaluation/open_corpus/retrieval_metrics.json
```

The script outputs:

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `Recall@10`
- `MRR`
- `nDCG@10`
- per-query breakdowns

The metric implementation is in:

- `backend/open_eval_metrics.py`

The scoring script reads the human-provided relevance labels from the annotated JSON file and computes:

- `Recall@1`
- `Recall@3`
- `Recall@5`
- `Recall@10`
- `MRR`
- `nDCG@10`

## 10. Build Calibration Records

Convert annotated claims into `/confidence/calibrate` input:

```bash
python scripts/open_eval_build_calibration.py \
  --claims Evaluation/open_corpus/csv_run/claim_annotations_merged.csv \
  --model-name msa_open_corpus_manual \
  --label open_corpus_manual \
  -o Evaluation/open_corpus/calibration_records.json
```

The same builder also accepts the earlier JSON claim annotation file:

```bash
python scripts/open_eval_build_calibration.py \
  --claims Evaluation/open_corpus/claim_annotations.json \
  --model-name msa_open_corpus_manual \
  --label open_corpus_manual \
  -o Evaluation/open_corpus/calibration_records.json
```

Output shape:

```json
{
  "model_name": "msa_open_corpus_manual",
  "label": "open_corpus_manual",
  "records": [
    {
      "query_id": "q1",
      "claim_id": "q1_c1",
      "sentence": "...",
      "evidence_text": "...",
      "label": "supported",
      "msa": {
        "M": 0.82,
        "S": 0.66,
        "A": 0.75
      }
    }
  ]
}
```

If `msa` is absent, the calibration endpoint can still backfill `M` from `sentence + evidence_text`, but `S` and `A` will not be as informative.

If you want line-oriented records instead:

```bash
python scripts/open_eval_build_calibration.py \
  --claims Evaluation/open_corpus/claim_annotations.json \
  --model-name msa_open_corpus_manual \
  --label open_corpus_manual \
  --format jsonl \
  -o Evaluation/open_corpus/calibration_records.jsonl
```

## 11. Evaluate Held-Out Calibration + Ablations

Before fitting final production weights, evaluate leakage-safe held-out performance:

```bash
python scripts/open_eval_eval_calibration.py \
  --claims Evaluation/open_corpus/csv_run/claim_annotations_merged.csv \
  --group-by query_id \
  --train-ratio 0.8 \
  --seed 42 \
  -o Evaluation/open_corpus/calibration_eval_report.json
```

The report includes train/test metrics for:

- `M+S+A`
- `M-only`
- `S-only`
- `A-only`
- fixed heuristic baseline

Reported metrics include Brier, ECE, PR-AUC, F1, confusion counts, and calibration bins.

## 12. Fit Logistic Regression Weights

Post calibration records to the existing endpoint:

```bash
python scripts/open_eval_fit_calibration.py \
  --input Evaluation/open_corpus/calibration_records.json \
  --base-url http://127.0.0.1:8000 \
  --output Evaluation/open_corpus/calibration_fit_response.json
```

This reuses the existing binary logistic fit in:

- `backend/app.py`

The fit script accepts either:

- `.json` payloads with a top-level `records` array
- `.jsonl` files where each line is one calibration record

Equivalent `curl` command:

```bash
curl -s -X POST http://127.0.0.1:8000/confidence/calibrate \
  -H "Content-Type: application/json" \
  -d @Evaluation/open_corpus/calibration_records.json | jq
```

The endpoint stores learned weights in `confidence_calibration`.

## 13. Suggested End-to-End Flow

```bash
python scripts/open_eval_generate_queries.py \
  --doc-ids 12 14 16 \
  --per-doc 4 \
  --cross-doc 6 \
  -o Evaluation/open_corpus/queries.json

python scripts/open_eval_export_csv.py \
  --queries Evaluation/open_corpus/queries.json \
  --out-dir Evaluation/open_corpus/csv_run \
  --retrieval-k 10 \
  --answer-k 8 \
  --compute-msa \
  --run-judge-llm

python scripts/open_eval_prepare_blind_claims.py \
  --claims Evaluation/open_corpus/csv_run/claim_annotations.csv \
  --out-dir Evaluation/open_corpus/csv_run \
  --shuffle \
  --seed 42
```

Then manual work:

1. annotate `csv_run/retrieval_annotations.csv`
2. optionally annotate `csv_run/corpus_doc_relevance.csv`
3. annotate blinded claims (`csv_run/claim_annotations_blind*.csv`)
4. adjudicate overlap and produce `csv_run/claim_annotations_blind_final.csv`

Then finish with:

```bash
python scripts/open_eval_score_retrieval.py \
  --annotations Evaluation/open_corpus/csv_run/retrieval_annotations.csv \
  --corpus-docs Evaluation/open_corpus/csv_run/corpus_doc_relevance.csv \
  -o Evaluation/open_corpus/retrieval_metrics.json

python scripts/open_eval_merge_blind_claims.py \
  --blind-annotations Evaluation/open_corpus/csv_run/claim_annotations_blind_final.csv \
  --blind-key Evaluation/open_corpus/csv_run/claim_annotations_blind_key.csv \
  -o Evaluation/open_corpus/csv_run/claim_annotations_merged.csv

python scripts/open_eval_build_calibration.py \
  --claims Evaluation/open_corpus/csv_run/claim_annotations_merged.csv \
  --model-name msa_open_corpus_manual \
  --label open_corpus_manual \
  -o Evaluation/open_corpus/calibration_records.json

python scripts/open_eval_eval_calibration.py \
  --claims Evaluation/open_corpus/csv_run/claim_annotations_merged.csv \
  --group-by query_id \
  --train-ratio 0.8 \
  --seed 42 \
  -o Evaluation/open_corpus/calibration_eval_report.json

python scripts/open_eval_fit_calibration.py \
  --input Evaluation/open_corpus/calibration_records.json \
  --base-url http://127.0.0.1:8000 \
  --output Evaluation/open_corpus/calibration_fit_response.json
```

## 14. Manual vs Automated

Automated:

- query-set generation scaffolding
- retrieval export
- answer generation
- claim splitting
- claim blinding / blind-key generation
- blinded annotation merge
- inter-annotator agreement reporting
- retrieval metric computation
- calibration record building
- held-out calibration and ablation reporting
- logistic calibration fit

Manual:

- reviewing or writing realistic queries
- relevance labeling for `corpus_docs`
- optional chunk-level relevance labeling
- claim support labeling
- optional citation correctness labeling
