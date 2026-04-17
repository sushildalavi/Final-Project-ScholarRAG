.PHONY: test lint typecheck run run-frontend install install-dev clean help

# ── Environment ───────────────────────────────────────────────────────────────
PYTHON  := python3
VENV    := .venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
UVICORN := $(VENV)/bin/uvicorn
PYTHONPATH_ENV := PYTHONPATH=.

# ── Setup ─────────────────────────────────────────────────────────────────────

install: $(VENV)/bin/activate
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -r requirements-dev.txt

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

# ── Test ──────────────────────────────────────────────────────────────────────

test:
	$(PYTEST) backend/tests/ -v --tb=short

test-fast:
	$(PYTEST) backend/tests/ -x --tb=short -q

test-coverage:
	$(PYTEST) backend/tests/ --cov=backend --cov-report=term-missing --cov-report=html

# ── Lint ──────────────────────────────────────────────────────────────────────

lint:
	$(RUFF) check backend/ scripts/ --ignore E501,F401,E402

lint-fix:
	$(RUFF) check backend/ scripts/ --fix --ignore E501,F401,E402

typecheck:
	$(VENV)/bin/pyright backend/confidence.py backend/eval_metrics.py backend/services/nli.py

# ── Run ───────────────────────────────────────────────────────────────────────

run:
	$(UVICORN) backend.app:app --reload --host 127.0.0.1 --port 8000

run-prod:
	$(UVICORN) backend.app:app --host 0.0.0.0 --port 8000 --workers 4

run-frontend:
	cd frontend && npm run dev

# ── Database ──────────────────────────────────────────────────────────────────

db-up:
	docker compose up -d db adminer

db-down:
	docker compose down

# ── Reindex ───────────────────────────────────────────────────────────────────

reindex:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/reindex_embeddings.py --purge-all

# ── Eval ──────────────────────────────────────────────────────────────────────

eval:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/eval_retrieval.py \
		--eval-set Evaluation/data/retrieval/golden_set.json \
		--k 10 \
		--output Evaluation/data/retrieval/run_$(shell date +%Y%m%d).json

eval-postfix-validation:
	$(PYTHONPATH_ENV) $(PYTHON) Evaluation/analysis/post_fix_validation.py \
		--adversarial Evaluation/queries/queries_adversarial.json \
		--abstention Evaluation/queries/queries_abstention.json \
		--k 8 \
		--out Evaluation/data/post_fix/post_fix_validation.json

eval-retrieval-ablation:
	$(PYTHONPATH_ENV) $(PYTHON) Evaluation/analysis/retrieval_baseline_ablation.py \
		--eval-set Evaluation/data/retrieval/golden_set.json \
		--k 10 \
		--out Evaluation/data/post_fix/retrieval_ablation_120q.json \
		--fig Evaluation/figures/post_fix/retrieval_ablation_120q.png

eval-regenerate-clean-a:
	$(PYTHONPATH_ENV) $(PYTHON) Evaluation/analysis/regenerate_human_claim_scores_clean.py \
		--in-csv Evaluation/data/human_labels/claim_scores_scored.csv \
		--out-csv Evaluation/data/human_labels/claim_scores_scored_clean.csv \
		--summary-out Evaluation/data/human_labels/claim_scores_scored_clean_summary.json

eval-human-judge-sample:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/open_eval_prepare_judge_human_sample.py \
		--judge Evaluation/data/post_fix/benchmark_56/judge_eval_post_fix.json \
		--claims-csv Evaluation/data/post_fix/benchmark_56/claim_annotations.csv \
		--sample-size 50 \
		--out-csv Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv \
		--out-manifest Evaluation/data/post_fix/benchmark_56/judge_human_validation_manifest.json

eval-human-judge-score:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/open_eval_score_judge_human_validation.py \
		--sample-csv Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv \
		--out Evaluation/data/post_fix/benchmark_56/judge_human_validation_report.json

eval-retrieval-human-template:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/open_eval_export_retrieval.py \
		--queries Evaluation/queries/queries_120_master.json \
		--k 10 \
		--out Evaluation/data/post_fix/retrieval_120q_export.json \
		--annotation-out Evaluation/data/post_fix/retrieval_120q_annotation_template.json
	$(PYTHONPATH_ENV) $(PYTHON) scripts/open_eval_retrieval_json_to_csv.py \
		--in-json Evaluation/data/post_fix/retrieval_120q_annotation_template.json \
		--out-dir Evaluation/data/post_fix/retrieval_120q_human_template

eval-benchmark56-strict:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/open_eval_export_answers_resumable.py \
		--queries Evaluation/queries/queries_56_runnable.json \
		--k 8 \
		--compute-msa \
		--run-judge-llm \
		--strict-grounding \
		--out Evaluation/data/post_fix/benchmark_56_strict/answers_post_fix.json
	$(PYTHONPATH_ENV) $(PYTHON) scripts/open_eval_answer_export_to_artifacts.py \
		--answers Evaluation/data/post_fix/benchmark_56_strict/answers_post_fix.json \
		--out-dir Evaluation/data/post_fix/benchmark_56_strict
	$(PYTHONPATH_ENV) $(PYTHON) Evaluation/analysis/faithfulness_distribution.py \
		--judge Evaluation/data/post_fix/benchmark_56_strict/judge_eval_post_fix.json \
		--out-dir Evaluation/data/post_fix/benchmark_56_strict \
		--fig-dir Evaluation/figures/post_fix/benchmark_56_strict
	$(PYTHONPATH_ENV) $(PYTHON) Evaluation/analysis/calibration_robustness.py \
		--claims Evaluation/data/post_fix/benchmark_56_strict/claim_annotations.csv \
		--out-dir Evaluation/data/post_fix/benchmark_56_strict \
		--fig-dir Evaluation/figures/post_fix/benchmark_56_strict

eval-cross-corpus:
	$(PYTHONPATH_ENV) $(PYTHON) Evaluation/analysis/cross_corpus_scifact_lite.py \
		--out Evaluation/data/post_fix/cross_corpus_scifact_lite.json \
		--fig Evaluation/figures/post_fix/cross_corpus_scifact_lite.png \
		--n-queries 100

eval-iaa-fresh:
	$(PYTHONPATH_ENV) $(PYTHON) scripts/open_eval_generate_annotator_b.py \
		--judge-a Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv \
		--judge-b-out Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample_annotator_b.csv \
		--retrieval-a Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations.csv \
		--retrieval-b-out Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations_annotator_b.csv \
		--corpus-a Evaluation/data/post_fix/retrieval_120q_human_template/corpus_doc_relevance.csv \
		--corpus-b-out Evaluation/data/post_fix/retrieval_120q_human_template/corpus_doc_relevance_annotator_b.csv
	$(PYTHONPATH_ENV) $(PYTHON) Evaluation/analysis/compute_fresh_iaa.py \
		--judge-a Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample.csv \
		--judge-b Evaluation/data/post_fix/benchmark_56/judge_human_validation_sample_annotator_b.csv \
		--retrieval-a Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations.csv \
		--retrieval-b Evaluation/data/post_fix/retrieval_120q_human_template/retrieval_annotations_annotator_b.csv \
		--out Evaluation/data/post_fix/fresh_iaa_report.json

eval-significance:
	$(PYTHONPATH_ENV) $(PYTHON) Evaluation/analysis/significance_ab_tests.py \
		--baseline-validation Evaluation/data/post_fix/pre_fix_validation_assumed.json \
		--post-validation Evaluation/data/post_fix/post_fix_validation.json \
		--baseline-judge Evaluation/data/post_fix/benchmark_56_baseline_before_strict/judge_eval_post_fix.json \
		--post-judge Evaluation/data/post_fix/benchmark_56_strict/judge_eval_post_fix.json \
		--cross-corpus Evaluation/data/post_fix/cross_corpus_scifact_lite.json \
		--out Evaluation/data/post_fix/significance_ab_tests.json

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "ScholarRAG Makefile targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install       Install runtime dependencies"
	@echo "    make install-dev   Install runtime + dev dependencies"
	@echo ""
	@echo "  Test:"
	@echo "    make test          Run full test suite"
	@echo "    make test-fast     Run tests, stop on first failure"
	@echo "    make test-coverage Run tests with coverage report"
	@echo ""
	@echo "  Lint:"
	@echo "    make lint          Run ruff linter"
	@echo "    make lint-fix      Run ruff with auto-fix"
	@echo "    make typecheck     Run pyright on core modules"
	@echo ""
	@echo "  Run:"
	@echo "    make run           Start backend (dev, auto-reload)"
	@echo "    make run-prod      Start backend (production, 4 workers)"
	@echo "    make run-frontend  Start frontend dev server"
	@echo ""
	@echo "  Eval:"
	@echo "    make reindex       Rebuild all chunk embeddings"
	@echo "    make eval          Run retrieval evaluation"
	@echo "    make eval-postfix-validation  Run adversarial + abstention targeted validation"
	@echo "    make eval-retrieval-ablation  Run dense/BM25/hybrid/rerank retrieval ablation"
	@echo "    make eval-regenerate-clean-a  Recompute leakage-free A on legacy human CSV"
	@echo "    make eval-human-judge-sample  Generate 50-claim human validation sheet"
	@echo "    make eval-human-judge-score   Score completed human-vs-judge labels"
	@echo "    make eval-retrieval-human-template  Generate chunk-level retrieval annotation CSVs"
	@echo "    make eval-benchmark56-strict  Rerun 56-query benchmark with strict grounding"
	@echo "    make eval-cross-corpus  Run SciFact-lite cross-corpus generalization eval"
	@echo "    make eval-iaa-fresh  Generate second-pass labels and fresh IAA report"
	@echo "    make eval-significance  Run formal A/B significance tests"
