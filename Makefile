PYTHON ?= python
SEED ?= 42
RUN_ID ?=
MITUL_STRICT ?= 0
KEY_TEST_MIN_SUPPORT ?= 0
KEY_CATEGORIES ?= Repairs & Maintenance,Rent & Utilities,Financial Fees,Income,Meals & Entertainment

.PHONY: setup pipeline smoke smoke-strict test validate-gold validate-taxonomy benchmark phase4 import-hf-mitulshah build-mitulshah-corpus train-mitulshah-baseline eval-mitul-robustness eval-transfer model-comparison synthetic-prepare synthetic-gapfill-dryrun synthetic-gapfill-core synthetic-gapfill-coverage synthetic-gapfill-training-rows synthetic-matching-pairs

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -e ".[dev]"

pipeline:
	$(PYTHON) run_pipeline.py --input data/samples/bank_sample.csv

smoke:
	$(PYTHON) tools/data_pipeline/smoke_run.py

smoke-strict:
	$(PYTHON) tools/data_pipeline/smoke_run.py --strict

validate-gold:
	$(PYTHON) -m pytest -q tests/test_gold_enforcement.py

validate-taxonomy:
	$(PYTHON) -m pytest -q tests/test_taxonomy_enforcement.py

benchmark:
	$(PYTHON) tools/evaluation/phase3_benchmark.py --allow-small

phase4:
	$(PYTHON) tools/data_pipeline/phase4_run.py

test:
	$(PYTHON) -m pytest -q

import-hf-mitulshah:
	$(PYTHON) tools/import_hf_mitulshah.py --mirror-only

build-mitulshah-corpus:
	$(PYTHON) tools/build_mitulshah_corpus.py

train-mitulshah-baseline:
	$(PYTHON) tools/training/train_mitulshah_baseline.py

eval-mitul-robustness:
	$(PYTHON) tools/evaluation/eval_mitul_robustness.py

eval-transfer:
	$(PYTHON) tools/evaluation/eval_transfer.py --seed $(SEED) --key-test-min-support $(KEY_TEST_MIN_SUPPORT) --key-categories "$(KEY_CATEGORIES)" $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(filter 1 true TRUE yes YES,$(MITUL_STRICT)),--strict-mitul,)

model-comparison:
	$(PYTHON) tools/evaluation/model_comparison.py

synthetic-prepare:
	$(PYTHON) -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['data/interim', 'data/ml', 'artifacts/synthetic']]"

synthetic-gapfill-dryrun: synthetic-prepare
	$(PYTHON) tools/generate_synthetic_gapfill.py --staging-db data/interim/staging.db --rows 200 --seed 42 --run-id gapfill_dryrun_v1 --start-date 2025-01-01 --dry-run

synthetic-gapfill-core: synthetic-prepare
	$(PYTHON) tools/generate_synthetic_gapfill.py --staging-db data/interim/staging.db --rows 1200 --seed 42 --run-id gapfill_core_v1 --start-date 2025-01-01 --categories "COGS,Payroll & Contractors,Taxes & Licenses,Insurance,Professional Services,Financial Fees,Rent & Utilities,Vehicle Expenses"

synthetic-gapfill-coverage:
	$(PYTHON) tools/gapfill_coverage_report.py --staging-db data/interim/staging.db --target-per-category 150

synthetic-gapfill-training-rows: synthetic-prepare
	$(PYTHON) tools/build_training_rows_from_staging.py --staging-db data/interim/staging.db --out data/ml/staging_training_rows_gapfill.jsonl --where "source = 'synthetic_gapfill'"

synthetic-matching-pairs: synthetic-prepare
	$(PYTHON) tools/generate_pairs.py --staging-db data/interim/staging.db --out-gold data/ml/synthetic_matching_gold.jsonl --positive-pairs 50 --negative-pairs 100 --seed 42
