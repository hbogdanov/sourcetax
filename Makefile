PYTHON ?= python
SEED ?= 42
RUN_ID ?=
MITUL_STRICT ?= 0
KEY_TEST_MIN_SUPPORT ?= 0
KEY_CATEGORIES ?= Repairs & Maintenance,Rent & Utilities,Financial Fees,Income,Meals & Entertainment

.PHONY: setup smoke smoke-strict test validate-gold validate-taxonomy benchmark phase4 import-hf-mitulshah build-mitulshah-corpus train-mitulshah-baseline eval-mitul-robustness eval-transfer model-comparison

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -e ".[dev]"

smoke:
	$(PYTHON) tools/smoke_run.py

smoke-strict:
	$(PYTHON) tools/smoke_run.py --strict

validate-gold:
	$(PYTHON) -m pytest -q tests/test_gold_enforcement.py

validate-taxonomy:
	$(PYTHON) -m pytest -q tests/test_taxonomy_enforcement.py

benchmark:
	$(PYTHON) tools/phase3_benchmark.py --allow-small

phase4:
	$(PYTHON) tools/phase4_run.py

test:
	$(PYTHON) -m pytest -q

import-hf-mitulshah:
	$(PYTHON) tools/import_hf_mitulshah.py --mirror-only

build-mitulshah-corpus:
	$(PYTHON) tools/build_mitulshah_corpus.py

train-mitulshah-baseline:
	$(PYTHON) tools/train_mitulshah_baseline.py

eval-mitul-robustness:
	$(PYTHON) tools/eval_mitul_robustness.py

eval-transfer:
	$(PYTHON) tools/eval_transfer.py --seed $(SEED) --key-test-min-support $(KEY_TEST_MIN_SUPPORT) --key-categories "$(KEY_CATEGORIES)" $(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(filter 1 true TRUE yes YES,$(MITUL_STRICT)),--strict-mitul,)

model-comparison:
	$(PYTHON) tools/model_comparison.py
