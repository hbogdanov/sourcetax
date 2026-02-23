PYTHON ?= python

.PHONY: setup smoke smoke-strict test benchmark phase4

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -e ".[dev]"

smoke:
	$(PYTHON) tools/smoke_run.py

smoke-strict:
	$(PYTHON) tools/smoke_run.py --strict

benchmark:
	$(PYTHON) tools/phase3_benchmark.py --allow-small

phase4:
	$(PYTHON) tools/phase4_run.py

test:
	$(PYTHON) -m pytest -q
