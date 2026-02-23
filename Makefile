PYTHON ?= python

.PHONY: setup smoke test benchmark phase4

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .
	$(PYTHON) -m pip install -e ".[dev]"

smoke:
	$(PYTHON) tools/smoke_run.py

benchmark:
	$(PYTHON) tools/phase3_benchmark.py --allow-small

phase4:
	$(PYTHON) tools/phase4_run.py

test:
	$(PYTHON) -m pytest -q
