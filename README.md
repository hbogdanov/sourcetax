# SourceTax

SourceTax turns messy transaction exports into categorized, accounting-ready records for tax prep.

It takes raw bank or POS data, normalizes merchants into a canonical schema, classifies each transaction, and exports enriched transactions, GL lines, and audit-ready logs.

## Killer Demo

Run one command:

```bash
python run_pipeline.py --input data/samples/bank_sample.csv
```

Before:

```text
STARBUCKS #1234  ATLANTA GA   -5.87
```

After:

```text
Category: Meals & Entertainment
Merchant: starbucks
Tax Class: Deductible Expense
```

## Why It Feels Like A Product

- One entrypoint runs normalization, classification, and export end to end.
- Outputs are product-shaped: enriched transactions CSV, GL lines CSV, audit trail JSONL, and QuickBooks import CSV.
- The taxonomy maps to Schedule C style small-business expense categories.

## Measured Lift

On a held-out split of the 589-row gold dataset, the ML baseline improved classification accuracy from `2.2%` rules-only to `68.5%` on the same test set.

Source artifact:

- `artifacts/metrics/gold_ml_baseline_metrics_sanity_gold_only_20260303_baseline.json`

## Repo Layout

```text
src/sourcetax/          core library
tools/data_pipeline/    ingestion, smoke runs, exports
tools/training/         model training scripts
tools/evaluation/       benchmarking and comparison
data/                   taxonomy, mappings, gold labels, sample inputs
tests/                  unit and integration tests
```

Legacy `tools/*.py` wrappers still work, but the grouped directories are the primary layout now.

## Quick Start

Install:

```bash
make setup
```

Run the product demo:

```bash
python run_pipeline.py --input data/samples/bank_sample.csv
```

Run validation and smoke checks:

```bash
pytest
make validate-gold
make validate-taxonomy
make smoke
```

## Core Commands

Evaluation:

```bash
python tools/evaluation/eval.py
python tools/evaluation/model_comparison.py --gold-only
```

Training:

```bash
python tools/training/train_ml_baseline.py
python tools/training/train_mitulshah_baseline.py
```

Pipeline exports:

```bash
python tools/data_pipeline/phase4_run.py
```

## What SourceTax Does

- Normalizes raw transactions into a canonical schema.
- Cleans merchant names and aliases.
- Classifies transactions with rules and ML.
- Produces accounting-grade exports with audit traceability.

## Current Dataset + Outputs

- Gold dataset: `589` validated labeled transactions.
- Versioned taxonomy: `data/taxonomy/sourcetax_v1.json`
- Export bundle:
  - `accounting_transactions_enriched.csv`
  - `gl_lines.csv`
  - `audit_trail.jsonl`
  - `quickbooks_import.csv`
