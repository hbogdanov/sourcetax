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

Launch the review UI:

```bash
streamlit run app_review.py
```

Run validation and smoke checks:

```bash
pytest
make validate-gold
make validate-taxonomy
make smoke
```

## Core Commands

UI workflow:

```text
Upload / Ingest -> Review Grid -> Exceptions -> Export
```

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

## Synthetic Data

SourceTax supports two separate synthetic workflows:

- `tools/generate_synthetic_gapfill.py` for categorization, mapping, and weak-category coverage
- `tools/generate_pairs.py` for receipt-to-bank matching realism

Standard staging path:

```text
data/interim/staging.db
```

Synthetic data policy:

- synthetic rows go to staging, never `data/gold`
- final reported categorization metrics stay gold-only
- any gold+synthetic experiment must preserve a gold-only holdout
- mixed experiments must report synthetic proportion and per-category metrics

Full workflow and exact commands:

- `docs/synthetic_data_workflow.md`

Experimental mixed-training ablations:

```bash
python tools/training/experimental_mixed_aux_train.py --synthetic-ratio 0.20
python tools/training/experimental_mixed_aux_train.py --synthetic-ratio 0.35
```

Current result on the locked gold holdout:

- gold-only ML macro-F1: `0.5777`
- gold + synthetic at `20%`: `0.6179`
- gold + synthetic at `35%`: `0.6119`

Current recommendation: keep `20%` as the experimental default if you mix synthetic rows at all. It improved the locked gold-holdout macro-F1 more than `35%`, while preserving the same gold-only evaluation anchor.

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
