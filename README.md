# SourceTax

SourceTax automatically converts messy financial transaction data (bank exports, card statements, receipt records) into categorized, accounting-ready transactions aligned with IRS Schedule C categories.

It ingests raw financial data, normalizes it into a canonical transaction schema, standardizes merchants, classifies transactions with rules and ML, and produces accounting-grade outputs such as General Ledger lines and audit-ready exports.

## Pipeline

```text
Raw financial data
      ↓
Normalization
      ↓
Canonical transaction schema
      ↓
Merchant normalization
      ↓
Categorization (rules + ML)
      ↓
Evaluation
      ↓
Accounting exports (GL lines, audit logs)
```

## Example

Input:

```text
Starbucks Store 1234 - $8.75
```

Output:

```text
merchant_norm: Starbucks
category: Meals & Entertainment
direction: expense
```

## Problem

Small businesses operate across fragmented financial systems:

- Bank CSV exports
- Card processor exports
- Receipt images and receipt platforms
- Manual bookkeeping

These sources are inconsistent and noisy, which makes reconciliation and tax categorization slow and error-prone.

SourceTax automates:

- Transaction normalization
- Merchant cleaning and alias mapping
- Tax category classification (rules and ML)
- Accounting-ready exports

## Current Status (Phase 3-4)

Implemented:

- Canonical transaction schema enforcement
- Gold dataset (validated, unique, canonical)
- Rule-based categorization engine
- TF-IDF + Logistic Regression baseline
- Optional SBERT embedding pipeline
- Hybrid rules and ML evaluation framework
- Accounting-grade exports:
  - Enriched transactions CSV
  - General Ledger lines CSV
  - Audit trail JSONL
- Dataset validation tests and full pytest suite

Gold dataset: 589 labeled transactions (validated and unique).

## Architecture Overview

Raw Data -> Normalization -> Canonical Schema -> Merchant Normalization -> Categorization (Rules / ML / Hybrid) -> Evaluation -> Accounting Exports

Core code:

- `src/sourcetax/` - main library (schema, normalization, categorization, exports)
- `tools/` - scripts for training, evaluation, and batch processing
- `tests/` - unit and integration tests

Data layout (versioned):

- `data/gold/` - curated labeled dataset
- `data/taxonomy/` - category taxonomy
- `data/mappings/` - mapping tables and rules inputs
- `data/samples/` - small demo inputs

External dataset reference:

- `docs/dataset_sources_guide.md` - curated list of useful/non-useful external datasets for SourceTax
- `docs/product_strategy_and_roadmap.md` - architecture decisions, positioning, metrics, and 4-week roadmap

Generated ML artifacts (not versioned):

- `data/ml/` - transient local split/export files
- `artifacts/models/` - model artifacts
- `artifacts/metrics/` - run metrics JSON
- `artifacts/reports/` - split IDs and comparison reports
- `artifacts/runs/<run_id>/run.json` - canonical index tying config, input hashes, outputs, and git commit (if available)

## Canonical Transaction Contract

Every transaction follows:

- `amount` (always positive)
- `direction` (`expense` or `income`)
- `currency`
- `transaction_date`
- `merchant_raw`
- `merchant_norm`
- `source`
- `source_record_id`
- `sourcetax_category_v1`

Gold data enforces:

- Unique `id`
- Unique `source_record_id`
- No missing required fields
- Valid taxonomy membership

## Quick Start

1) Install

```bash
make setup
```

or

```bash
pip install -e .
```

2) Run tests

```bash
pytest
```

Data-contract quick checks:

```bash
make validate-gold
make validate-taxonomy
make smoke
```

Run those three commands and you know the repository contract checks are passing.

3) Run evaluation on gold

```bash
python tools/eval.py
```

4) Train the ML baseline

```bash
python tools/train_ml_baseline.py
```

5) Compare configs on identical splits

```bash
python tools/model_comparison.py
```

Gold-only comparison mode (no external corpora or HF token required):

```bash
python tools/model_comparison.py --gold-only
```

6) Optional external warm-start (MitulShah)

```bash
make import-hf-mitulshah
make build-mitulshah-corpus
make train-mitulshah-baseline
python tools/train_ml_baseline.py --vectorizer-vocab-from data/external/mitulshah_corpus_train.parquet
```

If the warm-start path does not exist, `train_ml_baseline.py` exits with a clear message. The core flow runs without Mitul data/token by omitting warm-start flags.

## Evaluation Protocol

Treat these as separate tasks:

- Mitul task: predict Mitul labels from transaction description (internal sanity only)
- SourceTax task: predict `sourcetax_category_v1` on gold (this is the product metric)

Mitul internal sanity check (deterministic train/val/test):

```bash
python tools/train_mitulshah_baseline.py
```

Robustness sweep on Mitul preprocessing:

```bash
make eval-mitul-robustness
```

Gold transfer test on identical splits (real test):

```bash
python tools/model_comparison.py
```

One-liner end-to-end transfer evaluation (recommended):

```bash
make eval-transfer
```

This runs:

- Mitul sanity eval (sampled)
- Mitul robustness sweep
- Gold-only reference comparison
- Full transfer comparison

and writes one consolidated report:

- `artifacts/reports/eval_transfer_<run_id>.md`
- `artifacts/reports/eval_transfer_<run_id>.json`

Focus on `macro_f1` first, then per-class F1 for:

- `Repairs & Maintenance`
- `Rent & Utilities`
- `Financial Fees`
- `Income`
- `Meals & Entertainment`

`model_comparison.py` now emits an ROI table and confusion-pair deltas in its report to spot drift and class bias.

## Roadmap

Next:

- Expand gold dataset volume and diversity
- Active learning loop (high-uncertainty sampling)
- Labeling UI workflow for rapid gold growth
- Real ingestion integrations (Plaid-style, POS exports)
- Multi-user cloud deployment

## Vision

SourceTax aims to be an automated tax intelligence layer between raw financial data and accounting systems, reducing manual bookkeeping and enabling accurate, audit-ready reporting for small businesses.
