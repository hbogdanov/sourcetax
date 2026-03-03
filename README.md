# SourceTax

SourceTax is an AI-driven transaction classification and tax automation pipeline for small businesses.

It ingests raw financial transaction data (bank exports, card data, receipt records), normalizes everything into a clean canonical schema, categorizes transactions into IRS Schedule C-aligned tax categories, and produces accounting-grade outputs such as General Ledger lines and audit-ready exports.

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

3) Run evaluation on gold

```bash
python tools/eval.py
```

4) Train the ML baseline

```bash
python tools/train_ml_baseline.py
```

## Roadmap

Next:

- Expand gold dataset volume and diversity
- Active learning loop (high-uncertainty sampling)
- Labeling UI workflow for rapid gold growth
- Real ingestion integrations (Plaid-style, POS exports)
- Multi-user cloud deployment

## Vision

SourceTax aims to be an automated tax intelligence layer between raw financial data and accounting systems, reducing manual bookkeeping and enabling accurate, audit-ready reporting for small businesses.
