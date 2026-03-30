# Synthetic Data Workflow

SourceTax has two separate synthetic workflows:

1. `tools/generate_synthetic_gapfill.py`
For classification, mapping, and weak-category coverage.

2. `tools/generate_pairs.py`
For receipt-to-bank matching realism.

## Standard Path

Use this staging database path everywhere:

```text
data/interim/staging.db
```

This keeps staging separate from gold and matches the synthetic tooling defaults.

## Prepare Directories

PowerShell:

```powershell
New-Item -ItemType Directory -Force data\interim, data\ml, artifacts\synthetic
```

Make target:

```bash
make synthetic-prepare
```

## Synthetic Data Policy

- Synthetic rows are inserted into `staging_transactions` or `staging_receipts`, never `data/gold`.
- Synthetic rows may be used for rule coverage, mapping coverage, normalization robustness, auxiliary training experiments, and matching realism tests.
- Final reported categorization metrics must remain gold-only.
- Any experiment mixing synthetic and gold must preserve a gold-only holdout, report synthetic proportion, and report per-category metrics, not just accuracy.

## Phase A: Gapfill Transactions

### Dry Run First

```bash
python tools/generate_synthetic_gapfill.py \
  --staging-db data/interim/staging.db \
  --rows 200 \
  --seed 42 \
  --run-id gapfill_dryrun_v1 \
  --start-date 2025-01-01 \
  --dry-run
```

Check that:

- categories look sane
- amounts look sane
- descriptions are not goofy
- merchants are not too repetitive

### Core Batch

```bash
python tools/generate_synthetic_gapfill.py \
  --staging-db data/interim/staging.db \
  --rows 1200 \
  --seed 42 \
  --run-id gapfill_core_v1 \
  --start-date 2025-01-01 \
  --categories "COGS,Payroll & Contractors,Taxes & Licenses,Insurance,Professional Services,Financial Fees,Rent & Utilities,Vehicle Expenses"
```

This writes synthetic rows to `staging_transactions`.

### Coverage Report

```bash
python tools/gapfill_coverage_report.py \
  --staging-db data/interim/staging.db \
  --target-per-category 150
```

Good first-pass support is around `100-200` rows per weak category.

### Build Mapped Training Rows

```bash
python tools/build_training_rows_from_staging.py \
  --staging-db data/interim/staging.db \
  --out data/ml/staging_training_rows_gapfill.jsonl \
  --where "source = 'synthetic_gapfill'"
```

Use these rows for:

- mapping and rule audits
- staging-derived auxiliary experiments
- debugging category resolution behavior

## Synthetic Mixing Guidance

Use synthetic in layers:

1. Mapping / rule support
2. Staging-derived auxiliary training rows
3. Selective mixed auxiliary corpus

For mixed training:

- gold stays the anchor
- synthetic stays below `25-35%` of total training rows

Recommended ablation sequence:

1. gold-only baseline
2. gold + synthetic at `20%`
3. gold + synthetic at `35%`

Stop if larger synthetic mixes do not improve weak-category performance on the gold holdout.

## Optional LLM Diversification

Default recommendation: leave `--use-llm` off at first.

Template-driven synthetic data is more reproducible, auditable, and easier to debug.

If you want to probe LLM diversification, keep it tiny:

```bash
python tools/generate_synthetic_gapfill.py \
  --staging-db data/interim/staging.db \
  --rows 100 \
  --seed 44 \
  --run-id gapfill_llm_probe_v1 \
  --start-date 2025-01-01 \
  --categories "Professional Services,COGS" \
  --use-llm \
  --llm-model gpt-4.1-mini
```

Inspect outputs before trusting them.

## Phase B: Matching Realism

If `staging_receipts` already has receipts, generate pairs like this:

```bash
python tools/generate_pairs.py \
  --staging-db data/interim/staging.db \
  --out-gold data/ml/synthetic_matching_gold.jsonl \
  --positive-pairs 50 \
  --negative-pairs 100 \
  --seed 42
```

This produces:

- synthetic receipt rows
- synthetic bank rows
- a mini matching-gold JSONL

Use it for:

- threshold tuning
- hard-negative testing
- matching realism checks

Use it for matching evaluation, not categorization evaluation.

## Recommended Operating Procedure

1. Create a branch

```bash
git checkout -b synthetic-gapfill-v1
```

2. Generate the core synthetic staging batch
3. Run the coverage report
4. Build `data/ml/staging_training_rows_gapfill.jsonl`
5. Inspect `50-100` rows manually
6. Lock a gold-only baseline metrics file
7. Run one mixed auxiliary experiment capped at `20%`
8. Keep synthetic only if it improves weak-category holdout behavior without hurting overall quality

## Helper Make Targets

```bash
make synthetic-prepare
make synthetic-gapfill-dryrun
make synthetic-gapfill-core
make synthetic-gapfill-coverage
make synthetic-gapfill-training-rows
make synthetic-matching-pairs
```
