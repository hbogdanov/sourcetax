# Phase 3 Completion Summary

## Overview

Phase 3 established the ML and evaluation foundation for SourceTax's transaction categorization pipeline. The repo now pairs rule-based merchant and MCC heuristics with trainable text models, gold-set evaluation, and reproducible reporting.

## Current Dataset Status

- Locked gold dataset: 589 labeled transactions in `data/gold/gold_transactions.jsonl`
- Current reference split from the baseline transfer evaluation:
  - train: 412
  - validation: 88
  - test: 89
- Current taxonomy anchor: `data/taxonomy/sourcetax_v1.json`

This supersedes the early Phase 3 milestone where the project only had a tiny hand-labeled seed set. The working repo state now reflects the 589-row gold dataset referenced in `README.md` and the evaluation artifacts.

## What Phase 3 Added

### 1. Data Foundation
- Gold dataset loading and validation
- Reproducible train/validation/test splits
- Evaluation scripts for rules-only, ML, and comparison workflows

### 2. ML Baseline
- TF-IDF plus Logistic Regression baseline
- Saved pipeline artifacts for repeatable evaluation
- Locked-holdout reporting for accuracy and F1

### 3. Advanced Enhancements

#### Merchant Normalization
- Rule-based normalization of noisy merchant strings
- Alias cleanup for common abbreviations and variants
- Better shared text features for both rules and ML

#### SBERT Embeddings
- Support for semantic embeddings when optional dependencies are available
- Cached embedding workflow for repeated experiments
- Better handling of short, messy merchant descriptions than pure token overlap

#### Active Learning
- Uncertainty, margin, entropy, and diversity sampling strategies
- Label selection tooling to expand the gold set efficiently

#### Hierarchical Classification
- Multi-level category support aligned with business tax reporting structure
- A path toward more constrained downstream predictions

#### Visual Reporting
- Confusion matrix outputs
- Per-class precision and recall reporting
- Model comparison summaries

## Current Performance Snapshot

Primary artifact:
- `artifacts/metrics/gold_ml_baseline_metrics_gold_eval_20260303_transfer_baseline.json`

Locked gold evaluation snapshot:
- Rules-only test accuracy: 2.2%
- Rules-only test macro-F1: 0.0648
- ML baseline test accuracy: 68.5%
- ML baseline test macro-F1: 0.5777
- ML baseline test weighted F1: 0.6787

Classes with especially strong ML holdout performance in that artifact include:
- Financial Fees
- Repairs & Maintenance
- Vehicle Expenses
- Equipment & Software
- Payroll & Contractors

This supports the current project direction: rules are valuable for high-confidence known patterns, but text-based ML is necessary once descriptions become noisy or sparse.

## Recommended Next Steps

### Immediate
1. Keep expanding the gold set beyond 589 rows, especially for underrepresented categories.
2. Improve class balance so the model is less dominated by `Other Expense`.
3. Continue evaluating hybrid routing instead of choosing rules-only or ML-only.

### Near-Term
1. Turn outreach and partnership efforts into more realistic transaction inputs.
2. Keep synthetic augmentation separate from gold evaluation.
3. Add more business-specific examples for showcase and demo workflows.

### Longer-Term
1. Productionize hybrid inference policies.
2. Improve explainability for category decisions.
3. Add a tighter correction-feedback loop from review UI to retraining.

## Quality Assurance

- Integration tests exist for core Phase 3 components in `tests/test_phase3_integration.py`
- Gold-only evaluation remains the reporting anchor
- Synthetic-data experiments are documented separately in `docs/synthetic_data_workflow.md`

## Summary

Phase 3 is no longer just a prototype built around a 10-row seed. In the current repo state, it represents a reproducible categorization baseline backed by a 589-row gold dataset, evaluation artifacts, and a practical hybrid path that combines deterministic accounting heuristics with text-driven ML.
