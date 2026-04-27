# Showcase Summary

## Problem

Small businesses often export bank or POS transactions as messy CSVs with noisy merchant names and no tax-ready categorization. Manual cleanup is slow, inconsistent, and hard to audit.

## Pipeline

SourceTax ingests raw transaction files, normalizes merchant descriptions, applies rule-based and ML categorization, and exports accounting-oriented outputs:
- enriched transactions
- GL lines
- audit trail logs
- QuickBooks-style import CSVs

## Dataset

The current gold dataset contains 589 labeled transactions in `data/gold/gold_transactions.jsonl`. It supports baseline evaluation, taxonomy validation, and hybrid rules-plus-ML experiments.

## Outreach / Data Acquisition

Outreach to local small businesses and accounting-oriented firms produced mostly rejections because of privacy, compliance, and financial data sensitivity. One redacted sample was received, but it was too inconsistent to support reliable training ingestion. Real financial data access remains the main bottleneck.

## Results

On the locked gold evaluation artifact, rules-only accuracy was 2.2% while the ML baseline reached 68.5% accuracy. The repo evidence supports a hybrid conclusion: deterministic rules help on known merchants and explicit signals, while ML is needed for messy real-world text.

## Lessons Learned

- Merchant normalization matters more than it first appears.
- Gold-only evaluation needs to stay separate from synthetic augmentation.
- Imbalanced categories limit headline performance even when the pipeline is structurally sound.
- Real data access is harder than building the technical pipeline.

## Future Work

- grow the labeled dataset with better class balance
- secure stronger business partnerships for data-sharing
- improve hybrid routing policies
- expand showcase and demo scenarios with richer sample inputs
- keep synthetic augmentation as a support layer, not the evaluation source of truth
