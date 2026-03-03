# SourceTax Progress Log (March 2026)

This log summarizes the major work completed in the recent data-focused execution cycle.

## 1) Taxonomy Contract Enforcement

- Locked SourceTax v1 taxonomy as single source of truth.
- Enforced taxonomy usage in UI, validators, and export paths.
- Removed final-output dependency on `Uncategorized`; fallback is `Other Expense`.

## 2) Data Foundation and Staging

- Added/expanded staging schema and utilities for:
  - `staging_transactions`
  - `staging_receipts`
- Added import pipelines:
  - HF transaction corpus importer
  - DC p-card ArcGIS importer
  - SROIE receipt importer
- Added enriched staging build path with rule predictions, mapping reasons, and missing-field indicators.

## 3) Mapping System and Explainability

- Implemented deterministic precedence:
  - keyword -> MCC/MCC-description -> external category -> fallback
- Added explicit mapping evidence (`mapping_reason` / reason tokens).
- Expanded external and MCC mapping scaffolds from observed staging distributions.

## 4) Merchant Normalization and Noise Realism

- Strengthened merchant normalization:
  - punctuation/store-id cleanup
  - processor/acquirer prefix handling
- Added synthetic merchant noise utilities for realistic raw-merchant variants.

## 5) Matching Realism Utilities

- Added receipt-bank pair generator with:
  - date offsets
  - amount drift
  - merchant noise
  - non-match hard negatives
- Added mini matching-gold generation support for evaluation.

## 6) Gold Dataset Governance

- Enforced human-only gold usage for training/evaluation.
- Added formal labeling contract:
  - `sourcetax_category_v1`
  - `label_confidence` (`high|medium|low`)
  - `label_notes`
- Expanded gold through multiple targeted batches and merge flows.
- Added balance and coverage reporting (`tools/gold_balance_report.py`).

## 7) Batch Export Workflows

- Added targeted labeling exporters:
  - baseline balanced exporter
  - low-support category exporters
  - low-confidence/ambiguity exporters
  - sparse-category focused exporter
  - source-balance exporter
- Added dedupe controls (within-batch and against existing gold fingerprints).

## 8) Rules/ML/Hybrid Evaluation Stack

- Added:
  - error breakdown report
  - confusion reporting
  - rules vs ML vs hybrid comparison tooling
  - 5-fold CV utility
  - locked holdout train/eval utility
  - holdout policy sweep utility
- Added canonical enriched TF-IDF artifact train/load path with metadata:
  - taxonomy hash
  - feature builder version
  - training composition
  - git revision metadata

## 9) Shadow Mode and Hybrid Safety

- Extended shadow logging with:
  - rule prediction/confidence/reason
  - ML prediction/confidence
  - multiple hybrid variants (`t70`, `t85`)
  - prod-candidate policy output
- Added backfill tooling for large-scale shadow logging on staging/canonical tables.
- Added staging disagreement and sanity exports.
- Added Stage-1 prod-candidate review reporting with gate/intersection diagnostics.

## 10) RPM Repair Sprint (Major Quality Lift)

- Performed targeted Repairs & Maintenance error analysis.
- Added focused RPM lexical + MCC-description matcher.
- Fixed ambiguity routes (`electric`, `contractor`) that were causing systemic RPM misses.

Observed impact after RPM changes (gold eval):
- Overall categorization rose to high-80s.
- RPM recall improved from severe underperformance to high recall.

## 11) Current Constraints and Next Work

- Gold source skew remains heavy toward `dc_pcard`.
- Bank/Toast sample availability in this repo is small; source-balance batch exporter now reports explicit shortages.
- Hybrid policy shows potential but remains split-sensitive; continue staged rollout and review-queue validation before unconditional production flip.

Recommended next priorities:
1. Increase natural bank/Toast diversity in gold.
2. Continue targeted rule refinement for remaining low-recall categories.
3. Re-run locked-holdout policy checks after each major gold expansion cycle.

