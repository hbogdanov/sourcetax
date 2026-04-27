# Final Results Summary

## Dataset

- Gold dataset size: 589 labeled transactions
- Label space: 17 categories in the evaluated baseline artifact
- Reference metrics source: `artifacts/metrics/gold_ml_baseline_metrics_gold_eval_20260303_transfer_baseline.json`

## Results

- Rules-only test accuracy: 2.2%
- ML baseline test accuracy: 68.5%
- ML baseline macro-F1: 0.5777
- ML baseline weighted F1: 0.6787

Strongest holdout classes in the baseline artifact were generally:
- Financial Fees
- Repairs & Maintenance
- Vehicle Expenses
- Equipment & Software
- Payroll & Contractors

## Interpretation

The rule-based system is strongest when it sees known merchants, obvious keyword patterns, or MCC-like signals that map cleanly to a category.

The ML baseline improves materially once text features are used, especially on noisier descriptions and merchant variants. Even so, performance is still limited by class imbalance, sparse examples in some categories, and the dominance of fallback-heavy labels such as `Other Expense`.

## Practical Takeaway

The most realistic approach for SourceTax is a hybrid system:
- use rules for high-confidence merchant and mapping hits
- use ML to generalize beyond memorized patterns
- keep audit-friendly reasoning and exportability in the loop

## Future Work

- collect more labeled real-world transactions
- build better business partnerships for data access
- expand category coverage for underrepresented classes
- continue synthetic augmentation without contaminating gold-only evaluation
- refine hybrid routing and confidence thresholds
