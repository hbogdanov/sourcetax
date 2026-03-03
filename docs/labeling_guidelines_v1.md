# SourceTax Labeling Guidelines v1

This document defines the labeling contract for SourceTax gold data.

## Label Field (Locked)

- Label field name: `sourcetax_category_v1`
- Allowed values: SourceTax v1 taxonomy categories only (from `data/taxonomy/sourcetax_v1.json`)
- `Uncategorized` is not a valid gold label.

## Labeling Context Shown to Reviewers

Labelers should review the following fields when available:

- `merchant_raw`
- `merchant_norm`
- `description`
- `amount`
- `mcc`
- `mcc_description`
- `receipt_text` (optional)

## Ambiguity Rules

When uncertain, prefer conservative labeling and write a note.

1. `Meals & Entertainment` vs travel meals
- Meal/restaurant/coffee spend stays `Meals & Entertainment`.
- Use `Travel` only for transportation/lodging-related travel costs, not meals alone.

2. `Vehicle Expenses` vs `Travel`
- Fuel, tolls, parking, routine vehicle costs: `Vehicle Expenses`.
- Airfare, hotels, ride-share for trips, lodging: `Travel`.

3. `Equipment & Software` vs `Office Supplies` vs `Professional Services`
- SaaS subscriptions, software licenses, developer tools, cloud services: `Equipment & Software`.
- Physical office consumables: `Office Supplies`.
- Retainers or billable expert work (legal/accounting/consulting): `Professional Services`.

4. Owner draw vs payroll vs contractor
- Payroll wages/employee payroll service charges: `Payroll & Contractors`.
- 1099/vendor labor/contractors: `Payroll & Contractors`.
- Owner draw/distribution should not be treated as expense labeling; if forced in transaction labeling context and ambiguous, use `Other Expense` plus note.

## Hard Ambiguity Rule

- If truly ambiguous after context review, label `Other Expense` and add `label_notes` explaining why.

## Required Gold Metadata

Every gold record must include:

- `sourcetax_category_v1`: canonical SourceTax v1 label
- `label_confidence`: one of `high`, `medium`, `low`
- `label_notes`: free-text notes (empty string allowed)

## Reviewer Checklist

1. Confirm amount direction and merchant context.
2. Select `sourcetax_category_v1` from taxonomy.
3. Set `label_confidence` (`high`/`medium`/`low`).
4. Add `label_notes` for edge/ambiguous cases.
5. Save.
