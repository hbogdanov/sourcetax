# Gold Standard Dataset

This directory contains hand-labeled transaction data for evaluating SourceTax categorization, matching, and extraction.

## Files

- `gold_transactions.jsonl` - Canonical transactions with human labels

## Locked Labeling Contract

- Primary label field: `sourcetax_category_v1`
- Backward-compatible copy: `category_final`
- Confidence field: `label_confidence` (`high|medium|low`)
- Notes field: `label_notes` (free text, empty string allowed)
- Only SourceTax v1 taxonomy values are allowed for labels.

## Row Format

```json
{
  "id": "gold_001",
  "source": "bank|receipt|toast|quickbooks",
  "transaction_date": "2024-01-15",
  "merchant_raw": "STARBUCKS COFFEE",
  "merchant_norm": "Starbucks",
  "amount": 6.45,
  "direction": "expense|income",
  "category_final": "Meals & Entertainment",
  "sourcetax_category_v1": "Meals & Entertainment",
  "label_confidence": "high",
  "label_notes": "",
  "matched_transaction_id": "gold_003",
  "match_score": 0.87,
  "raw_payload": {
    "label_source": "human"
  }
}
```

## Ground Truth Fields

- `sourcetax_category_v1` - Canonical SourceTax v1 label for training/eval/export.
- `category_final` - Duplicate of the canonical label for backward compatibility.
- `label_confidence` - Human confidence in the assigned label.
- `label_notes` - Human notes for ambiguity/edge cases.
- `matched_transaction_id` - Verified receipt-to-bank linkage when applicable.
- `merchant_norm`, `amount`, `transaction_date`, `direction` - Human-verified values.

## Authoring Rules

- If truly ambiguous: label `Other Expense` and explain in `label_notes`.
- Do not write `Uncategorized` into gold.
- Ensure `raw_payload.label_source == "human"` for gold rows.
